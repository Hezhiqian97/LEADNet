"""
Implementation of Extreme-Aspect-Ratio Network: StarNet-SGD.
Full-Network Spectral Coverage Edition (Shallow + Deep)

Optimized for H=1300, W=160 Industrial Surface Defect Segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

# ==============================================================================
# 1. 频域与几何特征解耦组件
# ==============================================================================

class SpectralAsymmetricDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(2, 1)):
        super().__init__()
        self.stride = stride

        self.spatial_down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.complex_weight = nn.Parameter(
            torch.randn(in_channels, 16, 16, 2, dtype=torch.float32) * 0.02
        )

        # 【核心修复】：根据输入步长，动态调整重叠池化核的尺寸
        # 保证浅层 stride=(2,2) 和深层 stride=(2,1) 均能完美无损拼接
        if isinstance(stride, int):
            stride = (stride, stride)

        if stride[0] > 1 and stride[1] > 1:
            pool_kernel, pool_pad = (3, 3), (1, 1)
        elif stride[0] > 1:
            pool_kernel, pool_pad = (3, 1), (1, 0)
        else:
            pool_kernel, pool_pad = (1, 3), (0, 1)

        self.spectral_pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=stride, padding=pool_pad)

        self.spectral_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        feat_spatial = self.spatial_down(x)

        x_fft = torch.fft.rfft2(x, norm='ortho')
        fft_h, fft_w = x_fft.shape[-2], x_fft.shape[-1]

        # 👑 手术 3：频域权重插值必须使用 nearest，防止双线性破坏傅里叶相位！
        weight = self.complex_weight.permute(0, 3, 1, 2).contiguous()
        weight = F.interpolate(weight, size=(fft_h, fft_w), mode='nearest')  # 修改这里
        weight = weight.permute(0, 2, 3, 1).contiguous()
        weight = torch.view_as_complex(weight)

        x_filtered = torch.fft.irfft2(x_fft * weight, s=(H, W), norm='ortho')
        feat_spectral = self.spectral_proj(self.spectral_pool(x_filtered))

        out = self.act(feat_spatial + feat_spectral)
        return out



class DynamicOrthogonalProjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_h = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), groups=channels)
        self.conv_w = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), groups=channels)
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_h = self.conv_h(x_h).expand(-1, -1, H, W)
        x_w = self.conv_w(x_w).expand(-1, -1, H, W)
        attn_weights = self.attention(torch.cat([x_h, x_w], dim=1))
        return x * attn_weights# + x_h * 0.5 + x_w * 0.5

class SpectralGating(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.base_h, self.base_w = 16, 16
        self.complex_weight = nn.Parameter(
            torch.randn(channels, self.base_h, self.base_w, 2, dtype=torch.float32) * 0.02
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.rfft2(x, norm='ortho')
        fft_h, fft_w = x_fft.shape[-2], x_fft.shape[-1]

        weight = self.complex_weight.permute(0, 3, 1, 2).contiguous()
        weight = F.interpolate(weight, size=(fft_h, fft_w), mode='bilinear', align_corners=True)
        weight = weight.permute(0, 2, 3, 1).contiguous()
        weight = torch.view_as_complex(weight)

        x_fft_filtered = x_fft * weight
        x_filtered = torch.fft.irfft2(x_fft_filtered, s=(H, W), norm='ortho')
        return x_filtered



class SpectralGeometricBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spectral_branch = SpectralGating(channels)
        self.geometric_branch = DynamicOrthogonalProjection(channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        f_spec = self.spectral_branch(x)
        f_geom = self.geometric_branch(x)
        out = self.fusion(torch.cat([f_spec, f_geom], dim=1))
        return out + x

# ==============================================================================
# 2. 骨干网络组件 & 频域增强下采样
# ==============================================================================
class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=not with_bn))
        if with_bn:
            self.add_module('bn', nn.BatchNorm2d(out_planes))
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)  # <--- 就是这里，之前漏写了 nn.

class StripBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv_v = ConvBN(dim, dim, kernel_size=(9, 1), stride=1, padding=(4, 0), groups=4)
        self.dwconv_h = ConvBN(dim, dim, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=4)

        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 3, 1, padding=1, groups=4)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input_tensor = x
        x = self.dwconv_v(x) + self.dwconv_h(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        return input_tensor + self.drop_path(x)




# ==============================================================================
# 3. 主网络架构 (全网络频域覆盖版)
# ==============================================================================
class ExtremeStarNet(nn.Module):
    def __init__(self, base_dim=16, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = base_dim

        self.stem = nn.Sequential(
            ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1),
            nn.ReLU6()
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cur = 0

        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer

            # 步长策略保留：浅层全局压缩，深层锁定宽度
            if i_layer >= 2:
                stride = (2, 1)
            else:
                stride = (2, 2)

            # 【全网络覆盖】：无论浅层还是深层，全部启用频域增强下采样！
            down_sampler = SpectralAsymmetricDownsample(self.in_channel, embed_dim, stride=stride)
            #down_sampler = nn.Conv2d( in_channels=self.in_channel, out_channels=embed_dim, kernel_size=3,  stride=2, padding=1, bias=False)
            self.in_channel = embed_dim

            blocks = [StripBlock(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]

            # 【全网络覆盖】：无论浅层还是深层，全部注入频域-几何解耦模块！
            blocks.append(SpectralGeometricBlock(self.in_channel))

            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        F_out = []
        x = self.stem(x)
        F_out.append(x)
        for stage in self.stages:
            x = stage(x)
            F_out.append(x)
        return F_out

@register_model
def ExtremeStarNet_base(pretrained=False, **kwargs):
    return ExtremeStarNet(base_dim=16, depths=[1, 1, 3, 1], mlp_ratio=4, **kwargs)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ExtremeStarNet_base().to(device)

    # 严格按照 argparse 传入竖向长图: [Batch, Channel, Height, Width]
    dummy_input = torch.randn(1, 3, 160, 1280).to(device)

    print("开始前向传播，计算所有层级的频域特征...")
    feats = model(dummy_input)

    print("\n✅ 全网络频域覆盖版 ExtremeStarNet 执行成功！")
    print("-" * 60)




    from thop import profile

    # 1. 检查是否有可用的 GPU
    assert torch.cuda.is_available(), "必须使用 GPU 才能运行满血版 DCNv4！"
    device = torch.device('cuda')


    dummy_input = torch.randn(1, 3, 160, 1280).to(device)

    print("--- 开始计算参数量与 FLOPs ---")

    # 4. 使用 thop 进行计算 (thop 在 GPU 上会自动追踪张量运算)
    flops, params = profile(model, inputs=(dummy_input,))

    print(f"Total FLOPs: {flops / 1e9:.3f} G")  # 计算量 (Giga)
    print(f"Total Params: {params / 1e6:.3f} M")  # 参数量 (Million)

