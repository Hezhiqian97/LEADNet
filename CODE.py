"""
Implementation of Extreme-Aspect-Ratio Network: StarNet-SGD.
Full-Network Spectral Coverage Edition (Shallow + Deep)
Optimized for H=1300, W=160 Industrial Surface Defect Segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ==============================================================================
# 1. Frequency and Geometric Feature Decoupling Components
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

        # Dynamically adjust pooling kernel size based on stride input
        # Ensure lossless concatenation for different strides
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

        # Use nearest interpolation for frequency weights to preserve Fourier phase
        weight = self.complex_weight.permute(0, 3, 1, 2).contiguous()
        weight = F.interpolate(weight, size=(fft_h, fft_w), mode='nearest')
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
        return x * attn_weights


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
# 2. Backbone Components & Frequency-Enhanced Downsampling
# ==============================================================================

class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv',
                        nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                                  bias=not with_bn))
        if with_bn:
            self.add_module('bn', nn.BatchNorm2d(out_planes))
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)


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
# 3. Main Network Architecture
# ==============================================================================

class ExtremeStarNet(nn.Module):
    def __init__(self, base_dim=16, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, **kwargs):
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

            # Stride strategy: shallow layers use global compression, deep layers lock width
            if i_layer >= 2:
                stride = (2, 1)
            else:
                stride = (2, 2)

            # Apply frequency-enhanced downsampling across all layers
            down_sampler = SpectralAsymmetricDownsample(self.in_channel, embed_dim, stride=stride)
            self.in_channel = embed_dim

            blocks = [StripBlock(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]

            # Apply frequency-geometric decoupling module across all layers
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


# ==============================================================================
# 4. Decoder Components
# ==============================================================================

class WindmillOffsetBypass(nn.Module):
    """LayerScale-enabled pixel residual windmill bypass."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        mid_c = in_channels // 2
        self.reduce = nn.Conv2d(in_channels, mid_c, kernel_size=1, bias=False)

        self.pad_left = nn.ZeroPad2d((2, 0, 0, 0))
        self.conv_left = nn.Conv2d(mid_c, mid_c, kernel_size=(1, 3), bias=False)

        self.pad_right = nn.ZeroPad2d((0, 2, 0, 0))
        self.conv_right = nn.Conv2d(mid_c, mid_c, kernel_size=(1, 3), bias=False)

        self.pad_up = nn.ZeroPad2d((0, 0, 2, 0))
        self.conv_up = nn.Conv2d(mid_c, mid_c, kernel_size=(3, 1), bias=False)

        self.pad_down = nn.ZeroPad2d((0, 0, 0, 2))
        self.conv_down = nn.Conv2d(mid_c, mid_c, kernel_size=(3, 1), bias=False)

        self.pixel_gate = nn.Sequential(
            nn.Conv2d(mid_c * 4, mid_c * 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c * 2, mid_c * 4, kernel_size=1),
            nn.Sigmoid()
        )

        self.fuse_conv = nn.Conv2d(mid_c * 4, out_channels, kernel_size=1, bias=False)
        self.fuse_bn = nn.BatchNorm2d(out_channels)

        # LayerScale 1e-2 soft start
        nn.init.constant_(self.fuse_bn.weight, 1e-2)
        nn.init.constant_(self.fuse_bn.bias, 0)

    def forward(self, x):
        x_reduced = self.reduce(x)

        out_l = self.conv_left(self.pad_left(x_reduced))
        out_r = self.conv_right(self.pad_right(x_reduced))
        out_u = self.conv_up(self.pad_up(x_reduced))
        out_d = self.conv_down(self.pad_down(x_reduced))

        out_cat = torch.cat([out_l, out_r, out_u, out_d], dim=1)
        out_cat = out_cat * (1.0 + self.pixel_gate(out_cat))

        return self.fuse_bn(self.fuse_conv(out_cat))


class concat(nn.Module):
    """Windmill offset residual decoder."""

    def __init__(self, shallow_c, deep_c, out_size):
        super(concat, self).__init__()

        in_channels = shallow_c + deep_c

        # 1. Main stream
        self.conv_base = nn.Conv2d(in_channels, out_size, kernel_size=3, padding=1, bias=False)
        self.bn_base = nn.BatchNorm2d(out_size)

        # 2. Lightweight windmill bypass stream
        self.windmill_bypass = WindmillOffsetBypass(in_channels, out_size)

        # 3. Final smoothed output
        self.conv_final = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        # Dynamic alignment
        target_size = inputs1.shape[2:]
        outputs2 = F.interpolate(inputs2, size=target_size, mode='nearest')

        # Lossless concatenation
        x_concat = torch.cat([inputs1, outputs2], dim=1)

        # A. Main branch forward pass
        x_base = self.relu(self.bn_base(self.conv_base(x_concat)))

        # B. Offset bypass scan
        x_offset = self.windmill_bypass(x_concat)

        # C. Residual injection
        x_fused = x_base + x_offset

        # D. Final feature output
        outputs = self.relu(self.conv_final(x_fused))

        return outputs


class LEADNet(nn.Module):
    def __init__(self, num_classes=2):
        super(LEADNet, self).__init__()

        # 1. Backbone network (disable drop_path randomness)
        self.barbk = ExtremeStarNet_base(drop_path_rate=0.0)

        # 2. Cross-scale attention residual decoding modules
        self.up_concat4 = concat(64, 128, out_size=128)
        self.up_concat3 = concat(32, 128, out_size=64)
        self.up_concat2 = concat(16, 64, out_size=32)
        self.up_concat1 = concat(16, 32, out_size=32)

        # 3. Prediction head
        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1),
        )
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        feats = self.barbk(x)
        feat1, feat2, feat3, feat4, feat5 = feats

        # Top-down step-by-step decoding
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        up1 = self.up_conv(up1)
        final = self.final(up1)
        return final


# ==============================================================================
# Execution Script
# ==============================================================================
if __name__ == '__main__':
    try:
        from thop import profile
    except ImportError:
        print("Please install thop: pip install thop")
        exit()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU...")
    else:
        device = torch.device('cpu')
        print("Using CPU...")

    # Initialize the complete LEADNet model
    model = LEADNet(num_classes=2).to(device)

    # Create dummy input [Batch, Channel, Height, Width]
    dummy_input = torch.randn(1, 3, 160, 1280).to(device)

    print("Starting parameters and FLOPs calculation...")

    # Profile model
    flops, params = profile(model, inputs=(dummy_input,))

    print(f"Total FLOPs: {flops / 1e9:.3f} G")
    print(f"Total Params: {params / 1e6:.3f} M")

    # Verify forward pass
    out = model(dummy_input)
    print(f"Output shape: {out.shape}")