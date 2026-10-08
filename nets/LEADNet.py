import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# 屏蔽烦人的 timm 弃用警告和模型重注册警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 确保你的 starnet.py 路径正确
from nets.ExtremeStarNet import ExtremeStarNet_base
from nets.stane import starnet_s050



class WindmillOffsetBypass(nn.Module):
    """
    大圆满终极版：带 LayerScale 软启动的像素残差风车 (LayerScale WORD)
    【最后一击】：利用 1e-2 的软启动打破 Zero-Init 的早期梯度饥饿，榨干最后一丝 Recall！
    """

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

        # 👑 仅仅修改这里：LayerScale 1e-2 软启动
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
    """
    风车偏移残差解码器 (WORD)
    【架构哲学】：无损主路保底 + 风车偏移旁路追踪（极致速度与精度的平衡）。
    """

    def __init__(self, shallow_c, deep_c, out_size):
        super(concat, self).__init__()

        in_channels = shallow_c + deep_c

        # ==========================================================
        # 1. 绝对无损主路 (Main Stream)
        # ==========================================================
        self.conv_base = nn.Conv2d(in_channels, out_size, kernel_size=3, padding=1, bias=False)
        self.bn_base = nn.BatchNorm2d(out_size)

        # ==========================================================
        # 2. 👑 轻量级风车偏移旁路 (Bypass Stream)
        # ==========================================================
        self.windmill_bypass = WindmillOffsetBypass(in_channels, out_size)

        # 3. 最终平滑输出
        self.conv_final = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        # 动态对齐
        target_size = inputs1.shape[2:]
        outputs2 = F.interpolate(inputs2, size=target_size, mode='nearest')

        # 绝对无损拼接
        x_concat = torch.cat([inputs1, outputs2], dim=1)

        # A. 主干通路前向 (保住召回率的定海神针)
        x_base = self.relu(self.bn_base(self.conv_base(x_concat)))

        # B. 偏移旁路扫描 (用极高的速度扫出上下左右偏离的扭曲划痕)
        x_offset = self.windmill_bypass(x_concat)

        # C. 残差正交注入 (基底 + 偏移追踪特征)
        x_fused = x_base + x_offset

        # D. 最终特征输出
        outputs = self.relu(self.conv_final(x_fused))

        return outputs



class LEADNet(nn.Module):
    def __init__(self, num_classes=2):
        super(LEADNet, self).__init__()

        # 1. 骨干网络 (强行关闭 drop_path 随机性)
        self.barbk = ExtremeStarNet_base(drop_path_rate=0.0)  # 👑 必须加这个参数！
        # 2. 替换为全新的 跨尺度注意力残差 解码模块
        # 参数必须拆分为: (shallow_channels, deep_channels, out_channels)
        # 根据您的实际通道分布：feat4=64, feat5=128; feat3=32; feat2=16; feat1=16
        self.up_concat4 = concat(64,128, out_size=128)
        self.up_concat3 = concat(32,128, out_size=64)
        self.up_concat2 = concat(16,64,  out_size=32)
        self.up_concat1 = concat(16,32,  out_size=32)

        # 3. 预测头 (Head) 保持精简
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

        # 自顶向下逐层解码
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        up1 = self.up_conv(up1)
        final = self.final(up1)
        return final


# --- 测试脚本 ---
if __name__ == '__main__':
    import torch
    from thop import profile

    # 1. 检查是否有可用的 GPU
    assert torch.cuda.is_available(), "必须使用 GPU 才能运行满血版 DCNv4！"
    device = torch.device('cuda')

    # 2. 实例化 LRTISS 模型，并强制挂载到 GPU 上
    model = LEADNet(num_classes=2).to(device)

    # 3. 创建模拟输入，并同步挂载到 GPU 上
    # 针对你这极端的 160x1000 工业缺陷检测尺寸
    dummy_input = torch.randn(1, 3, 460, 1280).to(device)

    print("--- 开始计算参数量与 FLOPs ---")

    # 4. 使用 thop 进行计算 (thop 在 GPU 上会自动追踪张量运算)
    flops, params = profile(model, inputs=(dummy_input,))

    print(f"Total FLOPs: {flops / 1e9:.3f} G")    # 计算量 (Giga)
    print(f"Total Params: {params / 1e6:.3f} M")  # 参数量 (Million)

    # 5. 验证一次前向传播
    out = model(dummy_input)
    print(f"Output shape: {out.shape}")