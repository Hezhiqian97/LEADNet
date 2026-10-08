
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==========================================
# 1. 基础 CE (维持原样，压制大盘)
# ==========================================
class FocalLoss_CE(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super(FocalLoss_CE, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets, cls_weights=None):
        ce_loss = F.cross_entropy(logits, targets.long(), weight=cls_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = ((1.0 - pt) ** self.gamma) * ce_loss
        alpha_weight = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        return torch.mean(alpha_weight * loss)

'''
# ==========================================
# 2. 重构版 T：动态容错交叉熵 (Adaptive Margin BCE)
# 彻底解决 Dice 分母导致的 mIoU 乱跳问题
# ==========================================
class RefactoredToleranceLoss(nn.Module):
    def __init__(self, base_sigma=2.0, tolerance_drop=0.8):
        super(RefactoredToleranceLoss, self).__init__()
        self.base_sigma = base_sigma
        self.tolerance_drop = tolerance_drop

    def forward(self, probs, gt_hard):
        c = probs.size(1)
        pred_defect = probs[:, 1:, :, :]
        gt_defect = gt_hard[:, 1:, :, :]

        # 生成容错光晕 (机制保留，因为高斯扩散最合理)
        sigma = self.base_sigma
        size = int(2 * math.ceil(2 * sigma) + 1)
        x = torch.arange(size, dtype=torch.float32, device=probs.device) - size // 2
        gauss_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        gauss_2d = (gauss_1d[:, None] * gauss_1d[None, :]).unsqueeze(0).unsqueeze(0)
        gauss_2d = gauss_2d / gauss_2d.sum()

        smooth_target = F.conv2d(gt_hard, gauss_2d.expand(c, 1, size, size), padding=size // 2, groups=c)
        halo = smooth_target / (smooth_target.max() + 1e-8)
        halo_defect = halo[:, 1:, :, :]

        # 计算基础像素级 BCE
        bce_loss_map = F.binary_cross_entropy(pred_defect, gt_defect, reduction='none')

        # 构建动态惩罚权重图：
        # 对于漏检 (FN, gt=1), 必须全额惩罚，权重为 1.0
        # 对于误检 (FP, gt=0), 根据 halo_defect 动态减免惩罚
        fp_penalty_weight = 1.0 - (halo_defect * self.tolerance_drop)

        # 权重融合：真实缺陷处权重为1，背景处权重逐渐递增到1
        dynamic_weight_map = torch.where(gt_defect == 1, torch.tensor(1.0, device=gt_defect.device), fp_penalty_weight)

        # 加权平均
        return torch.mean(bce_loss_map * dynamic_weight_map)
    
'''






# ==========================================
# 2. 重构版 T：动态容错交叉熵
# Adaptive Margin BCE
# ==========================================
class RefactoredToleranceLoss(nn.Module):

    def __init__(self, sigma_l=1.0, sigma_s=1.0, tolerance_drop=0.8):

        super(RefactoredToleranceLoss, self).__init__()

        self.sigma_l = sigma_l
        self.sigma_s = sigma_s
        self.tolerance_drop = tolerance_drop


    def forward(self, probs, gt_hard):

        c = probs.size(1)

        pred_defect = probs[:, 1:, :, :]
        gt_defect = gt_hard[:, 1:, :, :]

        H = probs.size(2)
        W = probs.size(3)

        if H >= W:

            sigma_h = self.sigma_l
            sigma_w = self.sigma_s

        else:

            sigma_h = self.sigma_s
            sigma_w = self.sigma_l


        size_h = int(2 * math.ceil(2 * sigma_h) + 1)

        size_w = int(2 * math.ceil(2 * sigma_w) + 1)


        y = torch.arange(
            size_h,
            dtype=torch.float32,
            device=probs.device
        )

        y = y - size_h // 2


        x = torch.arange(
            size_w,
            dtype=torch.float32,
            device=probs.device
        )

        x = x - size_w // 2


        gauss_h = torch.exp(
            -y.pow(2) / (2 * sigma_h ** 2)
        )


        gauss_w = torch.exp(
            -x.pow(2) / (2 * sigma_w ** 2)
        )


        gauss_2d = gauss_h[:, None] * gauss_w[None, :]

        gauss_2d = gauss_2d.unsqueeze(0)

        gauss_2d = gauss_2d.unsqueeze(0)


        gauss_2d = gauss_2d / gauss_2d.sum()


        gaussian_kernel = gauss_2d.expand(
            c,
            1,
            size_h,
            size_w
        )


        smooth_target = F.conv2d(
            gt_hard,
            gaussian_kernel,
            padding=(
                size_h // 2,
                size_w // 2
            ),
            groups=c
        )


        max_value = smooth_target.max()

        halo = smooth_target / (
            max_value + 1e-8
        )


        halo_defect = halo[:, 1:, :, :]


        bce_loss_map = F.binary_cross_entropy(
            pred_defect,
            gt_defect,
            reduction='none'
        )


        fp_penalty_weight = 1.0 - (
            halo_defect * self.tolerance_drop
        )


        full_penalty = torch.tensor(
            1.0,
            device=gt_defect.device
        )


        dynamic_weight_map = torch.where(
            gt_defect == 1,
            full_penalty,
            fp_penalty_weight
        )


        weighted_loss = (
            bce_loss_map * dynamic_weight_map
        )


        tolerance_loss = torch.mean(
            weighted_loss
        )


        return tolerance_loss







# ==========================================
# 3. 重构版边界：拉普拉斯锐化损失 (Laplacian Edge Loss)
# 纯空间域微分，强迫网络学习 1 像素的锐利度，替代臃肿的频域
# ==========================================
class LaplacianEdgeLoss(nn.Module):
    def __init__(self):
        super(LaplacianEdgeLoss, self).__init__()
        # 定义拉普拉斯二阶微分卷积核
        kernel = torch.tensor([[[[0.0, 1.0, 0.0],
                                 [1.0, -4.0, 1.0],
                                 [0.0, 1.0, 0.0]]]])
        # 注册为 buffer，这样它会自动转移到 GPU，且不会被优化器更新
        self.register_buffer('laplacian_kernel', kernel)

    def forward(self, probs, gt_hard):
        pred_defect = probs[:, 1:, :, :]
        gt_defect = gt_hard[:, 1:, :, :]
        C = pred_defect.shape[1]
        weight = self.laplacian_kernel.expand(C, 1, 3, 3)
        edge_pred = F.conv2d(pred_defect, weight, padding=1, groups=C)
        edge_gt = F.conv2d(gt_defect, weight, padding=1, groups=C)
        # 提取预测概率图的物理边缘 (锐利度)
       # edge_pred = F.conv2d(pred_defect, self.laplacian_kernel, padding=1)
        # 提取真实标签的物理边缘 (绝对尖锐)
       # edge_gt = F.conv2d(gt_defect, self.laplacian_kernel, padding=1)

        # L1 距离：你的边缘必须像真实标签一样尖锐，越模糊，惩罚越大
        return F.l1_loss(torch.abs(edge_pred), torch.abs(edge_gt))


# ==========================================
# 终极重构调度器：CE + a*T(平滑版) + b*拉普拉斯边缘
# ==========================================
class UnifiedRefactoredLoss(nn.Module):
    def __init__(self, a=1, b=2):
        """
        重构后的参数变得极其简单好调：
        a=1.0: 容错损失权重 (保 Recall)
        b=2.0: 拉普拉斯锐化权重 (保 Precision，因为它算出来的 L1 比较小，可以直接给到 2.0~5.0)
        """
        super(UnifiedRefactoredLoss, self).__init__()
        self.weight_a = a
        self.weight_b = b

        self.loss_ce = FocalLoss_CE()
        self.loss_t = RefactoredToleranceLoss(sigma_l=3.0,sigma_s=1.5, tolerance_drop=0.6)
        self.loss_edge = LaplacianEdgeLoss()

    def forward(self, logits, targets, cls_weights=None):
        n, c, h, w = logits.size()

        if targets.dim() == 4:
            targets = targets.squeeze(1)
        if targets.size()[1:] != logits.size()[2:]:
            targets = F.interpolate(targets.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1).long()

        probs = torch.softmax(logits, dim=1)
        gt_hard = F.one_hot(targets.long(), num_classes=c).permute(0, 3, 1, 2).float()

        ce = self.loss_ce(logits, targets, cls_weights)
        t = self.loss_t(probs, gt_hard)
        edge = self.loss_edge(probs, gt_hard)

        # 由于三者全部统一在了 BCE 和 L1 这种稳定的空间尺度下
        # 它们再也不会出现梯度相互吞噬、或者某一项突然爆炸的情况。
        total_loss = ce + self.weight_a * t + self.weight_b * edge

        return total_loss






















def CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss




class TrueToleranceDiceLoss(nn.Module):
    def __init__(self, beta=0.8, smooth=1e-5, base_sigma=2.0, tolerance_drop=0.8):
        """
        死守 83.21% 阵地的终极微调版

        核心变动：
        - base_sigma=2.0, tolerance_drop=0.8：保持完美的高斯容错安全网绝对不变。
        - beta=0.8：打破 F1 的 1:1 诅咒。beta < 1 意味着在全局尺度上，
          稍微加重对 FP(误检) 的惩罚，从而在不破坏安全网的情况下拉升 Precision。
        """
        super(TrueToleranceDiceLoss, self).__init__()
        self.beta = beta
        self.smooth = smooth
        self.base_sigma = base_sigma
        self.tolerance_drop = tolerance_drop

    def forward(self, inputs, targets):
        n, c, h, w = inputs.size()

        # 1. 尺寸对齐
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        if targets.size()[1:] != inputs.size()[2:]:
            targets = F.interpolate(targets.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1).long()

        inputs_probs = torch.softmax(inputs, dim=1)
        gt_hard = F.one_hot(targets.long(), num_classes=c).permute(0, 3, 1, 2).float()

        # 2. 生成高斯容错光晕 (原汁原味，不收紧！)
        sigma = self.base_sigma
        size = int(2 * math.ceil(2 * sigma) + 1)
        x = torch.arange(size, dtype=torch.float32, device=inputs.device) - size // 2
        gauss_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        gauss_2d = (gauss_1d[:, None] * gauss_1d[None, :]).unsqueeze(0).unsqueeze(0)
        gauss_2d = gauss_2d / gauss_2d.sum()

        smooth_target = F.conv2d(gt_hard, gauss_2d.expand(c, 1, size, size), padding=size // 2, groups=c)
        halo = smooth_target / (smooth_target.max() + 1e-8)

        # 3. 剥离背景
        pred_defect = inputs_probs[:, 1:, :, :]
        gt_defect_hard = gt_hard[:, 1:, :, :]
        halo_defect = halo[:, 1:, :, :]

        # 4. 核心数学重构
        # TP 和 FN 依然使用硬标签
        tp = torch.sum(pred_defect * gt_defect_hard, dim=[2, 3])
        fn = torch.sum((1.0 - pred_defect) * gt_defect_hard, dim=[2, 3])

        # FP 使用宽广的容错网
        fp_raw = pred_defect * (1.0 - gt_defect_hard)
        fp_penalty_map = 1.0 - (halo_defect * self.tolerance_drop)
        fp_weighted = torch.sum(fp_raw * fp_penalty_map, dim=[2, 3])

        # 5. 计算 Dice (依靠 self.beta=0.8 施加全局 Precision 偏好)
        score = ((1 + self.beta ** 2) * tp + self.smooth) / (
                (1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp_weighted + self.smooth)

        return torch.mean(1.0 - score)







def CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss


def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs,
                                                                                                 temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss

def Adaptive_Gaussian_Dice_Loss(inputs, target, beta=1, smooth=1e-5, base_sigma=2.0):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        h, w = ht, wt

    temp_inputs = torch.softmax(inputs, dim=1)
    target_map = target.permute(0, 3, 1, 2).float()  # [n, ct, h, w]

    # ===== 只取前景部分 =====
    if ct > c:
        target_map = target_map[:, :c, :, :]  # 去掉背景通道
        ct = c

    # ---- 边缘检测 ----
    edge_kernel_x = torch.tensor([[1, 0, -1],
                                  [2, 0, -2],
                                  [1, 0, -1]], dtype=torch.float32, device=inputs.device).view(1, 1, 3, 3)
    edge_kernel_y = torch.tensor([[1, 2, 1],
                                  [0, 0, 0],
                                  [-1, -2, -1]], dtype=torch.float32, device=inputs.device).view(1, 1, 3, 3)
    edge = torch.sqrt(
        F.conv2d(target_map.sum(dim=1, keepdim=True), edge_kernel_x, padding=1).pow(2) +
        F.conv2d(target_map.sum(dim=1, keepdim=True), edge_kernel_y, padding=1).pow(2)
    )
    edge_weight = torch.exp(-4 * edge)
    edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min() + 1e-6)

    # ---- 自适应 σ ----
    aspect_ratio = max(h, w) / max(1.0, min(h, w))
    sigma = base_sigma * (0.5 + 0.5 * torch.tanh(torch.tensor(aspect_ratio / 4.0, device=inputs.device)))
    size = int(2 * math.ceil(2 * sigma.item()) + 1)

    x = torch.arange(size, dtype=torch.float32, device=inputs.device) - size // 2
    gauss_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    gauss_2d = (gauss_1d[:, None] * gauss_1d[None, :]).unsqueeze(0).unsqueeze(0)
    gauss_2d = gauss_2d / gauss_2d.sum()

    # ---- 高斯平滑 ----
    smooth_target = F.conv2d(target_map, gauss_2d.expand(ct, 1, size, size), padding=size // 2, groups=ct)
    smooth_target = edge_weight * smooth_target + (1 - edge_weight) * target_map

    # ---- Dice 计算 ----
    tp = torch.sum(smooth_target * temp_inputs, dim=[0, 2, 3])
    fp = torch.sum(temp_inputs, dim=[0, 2, 3]) - tp
    fn = torch.sum(smooth_target, dim=[0, 2, 3]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss

def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # --------------------------------------------#
    #   计算dice loss
    # --------------------------------------------#
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
