import argparse
import datetime
import os
from functools import partial
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
#from nets.LEA_BON import LEADNet
from nets.LEADNet import LEADNet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import NET_Dataset, dataset_collate
from utils.utils import seed_everything, show_config, worker_init_fn
from utils.utils_fit import fit_one_epoch
from nets.FDSNet import FDSNet

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')

    # ------------------ Configuration ------------------#
    parser.add_argument('--cuda', action='store_true', default=True, help='Whether to use CUDA')
    parser.add_argument('--seed', type=int, default=11, help='Random seed')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--input-shape', type=int, nargs=2, default=[1280,160],
                        help='Input image shape [H, W]')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Whether to use pretrained model')
    parser.add_argument('--model-path', type=str, default='', help='Path to pretrained model weights')

    parser.add_argument('--init-epoch', type=int, default=0, help='Initial epoch')
    parser.add_argument('--unfreeze-epoch', type=int, default=200, help='Total training epochs')
    parser.add_argument('--batch-size', type=int, default=6, help='Batch size')

    parser.add_argument('--init-lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--optimizer-type', type=str, default='adam', choices=['adam', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    parser.add_argument('--weight-decay', type=float, default=0, help='Weight decay for optimizer')

    parser.add_argument('--lr-decay-type', type=str, default='cos', choices=['cos', 'step'],
                        help='Learning rate decay type')
    parser.add_argument('--save-period', type=int, default=10, help='Period to save model checkpoints')
    parser.add_argument('--save-dir', type=str, default='logs/RSDDs1_LEADNet_3.0-1.5-0.6',  #Asymmetric  Symmetric   # F:\dataset\SAM\GSSDD\GSSDD
                        help='Directory to save logs and model checkpoints')
    parser.add_argument('--voc-path', type=str, default=r'F:\dataset\SAM\VOCdevkit_RSDDs1',   #F:\dataset\SAM\VOCdevkit_RSDDs2   [1280,64] F:\dataset\SAM\VOCdevkit_RSDDs1 1280*160  F:\dataset\SAM\VOC_BSData   460*1280 0
                        help='Path to VOCdevkit dataset')

    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--dice-loss', action='store_true', default=True, help='Use Adaptive Gaussian Dice loss')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ------------------ Seed ------------------#
    seed_everything(args.seed)
    cls_weights = np.ones(args.num_classes, np.float32)

    # ------------------ Device ------------------#
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # ------------------ Model ------------------#
    model = LEADNet(num_classes=args.num_classes).train()
    if not args.pretrained:
        weights_init(model)

        # 👑 手术 1：抢救被 weights_init 误杀的 LayerScale 软启动！
        print("=> 正在重置风车模块的 LayerScale 软启动权重 (1e-2)...")
        from nets.LEADNet import WindmillOffsetBypass  # 确保文件顶部导包

        for m in model.modules():
            if isinstance(m, WindmillOffsetBypass):
                nn.init.constant_(m.fuse_bn.weight, 1e-2)
                nn.init.constant_(m.fuse_bn.bias, 0)

    if args.model_path != '':
        print(f"Loading weights from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)

    model_train = model.to(device)

    # ------------------ Loss History ------------------#
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(args.save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=args.input_shape)

    # ------------------ Dataset ------------------#
    with open(os.path.join(args.voc_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(args.voc_path, "VOC2007/ImageSets/Segmentation/test.txt"), "r") as f:
        val_lines = f.readlines()

    num_train = len(train_lines)
    num_val = len(val_lines)

    show_config(
        num_classes=args.num_classes, backbone='LRTIIS', model_path=args.model_path,
        input_shape=args.input_shape, Init_Epoch=args.init_epoch, UnFreeze_Epoch=args.unfreeze_epoch,
        batch_size=args.batch_size, Init_lr=args.init_lr, Min_lr=args.min_lr, optimizer_type=args.optimizer_type,
        momentum=args.momentum, lr_decay_type=args.lr_decay_type, save_period=args.save_period,
        save_dir=args.save_dir, num_workers=args.num_workers, num_train=num_train, num_val=num_val
    )

    train_dataset = NET_Dataset(train_lines, args.input_shape, args.num_classes, True, args.voc_path)
    val_dataset = NET_Dataset(val_lines, args.input_shape, args.num_classes, False, args.voc_path)

    # ==========================================================
    # 👑 修复 1：创建全局绝对锁死的 DataLoader 发牌器
    # ==========================================================
    g = torch.Generator()
    g.manual_seed(args.seed)

    # 训练集：保持 drop_last=True (保护 BN 层)，加入 generator 锁死洗牌顺序
    gen = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers,
                     pin_memory=True, drop_last=True, collate_fn=dataset_collate,
                     worker_init_fn=partial(worker_init_fn, rank=0, seed=args.seed),
                     generator=g)  # 👈 必须加上这一行！

    # ==========================================================
    # 👑 修复 2：验证集必须设置 drop_last=False，严防漏算！
    # ==========================================================
    gen_val = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers,
                         pin_memory=True, drop_last=False, collate_fn=dataset_collate,  # 👈 改为 False！
                         worker_init_fn=partial(worker_init_fn, rank=0, seed=args.seed))
    # 注意：验证集 shuffle=False，所以不需要 generator

    # ------------------ Optimizer & LR ------------------#
    optimizer = {
        'adam': optim.Adam(model.parameters(), args.init_lr, betas=(args.momentum, 0.999),
                           weight_decay=args.weight_decay),
        'sgd': optim.SGD(model.parameters(), args.init_lr, momentum=args.momentum, nesterov=True,
                         weight_decay=args.weight_decay)
    }[args.optimizer_type]

    lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, args.init_lr, args.min_lr, args.unfreeze_epoch)

    # ------------------ Callbacks ------------------#
    eval_callback = EvalCallback(model, args.input_shape, args.num_classes, val_lines, args.voc_path,
                                 log_dir, args.cuda, eval_flag=True, period=5)

    # ------------------ Training Loop ------------------#
    for epoch in range(args.init_epoch, args.unfreeze_epoch):
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                      num_train // args.batch_size, num_val // args.batch_size, gen, gen_val, args.unfreeze_epoch,
                      args.cuda, args.dice_loss, cls_weights, args.num_classes,
                      save_period=args.save_period, save_dir=args.save_dir, log_dir=log_dir, local_rank=0,
                      eval_period=5)

    loss_history.writer.close()