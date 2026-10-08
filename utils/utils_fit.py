import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from nets.unet_training import CE_Loss, Dice_loss
from utils.utils import get_lr
from utils.utils_metrics import f_score
from nets.unet_training import TrueToleranceDiceLoss as TrueToleranceDiceLoss
from nets.unet_training import UnifiedRefactoredLoss as DecoupledComboLoss

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


#

# ==============================================================================
# 训练与验证 - 标准流程 (包含 Val)
# ==============================================================================
def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, cuda, agd_dice_loss, cls_weights, num_classes, save_period,
                  save_dir, log_dir, local_rank=0, eval_period=5, base_sigma=2.0,
                  min_alpha=0.4, max_alpha=0.8):
    total_loss, total_f_score = 0, 0
    val_loss, val_f_score = 0, 0
    weights = torch.from_numpy(cls_weights)
    atg_criterion = TrueToleranceDiceLoss().cuda()
    DecoupledComboLosss = DecoupledComboLoss().cuda()

    # ----------------------#
    #   1. Train 阶段
    # ----------------------#
    if local_rank == 0:
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch} [Train]', postfix=dict, mininterval=0.3)

    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: break
        imgs, pngs, labels = batch

        if cuda:
            imgs, pngs, labels = imgs.cuda(local_rank), pngs.cuda(local_rank), labels.cuda(local_rank)
            weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        outputs = model_train(imgs)

        # 👑 极简静态 Loss 计算：固定使用 max_alpha 作为 CE 的权重
        loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)
        if agd_dice_loss:
           # loss_agd = atg_criterion(outputs, pngs)
           # losss_boundary = loss_boundary(outputs, pngs,weights)

            loss = DecoupledComboLosss(outputs, pngs,weights)
        with torch.no_grad():
            _f_score = f_score(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(total_loss=total_loss / (iteration + 1), f_score=total_f_score / (iteration + 1),
                             lr=get_lr(optimizer))
            pbar.update(1)

    if local_rank == 0: pbar.close()

    # ----------------------#
    #   2. Valid 阶段
    # ----------------------#
    if local_rank == 0:
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch} [Valid]', postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val: break
        imgs, pngs, labels = batch

        with torch.no_grad():
            if cuda:
                imgs, pngs, labels = imgs.cuda(local_rank), pngs.cuda(local_rank), labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            outputs = model_train(imgs)

            # 👑 极简静态 Loss 计算
            loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)
            if agd_dice_loss:
                # loss_agd = atg_criterion(outputs, pngs)
                # losss_boundary = loss_boundary(outputs, pngs,weights)

                loss = DecoupledComboLosss(outputs, pngs, weights)

            _f_score = f_score(outputs, labels)

            val_loss += loss.item()
            val_f_score += _f_score.item()

            if local_rank == 0:
                pbar.set_postfix(val_loss=val_loss / (iteration + 1), f_score=val_f_score / (iteration + 1),
                                 lr=get_lr(optimizer))
                pbar.update(1)

    if local_rank == 0: pbar.close()

    # ----------------------#
    #   3. 回调与最高分保存 (👑 修复历史污染 Bug + mIoU精准比对)
    # ----------------------#
    if local_rank == 0:
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)

        # 【触发回调】：此处会自动计算测试集 mIoU 并写入 epoch_miou.txt
        eval_callback.on_epoch_end(epoch + 1, model_train)

        print(
            f"Epoch: {epoch + 1}/{Epoch} || Total Loss: {total_loss / epoch_step:.3f} || Val Loss: {val_loss / epoch_step_val:.3f}")

        if ((epoch + 1) % eval_period) == 0:
            miou_file_path = os.path.join(log_dir, "epoch_miou.txt")
            if os.path.exists(miou_file_path):
                with open(miou_file_path, 'r') as file:
                    lines = file.readlines()

                if lines:
                    # 1. 强制转为 float 浮点数，杜绝字符串比较陷阱
                    miou_list = [float(line.strip()) for line in lines if line.strip()]

                    # 2. 计算本次训练进行了多少次验证
                    current_eval_count = (epoch + 1) // eval_period

                    # 3. 切片截取：只保留属于“本次训练”的 mIoU 记录，屏蔽上一次/昨天的旧数据
                    current_run_mious = miou_list[-current_eval_count:]

                    if len(current_run_mious) > 0:
                        current_miou = current_run_mious[-1]

                        # 4. 判断：如果是本次训练的第一轮验证，或者打破了【本次训练】的历史最高分
                        if len(current_run_mious) == 1 or current_miou > max(current_run_mious[:-1]):
                            print(f'==========> 突破本次训练最高分! (mIoU: {current_miou:.2f}) <==========')
                            print('Save best model to best_epoch_weights.pth')
                            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        # 每轮都更新 last_epoch_weights 以防意外中断
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))


# ==============================================================================
# 训练 - 无验证流程 (通常用于测试或不开启 eval 时)
# ==============================================================================
def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda,
                         agd_dice_loss, cls_weights, num_classes, save_period, save_dir, local_rank=0,
                         base_sigma=2.0, min_alpha=0.4, max_alpha=0.8):
    total_loss, total_f_score = 0, 0
    weights = torch.from_numpy(cls_weights)
    atg_criterion = TrueToleranceDiceLoss().cuda()
    DecoupledComboLosss = DecoupledComboLoss().cuda()
    if local_rank == 0:
        print('Start Train (No Validation)')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: break
        imgs, pngs, labels = batch

        if cuda:
            imgs, pngs, labels = imgs.cuda(local_rank), pngs.cuda(local_rank), labels.cuda(local_rank)
            weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        outputs = model_train(imgs)

        # 👑 极简静态 Loss 计算
        loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)
        if agd_dice_loss:
            # loss_agd = atg_criterion(outputs, pngs)
            # losss_boundary = loss_boundary(outputs, pngs,weights)

            loss = DecoupledComboLosss(outputs, pngs, weights)

        with torch.no_grad():
            _f_score = f_score(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(total_loss=total_loss / (iteration + 1), f_score=total_f_score / (iteration + 1),
                             lr=get_lr(optimizer))
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, total_loss / epoch_step)
        print(f"Epoch: {epoch + 1}/{Epoch} || Total Loss: {total_loss / epoch_step:.3f}")

        # 基于 loss 的保存逻辑
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f'ep{(epoch + 1):03d}-loss{(total_loss / epoch_step):.3f}.pth'))

        if len(loss_history.losses) <= 1 or (total_loss / epoch_step) <= min(loss_history.losses[:-1]):
            print('Save best model to best_epoch_weights.pth based on training loss')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))