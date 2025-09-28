"""
!/usr/bin/env python
-*- coding: utf-8 -*-
@File:  my_utils.py
@Author: by Chen
@Date: 2025/9/26 19:56
@last Modified by: by Chen
@last Modified time: 2025/9/26 19:56
@Desc:
"""
import os
import sys
import warnings
import numpy as np
import random
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List

import torch.nn.functional as F
# 1. 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 2. 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)

# 3. 设置 numpy 错误处理
np.seterr(divide='ignore', invalid='ignore')






# Set random seed for reproducibility
def set_seed(seed=3407):
    """
    Set random seed for reproducibility.
    Args:
        seed: Random seed value (default: 3407) # why 3407? This is a trick of DL.
    Returns:

    """
    random.seed(seed) # Python built-in random module
    np.random.seed(seed) # numpy
    torch.manual_seed(seed) # torch
    torch.cuda.manual_seed(seed) # if you are using GPU
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU
    # 保证 cudnn 的确定性
    # 会强制 cuDNN 选择确定性算法，从而保证每次运行结果一致，
    # 但这通常会禁用一些高效的非确定性优化算法，因此训练速度会变慢。
    # 确定性和速度之间需要权衡，通常用于需要严格复现结果的场景。
    torch.backends.cudnn.deterministic = True # may slow down
    torch.backends.cudnn.benchmark = False  # may slow down

#绘制logits矩阵
def plot_logits_matrix(
        logits: np.ndarray,
        title: str = "Logits Matrix",
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'RdBu_r',
        annot: bool = True,
        fmt: str = '.5f',
        xticklabels: Optional[List[str]] = None,
        yticklabels: Optional[List[str]] = None,
        save_path: Optional[str] = None
) -> None:
    """
    绘制 logits 矩阵的热力图

    参数:
        logits: logits 值矩阵，形状为 (n_samples, n_classes) 或 (n_classes, n_classes)
        title: 图表标题
        figsize: 图表大小
        cmap: 颜色映射
        annot: 是否在单元格中显示数值
        fmt: 数值格式
        xticklabels: x 轴标签
        yticklabels: y 轴标签
        save_path: 保存路径（如果提供则保存图表）
    """
    # 创建图形和坐标轴
    plt.figure(figsize=figsize)

    # 绘制热力图
    ax = sns.heatmap(
        logits,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        linewidths=.5,
        cbar_kws={'label': 'Logit Value'}
    )

    # 设置标题和标签
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Predicted Class' if xticklabels is None else '', fontsize=12)
    plt.ylabel('True Class' if yticklabels is None else '', fontsize=12)

    # 设置刻度标签
    if xticklabels is not None:
        plt.xticks(np.arange(len(xticklabels)) + 0.5, xticklabels, rotation=45, ha='right')
    if yticklabels is not None:
        plt.yticks(np.arange(len(yticklabels)) + 0.5, yticklabels, rotation=0)

    # 调整布局
    plt.tight_layout()

    # 保存图表（如果需要）
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 显示图表
    plt.show()



def train_one_epoch_clip(
        model,
        optimizer,
        data_loader,
        device,
        epoch,
        tb_writer=None,
        accumulate_steps=1,
        plot_every_n_steps: Optional[int] = None,
        save_logits_dir: Optional[str] = None,
        target_count: int = 10  # 每个 batch 要覆盖的类别数
):
    """
    CLIP 训练一个 epoch，使用普通 DataLoader 并保证每个 batch 包含 target_count 个类别。

    Args:
        model: CLIP 模型，forward 返回相似度矩阵 [B, B]
        optimizer: 优化器
        data_loader: DataLoader，每个 batch 返回 images, labels
        device: 训练设备
        epoch: 当前 epoch
        tb_writer: TensorBoard writer
        accumulate_steps: 梯度累积步数
        plot_every_n_steps: 每隔多少 step 绘制一次 sim_matrix
        save_logits_dir: 保存矩阵图的路径（如果提供则保存）
        target_count: batch 内保证覆盖的类别数量

    Returns:
        metrics: dict，包含 loss, loss_i2t, loss_t2i, i2t/t2i top1/top5
    """
    model.train()
    accu_loss = accu_loss_i2t = accu_loss_t2i = 0.0
    correct_i2t_1 = correct_t2i_1 = 0
    correct_i2t_5 = correct_t2i_5 = 0
    sample_num = 0

    optimizer.zero_grad()
    step = 0

    # 使用 tqdm 包裹 data_loader
    data_loader_tqdm = tqdm(data_loader, desc=f"Train Epoch {epoch}", file=sys.stdout)

    for imgs, labels in data_loader_tqdm:
        # 确保 batch 中包含 target_count 个类别
        if torch.unique(labels).shape[0] < target_count:
            continue

        # 挑选 target_count 个类别
        selected_idx = []
        target_set = set()
        for i in range(len(labels)):
            if labels[i].item() not in target_set:
                target_set.add(labels[i].item())
                selected_idx.append(i)
            if len(target_set) == target_count:
                break

        imgs = imgs[selected_idx].to(device)
        labels = labels[selected_idx].to(device)
        batch_size = imgs.size(0)
        sample_num += batch_size

        # forward
        sim_matrix = model(imgs, labels)  # [B, B]
        targets = torch.arange(batch_size, device=device)

        # 对称交叉熵
        loss_i2t = F.cross_entropy(sim_matrix, targets)
        loss_t2i = F.cross_entropy(sim_matrix.T, targets)
        loss = (loss_i2t + loss_t2i) / 2

        # 梯度累积
        (loss / accumulate_steps).backward()
        if (step + 1) % accumulate_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # 累积
        accu_loss += loss.item() * batch_size
        accu_loss_i2t += loss_i2t.item() * batch_size
        accu_loss_t2i += loss_t2i.item() * batch_size

        # top-k accuracy
        preds_i2t_topk = sim_matrix.topk(5, dim=1).indices
        preds_t2i_topk = sim_matrix.topk(5, dim=0).indices.T
        correct_i2t_1 += (preds_i2t_topk[:, 0] == targets).sum().item()
        correct_t2i_1 += (preds_t2i_topk[:, 0] == targets).sum().item()
        correct_i2t_5 += (preds_i2t_topk == targets.unsqueeze(1)).any(dim=1).sum().item()
        correct_t2i_5 += (preds_t2i_topk == targets.unsqueeze(1)).any(dim=1).sum().item()

        step += 1

        # 更新 tqdm 显示
        data_loader_tqdm.set_postfix({
            'loss': f"{accu_loss/sample_num:.4f}",
            'i2t@1': f"{correct_i2t_1/sample_num:.4f}",
            't2i@1': f"{correct_t2i_1/sample_num:.4f}"
        })

        # 绘制 sim_matrix
        if plot_every_n_steps and (step % plot_every_n_steps == 0):
            sim_np = sim_matrix.detach().cpu().numpy()
            save_path = None
            title = f"Train Epoch {epoch} Step {step} Similarity Matrix"
            if save_logits_dir is not None:
                os.makedirs(save_logits_dir, exist_ok=True)
                save_path = os.path.join(save_logits_dir, f"train_sim_matrix_epoch{epoch}_step{step}.png")
            plot_logits_matrix(sim_np, title=title, save_path=save_path)

        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss at step {step}: {loss.item()}")

    # 平均指标
    metrics = {
        "loss": accu_loss / sample_num,
        "loss_i2t": accu_loss_i2t / sample_num,
        "loss_t2i": accu_loss_t2i / sample_num,
        "i2t_acc@1": correct_i2t_1 / sample_num,
        "t2i_acc@1": correct_t2i_1 / sample_num,
        "i2t_acc@5": correct_i2t_5 / sample_num,
        "t2i_acc@5": correct_t2i_5 / sample_num
    }

    # TensorBoard logging
    if tb_writer is not None:
        for key, value in metrics.items():
            tb_writer.add_scalar(f"train/{key}", value, epoch)
        tb_writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], epoch)

    return metrics




@torch.no_grad()
def evaluate_clip(
        model,
        data_loader,
        device,
        epoch=None,
        tb_writer=None,
        plot_every_n_steps: Optional[int] = None,
        save_logits_dir: Optional[str] = None
):
    """
    CLIP 验证/测试函数

    Args:
        model: CLIP 模型
        data_loader: DataLoader，每个 batch 返回 images, labels
        device: 训练设备
        epoch: 当前 epoch（可选，用于 tqdm 或 TensorBoard）
        tb_writer: TensorBoard writer
        plot_every_n_steps: 每隔多少 step 绘制一次 similarity matrix
        save_logits_dir: 保存 similarity matrix 的目录（None 不保存）

    Returns:
        metrics: dict，包含 loss, loss_i2t, loss_t2i, i2t_acc@1/5, t2i_acc@1/5
    """
    model.eval()
    accu_loss = accu_loss_i2t = accu_loss_t2i = 0.0
    correct_i2t_1 = correct_t2i_1 = 0
    correct_i2t_5 = correct_t2i_5 = 0
    sample_num = 0

    data_loader_iter = iter(data_loader)
    step = 0

    for images, labels in tqdm(data_loader, file=sys.stdout):
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        sample_num += batch_size

        # forward
        sim_matrix = model(images, labels)
        targets = torch.arange(batch_size, device=device)

        # 对称交叉熵
        loss_i2t = F.cross_entropy(sim_matrix, targets)
        loss_t2i = F.cross_entropy(sim_matrix.T, targets)
        loss = (loss_i2t + loss_t2i) / 2

        # 累积
        accu_loss += loss.item() * batch_size
        accu_loss_i2t += loss_i2t.item() * batch_size
        accu_loss_t2i += loss_t2i.item() * batch_size

        # top-k accuracy
        preds_i2t_topk = sim_matrix.topk(5, dim=1).indices
        preds_t2i_topk = sim_matrix.topk(5, dim=0).indices.T
        correct_i2t_1 += (preds_i2t_topk[:, 0] == targets).sum().item()
        correct_t2i_1 += (preds_t2i_topk[:, 0] == targets).sum().item()
        correct_i2t_5 += (preds_i2t_topk == targets.unsqueeze(1)).any(dim=1).sum().item()
        correct_t2i_5 += (preds_t2i_topk == targets.unsqueeze(1)).any(dim=1).sum().item()

        # tqdm 描述
        step += 1
        desc = f"loss: {accu_loss/sample_num:.4f}, i2t@1: {correct_i2t_1/sample_num:.4f}, t2i@1: {correct_t2i_1/sample_num:.4f}"
        if epoch is not None:
            tqdm.write(f"[valid epoch {epoch}] {desc}")

        # 绘制 similarity matrix
        if plot_every_n_steps and (step % plot_every_n_steps == 0):
            sim_np = sim_matrix.detach().cpu().numpy()
            save_path = None
            title = f"Eval Epoch {epoch} Step {step} Similarity Matrix"
            if save_logits_dir is not None:
                import os
                os.makedirs(save_logits_dir, exist_ok=True)
                save_path = os.path.join(save_logits_dir, f"eval_sim_matrix_epoch{epoch}_step{step}.png")
            plot_logits_matrix(sim_np, title=title, save_path=save_path)

    # 平均指标
    metrics = {
        "loss": accu_loss / sample_num,
        "loss_i2t": accu_loss_i2t / sample_num,
        "loss_t2i": accu_loss_t2i / sample_num,
        "i2t_acc@1": correct_i2t_1 / sample_num,
        "t2i_acc@1": correct_t2i_1 / sample_num,
        "i2t_acc@5": correct_i2t_5 / sample_num,
        "t2i_acc@5": correct_t2i_5 / sample_num
    }

    # TensorBoard logging
    if tb_writer is not None and epoch is not None:
        for key, value in metrics.items():
            tb_writer.add_scalar(f"val/{key}", value, epoch)

    return metrics
