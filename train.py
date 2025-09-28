"""
!/usr/bin/env python
-*- coding: utf-8 -*-
@File:  train.py
@Author: by Chen
@Date: 2025/9/26 19:52
@last Modified by: by Chen
@last Modified time: 2025/9/27 17:49
@Desc: 训练函数
"""
import argparse
import os
import shutil

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_utils import set_seed, train_one_epoch_clip, evaluate_clip

from my_dataset import MyDataset
from clip import CLIP
import os, shutil, torch, argparse
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from tqdm import tqdm


def main(args):


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # 清理/创建保存目录
    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
    os.makedirs(args.save_path)

    # 1. Set random seed
    set_seed(args.seed)

    # 2. TensorBoard writer
    tb_writer = SummaryWriter(log_dir=args.save_path)

    # 3. Data transforms
    data_transform = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.1307]*3, [0.3081]*3)
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.1307]*3, [0.3081]*3)
        ])
    }

    # 4. Dataset & split
    full_train_dataset = MyDataset(data_path=args.data_path, is_train=True, transform=data_transform['train'])
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    print(f"Using {len(train_dataset)} images for training, {len(val_dataset)} images for validation.")

    # 5. DataLoader
    batch_size = args.batch_size
    nw = min(os.cpu_count(), batch_size if batch_size > 1 else 0, 8)
    print(f'Using {nw} dataloader workers per process')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    # 6. Model
    model = CLIP(args.init_temperature).to(device)

    # 7. Load pretrained weights if provided
    if args.weights != "":
        assert os.path.exists(args.weights), f"weights file '{args.weights}' not exist."
        state_dict = torch.load(args.weights, map_location=device)
        try:
            model.load_state_dict(state_dict, strict=True)
            print(f"Loaded pretrained weights from '{args.weights}' (strict=True).")
        except RuntimeError as e:
            print(f"Strict load failed: {e}, trying strict=False...")
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from '{args.weights}' (strict=False).")

    # 8. Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 9. Training loop
    best_acc = 0.0
    for epoch in tqdm(range(args.epochs), desc="Training Epochs"):
        # Train one epoch
        train_metrics = train_one_epoch_clip(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            tb_writer=tb_writer,
            accumulate_steps=args.accumulate_steps,
            plot_every_n_steps=args.plot_interval,
            save_logits_dir=os.path.join(args.save_path, "train_logits"),
            target_count=args.num_classes  # 每个 batch 至少覆盖类别数
        )

        scheduler.step()

        # Validate
        val_metrics = evaluate_clip(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch,
            tb_writer=tb_writer,
            plot_every_n_steps=args.plot_interval,
            save_logits_dir=os.path.join(args.save_path, "val_logits")
        )

        # TensorBoard logging
        for key, value in train_metrics.items():
            tb_writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in val_metrics.items():
            tb_writer.add_scalar(f"val/{key}", value, epoch)
        tb_writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], epoch)

        # Save checkpoints
        torch.save(model.state_dict(), os.path.join(args.save_path, f"epoch_{epoch}.pth"))

        # Save best model
        if val_metrics["i2t_acc@1"] > best_acc:
            best_acc = val_metrics["i2t_acc@1"]
            torch.save(model.state_dict(), os.path.join(args.save_path, "best_model.pth"))

        # Logging
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"i2t@1={val_metrics['i2t_acc@1']:.4f}, "
            f"t2i@1={val_metrics['t2i_acc@1']:.4f}, "
            f"best_i2t@1={best_acc:.4f}"
        )



def get_args():
    parser = argparse.ArgumentParser(description="Train CLIP on MNIST")

    # 数据与保存
    parser.add_argument('--data_path', type=str, default='./mnist', help='MNIST dataset path')
    parser.add_argument('--save_path', type=str, default='./runs', help='Path to save checkpoints and logs')

    # 训练超参数
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--accumulate_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--init_temperature', type=float, default=0.07, help='Initial temperature for CLIP')
    parser.add_argument('--plot_interval', type=int, default=100, help='Steps interval to plot similarity matrix')

    # 其他
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--weights', type=str, default='', help='Path to pretrained weights (optional)')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    main(args)


