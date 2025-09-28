"""
!/usr/bin/env python
-*- coding: utf-8 -*-
@File:  mydataset.py
@Author: by Chen
@Date: 2025/9/26 20:25
@last Modified by: by Chen
@last Modified time: 2025/9/26 20:25
@Desc: 
"""
import os
import warnings

import numpy as np
from nltk import download
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor,Compose
import torchvision
# 1. 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 2. 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)

# 3. 设置 numpy 错误处理
np.seterr(divide='ignore', invalid='ignore')


import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, PILToTensor
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, data_path='./mnist/', is_train=True, transform=None):
        """
        自定义 MNIST 数据集
        Args:
            data_path: 数据存储路径
            is_train: 是否为训练集
            transform: 图像变换
        """
        super().__init__()
        self.ds = torchvision.datasets.MNIST(
            root=data_path,
            train=is_train,
            download=True
        )
        self.targets = self.ds.targets  # 暴露标签，供 BalancedBatchSampler 使用

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.PILToTensor()])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        img, label = self.ds[index]
        if self.transform:
            img = self.transform(img)
        return img, label


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ds = MyDataset()
    img, label = ds[0]
    print(label)
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
