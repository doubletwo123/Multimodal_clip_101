"""
!/usr/bin/env python
-*- coding: utf-8 -*-
@File:  text_encoder
@Author: by Chen
@Date: 2025/9/25 22:05
@last Modified by: by Chen
@last Modified time: 2025/9/25 22:05
@Desc: 这里采用简化之后的文本编码器，文本采用数字替代
"""
from torch import nn
import torch
import torch.nn.functional as F

class TextEncoder(nn.Module):
    def __init__(self, out_dim=16):
        """
        初始化文本编码器模型
        参数:
            out_dim (int): 输出特征的维度，默认为16
        """
        super().__init__()
        # 词嵌入层，将离散的token转换为密集向量
        self.embedding = nn.Embedding(10, 32)
        # 第一个全连接层，将32维输入映射到64维
        self.dense1 = nn.Linear(32, 64)
        # 第二个全连接层，将64维输入映射到32维
        self.dense2 = nn.Linear(64, 32)
        # 第三个全连接层，将32维输入映射到out_dim维
        self.dense3 = nn.Linear(32, out_dim)
        # 层归一化层，对输出进行归一化处理
        self.layernorm = nn.LayerNorm(out_dim)
        # sigmoid层
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播过程
        参数:
            x (torch.Tensor): 输入的文本token序列
        返回:
            torch.Tensor: 编码后的特征向量
        """
        # 将输入token转换为嵌入向量
        x = self.embedding(x)
        # 通过第一个全连接层并应用ReLU激活函数
        x = F.relu(self.dense1(x))
        # 通过第二个全连接层并应用ReLU激活函数
        x = F.relu(self.dense2(x))
        # 通过第三个全连接层
        x = self.dense3(x)
        # 应用层归一化
        x = self.layernorm(x)
        return self.sigmoid(x)
if __name__ == '__main__':
    text_encoder=TextEncoder()
    x=torch.tensor([1,2,3,4,5,6,7,8,9,0])
    y=text_encoder(x)
    print(y.shape)

