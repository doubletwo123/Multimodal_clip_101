"""
!/usr/bin/env python
-*- coding: utf-8 -*-
@File:  clip
@Author: by Chen
@Date: 2025/9/25 22:16
@last Modified by: by Chen
@last Modified time: 2025/9/27 22:55
@Desc: 进行简单的点积求相似度
@Log: 1.bug:求得的相似度矩阵的元素居然有负数？？？
"""
import numpy as np
from torch import nn
import torch
from img_encoder import ImgEncoder
from text_encoder import TextEncoder
import torch.nn.functional as F
from my_utils import plot_logits_matrix
from text_encoder_plus import ALBertTextEncoder


class CLIP(nn.Module):
    def __init__(self, init_temperature=0.07):

        # 初始化函数
        super().__init__()  # 调用父类nn.Module的初始化函数
        self.img_enc = ImgEncoder()  # 初始化图像编码器
        self.text_enc = TextEncoder()  # 初始化文本编码器
        # self.text_enc = ALBertTextEncoder()  # 使用优化版的文本编码器
        self.log_temperature = nn.Parameter(torch.ones([]) * np.log(init_temperature)) # 设置可学习的温度参数

    def forward(self, img_x, text_x):

        # 前向传播函数
        # img_x: 输入的图像数据
        # text_x: 输入的文本数据
        img_emb = self.img_enc(img_x)  # 通过图像编码器获取图像特征
        # print("图像特征编码之后的形状：",img_emb.shape)
        text_emb = self.text_enc(text_x)  # 通过文本编码器获取文本特征
        # print("文本特征编码之后的形状：",text_emb.shape)
        # 归一化到单位向量
        img_emb = F.normalize(img_emb, p=2, dim=-1)  # [N_img, D]
        text_emb = F.normalize(text_emb, p=2, dim=-1)  # [N_text, D]
        # 计算余弦相似度矩阵
        sim_matrix = (img_emb @ text_emb.T) * torch.exp(self.log_temperature)# [N_img, N_text]  #   # 计算图像特征和文本特征的余弦相似度矩阵
        return  sim_matrix

if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip = CLIP().to(DEVICE)
    img_x = torch.randn(5, 3, 28, 28).to(DEVICE)
    text_x = torch.randint(0, 10, (5,)).to(DEVICE)
    print("img_x's shape is", img_x.shape)
    print("text_x's shape is", text_x.shape)
    logits=clip(img_x,text_x)
    print("logits's shape is", logits.shape)
    x_names = ['img 1', 'img 2', 'img 3', 'img 4', 'img 5']
    y_names = ['text 1', 'text 2', 'text 3', 'text 4', 'text 5']

    # print(logits)
    # 绘制 logits 矩阵
    plot_logits_matrix(
        logits.detach().cpu().numpy(),
        title="Example Logits Matrix",
        xticklabels=x_names,
        yticklabels=y_names,
        save_path="logits_matrix.png"
    )

