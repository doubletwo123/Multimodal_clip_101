"""
!/usr/bin/env python
-*- coding: utf-8 -*-
@File:  img_encoder.py
@Author: by Chen
@Date: 2025/9/25 21:55
@last Modified by: by Chen
@last Modified time: 2025/9/25 21:55
@Desc:  采用resnet50作为图像编码器
"""
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn  as nn
import torch

class ImgEncoder(nn.Module):
    def __init__(self, pretrained=True, out_dim=16, pool_type="avg"):
        """
        参数:
            pretrained: 是否加载ImageNet预训练权重
            out_dim: 最终输出的特征维度
            pool_type: 池化方式 ("avg" 或 "max" 或 None)
        """
        super(ImgEncoder, self).__init__()
        if pretrained:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet50()
        # 去掉分类头 (fc + avgpool)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])  # 输出 [B, 2048, H/32, W/32]

        # 全局池化层 (控制是否输出向量)
        if pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.pool = nn.Identity()  # 保留空间特征图

        # 映射到 dim
        self.fc = nn.Linear(2048, out_dim) if pool_type is not None else nn.Conv2d(2048, out_dim, kernel_size=1)
        # sigmoid层
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.feature_extractor(x)  # [B, 2048, H/32, W/32]

        if isinstance(self.pool, nn.Identity):
            # 输出 feature map [B, dim, H/32, W/32]
            x = self.fc(x)
        else:
            # 输出向量 [B, dim]
            x = self.pool(x)        # [B, 2048, 1, 1]
            x = torch.flatten(x, 1) # [B, 2048]
            x = self.fc(x)          # [B, dim]

        return self.sigmoid(x)


if __name__ == '__main__':
    img_encoder = ImgEncoder()
    x = torch.randn(1, 3, 224, 224)
    print("x's shape is", x.shape)
    print("x 's encoder shape is", img_encoder(x).shape)

