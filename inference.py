"""
!/usr/bin/env python
-*- coding: utf-8 -*-
@File:  inference.py
@Author: by Chen
@Date: 2025/9/28 16:00
@last Modified by: by Chen
@last Modified time: 2025/9/28 16:00
@Desc: 推理函数
"""
from clip import CLIP
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import cv2

if __name__ == 'main':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.1307] * 3, [0.3081] * 3)
    ])

    img = cv2.imread('./test.jpg')
    # 转为灰度图
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    img = transform(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # 构造文本输入
    labels = torch.arange(10).unsqueeze(0).to(device)  # [1, 10]
    print('labels is ',labels)

    # 加载模型
    model = CLIP().to(device)
    model.load_state_dict(torch.load('./best_model.pth', map_location=device))

    # 推理
    model.eval()
    with torch.no_grad():
        logits = model(img, labels)  # [1, 10]

    # 取出相似度最高的类别
    pred = logits.argmax(dim=-1).item()
    print(f'Predicted label: {pred}')



