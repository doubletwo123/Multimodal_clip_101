"""
!/usr/bin/env python
-*- coding: utf-8 -*-
@File:  text_encoder_plus
@Author: by Chen
@Date: 2025/9/28 19:46
@last Modified by: by Chen
@last Modified time: 2025/9/28 19:46
@Desc: 这里并未直接采用标签作为文本，而是采用了更加复杂的方式进行了编排。首先将标签嵌入到我们的模板中，然后再采用文本编码器进行编码。这里我们采用了bert-base-uncased作为文本编码器。
"""
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


# 设置环境变量，指定模型缓存目录和禁用离线模式
os.environ['HF_HOME'] = 'D:/huggingface'  # 模型缓存目录，可选
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_DATASETS_OFFLINE'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# 设置嵌入句子模板
sentence = ['this is a written digit {}',
            'this is a photo of digit {}',
            'this is a image of digit {}',
            'the digit is {}',
            'the number is {}',
            'the number of digit is {}',
            'the digit in the image is {}',
            'the digit in the photo is {}',
            'the digit in the picture is {}',
            'the digit shown is {}']

# 数字到单词的映射
num_to_word = {
    0: 'zero',
    1: 'one',
    2: 'two',
    3: 'three',
    4: 'four',
    5: 'five',
    6: 'six',
    7: 'seven',
    8: 'eight',
    9: 'nine'
}


class ALBertTextEncoder(nn.Module):
    def __init__(self, pretrained_model_name='albert-base-v2', out_dim=16, device='cuda', freeze_albert=False):
        """
        优化版 ALBERT 句子编码器
        参数:
            pretrained_model_name (str): 预训练 ALBERT 模型名
            out_dim (int): 输出特征维度
            device (str): 运行设备
            freeze_albert (bool): 是否冻结 ALBERT 参数
        """
        super().__init__()
        self.device = device
        self.albert = AutoModel.from_pretrained(pretrained_model_name, mirror="tuna" )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, mirror="tuna")

        # 冻结 ALBERT 参数（如果需要）
        if freeze_albert:
            for param in self.albert.parameters():
                param.requires_grad = False

        self.albert.to(device)
        # 映射到目标维度
        self.dense = nn.Linear(self.albert.config.hidden_size, out_dim)
        self.layernorm = nn.LayerNorm(out_dim)

    def forward(self, x):
        """
        前向传播
        参数:
            x : 输入数字标签
        返回:
            torch.Tensor: L2 归一化的句子嵌入 [batch_size, out_dim]
        """
        # 将数字嵌入到模板中
        texts = x.tolist()  # tensor -> Python list
        texts = [random.choice(sentence).format(num_to_word[int(t)]) for t in texts]

        # 对文本进行编码
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        # 迁移到模型所在设备
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        # ALBERT 输出
        outputs = self.albert(**encoded_input)  # last_hidden_state: [B, L, H]
        hidden_states = outputs.last_hidden_state

        # Mean Pooling: 考虑 attention_mask
        attention_mask = encoded_input['attention_mask'].unsqueeze(-1)  # [B, L, 1]
        masked_hidden = hidden_states * attention_mask
        sum_hidden = masked_hidden.sum(1)
        lengths = attention_mask.sum(1).clamp(min=1e-9)
        sentence_embeddings = sum_hidden / lengths  # [B, H]

        # 映射到目标维度
        x = self.dense(sentence_embeddings)
        x = self.layernorm(x)

        # L2 归一化
        return F.normalize(x, p=2, dim=1)


if __name__ == "__main__":
    DEVICE = 'cuda'
    model = ALBertTextEncoder(out_dim=16, device=DEVICE, freeze_albert=False)
    model.to(DEVICE)
    x = torch.randint(0, 10, (5,)).to(device=DEVICE)
    embeddings = model(x)
    print(embeddings.shape)  # [10, 16]
    print(embeddings)  # 打印归一化后的句子向量
