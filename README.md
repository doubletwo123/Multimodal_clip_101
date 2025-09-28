# RAEDME
## 简介
[![image.png](https://pic1.imgdb.cn/item/68d939d3c5157e1a8840ec16.png)](https://pic1.imgdb.cn/item/68d939d3c5157e1a8840ec16.png)
这是一个基于 CLIP（Contrastive Language-Image Pretraining）模型的 MNIST 图像与文本匹配demo。该项目旨在通过对 MNIST 图像和对应文本标签进行对比学习，训练一个能够理解图像与文本之间语义关联的模型。该项目参考自B站up主[小鱼儿at青岛]的`【多模态】复现OpenAI的CLIP模型`视频，强烈推荐大家去学习一下。该项目在Nvidia 4090 24G显存上，训练50个epoch,在batch_size=32的情况下进行训练，最后训练结果和测试结果（部分）如下所示：
[![image.png](https://pic1.imgdb.cn/item/68d93c5cc5157e1a8840f0e8.png)](https://pic1.imgdb.cn/item/68d93c5cc5157e1a8840f0e8.png)
[![image.png](https://pic1.imgdb.cn/item/68d93cc0c5157e1a8840f1c7.png)](https://pic1.imgdb.cn/item/68d93cc0c5157e1a8840f1c7.png)
ps:模型的表现虽然不是特别好，但是已经能够体现出图像和文本之间的语义关联了。例如，现在有如下一组数据，并已经给出了真实标签和预测标签
[![image.png](https://pic1.imgdb.cn/item/68d93d45c5157e1a8840f2c7.png)](https://pic1.imgdb.cn/item/68d93d45c5157e1a8840f2c7.png)
- 第一行：真实标签
- 第二行：预测标签
- 上述数据的相似度矩阵（横轴为数字标签，纵轴为图片的index）如下所示：
- [![image.png](https://pic1.imgdb.cn/item/68d93d96c5157e1a8840f344.png)](https://pic1.imgdb.cn/item/68d93d96c5157e1a8840f344.png)
可以看到，模型能够较好地将图像与对应的文本标签进行匹配，尽管有些预测结果并不完全准确，但整体趋势是正确的。这表明模型在学习图像与文本之间的语义关联方面取得了一定的成功。同样，除了文本和图片这两种常见的模态，事实上还可以有音频、视频、时间序列数据等多种模态。多模态学习的目标是通过结合不同模态的信息，提升模型的理解和生成能力，从而在各种任务中取得更好的表现。**CLIP的诞生提供了一种新的思路，能够将不同模态的数据映射到同一个特征空间中，从而实现跨模态的理解和生成。**


## 项目结构

```
minist_clip/
│—— __init__.py # 包初始化文件
|—— MINIST_CLIP # 数据集
├── clip.py # CLIP 模型定义
|—— text_encoder.py # 文本编码器定义
|—— text_encoder_plus.py # 文本编码器plus。 如果需要尝试使用，请在clip.py中注释28行，启用29行即可
|—— image_encoder.py # 图像编码器定义
├── my_dataset.py # MNIST 自定义数据集
├── my_utils.py # 工具函数，如绘制相似度矩阵
├── inference.jpynb # 推理脚本（便于直接查看运行结果）
├── train.py # 训练脚本
├── runs/ # 训练结果保存目录（模型权重、日志等）
│ └── best_model.pth # 最佳模型权重
│ └── 训练日志文件 # 采用 'tensorboard --logdir runs' 查看
├── inference_test/ # 推理保存图片目录
├── requirements.txt # 依赖库
└── README.md # 项目说明文档
```

## 说明
- 该项目实现了一个基于 CLIP（Contrastive Language-Image Pretraining）的 MNIST 图像与文本匹配模型。
- 通过对 MNIST **图像**和对应**文本标签**进行对比学习，模型能够学习到图像与文本之间的语义关联。
- 项目包含数据集处理、模型定义、训练和推理等完整流程
- 训练过程中使用了对比损失函数（Contrastive Loss）来优化模型，使得相似的图像-文本对在嵌入空间中距离更近，而不相似的对距离更远。
- 训练结果和日志保存在 `runs/` 目录下，可以使用 TensorBoard
- 推理结果保存在 `inference_test/` 目录下，包含图像与文本的相似度矩阵可视化。
- 项目依赖于 PyTorch 和 torchvision 等深度学习库，具体依赖项列在 `requirements.txt` 文件中。
- PS: 该项目仅用于学习目的，数据集使用的是 MNIST，模型结构较为简单，实际应用中可能需要更复杂的模型和更大规模的数据集。
- 实际上，这里针对文本这里进行了简化，只使用了数字标签作为文本输入，在实际应用中应当使用更复杂的文本描述，例如将标签嵌入到更丰富的文本中。例如，将标签 "0" 转换为 "This is a handwritten digit 0." 以提供更多的上下文信息。则需要对于文本编码器进行更复杂的设计。
- 我们也text_encoder_plus.py中给出了一个简单的demo，具体流程是将数字标签映射到单词，并将单词嵌入到随机一个文本模板中，采用albert-base-v2进行句子的编码，随后采用了平均池化、L2归一化等操作， 得到文本的特征向量 
## 参考
  - [BERT的youxiu变体：ALBERT论文图解介绍](http://zhuanlan.zhihu.com/p/142416395)
  - [CLIP 论文逐段精读【论文精读】](https://www.bilibili.com/video/BV1SL4y1s7LQ/?spm_id_from=333.337.search-card.all.click)
  - [【多模态】复现OpenAI的CLIP模型](https://www.bilibili.com/video/BV13K421v7Ar/?spm_id_from=333.337.search-card.all.click&vd_source=16c13db1bc47b1b98f0b2e5bbe63cdbe)
  - 代码参考： https://github.com/owenliang/mnist-clip（强烈推荐）
