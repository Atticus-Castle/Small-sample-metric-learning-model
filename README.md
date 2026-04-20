# 铁路轨道小样本故障检测（Few-shot Prototypical Network）

## 1. 项目简介

本项目针对铁路运维场景中故障样本稀缺、设备差异大、模型泛化能力不足的问题，提出了一种轻量化小样本故障检测方法。

方法核心包括：

- 基于 MobileNetV3-Small 的轻量化特征提取网络
- 多尺度特征融合（第 4、7、11 层）
- 原型网络（Prototypical Network）进行小样本分类
- 128 维归一化嵌入空间 + 欧氏距离度量

该方法在小样本条件下具有良好的分类性能与泛化能力。

## 2. 方法概述

### 2.1 模型结构

整体流程如下：

输入图像 -> MobileNetV3-Small -> 多层特征提取（4/7/11）  
-> GAP -> 特征拼接 -> 投影头（128维）  
-> L2 归一化 -> 原型网络分类

### 2.2 多尺度特征融合

- 第 4 层：纹理/边缘信息（浅层）
- 第 7 层：局部结构信息（中层）
- 第 11 层：语义信息（深层）

融合方式：

- Global Average Pooling（GAP）
- 向量拼接
- 非线性投影（MLP）

### 2.3 原型网络分类

- 支持集：构建类别原型（均值）
- 查询集：计算与各原型的欧氏距离
- 使用交叉熵损失训练

## 3. 项目结构

```text
project/
├── models/
│   ├── backbone.py        # MobileNetV3-Small
│   ├── fusion.py          # 多尺度融合
│   └── proto_net.py       # 原型网络
│
├── datasets/
│   └── railway_dataset.py
│
├── utils/
│   ├── loss.py
│   ├── metrics.py
│   └── sampler.py         # N-way K-shot任务构建
│
├── train.py
├── test.py
├── config.py
└── README.md
```

## 4. 环境配置

### 4.1 硬件环境

- GPU：NVIDIA RTX 5090（或其他支持 CUDA 的 GPU）
- CPU：Intel Xeon / i7 / i9

### 4.2 软件环境

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 11.8

安装依赖：

```bash
pip install torch torchvision numpy matplotlib tqdm
```

## 5. 数据集说明

### 5.1 数据来源

铁路轨道图像数据集（如紧固件、枕木等部件的缺陷与非缺陷图像）。

### 5.2 数据组织方式

本项目代码默认读取如下路径（与当前 `config.py` 一致）：

```text
dataset/
├── Train/
│   ├── Defective/
│   └── Non defective/
├── Validation/
│   ├── Defective/
│   └── Non defective/
└── Test/
    ├── Defective/
    └── Non defective/
```

> 如果你希望使用 `train/val/test` 小写命名，可在 `config.py` 中修改对应路径。

### 5.3 数据增强

- 随机裁剪（224 x 224，训练阶段）
- 随机水平翻转（训练阶段）
- 标准化（ImageNet mean/std）

## 6. 使用方法

### 6.1 训练模型

```bash
python train.py
```

> 当前版本主要通过 `config.py` 管理参数（如 `n_way`、`k_shot`、`n_query`、`num_epochs`）。

### 6.2 测试模型

```bash
python test.py
```

### 6.3 小样本任务设置

- N-way：类别数（如 2 类）
- K-shot：每类支持集样本数（如 5-shot）
- Query：每类查询集样本数

## 7. 评价指标

当前测试脚本输出以下指标：

- Accuracy
- Precision（macro）
- Recall（macro）
- F1 Score（macro）

你可以在 `utils/metrics.py` 中扩展为 binary/micro/weighted 等统计方式。

## 8. 实验结果（示例）

> 以下为示例展示，具体数值请以你本地训练与测试结果为准。

### 8.1 分类结果（示例）

| Model | Precision | Recall | F1 |
| --- | --- | --- | --- |
| OUR MODEL | 0.95 | 0.95 | 0.95 |

### 8.2 ROC（示例）

- AUC = 0.834
- 明显优于随机分类基线

### 8.3 消融实验（示例）

| 特征组合 | Precision | Recall |
| --- | --- | --- |
| (4,7) | 91.1 | 93.0 |
| (4,11) | 94.5 | 94.5 |
| (7,11) | 92.6 | 93.7 |

## 9. 项目特点

- 轻量化（MobileNetV3）
- 小样本适应能力强
- 多尺度特征融合
- 可扩展至多设备场景

## 10. 可扩展方向

- 多模态融合（图像 + 振动信号）
- 跨设备迁移学习
- 实时检测系统部署
