# 🚀 翻译模型从零开始训练项目

## 📋 项目简介

这是一个从零开始实现翻译模型的完整项目，专为 Mac M4 + 32GB 内存环境优化。项目包含多个版本，从简单到复杂，适合不同学习阶段。

## 🎯 核心功能

- ✅ **多种模型架构**：GRU、LSTM、Transformer
- ✅ **稳定训练**：解决 NaN 问题，确保训练成功
- ✅ **硬件适配**：针对 Mac M4 优化，支持 MPS 加速
- ✅ **完整流程**：数据处理、模型训练、推理部署

## 🚀 快速开始

### 新手推荐（最稳定）

```bash
python final_stable_demo.py
```

### 进阶学习

```bash
# LSTM版本
python simple_demo.py

# Transformer版本
python stable_transformer.py
```

### 完整训练

```bash
python src/training/train.py --config configs/train_config.yaml
```

## 📊 版本对比

| 版本                    | 架构        | 稳定性     | 参数数量 | 推荐程度   |
| ----------------------- | ----------- | ---------- | -------- | ---------- |
| `final_stable_demo.py`  | GRU         | ⭐⭐⭐⭐⭐ | 15,808   | ⭐⭐⭐⭐⭐ |
| `simple_demo.py`        | LSTM        | ⭐⭐⭐⭐⭐ | 72,800   | ⭐⭐⭐⭐   |
| `stable_transformer.py` | Transformer | ⭐⭐⭐⭐   | 208,884  | ⭐⭐⭐     |

## 📁 项目结构

```
model_training/
├── final_stable_demo.py      # 最稳定版本（GRU）
├── simple_demo.py            # 简化版本（LSTM）
├── stable_transformer.py     # 稳定版本（Transformer）
├── src/                      # 完整项目源码
│   ├── models/              # 模型定义
│   ├── data/                # 数据处理
│   ├── training/            # 训练相关
│   └── inference.py         # 推理脚本
├── configs/                 # 配置文件
├── requirements.txt         # 依赖包
└── 文档/                    # 详细文档
```

## 🎓 学习路径

1. **第 1 周**：运行 `final_stable_demo.py`，理解基础概念
2. **第 2-3 周**：运行 `simple_demo.py`，学习 LSTM 原理
3. **第 4-6 周**：运行 `stable_transformer.py`，理解注意力机制

## 📚 详细文档

- [使用指南](USAGE_GUIDE.md) - 详细的使用说明
- [项目总结](FINAL_SUMMARY.md) - 完整的项目总结
- [精简文档](FINAL_CLEAN_DOCUMENT.md) - 精简版文档

## 🔧 环境要求

- Python 3.8+
- PyTorch 2.0+
- Mac M 系列芯片（推荐 M4）

## 📦 安装依赖

```bash
pip install -r requirements.txt
```

## 🎉 项目特色

- **教育价值**：完整的深度学习项目开发流程
- **实用价值**：可扩展的代码框架，针对特定硬件优化
- **研究价值**：为后续研究提供基础，可复现的实验设置

## 🏆 学习收获

通过这个项目，你将掌握：

1. 深度学习模型设计的核心原理
2. 工程化的项目开发方法
3. 针对特定硬件的优化技巧
4. 完整的训练和评估流程

---

**开始你的深度学习之旅吧！** 🚀
