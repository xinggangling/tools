# 简单 Transformer 模型实现

这个项目实现了一个简单的 Transformer 模型，用于识别 MNIST 数据集中的数字 1 和 2。该项目主要用于学习 Transformer 的基本概念和实现。

## 项目结构

```
.
├── simple_transformer.py    # Transformer模型的核心实现
├── train_digit_transformer.py  # 训练脚本
├── requirements.txt         # 项目依赖
└── README.md               # 项目文档
```

## 核心组件说明

### 1. 位置编码 (PositionalEncoding)

位置编码是 Transformer 的重要组成部分，用于为序列中的每个位置添加位置信息。

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        # d_model: 模型的维度
        # max_len: 序列的最大长度
```

主要功能：

- 使用正弦和余弦函数生成位置编码
- 为每个位置生成唯一的编码向量
- 帮助模型理解序列中元素的位置关系

### 2. 多头注意力机制 (MultiHeadAttention)

多头注意力机制允许模型同时关注输入序列的不同部分。

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        # d_model: 模型的维度
        # num_heads: 注意力头的数量
```

主要功能：

- 将输入分成多个头，每个头独立计算注意力
- 实现缩放点积注意力机制
- 合并多个头的结果得到最终输出

### 3. 数字识别 Transformer (DigitTransformer)

整合上述组件，构建完整的模型用于数字识别。

```python
class DigitTransformer(nn.Module):
    def __init__(self, d_model=64, num_heads=4):
        # d_model: 模型的维度
        # num_heads: 注意力头的数量
```

主要功能：

- 输入处理：将 28x28 的图像转换为序列
- 位置编码：添加位置信息
- 注意力计算：使用多头注意力机制
- 输出层：进行二分类（1 和 2）

## 训练流程

### 1. 数据准备

```python
def get_mnist_data():
    # 加载MNIST数据集
    # 只选择数字1和2
    # 进行数据预处理和标准化
```

### 2. 模型训练

```python
def train_model(model, train_loader, criterion, optimizer, device):
    # 训练一个epoch
    # 计算损失和准确率
    # 更新模型参数
```

### 3. 模型评估

```python
def evaluate_model(model, test_loader, device):
    # 在测试集上评估模型
    # 计算准确率
```

### 4. 训练可视化

```python
def plot_training_history(train_losses, train_accs, test_accs):
    # 绘制训练损失曲线
    # 绘制训练和测试准确率曲线
```

## 使用方法

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 运行训练：

```bash
python train_digit_transformer.py
```

## 主要参数说明

- `d_model`：模型的维度，默认 64
- `num_heads`：注意力头的数量，默认 4
- `batch_size`：批次大小，默认 64
- `learning_rate`：学习率，默认 0.001
- `num_epochs`：训练轮数，默认 10

## 输出文件

- `digit_transformer.pth`：训练好的模型
- `training_history.png`：训练过程的可视化结果

## 注意事项

1. 数据预处理：

   - 图像被标准化到[0,1]范围
   - 标签从[1,2]转换为[0,1]

2. 模型特点：

   - 使用残差连接
   - 包含层归一化
   - 使用 ReLU 激活函数

3. 训练过程：
   - 使用 Adam 优化器
   - 使用交叉熵损失函数
   - 支持 GPU 训练（如果可用）

```mermaid
输入序列
    ↓
位置编码层 (PositionalEncoding)
    ↓
多头注意力层 (MultiHeadAttention)
    ↓
层归一化 (LayerNorm) + 残差连接
    ↓
前馈神经网络 (Feed Forward)
    ↓
层归一化 (LayerNorm) + 残差连接
    ↓
输出序列
```
