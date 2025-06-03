# 基础理论学习

## 学习目标

通过代码和可视化方式，深入理解机器学习和深度学习的基础理论。

## 学习内容

1. **机器学习基础**

   - **监督学习、无监督学习的概念**：监督学习是指从标记数据中学习，而无监督学习则是从未标记数据中学习。
     - [监督学习示例代码](linear_regression.py)
     - [无监督学习示例代码](unsupervised_learning.py)
   - **损失函数（Loss Function）**：用于衡量模型预测值与真实值之间的差距，常见的损失函数包括均方误差（MSE）和交叉熵损失。
   - **优化器（Optimizer）**：用于更新模型参数以最小化损失函数，常见的优化器包括随机梯度下降（SGD）和 Adam。
   - **学习率（Learning Rate）**：控制每次参数更新的步长，过大的学习率可能导致训练不稳定，过小则可能导致训练过慢。
   - **过拟合和欠拟合**：过拟合是指模型在训练数据上表现良好但在测试数据上表现差，欠拟合则是指模型在训练数据上表现不佳。

2. **深度学习基础**
   - 神经网络的基本结构
   - 反向传播算法
   - 激活函数
   - 批量归一化
   - 正则化方法

## 学习方式

- 使用 Python 和 PyTorch 编写代码示例
- 使用 Matplotlib 或 Seaborn 进行数据可视化
- 通过 Jupyter Notebook 进行交互式学习

## 示例代码

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 示例：线性回归模型
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# 训练代码
model = LinearModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# 可视化训练过程
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
```

## 学习资源

- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [Matplotlib 官方文档](https://matplotlib.org/stable/contents.html)
- [Seaborn 官方文档](https://seaborn.pydata.org/)

## 下一步

完成基础理论学习后，进入实践学习路线，开始动手实现简单的模型。
