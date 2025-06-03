import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Songti SC']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 生成数据
x = torch.linspace(0, 10, 100).view(-1, 1)
y = 2 * x + 1 + torch.randn(100, 1) * 0.5  # 生成带有噪声的标签数据，体现监督学习

# 定义模型


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# 训练代码
model = LinearModel()
criterion = nn.MSELoss()  # 使用均方误差作为损失函数
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01)  # 使用随机梯度下降优化器，学习率为0.01

# 训练循环
losses = []
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新模型参数
    losses.append(loss.item())

# 可视化训练过程
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# 可视化模型预测
plt.scatter(x.numpy(), y.numpy(), label='Data')
plt.plot(x.numpy(), model(x).detach().numpy(), label='Model', color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()
plt.show()
