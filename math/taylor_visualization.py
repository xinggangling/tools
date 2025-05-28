import numpy as np
import matplotlib.pyplot as plt
from math import factorial
import matplotlib as mpl

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # MacOS 系统
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def taylor_series(x, n):
    """
    计算 e^x 的n阶泰勒展开式
    """
    result = 0
    for i in range(n + 1):
        result += x**i / factorial(i)
    return result

def get_taylor_formula(n):
    """
    返回n阶泰勒展开式的公式字符串
    """
    terms = []
    for i in range(n + 1):
        if i == 0:
            terms.append("1")
        elif i == 1:
            terms.append("x")
        else:
            terms.append(f"x^{i}/{i}!")
    return " + ".join(terms)

# 创建数据点
x = np.linspace(-2, 2, 1000)
y_exact = np.exp(x)  # 精确的 e^x 值

# 绘制不同阶数的泰勒展开式
plt.figure(figsize=(12, 8))
plt.plot(x, y_exact, 'k-', label='e^x (精确值)', linewidth=2)

# 绘制不同阶数的泰勒展开式
orders = [1, 2, 3, 4, 5]
colors = ['r', 'g', 'b', 'c', 'm']

for order, color in zip(orders, colors):
    y_taylor = taylor_series(x, order)
    formula = get_taylor_formula(order)
    plt.plot(x, y_taylor, color + '--', 
             label=f'{order}阶泰勒展开: {formula}', 
             linewidth=1.5)

plt.title('e^x 的泰勒展开式可视化')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, alpha=0.3)  # 降低网格线的透明度
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 将图例放在图形右侧

# 设置坐标轴
plt.axis([-2, 2, -1, 8])
# 添加坐标轴，使用醒目的颜色
plt.axhline(y=0, color='#FF4500', linestyle='-', linewidth=1.5, alpha=0.8)  # x轴，使用橙红色
plt.axvline(x=0, color='#4169E1', linestyle='-', linewidth=1.5, alpha=0.8)  # y轴，使用皇家蓝

# 添加坐标轴标签
plt.text(2.1, 0, 'x轴', color='#FF4500', fontsize=10)
plt.text(0, 8.1, 'y轴', color='#4169E1', fontsize=10)

plt.tight_layout()  # 自动调整布局，确保图例不被切掉
plt.show() 