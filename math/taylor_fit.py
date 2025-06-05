import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import factorial

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # MacOS 系统
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def taylor_series(x, *coeffs):
    """
    泰勒级数函数
    coeffs: 系数列表 [a0, a1, a2, ..., an]
    返回: a0 + a1*x + a2*x^2/2! + a3*x^3/3! + ... + an*x^n/n!
    """
    result = np.zeros_like(x, dtype=float)
    for i, coeff in enumerate(coeffs):
        result += coeff * x**i / factorial(i)
    return result

# 生成示例数据（这里使用一个复杂的函数作为示例）


def original_function(x):
    return np.sin(x) * np.exp(-0.1*x) + 2


# 生成数据点
x_data = np.linspace(0, 10, 100)
y_data = original_function(x_data) + np.random.normal(0, 0.1, 100)  # 添加一些噪声

# 尝试不同阶数的泰勒级数拟合
orders = [3, 5, 7, 9]  # 尝试不同的阶数
plt.figure(figsize=(12, 8))

# 绘制原始数据点
plt.scatter(x_data, y_data, color='gray', label='原始数据点', alpha=0.5)

# 绘制原始函数（如果知道的话）
x_smooth = np.linspace(0, 10, 200)
y_original = original_function(x_smooth)
plt.plot(x_smooth, y_original, 'k--', label='原始函数', linewidth=2)

# 对每个阶数进行拟合
colors = ['r', 'g', 'b', 'c']
for order, color in zip(orders, colors):
    # 初始猜测值
    p0 = [1.0] * (order + 1)

    # 进行拟合
    popt, pcov = curve_fit(taylor_series, x_data, y_data, p0=p0)

    # 计算拟合曲线
    y_fit = taylor_series(x_smooth, *popt)

    # 绘制拟合曲线
    plt.plot(x_smooth, y_fit, color=color,
             label=f'{order}阶泰勒展开',
             linewidth=2, alpha=0.7)

    # 计算拟合误差
    mse = np.mean((y_data - taylor_series(x_data, *popt))**2)
    print(f'{order}阶泰勒展开的均方误差: {mse:.6f}')

plt.title('使用泰勒级数进行曲线拟合')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 添加坐标轴
plt.axhline(y=0, color='#FF4500', linestyle='-', linewidth=1, alpha=0.8)
plt.axvline(x=0, color='#4169E1', linestyle='-', linewidth=1, alpha=0.8)

plt.tight_layout()
plt.show()
