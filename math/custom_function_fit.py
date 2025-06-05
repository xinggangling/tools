import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # MacOS 系统
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def custom_growth(x, a, b, c):
    """
    自定义增长函数：f(x) = a * (1 - e^(-b*x)) + c
    其中：
    a: 控制最大增长幅度
    b: 控制增长速度
    c: 控制初始值
    """
    return a * (1 - np.exp(-b * x)) + c


# 生成示例数据
np.random.seed(42)  # 设置随机种子，确保结果可重现
x_data = np.linspace(0, 10, 50)
y_data = custom_growth(x_data, 5, 0.5, 1) + \
    np.random.normal(0, 0.2, 50)  # 添加一些噪声

# 使用 curve_fit 进行拟合
popt, pcov = curve_fit(custom_growth, x_data, y_data)

# 生成用于绘图的平滑曲线
x_smooth = np.linspace(0, 10, 200)
y_smooth = custom_growth(x_smooth, *popt)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color='blue', label='原始数据点', alpha=0.5)
plt.plot(x_smooth, y_smooth, 'r-', label='拟合曲线', linewidth=2)

# 添加拟合参数信息
a, b, c = popt
plt.title(f'自定义增长函数拟合\n参数: a={a:.2f}, b={b:.2f}, c={c:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, alpha=0.3)
plt.legend()

# 添加坐标轴
plt.axhline(y=0, color='#FF4500', linestyle='-', linewidth=1, alpha=0.8)
plt.axvline(x=0, color='#4169E1', linestyle='-', linewidth=1, alpha=0.8)

plt.tight_layout()
plt.show()
