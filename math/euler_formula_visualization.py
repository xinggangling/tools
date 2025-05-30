import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.animation import FuncAnimation

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # MacOS 系统
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def create_euler_animation():
    """
    创建欧拉公式的动画，包含辅助线、公式数值联动和简明解释
    """
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 第一个子图：复平面上的单位圆
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # 绘制单位圆
    circle = Circle((0, 0), 1, fill=False, color='gray',
                    linestyle='--', alpha=0.5)
    ax1.add_patch(circle)

    # 设置坐标轴
    ax1.axhline(y=0, color='#FF4500', linestyle='-', linewidth=1.5, alpha=0.8)
    ax1.axvline(x=0, color='#4169E1', linestyle='-', linewidth=1.5, alpha=0.8)

    # 设置标题和标签
    ax1.set_title('复平面上的欧拉公式')
    ax1.set_xlabel('实部 (cos(θ))')
    ax1.set_ylabel('虚部 (sin(θ))')
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)

    # 创建箭头和点的对象
    arrow = FancyArrowPatch((0, 0), (1, 0), color='#4169E1',
                            arrowstyle='->', mutation_scale=15, linewidth=2)
    ax1.add_patch(arrow)
    point, = ax1.plot([], [], 'ro', markersize=8)
    angle_text = ax1.text(0.2, 0.2, '', bbox=dict(
        facecolor='white', alpha=0.8))

    # 辅助线和投影点
    hline, = ax1.plot([], [], 'b--', lw=1)  # 实部投影线
    vline, = ax1.plot([], [], 'r--', lw=1)  # 虚部投影线
    proj_x, = ax1.plot([], [], 'bo', markersize=7, alpha=0.7)  # 实部投影点
    proj_y, = ax1.plot([], [], 'ro', markersize=7, alpha=0.7)  # 虚部投影点

    # 公式数值显示
    formula_text = ax1.text(-1.45, 1.2, '', fontsize=13,
                            color='black', bbox=dict(facecolor='white', alpha=0.8))

    # 简明解释文本
    explain_text = ax1.text(-1.45, 1.35, '', fontsize=15,
                            color='#008B8B', bbox=dict(facecolor='white', alpha=0.8))

    # 第二个子图：三角函数曲线
    t = np.linspace(0, 2*np.pi, 1000)
    ax2.plot(t, np.cos(t), 'b-', label='cos(θ)', linewidth=2)
    ax2.plot(t, np.sin(t), 'r-', label='sin(θ)', linewidth=2)

    # 创建垂直线和点的对象
    vline2 = ax2.axvline(x=0, color='g', linestyle='--', alpha=0.5)
    cos_point, = ax2.plot([], [], 'bo', markersize=8)
    sin_point, = ax2.plot([], [], 'ro', markersize=8)

    # 设置坐标轴
    ax2.axhline(y=0, color='#FF4500', linestyle='-', linewidth=1.5, alpha=0.8)
    ax2.axvline(x=0, color='#4169E1', linestyle='-', linewidth=1.5, alpha=0.8)

    # 设置标题和标签
    ax2.set_title('三角函数曲线')
    ax2.set_xlabel('θ (弧度)')
    ax2.set_ylabel('值')
    ax2.set_xlim(0, 2*np.pi)
    ax2.set_ylim(-1.5, 1.5)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 添加欧拉公式的数学表达式
    plt.figtext(0.5, 0.01,
                '欧拉公式: e^(iθ) = cos(θ) + i·sin(θ)',
                ha='center', fontsize=13,
                bbox=dict(facecolor='white', alpha=0.8))

    def update(frame):
        theta = frame * 2 * np.pi / 100  # 将帧数转换为角度
        x = np.cos(theta)
        y = np.sin(theta)
        arrow.set_positions((0, 0), (x, y))
        point.set_data([x], [y])
        angle_text.set_text(f'θ = {theta:.2f}弧度\n≈ {np.degrees(theta):.1f}°')

        # 辅助线和投影点
        hline.set_data([0, x], [y, y])  # 水平线
        vline.set_data([x, x], [0, y])  # 垂直线
        proj_x.set_data([x], [0])       # 实部投影点
        proj_y.set_data([0], [y])       # 虚部投影点

        # 公式数值联动
        formula_text.set_text(
            f"e^{{iθ}} = cos(θ) + i·sin(θ)\n= {x:.2f} + i·{y:.2f}")

        # 第二个子图
        vline2.set_xdata([theta, theta])
        cos_point.set_data([theta], [np.cos(theta)])
        sin_point.set_data([theta], [np.sin(theta)])

        # 简明解释
        explain = (
            f"e^{{iθ}} 在复平面上是单位圆上的一个点。\n"
            f"蓝色虚线是实部（cos(θ)），红色虚线是虚部（sin(θ)）。\n"
            f"当前 θ = {theta:.2f}，e^{{iθ}} = {x:.2f} + i·{y:.2f}。"
        )
        explain_text.set_text(explain)

        return (arrow, point, angle_text, hline, vline, proj_x, proj_y, formula_text, vline2, cos_point, sin_point, explain_text)

    # 创建动画
    anim = FuncAnimation(fig, update, frames=100, interval=60, blit=True)
    plt.tight_layout()
    plt.show()


# 运行程序
if __name__ == "__main__":
    create_euler_animation()
