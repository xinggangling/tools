import torch  # 导入PyTorch库，用于生成随机数据
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于数据可视化
from sklearn.cluster import KMeans  # 导入KMeans聚类算法

# 生成数据
x = torch.randn(100, 2)  # 生成100个随机二维数据点，用于聚类

# 使用K-means进行聚类
kmeans = KMeans(n_clusters=3)  # 初始化KMeans模型，设置聚类数为3
kmeans.fit(x.numpy())  # 使用生成的数据进行聚类

# 获取聚类结果
labels = kmeans.labels_  # 获取每个数据点的聚类标签
centers = kmeans.cluster_centers_  # 获取每个聚类的中心点

plt.rcParams['font.sans-serif'] = ['Songti SC']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 可视化聚类结果
plt.scatter(x[:, 0].numpy(), x[:, 1].numpy(), c=labels,
            cmap='viridis')  # 绘制数据点，颜色根据聚类标签区分
plt.scatter(centers[:, 0], centers[:, 1], c='red',
            marker='x', s=100, label='Centroids')  # 绘制聚类中心点
plt.xlabel('X')  # 设置X轴标签
plt.ylabel('Y')  # 设置Y轴标签
plt.title('K-means Clustering')  # 设置图表标题
plt.legend()  # 显示图例
plt.show()  # 显示图表
