import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Songti SC']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载图像
image = io.imread('dolphin.png')  # 加载图像

# 将图像转换为二维数组
# 每个像素点都是一个RGB值，可以看作是三维空间中的一个点
# 聚类就是要将这些点按照颜色相似度分组
rows, cols, dims = image.shape
image_2d = image.reshape(rows * cols, dims)

# 使用K-means进行聚类
# n_clusters=3 表示将图像分成3个颜色区域
# 每个区域的颜色由该区域所有像素点的平均颜色决定
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(image_2d)

# 获取聚类结果
labels = kmeans.labels_  # 每个像素点属于哪个聚类
centers = kmeans.cluster_centers_  # 每个聚类的中心点（代表该聚类的颜色）

# 将聚类结果转换回图像
# 用聚类中心点的颜色替换该聚类中所有像素点的颜色
segmented_image = centers[labels].reshape(rows, cols, dims).astype(np.uint8)

# 可视化原始图像和分割后的图像
plt.figure(figsize=(15, 5))

# 显示原始图像
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('原始图像')
plt.axis('off')

# 显示分割后的图像
plt.subplot(1, 3, 2)
plt.imshow(segmented_image)
plt.title('聚类分割后的图像\n(3个颜色区域)')
plt.axis('off')

# 显示聚类中心点的颜色
plt.subplot(1, 3, 3)
colors = centers.astype(np.uint8)
plt.imshow([colors])
plt.title('聚类中心点的颜色\n(每个区域的平均颜色)')
plt.axis('off')

plt.tight_layout()
plt.show()

# 打印每个聚类的大小（像素点数量）
unique_labels, counts = np.unique(labels, return_counts=True)
print("\n每个聚类的大小（像素点数量）：")
for label, count in zip(unique_labels, counts):
    print(f"聚类 {label}: {count} 个像素点")
