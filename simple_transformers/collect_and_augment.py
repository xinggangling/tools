import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# 数据增强函数


def augment(img):
    imgs = [img]
    rows, cols = img.shape

    # 旋转
    for angle in [-15, 15]:
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        imgs.append(cv2.warpAffine(img, M, (cols, rows), borderValue=255))

    # 平移
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        imgs.append(cv2.warpAffine(img, M, (cols, rows), borderValue=255))

    # 缩放
    for scale in [0.9, 1.1]:
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 0, scale)
        imgs.append(cv2.warpAffine(img, M, (cols, rows), borderValue=255))

    # 加高斯噪声
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    imgs.append(cv2.add(img, noise))

    return imgs

# 标准化为MNIST风格


def mnist_style(img):
    # 灰度
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 自适应阈值二值化
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)
    # 裁剪数字区域
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    digit = img[y:y+h, x:x+w]
    # 缩放到20x20
    digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
    # 四周补白到28x28
    padded = np.pad(digit, ((4, 4), (4, 4)), 'constant', constant_values=0)
    return padded


def process_folder(input_folder, output_folder, label):
    os.makedirs(output_folder, exist_ok=True)
    img_files = [f for f in os.listdir(
        input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    idx = 0
    for file in tqdm(img_files, desc=f'Processing {label}'):
        img_path = os.path.join(input_folder, file)
        img = cv2.imread(img_path)
        for aug_img in augment(mnist_style(img)):
            save_path = os.path.join(output_folder, f'{label}_{idx}.png')
            cv2.imwrite(save_path, aug_img)
            idx += 1


if __name__ == '__main__':
    # 假设你有两个文件夹：my_1/ 和 my_2/，分别存放你拍的"1"和"2"
    # 输出到 processed_1/ 和 processed_2/
    process_folder('my_1', 'processed_1', label=1)
    process_folder('my_2', 'processed_2', label=2)
    print('采集、增强、标准化完成！')
