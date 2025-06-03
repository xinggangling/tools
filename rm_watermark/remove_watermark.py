from PIL import Image
import numpy as np
import os
import cv2


def preprocess_image(img):
    """预处理图片，统一格式和颜色空间"""
    # 确保图片是BGR格式
    if len(img.shape) == 2:  # 如果是灰度图
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # 如果是RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    # 转换为灰度图进行匹配
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def find_watermark_location(img, watermark, method=cv2.TM_CCOEFF_NORMED):
    """使用不同的匹配方法查找水印位置"""
    # 预处理图片
    img_gray = preprocess_image(img)
    watermark_gray = preprocess_image(watermark)

    # 获取水印模板的尺寸
    h, w = watermark_gray.shape[:2]

    # 使用模板匹配
    result = cv2.matchTemplate(img_gray, watermark_gray, method)

    # 根据不同的匹配方法获取最佳匹配位置
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = min_loc
    else:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)
    return top_left, bottom_right, max_val


def remove_watermark(input_path, output_path, top_left, bottom_right, fill_mode='mean'):
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("无法读取图片")

    h_img, w_img = img.shape[:2]
    print(f"输入图片尺寸：宽 {w_img} 像素，高 {h_img} 像素")

    # 取水印区域四周的像素
    pad = 5
    y1, y2 = max(top_left[1] - pad, 0), min(bottom_right[1] + pad, h_img)
    x1, x2 = max(top_left[0] - pad, 0), min(bottom_right[0] + pad, w_img)
    region = img[y1:y2, x1:x2]
    h_wm = bottom_right[1] - top_left[1]
    w_wm = bottom_right[0] - top_left[0]
    mask = np.ones((y2 - y1, x2 - x1), dtype=np.uint8)
    mask[pad:pad+h_wm, pad:pad+w_wm] = 0
    bg_pixels = region[mask.astype(bool)].reshape(-1, 3)

    # 计算中值和均值
    median_color = np.median(bg_pixels, axis=0).astype(np.uint8)
    mean_color = np.mean(bg_pixels, axis=0).astype(np.uint8)

    # 随机采样，数量不够就重复填充
    need_pixels = h_wm * w_wm
    if len(bg_pixels) < need_pixels:
        bg_pixels = np.resize(bg_pixels, (need_pixels, 3))
    else:
        np.random.shuffle(bg_pixels)
    sampled = bg_pixels[:need_pixels].reshape((h_wm, w_wm, 3))

    img_filled = img.copy()

    if fill_mode == 'mean':
        img_filled[top_left[1]:bottom_right[1],
                   top_left[0]:bottom_right[0]] = mean_color
    elif fill_mode == 'median':
        img_filled[top_left[1]:bottom_right[1],
                   top_left[0]:bottom_right[0]] = median_color
    elif fill_mode == 'sampled':
        blend = 0.5 * sampled + 0.25 * median_color + 0.25 * mean_color
        blend = blend.astype(np.uint8)
        img_filled[top_left[1]:bottom_right[1],
                   top_left[0]:bottom_right[0]] = blend
    elif fill_mode == 'linear':
        top_line = img[top_left[1], top_left[0]:bottom_right[0]].astype(np.float32)
        bottom_line = img[bottom_right[1]-1, top_left[0]:bottom_right[0]].astype(np.float32)
        for i in range(h_wm):
            alpha = i / max(h_wm-1, 1)
            line = (1-alpha) * top_line + alpha * bottom_line
            img_filled[top_left[1]+i, top_left[0]:bottom_right[0]] = line.astype(np.uint8)
    elif fill_mode == 'bilinear':
        top_line = img[top_left[1], top_left[0]:bottom_right[0]].astype(np.float32)
        bottom_line = img[bottom_right[1]-1, top_left[0]:bottom_right[0]].astype(np.float32)
        left_line = img[top_left[1]:bottom_right[1],
                        top_left[0]].astype(np.float32)
        right_line = img[top_left[1]:bottom_right[1],
                         bottom_right[0]-1].astype(np.float32)
        for i in range(h_wm):
            alpha = i / max(h_wm-1, 1)
            row = (1-alpha) * top_line + alpha * bottom_line
            for j in range(w_wm):
                beta = j / max(w_wm-1, 1)
                pixel = (1-beta) * left_line[i] + beta * right_line[i]
                blend_pixel = 0.5 * row[j] + 0.5 * pixel
                img_filled[top_left[1]+i, top_left[0] +
                           j] = blend_pixel.astype(np.uint8)
    elif fill_mode == 'gaussian':
        # 高斯模糊融合
        blur_img = cv2.GaussianBlur(img, (31, 31), 0)
        img_filled[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = \
            blur_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    elif fill_mode == 'plane':
        # 区域平面拟合（最小二乘平面）
        # 采集四周像素点坐标和值
        coords = []
        values = []
        for i in range(y1, y2):
            for j in range(x1, x2):
                if mask[i-y1, j-x1]:
                    coords.append([j, i, 1])
                    values.append(img[i, j])
        coords = np.array(coords)
        values = np.array(values)
        # 对每个通道做平面拟合
        plane = []
        for c in range(3):
            A, _, _, _ = np.linalg.lstsq(coords, values[:, c], rcond=None)
            plane.append(A)
        # 生成平面填充区域
        for i in range(h_wm):
            for j in range(w_wm):
                x = top_left[0] + j
                y = top_left[1] + i
                val = [int(plane[c][0]*x + plane[c][1]*y + plane[c][2])
                       for c in range(3)]
                img_filled[y, x] = np.clip(val, 0, 255)
    elif fill_mode == 'neighbor':
        img_filled = fill_by_neighbor_mean(img, top_left, bottom_right)
    elif fill_mode == 'mirror':
        for i in range(h_wm):
            for j in range(w_wm):
                y = top_left[1] + i
                x = top_left[0] + j
                mirror_x = w_img - 1 - x
                if 0 <= mirror_x < w_img:
                    img_filled[y, x] = img[y, mirror_x]
    else:
        raise ValueError(f"未知的填充模式: {fill_mode}")

    # 可视化，添加坐标轴
    debug_img = img.copy()
    cv2.rectangle(debug_img, top_left, bottom_right, (0, 0, 255), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for x in range(0, w_img, 100):
        cv2.line(debug_img, (x, 0), (x, h_img), (0, 255, 0), 1)
        cv2.putText(debug_img, str(x), (x+2, 20), font, 0.5, (0, 255, 0), 1)
    for y in range(0, h_img, 100):
        cv2.line(debug_img, (0, y), (w_img, y), (0, 255, 0), 1)
        cv2.putText(debug_img, str(y), (5, y+18), font, 0.5, (0, 255, 0), 1)
    debug_path = os.path.join(os.path.dirname(output_path), 'debug_match.png')
    cv2.imwrite(debug_path, debug_img)
    print(f"调试图片已保存: {debug_path}")

    # 保存结果
    cv2.imwrite(output_path, img_filled)
    print(f"处理完成！输出文件保存在: {output_path}")
    print(f"水印位置：左上角坐标 {top_left}, 右下角坐标 {bottom_right}")


def fill_by_neighbor_mean(img, top_left, bottom_right):
    img_filled = img.copy()
    h_wm = bottom_right[1] - top_left[1]
    w_wm = bottom_right[0] - top_left[0]
    for i in range(h_wm):
        for j in range(w_wm):
            y = top_left[1] + i
            x = top_left[0] + j
            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if (dy != 0 or dx != 0) and \
                       (0 <= ny < img.shape[0]) and (0 <= nx < img.shape[1]) and \
                       not (top_left[0] <= nx < bottom_right[0] and top_left[1] <= ny < bottom_right[1]):
                        neighbors.append(img[ny, nx])
            if neighbors:
                img_filled[y, x] = np.mean(neighbors, axis=0)
    return img_filled.astype(np.uint8)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, 'dolphin.png')
    output_path = os.path.join(current_dir, 'no_watermark_dolphin.png')

    # 你可以手动微调这两个坐标
    top_left = (1400, 1485)      # 左上角 (x, y)
    bottom_right = (1510, 1510)  # 右下角 (x, y)

    # fill_mode 可选: 'mean', 'median', 'sampled', 'linear', 'bilinear', 'gaussian', 'plane', 'neighbor', 'mirror'
    # 'mean'：均值填充（适合纯色背景，最自然）
    # 'median'：中值填充（适合有少量杂色的背景）
    # 'sampled'：随机采样融合（适合有纹理的背景）
    # 'linear'：上下边界线性插值（适合有渐变的背景）
    # 'bilinear'：双线性插值（适合复杂渐变背景）
    # 'gaussian'：高斯模糊融合（适合大面积模糊背景）
    # 'plane'：区域平面拟合（适合大面积渐变或光照变化背景）
    # 'neighbor'：邻域均值填充（适合纯色或边界明显的卡通图）
    # 'mirror'：对称镜像填充（适合对称背景）
    remove_watermark(input_path, output_path, top_left,
                     bottom_right, fill_mode='mirror')
