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


def remove_watermark(input_path, output_path, top_left, bottom_right):
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("无法读取图片")

    h_img, w_img = img.shape[:2]

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

    # 融合：中值+均值+随机采样
    img_filled = img.copy()
    blend = 0.5 * sampled + 0.25 * median_color + 0.25 * mean_color
    blend = blend.astype(np.uint8)
    img_filled[top_left[1]:bottom_right[1],
               top_left[0]:bottom_right[0]] = blend

    # 可视化
    debug_img = img.copy()
    cv2.rectangle(debug_img, top_left, bottom_right, (0, 0, 255), 2)
    debug_path = os.path.join(os.path.dirname(output_path), 'debug_match.png')
    cv2.imwrite(debug_path, debug_img)
    print(f"调试图片已保存: {debug_path}")

    # 保存结果
    cv2.imwrite(output_path, img_filled)
    print(f"处理完成！输出文件保存在: {output_path}")
    print(f"水印位置：左上角坐标 {top_left}, 右下角坐标 {bottom_right}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, 'dadong.png')
    output_path = os.path.join(current_dir, 'no_watermark_dadong.png')

    # 你可以手动微调这两个坐标
    top_left = (955, 1070)      # 左上角 (x, y)
    bottom_right = (1040, 1090)  # 右下角 (x, y)

    remove_watermark(input_path, output_path, top_left, bottom_right)
