from PIL import Image
import numpy as np
import os
import cv2

def remove_watermark(input_path, output_path, watermark_template_path):
    # 读取原始图片和水印模板
    img = cv2.imread(input_path)
    watermark = cv2.imread(watermark_template_path)
    
    # 确保图片和水印模板都成功读取
    if img is None or watermark is None:
        raise ValueError("无法读取图片或水印模板")
    
    # 获取水印模板的尺寸
    h, w = watermark.shape[:2]
    
    # 使用模板匹配找到水印位置
    result = cv2.matchTemplate(img, watermark, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # 如果匹配度不够高，可能需要调整阈值
    if max_val < 0.5:
        print(f"警告：水印匹配度较低 ({max_val:.2f})，可能无法准确定位水印")
    
    # 获取水印位置
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    # 创建掩码，稍微扩大一点区域以确保完全覆盖水印
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    # 扩大掩码区域，确保完全覆盖水印
    padding = 2
    mask[top_left[1]-padding:bottom_right[1]+padding, 
         top_left[0]-padding:bottom_right[0]+padding] = 255
    
    # 使用改进的Inpainting方法
    # 首先使用NS方法进行初步修复
    result_ns = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
    
    # 然后使用TELEA方法进行精细修复
    result = cv2.inpaint(result_ns, mask, 1, cv2.INPAINT_TELEA)
    
    # 对修复区域进行边缘平滑处理
    kernel = np.ones((3,3), np.float32)/9
    result = cv2.filter2D(result, -1, kernel)
    
    # 保存结果
    cv2.imwrite(output_path, result)
    print(f"处理完成！输出文件保存在: {output_path}")
    print(f"水印位置：左上角坐标 {top_left}, 右下角坐标 {bottom_right}")

if __name__ == "__main__":
    # 获取当前目录下的所有PNG文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 只处理原始图片，排除已处理过的文件和模板图片
    png_files = [f for f in os.listdir(current_dir) 
                if f.lower().endswith('.png') 
                and not f.startswith('no_watermark_') 
                and f != 'image.png']
    
    # 水印模板图片路径
    watermark_template_path = os.path.join(current_dir, 'image.png')
    
    for png_file in png_files:
        input_path = os.path.join(current_dir, png_file)
        output_path = os.path.join(current_dir, f"no_watermark_{png_file}")
        remove_watermark(input_path, output_path, watermark_template_path) 