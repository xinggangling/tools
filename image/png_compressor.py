import os
from PIL import Image
import argparse
from pathlib import Path

def compress_png(input_path, output_path=None, quality=80, resize=False, max_size=1024):
    """
    压缩PNG图片
    
    Args:
        input_path (str): 输入图片路径
        output_path (str, optional): 输出图片路径，如果不指定则覆盖原文件
        quality (int, optional): 压缩质量，范围0-100，默认80
        resize (bool, optional): 是否缩小图片尺寸，默认False
        max_size (int, optional): 图片最大边长，默认1024像素
    """
    try:
        # 如果没有指定输出路径，则覆盖原文件
        if output_path is None:
            output_path = input_path
            
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 打开图片
        img = Image.open(input_path)
        
        # 如果是PNG格式，进行压缩
        if img.format == 'PNG':
            # 获取原始文件大小
            original_size = os.path.getsize(input_path)
            
            # 如果图片是RGBA模式，转换为RGB模式
            if img.mode == 'RGBA':
                # 创建白色背景
                background = Image.new('RGB', img.size, (255, 255, 255))
                # 将RGBA图片粘贴到白色背景上
                background.paste(img, mask=img.split()[3])
                img = background
            
            # 如果需要缩小图片尺寸
            if resize:
                # 获取原始尺寸
                width, height = img.size
                # 计算缩放比例
                if width > height:
                    if width > max_size:
                        ratio = max_size / width
                        new_width = max_size
                        new_height = int(height * ratio)
                else:
                    if height > max_size:
                        ratio = max_size / height
                        new_height = max_size
                        new_width = int(width * ratio)
                
                # 如果图片需要缩小
                if 'new_width' in locals():
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    print(f"图片尺寸从 {width}x{height} 缩小到 {new_width}x{new_height}")
            
            # 计算颜色数量（基于quality参数）
            colors = max(2, int(256 * (quality / 100)))
            
            # 量化图片（减少颜色数量）
            img = img.quantize(colors=colors, method=2)
            
            # 转换回RGB模式
            img = img.convert('RGB')
            
            # 保存图片，使用optimize=True来启用PNG优化
            img.save(output_path, 'PNG', optimize=True)
            
            # 获取压缩后的文件大小
            compressed_size = os.path.getsize(output_path)
            
            # 计算压缩率
            compression_ratio = (1 - compressed_size / original_size) * 100
            
            print(f"压缩完成：{input_path} -> {output_path}")
            print(f"原始大小: {original_size/1024:.2f}KB")
            print(f"压缩后大小: {compressed_size/1024:.2f}KB")
            print(f"压缩率: {compression_ratio:.2f}%")
        else:
            print(f"警告：{input_path} 不是PNG格式的图片")
            
    except Exception as e:
        print(f"压缩失败：{str(e)}")

def main():
    parser = argparse.ArgumentParser(description='PNG图片压缩工具')
    parser.add_argument('input', help='输入图片路径或目录')
    parser.add_argument('-o', '--output', help='输出图片路径或目录')
    parser.add_argument('-q', '--quality', type=int, default=80, help='压缩质量(0-100)，默认80')
    parser.add_argument('-r', '--resize', action='store_true', help='是否缩小图片尺寸')
    parser.add_argument('-s', '--max-size', type=int, default=1024, help='图片最大边长（像素），默认1024')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 单个文件处理
        compress_png(str(input_path), args.output, args.quality, args.resize, args.max_size)
    elif input_path.is_dir():
        # 目录处理
        output_dir = Path(args.output) if args.output else input_path
        
        for png_file in input_path.glob('**/*.png'):
            relative_path = png_file.relative_to(input_path)
            output_path = output_dir / relative_path
            compress_png(str(png_file), str(output_path), args.quality, args.resize, args.max_size)
    else:
        print(f"错误：{args.input} 不是有效的文件或目录")

if __name__ == '__main__':
    main() 