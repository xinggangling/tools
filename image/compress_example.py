import os
from png_compressor import compress_png

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建输入和输出文件的完整路径
input_file = os.path.join(current_dir, "logo.png")
output_file = os.path.join(current_dir, "logo_compressed.png")

# 检查输入文件是否存在
if not os.path.exists(input_file):
    print(f"错误：找不到输入文件 {input_file}")
else:
    try:
        # 压缩图片
        compress_png(
            input_path=input_file,
            output_path=output_file,
            quality=80,
            resize=True,
            max_size=128
        )
    except Exception as e:
        print(f"压缩过程中出现错误：{str(e)}")

# # 示例2：直接覆盖原文件
# compress_png(
#     input_path="your_image.png",  # 替换为您的图片路径
#     quality=70  # 使用较低的压缩质量
# )
