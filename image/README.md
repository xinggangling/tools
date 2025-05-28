# PNG 图片无损压缩工具

这是一个用于无损压缩 PNG 图片的命令行工具。它使用 pngquant 库来实现高质量的图片压缩，同时保持图片的视觉质量。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 压缩单个文件

```bash
python png_compressor.py input.png -o output.png -q 80
```

### 压缩整个目录

```bash
python png_compressor.py /Users/lingxing.gang/workspace/ai-projects/gk_miniapp/assets/images -o /Users/lingxing.gang/workspace/ai-projects/gk_miniapp/assets/images -q 80
```

### 参数说明

- `input`: 输入文件或目录的路径（必需）
- `-o, --output`: 输出文件或目录的路径（可选，默认覆盖原文件）
- `-q, --quality`: 压缩质量，范围 0-100（可选，默认 80）

## 注意事项

1. 该工具仅支持 PNG 格式的图片
2. 如果不指定输出路径，将覆盖原文件
3. 压缩质量越高，文件大小越大，但图片质量越好
