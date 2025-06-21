# 音频转换工具

一个功能强大的 Python 音频转换工具，专门用于将 WAV 格式音频文件转换为 MP3 格式。

## 功能特性

- 🎵 **WAV 到 MP3 转换**: 支持高质量音频格式转换
- 📁 **批量处理**: 支持整个目录的批量转换
- ⚙️ **可调参数**: 支持自定义比特率和质量设置
- 📊 **文件信息**: 提供详细的音频文件信息查看
- 🚀 **高效处理**: 基于 pydub 库，处理速度快
- 📝 **详细日志**: 完整的转换过程日志记录

## 安装要求

### 系统依赖

在 macOS 上，您需要安装 FFmpeg：

```bash
# 使用Homebrew安装FFmpeg
brew install ffmpeg
```

### Python 依赖

```bash
# 安装Python依赖
pip install -r requirements.txt
```

## 使用方法

### 命令行使用

#### 单个文件转换

```bash
# 基本转换
python audio_converter.py input.wav

# 指定输出文件
python audio_converter.py input.wav -o output.mp3

# 自定义比特率和质量
python audio_converter.py input.wav -b 320k -q 7
```

#### 批量转换

```bash
# 转换目录中的所有WAV文件
python audio_converter.py ./wav_files/

# 指定输出目录
python audio_converter.py ./wav_files/ -o ./mp3_files/
```

#### 查看文件信息

```bash
# 显示音频文件详细信息
python audio_converter.py input.wav -i
```

### 编程接口使用

```python
from audio_converter import AudioConverter

# 创建转换器实例
converter = AudioConverter()

# 单个文件转换
success, message = converter.convert_wav_to_mp3(
    'input.wav',
    'output.mp3',
    bitrate='192k',
    quality=5
)

# 批量转换
results = converter.batch_convert(
    './wav_files/',
    './mp3_files/',
    bitrate='320k',
    quality=7
)

# 获取文件信息
info = converter.get_file_info('audio.wav')
if info:
    print(f"时长: {info['duration']}秒")
    print(f"大小: {info['file_size']}字节")
```

## 参数说明

### 命令行参数

- `input`: 输入的 WAV 文件或目录路径
- `-o, --output`: 输出文件或目录路径
- `-b, --bitrate`: MP3 比特率 (默认: 192k)
- `-q, --quality`: 转换质量 0-9 (默认: 5)
- `-i, --info`: 显示文件信息而不转换

### 比特率选项

- `64k`: 低质量，文件小
- `128k`: 标准质量
- `192k`: 高质量 (默认)
- `256k`: 更高质量
- `320k`: 最高质量

### 质量参数

- `0`: 最高质量，文件最大
- `5`: 平衡质量和文件大小 (默认)
- `9`: 最低质量，文件最小

## 文件结构

```
audio/
├── audio_converter.py    # 主要转换工具
├── example.py           # 使用示例
├── requirements.txt     # Python依赖
├── README.md           # 说明文档
└── test_audio.wav      # 测试音频文件 (可选)
```

## 示例输出

### 转换过程

```
2024-01-01 12:00:00 - INFO - 开始转换: input.wav -> output.mp3
2024-01-01 12:00:00 - INFO - 比特率: 192k, 质量: 5
2024-01-01 12:00:01 - INFO - 转换完成!
2024-01-01 12:00:01 - INFO - 原始文件大小: 10.5 MB
2024-01-01 12:00:01 - INFO - 输出文件大小: 2.1 MB
2024-01-01 12:00:01 - INFO - 压缩率: 80.0%
转换成功! 输出文件: output.mp3
```

### 文件信息

```
文件信息:
文件名: audio.wav
格式: wav
大小: 10.5 MB
时长: 120.50 秒
声道: 2
采样率: 44100 Hz
采样宽度: 2 bytes
```

## 错误处理

工具包含完善的错误处理机制：

- 文件不存在检查
- 格式验证
- 转换失败重试
- 详细的错误信息输出

## 性能优化

- 使用 pydub 库进行高效音频处理
- 支持多文件并行处理
- 内存优化的文件处理
- 进度显示和状态更新

## 常见问题

### Q: 转换失败怎么办？

A: 请检查：

1. 输入文件是否为有效的 WAV 格式
2. 是否已安装 FFmpeg
3. 输出目录是否有写入权限

### Q: 如何提高转换质量？

A: 可以：

1. 提高比特率 (如使用 320k)
2. 降低质量参数 (如使用 0-3)
3. 确保输入文件质量良好

### Q: 支持其他音频格式吗？

A: 当前版本专注于 WAV 到 MP3 转换，未来版本将支持更多格式。

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个工具！
