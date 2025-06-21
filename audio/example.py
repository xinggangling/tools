#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频转换工具使用示例
"""

from audio_converter import AudioConverter
import os


def main():
    """使用示例"""
    converter = AudioConverter()

    print("=== 音频转换工具使用示例 ===\n")

    # 示例1: 单个文件转换
    print("1. 单个文件转换示例:")
    print("   converter.convert_wav_to_mp3('input.wav', 'output.mp3')")
    print("   converter.convert_wav_to_mp3('input.wav', bitrate='320k', quality=7)")
    print()

    # 示例2: 批量转换
    print("2. 批量转换示例:")
    print("   converter.batch_convert('./wav_files/', './mp3_files/')")
    print()

    # 示例3: 获取文件信息
    print("3. 获取文件信息示例:")
    print("   info = converter.get_file_info('audio.wav')")
    print("   if info:")
    print("       print(f'时长: {info[\"duration\"]}秒')")
    print("       print(f'大小: {info[\"file_size\"]}字节')")
    print()

    # 示例4: 创建测试文件（如果不存在）
    test_wav = "test_audio.wav"
    if not os.path.exists(test_wav):
        print("4. 创建测试音频文件...")
        try:
            from pydub import AudioSegment
            from pydub.generators import Sine

            # 创建一个简单的测试音频文件
            generator = Sine(440)  # 440Hz正弦波
            audio = generator.to_audio_segment(duration=3000)  # 3秒
            audio.export(test_wav, format="wav")
            print(f"   已创建测试文件: {test_wav}")
        except Exception as e:
            print(f"   创建测试文件失败: {e}")
    else:
        print(f"4. 测试文件已存在: {test_wav}")

    print("\n=== 命令行使用示例 ===")
    print("python audio_converter.py input.wav -o output.mp3")
    print("python audio_converter.py input.wav -b 320k -q 7")
    print("python audio_converter.py ./wav_folder/ -o ./mp3_folder/")
    print("python audio_converter.py input.wav -i  # 显示文件信息")
    print()

    print("=== 支持的参数 ===")
    print("-o, --output: 输出文件路径")
    print("-b, --bitrate: MP3比特率 (默认: 192k)")
    print("-q, --quality: 转换质量 0-9 (默认: 5)")
    print("-i, --info: 显示文件信息")


if __name__ == "__main__":
    main()
