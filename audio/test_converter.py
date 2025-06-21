#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频转换工具测试脚本
"""

import os
import sys
from audio_converter import AudioConverter
from pydub import AudioSegment
from pydub.generators import Sine


def create_test_audio(filename="test_audio.wav", duration=3000):
    """创建测试音频文件"""
    print(f"创建测试音频文件: {filename}")

    # 创建一个包含多个频率的测试音频
    generator1 = Sine(440)  # A4音符
    generator2 = Sine(880)  # A5音符

    # 混合两个频率
    audio1 = generator1.to_audio_segment(duration=duration//2)
    audio2 = generator2.to_audio_segment(duration=duration//2)

    # 合并音频
    combined_audio = audio1 + audio2

    # 导出为WAV文件
    combined_audio.export(filename, format="wav")
    print(f"测试音频文件已创建: {filename}")
    return filename


def test_single_conversion():
    """测试单个文件转换"""
    print("\n=== 测试单个文件转换 ===")

    converter = AudioConverter()

    # 创建测试文件
    test_file = create_test_audio()

    try:
        # 测试文件信息获取
        print("\n1. 测试文件信息获取:")
        info = converter.get_file_info(test_file)
        if info:
            print(f"   文件名: {info['file_name']}")
            print(f"   格式: {info['format']}")
            print(f"   大小: {converter._format_size(info['file_size'])}")
            print(f"   时长: {info['duration']:.2f} 秒")
            print(f"   声道: {info['channels']}")
            print(f"   采样率: {info['sample_rate']} Hz")
        else:
            print("   获取文件信息失败")
            return False

        # 测试转换
        print("\n2. 测试WAV到MP3转换:")
        output_file = "test_output.mp3"
        success, message = converter.convert_wav_to_mp3(
            test_file,
            output_file,
            bitrate="192k",
            quality=5
        )

        print(f"   转换结果: {message}")

        if success and os.path.exists(output_file):
            print(
                f"   输出文件大小: {converter._format_size(os.path.getsize(output_file))}")
            return True
        else:
            print("   转换失败")
            return False

    except Exception as e:
        print(f"   测试过程中发生错误: {e}")
        return False


def test_batch_conversion():
    """测试批量转换"""
    print("\n=== 测试批量转换 ===")

    converter = AudioConverter()

    # 创建测试目录
    test_dir = "test_wav_files"
    output_dir = "test_mp3_files"

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # 创建多个测试文件
    test_files = []
    for i in range(3):
        filename = os.path.join(test_dir, f"test_audio_{i+1}.wav")
        create_test_audio(filename, duration=2000)
        test_files.append(filename)

    try:
        # 执行批量转换
        print(f"\n批量转换 {len(test_files)} 个文件...")
        results = converter.batch_convert(
            test_dir,
            output_dir,
            bitrate="128k",
            quality=7
        )

        print(f"批量转换结果:")
        print(f"   总计: {results['total']}")
        print(f"   成功: {results['success']}")
        print(f"   失败: {results['failed']}")

        if results['errors']:
            print("   错误详情:")
            for error in results['errors']:
                print(
                    f"     - {os.path.basename(error['file'])}: {error['error']}")

        return results['success'] > 0

    except Exception as e:
        print(f"批量转换测试失败: {e}")
        return False


def cleanup_test_files():
    """清理测试文件"""
    print("\n=== 清理测试文件 ===")

    files_to_remove = [
        "test_audio.wav",
        "test_output.mp3"
    ]

    dirs_to_remove = [
        "test_wav_files",
        "test_mp3_files"
    ]

    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"已删除: {file}")

    for dir_path in dirs_to_remove:
        if os.path.exists(dir_path):
            import shutil
            shutil.rmtree(dir_path)
            print(f"已删除目录: {dir_path}")


def main():
    """主测试函数"""
    print("音频转换工具测试")
    print("=" * 50)

    # 检查依赖
    try:
        from pydub import AudioSegment
        print("✓ pydub库已安装")
    except ImportError:
        print("✗ pydub库未安装，请运行: pip install -r requirements.txt")
        return

    # 检查FFmpeg
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ FFmpeg已安装")
        else:
            print("✗ FFmpeg未正确安装")
            return
    except FileNotFoundError:
        print("✗ FFmpeg未安装，请安装FFmpeg")
        print("  macOS: brew install ffmpeg")
        return

    # 运行测试
    test_results = []

    # 测试单个文件转换
    test_results.append(test_single_conversion())

    # 测试批量转换
    test_results.append(test_batch_conversion())

    # 显示测试结果
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print(f"单个文件转换: {'✓ 通过' if test_results[0] else '✗ 失败'}")
    print(f"批量转换: {'✓ 通过' if test_results[1] else '✗ 失败'}")

    if all(test_results):
        print("\n🎉 所有测试通过！音频转换工具工作正常。")
    else:
        print("\n❌ 部分测试失败，请检查错误信息。")

    # 询问是否清理测试文件
    try:
        response = input("\n是否清理测试文件？(y/n): ").lower().strip()
        if response in ['y', 'yes', '是']:
            cleanup_test_files()
        else:
            print("测试文件已保留，您可以手动检查转换结果。")
    except KeyboardInterrupt:
        print("\n测试文件已保留。")


if __name__ == "__main__":
    main()
