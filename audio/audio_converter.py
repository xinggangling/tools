#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频转换工具
支持将WAV格式音频文件转换为MP3格式
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple
from pydub import AudioSegment
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioConverter:
    """音频转换器类"""

    def __init__(self):
        """初始化音频转换器"""
        self.supported_formats = {
            'wav': 'WAV',
            'mp3': 'MP3',
            'flac': 'FLAC',
            'ogg': 'OGG',
            'aac': 'AAC'
        }

    def get_file_info(self, file_path: str) -> Optional[dict]:
        """
        获取音频文件信息

        Args:
            file_path: 音频文件路径

        Returns:
            包含文件信息的字典，如果文件不存在或格式不支持则返回None
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"文件不存在: {file_path}")
                return None

            audio = AudioSegment.from_file(file_path)

            info = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'duration': len(audio) / 1000.0,  # 转换为秒
                'channels': audio.channels,
                'sample_rate': audio.frame_rate,
                'sample_width': audio.sample_width,
                'format': self._get_format_from_path(file_path)
            }

            return info

        except Exception as e:
            logger.error(f"获取文件信息失败: {e}")
            return None

    def _get_format_from_path(self, file_path: str) -> str:
        """从文件路径获取格式"""
        return Path(file_path).suffix.lower()[1:]

    def convert_wav_to_mp3(self,
                           input_path: str,
                           output_path: Optional[str] = None,
                           bitrate: str = "192k",
                           quality: int = 5) -> Tuple[bool, str]:
        """
        将WAV文件转换为MP3格式

        Args:
            input_path: 输入的WAV文件路径
            output_path: 输出的MP3文件路径，如果为None则自动生成
            bitrate: MP3比特率，默认192k
            quality: 转换质量 (0-9)，默认5

        Returns:
            (成功标志, 消息)
        """
        try:
            # 检查输入文件
            if not os.path.exists(input_path):
                return False, f"输入文件不存在: {input_path}"

            input_format = self._get_format_from_path(input_path)
            if input_format != 'wav':
                return False, f"输入文件格式不是WAV: {input_format}"

            # 生成输出路径
            if output_path is None:
                output_path = self._generate_output_path(input_path, 'mp3')

            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            logger.info(f"开始转换: {input_path} -> {output_path}")
            logger.info(f"比特率: {bitrate}, 质量: {quality}")

            # 加载音频文件
            audio = AudioSegment.from_wav(input_path)

            # 转换并导出
            audio.export(
                output_path,
                format="mp3",
                bitrate=bitrate,
                parameters=["-q:a", str(quality)]
            )

            # 验证输出文件
            if os.path.exists(output_path):
                input_size = os.path.getsize(input_path)
                output_size = os.path.getsize(output_path)
                compression_ratio = (1 - output_size / input_size) * 100

                logger.info(f"转换完成!")
                logger.info(f"原始文件大小: {self._format_size(input_size)}")
                logger.info(f"输出文件大小: {self._format_size(output_size)}")
                logger.info(f"压缩率: {compression_ratio:.1f}%")

                return True, f"转换成功! 输出文件: {output_path}"
            else:
                return False, "转换失败: 输出文件未生成"

        except Exception as e:
            error_msg = f"转换过程中发生错误: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def _generate_output_path(self, input_path: str, output_format: str) -> str:
        """生成输出文件路径"""
        input_path_obj = Path(input_path)
        output_name = input_path_obj.stem + f".{output_format}"
        return str(input_path_obj.parent / output_name)

    def _format_size(self, size_bytes: int) -> str:
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def batch_convert(self,
                      input_dir: str,
                      output_dir: Optional[str] = None,
                      bitrate: str = "192k",
                      quality: int = 5) -> dict:
        """
        批量转换目录中的WAV文件

        Args:
            input_dir: 输入目录
            output_dir: 输出目录，如果为None则使用输入目录
            bitrate: MP3比特率
            quality: 转换质量

        Returns:
            转换结果统计
        """
        if not os.path.exists(input_dir):
            return {"success": False, "message": f"输入目录不存在: {input_dir}"}

        if output_dir is None:
            output_dir = input_dir

        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 查找所有WAV文件
        wav_files = []
        for file in os.listdir(input_dir):
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(input_dir, file))

        if not wav_files:
            return {"success": False, "message": "未找到WAV文件"}

        logger.info(f"找到 {len(wav_files)} 个WAV文件")

        # 批量转换
        results = {
            "total": len(wav_files),
            "success": 0,
            "failed": 0,
            "errors": []
        }

        for i, wav_file in enumerate(wav_files, 1):
            logger.info(
                f"处理文件 {i}/{len(wav_files)}: {os.path.basename(wav_file)}")

            # 生成输出路径
            output_path = os.path.join(
                output_dir,
                os.path.splitext(os.path.basename(wav_file))[0] + ".mp3"
            )

            success, message = self.convert_wav_to_mp3(
                wav_file, output_path, bitrate, quality
            )

            if success:
                results["success"] += 1
            else:
                results["failed"] += 1
                results["errors"].append({
                    "file": wav_file,
                    "error": message
                })

        logger.info(f"批量转换完成: 成功 {results['success']}, 失败 {results['failed']}")
        return results


def main():
    """主函数 - 命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(description="WAV到MP3音频转换工具")
    parser.add_argument("input", help="输入的WAV文件或目录路径")
    parser.add_argument("-o", "--output", help="输出文件或目录路径")
    parser.add_argument("-b", "--bitrate", default="192k",
                        help="MP3比特率 (默认: 192k)")
    parser.add_argument("-q", "--quality", type=int,
                        default=5, help="转换质量 0-9 (默认: 5)")
    parser.add_argument("-i", "--info", action="store_true", help="显示文件信息而不转换")

    args = parser.parse_args()

    converter = AudioConverter()

    if args.info:
        # 显示文件信息
        if os.path.isfile(args.input):
            info = converter.get_file_info(args.input)
            if info:
                print(f"\n文件信息:")
                print(f"文件名: {info['file_name']}")
                print(f"格式: {info['format']}")
                print(f"大小: {converter._format_size(info['file_size'])}")
                print(f"时长: {info['duration']:.2f} 秒")
                print(f"声道: {info['channels']}")
                print(f"采样率: {info['sample_rate']} Hz")
                print(f"采样宽度: {info['sample_width']} bytes")
            else:
                print("无法获取文件信息")
        else:
            print("请指定一个有效的文件路径")
    else:
        # 执行转换
        if os.path.isfile(args.input):
            # 单个文件转换
            success, message = converter.convert_wav_to_mp3(
                args.input, args.output, args.bitrate, args.quality
            )
            print(message)
        elif os.path.isdir(args.input):
            # 批量转换
            results = converter.batch_convert(
                args.input, args.output, args.bitrate, args.quality
            )
            print(f"\n批量转换结果:")
            print(f"总计: {results['total']}")
            print(f"成功: {results['success']}")
            print(f"失败: {results['failed']}")

            if results['errors']:
                print(f"\n错误详情:")
                for error in results['errors']:
                    print(
                        f"- {os.path.basename(error['file'])}: {error['error']}")
        else:
            print(f"输入路径不存在: {args.input}")


if __name__ == "__main__":
    main()
