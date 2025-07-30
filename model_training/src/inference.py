#!/usr/bin/env python3
"""
翻译模型推理脚本
用于使用训练好的模型进行翻译
"""

from src.training.utils import setup_device
from src.data import Tokenizer
from src.models import TranslationModel
import sys
import torch
import argparse
import json
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class Translator:
    """翻译器类"""

    def __init__(self, model_path: str, src_tokenizer_path: str, tgt_tokenizer_path: str):
        """
        初始化翻译器
        Args:
            model_path: 模型检查点路径
            src_tokenizer_path: 源语言分词器路径
            tgt_tokenizer_path: 目标语言分词器路径
        """
        # 设置设备
        self.device = setup_device()

        # 加载分词器
        self.src_tokenizer = Tokenizer.load(src_tokenizer_path)
        self.tgt_tokenizer = Tokenizer.load(tgt_tokenizer_path)

        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()

        print(f"翻译器初始化完成，设备: {self.device}")

    def _load_model(self, model_path: str) -> TranslationModel:
        """加载模型"""
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device)

        # 获取模型配置
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})

        # 创建模型
        model = TranslationModel(
            src_vocab_size=self.src_tokenizer.vocab_size,
            tgt_vocab_size=self.tgt_tokenizer.vocab_size,
            d_model=model_config.get('d_model', 256),
            nlayers=model_config.get('nlayers', 2),
            nhead=model_config.get('nhead', 2),
            dim_feedforward=model_config.get('dim_feedforward', 1024),
            dropout=model_config.get('dropout', 0.1)
        )

        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)

        return model

    def translate(self, text: str, max_length: int = 50, temperature: float = 1.0) -> str:
        """
        翻译文本
        Args:
            text: 源文本
            max_length: 最大生成长度
            temperature: 采样温度
        Returns:
            翻译结果
        """
        # 编码源文本
        src_ids = self.src_tokenizer.encode(text, add_special_tokens=True)
        src_tensor = torch.tensor([src_ids], dtype=torch.long).to(self.device)

        # 生成翻译
        with torch.no_grad():
            output_ids = self.model.generate(
                src_tensor,
                max_len=max_length,
                start_token=self.tgt_tokenizer.sos_token_id,
                end_token=self.tgt_tokenizer.eos_token_id,
                temperature=temperature
            )

        # 解码结果
        output_ids = output_ids[0].cpu().numpy().tolist()
        translation = self.tgt_tokenizer.decode(
            output_ids, skip_special_tokens=True)

        return translation

    def translate_batch(self, texts: list, max_length: int = 50, temperature: float = 1.0) -> list:
        """
        批量翻译
        Args:
            texts: 源文本列表
            max_length: 最大生成长度
            temperature: 采样温度
        Returns:
            翻译结果列表
        """
        results = []

        for text in texts:
            translation = self.translate(text, max_length, temperature)
            results.append(translation)

        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="翻译模型推理")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型检查点路径"
    )
    parser.add_argument(
        "--src_tokenizer",
        type=str,
        required=True,
        help="源语言分词器路径"
    )
    parser.add_argument(
        "--tgt_tokenizer",
        type=str,
        required=True,
        help="目标语言分词器路径"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="",
        help="要翻译的文本"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="",
        help="输入文件路径（每行一个文本）"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="输出文件路径"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="最大生成长度"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="采样温度"
    )

    args = parser.parse_args()

    # 创建翻译器
    translator = Translator(
        model_path=args.model_path,
        src_tokenizer_path=args.src_tokenizer,
        tgt_tokenizer_path=args.tgt_tokenizer
    )

    # 处理输入
    if args.text:
        # 翻译单个文本
        translation = translator.translate(
            args.text,
            max_length=args.max_length,
            temperature=args.temperature
        )
        print(f"原文: {args.text}")
        print(f"译文: {translation}")

    elif args.input_file:
        # 批量翻译
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        print(f"开始批量翻译 {len(texts)} 个文本...")
        translations = translator.translate_batch(
            texts,
            max_length=args.max_length,
            temperature=args.temperature
        )

        # 输出结果
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for text, translation in zip(texts, translations):
                    f.write(f"{text}\t{translation}\n")
            print(f"翻译结果已保存到: {args.output_file}")
        else:
            for text, translation in zip(texts, translations):
                print(f"原文: {text}")
                print(f"译文: {translation}")
                print("-" * 50)

    else:
        # 交互式翻译
        print("进入交互式翻译模式（输入 'quit' 退出）")
        while True:
            text = input("请输入要翻译的文本: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break

            if text:
                translation = translator.translate(
                    text,
                    max_length=args.max_length,
                    temperature=args.temperature
                )
                print(f"译文: {translation}")
                print()


if __name__ == "__main__":
    main()
