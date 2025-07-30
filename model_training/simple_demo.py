#!/usr/bin/env python3
"""
简化的翻译模型演示
使用更简单的架构来避免训练不稳定问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import os
from typing import List, Dict, Tuple

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


class SimpleTokenizer:
    """简单的字符级分词器"""

    def __init__(self, texts: List[str], max_vocab_size: int = 1000):
        # 构建词汇表
        char_counts = {}
        for text in texts:
            for char in text:
                char_counts[char] = char_counts.get(char, 0) + 1

        # 选择最常见的字符
        sorted_chars = sorted(char_counts.items(),
                              key=lambda x: x[1], reverse=True)
        vocab = ['<pad>', '<unk>', '<sos>', '<eos>'] + \
            [char for char, _ in sorted_chars[:max_vocab_size-4]]

        self.char_to_id = {char: i for i, char in enumerate(vocab)}
        self.id_to_char = {i: char for i, char in enumerate(vocab)}
        self.vocab_size = len(vocab)

    def encode(self, text: str) -> List[int]:
        return [self.char_to_id.get(char, 1) for char in text]  # 1 = <unk>

    def decode(self, ids: List[int]) -> str:
        return ''.join([self.id_to_char.get(id, '<unk>') for id in ids if id not in [0, 1, 2, 3]])


class SimpleTranslationModel(nn.Module):
    """简化的翻译模型"""

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 hidden_size: int = 128, num_layers: int = 2):
        super().__init__()

        self.hidden_size = hidden_size
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        # 嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, hidden_size)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, hidden_size)

        # 编码器（简单的LSTM）
        self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers,
                               batch_first=True, dropout=0.1)

        # 解码器（简单的LSTM）
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers,
                               batch_first=True, dropout=0.1)

        # 输出层
        self.output_layer = nn.Linear(hidden_size, tgt_vocab_size)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def encode(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码源序列"""
        embedded = self.src_embedding(src)
        output, (hidden, cell) = self.encoder(embedded)
        return output, (hidden, cell)

    def decode(self, tgt: torch.Tensor, encoder_hidden: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """解码目标序列"""
        embedded = self.tgt_embedding(tgt)
        output, _ = self.decoder(embedded, encoder_hidden)
        return self.output_layer(output)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 编码
        _, encoder_hidden = self.encode(src)

        # 解码（去掉最后一个token）
        tgt_input = tgt[:, :-1]
        output = self.decode(tgt_input, encoder_hidden)

        return output

    def generate(self, src: torch.Tensor, max_len: int = 50,
                 start_token: int = 2, end_token: int = 3) -> torch.Tensor:
        """生成翻译"""
        self.eval()
        batch_size = src.size(0)

        # 编码
        _, encoder_hidden = self.encode(src)

        # 初始化目标序列
        tgt = torch.full((batch_size, 1), start_token,
                         dtype=torch.long, device=src.device)

        with torch.no_grad():
            for _ in range(max_len - 1):
                # 解码
                output = self.decode(tgt, encoder_hidden)
                logits = output[:, -1:]

                # 采样下一个token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs.squeeze(1), 1)

                # 添加到序列
                tgt = torch.cat([tgt, next_token], dim=1)

                # 检查是否结束
                if (tgt == end_token).any(dim=1).all():
                    break

        return tgt


def create_sample_data() -> Tuple[List[str], List[str]]:
    """创建示例数据"""
    src_texts = [
        "Hello",
        "Good morning",
        "How are you?",
        "Thank you",
        "Goodbye",
        "I love you",
        "What is your name?",
        "Where are you from?",
        "Nice to meet you",
        "Have a nice day"
    ]

    tgt_texts = [
        "你好",
        "早上好",
        "你好吗？",
        "谢谢",
        "再见",
        "我爱你",
        "你叫什么名字？",
        "你来自哪里？",
        "很高兴认识你",
        "祝你愉快"
    ]

    return src_texts, tgt_texts


def prepare_data(src_texts: List[str], tgt_texts: List[str],
                 src_tokenizer: SimpleTokenizer, tgt_tokenizer: SimpleTokenizer,
                 max_length: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
    """准备训练数据"""
    src_ids = []
    tgt_ids = []

    for src_text, tgt_text in zip(src_texts, tgt_texts):
        # 编码
        # <sos> + text + <eos>
        src_seq = [2] + src_tokenizer.encode(src_text) + [3]
        tgt_seq = [2] + tgt_tokenizer.encode(tgt_text) + [3]

        # 填充到固定长度
        src_seq = src_seq[:max_length] + [0] * \
            max(0, max_length - len(src_seq))
        tgt_seq = tgt_seq[:max_length] + [0] * \
            max(0, max_length - len(tgt_seq))

        src_ids.append(src_seq)
        tgt_ids.append(tgt_seq)

    return torch.tensor(src_ids), torch.tensor(tgt_ids)


def train_model(model: SimpleTranslationModel, src_data: torch.Tensor,
                tgt_data: torch.Tensor, num_epochs: int = 100,
                learning_rate: float = 0.001) -> List[float]:
    """训练模型"""
    device = torch.device(
        'mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    src_data = src_data.to(device)
    tgt_data = tgt_data.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding token

    losses = []

    print(f"开始训练，设备: {device}")
    print(f"数据形状: src={src_data.shape}, tgt={tgt_data.shape}")

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # 前向传播
        output = model(src_data, tgt_data)

        # 计算损失
        loss = criterion(output.reshape(-1, output.size(-1)),
                         tgt_data[:, 1:].reshape(-1))

        # 反向传播
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    return losses


def test_translation(model: SimpleTranslationModel, src_tokenizer: SimpleTokenizer,
                     tgt_tokenizer: SimpleTokenizer, test_texts: List[str]):
    """测试翻译"""
    device = next(model.parameters()).device

    print("\n🧪 测试翻译结果:")
    print("=" * 50)

    for text in test_texts:
        # 编码
        src_seq = [2] + src_tokenizer.encode(text) + [3]
        src_tensor = torch.tensor([src_seq]).to(device)

        # 生成翻译
        tgt_ids = model.generate(src_tensor, max_len=20)

        # 解码
        tgt_seq = tgt_ids[0].cpu().numpy().tolist()
        translation = tgt_tokenizer.decode(tgt_seq)

        print(f"原文: {text}")
        print(f"译文: {translation}")
        print("-" * 30)


def main():
    """主函数"""
    print("🚀 简化翻译模型演示")
    print("=" * 50)

    # 1. 创建示例数据
    print("📊 步骤1: 准备数据...")
    src_texts, tgt_texts = create_sample_data()
    print(f"✅ 数据准备完成，共 {len(src_texts)} 个样本")

    # 2. 构建分词器
    print("\n🔤 步骤2: 构建分词器...")
    src_tokenizer = SimpleTokenizer(src_texts, max_vocab_size=100)
    tgt_tokenizer = SimpleTokenizer(tgt_texts, max_vocab_size=100)
    print(f"✅ 源语言词汇表大小: {src_tokenizer.vocab_size}")
    print(f"✅ 目标语言词汇表大小: {tgt_tokenizer.vocab_size}")

    # 3. 准备训练数据
    print("\n📦 步骤3: 准备训练数据...")
    src_data, tgt_data = prepare_data(
        src_texts, tgt_texts, src_tokenizer, tgt_tokenizer)
    print(f"✅ 数据形状: src={src_data.shape}, tgt={tgt_data.shape}")

    # 4. 创建模型
    print("\n🧠 步骤4: 创建模型...")
    model = SimpleTranslationModel(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        hidden_size=64,  # 更小的隐藏层
        num_layers=1     # 更少的层数
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型参数数量: {total_params:,}")

    # 5. 训练模型
    print("\n🎯 步骤5: 训练模型...")
    losses = train_model(model, src_data, tgt_data,
                         num_epochs=50, learning_rate=0.01)
    print(f"✅ 训练完成，最终损失: {losses[-1]:.4f}")

    # 6. 测试翻译
    test_texts = ["Hello", "Thank you", "Goodbye", "I love you"]
    test_translation(model, src_tokenizer, tgt_tokenizer, test_texts)

    print("\n🎉 演示完成！")
    print("这个简化版本展示了翻译模型的基本原理，")
    print("虽然效果有限，但避免了复杂的Transformer架构可能出现的训练不稳定问题。")


if __name__ == "__main__":
    main()
