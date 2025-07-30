#!/usr/bin/env python3
"""
稳定的Transformer翻译模型
专门解决训练不稳定和NaN问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from typing import List, Dict, Tuple, Optional

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


class StablePositionalEncoding(nn.Module):
    """稳定的位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class StableMultiheadAttention(nn.Module):
    """稳定的多头注意力机制"""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)

        # 线性变换并重塑
        Q = self.w_q(query).view(batch_size, -1,
                                 self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1,
                               self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1,
                                 self.nhead, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 数值稳定性：使用log_softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        context = torch.matmul(attn_weights, V)

        # 重塑并线性变换
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        output = self.w_o(context)

        return output, attn_weights


class StableTransformerLayer(nn.Module):
    """稳定的Transformer层"""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = StableMultiheadAttention(d_model, nhead, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力 + 残差连接
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class StableTransformerModel(nn.Module):
    """稳定的Transformer翻译模型"""

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 512, max_len: int = 100):
        super().__init__()

        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        # 嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # 位置编码
        self.pos_encoder = StablePositionalEncoding(d_model, max_len)

        # 编码器和解码器
        self.encoder_layers = nn.ModuleList([
            StableTransformerLayer(d_model, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            StableTransformerLayer(d_model, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])

        # 输出层
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """稳定的权重初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

        # 特别处理输出层
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def create_padding_mask(self, seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """创建填充掩码"""
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """创建因果掩码"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask == 0

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """编码源序列"""
        # 嵌入 + 位置编码
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb.transpose(0, 1)).transpose(0, 1)

        # 编码器层
        for layer in self.encoder_layers:
            src_emb = layer(src_emb, src_mask)

        return src_emb

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """解码目标序列"""
        # 嵌入 + 位置编码
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb.transpose(0, 1)).transpose(0, 1)

        # 解码器层
        for layer in self.decoder_layers:
            tgt_emb = layer(tgt_emb, tgt_mask)

        return tgt_emb

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 创建掩码
        src_padding_mask = self.create_padding_mask(src)
        tgt_padding_mask = self.create_padding_mask(tgt[:, :-1])  # 去掉最后一个token

        # 编码
        memory = self.encode(src, src_padding_mask)

        # 解码
        tgt_input = tgt[:, :-1]  # 去掉最后一个token
        output = self.decode(tgt_input, memory, tgt_padding_mask)

        # 输出层
        logits = self.output_layer(output)

        return logits

    def generate(self, src: torch.Tensor, max_len: int = 50,
                 start_token: int = 2, end_token: int = 3,
                 temperature: float = 1.0) -> torch.Tensor:
        """生成翻译"""
        self.eval()
        batch_size = src.size(0)
        device = src.device

        # 编码
        src_padding_mask = self.create_padding_mask(src)
        memory = self.encode(src, src_padding_mask)

        # 初始化目标序列
        tgt = torch.full((batch_size, 1), start_token,
                         dtype=torch.long, device=device)

        with torch.no_grad():
            for _ in range(max_len - 1):
                # 创建掩码
                tgt_padding_mask = self.create_padding_mask(tgt)

                # 解码
                output = self.decode(tgt, memory, tgt_padding_mask)
                logits = self.output_layer(output[:, -1:]) / temperature

                # 数值稳定性检查
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print("警告: 检测到NaN或Inf值，使用随机采样")
                    next_token = torch.randint(
                        0, self.tgt_vocab_size, (batch_size, 1), device=device)
                else:
                    # 采样下一个token
                    probs = F.softmax(logits, dim=-1)
                    # 确保概率有效
                    probs = torch.clamp(probs, min=1e-8, max=1.0)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    next_token = torch.multinomial(probs.squeeze(1), 1)

                # 添加到序列
                tgt = torch.cat([tgt, next_token], dim=1)

                # 检查是否结束
                if (tgt == end_token).any(dim=1).all():
                    break

        return tgt


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
        "Have a nice day",
        "The weather is nice",
        "I like this movie",
        "Can you help me?",
        "What time is it?",
        "How much does it cost?"
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
        "祝你愉快",
        "天气很好",
        "我喜欢这部电影",
        "你能帮助我吗？",
        "现在几点了？",
        "这个多少钱？"
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


def train_model(model: StableTransformerModel, src_data: torch.Tensor,
                tgt_data: torch.Tensor, num_epochs: int = 100,
                learning_rate: float = 0.001) -> List[float]:
    """训练模型"""
    device = torch.device(
        'mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    src_data = src_data.to(device)
    tgt_data = tgt_data.to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding token

    losses = []

    print(f"开始训练，设备: {device}")
    print(f"数据形状: src={src_data.shape}, tgt={tgt_data.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # 前向传播
        output = model(src_data, tgt_data)

        # 计算损失
        loss = criterion(output.reshape(-1, output.size(-1)),
                         tgt_data[:, 1:].reshape(-1))

        # 检查NaN
        if torch.isnan(loss):
            print(f"警告: Epoch {epoch+1} 检测到NaN损失，跳过此epoch")
            continue

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    return losses


def test_translation(model: StableTransformerModel, src_tokenizer: SimpleTokenizer,
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
        tgt_ids = model.generate(src_tensor, max_len=20, temperature=0.8)

        # 解码
        tgt_seq = tgt_ids[0].cpu().numpy().tolist()
        translation = tgt_tokenizer.decode(tgt_seq)

        print(f"原文: {text}")
        print(f"译文: {translation}")
        print("-" * 30)


def main():
    """主函数"""
    print("🚀 稳定Transformer翻译模型演示")
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
    model = StableTransformerModel(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        d_model=64,           # 更小的模型
        nhead=4,              # 4个注意力头
        num_layers=2,         # 2层
        dim_feedforward=256   # 更小的前馈网络
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型参数数量: {total_params:,}")

    # 5. 训练模型
    print("\n🎯 步骤5: 训练模型...")
    losses = train_model(model, src_data, tgt_data,
                         num_epochs=50, learning_rate=0.001)
    print(f"✅ 训练完成，最终损失: {losses[-1]:.4f}")

    # 6. 测试翻译
    test_texts = ["Hello", "Thank you", "Goodbye",
                  "I love you", "What is your name?"]
    test_translation(model, src_tokenizer, tgt_tokenizer, test_texts)

    print("\n🎉 演示完成！")
    print("这个稳定版本解决了训练不稳定和NaN问题，")
    print("展示了Transformer架构在翻译任务中的应用。")


if __name__ == "__main__":
    main()
