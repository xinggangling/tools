import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    位置编码：给输入序列添加位置信息
    使用正弦和余弦函数生成位置编码，帮助模型理解序列中token的相对位置
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # 创建位置矩阵 [max_len, 1]
        position = torch.arange(max_len).unsqueeze(1).float()

        # 创建除数项，用于生成不同频率的正弦/余弦波
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, 1, d_model)

        # 偶数位置使用正弦函数
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        # 奇数位置使用余弦函数
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # 注册为buffer，不参与梯度更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: 输入张量 [seq_len, batch_size, d_model]
        Returns:
            添加位置编码后的张量
        """
        return x + self.pe[:x.size(0)]


class TransformerEncoderLayer(nn.Module):
    """
    简化的Transformer Encoder层
    包含自注意力机制和前馈神经网络，使用残差连接和层归一化
    """

    def __init__(self, d_model, nhead=2, dim_feedforward=1024, dropout=0.1):
        super().__init__()

        # 多头自注意力层
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # 前馈神经网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            src_mask: 源序列掩码 [batch_size, seq_len] (填充掩码)
        Returns:
            编码后的张量
        """
        # 将填充掩码转换为注意力掩码
        if src_mask is not None:
            # 创建注意力掩码 [seq_len, seq_len]
            seq_len = x.size(1)
            attn_mask = torch.zeros(seq_len, seq_len, device=x.device)
            # 对于每个序列，将填充位置标记为-inf
            for i in range(x.size(0)):
                valid_len = src_mask[i].sum().item()
                attn_mask[valid_len:, :] = float('-inf')
                attn_mask[:, valid_len:] = float('-inf')
        else:
            attn_mask = None

        # 自注意力 + 残差连接
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # 前馈网络 + 残差连接
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


class TransformerDecoderLayer(nn.Module):
    """
    简化的Transformer Decoder层
    包含掩码自注意力、交叉注意力和前馈神经网络
    """

    def __init__(self, d_model, nhead=2, dim_feedforward=1024, dropout=0.1):
        super().__init__()

        # 掩码多头自注意力（用于目标序列）
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # 交叉注意力（目标序列关注编码器输出）
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # 前馈神经网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        """
        Args:
            x: 目标序列 [batch_size, tgt_len, d_model]
            memory: 编码器输出 [batch_size, src_len, d_model]
            tgt_mask: 目标序列掩码 [tgt_len, tgt_len] (因果掩码)
            memory_mask: 内存掩码 [src_len, src_len] (填充掩码)
        Returns:
            解码后的张量
        """
        # 掩码自注意力
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # 交叉注意力
        attn_output, _ = self.cross_attn(
            x, memory, memory, attn_mask=memory_mask)
        x = x + self.dropout(attn_output)
        x = self.norm2(x)

        # 前馈网络
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout(ff_output)
        x = self.norm3(x)

        return x


class TranslationModel(nn.Module):
    """
    完整的翻译模型：Encoder-Decoder架构
    针对Mac M4优化的轻量化设计
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256,
                 nlayers=2, nhead=2, dim_feedforward=1024, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        # 词嵌入层
        self.embedding_src = nn.Embedding(src_vocab_size, d_model)
        self.embedding_tgt = nn.Embedding(tgt_vocab_size, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # 编码器：多层Transformer Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(nlayers)
        ])

        # 解码器：多层Transformer Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(nlayers)
        ])

        # 输出层：将隐藏状态映射到目标词汇表
        self.fc = nn.Linear(d_model, tgt_vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                # 使用Xavier初始化
                nn.init.xavier_uniform_(p)
            else:
                # 偏置项初始化为0
                nn.init.zeros_(p)

        # 特别处理输出层
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def generate_square_subsequent_mask(self, sz):
        """
        生成下三角掩码矩阵，用于解码器的自注意力
        确保模型只能看到当前位置及之前的信息
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, src, src_mask=None):
        """
        编码源序列
        Args:
            src: 源序列 [batch_size, src_len]
            src_mask: 源序列掩码
        Returns:
            编码后的表示 [batch_size, src_len, d_model]
        """
        # 词嵌入 + 位置编码
        src_emb = self.embedding_src(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb.transpose(0, 1)).transpose(0, 1)
        src_emb = self.dropout(src_emb)

        # 通过编码器层
        memory = src_emb
        for encoder_layer in self.encoder_layers:
            memory = encoder_layer(memory, src_mask)

        return memory

    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        解码目标序列
        Args:
            tgt: 目标序列 [batch_size, tgt_len]
            memory: 编码器输出 [batch_size, src_len, d_model]
            tgt_mask: 目标序列掩码
            memory_mask: 内存掩码
        Returns:
            解码后的表示 [batch_size, tgt_len, d_model]
        """
        # 词嵌入 + 位置编码
        tgt_emb = self.embedding_tgt(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb.transpose(0, 1)).transpose(0, 1)
        tgt_emb = self.dropout(tgt_emb)

        # 通过解码器层
        output = tgt_emb
        for decoder_layer in self.decoder_layers:
            output = decoder_layer(output, memory, tgt_mask, memory_mask)

        return output

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        前向传播
        Args:
            src: 源序列 [batch_size, src_len]
            tgt: 目标序列 [batch_size, tgt_len]
            src_mask: 源序列掩码 [batch_size, src_len] (填充掩码)
            tgt_mask: 目标序列掩码 [batch_size, tgt_len] (填充掩码)
        Returns:
            预测的logits [batch_size, tgt_len, tgt_vocab_size]
        """
        # 编码
        memory = self.encode(src, src_mask)

        # 解码（去掉最后一个token，用于教师强制训练）
        tgt_input = tgt[:, :-1]

        # 为解码器生成掩码
        if tgt_mask is not None:
            tgt_mask = tgt_mask[:, :-1]  # 去掉最后一个token的掩码
            # 生成下三角掩码用于自注意力
            seq_len = tgt_input.size(1)
            causal_mask = self.generate_square_subsequent_mask(
                seq_len).to(src.device)
        else:
            causal_mask = None

        output = self.decode(tgt_input, memory, causal_mask)

        # 输出层
        logits = self.fc(output)

        return logits

    def count_parameters(self):
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def generate(self, src, max_len=50, start_token=1, end_token=2, temperature=1.0):
        """
        生成翻译结果
        Args:
            src: 源序列 [batch_size, src_len]
            max_len: 最大生成长度
            start_token: 开始标记
            end_token: 结束标记
            temperature: 采样温度
        Returns:
            生成的序列 [batch_size, generated_len]
        """
        self.eval()
        batch_size = src.size(0)

        # 编码源序列
        memory = self.encode(src)

        # 初始化目标序列
        tgt = torch.full((batch_size, 1), start_token,
                         dtype=torch.long, device=src.device)

        with torch.no_grad():
            for i in range(max_len - 1):
                # 生成掩码
                tgt_mask = self.generate_square_subsequent_mask(
                    tgt.size(1)).to(src.device)

                # 解码
                output = self.decode(tgt, memory, tgt_mask)
                logits = self.fc(output[:, -1:]) / temperature

                # 数值稳定性检查
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print("警告: 检测到NaN或Inf值，使用随机采样")
                    next_token = torch.randint(
                        0, self.tgt_vocab_size, (batch_size, 1), device=src.device)
                else:
                    # 采样下一个token
                    probs = F.softmax(logits, dim=-1)
                    # 确保概率有效
                    probs = torch.clamp(probs, min=1e-8, max=1.0)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    next_token = torch.multinomial(probs.squeeze(1), 1)

                # 添加到序列
                tgt = torch.cat([tgt, next_token], dim=1)

                # 检查是否所有序列都结束
                if (tgt == end_token).any(dim=1).all():
                    break

        return tgt


def create_translation_model(src_vocab_size, tgt_vocab_size, config=None):
    """
    创建翻译模型的工厂函数
    Args:
        src_vocab_size: 源语言词汇表大小
        tgt_vocab_size: 目标语言词汇表大小
        config: 配置字典
    Returns:
        翻译模型实例
    """
    if config is None:
        config = {
            'd_model': 256,
            'nlayers': 2,
            'nhead': 2,
            'dim_feedforward': 1024,
            'dropout': 0.1
        }

    model = TranslationModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config['d_model'],
        nlayers=config['nlayers'],
        nhead=config['nhead'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    )

    return model
