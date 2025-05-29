import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    位置编码类：为序列中的每个位置添加位置信息
    使用正弦和余弦函数生成位置编码，这样可以让模型知道序列中每个元素的位置
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 创建位置编码矩阵，形状为 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        # 创建位置索引，形状为 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算除数项，用于生成不同频率的正弦和余弦函数
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 偶数位置使用正弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数位置使用余弦函数
        pe[:, 1::2] = torch.cos(position * div_term)
        # 增加批次维度，形状变为 [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        # 注册为缓冲区，不参与反向传播
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将位置编码加到输入上，只取需要的长度
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制：允许模型同时关注输入序列的不同部分
    将输入分成多个头，每个头独立计算注意力，最后合并结果
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads  # 注意力头的数量
        self.d_model = d_model  # 模型的维度
        # 确保维度能被头数整除
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads  # 每个头的维度
        # 定义Q、K、V的线性变换层
        self.W_q = nn.Linear(d_model, d_model)  # 查询变换
        self.W_k = nn.Linear(d_model, d_model)  # 键变换
        self.W_v = nn.Linear(d_model, d_model)  # 值变换
        self.W_o = nn.Linear(d_model, d_model)  # 输出变换
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        计算缩放点积注意力
        Q: 查询矩阵
        K: 键矩阵
        V: 值矩阵
        mask: 掩码矩阵（可选）
        """
        # 计算注意力分数，并除以sqrt(d_k)进行缩放
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # 如果提供了掩码，将掩码位置的值设为很小的负数
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # 对分数进行softmax得到注意力权重
        attention_weights = torch.softmax(scores, dim=-1)
        # 计算加权和得到输出
        output = torch.matmul(attention_weights, V)
        return output
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 线性变换并重塑为多头形式
        # 将输入分成多个头，每个头独立计算注意力
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 重塑输出并应用输出变换
        # 将多个头的结果合并
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)

class SimpleTransformer(nn.Module):
    """
    简单的Transformer模型
    包含位置编码、多头注意力、层归一化和前馈网络
    """
    def __init__(self, d_model=512, num_heads=8, max_len=5000):
        super(SimpleTransformer, self).__init__()
        # 位置编码层
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        # 多头注意力层
        self.attention = MultiHeadAttention(d_model, num_heads)
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # 前馈神经网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),  # 扩展维度
            nn.ReLU(),  # 激活函数
            nn.Linear(d_model * 4, d_model)  # 压缩维度
        )
        
    def forward(self, x, mask=None):
        # 添加位置编码
        x = self.positional_encoding(x)
        
        # 多头注意力 + 残差连接 + 层归一化
        attention_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attention_output)  # 残差连接后归一化
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)  # 残差连接后归一化
        
        return x

# 使用示例
if __name__ == "__main__":
    # 创建一个简单的transformer模型
    # d_model=512: 模型维度为512
    # num_heads=8: 使用8个注意力头
    model = SimpleTransformer(d_model=512, num_heads=8)
    
    # 创建一个示例输入
    batch_size = 2  # 批次大小为2
    seq_length = 10  # 序列长度为10
    x = torch.randn(batch_size, seq_length, 512)  # 随机生成输入张量
    
    # 前向传播
    output = model(x)
    print(f"输入形状: {x.shape}")  # 输出输入张量的形状
    print(f"输出形状: {output.shape}")  # 输出输出张量的形状 