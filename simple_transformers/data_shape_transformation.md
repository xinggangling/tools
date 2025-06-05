# Transformer 模型数据形态变换过程

## 1. 输入数据

- 初始输入：MNIST 图像数据（手写数字图像）
- 形状：`[batch_size, 784]` (28×28 图像展平为 1 维向量)
- 数据含义：
  - 每个数值范围在 [0,1] 之间，表示像素的灰度值
  - 0 表示黑色（背景），1 表示白色（数字笔画）
- 示例：
  ```
  batch_size = 2 时：
  [
    [0.1, 0.2, ..., 0.784],  # 第一张图片的像素值，0.1表示较暗的像素，0.784表示较亮的像素
    [0.2, 0.3, ..., 0.785]   # 第二张图片的像素值
  ]
  ```
- 代码实现：
  ```python
  # 假设我们有一个MNIST数据加载器
  batch_size = 2
  x = torch.randn(batch_size, 784)  # 模拟MNIST图像数据
  ```

## 2. 输入重塑

- 操作：将输入重塑为序列形式
- 变换：`[batch_size, 784]` → `[batch_size, 1, 784]`
- 数据含义：
  - 将每个图像视为一个长度为 1 的序列
  - 每个序列包含 784 个特征（像素值）
- 示例：
  ```
  [
    [[0.1, 0.2, ..., 0.784]],  # 第一张图片的像素序列
    [[0.2, 0.3, ..., 0.785]]   # 第二张图片的像素序列
  ]
  ```
- 代码实现：
  ```python
  # 将输入重塑为序列形式
  x = x.view(batch_size, 1, 784)  # [batch_size, 1, 784]
  ```

## 3. 输入投影

- 操作：通过线性层将输入投影到模型维度
- 变换：`[batch_size, 1, 784]` → `[batch_size, 1, d_model]`
- 数据含义：
  - 将 784 维的像素特征压缩/转换到 d_model 维的特征空间
  - 每个维度代表一个学习到的特征
- 示例（假设 d_model = 64）：
  ```
  [
    [[0.1, 0.2, ..., 0.64]],  # 第一张图片的64维特征表示
    [[0.2, 0.3, ..., 0.64]]   # 第二张图片的64维特征表示
  ]
  ```
- 代码实现：

  ```python
  # 定义输入投影层
  d_model = 64
  input_projection = nn.Linear(784, d_model)

  # 应用投影
  x = input_projection(x)  # [batch_size, 1, d_model]
  ```

## 4. 位置编码

- 操作：添加位置编码信息
- 变换：`[batch_size, 1, d_model]` → `[batch_size, 1, d_model]`
- 位置编码矩阵形状：`[1, max_len, d_model]`
- 数据含义：
  - pe_0, pe_1 等是使用正弦和余弦函数生成的位置编码值
  - 这些值帮助模型理解序列中元素的位置信息
- 示例：
  ```
  [
    [[0.1+pe_0, 0.2+pe_1, ..., 0.64+pe_63]],  # 第一张图片的特征加上位置信息
    [[0.2+pe_0, 0.3+pe_1, ..., 0.64+pe_63]]   # 第二张图片的特征加上位置信息
  ]
  ```
- 代码实现：

  ```python
  class PositionalEncoding(nn.Module):
      def __init__(self, d_model, max_len=5000):
          super().__init__()
          pe = torch.zeros(max_len, d_model)
          position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
          div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
          pe[:, 0::2] = torch.sin(position * div_term)
          pe[:, 1::2] = torch.cos(position * div_term)
          pe = pe.unsqueeze(0)
          self.register_buffer('pe', pe)

      def forward(self, x):
          return x + self.pe[:, :x.size(1)]

  # 应用位置编码
  pos_encoder = PositionalEncoding(d_model)
  x = pos_encoder(x)  # [batch_size, 1, d_model]
  ```

## 5. 多头注意力

- 操作：计算自注意力
- 输入形状：`[batch_size, 1, d_model]`
- 中间变换：
  1. Q/K/V 投影：`[batch_size, 1, d_model]` → `[batch_size, 1, d_model]`
     - Q(Query): 查询向量，用于寻找相关信息
     - K(Key): 键向量，用于匹配查询
     - V(Value): 值向量，包含实际信息
  2. 多头重塑：`[batch_size, 1, d_model]` → `[batch_size, num_heads, 1, d_k]`
     - 将注意力分成多个头，每个头关注不同的特征模式
  3. 注意力计算后：`[batch_size, num_heads, 1, d_k]` → `[batch_size, 1, d_model]`
     - 合并多个头的注意力结果
- 示例（假设 num_heads = 4, d_k = 16）：
  ```
  多头注意力输出：
  [
    [[0.3, 0.4, ..., 0.64]],  # 第一张图片的注意力加权特征
    [[0.4, 0.5, ..., 0.64]]   # 第二张图片的注意力加权特征
  ]
  ```
- 代码实现：

  ```python
  class MultiHeadAttention(nn.Module):
      def __init__(self, d_model, num_heads):
          super().__init__()
          self.num_heads = num_heads
          self.d_k = d_model // num_heads

          self.W_q = nn.Linear(d_model, d_model)
          self.W_k = nn.Linear(d_model, d_model)
          self.W_v = nn.Linear(d_model, d_model)
          self.W_o = nn.Linear(d_model, d_model)

      def forward(self, Q, K, V, mask=None):
          batch_size = Q.size(0)

          # 线性变换
          Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
          K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
          V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

          # 计算注意力
          scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
          if mask is not None:
              scores = scores.masked_fill(mask == 0, -1e9)
          attention_weights = torch.softmax(scores, dim=-1)
          output = torch.matmul(attention_weights, V)

          # 重塑并应用输出变换
          output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
          return self.W_o(output)

  # 应用多头注意力
  num_heads = 4
  attention = MultiHeadAttention(d_model, num_heads)
  x = attention(x, x, x)  # [batch_size, 1, d_model]
  ```

## 6. 残差连接和层归一化

- 操作：添加残差连接并进行层归一化
- 变换：保持形状不变 `[batch_size, 1, d_model]`
- 数据含义：
  - 残差连接：保留原始特征信息
  - 层归一化：将特征值标准化到合适的范围
- 数据结构：
  1. 残差连接前：
     ```
     [
       [[0.3, 0.4, ..., 0.64]],  # 第一张图片的注意力输出
       [[0.4, 0.5, ..., 0.64]]   # 第二张图片的注意力输出
     ]
     ```
  2. 残差连接（与原始输入相加）：
     ```
     [
       [[0.3+0.1, 0.4+0.2, ..., 0.64+0.64]],  # 第一张图片的残差连接结果
       [[0.4+0.2, 0.5+0.3, ..., 0.64+0.64]]   # 第二张图片的残差连接结果
     ]
     ```
  3. 层归一化后：
     ```
     [
       [[0.2, 0.3, ..., 0.5]],  # 第一张图片的归一化特征
       [[0.3, 0.4, ..., 0.6]]   # 第二张图片的归一化特征
     ]
     ```
- 代码实现：
  ```python
  # 残差连接和层归一化
  norm1 = nn.LayerNorm(d_model)
  x = norm1(x + x)  # 残差连接后归一化
  ```

## 7. 前馈神经网络

- 操作：通过两层线性变换
- 变换：
  1. 第一层：`[batch_size, 1, d_model]` → `[batch_size, 1, d_model*4]`
     - 扩展特征维度，增加模型容量
  2. ReLU 激活
     - 引入非线性，去除负值
  3. 第二层：`[batch_size, 1, d_model*4]` → `[batch_size, 1, d_model]`
     - 压缩回原始维度
- 数据结构：
  1. 输入数据（第一层前）：
     ```
     [
       [[0.2, 0.3, ..., 0.5]],  # 第一张图片的特征
       [[0.3, 0.4, ..., 0.6]]   # 第二张图片的特征
     ]
     ```
  2. 第一层线性变换后（扩展维度）：
     ```
     [
       [[0.1, 0.2, ..., 0.256]],  # 第一张图片的扩展特征 (d_model*4=256)
       [[0.2, 0.3, ..., 0.256]]   # 第二张图片的扩展特征
     ]
     ```
  3. ReLU 激活后：
     ```
     [
       [[0.1, 0.2, ..., 0.256]],  # 第一张图片的激活特征（负值变为0）
       [[0.2, 0.3, ..., 0.256]]   # 第二张图片的激活特征
     ]
     ```
  4. 第二层线性变换后（压缩维度）：
     ```
     [
       [[0.3, 0.4, ..., 0.64]],  # 第一张图片的压缩特征
       [[0.4, 0.5, ..., 0.64]]   # 第二张图片的压缩特征
     ]
     ```
- 代码实现：
  ```python
  # 前馈神经网络
  feed_forward = nn.Sequential(
      nn.Linear(d_model, d_model * 4),
      nn.ReLU(),
      nn.Linear(d_model * 4, d_model)
  )
  x = feed_forward(x)  # [batch_size, 1, d_model]
  ```

## 8. 输出层

- 操作：展平并投影到输出维度
- 变换：
  1. 展平：`[batch_size, 1, d_model]` → `[batch_size, d_model]`
  2. 线性投影：`[batch_size, d_model]` → `[batch_size, 2]`
- 最终输出形状：`[batch_size, 2]`
- 数据含义：
  - 两个数值分别表示属于类别 1 和类别 2 的原始分数
- 示例：
  ```
  [
    [0.7, 0.3],  # 第一张图片的原始分数：更可能属于类别1
    [0.2, 0.8]   # 第二张图片的原始分数：更可能属于类别2
  ]
  ```
- 代码实现：
  ```python
  # 输出层
  x = x.view(batch_size, -1)  # 展平
  output_layer = nn.Linear(d_model, 2)
  x = output_layer(x)  # [batch_size, 2]
  ```

## 9. Softmax 激活

- 操作：对输出进行 softmax 激活
- 变换：保持形状不变 `[batch_size, 2]`
- 输出：每个类别的概率分布
- 数据含义：
  - 将原始分数转换为概率值
  - 所有类别的概率之和为 1
- 示例：
  ```
  [
    [0.67, 0.33],  # 第一张图片：67%概率是类别1，33%概率是类别2
    [0.25, 0.75]   # 第二张图片：25%概率是类别1，75%概率是类别2
  ]
  ```
- 代码实现：
  ```python
  # Softmax激活
  predictions = torch.softmax(x, dim=1)  # [batch_size, 2]
  ```
