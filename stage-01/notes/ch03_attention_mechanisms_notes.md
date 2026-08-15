# Stage 01 · Chapter 3 学习笔记：Coding Attention Mechanisms

> 教材：Sebastian Raschka《Build a Large Language Model From Scratch》
> 代码路径：`stage-01/code/ch03/`
> 学习日期：2026-08-15

---

## 一、章节总览

本章是整本书的**核心技术章节**，从零开始手写实现 Transformer 架构中最关键的组件——**注意力机制（Attention Mechanism）**。

### 学习路线图

```
3.1 问题背景：RNN 搞不定长序列
        ↓
3.2 注意力机制：让模型"选择性关注"
        ↓
3.3 自注意力入门：无训练权重的简化版（理解原理）
        ↓
3.4 带可训练权重的自注意力：Q/K/V 三矩阵登场
        ↓
3.5 因果注意力：用掩码遮住"未来"的词（自回归生成的关键）
        ↓
3.6 多头注意力：多个注意力头并行工作，捕捉不同模式
```

### 核心概念速查表

| 概念 | 一句话解释 |
|------|-----------|
| **Attention Scores (ω)** | 未归一化的注意力分数，query 和 key 的点积 |
| **Attention Weights** | 经 softmax 归一化后的分数，每行和为 1 |
| **Query (Q)** | "我在找什么"——当前 token 的查询向量 |
| **Key (K)** | "我有什么"——每个 token 的键向量 |
| **Value (V)** | "我的内容是什么"——每个 token 的值向量 |
| **Context Vector** | 注意力加权求和后的输出向量 |
| **Causal Mask** | 下三角掩码，禁止看到未来 token |
| **Scaled Dot-Product** | 点积后除以 √d_k，防止梯度消失 |

---

## 二、3.1 长序列建模的问题

### 为什么需要注意力？

在 Transformer 出现之前，机器翻译主要用 **Encoder-Decoder RNN** 架构：

- **Encoder**：把源语言句子逐个 token 读入，压缩成一个**固定长度的隐藏状态向量**
- **Decoder**：从这个向量出发，逐个生成目标语言 token

### RNN 的致命缺陷

1. **信息瓶颈**：整个句子的所有信息都要挤进一个固定维度的向量，长句子必然丢失信息
2. **顺序依赖**：必须按时间步逐个计算，无法并行化，训练极慢
3. **长距离遗忘**：梯度在时间维度上反复传播，容易梯度消失/爆炸，句子开头的信息到结尾基本忘了

> 类比：RNN 就像一个人读完整篇文章后，只能用一句话总结，然后让你根据这一句话翻译全文——信息损失是必然的。

---

## 三、3.2 注意力机制的核心思想

### 解决思路

注意力机制的本质：**Decoder 在生成每个 token 时，不再只依赖一个压缩向量，而是可以"回头看"Encoder 的所有输入 token，并根据相关性给每个 token 分配不同的权重。**

### 自注意力（Self-Attention）

在 Transformer 的 Decoder 中，注意力不是跨 Encoder-Decoder，而是**在同一个序列内部**进行的：

- 序列中的每个位置都可以关注序列中的其他所有位置
- 通过学习到的权重，决定"当前 token 应该多大程度上参考其他 token"

> 类比：自注意力就像开会时每个人都可以听其他人发言，但每个人会根据话题相关性，决定更认真听谁的话。

---

## 四、3.3 自注意力机制（无训练权重版）

> ⚠️ 本节是**纯概念演示**，没有可训练参数，不是真正 Transformer 用的注意力。目的是先把计算流程搞懂。

### 3.3.1 简化版自注意力的计算步骤

假设输入序列已经被转成 embedding 向量（示例用 6 个 token，每个 3 维）：

```python
inputs = torch.tensor([
  [0.43, 0.15, 0.89],  # token 1
  [0.55, 0.87, 0.66],  # token 2 (作为 query)
  [0.57, 0.85, 0.64],  # token 3
  [0.22, 0.58, 0.33],  # token 4
  [0.77, 0.25, 0.10],  # token 5
  [0.05, 0.80, 0.55]   # token 6
])
```

#### Step 1：计算注意力分数（Attention Scores）

以第 2 个 token 作为 query，计算它和所有 token 的点积：

```python
query = inputs[1]  # 第2个token作为query
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
```

- 点积的几何意义：两个向量越相似（方向一致），点积越大
- `ω_21` 表示"第2个 token 作为 query，和第1个 token 的相关性分数"

#### Step 2：归一化为注意力权重（Attention Weights）

```python
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
```

- softmax 把任意实数转成概率分布，所有值在 (0,1) 之间且和为 1
- **为什么不用简单的除法归一化？** softmax 对大值更敏感，能放大差异；且数值稳定性更好

#### Step 3：计算上下文向量（Context Vector）

```python
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
```

- 上下文向量 = 所有输入 token 的**加权平均**
- 权重越大的 token，对最终输出的贡献越大

### 3.3.2 推广到所有 token：矩阵化实现

上面只算了一个 query 的情况，实际要对每个 token 都算一遍。用矩阵乘法一次性搞定：

```python
# 所有注意力分数：(6, 6) 矩阵，第 i 行是 token i 作为 query 的分数
attn_scores = inputs @ inputs.T

# 每行做 softmax 归一化
attn_weights = torch.softmax(attn_scores, dim=-1)

# 所有上下文向量一次性算出
all_context_vecs = attn_weights @ inputs  # (6, 3)
```

**关键理解**：
- `attn_scores[i, j]` = token i 和 token j 的相似度
- `attn_weights[i, :]` = token i 对所有 token 的关注权重（和为1）
- `all_context_vecs[i, :]` = token i 的上下文向量

---

## 五、3.4 带可训练权重的自注意力

### 核心升级：引入 Q / K / V 三个投影矩阵

上面的简化版直接用原始 embedding 做点积，问题是：
- 同一个向量既要当 query 又要当 key 还要当 value，角色混乱
- 没有可训练参数，无法学习"该关注什么"

真正的自注意力通过三个独立的**可训练权重矩阵**把输入投影到三个不同空间：

| 矩阵 | 作用 | 维度 |
|------|------|------|
| `W_query` | 把输入投影成查询向量 Q | `(d_in, d_out)` |
| `W_key` | 把输入投影成键向量 K | `(d_in, d_out)` |
| `W_value` | 把输入投影成值向量 V | `(d_in, d_out)` |

### 逐步计算流程

```python
d_in = 3   # 输入embedding维度
d_out = 2  # 输出维度

# 初始化三个可训练矩阵
W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out))
W_value = torch.nn.Parameter(torch.rand(d_in, d_out))

# 投影得到 Q, K, V
queries = inputs @ W_query  # (6, 2)
keys    = inputs @ W_key    # (6, 2)
values  = inputs @ W_value  # (6, 2)

# 计算注意力分数
attn_scores = queries @ keys.T  # (6, 6)

# 缩放点积（Scaled Dot-Product）
d_k = keys.shape[1]  # = 2
attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)

# 计算上下文向量
context_vec = attn_weights @ values  # (6, 2)
```

### 为什么要除以 √d_k？（Scaled Dot-Product）

当 `d_k` 很大时，点积的数值会变得很大，导致 softmax 进入梯度极小的饱和区域（类似 sigmoid 的两端）。

- 除以 √d_k 把点积的方差拉回到 ~1
- 保证 softmax 的梯度健康，训练能正常收敛

> 类比：就像考试分数如果满分是10000分，大家的分数差异在百分比上就不明显了；除以一个缩放因子让分数分布更合理。

### 类封装：SelfAttention_v1 → SelfAttention_v2

**v1（手写 Parameter）**：
```python
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec
```

**v2（用 nn.Linear 简化）**：
```python
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec
```

> `nn.Linear` 内部就是 `x @ W.T + b`，和手动 Parameter 等价，但初始化更规范（Kaiming/Xavier），代码更简洁。

---

## 六、3.5 因果注意力（Causal Attention）

### 为什么需要因果掩码？

在自回归语言模型中，生成第 `t` 个 token 时，**只能看到前 `t` 个 token**，不能看到未来的 token（否则就是作弊）。

因果注意力通过**下三角掩码**实现这一约束：

```
token 1: [1, 0, 0, 0, 0, 0]  只能看自己
token 2: [1, 1, 0, 0, 0, 0]  能看 1,2
token 3: [1, 1, 1, 0, 0, 0]  能看 1,2,3
token 4: [1, 1, 1, 1, 0, 0]  ...
token 5: [1, 1, 1, 1, 1, 0]
token 6: [1, 1, 1, 1, 1, 1]  能看全部
```

### 两种掩码实现方式

#### 方式一：softmax 后乘掩码（不推荐）

```python
mask_simple = torch.tril(torch.ones(context_length, context_length))
masked_simple = attn_weights * mask_simple
# 问题：乘完后每行和不再是1，需要重新归一化
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
```

缺点：先 softmax 再掩码，被掩码的位置虽然变成0，但其他位置的权重分布已经被"污染"了，需要额外重新归一化。

#### 方式二：softmax 前填 -∞（标准做法）

```python
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
```

原理：`softmax(-∞) = 0`，未来位置的注意力权重自然变成0，且其他位置的权重自动归一化，无需额外处理。

### Dropout 正则化

在注意力权重上应用 dropout，随机把一部分权重置零，防止过拟合：

```python
dropout = torch.nn.Dropout(0.5)  # 50%概率置零
attn_weights = dropout(attn_weights)
```

- 训练时启用，推理时自动关闭（`model.eval()`）
- 未被置零的值会按 `1/(1-p)` 缩放，保证期望不变

### 完整 CausalAttention 类

```python
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # register_buffer: 不参与梯度更新，但会随模型保存/移动到GPU
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # 新增batch维度b
        keys = self.W_key(x)       # (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 注意转置维度：(b, num_tokens, d_out) -> 对后两维转置
        attn_scores = queries @ keys.transpose(1, 2)  # (b, num_tokens, num_tokens)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf
        )
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values  # (b, num_tokens, d_out)
        return context_vec
```

### 关键细节：register_buffer

`register_buffer` 注册的张量：
- ✅ 不参与梯度更新（不是 Parameter）
- ✅ 会随 `model.to(device)` 移动到 GPU/CPU
- ✅ 会保存在 `state_dict` 中，模型保存/加载时自动带上
- 适合存放掩码这种"固定不变但需要跟着模型走"的张量

---

## 七、3.6 多头注意力（Multi-Head Attention）

### 为什么需要多个头？

单个注意力头只能学习一种"关注模式"。多头注意力让模型**同时从多个不同角度关注输入**：
- 有的头可能关注语法关系（主谓宾）
- 有的头可能关注指代关系（代词指向谁）
- 有的头可能关注位置邻近性

### 实现方式一：MultiHeadAttentionWrapper（直观但低效）

直接堆叠多个独立的 `CausalAttention`，最后在特征维度拼接：

```python
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        # 每个头输出 (b, num_tokens, d_out)，拼接后 (b, num_tokens, d_out * num_heads)
        return torch.cat([head(x) for head in self.heads], dim=-1)
```

缺点：每个头独立做矩阵乘法，无法利用并行计算优化。

### 实现方式二：MultiHeadAttention（标准高效实现）

**核心思想**：用一个大的 Q/K/V 矩阵一次性投影，然后在特征维度上"切"成多个头。

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 每个头的维度

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 头输出融合投影
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # 一次性投影到 d_out 维度
        keys = self.W_key(x)      # (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 切分成多个头：(b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 把头维度放到前面，方便并行计算：(b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 每个头独立计算注意力分数
        attn_scores = queries @ keys.transpose(2, 3)  # (b, num_heads, num_tokens, num_tokens)

        # 因果掩码
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 缩放 + softmax + dropout
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算上下文向量：(b, num_heads, num_tokens, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)  # 换回 (b, num_tokens, num_heads, head_dim)

        # 合并多头：(b, num_tokens, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # 可选的输出投影

        return context_vec
```

### 维度变换全景图

```
输入 x:               (b, num_tokens, d_in)
    ↓ W_query/W_key/W_value 投影
Q, K, V:              (b, num_tokens, d_out)
    ↓ view 切分
                      (b, num_tokens, num_heads, head_dim)
    ↓ transpose(1,2)
                      (b, num_heads, num_tokens, head_dim)
    ↓ Q @ K^T
attn_scores:          (b, num_heads, num_tokens, num_tokens)
    ↓ mask + softmax + dropout
attn_weights:         (b, num_heads, num_tokens, num_tokens)
    ↓ @ V
context_vec:          (b, num_heads, num_tokens, head_dim)
    ↓ transpose(1,2) + contiguous + view
                      (b, num_tokens, d_out)
    ↓ out_proj
输出:                 (b, num_tokens, d_out)
```

### 输出投影层（out_proj）的作用

多头拼接后，加一个线性层 `out_proj`：
- 让模型学会如何融合不同头的输出
- 是 Transformer 标准实现的一部分
- 原论文中这个投影层是必须的

---

## 八、Bonus 材料

### Bonus 1：高效多头注意力实现比较

路径：`02_bonus_efficient-multihead-attention/`

对比多种 MHA 实现的性能：
- 朴素循环实现
- `nn.MultiheadAttention`（PyTorch 内置）
- `scaled_dot_product_attention`（PyTorch 2.0+，融合算子，支持 Flash Attention）
- 编译优化（`torch.compile`）

**结论**：`scaled_dot_product_attention` + `torch.compile` 性能最优，前向+反向传播比朴素实现快数倍。

### Bonus 2：理解 PyTorch Buffers

路径：`03_understanding-buffers/`

深入讲解 `register_buffer` 的机制：
- Buffer vs Parameter 的区别
- 为什么掩码要用 buffer 而不是普通张量
- buffer 在 `state_dict`、设备迁移、模型保存中的行为

---

## 九、易错点与面试高频题

### 1. 为什么注意力分数要除以 √d_k？
- 防止点积数值过大导致 softmax 梯度消失
- 保持点积方差约为1，训练稳定

### 2. 因果掩码为什么用 -∞ 而不是 0？
- softmax(-∞) = 0，自然实现权重为0且自动归一化
- 如果在 softmax 后乘0掩码，需要额外重新归一化，且梯度计算更复杂

### 3. MultiHeadAttention 中 d_out 和 num_heads 的关系？
- `d_out` 必须能被 `num_heads` 整除
- 每个头的维度 `head_dim = d_out / num_heads`
- 实际 Transformer 中 d_out 通常等于 d_model（如 512、768、4096）

### 4. register_buffer 和 nn.Parameter 的区别？
| | Parameter | Buffer |
|---|---|---|
| 梯度更新 | ✅ 是 | ❌ 否 |
| 计入 model.parameters() | ✅ | ❌ |
| 计入 state_dict | ✅ | ✅ |
| 随 .to(device) 迁移 | ✅ | ✅ |
| 典型用途 | 权重、偏置 | 掩码、位置编码 |

### 5. 为什么自注意力是 O(n²) 复杂度？
- 注意力分数矩阵大小为 `(num_tokens, num_tokens)`
- 序列长度翻倍，计算量和内存都翻4倍
- 这也是长上下文 LLM 的核心瓶颈（催生了 Flash Attention、线性注意力等优化）

### 6. Q、K、V 为什么要用三个不同的投影矩阵？
- Q：当前 token"想找什么"
- K：每个 token"能提供什么索引"
- V：每个 token"实际内容是什么"
- 三者分离让模型能学习更灵活的关注模式；如果共用一个矩阵，表达能力会受限

---

## 十、本章代码文件索引

| 文件 | 内容 |
|------|------|
| `01_main-chapter-code/ch03.ipynb` | 主章节完整代码 notebook |
| `01_main-chapter-code/multihead-attention.ipynb` | 精简版多头注意力实现 |
| `01_main-chapter-code/exercise-solutions.ipynb` | 章节习题解答 |
| `02_bonus_efficient-multihead-attention/` | 高效 MHA 实现对比 + benchmark |
| `03_understanding-buffers/` | PyTorch Buffer 机制详解 |

---

## 十一、下一步衔接

学完本章后，你已经掌握了 Transformer 最核心的组件。后续章节将：
- **Ch04**：用本章的 MultiHeadAttention 搭建完整的 GPT 模型架构
- **Ch05**：在真实数据上预训练 GPT
- **Ch06**：指令微调（Instruction Fine-tuning）

> 建议在进入 Ch04 前，能手写一遍 `MultiHeadAttention` 的 forward 方法，特别是维度变换部分——这是面试和实际开发中最常考的点。
