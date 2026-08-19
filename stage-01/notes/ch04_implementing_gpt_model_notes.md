# Ch4 从零实现 GPT 模型

> 教材：Sebastian Raschka《Build a Large Language Model From Scratch》
> 日期：2026-08-18
> 代码位置：`code/ch04/01_main-chapter-code/gpt.py`

---

## 章节总览

Ch4 是整个教材的**架构核心章**——把前两章学的零件（分词、注意力）组装成一个完整可运行的 GPT 模型。本章只搭架构 + 做贪心解码推理，**不训练**（训练留到 Ch5）。

最终目标：手写一个 GPT-2 small（124M 参数）级别的完整模型，输入 "Hello, I am"，能输出一串 token（虽然随机权重下输出是乱码，但前向传播是通的）。

### 章节路线图

| 小节 | 内容 | 核心产出 |
|------|------|----------|
| 4.1 | LLM 架构总览 | GPT_CONFIG_124M 配置字典 |
| 4.2 | LayerNorm 层归一化 | `LayerNorm` 类 |
| 4.3 | GELU 激活 + 前馈网络 | `GELU`、`FeedForward` 类 |
| 4.4 | 残差连接（shortcut） | 梯度消失实验对比 |
| 4.5 | Transformer 块 | `TransformerBlock` 类 |
| 4.6 | 完整 GPT 模型 | `GPTModel` 类 |
| 4.7 | 文本生成 | `generate_text_simple` 贪心解码 |

---

## 4.1 GPT-2 124M 模型配置

### 配置字典

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # 词表大小（GPT-2 BPE 分词器）
    "context_length": 1024,  # 最大上下文长度（位置编码上限）
    "emb_dim": 768,          # 嵌入维度（每个 token 变成 768 维向量）
    "n_heads": 12,           # 注意力头数（768 / 12 = 64 维/头）
    "n_layers": 12,          # Transformer 块堆叠层数
    "drop_rate": 0.1,        # Dropout 比率
    "qkv_bias": False        # QKV 投影是否带偏置（GPT-2 不用）
}
```

### 通俗理解

- **vocab_size=50257**：GPT-2 的 BPE 词表有 50257 个 token，输入输出都在这个空间里
- **context_length=1024**：模型最多一次看 1024 个 token，超出的部分位置编码没见过
- **emb_dim=768**：每个 token 被映射成 768 维的稠密向量，这是模型内部的"思考维度"
- **n_heads=12**：12 个注意力头并行工作，每个头负责 768/12=64 维
- **n_layers=12**：12 个 Transformer 块叠起来，逐层提炼语义
- **drop_rate=0.1**：训练时随机丢弃 10% 神经元防过拟合
- **qkv_bias=False**：GPT-2 风格，QKV 线性层不加偏置项

### GPT-2 家族其他配置（习题参考）

| 模型 | emb_dim | n_layers | n_heads | 参数量 |
|------|---------|----------|---------|--------|
| GPT-2 small | 768 | 12 | 12 | 124M |
| GPT-2 medium | 1024 | 24 | 16 | 355M |
| GPT-2 large | 1280 | 36 | 20 | 774M |
| GPT-2 XL | 1600 | 48 | 25 | 1558M |

---

## 4.2 LayerNorm 层归一化

### 为什么需要 LayerNorm

深度神经网络中，每一层的输入分布会随着前层参数更新而漂移（内部协变量偏移），导致训练不稳定、收敛慢。LayerNorm 把每个样本的特征归一化到均值 0、方差 1，让梯度更平稳。

### 核心公式

```
x_norm = (x - mean) / sqrt(var + eps)
output = scale * x_norm + shift
```

- `mean`、`var`：沿最后一个维度（特征维度）计算
- `eps=1e-5`：防止除零
- `scale`（初始=1）、`shift`（初始=0）：可训练参数，模型自己学要不要调整归一化后的分布

### 代码实现

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 可训练缩放
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可训练偏移

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 总体方差（除以N，不是N-1）
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

### 易错点

1. **`unbiased=False`**：PyTorch 默认 `var()` 用无偏估计（除以 N-1），但 LayerNorm 论文用总体方差（除以 N），必须显式设 `unbiased=False`
2. **`dim=-1`**：沿最后一维（特征维）归一化，不是沿 batch 维（那是 BatchNorm）
3. **`keepdim=True`**：保持维度以便广播
4. **scale/shift 是 nn.Parameter**：不是 buffer，会被优化器更新

### LayerNorm vs BatchNorm 对比

| 维度 | LayerNorm | BatchNorm |
|------|-----------|-----------|
| 归一化轴 | 每个样本的特征维 | 整个 batch 的同一特征 |
| 依赖 batch 大小 | 不依赖 | 依赖（batch 小效果差） |
| 推理时 | 直接用 | 需要保存运行时均值/方差 |
| NLP 适用性 | 适合（序列长度可变） | 不适合 |

---

## 4.3 GELU 激活 + 前馈网络

### GELU vs ReLU

**ReLU**：`max(0, x)`，简单粗暴，负数直接清零，梯度在负数区为 0。

**GELU**（Gaussian Error Linear Unit）：平滑版的 ReLU，负数区有微小的非零梯度。

精确公式：`GELU(x) = x * Φ(x)`，其中 Φ 是标准正态分布的累积分布函数。

GPT-2 使用的近似公式（计算更快）：

```
GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

### 代码实现

```python
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
```

### 前馈网络 FeedForward

Transformer 块中的"MLP 部分"，先升维再降维：

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 升维 768 -> 3072
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 降维 3072 -> 768
        )

    def forward(self, x):
        return self.layers(x)
```

### 为什么是 4 倍扩展？

- 768 → 3072 → 768，中间层是输入的 4 倍
- 这是 Transformer 论文的标准配置，给模型足够的非线性表达能力
- 参数量：768×3072×2 ≈ 4.7M（每个 FFN），12 层就是 ~56M
- 现代模型（Llama、Qwen）常用 SwiGLU 替代 GELU，扩展比例约 8/3 倍

---

## 4.4 残差连接（Shortcut / Residual Connection）

### 核心思想

把输入直接加到输出上：`output = x + sublayer(x)`，给梯度一条"高速公路"。

### 为什么能缓解梯度消失？

反向传播时，梯度 = 直接路径的梯度（恒等映射，梯度=1）+ 子层路径的梯度。即使子层梯度很小，直接路径保证梯度不会消失。

### 代码模式

```python
shortcut = x
x = self.norm(x)
x = self.sublayer(x)
x = x + shortcut  # 残差连接
```

### Pre-LN vs Post-LN

本章 GPT-2 使用 **Pre-LN**（先归一化再进子层）：

```
x = x + Attention(LayerNorm(x))      # 注意力残差
x = x + FeedForward(LayerNorm(x))    # 前馈残差
```

原始 Transformer 论文用 **Post-LN**（先子层再归一化）：

```
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FeedForward(x))
```

Pre-LN 训练更稳定，不需要 warmup；Post-LN 效果可能更好但训练更难。现代 LLM 基本都用 Pre-LN。

---

## 4.5 TransformerBlock Transformer 块

### 架构图（文字版）

```
输入 x (b, seq_len, 768)
  │
  ├─ shortcut ────────────────────┐
  │                               │
  ├─ LayerNorm1                   │
  ├─ MultiHeadAttention           │
  ├─ Dropout                      │
  └─ + (残差相加) ◄────────-───────┘
  │
  ├─ shortcut ────────────────────┐
  │                               │
  ├─ LayerNorm2                   │
  ├─ FeedForward                  │
  ├─ Dropout                      │
  └─ + (残差相加) ◄───────-────────┘
  │
输出 x (b, seq_len, 768)
```

### 代码实现

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 注意力子层 + 残差
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # 前馈子层 + 残差
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
```

### 关键观察

- 输入输出形状完全相同：`(b, seq_len, emb_dim)`，所以可以无限堆叠
- 每个块有两个残差连接（注意力后、前馈后）
- Dropout 只加在残差相加之前，不影响 shortcut 路径
- LayerNorm 在子层之前（Pre-LN）

---

## 4.6 GPTModel 完整模型

### 整体架构

```
输入 token ids (b, seq_len)
  │
  ├─ Token Embedding (50257 -> 768)
  ├─ + Positional Embedding (1024 -> 768)
  ├─ Dropout
  │
  ├─ TransformerBlock × 12
  │
  ├─ Final LayerNorm
  ├─ Output Linear (768 -> 50257, 无偏置)
  │
输出 logits (b, seq_len, 50257)
```

### 代码实现

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

### 位置编码的实现

GPT-2 使用**可学习的位置嵌入**（不是正弦编码）：

```python
self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
# 前向时：
pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
```

- 直接用 `nn.Embedding` 查表，位置 0~1023 各对应一个 768 维向量
- 这些向量是可训练参数，训练过程中模型自己学会位置信息
- `torch.arange(seq_len)` 生成 `[0, 1, 2, ..., seq_len-1]` 作为位置索引

### 参数量分析

```python
total_params = sum(p.numel() for p in model.parameters())
# 结果：163,008,384 ≈ 163M
```

为什么不是 124M？因为**没有做权重共享（weight tying）**。

- `tok_emb`：50257 × 768 = 38.6M
- `out_head`：768 × 50257 = 38.6M（和 tok_emb 形状完全一样）
- 12 层 TransformerBlock：约 85.8M
- 总计：38.6 + 38.6 + 85.8 = 163M

GPT-2 原论文让 `out_head.weight = tok_emb.weight`（共享权重），减去 38.6M 后就是 124M。

> 教材选择不做权重共享，因为训练更简单；Ch5 加载预训练权重时会再应用。

### 内存估算

```python
# 每个参数 float32 占 4 字节
memory_bytes = total_params * 4  # ≈ 652 MB
# 加上梯度、优化器状态，训练时实际占用约 2-3 GB
```

---

## 4.7 文本生成（贪心解码）

### 自回归生成原理

LLM 一次只预测**下一个 token**，然后把预测结果拼回输入，再预测下一个，循环往复。

```
输入: "Hello, I am"
  → 模型预测下一个 token: "Feature" (随机权重下是乱的)
  → 输入变为: "Hello, I am Feature"
  → 模型预测下一个: "iman"
  → ... 循环 max_new_tokens 次
```

### 代码实现

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        # 截断：只保留最后 context_size 个 token
        idx_cond = idx[:, -context_size:]

        # 前向传播（不计算梯度）
        with torch.no_grad():
            logits = model(idx_cond)

        # 只取最后一个位置的 logits: (b, vocab_size)
        logits = logits[:, -1, :]

        # 贪心：选概率最大的 token（argmax logits 等价于 argmax prob）
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (b, 1)

        # 拼接到序列末尾
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
```

### 关键细节

1. **`idx[:, -context_size:]`**：上下文截断。如果输入超过模型支持的最大长度，只保留最后 `context_size` 个 token（因为位置编码只支持到 1024）
2. **`logits[:, -1, :]`**：只取最后一个时间步的预测。模型输出是 `(b, seq_len, vocab_size)`，每个位置都预测下一个 token，但生成时只关心最后一个位置
3. **`torch.argmax`**：贪心解码，永远选概率最高的。简单快速但输出单调
4. **`torch.no_grad()`**：推理时不需要计算梯度，省内存
5. **`keepdim=True`**：保持 `(b, 1)` 形状，才能和原序列 `(b, n)` 在 dim=1 上拼接

### 贪心解码的局限

- 永远选最高概率 token，输出容易重复、单调
- 没有随机性，相同输入永远得到相同输出
- Ch5 会引入温度缩放、top-k、top-p 等更高级的采样策略

---

## 完整前向传播维度追踪

以 batch=2, seq_len=4 为例：

```
输入 in_idx:              (2, 4)                    token id
    ↓ tok_emb
tok_embeds:               (2, 4, 768)
    ↓ + pos_emb (4, 768)  广播
x:                        (2, 4, 768)
    ↓ drop_emb
x:                        (2, 4, 768)
    ↓ trf_blocks × 12（每层形状不变）
x:                        (2, 4, 768)
    ↓ final_norm
x:                        (2, 4, 768)
    ↓ out_head (768 -> 50257)
logits:                   (2, 4, 50257)
```

单个 TransformerBlock 内部：

```
x:                        (2, 4, 768)
    ↓ norm1
x:                        (2, 4, 768)
    ↓ MHA（内部维度变换见 Ch3 笔记）
x:                        (2, 4, 768)
    ↓ dropout + 残差
x:                        (2, 4, 768)
    ↓ norm2
x:                        (2, 4, 768)
    ↓ FFN (768->3072->768)
x:                        (2, 4, 768)
    ↓ dropout + 残差
x:                        (2, 4, 768)
```

---

## 易错点汇总

| 易错点 | 正确做法 | 原因 |
|--------|----------|------|
| LayerNorm 方差 | `unbiased=False` | 论文用总体方差（除以N），PyTorch 默认是无偏（除以N-1） |
| 位置编码设备 | `torch.arange(seq_len, device=in_idx.device)` | 必须和输入在同一设备，否则 GPU/CPU 报错 |
| out_head 偏置 | `bias=False` | GPT-2 风格，输出层不加偏置 |
| 生成时取最后位置 | `logits[:, -1, :]` | 模型输出每个位置都预测下一个，但生成只关心最后一个 |
| 上下文截断 | `idx[:, -context_size:]` | 超过 context_length 会导致位置编码越界 |
| 推理模式 | `model.eval()` + `torch.no_grad()` | 关闭 dropout 和梯度计算 |
| 残差连接 | `x = x + shortcut`，不是 `x += shortcut` | 避免 in-place 操作导致梯度计算问题 |

---

## 面试高频题

### Q1: GPT 模型的完整架构包含哪些组件？

**答**：Token 嵌入 + 可学习位置嵌入 → Dropout → N 个 TransformerBlock（每个包含 Pre-LN + 多头因果注意力 + 残差、Pre-LN + 前馈网络 + 残差）→ Final LayerNorm → 输出线性层（无偏置）。

### Q2: 为什么 Transformer 用 Pre-LN 而不是 Post-LN？

**答**：Pre-LN（归一化在子层之前）训练更稳定，梯度可以通过残差路径直接回传，不需要学习率 warmup。Post-LN（归一化在残差相加之后）在深层网络中容易梯度爆炸/消失，需要精心调参。现代 LLM（GPT-2、Llama、Qwen）基本都用 Pre-LN。

### Q3: GELU 和 ReLU 的区别？为什么 LLM 用 GELU？

**答**：ReLU 是 `max(0,x)`，负数区梯度为 0；GELU 是平滑的近似 ReLU，负数区有微小非零梯度。GELU 的平滑性让训练更稳定，梯度信息更丰富。现代模型常用 SwiGLU（Llama、Qwen），效果更好但计算量稍大。

### Q4: 什么是权重共享（weight tying）？GPT-2 为什么用它？

**答**：让输出层的权重矩阵和 token 嵌入层共享同一份参数。两者形状都是 `(vocab_size, emb_dim)`，语义上也相关（输入映射和输出映射是对称的）。好处是减少参数量（124M vs 163M）、正则化效果、可能提升泛化。

### Q5: 前馈网络为什么要先升维 4 倍再降维？

**答**：升维给模型更大的非线性表达空间，让模型能学习更复杂的特征变换。4 倍是 Transformer 论文的标准配置。参数量主要集中在这里（每个 FFN 约 4.7M）。

### Q6: 贪心解码有什么问题？有哪些改进方法？

**答**：贪心解码永远选最高概率 token，输出单调、容易重复。改进方法：温度缩放（控制随机性）、top-k 采样（只在概率最高的 k 个 token 中采样）、top-p / nucleus 采样（在累积概率达到 p 的最小集合中采样）。

### Q7: LayerNorm 和 BatchNorm 的区别？为什么 NLP 用 LayerNorm？

**答**：LayerNorm 沿特征维归一化每个样本，不依赖 batch 大小，适合变长序列；BatchNorm 沿 batch 维归一化同一特征，依赖 batch 大小，推理时需要保存运行统计量。NLP 中序列长度可变、batch 中序列长度不一致，BatchNorm 难以应用，所以用 LayerNorm。

---

## 代码运行验证

运行 `gpt.py` 输出（随机权重，输出文本无意义但前向传播正确）：

```
输入: "Hello, I am"
编码: [15496, 11, 314, 716]
输出长度: 14 (4 输入 + 10 生成)
输出文本: Hello, I am Featureiman Byeswickattribute argue logger Normandy Compton analogous
```

> 输出是乱码因为权重随机初始化，Ch5 训练后才有意义。

---

## 与前后章节的联系

- **Ch2**：提供了 `GPTDatasetV1`、`create_dataloader_v1`、BPE 分词器，本章 `gpt.py` 直接复用
- **Ch3**：提供了 `MultiHeadAttention`，是 TransformerBlock 的核心组件
- **Ch5**：将训练本章搭建的模型，实现更高级的采样策略，加载预训练权重
- **Ch6/7**：在训练好的模型基础上做 SFT、DPO、LoRA

---

## 参考资源

- 教材代码：`code/ch04/01_main-chapter-code/`
- GPT-2 论文：Radford et al., "Language Models are Unsupervised Multitask Learners"
- LayerNorm 论文：Ba et al., 2016, arXiv:1607.06450
- GELU 论文：Hendrycks & Gimpel, 2016, arXiv:1606.08415
- Bonus 材料：KV cache、GQA、MLA、SWA、MoE、DeltaNet 等注意力变体在 `code/ch04/` 其他子目录
