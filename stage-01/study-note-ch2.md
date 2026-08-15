# Ch2 学习笔记：Working with Text Data（文本数据处理）

> 教材：Build a Large Language Model From Scratch
> 章节：Chapter 2
> 目标：理解 LLM 如何处理文本——从原始字符串到模型可消费的 token 嵌入向量

---

## 章节概览

Ch2 是整个 LLM 数据管道的入口，回答一个核心问题：**如何把一段自然语言文本，转换成 Transformer 模型能处理的数字张量？**

完整管道：
```
原始文本 → 分词(Tokenize) → Token ID → 滑动窗口采样 → 嵌入(Embedding) → 位置编码 → 模型输入
```

---

## 2.1 Understanding Word Embeddings（词嵌入）

### 核心概念

- **离散表示**：文本本质是离散的符号（字符、单词、子词），无法直接输入神经网络
- **嵌入层（Embedding Layer）**：将离散的 token ID 映射为连续的低维稠密向量
- 每个 token 对应一个可学习的向量，向量空间中的距离反映语义相似度

### 关键直觉

- 嵌入矩阵形状：`[vocab_size, embed_dim]`
- 查表操作：`embedding = weight[token_id]`，本质是一次索引查找
- 嵌入向量在训练过程中被优化，相似语义的 token 会在向量空间中靠近

---

## 2.2 Tokenizing Text（文本分词）

### 分词策略对比

| 策略 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| 字符级 | 逐字符切分 | 词表极小，无 OOV | 序列过长，语义粒度太细 |
| 单词级 | 按空格/标点切分 | 语义粒度合适 | 词表巨大，OOV 问题严重 |
| **子词级（BPE）** | 高频词保留，低频词拆为子词 | 平衡词表大小与序列长度，无 OOV | 实现较复杂 |

### GPT 系列使用子词分词

- GPT-2 使用 BPE（Byte Pair Encoding）
- 词表大小约 50,257
- 既能处理生僻词（拆成子词），又不会让序列过长

### tiktoken 库

OpenAI 官方分词库，直接使用预训练好的 BPE 词表：

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
text = "Hello, world!"
tokens = tokenizer.encode(text)       # [15496, 11, 995, 0]
decoded = tokenizer.decode(tokens)    # "Hello, world!"
```

---

## 2.3 Converting Tokens into Token IDs（Token 转 ID）

### 流程

1. 分词器将文本切分为 token 字符串
2. 通过词表（vocabulary）查找每个 token 对应的整数 ID
3. 得到一个整数序列，即模型的输入索引

### 示例

```python
text = "Every effort moves you"
# 分词 → ["Every", " effort", " moves", " you"]
# 查词表 → [6108, 3626, 6193, 345]
```

注意：GPT-2 的 BPE 分词中，空格会被包含在 token 中（如 `" effort"` 前面有空格），这是为了区分词首和词中位置。

---

## 2.4 Adding Special Context Tokens（特殊上下文 Token）

### 常用特殊 Token

| Token | 作用 |
|-------|------|
| `<|endoftext|>` | 文本结束标记，分隔不同文档/段落 |
| `<|unk|>` | 未知 token（BPE 通常不需要，因为可以逐字节回退） |
| `<|pad|>` | 填充 token，用于 batch 中对齐序列长度 |

### 为什么需要特殊 token

- **文档边界**：预训练数据是多个文本拼接而成，需要 `<|endoftext|>` 标记边界，防止模型跨文档学习不合理的上下文
- **批处理对齐**：不同长度的序列需要 padding 到相同长度才能组成 batch

### 示例

```python
# 两段文本拼接
text = "First text." + "<|endoftext|>" + "Second text."
```

---

## 2.5 BytePair Encoding（BPE 编码）

### BPE 原理

BPE 是一种数据压缩算法，被借用到 NLP 分词中：

1. **初始词表**：所有单个字符（或字节）
2. **迭代合并**：统计语料中相邻 token 对的出现频率，将最高频的对合并为一个新 token
3. **终止条件**：达到预设的词表大小（如 GPT-2 的 50,257）

### 直观理解

- 假设语料中 `"hug"` + `"ging"` 频繁共现 → 合并为 `"hugging"`
- 假设 `"ing"` 是高频后缀 → 会被合并为一个子词 token
- 生僻词如 `"supercalifragilistic"` 会被拆成多个子词

### BPE 的优势

- **无 OOV**：任何词都可以拆成子词，最终回退到单字节
- **词表可控**：通过合并次数控制词表大小
- **压缩效率高**：高频词用一个 token 表示，低频词拆成多个子词

### 什么是 OOV（Out Of Vocabulary，词表外/未登录词）

OOV 指模型词表中**没有收录**的词。

**单词级分词的 OOV 问题**：词表再大也不可能装下所有词（人名、新词、生僻词、拼写错误……），遇到 OOV 只能用统一的 `<unk>` 代替，语义信息直接丢失。

```
词表: ["我", "爱", "吃", "苹果"]
"我爱吃苹果" → 全部命中 ✅
"我爱吃榴莲" → "榴莲" 不在词表 → OOV → 变成 <unk> ❌
```

**BPE 如何解决 OOV**：BPE 词表最底层是**单个字节**（256 个，覆盖所有可能的字符），任何词都能逐层拆成子词，最终拆到字节级别，永远不会 OOV：

```
"supercalifragilistic"
→ 词表里没有整词
→ 拆成 ["super", "cali", "fragil", "istic"]
→ 还没有就继续拆到单字节
→ 一定能表示
```

这就是 BPE "无 OOV" 的根本原因——用**变长的子词切分**换来了对任意文本的覆盖能力。

### GPT-2 BPE 的特殊处理

- 基于字节（byte）而非 Unicode 字符，避免多语言字符集问题
- 使用 regex 预分割，确保标点、数字等不会被错误合并
- 详见 bonus 材料 `05_bpe-from-scratch/`

---

## 2.6 Data Sampling with a Sliding Window（滑动窗口数据采样）

### 核心问题

LLM 的训练目标是 **next-token prediction**（下一个 token 预测），需要构造输入-目标对。

### 滑动窗口方法

给定 token ID 序列，以固定的上下文窗口大小（context length）滑动采样：

```
原始序列: [x1, x2, x3, x4, x5, x6, x7, x8]
窗口大小: 4

输入 x: [x1, x2, x3, x4]  →  目标 y: [x2, x3, x4, x5]
输入 x: [x2, x3, x4, x5]  →  目标 y: [x3, x4, x5, x6]
...
```

- 输入和目标**形状相同**，目标是输入向右偏移一个位置
- 每个位置预测下一个 token

### 关键代码逻辑

```python
def create_sample(token_ids, context_size, index):
    x = token_ids[index : index + context_size]
    y = token_ids[index + 1 : index + context_size + 1]
    return x, y
```

### 步长（stride）

- 步长 = 1：每个位置都作为起点，样本最多但重叠严重
- 步长 = context_size：不重叠采样，样本数最少
- 实际中常用步长 = context_size（减少冗余）或较小步长（数据增强）

---

## 2.7 Creating Token Embeddings（创建 Token 嵌入）

### PyTorch 实现

```python
import torch.nn as nn

vocab_size = 50257
embed_dim = 768

token_embedding = nn.Embedding(vocab_size, embed_dim)

# 输入: [batch_size, context_length] 的整数张量
# 输出: [batch_size, context_length, embed_dim] 的浮点张量
input_ids = torch.tensor([[6108, 3626, 6193, 345]])
embeddings = token_embedding(input_ids)  # shape: [1, 4, 768]
```

### 嵌入层的本质

嵌入层等价于：将 token ID 转成 one-hot 向量，再与权重矩阵做矩阵乘法。

```
one_hot: [1, vocab_size]  ×  weight: [vocab_size, embed_dim]  →  [1, embed_dim]
```

但实际用索引查找（`weight[token_id]`），比 one-hot + matmul 高效得多。

> 详见 bonus：`03_bonus_embedding-vs-matmul/`

---

## 2.8 Encoding Word Positions（位置编码）

### 为什么需要位置编码

- 自注意力是**置换不变**（permutation invariant）的：打乱 token 顺序，注意力输出不变
- 但语言中词序至关重要，需要显式注入位置信息

### 绝对位置编码（GPT 采用）

为每个位置分配一个可学习的嵌入向量：

```python
context_length = 4
pos_embedding = nn.Embedding(context_length, embed_dim)

positions = torch.arange(context_length)  # [0, 1, 2, 3]
pos_emb = pos_embedding(positions)       # [4, 768]
```

### 最终输入 = Token 嵌入 + 位置嵌入

```python
x = token_embedding(input_ids) + pos_embedding(positions)
# shape: [batch_size, context_length, embed_dim]
```

### 位置编码的限制

- 绝对位置编码只能处理训练时见过的最大上下文长度
- 外推到更长序列需要特殊处理（如 RoPE、ALiBi 等，Ch4 及后续会涉及）

---

## 数据加载管道总结（dataloader.ipynb）

完整的 `Dataset` + `DataLoader` 实现：

```python
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
```

### 关键参数

- `max_length`：上下文窗口大小
- `stride`：滑动步长
- `batch_size`：批大小（DataLoader 参数）

---

## Bonus 材料导读

| 目录 | 内容 | 建议 |
|------|------|------|
| `02_bonus_bytepair-encoder/` | 不同 BPE 实现（tiktoken vs 自写）的性能对比 | 选读，理解工程优化 |
| `03_bonus_embedding-vs-matmul/` | 证明嵌入层 = one-hot + 矩阵乘法 | 推荐阅读，加深理解 |
| `04_bonus_dataloader-intuition/` | 用简单数字演示数据加载器逻辑 | 推荐阅读，初学者友好 |
| `05_bpe-from-scratch/` | 从零实现 GPT-2 BPE 分词器（含训练） | 进阶选读，深入理解 BPE |

---

## 核心概念速查表

| 概念 | 一句话解释 |
|------|-----------|
| Tokenization | 将文本切分为离散单元（子词） |
| OOV (Out Of Vocabulary) | 词表外/未登录词，BPE 通过字节级回退彻底解决 |
| BPE | 字节对编码，通过高频合并构建子词词表 |
| Token ID | 每个 token 在词表中的整数索引 |
| Context Window | 模型一次能处理的最大 token 数 |
| Sliding Window | 以固定窗口滑动构造训练样本 |
| Next-token Prediction | 每个位置预测下一个 token（自回归） |
| Embedding | 离散 ID → 连续向量的可学习映射 |
| Positional Encoding | 为 token 注入位置信息，解决注意力的置换不变性 |
| `<\|endoftext\|>` | 文本结束标记，分隔不同文档 |

---

## 常见问题与注意事项

1. **为什么目标是输入偏移一位，而不是单独的标签？**
   因为 LLM 是自回归模型，每个位置都在预测下一个 token，所以整个序列同时提供输入和监督信号。

2. **为什么用子词而不是单词？**
   单词级分词词表过大且有 OOV 问题；子词平衡了词表大小和序列长度，且无 OOV（可回退到字节）。

3. **嵌入层和线性层有什么区别？**
   本质相同，都是矩阵乘法。嵌入层是用索引查找代替 one-hot 乘法，更高效。

4. **位置编码为什么不能省略？**
   自注意力本身不感知顺序，没有位置编码的话，"猫追狗"和"狗追猫"对模型来说是一样的。

5. **MPS/CPU 上的注意事项**
   Ch2 全部是数据处理和小矩阵操作，CPU 完全够用，不需要 GPU。

---

## 学习检查清单

- [ ] 能解释 BPE 的基本原理和合并过程
- [ ] 能用 tiktoken 进行编码和解码
- [ ] 理解滑动窗口如何构造输入-目标对
- [ ] 能手写一个简单的 Dataset 类
- [ ] 理解嵌入层的本质是查表
- [ ] 说清为什么需要位置编码
- [ ] 完成 exercise-solutions.ipynb 中的习题
