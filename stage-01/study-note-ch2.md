# Ch2 学习笔记：Working with Text Data（文本数据处理）

> 教材：Build a Large Language Model From Scratch, Sebastian Raschka
> 章节：Chapter 2
> 目标：完整理解从原始文本到 LLM 输入张量的全流程——分词 → Token ID → 滑动窗口采样 → 嵌入 → 位置编码

---

## 章节概述

本章覆盖**数据准备和采样**，将原始文本转换为 LLM 可消费的输入张量。

完整管道：
```
原始文本 → 分词(Tokenize) → Token ID → 滑动窗口采样(input/target) → Token Embedding → + Positional Embedding → 模型输入
```

---

## 2.1 Understanding Word Embeddings（理解词嵌入）

### 核心概念

- **离散 vs 连续**：文本本质是离散符号（字符、单词），神经网络只能处理连续数值，需要嵌入层做桥梁
- **嵌入层**：将离散的 token ID 映射为连续的低维稠密向量
- **高维空间**：LLM 使用数千维的嵌入空间，人类只能可视化 1/2/3 维
- **语义相似度**：语义相近的词在向量空间中距离更近

### 直觉

每个 token 对应一个可学习的向量，训练过程中这些向量被优化，使得相似语义的 token 在空间中靠近。

---

## 2.2 Tokenizing Text（文本分词）

### 分词定义

将文本切分为更小的单元（单词、标点符号等）。

### 示例文本

使用 Edith Wharton 的短篇小说 *The Verdict*（公有领域）作为训练语料。

### 逐步构建正则表达式分词器

从简单到复杂，逐步完善分词规则：

**步骤1：按空格分割**
```python
import re
text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
# ['Hello,', ' ', 'world.', ' ', 'This,', ' ', 'is', ' ', 'a', ' ', 'test.']
```

**步骤2：加入逗号和句号分割**
```python
result = re.split(r'([,.]|\s)', text)
# ['Hello', ',', '', ' ', 'world', '.', '', ' ', ...]
```

**步骤3：去除空字符串和空格**
```python
result = [item.strip() for item in result if item.strip()]
# ['Hello', ',', 'world', '.', 'This', ',', 'is', 'a', 'test', '.']
```

**步骤4：处理更多标点符号（最终版）**
```python
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
# ['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']
```

正则表达式 `([,.:;?_!"()\']|--|\s)` 的含义：
- `[,.:;?_!"()\']`：匹配任意单个标点符号
- `--`：匹配双连字符
- `\s`：匹配空白字符
- 括号 `()` 表示保留分隔符本身

### 应用到完整文本

```python
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))  # 总 token 数
```

---

## 2.3 Converting Tokens into Token IDs（Token 转 ID）

### 构建词表（Vocabulary）

```python
all_words = sorted(set(preprocessed))   # 去重并排序
vocab_size = len(all_words)
vocab = {token: integer for integer, token in enumerate(all_words)}
```

- 词表是 token → 整数 ID 的映射字典
- 词表大小 = 唯一 token 的数量
- 排序保证词表确定性（每次运行结果一致）

### SimpleTokenizerV1：第一个分词器类

```python
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab                              # token → ID
        self.int_to_str = {i: s for s, i in vocab.items()}  # ID → token

    def encode(self, text):
        """文本 → token ID 列表"""
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        """token ID 列表 → 文本"""
        text = " ".join([self.int_to_str[i] for i in ids])
        # 修复标点前的空格（如 " ," → ","）
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
```

### 使用示例

```python
tokenizer = SimpleTokenizerV1(vocab)
text = '"It\'s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'
ids = tokenizer.encode(text)   # [28, ..., 整数列表]
tokenizer.decode(ids)          # 还原为原始文本
```

---

## 2.4 Adding Special Context Tokens（特殊上下文 Token）

### 特殊 Token 类型

| Token | 全称 | 作用 |
|-------|------|------|
| `[BOS]` | Beginning of Sequence | 标记文本开始 |
| `[EOS]` | End of Sequence | 标记文本结束；用于拼接多个不相关文本（不同文章、书籍等） |
| `[PAD]` | Padding | batch 中对齐不同长度的序列 |
| `[UNK]` | Unknown | 表示词表外的词（OOV） |

### GPT-2 的简化设计

- **只使用 `<|endoftext|>` 一个特殊 token**，同时充当 EOS 和 PAD 的角色
- **不使用 `[UNK]`**：因为 BPE 分词器可以将任何词拆成子词，从根本上解决 OOV 问题
- 训练时使用 mask，padding 位置不参与注意力计算，所以用什么 token 做 padding 无所谓

### OOV 问题演示

```python
tokenizer = SimpleTokenizerV1(vocab)
text = "Hello, do you like tea. Is this-- a test?"
tokenizer.encode(text)
# KeyError: 'Hello'  ← "Hello" 不在词表中，报错
```

### SimpleTokenizerV2：支持特殊 Token

```python
# 扩展词表，加入特殊 token
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # 关键：未知词替换为 <|unk|>
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
```

### 用 `<|endoftext|>` 拼接多段文本

```python
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
# "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."

tokenizer.decode(tokenizer.encode(text))
```

---

## 2.5 BytePair Encoding（BPE 编码）

### 为什么需要 BPE

SimpleTokenizerV2 用 `<|unk|>` 处理 OOV，但这会丢失信息（所有未知词都变成同一个 token）。BPE 从根本上解决这个问题。

### BPE 原理

- BPE 是一种数据压缩算法，被借用到 NLP 分词中
- 初始词表：所有单个字节（256 个，覆盖所有可能的字符）
- 迭代合并：统计语料中相邻 token 对的频率，将最高频的对合并为新 token
- 终止条件：达到预设词表大小（GPT-2 为 50,257）

### BPE 如何解决 OOV

任何词都能逐层拆成子词，最终回退到单字节，**永远不会 OOV**：

```
"supercalifragilistic"
→ 词表里没有整词
→ 拆成 ["super", "cali", "fragil", "istic"]
→ 还没有就继续拆到单字节
→ 一定能表示
```

示例：`"unfamiliarword"` 可能被分为 `["unfam", "iliar", "word"]`

### tiktoken 库

OpenAI 官方 BPE 分词器，核心算法用 Rust 实现，比原始 Python 实现快约 5 倍。

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, ...]

strings = tokenizer.decode(integers)
# 还原为原始文本
```

- `allowed_special={"<|endoftext|>"}`：允许解析特殊 token，否则会报错
- `someunknownPlace` 会被 BPE 拆成多个子词，不会变成 `<unk>`

### BPE vs 简单分词的关键区别

| 特性 | SimpleTokenizer | BPE (GPT-2) |
|------|-----------------|-------------|
| 分词粒度 | 单词+标点 | 子词（可变长度） |
| OOV 处理 | 替换为 `<unk>`，丢失信息 | 拆成子词，无信息丢失 |
| 词表大小 | 由语料唯一词数决定 | 固定（50,257） |
| 空格处理 | 空格作为分隔符被去掉 | 空格包含在 token 中（如 `" effort"`） |

---

## 2.6 Data Sampling with a Sliding Window（滑动窗口数据采样）

### 训练目标：Next-Token Prediction

LLM 逐词生成，训练时每个位置都要预测下一个 token：

```
输入 x: [token_1, token_2, token_3, token_4]
目标 y: [token_2, token_3, token_4, token_5]
```

目标是输入**向右偏移一位**。

### 逐个位置预测的直观理解

```
[token_1]                → 预测 token_2
[token_1, token_2]       → 预测 token_3
[token_1, token_2, token_3] → 预测 token_4
...
```

但实际训练中，我们用固定上下文窗口一次性处理，而不是逐步增长。

### 滑动窗口方法

以固定的上下文窗口大小（context size / max_length）在 token 序列上滑动：

```
原始序列: [t1, t2, t3, t4, t5, t6, t7, t8, ...]
窗口大小: 4, 步长: 1

第1个样本: x=[t1,t2,t3,t4], y=[t2,t3,t4,t5]
第2个样本: x=[t2,t3,t4,t5], y=[t3,t4,t5,t6]
第3个样本: x=[t3,t4,t5,t6], y=[t4,t5,t6,t7]
...
```

### 步长（stride）的选择

- **stride = 1**：最大重叠，样本数最多，但相邻样本高度相似，可能过拟合
- **stride = max_length**：无重叠，样本数最少，数据利用率低
- **实际常用**：stride = max_length / 2，平衡数据利用率和多样性

### GPTDatasetV1 完整实现

```python
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 对整个文本分词
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 滑动窗口构造输入-目标对
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

### create_dataloader_v1 工厂函数

```python
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,   # 丢弃最后一个不完整的 batch
        num_workers=num_workers
    )
    return dataloader
```

### 参数说明

| 参数 | 作用 |
|------|------|
| `batch_size` | 每个 batch 的样本数 |
| `max_length` | 上下文窗口大小（每个样本的 token 数） |
| `stride` | 滑动步长 |
| `shuffle` | 是否打乱样本顺序 |
| `drop_last` | 是否丢弃最后一个不完整的 batch |
| `num_workers` | 数据加载进程数（Windows 上建议 0） |

### 测试示例

```python
# batch_size=1, max_length=4, stride=1
dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)   # (tensor([[...]]), tensor([[...]]))

# batch_size=8, max_length=4, stride=4（无重叠）
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
inputs, targets = next(data_iter)
# inputs.shape: torch.Size([8, 4])
# targets.shape: torch.Size([8, 4])
```

---

## 2.7 Creating Token Embeddings（创建 Token 嵌入）

### 嵌入层的作用

将整数 token ID 转换为连续的向量表示。嵌入层是 LLM 的一部分，**在训练过程中被更新**。

### 小例子理解

```python
import torch

input_ids = torch.tensor([2, 3, 5, 1])    # 4 个 token ID
vocab_size = 6                             # 词表大小
output_dim = 3                             # 嵌入维度

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# 单个 token 嵌入
print(embedding_layer(torch.tensor([3])))
# tensor([[ 0.3074, -0.5059, 0.4283]])  ← 权重矩阵第 4 行

# 批量嵌入
print(embedding_layer(input_ids))
# shape: [4, 3]
```

### 嵌入层的本质

嵌入层等价于 **one-hot 编码 + 全连接层矩阵乘法**，但是用查表操作实现，更高效：

```
one_hot(token_id=3): [0, 0, 0, 1, 0, 0]  (1×6)
weight matrix:                            (6×3)
矩阵乘法结果:                              (1×3) = 权重矩阵第 4 行
```

因为 one-hot 向量只有一个位置是 1，矩阵乘法等价于直接取权重矩阵的对应行，即查表。

### 关键性质

- 嵌入层是可学习的神经网络层，通过反向传播优化
- 权重矩阵形状：`[vocab_size, output_dim]`
- 输出形状：`[batch_size, seq_length, output_dim]`

---

## 2.8 Encoding Word Positions（位置编码）

### 为什么需要位置编码

**自注意力是置换不变（permutation invariant）的**：打乱 token 顺序，注意力输出不变。但语言中词序至关重要，需要显式注入位置信息。

嵌入层本身不感知位置：同一个 token ID 无论在序列的哪个位置，得到的嵌入向量都相同。

### 绝对位置编码（GPT-2 采用）

为每个位置分配一个可学习的嵌入向量，使用另一个 `nn.Embedding` 层：

```python
context_length = max_length   # 4
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)  # [4, 256]

positions = torch.arange(context_length)   # [0, 1, 2, 3]
pos_embeddings = pos_embedding_layer(positions)  # [4, 256]
```

### 最终输入 = Token 嵌入 + 位置嵌入

```python
# token_embeddings: [batch_size=8, seq_len=4, embed_dim=256]
# pos_embeddings:   [seq_len=4, embed_dim=256]
# 广播机制自动扩展 pos_embeddings 到每个 batch
input_embeddings = token_embeddings + pos_embeddings  # [8, 4, 256]
```

### 完整流程示例

```python
vocab_size = 50257
output_dim = 256
max_length = 4

# Token 嵌入层
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# 从 dataloader 获取一个 batch
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length,
                                   stride=max_length, shuffle=False)
inputs, targets = next(iter(dataloader))
# inputs.shape: [8, 4]

token_embeddings = token_embedding_layer(inputs)  # [8, 4, 256]

# 位置嵌入层
pos_embedding_layer = torch.nn.Embedding(max_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))  # [4, 256]

# 相加得到最终输入
input_embeddings = token_embeddings + pos_embeddings  # [8, 4, 256]
```

### 形状变化总结

```
原始文本
  → encode → token IDs: [N]  (N = 文本总 token 数)
  → 滑动窗口 → inputs: [batch_size, max_length]
  → token embedding → [batch_size, max_length, output_dim]
  → + positional embedding → [batch_size, max_length, output_dim]
  → 输入 Transformer
```

---

## 完整管道代码总结

```python
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

# 1. 分词器
tokenizer = tiktoken.get_encoding("gpt2")

# 2. 数据集
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i+max_length]))
            self.target_ids.append(torch.tensor(token_ids[i+1:i+max_length+1]))

    def __len__(self): return len(self.input_ids)
    def __getitem__(self, idx): return self.input_ids[idx], self.target_ids[idx]

# 3. DataLoader
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True):
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

# 4. 嵌入层
vocab_size = 50257
output_dim = 256
max_length = 4
token_embedding = torch.nn.Embedding(vocab_size, output_dim)
pos_embedding = torch.nn.Embedding(max_length, output_dim)

# 5. 前向传播
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length)
inputs, targets = next(iter(dataloader))
x = token_embedding(inputs) + pos_embedding(torch.arange(max_length))
# x.shape: [8, 4, 256] ← 这就是送入 Transformer 的输入
```

---

## Bonus 材料导读

| 目录 | 内容 | 建议 |
|------|------|------|
| `02_bonus_bytepair-encoder/` | 对比 tiktoken 与原始 Python BPE 实现性能（约 5x 差距） | 选读，理解工程优化 |
| `03_bonus_embedding-vs-matmul/` | 证明嵌入层 = one-hot + 矩阵乘法 | 推荐阅读，加深理解 |
| `04_bonus_dataloader-intuition/` | 用简单数字演示数据加载器逻辑 | 推荐阅读，初学者友好 |
| `05_bpe-from-scratch/` | 从零实现 GPT-2 BPE 分词器（含训练合并规则） | 进阶选读，深入理解 BPE |

---

## 核心概念速查表

| 概念 | 一句话解释 |
|------|-----------|
| Tokenization | 将文本切分为离散单元（子词/单词/字符） |
| BPE | 字节对编码，通过高频合并构建子词词表，从根本上解决 OOV |
| OOV | Out Of Vocabulary，词表外未登录词 |
| Token ID | 每个 token 在词表中的整数索引 |
| `<\|endoftext\|>` | GPT-2 唯一特殊 token，兼作 EOS 和 PAD |
| Context Window | 模型一次能处理的最大 token 数（max_length） |
| Sliding Window | 以固定窗口滑动构造训练样本 |
| Stride | 滑动步长，控制相邻样本的重叠程度 |
| Next-token Prediction | 每个位置预测下一个 token，目标是输入偏移一位 |
| Embedding | 离散 ID → 连续向量的可学习映射，本质是查表 |
| Positional Encoding | 为 token 注入位置信息，解决注意力的置换不变性 |
| Absolute Position Embedding | GPT-2 使用的可学习位置嵌入，每个位置一个向量 |

---

## 常见问题与注意事项

1. **为什么目标是输入偏移一位，而不是单独的标签？**
   LLM 是自回归模型，每个位置都在预测下一个 token，整个序列同时提供输入和监督信号，效率最高。

2. **为什么用子词（BPE）而不是单词？**
   单词级分词词表过大且有 OOV 问题；子词平衡了词表大小和序列长度，且通过字节级回退彻底消除 OOV。

3. **嵌入层和线性层有什么区别？**
   本质相同，都是矩阵乘法。嵌入层用索引查找代替 one-hot 乘法，更高效。

4. **位置编码为什么不能省略？**
   自注意力本身不感知顺序，没有位置编码的话，"猫追狗"和"狗追猫"对模型来说是一样的。

5. **GPT-2 为什么只用 `<|endoftext|>` 一个特殊 token？**
   简化设计：EOS 和 PAD 合并；BPE 消除了对 UNK 的需求；BOS 不是必须的，因为模型从第一个 token 开始预测即可。

6. **stride 怎么选？**
   stride=max_length 无重叠但数据利用率低；stride=1 重叠最大但可能过拟合；常用 stride=max_length/2 折中。

7. **MPS/CPU 上的注意事项**
   Ch2 全部是数据处理和小矩阵操作，CPU 完全够用，不需要 GPU。

---

## 学习检查清单

- [ ] 能解释 BPE 的基本原理和合并过程
- [ ] 能用 tiktoken 进行编码和解码，理解 allowed_special 参数
- [ ] 理解滑动窗口如何构造输入-目标对
- [ ] 能手写 GPTDatasetV1 和 create_dataloader_v1
- [ ] 理解嵌入层的本质是查表，等价于 one-hot + 矩阵乘法
- [ ] 说清为什么需要位置编码，以及绝对位置编码的实现方式
- [ ] 能跟踪完整管道的形状变化：文本 → IDs → [batch, seq_len] → [batch, seq_len, embed_dim]
- [ ] 完成 exercise-solutions.ipynb 中的习题
