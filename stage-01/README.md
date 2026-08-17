# 阶段1：LLMs-from-scratch 原理与基础实现

> 教材：Sebastian Raschka《Build a Large Language Model From Scratch》
> 周期：4 周
> 目标：从零理解并实现 GPT 模型核心组件，跑通预训练 / SFT / DPO / LoRA 全流程

## 教材代码

教材代码位于 `./code/` 目录（即 `stage-01/code/`），是从 [LLMs-from-scratch 官方仓库](https://github.com/rasbt/LLMs-from-scratch) 克隆的独立 Git 仓库，已加入 `.gitignore`。

```
stage-01/
├── README.md          # 本文件（学习记录）
└── code/              # 教材代码
    ├── ch01/ ~ ch07/  # 各章节 Notebook
    ├── appendix-A/ ~ appendix-E/
    ├── pkg/           # 可安装的 gpt-from-scratch 包
    └── setup/         # 环境配置
```

更新教材代码：
```bash
cd stage-01/code
git pull
```

---

## 学习路线总览

| 周次 | 章节 | 核心内容 | 完成 |
|------|------|----------|------|
| 第1周 | Ch2 | 文本数据、BPE 分词、数据加载器 | ☑ |
| 第1-2周 | Ch3 | 多头自注意力、因果掩码 | ☑ 核心课程完成，习题留作下周复习 |
| 第2周 | Ch4 | 完整 GPT 模型搭建（嵌入、位置编码、Transformer 块、LN、残差） | ☐ |
| 第3周 | Ch5 | 预训练与文本生成 | ☐ |
| 第3-4周 | Ch6 + Ch7 + 附录E | SFT 监督微调、DPO 偏好对齐、LoRA | ☐ |
| 第4周 | 复习验收 | 复盘核心代码、完成习题、手写 GPT 推理 | ☐ |

---

## 第1周：Ch2 文本数据、BPE 分词、数据加载器

### 学习目标
- [x] 运行章节 Notebook
- [x] 理解 token、上下文窗口、padding
- [x] 完成习题练习
- [x] 看懂 BPE 分词流程，能构造 LLM 输入数据集

> ✅ 已学完（2026-08-15）

### 学习笔记

#### 核心概念

#### 代码实现

#### 习题记录

### 遇到的问题与解决

---

## 第1-2周：Ch3 多头自注意力

### 学习目标
- [x] 手写因果多头注意力
- [x] 对比普通注意力 vs GPT 因果掩码
- [ ] 完成习题练习（📅 下周复习）
- [x] 清楚因果注意力原理，能独立写出注意力模块

> ✅ 核心课程完成（2026-08-17），课后习题留作下周复习

### 学习笔记

详细笔记见：[`notes/ch03_attention_mechanisms_notes.md`](notes/ch03_attention_mechanisms_notes.md)

#### 核心概念
- 注意力分数（Attention Scores）vs 注意力权重（Attention Weights）
- Q/K/V 三投影矩阵的作用
- Scaled Dot-Product（÷√d_k）防止梯度消失
- 因果掩码：softmax 前填 -∞（标准做法）vs softmax 后乘 0（不推荐）
- register_buffer vs nn.Parameter 的区别
- 多头注意力：多个头并行捕捉不同关注模式

#### 代码实现
- `SelfAttention_v1`（手写 Parameter）→ `SelfAttention_v2`（nn.Linear 简化）
- `CausalAttention`：支持 batch + 因果掩码 + dropout
- `MultiHeadAttentionWrapper`：直观版，堆叠多个单头
- `MultiHeadAttention`：标准高效版，view 切分 + transpose 并行 + 合并

#### 维度变换全景图（MHA forward）
```
输入 x:                  (b, num_tokens, d_in)
    ↓ W_query/W_key/W_value 投影
Q, K, V:                 (b, num_tokens, d_out)
    ↓ view 切分
                         (b, num_tokens, num_heads, head_dim)
    ↓ transpose(1,2)
                         (b, num_heads, num_tokens, head_dim)
    ↓ Q @ K^T
attn_scores:             (b, num_heads, num_tokens, num_tokens)
    ↓ mask + 缩放 + softmax + dropout
attn_weights:            (b, num_heads, num_tokens, num_tokens)
    ↓ @ V
context_vec:             (b, num_heads, num_tokens, head_dim)
    ↓ transpose(1,2) + contiguous + view 合并
                         (b, num_tokens, d_out)
    ↓ out_proj
输出:                    (b, num_tokens, d_out)
```

#### 习题记录
- 📅 待完成（下周复习）

### 难点标记
- ⚠️ MultiHeadAttention 的 `view → transpose → @ → transpose → contiguous → view` 维度变换链，需反复巩固

### 下周复习计划（2026-08-18 ~ 2026-08-24）
| 任务 | 预计耗时 | 状态 |
|------|---------|------|
| 重读 ch03 笔记，重点看 MHA 维度变换 | 30min | ☐ |
| 不看代码手写 MultiHeadAttention 类 | 45min | ☐ |
| 完成课后习题 exercise-solutions.ipynb | 1h | ☐ |
| 跑通 ch03.ipynb 全部代码 | 1h | ☐ |
| 阅读 Bonus 材料（高效MHA实现 + PyTorch Buffers） | 1h | ☐ |

### 遇到的问题与解决

---

## 第2周：Ch4 完整 GPT 模型搭建

### 学习目标
- [ ] 完整 GPT 前向传播
- [ ] 拆解：嵌入、位置编码、Transformer 块、LN、残差连接
- [ ] 从零搭建可推理的基础 GPT 模型

### 学习笔记

#### 核心概念

#### 代码实现

#### 模型架构记录

### 遇到的问题与解决

---

## 第3周：Ch5 预训练与文本生成

### 学习目标
- [ ] 训练小参数量 GPT
- [ ] 文本采样生成
- [ ] 加载轻量 Qwen/Llama 小版本做推理
- [ ] 跑通预训练循环，理解训练/推理区别

### 学习笔记

#### 核心概念

#### 训练记录

| 实验 | 模型规模 | 数据集 | 训练时长 | Loss | 生成效果 |
|------|----------|--------|----------|------|----------|
|      |          |        |          |      |          |

### 遇到的问题与解决

---

## 第3-4周：Ch6 SFT + Ch7 DPO + 附录E LoRA

### 学习目标
- [ ] 理解 SFT 监督微调原理，完成 notebook 实验
- [ ] 理解 DPO 偏好对齐原理，完成 notebook 实验
- [ ] 理解 LoRA 低秩适配原理，完成 notebook 实验
- [ ] 对比三种方法的适用场景

### 学习笔记

#### SFT 监督微调

#### DPO 偏好对齐

#### LoRA 低秩适配

### 实验记录

| 实验 | 方法 | 基座模型 | 数据集 | 训练时长 | 效果 |
|------|------|----------|--------|----------|------|
|      |      |          |        |          |      |

### 遇到的问题与解决

---

## 第4周：复习 + 验收自测

### 验收清单
- [ ] 能手写 GPT 核心模块（嵌入、注意力、Transformer 块）
- [ ] 能讲清预训练 / SFT / DPO 的作用与区别
- [ ] 完成书本全部习题
- [ ] 手写 GPT 推理代码并运行通过

### 复盘总结

#### 核心知识点梳理

#### 薄弱环节

#### 后续改进方向

---

## 资源链接

- 教材官方仓库：https://github.com/rasbt/LLMs-from-scratch
- 章节 Notebook：`code/ch01/` ~ `code/ch07/`
- 参考资料：

## 笔记文件索引

| 章节 | 笔记路径 |
|------|---------|
| Ch03 注意力机制 | [`notes/ch03_attention_mechanisms_notes.md`](notes/ch03_attention_mechanisms_notes.md) |
