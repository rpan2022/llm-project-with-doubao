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
| 第2周 | Ch4 | 完整 GPT 模型搭建（嵌入、位置编码、Transformer 块、LN、残差） | ☑ |
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
- [x] 完整 GPT 前向传播
- [x] 拆解：嵌入、位置编码、Transformer 块、LN、残差连接
- [x] 从零搭建可推理的基础 GPT 模型

> ✅ 已学完（2026-08-18）

### 学习笔记

详细笔记见：[`notes/ch04_implementing_gpt_model_notes.md`](notes/ch04_implementing_gpt_model_notes.md)

#### 核心概念
- GPT-2 124M 配置：vocab=50257, ctx=1024, emb=768, 12头, 12层
- LayerNorm（Pre-LN）：沿特征维归一化，scale/shift 可训练参数
- GELU 激活：平滑版 ReLU，负数区有非零梯度
- 前馈网络：768→3072→768（4倍扩展）
- 残差连接：`x = x + sublayer(LN(x))`，缓解梯度消失
- TransformerBlock：注意力子层 + 前馈子层，各带残差
- 可学习位置编码（非正弦）
- 权重共享（weight tying）：124M vs 163M 的区别
- 贪心解码：逐 token 自回归生成

#### 代码实现
- `LayerNorm`、`GELU`、`FeedForward`、`TransformerBlock`、`GPTModel`
- `generate_text_simple`：贪心解码文本生成
- 完整可运行脚本：`code/ch04/01_main-chapter-code/gpt.py`

#### 模型架构记录
```
token ids → tok_emb + pos_emb → dropout → TransformerBlock×12 → final_norm → out_head → logits
```
- 参数量：163M（无权重共享）/ 124M（有权重共享）
- 单卡内存：约 652MB（float32 权重），训练时 ~2-3GB
- 运行验证：输入 "Hello, I am" → 输出 14 个 token（随机权重下为乱码，前向传播正确）

### 遇到的问题与解决
- 无（代码直接运行通过，输出与 tests.py 预期一致）

---

## 第3周：Ch5 预训练与文本生成

### 学习目标
- [x] 训练小参数量 GPT（代码已整理，待实际运行）
- [x] 文本采样生成（greedy + temperature + top-k）
- [ ] 加载轻量 Qwen/Llama 小版本做推理
- [x] 跑通预训练循环，理解训练/推理区别

> 📝 代码与核心概念已整理完成（2026-08-20），实际训练运行待执行

### 学习笔记

详细笔记见：[`notes/ch05_pretraining_on_unlabeled_data_notes.md`](notes/ch05_pretraining_on_unlabeled_data_notes.md)

#### 核心概念
- **预训练目标**：下一个 token 预测的交叉熵损失（自监督学习，无需人工标注）
- **训练循环四步法**：`zero_grad()` → `forward()` → `backward()` → `step()`
- **损失计算**：`F.cross_entropy(logits.flatten(0,1), target.flatten())`，把 batch×seq 合并成 N
- **模型评估**：`model.eval()` + `torch.no_grad()`，定期计算 train/val loss
- **训练/验证集划分**：90:10 字符级切分，训练集 shuffle+drop_last，验证集不 shuffle
- **AdamW 优化器**：学习率 5e-4，权重衰减 0.1
- **模型保存/加载**：`torch.save(model.state_dict())` / `load_state_dict(weights_only=True)`
- **GPT-2 权重下载**：从 OpenAI Azure Blob 下载 TF checkpoint，支持 backup URL
- **TF→PyTorch 权重映射**：变量名解析 + 转置（TF 是 in×out，PyTorch 是 out×in）+ QKV split
- **Weight Tying**：输出层与 token embedding 共享权重，124M vs 163M 的区别
- **高级采样策略**：
  - top-k：只保留概率最高的 k 个候选，过滤离谱选项
  - temperature：缩放 logits，<1 更尖锐保守，>1 更平坦多样
  - 两者可叠加使用：先 top-k 过滤，再 temperature 缩放，最后 softmax+采样
- **数值稳定性**：softmax 前减去最大值防止 exp 溢出

#### 关键文件
- `gpt_train.py`：独立训练脚本（本章训练代码总结版）
- `gpt_generate.py`：独立生成脚本（加载 GPT-2 权重 + 高级采样）
- `gpt_download.py`：GPT-2 权重下载工具（带 backup URL）
- `previous_chapters.py`：前 4 章代码汇总（GPTModel, MHA, DataLoader 等）

#### 训练记录

| 实验 | 模型规模 | 数据集 | 训练时长 | Loss | 生成效果 |
|------|----------|--------|----------|------|----------|
| 待运行 | 124M (ctx=256) | the-verdict.txt (~20k tokens) | - | - | - |

### 遇到的问题与解决
- 无（代码整理阶段，实际运行待执行）

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
| Ch02 文本数据 | [`notes/ch02_working_with_text_data.md`](notes/ch02_working_with_text_data.md) |
| Ch03 注意力机制 | [`notes/ch03_attention_mechanisms_notes.md`](notes/ch03_attention_mechanisms_notes.md) |
| Ch04 实现 GPT 模型 | [`notes/ch04_implementing_gpt_model_notes.md`](notes/ch04_implementing_gpt_model_notes.md) |
| Ch05 预训练与文本生成 | [`notes/ch05_pretraining_on_unlabeled_data_notes.md`](notes/ch05_pretraining_on_unlabeled_data_notes.md) |
