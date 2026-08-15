# LLM 学习项目

> 大语言模型系统学习计划：从原理实现到工业工程落地
> 周期：8-12 周，每日 2-3h
> 基线：已有机器学习基础

## 学习路线

**项目链路：LLMs-from-scratch → nanochat → ai-engineering-from-scratch**

| 阶段 | 内容 | 周期 | 状态 |
|------|------|------|------|
| [阶段1](./stage-01/) | LLMs-from-scratch 原理与基础实现 | 4 周 | 进行中 |
| 阶段2 | nanochat 端到端对话模型训练 | 3 周 | 未开始 |
| 阶段3 | ai-engineering-from-scratch 工业工程落地 | 3-5 周 | 未开始 |

---

## 项目架构

### 设计原则

- **每个阶段独立目录**：`stage-01/`、`stage-02/`、`stage-03/`，包含该阶段的学习记录、代码、笔记
- **第三方代码独立管理**：教材代码克隆到各阶段 `code/` 子目录，加入 `.gitignore`，不混入学习仓库
- **文档驱动**：每个阶段有 `README.md` 作为学习进度追踪表，记录目标、笔记、问题、实验结果
- **环境统一**：项目根目录 `.venv` 虚拟环境，所有阶段共用

### 目录结构

```
llm-project-with-doubao/
├── README.md                          # 项目总览（本文件）
├── LLM学习项目完整计划.md              # 完整学习计划、硬件约束、预算方案
├── .gitignore                         # 忽略 .venv/、教材代码、临时文件等
│
├── stage-01/                          # 阶段1：LLMs-from-scratch（原理与基础实现）
│   ├── README.md                      # 阶段1学习记录（周计划、笔记、实验、问题）
│   └── code/                          # 教材代码（独立仓库，已 gitignore）
│       ├── ch01/ ~ ch07/              # 各章节 Notebook
│       ├── appendix-A/ ~ appendix-E/  # 附录（LoRA 等）
│       └── pkg/                       # 可安装的 gpt-from-scratch 包
│
├── stage-02/                          # 阶段2：nanochat（端到端对话模型训练）
│   ├── README.md                      # 阶段2学习记录（待创建）
│   └── code/                          # nanochat 代码（独立仓库，待克隆）
│
├── stage-03/                          # 阶段3：ai-engineering-from-scratch（工业工程落地）
│   ├── README.md                      # 阶段3学习记录（待创建）
│   └── code/                          # 教材代码（独立仓库，待克隆）
│
└── .venv/                             # Python 虚拟环境（已 gitignore）
    └── Python 3.13.13 + torch/jupyterlab/tiktoken/...
```

### 执行环境架构

```
┌─────────────────────────┐         SSH / VSCode Remote         ┌──────────────────────┐
│  本地客户端              │ ──────────────────────────────────▶ │  远端 GPU 服务器      │
│  旧笔记本 i7-4700HQ     │                                      │  RunPod RTX 3090 24GB│
│                         │                                      │                      │
│  · VSCode + Remote-SSH  │                                      │  · PyTorch (CUDA)    │
│  · Git / GPG 签名       │                                      │  · 训练 / 微调 / 推理 │
│  · Jupyter 前端浏览     │                                      │  · tmux 保活训练     │
│  · 阶段1轻量代码本地跑  │                                      │  · 大模型 / 大数据集  │
└─────────────────────────┘                                      └──────────────────────┘
```

- **本地**：仅做编辑、浏览、轻量实验（Ch2-Ch4 可本地 CPU 跑通）
- **远端**：所有训练、微调、大模型推理任务
- **数据同步**：代码通过 Git 同步；数据集、模型权重按需 rsync

---

## Milestones

### Milestone 1：LLM 原理与基础实现闭环

| 项目 | 内容 |
|------|------|
| **对应阶段** | 阶段1（第1-4周） |
| **教材** | Sebastian Raschka《Build a Large Language Model From Scratch》 |
| **目标** | 从零理解并实现 GPT 模型核心组件，跑通预训练 / SFT / DPO / LoRA 全流程 |

**关键交付物：**
- 手写 GPT 核心模块代码（嵌入、因果多头注意力、Transformer 块、LN、残差连接）
- 小参数量 GPT 预训练实验记录
- SFT 监督微调实验记录
- DPO 偏好对齐实验记录
- LoRA 低秩适配实验记录
- 阶段1学习笔记与习题解答

**验收标准：**
- [ ] 能手写 GPT 推理代码并运行通过
- [ ] 能讲清 tokenization、上下文窗口、因果注意力原理
- [ ] 能说清预训练 / SFT / DPO / LoRA 的区别与适用场景
- [ ] 完成教材 Ch2-Ch7 全部 Notebook 与习题
- [ ] 阶段1 README.md 学习记录完整

---

### Milestone 2：端到端对话模型训练流程

| 项目 | 内容 |
|------|------|
| **对应阶段** | 阶段2（第5-7周） |
| **教材** | Karpathy nanochat |
| **目标** | 完整跑通预训练 → SFT → RL → 对话推理的端到端流程，理解工业训练工程优化 |

**关键交付物：**
- nanochat 小模型（`--depth 12`）预训练实验
- SFT 监督微调实验
- RL 强化学习对齐实验
- 交互式对话推理演示
- 轻量子集评测结果
- 分布式训练、混合精度、MFU 等概念梳理笔记

**验收标准：**
- [ ] 完整跑通预训练 → SFT → RL → 对话全流程（小模型）
- [ ] 理解 bpb 损失、梯度累积、bfloat16 混合精度
- [ ] 能对比 nanochat 与 rasbt 版本 GPT 实现差异
- [ ] 理解 speedrun.sh 中的高性能训练 trick（只读不执行）
- [ ] 阶段2 README.md 学习记录完整

---

### Milestone 3：LLM 工业工程全栈

| 项目 | 内容 |
|------|------|
| **对应阶段** | 阶段3（第8-12周） |
| **教材** | ai-engineering-from-scratch |
| **目标** | 掌握 LLM 从数据到部署的完整工程链路：数据工程 → 评估 → RAG → 部署监控 |

**关键交付物：**
- LLM 数据集构建、清洗、去重流程实践
- LLM-as-judge 自动评估方案
- 幻觉检测与事实一致性评估
- 完整 RAG 系统实现（向量库、召回、重排）
- 模型 API 服务（流式输出、实验追踪）
- 生产监控方案（延迟、吞吐量、幻觉）

**验收标准：**
- [ ] 掌握 LLM 数据处理完整流程
- [ ] 能对大模型输出做标准化评估
- [ ] 实现可用的 RAG 应用
- [ ] 能把模型封装成可对外服务的 API
- [ ] 梳理数据 → 训练 → 对齐 → 评估 → RAG → 部署全链路
- [ ] 阶段3 README.md 学习记录完整

---

### 整体验收清单

- [ ] 能手写 GPT 核心模块（嵌入、注意力、Transformer 块）
- [ ] 说清预训练 / SFT / DPO / RL 的区别与用途
- [ ] 区分教学向实现(rasbt)与高性能训练实现(nanochat)的代码差异
- [ ] 熟练掌握远程开发：VSCode Remote-SSH、端口转发、tmux 保活
- [ ] 能完成数据集处理、评估、RAG、模型部署的完整流程

---

## 环境配置

### Python 虚拟环境

```bash
# 创建（已完成）
python -m venv .venv

# 激活（Windows）
.venv\Scripts\activate

# 核心依赖
pip install torch jupyterlab tiktoken matplotlib numpy pandas tqdm psutil
```

### 教材代码克隆

```bash
# 阶段1
git clone https://github.com/rasbt/LLMs-from-scratch.git stage-01/code

# 阶段2（待开始）
git clone https://github.com/karpathy/nanochat.git stage-02/code

# 阶段3（待开始）
git clone https://github.com/rasbt/ai-engineering-from-scratch.git stage-03/code
```

### Git 配置

- 用户名：`rpan2022`
- 邮箱：GitHub noreply 邮箱
- 提交签名：GPG 已启用（`commit.gpgsign=true`）
- 默认分支：`main`

---

## 参考资源

- [LLMs-from-scratch 官方仓库](https://github.com/rasbt/LLMs-from-scratch)
- [nanochat 官方仓库](https://github.com/karpathy/nanochat)
- [ai-engineering-from-scratch](https://github.com/rasbt/ai-engineering-from-scratch)
