# LLM 学习项目

> 基于 MacBook Pro M5 Pro 24GB（后调整为旧笔记本 + 远程 GPU 云服务器）的大语言模型系统学习计划
> 周期：8-12 周，每日 2-3h
> 基线：已有机器学习基础

## 学习路线

| 阶段 | 内容 | 周期 | 状态 |
|------|------|------|------|
| [阶段1](./stage-01/) | LLMs-from-scratch 原理与基础实现 | 4 周 | 进行中 |
| 阶段2 | nanochat 端到端对话模型训练 | 3 周 | 未开始 |
| 阶段3 | ai-engineering-from-scratch 工业工程落地 | 3-5 周 | 未开始 |

### 项目链路
**LLMs-from-scratch → nanochat → ai-engineering-from-scratch**

## 执行环境

- **本地客户端**：旧笔记本（i7-4700HQ），仅作 SSH 终端、VSCode Remote-SSH、代码编辑
- **远端计算**：公有云 GPU（RunPod 社区云，RTX 3090 24GB），承担全部训练/推理任务
- **硬件约束**：无 CUDA 本地环境；训练只跑小模型，微调必须 LoRA/QLoRA；nanochat 仅用 `--depth 12`

## 目录结构

```
llm-project-with-doubao/
├── README.md                          # 项目总览（本文件）
├── LLM学习项目完整计划.md              # 完整学习计划与预算方案
├── .gitignore
├── stage-01/                          # 阶段1：LLMs-from-scratch
│   ├── README.md                      # 阶段1学习记录
│   └── code/                          # 教材代码（独立仓库，已 gitignore）
│       ├── ch01/ ~ ch07/
│       ├── appendix-A/ ~ appendix-E/
│       └── pkg/
├── stage-02/                          # 阶段2：nanochat（待创建）
└── stage-03/                          # 阶段3：ai-engineering-from-scratch（待创建）
```

## Git 配置

- 用户名：`rpan2022`
- 邮箱：GitHub noreply 邮箱
- 提交签名：GPG 已启用（`commit.gpgsign=true`）
- 默认分支：`main`

## 快速开始

```bash
# 克隆本仓库
git clone https://github.com/rpan2022/llm-project-with-doubao.git
cd llm-project-with-doubao

# 拉取阶段1教材代码
git clone https://github.com/rasbt/LLMs-from-scratch.git stage-01/code

# 进入阶段1学习
cd stage-01
```

## 整体验收清单

- [ ] 能手写 GPT 核心模块（嵌入、注意力、Transformer 块）
- [ ] 说清预训练 / SFT / DPO / RL 的区别与用途
- [ ] 区分教学向实现(rasbt)与高性能训练实现(nanochat)的代码差异
- [ ] 熟练掌握远程开发：VSCode Remote-SSH、端口转发、tmux 保活
- [ ] 能完成数据集处理、评估、RAG、模型部署的完整流程

## 参考资源

- [LLMs-from-scratch 官方仓库](https://github.com/rasbt/LLMs-from-scratch)
- [nanochat 官方仓库](https://github.com/karpathy/nanochat)
- [ai-engineering-from-scratch](https://github.com/rasbt/ai-engineering-from-scratch)
