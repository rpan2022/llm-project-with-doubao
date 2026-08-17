# AGENTS.md

> 本文件为 AI Agent 提供项目级操作指南，确保虚拟环境、目录结构、笔记管理等操作一致。

## 项目概述

基于 Sebastian Raschka《Build a Large Language Model From Scratch》的 LLM 从零实现学习项目。
GitHub: `rpan2022/llm-project-with-doubao`

## 虚拟环境

### 解释器路径

```
C:\Users\renbo\Desktop\llm project\.venv\Scripts\python.exe
```

- **Python 版本**：3.13.13
- **运行所有 Python 脚本必须使用此解释器**，不要用系统 Python

### 已安装关键包

| 包 | 版本 | 用途 |
|----|------|------|
| torch | 2.13.0 | 深度学习框架 |
| numpy | 2.5.2 | 数值计算 |
| matplotlib | 3.11.1 | 绘图 |
| jupyterlab | 4.6.3 | Notebook 环境 |
| ipykernel | 7.3.0 | Jupyter 内核 |

### matplotlib 后端约束

环境中**没有 Tcl/Tk GUI**，matplotlib 不能使用默认的 TkAgg 后端。所有绘图脚本必须在导入 pyplot 之前设置非交互式后端：

```python
import matplotlib
matplotlib.use('Agg')  # 必须在 import matplotlib.pyplot 之前
import matplotlib.pyplot as plt
```

否则会报 `_tkinter.TclError: Can't find a usable init.tcl`。

### 中文字体

matplotlib 中文字体配置：

```python
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
```

注意：Unicode 下标字符（如 ₁₂₃）在 Microsoft YaHei 中缺失，会显示为方块，应避免使用，改用普通数字（x1, x2, x3）。

## 项目结构

```
llm project/
├── AGENTS.md              # 本文件
├── .venv/                 # 虚拟环境（不纳入版本控制）
├── temp/                  # 临时文件目录（仅 .gitkeep 纳入版本控制）
├── stage-01/              # 第一阶段：原理与基础实现
│   ├── README.md          # ⚠️ 唯一的学习进度记录文件
│   ├── notes/             # 学习笔记（Markdown）
│   │   └── ch03_attention_mechanisms_notes.md
│   └── code/              # 教材代码（独立 Git 仓库，已 .gitignore）
│       ├── ch01/ ~ ch07/
│       ├── appendix-A/ ~ appendix-E/
│       ├── pkg/           # 可安装的 gpt-from-scratch 包
│       └── setup/         # 环境配置
```

### 目录规范

- **`stage-01/code/`** 是从 [LLMs-from-scratch 官方仓库](https://github.com/rasbt/LLMs-from-scratch) 克隆的独立 Git 仓库，不要在此目录下创建学习笔记或临时文件。
- **`stage-01/notes/`** 存放各章节学习笔记，命名格式：`chXX_主题_notes.md`。
- **临时文件**（解析脚本、中间提取文件、生成的图片等）统一放在项目根目录的 **`temp/`** 文件夹下，不得散落在项目根目录或 stage 目录中。`temp/` 下除 `.gitkeep` 外的文件均不纳入版本控制，任务完成后应及时清理。

## 学习进度管理

- **唯一进度文件**：`stage-01/README.md`
- 不要创建独立的 `learning_progress.md` 等重复进度文件
- 进度更新直接修改 README.md 对应章节
- 笔记文件在 README.md 末尾的「笔记文件索引」中登记

## 笔记规范

- 格式：结构化 Markdown，适配 Obsidian / Notion
- 内容包含：章节总览、核心概念、代码实现、易错点、面试高频题
- 技术解释偏好通俗易懂的非官方语言转述，而非堆砌原文
- 代码块保留完整可运行实现

## Git 规范

- 提交使用 GPG 签名
- 按阶段（stage-01, stage-02...）划分顶层目录
- `.venv/`、`code/`（教材子仓库）、临时文件需加入 `.gitignore`

## 运行 Python 脚本的标准命令

```powershell
& "C:\Users\renbo\Desktop\llm project\.venv\Scripts\python.exe" "<脚本路径>"
```

不要使用 `python <脚本>`（会调用系统 Python，缺少依赖）。
