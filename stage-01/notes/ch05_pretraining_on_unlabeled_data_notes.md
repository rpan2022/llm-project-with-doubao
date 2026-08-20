# Chapter 5: Pretraining on Unlabeled Data（在无标签数据上预训练）

> 教材：Sebastian Raschka《Build a Large Language Model From Scratch》
> 代码目录：`stage-01/code/ch05/01_main-chapter-code/`

---

## 一、章节总览

第 5 章是整个学习路径的**第一个实战里程碑**：把前 4 章搭好的 GPT 模型真正"喂数据、跑起来"。

本章核心做两件事：

1. **从零预训练一个小 GPT**：用一篇短篇小说（《The Verdict》，约 2 万 token）作为训练语料，跑通完整的训练循环——前向传播、算 loss、反向传播、参数更新、验证集评估、生成样本观察。
2. **加载 OpenAI 官方 GPT-2 预训练权重**：把官方 124M 参数的 GPT-2 权重下载下来，手动映射到我们自己写的 `GPTModel` 里，然后用更高级的采样策略（temperature + top-k）生成文本。

学完本章你应该能回答：
- 大模型预训练到底在优化什么目标函数？
- 训练集和验证集怎么切？为什么 stride 要等于 context_length？
- 一个标准的训练循环长什么样？
- 为什么生成文本不能永远用 greedy（argmax）？temperature 和 top-k 分别解决什么问题？
- OpenAI 的 GPT-2 权重是 TensorFlow 格式的，怎么"翻译"成 PyTorch 参数？

---

## 二、核心概念

### 2.1 预训练（Pretraining）是什么

**一句话版**：给模型看海量无标签文本，让它学会"根据前面的词预测下一个词"。

更正式地说，预训练阶段优化的是**下一个 token 预测的交叉熵损失**：

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P(x_i \mid x_{<i})
$$

其中 $x_i$ 是第 $i$ 个 token，$x_{<i}$ 是它前面的所有 token。模型的任务就是：给定上下文，让正确的下一个 token 的概率尽可能大。

**关键点**：
- 不需要人工标注——文本本身就是标签（target = input 右移一位）
- 这是一个**自监督学习**（self-supervised learning）任务
- 预训练好的模型学到的是通用语言能力，后续可以通过微调（fine-tuning）适配具体任务

### 2.2 输入与目标的构造

在第 2 章的 `GPTDatasetV1` 中已经实现了滑动窗口切分：

```
原始 token 序列: [t0, t1, t2, t3, t4, t5, ...]
input_chunk:    [t0, t1, t2, t3]   (max_length=4)
target_chunk:   [t1, t2, t3, t4]   (整体右移一位)
```

**为什么 target 是 input 右移一位？**
因为模型在位置 $i$ 输出的 logits，预测的是位置 $i+1$ 的 token。所以 input[t0,t1,t2,t3] 对应的监督信号就是 [t1,t2,t3,t4]。

### 2.3 训练集 / 验证集划分

本章用 90:10 的比例切分：

```python
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_text = text_data[:split_idx]
val_text = text_data[split_idx:]
```

**注意**：切分是在**原始文本字符级别**做的，不是在 token 级别。这没问题，因为后面 dataloader 会各自 tokenize。

**训练集和验证集的 dataloader 配置差异**：

| 配置项 | 训练集 | 验证集 |
|--------|--------|--------|
| `shuffle` | `True` | `False` |
| `drop_last` | `True` | `False` |
| `stride` | `context_length` | `context_length` |

- 训练集 `shuffle=True`：打乱样本顺序，防止模型学到顺序伪规律
- 训练集 `drop_last=True`：最后一个不完整的 batch 丢掉，避免 batch norm 等问题（虽然我们没用 BN，但保持一致性）
- 验证集 `shuffle=False, drop_last=False`：评估要确定性，且所有样本都要算到

**关于 stride 的选择**：
- 本章训练时 `stride = context_length`（无重叠），最大化数据利用率
- 如果 `stride < context_length`（有重叠），相当于做了数据增强，但会让相邻样本高度相关，可能导致过拟合
- 教材在第 2 章用 `stride = max_length // 2`（50% 重叠）是为了演示，实际预训练用无重叠更合理

### 2.4 损失函数计算

#### 单 batch 损失：`calc_loss_batch`

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss
```

**形状变换解析**：
- `input_batch` 形状：`(batch_size, context_length)`
- `logits` 形状：`(batch_size, context_length, vocab_size)`
- `logits.flatten(0, 1)` 形状：`(batch_size * context_length, vocab_size)`
- `target_batch.flatten()` 形状：`(batch_size * context_length,)`

**为什么要 flatten？**
`F.cross_entropy` 要求输入是 `(N, C)` 的二维张量（N 个样本，C 个类别），所以要把 batch 和 sequence 两个维度合并成一个维度。每个位置的预测都是一个独立的分类问题。

#### 整个 dataloader 的平均损失：`calc_loss_loader`

```python
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
```

**`num_batches` 参数的意义**：
验证集评估时，如果数据集很大，遍历所有 batch 太慢。可以只采样前 `num_batches` 个 batch 来估算平均损失，这是一种**蒙特卡洛估计**。

### 2.5 模型评估：`evaluate_model`

```python
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
```

**两个关键操作**：
1. `model.eval()`：切换到评估模式，关闭 dropout（推理时不需要随机丢弃）
2. `torch.no_grad()`：关闭梯度计算，节省内存和计算（评估不需要反向传播）

评估完后记得 `model.train()` 切回训练模式。

### 2.6 训练循环：`train_model_simple`

这是本章最核心的函数，完整实现了一个训练循环：

```python
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()           # 1. 清零梯度
            loss = calc_loss_batch(...)     # 2. 前向传播 + 算 loss
            loss.backward()                 # 3. 反向传播（算梯度）
            optimizer.step()                # 4. 更新参数
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:   # 定期评估
                train_loss, val_loss = evaluate_model(...)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): ...")

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen
```

**训练循环四步法**（每个 batch 都执行）：
1. `optimizer.zero_grad()`：清零上一步累积的梯度（PyTorch 默认会累积梯度，不清零会出错）
2. 前向传播 + 算 loss：`loss = calc_loss_batch(...)`
3. `loss.backward()`：反向传播，自动计算所有参数的梯度（`param.grad`）
4. `optimizer.step()`：优化器根据梯度更新参数（`param -= lr * param.grad`，加上 AdamW 的动量等）

**`global_step` vs `epoch`**：
- `epoch`：遍历完一遍整个训练集叫一个 epoch
- `global_step`：处理完一个 batch 叫一个 step
- 一个 epoch 包含 `len(train_loader)` 个 step
- 用 `global_step` 来控制评估频率比用 epoch 更精细

**`tokens_seen` 追踪**：
记录模型已经看过多少个 token，这在大模型训练中是一个重要指标（类似"训练算力"的衡量）。`input_batch.numel()` = `batch_size * context_length`。

### 2.7 训练中生成样本：`generate_and_print_sample`

```python
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))
    model.train()
```

每个 epoch 结束后，用固定的起始文本（`"Every effort moves you"`）生成 50 个 token，直观观察模型训练进度。

**为什么用 `generate_text_simple`（greedy decoding）？**
因为这是训练过程中的监控，只需要看个大概趋势，不需要多样性。greedy 解码确定性高，方便对比不同 epoch 的输出变化。

### 2.8 损失可视化：`plot_losses`

```python
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    ax2 = ax1.twiny()  # 第二个 x 轴，共享 y 轴
    ax2.plot(tokens_seen, train_losses, alpha=0)  # 不可见的图，仅用于对齐刻度
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
```

**双 x 轴设计**：
- 底部 x 轴：Epochs（训练轮次）
- 顶部 x 轴：Tokens seen（模型看过的 token 数）

两个轴表示同一个训练进度的不同度量方式，`twiny()` 创建共享 y 轴的第二个 x 轴。`alpha=0` 的不可见图只是为了让顶部轴的刻度范围和底部对齐。

### 2.9 模型保存与加载

```python
# 保存
torch.save(model.state_dict(), "model.pth")

# 加载
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", weights_only=True))
```

**`state_dict` 是什么？**
是一个有序字典，key 是参数名（如 `tok_emb.weight`），value 是参数张量。只保存模型的**可学习参数**，不保存模型结构本身（所以加载时要先实例化同结构的模型）。

**`weights_only=True` 的意义**：
PyTorch 2.6+ 推荐使用，反序列化时只加载张量，不执行任意 Python 对象，防止 `pickle` 反序列化漏洞。

### 2.10 下载 OpenAI GPT-2 预训练权重

#### 下载函数：`download_and_load_gpt2`

```python
def download_and_load_gpt2(model_size, models_dir):
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params
```

**GPT-2 四个规格**：

| 模型 | 参数 | emb_dim | n_layers | n_heads |
|------|------|---------|----------|---------|
| gpt2-small | 124M | 768 | 12 | 12 |
| gpt2-medium | 355M | 1024 | 24 | 16 |
| gpt2-large | 774M | 1280 | 36 | 20 |
| gpt2-xl | 1558M | 1600 | 48 | 25 |

**下载的文件说明**：
- `checkpoint`：TensorFlow 检查点索引文件
- `encoder.json`：BPE tokenizer 的编码器（token → id 映射）
- `hparams.json`：模型超参数配置
- `model.ckpt.data-00000-of-00001`：实际的权重数据（最大的文件）
- `model.ckpt.index`：权重索引
- `model.ckpt.meta`：计算图结构
- `vocab.bpe`：BPE 合并规则

#### 带进度条的下载：`download_file`

```python
def download_file(url, destination):
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    file_size = int(response.headers.get("Content-Length", 0))

    # 已存在且大小一致则跳过
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size and file_size == file_size_local:
            print(f"File already exists and is up-to-date: {destination}")
            return

    block_size = 1024  # 1 KB
    with tqdm(total=file_size, unit="iB", unit_scale=True) as progress_bar:
        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))
```

**关键点**：
- `stream=True`：流式下载，不一次性把整个文件加载到内存
- `tqdm` 进度条：实时显示下载进度
- 断点续传（简化版）：如果文件已存在且大小一致，跳过下载

### 2.11 TensorFlow 检查点 → PyTorch 参数字典

#### 解析 TF checkpoint：`load_gpt2_params_from_tf_ckpt`

```python
def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        variable_name_parts = name.split("/")[1:]  # 跳过 'model/' 前缀

        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params
```

**TF 变量名 → 嵌套字典的映射逻辑**：

TF 的变量名类似：
```
model/wte
model/wpe
model/h0/attn/c_attn/w
model/h0/attn/c_attn/b
model/h0/attn/c_proj/w
model/h0/ln_1/g
model/h0/ln_1/b
model/h0/mlp/c_fc/w
...
model/ln_f/g
model/ln_f/b
```

解析后变成：
```python
params = {
    "wte": np.array(...),      # token embedding
    "wpe": np.array(...),      # position embedding
    "blocks": [
        {
            "attn": {
                "c_attn": {"w": ..., "b": ...},   # QKV 合并的权重
                "c_proj": {"w": ..., "b": ...},   # 输出投影
            },
            "ln_1": {"g": ..., "b": ...},          # 第一个 LayerNorm
            "mlp": {
                "c_fc": {"w": ..., "b": ...},      # FFN 第一层
                "c_proj": {"w": ..., "b": ...},    # FFN 第二层
            },
            "ln_2": {"g": ..., "b": ...},          # 第二个 LayerNorm
        },
        ...  # 共 n_layer 个
    ],
    "g": ...,   # 最终 LayerNorm 的 scale
    "b": ...,   # 最终 LayerNorm 的 shift
}
```

**注意 TF 和 PyTorch 的命名差异**：
- TF 用 `h0, h1, ...` 表示层，PyTorch 用 `trf_blocks[0], trf_blocks[1], ...`
- TF 用 `ln_1/g`（gamma），PyTorch 用 `norm1/scale`
- TF 用 `ln_1/b`（beta），PyTorch 用 `norm1/shift`
- TF 用 `mlp`，PyTorch 用 `ff`
- TF 用 `ln_f`（final norm），PyTorch 用 `final_norm`

### 2.12 权重加载到自定义 GPT：`load_weights_into_gpt`

```python
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    # 1. Embedding 层
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        # 2. Attention 层：QKV 是合并存储的，需要 split
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(..., q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(..., k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(..., v_b)

        # 输出投影
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            ..., params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            ..., params["blocks"][b]["attn"]["c_proj"]["b"])

        # 3. FFN 层
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            ..., params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            ..., params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            ..., params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            ..., params["blocks"][b]["mlp"]["c_proj"]["b"])

        # 4. LayerNorm
        gpt.trf_blocks[b].norm1.scale = assign(..., params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(..., params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(..., params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(..., params["blocks"][b]["ln_2"]["b"])

    # 5. 最终 LayerNorm
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])

    # 6. 输出头：权重与 token embedding 共享（weight tying）
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
```

**几个关键细节**：

#### (1) 转置（.T）的原因

TensorFlow 的 `Linear` 层权重形状是 `(in_features, out_features)`，而 PyTorch 的 `nn.Linear` 权重形状是 `(out_features, in_features)`。所以加载时需要转置。

验证：
- TF `c_attn/w` 形状：`(emb_dim, 3 * emb_dim)`（QKV 合并）
- split 后 `q_w` 形状：`(emb_dim, emb_dim)`
- PyTorch `W_query.weight` 形状：`(emb_dim, emb_dim)`
- 所以需要 `q_w.T`，形状变为 `(emb_dim, emb_dim)`——刚好匹配

#### (2) QKV 合并存储

GPT-2 的实现中，Q、K、V 三个投影矩阵是合并成一个大矩阵 `c_attn` 存储的，形状为 `(emb_dim, 3 * emb_dim)`。加载时需要用 `np.split(..., 3, axis=-1)` 切成三份。

#### (3) Weight Tying（权重共享）

```python
gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
```

GPT-2 的输出层（`out_head`）和 token embedding 层（`tok_emb`）**共享同一套权重**。这是一种常见的优化技巧：
- 减少参数量（emb_dim × vocab_size 的矩阵只存一份）
- 直觉上合理：embedding 是"词 → 向量"，输出头是"向量 → 词"，两者是逆操作，共享权重有语义一致性

注意：我们的 `GPTModel` 中 `out_head` 是 `nn.Linear(..., bias=False)`，而 `tok_emb` 是 `nn.Embedding`。虽然都叫 weight，但它们是两个独立的 `nn.Parameter`。加载时把同一个 `wte` 数组分别赋给两者，就实现了权重共享（值相同，但内存中是两份拷贝——严格来说这不是运行时共享，只是初始化时相同。如果要真正运行时共享，需要 `gpt.out_head.weight = gpt.tok_emb.weight`）。

#### (4) `assign` 函数的作用

```python
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))
```

- 先做形状校验，不匹配直接报错（防止静默加载错误）
- 把 numpy 数组转成 `torch.nn.Parameter`（可学习参数）
- 直接赋值给模型的属性（如 `gpt.pos_emb.weight = ...`）

### 2.13 高级文本生成：带 temperature 和 top-k 采样

第 4 章的 `generate_text_simple` 用的是 greedy decoding（永远选概率最大的 token）。本章引入了更灵活的生成策略：

```python
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # --- top-k 采样 ---
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]  # 第 k 大的 logit 值
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits
            )

        # --- temperature 缩放 ---
        if temperature > 0.0:
            logits = logits / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values  # 数值稳定性
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # 按概率采样
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # greedy

        if idx_next == eos_id:  # 遇到结束符提前停止
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx
```

#### top-k 采样

**问题**：greedy 解码只选概率最高的 token，生成的文本容易重复、单调。

**top-k 的思路**：只保留概率最高的 k 个候选 token，把其余 token 的概率置为 0（logits 置为 -inf），然后在这 k 个里面按概率采样。

**示例**（vocab_size=5, top_k=2）：
```
原始 logits: [2.0, 1.5, 0.8, 0.3, -1.0]
top-2 的值:  [2.0, 1.5]，min_val = 1.5
过滤后:      [2.0, 1.5, -inf, -inf, -inf]
softmax 后:  [0.62, 0.38, 0, 0, 0]
```
这样有 62% 概率选第一个，38% 概率选第二个，不会选后面的。

**top_k 常见取值**：10, 50, 100。k 越小越保守，k 越大越多样。

#### temperature 缩放

**问题**：即使做了 top-k，概率分布可能还是过于集中（比如第一个 token 概率 99%），采样结果和 greedy 差不多。

**temperature 的思路**：在 softmax 之前把 logits 除以 temperature：
- `temperature < 1`：logits 被放大，概率分布更尖锐（更保守、更确定）
- `temperature = 1`：不改变原始分布
- `temperature > 1`：logits 被缩小，概率分布更平坦（更随机、更多样）

**示例**（logits=[2.0, 1.0]）：
```
temperature=0.5: logits/0.5 = [4.0, 2.0] → softmax ≈ [0.88, 0.12]  (更尖锐)
temperature=1.0: logits/1.0 = [2.0, 1.0] → softmax ≈ [0.73, 0.27]  (原始)
temperature=2.0: logits/2.0 = [1.0, 0.5] → softmax ≈ [0.62, 0.38]  (更平坦)
```

**temperature 常见取值**：0.7（保守但有一定多样性）、1.0（标准）、1.5（更有创意）。

#### 数值稳定性技巧

```python
logits = logits - logits.max(dim=-1, keepdim=True).values
```

在 softmax 之前减去最大值，防止 `exp(logits)` 溢出（当 logits 很大时，exp 会变成 inf）。这是 softmax 的标准数值稳定技巧，因为：
$$\text{softmax}(x_i - c) = \frac{e^{x_i - c}}{\sum_j e^{x_j - c}} = \frac{e^{x_i}/e^c}{\sum_j e^{x_j}/e^c} = \text{softmax}(x_i)$$
减去常数不改变 softmax 结果。

#### eos_id（结束符）

如果指定了 `eos_id`，生成到结束 token 时提前停止。GPT-2 的结束 token 是 `<|endoftext|>`（id=50256）。

---

## 三、代码实现总览

### 3.1 文件结构

```
01_main-chapter-code/
├── ch05.ipynb              # 章节主 notebook（所有代码的完整呈现）
├── previous_chapters.py    # 前 4 章代码汇总（GPTModel, MHA, DataLoader 等）
├── gpt_train.py            # 独立训练脚本（本章训练代码的总结版）
├── gpt_generate.py         # 独立生成脚本（加载 GPT-2 权重 + 高级采样）
├── gpt_download.py         # GPT-2 权重下载工具（带 backup URL）
├── exercise-solutions.ipynb # 习题解答
├── tests.py                # 单元测试
└── README.md               # 目录说明
```

### 3.2 训练脚本完整流程（gpt_train.py）

```
1. 设置随机种子 + 选择设备（cuda/cpu）
2. 下载/读取训练数据（the-verdict.txt）
3. 实例化 GPTModel + 移到设备
4. 创建 AdamW 优化器
5. 90:10 切分训练/验证集
6. 创建 train_loader / val_loader
7. 初始化 tokenizer（gpt2）
8. 调用 train_model_simple 开始训练
   ├── 每个 epoch:
   │   ├── 每个 batch: zero_grad → forward → loss → backward → step
   │   └── 每 eval_freq 步: 评估 train/val loss
   └── 每个 epoch 结束: 生成样本文本
9. 绘制 loss 曲线 → 保存为 loss.pdf
10. 保存模型权重 → model.pth
```

### 3.3 生成脚本完整流程（gpt_generate.py）

```
1. 解析命令行参数（--prompt, --device）
2. 选择模型规格（默认 gpt2-small 124M）
3. 构建模型配置（BASE_CONFIG + model_configs[CHOOSE_MODEL]）
4. 调用 main():
   ├── download_and_load_gpt2: 下载 TF checkpoint + 解析为 params 字典
   ├── 实例化 GPTModel
   ├── load_weights_into_gpt: 把 params 映射到 PyTorch 模型
   ├── 模型移到设备 + eval() 模式
   ├── 初始化 tokenizer
   ├── 调用 generate() 生成文本（top_k=50, temperature=1.0）
   └── 打印输出
```

### 3.4 超参数配置

#### 训练用配置（小规模，适合 CPU 演示）

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,      # 词表大小（gpt2 固定）
    "context_length": 256,    # 缩短上下文（原版 1024，训练数据小用不上）
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

OTHER_SETTINGS = {
    "learning_rate": 5e-4,    # AdamW 学习率
    "num_epochs": 10,
    "batch_size": 2,          # 小 batch，CPU 也能跑
    "weight_decay": 0.1       # AdamW 权重衰减
}
```

#### 推理用配置（加载官方 GPT-2 权重）

```python
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,   # 原版完整上下文
    "drop_rate": 0.0,         # 推理时不需要 dropout
    "qkv_bias": True          # 官方 GPT-2 用了 qkv bias
}

model_configs = {
    "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
```

**训练配置 vs 推理配置的关键差异**：

| 配置项 | 训练（自己训） | 推理（加载官方权重） |
|--------|---------------|---------------------|
| context_length | 256 | 1024 |
| drop_rate | 0.1 | 0.0 |
| qkv_bias | False | True |

**为什么 qkv_bias 不同？**
- 我们自己训练时用 `qkv_bias=False` 是为了简化（少几个参数，小数据集上影响不大）
- 官方 GPT-2 用了 `qkv_bias=True`，所以加载官方权重时必须设为 True，否则形状不匹配会报错

---

## 四、易错点

### 4.1 忘记 `optimizer.zero_grad()`

PyTorch 的梯度是**累积**的，每次 `backward()` 会把新梯度加到 `param.grad` 上，而不是替换。如果忘记 `zero_grad()`，梯度会越来越大，训练完全异常。

**正确顺序**：`zero_grad()` → `forward()` → `backward()` → `step()`

### 4.2 评估时忘记 `model.eval()` 和 `torch.no_grad()`

- 不调用 `model.eval()`：dropout 仍然在随机丢弃神经元，评估结果不稳定
- 不使用 `torch.no_grad()`：仍然计算梯度，浪费内存和速度，且可能因为梯度累积影响后续训练

### 4.3 加载官方权重时形状不匹配

常见原因：
- `qkv_bias` 设置错误（官方是 True，自己训练时可能用了 False）
- `context_length` 不匹配（位置嵌入权重形状是 `(context_length, emb_dim)`）
- 忘记转置（TF 权重是 `(in, out)`，PyTorch 是 `(out, in)`）

### 4.4 `F.cross_entropy` 的输入形状

`F.cross_entropy(input, target)` 要求：
- `input` 形状：`(N, C)` 或 `(N, C, d1, d2, ...)`
- `target` 形状：`(N,)` 或 `(N, d1, d2, ...)`（类别索引，不是 one-hot）

所以必须把 `(batch, seq_len, vocab_size)` flatten 成 `(batch*seq_len, vocab_size)`。

### 4.5 训练数据太小导致过拟合

本章用的《The Verdict》只有约 2 万 token，训练 10 个 epoch 后训练 loss 会降得很低，但验证 loss 可能开始上升——这是过拟合。这是正常的，因为本章目的是**跑通训练流程**，不是训练一个好用的模型。

### 4.6 `stride = context_length` 时的样本数

当 `stride = context_length`（无重叠）时，样本数大约是 `len(token_ids) // context_length`。如果训练文本很短，可能只有几个 batch，训练很快就结束了。

### 4.7 matplotlib 后端问题（本项目环境）

本项目虚拟环境**没有 Tcl/Tk GUI**，不能用默认的 TkAgg 后端。绘图脚本必须在 `import matplotlib.pyplot` 之前设置：

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

否则会报 `_tkinter.TclError: Can't find a usable init.tcl`。

---

## 五、面试高频题

### Q1: 大模型预训练的目标函数是什么？为什么用交叉熵？

**答**：预训练优化的是下一个 token 预测的交叉熵损失（也叫负对数似然 NLL）。对于每个位置，模型输出一个 vocab_size 维的概率分布，交叉熵衡量这个分布和真实 token（one-hot）之间的差异。

用交叉熵的原因：
1. 它等价于最大化真实 token 的对数概率（MLE），直觉清晰
2. 梯度形式简单：`∂L/∂logits_i = softmax(logits)_i - one_hot_i`，实现高效
3. 对分类问题是标准且有效的损失函数

### Q2: 训练循环中为什么要先 `zero_grad()` 再 `backward()`？

**答**：PyTorch 的 `backward()` 会把梯度**累积**到 `param.grad` 上（即 `param.grad += new_grad`），而不是替换。这是为了支持梯度累积（gradient accumulation）等技巧——当显存不够大 batch 时，可以分多个小 batch 累积梯度再更新。

但在普通训练中，每个 batch 的梯度应该独立计算，所以必须在 `backward()` 之前调用 `zero_grad()` 清零上一步的梯度，否则梯度会越来越大，训练异常。

### Q3: `model.eval()` 和 `torch.no_grad()` 有什么区别？

**答**：两者作用不同，通常一起使用：
- `model.eval()`：切换模型的**模式**，会影响 dropout（关闭）和 batch norm（用运行时统计量而非当前 batch 统计量）等层的行为。它不影响梯度计算。
- `torch.no_grad()`：是一个**上下文管理器**，在其范围内禁用梯度计算（不构建计算图），节省内存和速度。它不影响模型模式。

评估时两者都需要：`model.eval()` 保证行为正确，`torch.no_grad()` 保证效率。

### Q4: temperature 和 top-k 采样分别解决什么问题？可以同时用吗？

**答**：
- **greedy 解码**（argmax）的问题：生成文本单调、重复、缺乏多样性。
- **top-k**：只保留概率最高的 k 个候选，过滤掉低概率的"离谱"选项，在 k 个里面采样。解决"可能选到完全不合理的词"的问题。
- **temperature**：通过缩放 logits 来调整概率分布的尖锐程度。temperature 高→分布平坦→更多样；temperature 低→分布尖锐→更确定。解决"概率太集中导致采样和 greedy 没区别"的问题。

**可以同时用**：先 top-k 过滤候选，再 temperature 缩放，最后 softmax + 采样。这是 GPT-2 等模型的标准生成配置。

### Q5: 什么是 weight tying？为什么要这样做？

**答**：Weight tying（权重共享）是指让模型的 token embedding 层和输出投影层使用同一套权重矩阵。

**原因**：
1. **减少参数量**：embedding 和 output head 都是 `(vocab_size, emb_dim)` 的大矩阵，共享后参数减半。
2. **语义直觉**：embedding 是"词 → 向量"的编码，output head 是"向量 → 词"的解码，两者是逆操作，共享权重有天然的语义一致性。
3. **实证效果好**：在语言模型中，weight tying 通常不会降低性能，有时还能提升。

GPT-2、GPT-3 等模型都使用了 weight tying。

### Q6: 为什么加载 TensorFlow 权重时需要转置？

**答**：因为 TensorFlow 和 PyTorch 的全连接层权重存储约定不同：
- TensorFlow / Keras 的 `Dense` 层权重形状是 `(input_dim, output_dim)`（行是输入，列是输出）
- PyTorch 的 `nn.Linear` 层权重形状是 `(output_dim, input_dim)`（行是输出，列是输入）

这是因为两者的矩阵乘法约定不同：
- TF: `output = input @ W + b`，所以 W 是 `(in, out)`
- PyTorch: `output = input @ W.T + b`，所以 W 是 `(out, in)`

因此从 TF 加载权重到 PyTorch 时需要转置。

### Q7: 训练集和验证集的 loss 曲线应该怎么看？什么是过拟合？

**答**：
- **正常情况**：训练 loss 和验证 loss 都下降，最终验证 loss 略高于训练 loss。
- **过拟合**：训练 loss 持续下降，但验证 loss 先降后升——模型记住了训练数据的细节，泛化能力下降。
- **欠拟合**：训练 loss 和验证 loss 都很高且不降——模型容量不足或训练不够。

本章因为训练数据极小（2 万 token），很容易观察到过拟合现象。实际大模型预训练用 TB 级数据，过拟合不是主要问题。

### Q8: AdamW 优化器和普通 SGD 有什么区别？为什么大模型训练常用 AdamW？

**答**：
- **SGD**：只使用当前梯度更新参数，`θ -= lr * grad`。收敛慢，对学习率敏感，但泛化往往更好。
- **Adam**：结合了动量（momentum）和自适应学习率（RMSprop），为每个参数维护一阶矩（梯度的指数移动平均）和二阶矩（梯度平方的指数移动平均），收敛快，对超参数不敏感。
- **AdamW**：是 Adam 的修正版，把权重衰减（weight decay）从梯度计算中解耦出来（Adam 中 weight decay 被错误地混进了梯度），更符合"L2 正则化"的原始定义。

大模型训练常用 AdamW 的原因：收敛快、训练稳定、对学习率不敏感、权重衰减处理正确。

### Q9: 预训练和微调有什么区别？

**答**：
- **预训练（Pretraining）**：在大规模无标签文本上训练，目标是下一个 token 预测。模型学到通用语言能力。数据量大、计算量大、时间长。
- **微调（Fine-tuning）**：在预训练模型的基础上，用特定任务的有标签数据继续训练，适配具体任务（如分类、问答、摘要）。数据量小、计算量小、时间短。

类比：预训练像"读万卷书"（广泛学习通用知识），微调像"专业实习"（针对具体岗位强化技能）。

### Q10: 为什么 GPT 是 decoder-only 架构？和 encoder-decoder（如 T5）有什么区别？

**答**：
- **GPT（decoder-only）**：只有 Transformer decoder 层，使用因果注意力（causal attention，每个位置只能看到前面的 token）。适合生成任务，通过"预测下一个 token"统一所有任务。
- **T5/BART（encoder-decoder）**：有 encoder 和 decoder 两部分。encoder 用双向注意力（能看到完整输入），decoder 用因果注意力。适合输入-输出映射明确的任务（如翻译、摘要）。

decoder-only 的优势：架构简单、训练目标统一（next token prediction）、规模化效果好（Scaling Law）。GPT 系列证明了 decoder-only + 大规模预训练 + 指令微调的路径非常有效。

---

## 六、本章关键函数速查表

| 函数 | 所在文件 | 作用 |
|------|---------|------|
| `text_to_token_ids` | gpt_train.py / gpt_generate.py | 文本 → token id 张量（加 batch 维） |
| `token_ids_to_text` | gpt_train.py / gpt_generate.py | token id 张量 → 文本（去 batch 维） |
| `calc_loss_batch` | gpt_train.py | 计算单个 batch 的交叉熵损失 |
| `calc_loss_loader` | gpt_train.py | 计算整个 dataloader（或前 N 个 batch）的平均损失 |
| `evaluate_model` | gpt_train.py | 评估模型在训练集和验证集上的损失 |
| `generate_and_print_sample` | gpt_train.py | 训练中生成样本文本并打印 |
| `train_model_simple` | gpt_train.py | 完整训练循环（核心函数） |
| `plot_losses` | gpt_train.py | 绘制训练/验证 loss 曲线（双 x 轴） |
| `download_and_load_gpt2` | gpt_download.py / gpt_generate.py | 下载 GPT-2 权重并解析为 params 字典 |
| `download_file` | gpt_download.py / gpt_generate.py | 带进度条的文件下载 |
| `load_gpt2_params_from_tf_ckpt` | gpt_download.py / gpt_generate.py | TF checkpoint → 嵌套字典 |
| `assign` | gpt_generate.py | 形状校验 + numpy → nn.Parameter |
| `load_weights_into_gpt` | gpt_generate.py | params 字典 → 自定义 GPTModel 权重 |
| `generate` | gpt_generate.py | 带 temperature + top-k 的文本生成 |
| `generate_text_simple` | previous_chapters.py | greedy 文本生成（第 4 章） |

---

## 七、运行命令参考

### 训练模型

```powershell
& "C:\Users\renbo\Desktop\llm project\.venv\Scripts\python.exe" "C:\Users\renbo\Desktop\llm project\stage-01\code\ch05\01_main-chapter-code\gpt_train.py"
```

运行后会在当前目录生成：
- `the-verdict.txt`：训练数据
- `loss.pdf`：损失曲线图
- `model.pth`：训练好的模型权重

### 用官方 GPT-2 生成文本

```powershell
& "C:\Users\renbo\Desktop\llm project\.venv\Scripts\python.exe" "C:\Users\renbo\Desktop\llm project\stage-01\code\ch05\01_main-chapter-code\gpt_generate.py" --prompt "Every effort moves you" --device cpu
```

首次运行会下载 GPT-2 124M 权重（约 500MB）到 `gpt2/124M/` 目录。

> **注意**：`gpt_generate.py` 依赖 `tensorflow`（用于读取 TF checkpoint），如果环境中没装需要先安装。`gpt_train.py` 只依赖 PyTorch，不需要 tensorflow。

---

## 八、与前后章节的关联

- **第 2 章**：提供了 `GPTDatasetV1` 和 `create_dataloader_v1`，本章直接复用
- **第 3 章**：提供了 `MultiHeadAttention`，是 GPT 模型的核心组件
- **第 4 章**：提供了 `GPTModel`、`LayerNorm`、`FeedForward`、`TransformerBlock` 和 `generate_text_simple`，本章在其基础上添加训练逻辑和高级生成策略
- **第 6 章**：将在预训练基础上进行微调（fine-tuning），适配具体任务

---

*笔记整理完成。建议配合 `ch05.ipynb` 逐 cell 运行，观察训练过程中 loss 下降和生成文本的变化。*
