# Transformer 中文聊天机器人

从零手写 Transformer，基于 PyTorch 实现的中文对话机器人（小黄鸡），参考论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)。

[📖 English Docs](README_ENG.md)

---

## 特色

- **纯手写 Transformer** — 不依赖 `torch.nn.Transformer`，逐模块实现 Encoder/Decoder/Multi-Head Attention
- **Pre-LN 架构** — 使用 Pre-LayerNorm 结构，训练更稳定
- **权重共享** — Encoder 嵌入、Decoder 嵌入、输出投影共享权重矩阵
- **Noam 学习率调度** — 内置 Warmup 机制，复现原论文训练策略
- **标签平滑** — 内存高效的 Label Smoothing Cross-Entropy 实现
- **混合精度训练** — 支持 AMP 自动混合精度，节省显存
- **多种解码策略** — Beam Search / 贪心 / 温度采样 (Top-K + Top-P)

---

## 项目结构

```
.
├── model.py           # Transformer 模型（Encoder/Decoder/Attention/FFN）
├── config.py          # 超参数配置（模型/训练/推理）
├── data_loader.py     # 数据预处理、词汇表构建、DataLoader
├── train.py           # 训练脚本（含 Noam 调度器 & 标签平滑损失）
├── inference.py       # 推理 & 交互式聊天（Beam Search / 采样）
├── vocab.json         # 词汇表文件（自动生成）
├── checkpoints/       # 模型检查点保存目录
│   ├── best_model.pt
│   └── history.json
├── xiaohuangji50w_fenciA.conv  # 小黄鸡对话语料（50w 条）
└── requirements.txt   # 依赖
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

- Python 3.10+
- PyTorch 2.0+（推荐 CUDA 版本以启用 GPU 加速）

### 2. 训练模型

```bash
python train.py
```

训练过程会：
- 自动解析对话语料并构建词汇表（保存为 `vocab.json`）
- 使用 Noam 调度器 + 标签平滑进行训练
- 每个 epoch 验证一次，自动保存最佳模型到 `checkpoints/best_model.pt`
- 训练历史记录在 `checkpoints/history.json`

### 3. 交互式聊天

```bash
python inference.py
```

聊天命令：

| 命令 | 说明 |
|------|------|
| `/beam` | 切换为 Beam Search 解码 |
| `/sample` | 切换为温度采样解码 |
| `/greedy` | 切换为贪心解码 |
| `quit` / `exit` | 退出 |

---

## 配置

在 [config.py](config.py) 中调整超参数：

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `d_model` | 128 | 词向量 / 隐层维度 |
| `n_heads` | 8 | 多头注意力头数 |
| `n_layers` | 6 | Encoder / Decoder 层数 |
| `d_ff` | 2048 | 前馈网络隐层维度 |
| `dropout` | 0.1 | Dropout 比例 |
| `max_len` | 60 | 最大序列长度 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 128 | 批次大小 |
| `epochs` | 30 | 训练轮数 |
| `warmup_steps` | 4000 | Noam 调度器预热步数 |
| `label_smoothing` | 0.1 | 标签平滑系数 |
| `grad_clip` | 1.0 | 梯度裁剪阈值 |

### 推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `beam_size` | 5 | Beam Search 宽度 |
| `temperature` | 0.8 | 采样温度 |
| `length_penalty` | 0.6 | 长度惩罚系数（<1 鼓励短句） |
| `max_decode_len` | 50 | 最大解码长度 |

---

## 模型架构

```
输入文本 → Token Embedding + Positional Encoding
         → Encoder (×N layers)
            ├── Pre-LN → Multi-Head Self-Attention → Residual
            └── Pre-LN → FeedForward → Residual
         → Decoder (×N layers)
            ├── Pre-LN → Masked Multi-Head Self-Attention → Residual
            ├── Pre-LN → Multi-Head Cross-Attention → Residual
            └── Pre-LN → FeedForward → Residual
         → Linear Projection → Softmax → 输出文本
```

### 关键实现细节

- **位置编码**: 正弦位置编码，预计算并注册为 buffer
- **多头注意力**: 手写 `_split_heads` / `_merge_heads`，支持因果 mask 和 padding mask
- **Pre-LN**: LayerNorm 放在子层之前，相比 Post-LN 训练更稳定
- **标签平滑损失**: 用数学展开避免 `(N, V)` 维度的大张量分配，内存高效

---

## 训练日志示例

```
Epoch  1 | Step   100 | Loss: 4.1234 | PPL: 61.7 | LR: 0.000123 | Time: 45s
Epoch  1 | Step   200 | Loss: 3.8567 | PPL: 47.3 | LR: 0.000174 | Time: 89s
...
-------------------------------------------------------------
Epoch  1/30 | Train Loss: 3.2145 | Val Loss: 3.0123 | Val PPL: 20.3 | Time: 120s
-------------------------------------------------------------
```

---

## 数据集

使用小黄鸡对话语料（`xiaohuangji50w_fenciA.conv`），约 50 万条中文对话对。

**数据格式** — 每个对话段以 `E` 标记结尾，对话消息以 `M ` 前缀开头，消息中词语用 `/` 分隔：

```
M 你/在/干嘛
M 在/跟/你/聊天/呀
E
M 今天/天气/怎么样
M 很/好/呀
E
```

相邻的 `M` 消息两两组为 Query-Response 对话对。

---

## 自定义数据集

替换 `xiaohuangji50w_fenciA.conv` 文件即可使用自己的对话数据，格式保持一致即可。在 [config.py](config.py) 中修改 `data_path` 指定新文件路径。

---

## 依赖

- **Python** ≥ 3.10
- **PyTorch** ≥ 2.0.0
- **NumPy**
- **tqdm**

---

## 参考文献

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., NeurIPS 2017
- [Pre-LN Transformer](https://arxiv.org/abs/2002.04745) — Xiong et al.
- [Rethinking Label Smoothing](https://arxiv.org/abs/1906.02629) — Müller et al.
---

## 语料下载地址

https://github.com/candlewill/Dialog_Corpus

## License

MIT License
