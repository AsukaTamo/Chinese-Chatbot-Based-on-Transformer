# Transformer 中文聊天机器人

从零手写 Transformer，基于 PyTorch 实现的中文对话机器人（小黄鸡），参考论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)。

[📖 English Docs](README_ENG.md)

---

## 特色

- **双架构支持** — Encoder-Decoder (Transformer) 和 Decoder-Only (GPT) 两种架构，一行配置切换
- **短期记忆** — GPT 模式支持多轮对话上下文，自动维护历史记录
- **权重共享** — Encoder 嵌入、Decoder 嵌入、输出投影共享权重矩阵
- **Noam 学习率调度** — 内置 Warmup 机制，复现原论文训练策略
- **标签平滑** — 内存高效的 Label Smoothing Cross-Entropy 实现
- **混合精度训练** — 支持 AMP 自动混合精度，节省显存
- **多种解码策略** — Beam Search / 贪心 / 温度采样 (Top-K + Top-P)
- **SDPA 加速** — 使用 PyTorch `scaled_dot_product_attention`，自动启用 Flash Attention 后端
- **KV-Cache 增量解码** — 推理时复用历史 K/V，速度提升 5-10 倍
- **重复惩罚 + N-gram 阻断** — 消除模型重复输出"我我我"等退化现象
- **多语料支持** — 一行配置切换单语料 / 多语料联合训练，结果按语料名自动分目录管理
- **命令行训练** — 支持 `--corpora`、`--epoch`、`--batch` 等参数，无需修改配置文件即可启动训练

---

## 项目结构

```
.
├── model.py           # Encoder-Decoder Transformer 模型
├── model_gpt.py       # Decoder-Only GPT 模型（支持多轮记忆）
├── config.py          # 超参数配置 + 语料库/模型选择
├── data_loader.py     # 数据预处理、词汇表构建、DataLoader
├── train.py           # 训练脚本（支持命令行参数）
├── inference.py       # 推理 & 交互式聊天（Beam Search / 采样）
├── requirements.txt   # 依赖
│
├── data/              # 语料目录（每个语料一个子文件夹）
│   ├── xiaohuangji/
│   │   ├── xiaohuangji50w_fenciA.conv
│   │   └── vocab.json
│   └── xiaohuangji+weibo/     ← 多语料时自动创建
│       └── vocab.json         ← 合并词表
│
└── checkpoints/       # 模型检查点（每个语料一个子文件夹）
    ├── xiaohuangji/
    │   ├── best_model.pt
    │   └── history.json
    └── xiaohuangji+weibo/
        ├── best_model.pt
        └── history.json
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
# GPT 模型训练（默认，支持多轮记忆）
python train.py --model gpt --corpora xiaohuangji

# Encoder-Decoder 训练
python train.py --model transformer --corpora xiaohuangji

# 多语料联合训练
python train.py --model gpt --corpora xiaohuangji,weibo

# 自定义训练参数
python train.py --model gpt --corpora xiaohuangji --epoch 50 --batch 64

# 查看所有命令行参数
python train.py --help
```

> `config.py` 中 `model_type = "gpt"` 控制默认架构，`d_model = 512` 已提升模型容量。

训练过程会：
- 自动解析对话语料并构建词汇表（保存到 `data/<语料名>/vocab.json`）
- 使用 Noam 调度器 + 标签平滑进行训练
- 每个 epoch 验证一次，自动保存最佳模型到 `checkpoints/<语料名>/best_model.pt`
- 训练历史记录在 `checkpoints/<语料名>/history.json`

### 3. 使用模型推理

训练完成后，模型保存在 `checkpoints/<语料名>/best_model.pt`。

#### 交互式聊天

```bash
python inference.py
```

根据 `config.py` 中 `model_type` 自动选择对应架构。GPT 模式下额外支持：

| 命令 | 说明 |
|------|------|
| `/beam` | 切换为 Beam Search 解码（质量最高，默认） |
| `/sample` | 切换为温度采样解码（多样性高） |
| `/greedy` | 切换为贪心解码（速度最快） |
| `/clear` | **清空对话记忆**（仅 GPT 模式） |
| `/history` | **查看当前记忆**（仅 GPT 模式） |
| `quit` / `exit` | 退出 |

#### GPT 多轮对话示例

```
你: 我叫小明
小黄鸡: 小明你好呀~

你: 我叫什么名字？
小黄鸡: 你叫小明呀，刚告诉我的~          ← 引用了上文

你: /clear
[记忆] 已清空

你: 我叫什么名字？
小黄鸡: 你没有告诉过我呀...              ← 记忆已清除
```

#### 程序化调用

```python
from config import Config
from inference import ChatBot

config = Config()
# 如需切换语料：config.corpora = ("xiaohuangji",) → 重新实例化 Config

bot = ChatBot(config.best_model_path, config)

# Beam Search（默认，质量最高）
reply = bot.reply("你好", use_beam=True)
print(reply)

# 随机采样（更多样化）
reply = bot.reply("你好", use_sample=True)
print(reply)

# 贪心解码（速度最快）
reply = bot.reply("你好", use_beam=False, use_sample=False)
print(reply)
```

#### 推理加速与质量优化

当前推理已内置以下优化（无需额外配置）：

| 优化 | 说明 |
|------|------|
| **KV-Cache** | 增量解码，每步只计算新 token，历史 K/V 自动复用 |
| **SDPA** | `scaled_dot_product_attention` 自动选择最优注意力后端 |
| **重复惩罚** (×1.2) | 对已出现 token 降低概率，减少"我我我"重复 |
| **3-gram 阻断** | 禁止生成与前面重复的三连词，消除机械重复 |

可在 `inference.py` 的 `BeamSearchDecoder.__init__` 中调整 `repetition_penalty` 和 `ngram_block` 参数。

#### 推理参数调优

在 [config.py](config.py) 中调整推理效果：

| 参数 | 推荐场景 |
|------|---------|
| `beam_size` ↑ | 提高回复质量，但速度变慢 |
| `temperature` ↑ | 提高回复多样性（采样模式下） |
| `length_penalty` < 1 | 鼓励短回复；> 1 鼓励长回复 |

---

## 配置

在 [config.py](config.py) 中调整超参数，也可通过命令行覆盖部分参数：

### 语料库选择

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `corpora` | `("xiaohuangji",)` | 语料库名称（`data/` 下的文件夹名）。多语料联合训练写 `("a", "b")` |
| `model_type` | `"gpt"` | 模型架构：`"gpt"` (Decoder-Only) 或 `"transformer"` (Encoder-Decoder) |

### 命令行参数

| 参数 | 说明 |
|------|------|
| `--model` | 模型架构：`gpt` (默认) 或 `transformer` |
| `--corpora` | 语料库名称，多语料用逗号分隔（例: `xiaohuangji,weibo`） |
| `--epoch` | 训练轮数 |
| `--batch` | 批次大小 |
| `--device` | 训练设备：`cuda` / `cpu` |
| `--resume` | 从指定 checkpoint 恢复训练 |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `d_model` | 512 | 词向量 / 隐层维度 |
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

默认使用小黄鸡对话语料（`xiaohuangji50w_fenciA.conv`），约 50 万条中文对话对。

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

## 语料库管理

### 单语料训练

1. 将 `.conv` 文件放入 `data/<语料名>/` 目录
2. 在 [config.py](config.py) 中设置 `corpora = ("<语料名>",)`，或通过命令行指定：
   ```bash
   python train.py --corpora <语料名>
   ```
3. 词表和模型自动保存到对应目录

### 多语料联合训练

合并多个语料库以扩大对话覆盖范围：

1. 每个语料放在 `data/` 下的独立文件夹中：
   ```
   data/
   ├── xiaohuangji/
   │   └── xiaohuangji50w_fenciA.conv
   └── weibo/
       └── weibo.conv
   ```
2. 通过 config 或命令行指定：
   ```bash
   python train.py --corpora xiaohuangji,weibo
   ```
3. 所有对话对合并训练，自动构建统一词表，结果保存到 `checkpoints/xiaohuangji+weibo/`

### 添加新语料

只需在 `data/` 下新建文件夹并放入 `.conv` 文件即可。数据格式要求：
- 每个对话段以 `E` 标记结尾
- 消息以 `M ` 前缀开头
- 词语用 `/` 分隔

文件名无限制，文件夹名即为语料标识符。

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
