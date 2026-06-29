# Transformer 中文聊天机器人

从零手写 Transformer，基于 PyTorch 实现的中文对话机器人（小黄鸡），参考论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)。

[📖 English Docs](README_ENG.md)

---

## 特色

- **三档模型** — Lite (Transformer) / Middle (GPT) / Pro (Qwen-Chinese 预训练)，命令行自由切换
- **短期记忆** — GPT / Pro 模式支持多轮对话上下文，自动维护历史记录
- **预训练微调** — Pro 版基于 `uer/qwen-chinese-cluecorpussmall`（100M 参数），中文知识 + 对话微调
- **权重共享** — Encoder 嵌入、Decoder 嵌入、输出投影共享权重矩阵
- **Noam 学习率调度** — 内置 Warmup 机制，复现原论文训练策略
- **标签平滑** — 内存高效的 Label Smoothing Cross-Entropy 实现
- **混合精度训练** — 支持 AMP 自动混合精度，节省显存
- **多种解码策略** — Beam Search / 贪心 / 温度采样 (Top-K + Top-P)
- **SDPA 加速** — 使用 PyTorch `scaled_dot_product_attention`，自动启用 Flash Attention 后端
- **KV-Cache 增量解码** — 推理时复用历史 K/V，速度提升 5-10 倍
- **重复惩罚 + N-gram 阻断** — 消除模型重复输出"我我我"等退化现象
- **多语料支持** — 一行配置切换单语料 / 多语料联合训练，结果按语料名自动分目录管理
- **命令行训练** — 支持 `--corpora`、`--epoch`、`--batch`、`--fenci` 等参数，无需修改配置文件即可启动训练
- **双分词引擎** — `--fenci jieba`（词级）和 `--fenci space`（空格切分），适配不同语料格式
- **双格式语料** — 支持 LCCC JSON（空格预分词）和 .conv（原始文本）两种语料格式

---

## 项目结构

```
.
├── model.py           # Encoder-Decoder Transformer (Lite)
├── model_gpt.py       # Decoder-Only GPT 手写 (Middle)
├── model_qwen.py      # Qwen-Chinese 预训练封装 (Pro)
├── config.py          # 超参数配置 + 语料库/模型选择
├── data_loader.py     # 数据预处理、词汇表构建、DataLoader
├── train.py           # 训练脚本（支持命令行参数）
├── inference.py       # 推理 & 交互式聊天（Beam Search / 采样）
├── requirements.txt   # 依赖
│
├── data/              # 语料目录（每个语料一个子文件夹）
│   ├── xiaohuangji/
│   │   ├── xiaohuangji50w_fenciA.conv
│   │   ├── vocab_gpt.json            ← GPT 专用词表
│   │   └── vocab_transformer.json    ← Transformer 专用词表
│   ├── LCCC-base-split/              ← LCCC 大规模中文对话
│   │   ├── LCCC-base_train.json      ← 8.9M 对话对（JSON 格式）
│   │   ├── LCCC-base_valid.json
│   │   └── LCCC-base_test.json
│   └── xiaohuangji+weibo/     ← 多语料时自动创建
│       └── vocab_gpt.json     ← 合并词表
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
# Middle 版：从零训练手写 GPT（LCCC 大语料 + 空格分词）
python train.py --model gpt --corpora LCCC-base-split --fenci space --epoch 10 --batch 32

# Middle 版：从零训练手写 GPT（小黄鸡小语料 + jieba 分词）
python train.py --model gpt --corpora xiaohuangji --fenci jieba

# Lite 版：Encoder-Decoder
python train.py --model transformer --corpora xiaohuangji

# Pro 版：微调预训练 Qwen-Chinese（首次自动下载 ~500MB 权重）
python train.py --model qwen --corpora xiaohuangji --epoch 5 --batch 32

# 查看所有命令行参数
python train.py --help
```

> **分词选择**：LCCC 语料已用空格预分词，使用 `--fenci space` 速度提升约 20 倍且不需 jieba；原始中文文本（xiaohuangji）使用 `--fenci jieba`（默认）自动分词。

> `config.py` 中 `model_type = "gpt"` 控制默认架构。Pro 版首次运行需联网下载权重。

### 模型对比

| 版本 | --model | 架构 | 参数量 | 记忆 | 效果 |
|------|---------|------|--------|------|------|
| Lite | `transformer` | Encoder-Decoder 手写 | ~15M | 无 | 基础 |
| Middle | `gpt` | LLaMA-style Decoder-Only 手写 | ~76M* | ✓ | 中等 |
| **Pro** | `qwen` | Qwen-Chinese 预训练 | ~100M | ✓ | **最强** |

> \*GPT 参数量与 `vocab_size` 联动（输出投影 = vocab_size × d_model），vocab_size=100000 时约 76M，50000 时约 51M。

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

### 语料与模型选择

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `corpora` | `("xiaohuangji",)` | 语料库名称（`data/` 下的文件夹名）。多语料联合训练写 `("a", "b")` |
| `model_type` | `"gpt"` | 模型架构：`"gpt"` (Decoder-Only) \| `"transformer"` (Encoder-Decoder) \| `"qwen"` (预训练) |
| `fenci_mode` | `"jieba"` | 分词模式：`"jieba"` (词级别) \| `"space"` (空格切分，适用于 LCCC) |
| `vocab_size` | 100000 | 词汇表上限（建议 LCCC 语料 50000，小语料按需调低） |
| `min_freq` | 3 | 最低词频阈值，低于此频次的 token 被丢弃 |

### 命令行参数

| 参数 | 说明 |
|------|------|
| `--model` | 模型架构：`transformer` \| `gpt` \| `qwen` |
| `--corpora` | 语料库名称，多语料用逗号分隔（例: `xiaohuangji,weibo`） |
| `--fenci` | 分词模式：`jieba` (默认) \| `space` |
| `--epoch` | 训练轮数 |
| `--batch` | 批次大小 |
| `--device` | 训练设备：`cuda` / `cpu` |
| `--resume` | 从指定 checkpoint 恢复训练 |

### 模型架构参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `d_model` | 512 | 词向量 / 隐层维度 |
| `n_heads` | 8 | 多头注意力头数 |
| `n_layers` | 6 | Decoder 层数 |
| `d_ff` | 2048 | SwiGLU 前馈网络隐层维度 |
| `dropout` | 0.1 | Dropout 比例 |
| `max_len` | 120 | 最大序列长度（GPT 模式需容纳 User + Query + Assistant + Response + EOS） |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 64 | 批次大小（vocab_size=100000 时勿超过 64） |
| `epochs` | 30 | 训练轮数（LCCC 大语料建议 5-10） |
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

### GPT Decoder-Only（LLaMA-style）

```
Token IDs → Embedding
          → DecoderLayer × N
             ├── RMSNorm → RoPE Multi-Head Self-Attention (无 bias) → Residual
             ├── RMSNorm → SwiGLU Feed-Forward Network → Residual
          → RMSNorm → Linear (共享 Embedding 权重) → Vocab Logits
```

GPT 架构特性：
- **RMSNorm** — 比 LayerNorm 更轻量，去除了平移参数 (bias)
- **RoPE** (Rotary Position Embedding) — 旋转位置编码，支持序列外推
- **SwiGLU** — 现代 LLM 标配激活函数，优于 ReLU
- **无 bias** — 所有线性层无 bias，遵循 LLaMA 风格
- **Pre-Norm** — Normalization 在子层之前，训练更稳定
- **Speaker Token** — `<|user|>` / `<|assistant|>` 区分对话角色
- **全序列 Loss** — 模型学习完整对话流，而非仅助词回复
- **KV-Cache 增量解码** — 推理时每步只计算新 token

### Transformer Encoder-Decoder

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

### 内置语料

| 语料 | 标识符 | 规模 | 格式 | 推荐分词 |
|------|--------|------|------|----------|
| 小黄鸡 | `xiaohuangji` | ~50 万对话对 | `.conv` | `--fenci jieba` |
| LCCC-base | `LCCC-base-split` | ~890 万对话对 | `.json` | `--fenci space` |

> LCCC (Large-scale Clean Chinese Conversation) — 清华大学发布的大规模中文日常对话数据集。

### .conv 语料格式规范

所有 `.conv` 语料文件格式要求如下：

```
M <原始中文文本>
M <原始中文文本>
E
```

- 每条消息以 `M `（大写 M + 一个空格）开头，后跟**原始中文文本**（无需预分词，jieba 自动处理）
- 每个对话段以单独一行 `E` 结尾
- 段内相邻的 `M` 消息两两组为 Query-Response 对（第 1、2 条配对，第 3、4 条配对...）
- 文件编码必须为 UTF-8

**示例：**

```
M 你在干嘛
M 在跟你聊天呀
E
M 今天天气怎么样
M 很好呀
E
```

### JSON 语料格式规范（LCCC 标准）

```json
[
  ["消息1", "消息2", "消息3"],
  ["消息1", "消息2"],
  ...
]
```

- 顶层是数组，每个元素是一次对话
- 每次对话包含 N 条消息（N ≥ 2），消息以空格预分词（如 `"我 饿 了 。"`）
- 相邻消息两两配对：(msg[0], msg[1]), (msg[2], msg[3]), ...
- 文件编码 UTF-8

**示例：**

```json
[
  ["你 好 呀", "你 好 你 好"],
  ["吃 了 吗", "还 没 呢", "那 一 起 呀"]
]
```

> 第一条为 1 个 QA 对（"你好呀"→"你好你好"），第二条配对前两条（"吃了吗"→"还没呢"），第三条消息丢弃。

### 分词引擎说明

| --fenci | 处理方式 | 速度 | 适用场景 |
|---------|---------|------|---------|
| `jieba` (默认) | 先去除空格/`/` 分隔符，再 jieba 词级分词 | 慢 | `.conv` 原始中文文本 |
| `space` | 直接按空格切分 | 快 (~20×) | `.json` LCCC 等预分词语料 |

### 兼容性

- 已有带 `/` 分隔符的旧版 `.conv` 文件也能正常使用——`tokenize()` 会自动移除 `/`，再用 jieba 重新分词
- `.conv` 和 `.json` 可混合使用，程序按后缀自动选择解析器

### 语料目录规范

```
data/
└── <语料名>/           # 文件夹名 = 语料标识符（如 xiaohuangji、weibo）
    └── *.conv          # 一个或多个 .conv 文件（文件名不限）
```

`<语料名>` 同时也是词表和检查点的目录名，多语料联合训练时以 `+` 连接（如 `xiaohuangji+weibo`）。

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

只需两步：

1. 在 `data/` 下新建文件夹（文件夹名即为语料标识符）
2. 放入 `.json` 或 `.conv` 格式的数据文件

```bash
# JSON 格式（推荐，LCCC 标准）
mkdir -p data/mycorpus
cp /path/to/data.json data/mycorpus/
python train.py --corpora mycorpus --fenci space

# .conv 格式（传统）
cp /path/to/data.conv data/mycorpus/
python train.py --corpora mycorpus --fenci jieba
```

JSON 语料如已用空格预分词，使用 `--fenci space` 速度显著更快。`.conv` 格式无需手动分词——jieba 会在构建词表时自动处理。旧版带 `/` 分隔符的语料也兼容，程序会自动移除并重新分词。

---

## 依赖

- **Python** ≥ 3.10
- **PyTorch** ≥ 2.0.0
- **NumPy**
- **tqdm**（进度条）
- **jieba**（`--fenci jieba` 模式下需要，`--fenci space` 不需要）
- **transformers** ≥ 4.30.0（仅 Qwen Pro 版需要）

---

## 参考文献

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., NeurIPS 2017
- [Pre-LN Transformer](https://arxiv.org/abs/2002.04745) — Xiong et al.
- [Rethinking Label Smoothing](https://arxiv.org/abs/1906.02629) — Müller et al.
---

## 语料下载

| 语料 | 链接 | 说明 |
|------|------|------|
| 小黄鸡 | [Dialog_Corpus](https://github.com/candlewill/Dialog_Corpus) | 50 万中文对话对 |
| LCCC-base | [LCCC](https://github.com/thu-coai/CDial-GPT) | 890 万中文日常对话（清华） |

## License

MIT License
