# Transformer Chinese Chatbot

A Transformer chatbot built from scratch in PyTorch, trained on the XiaoHuangJi (小黄鸡) Chinese conversation dataset. Based on [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

[📖 中文文档](README.md)

---

## Features

- **Transformer from scratch** — Encoder/Decoder/Multi-Head Attention implemented by hand, zero reliance on `torch.nn.Transformer`
- **Three-tier models** — Lite (Transformer) / Middle (GPT from scratch) / Pro (Qwen-Chinese pretrained), switch via CLI
- **Short-term memory** — GPT / Pro mode maintains conversation history for multi-turn context
- **Pretrained finetuning** — Pro is based on `uer/qwen-chinese-cluecorpussmall` (100M params) with Chinese knowledge
- **Weight tying** — Shared weights across Encoder embedding, Decoder embedding, and output projection
- **Noam scheduler** — Learning rate warmup strategy from the original paper
- **Label smoothing** — Memory-efficient cross-entropy with label smoothing
- **Mixed precision training** — AMP support to reduce GPU memory usage
- **Multiple decoding strategies** — Beam Search / Greedy / Temperature Sampling (Top-K + Top-P)
- **SDPA acceleration** — Uses PyTorch `scaled_dot_product_attention` with automatic Flash Attention backend
- **KV-Cache incremental decoding** — Reuses historical K/V across steps, 5-10× inference speedup
- **Repetition penalty + N-gram blocking** — Eliminates degenerate repetitive outputs
- **Multi-corpus support** — Train on single or multiple corpora with one-line config switch; results auto-organized by corpus name
- **CLI-driven training** — Start training with `--corpora`, `--epoch`, `--batch`, `--fenci` flags — no need to edit config files
- **Dual tokenization** — `--fenci jieba` (word-level) and `--fenci space` (whitespace split) for different corpus formats
- **Dual format corpora** — Supports LCCC JSON (space-pretokenized) and .conv (raw text) corpus formats

---

## Project Structure

```
.
├── model.py           # Encoder-Decoder Transformer (Lite)
├── model_gpt.py       # Decoder-Only GPT hand-written (Middle)
├── model_qwen.py      # Qwen-Chinese pretrained wrapper (Pro)
├── config.py          # Hyperparameters + corpus/model selection
├── data_loader.py     # Data preprocessing, vocabulary builder, DataLoader
├── train.py           # Training script (with CLI argument support)
├── inference.py       # Inference & interactive chat (Beam Search / Sampling)
├── requirements.txt   # Dependencies
│
├── data/              # Corpus directory (one subfolder per corpus)
│   ├── xiaohuangji/
│   │   ├── xiaohuangji50w_fenciA.conv
│   │   ├── vocab_gpt.json             ← GPT vocabulary
│   │   └── vocab_transformer.json     ← Transformer vocabulary
│   ├── LCCC-base-split/               ← Large-scale Chinese daily dialog
│   │   ├── LCCC-base_train.json       ← 8.9M QA pairs (JSON format)
│   │   ├── LCCC-base_valid.json
│   │   └── LCCC-base_test.json
│   └── xiaohuangji+weibo/     ← auto-created for multi-corpus
│       └── vocab_gpt.json     ← merged vocabulary
│
└── checkpoints/       # Model checkpoints (one subfolder per corpus)
    ├── xiaohuangji/
    │   ├── best_model.pt
    │   └── history.json
    └── xiaohuangji+weibo/
        ├── best_model.pt
        └── history.json
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

- Python 3.10+
- PyTorch 2.0+ (CUDA recommended for GPU acceleration)

### 2. Train the Model

```bash
# Middle: train GPT from scratch (LCCC dataset + space tokenization)
python train.py --model gpt --corpora LCCC-base-split --fenci space --epoch 10 --batch 32

# Middle: train GPT from scratch (xiaohuangji + jieba tokenization)
python train.py --model gpt --corpora xiaohuangji --fenci jieba

# Lite: Encoder-Decoder
python train.py --model transformer --corpora xiaohuangji

# Pro: finetune pretrained Qwen-Chinese (auto-downloads ~500MB weights first run)
python train.py --model qwen --corpora xiaohuangji --epoch 5 --batch 32

# Show all CLI options
python train.py --help
```

> **Tokenization**: LCCC corpora are pre-tokenized with spaces — use `--fenci space` for ~20× faster vocab building. Raw Chinese text (xiaohuangji) uses `--fenci jieba` (default).

### Model Comparison

| Tier | --model | Architecture | Params | Memory | Quality |
|------|---------|-------------|--------|--------|---------|
| Lite | `transformer` | Encoder-Decoder (handwritten) | ~15M | No | Basic |
| Middle | `gpt` | LLaMA-style Decoder-Only (handwritten) | ~76M* | Yes | Moderate |
| **Pro** | **`qwen`** | **Qwen-Chinese (pretrained)** | **~100M** | **Yes** | **Best** |

> \*GPT params scale with `vocab_size` (output projection = vocab_size × d_model): ~76M at vocab_size=100000, ~51M at 50000.

The training pipeline will:
- Parse the conversation corpus and build a vocabulary (saved to `data/<corpus>/vocab_<model_type>.json` — isolated per model architecture)
- Train with Noam scheduler + label smoothing
- Validate after each epoch and save the best model to `checkpoints/<corpus>/best_model.pt`
- Record training history in `checkpoints/<corpus>/history.json`

### 3. Run Inference

After training, the model is saved at `checkpoints/<corpus>/best_model.pt`, with the vocabulary at `data/<corpus>/vocab_<model_type>.json`.

#### Interactive Chat

```bash
python inference.py
```

Auto-selects architecture based on `config.model_type`. GPT mode adds memory commands:

| Command | Description |
|---------|-------------|
| `/beam` | Switch to Beam Search decoding (best quality, default) |
| `/sample` | Switch to temperature sampling (more diverse) |
| `/greedy` | Switch to greedy decoding (fastest) |
| `/clear` | **Clear conversation memory** (GPT only) |
| `/history` | **View current memory** (GPT only) |
| `quit` / `exit` | Exit |

#### GPT Multi-Turn Example

```
你: 我叫小明
小黄鸡: 小明你好呀~

你: 我叫什么名字？
小黄鸡: 你叫小明呀，刚告诉我的~          ← References earlier turn

你: /clear
[Memory] Cleared

你: 我叫什么名字？
小黄鸡: 你没有告诉过我呀...              ← Memory is gone
```

#### Programmatic API

```python
from config import Config
from inference import ChatBot

config = Config()
# To switch corpora: config.corpora = ("xiaohuangji",) → re-instantiate Config

bot = ChatBot(config.best_model_path, config)

# Beam Search (default, best quality)
reply = bot.reply("你好", use_beam=True)
print(reply)

# Random sampling (more diverse)
reply = bot.reply("你好", use_sample=True)
print(reply)

# Greedy decoding (fastest)
reply = bot.reply("你好", use_beam=False, use_sample=False)
print(reply)
```

#### Inference Optimizations

The following optimizations are enabled by default:

| Feature | Description |
|---------|-------------|
| **KV-Cache** | Incremental decoding — only the new token is computed per step |
| **SDPA** | `scaled_dot_product_attention` auto-selects optimal attention backend |
| **Repetition penalty** (×1.2) | Reduces probability of already-generated tokens |
| **3-gram blocking** | Bans repeated trigrams to prevent mechanical loops |

Tune `repetition_penalty` and `ngram_block` in `BeamSearchDecoder.__init__` within `inference.py`.

#### Tuning Inference

Adjust in [config.py](config.py):

| Parameter | Guidance |
|-----------|----------|
| `beam_size` ↑ | Better quality, slower |
| `temperature` ↑ | More diverse output (sampling mode) |
| `length_penalty` < 1 | Favors shorter replies; > 1 favors longer replies |

---

## Configuration

Adjust hyperparameters in [config.py](config.py), or override them via command-line flags:

### Corpus & Model Selection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `corpora` | `("xiaohuangji",)` | Corpus names (folders under `data/`). Use `("a", "b")` for joint training |
| `model_type` | `"gpt"` | Model architecture: `"gpt"` \| `"transformer"` \| `"qwen"` |
| `fenci_mode` | `"jieba"` | Tokenization: `"jieba"` (word-level) \| `"space"` (whitespace split, for LCCC) |
| `vocab_size` | 100000 | Vocabulary size cap (recommend 50000 for LCCC, lower for small corpora) |
| `min_freq` | 3 | Minimum token frequency threshold |

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `--model` | Architecture: `transformer` \| `gpt` \| `qwen` |
| `--corpora` | Corpus name(s), comma-separated (e.g. `xiaohuangji,weibo`) |
| `--fenci` | Tokenization: `jieba` (default) \| `space` |
| `--epoch` | Number of training epochs |
| `--batch` | Batch size |
| `--device` | Training device: `cuda` / `cpu` |
| `--resume` | Resume training from a checkpoint |

### Model Architecture Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 512 | Embedding / hidden dimension |
| `n_heads` | 8 | Number of attention heads |
| `n_layers` | 6 | Decoder layers |
| `d_ff` | 2048 | SwiGLU feed-forward hidden dimension |
| `dropout` | 0.1 | Dropout rate |
| `max_len` | 120 | Maximum sequence length (accommodates User+Query+Assistant+Response+EOS) |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 64 | Batch size (do not exceed 64 with vocab_size=100000) |
| `epochs` | 30 | Number of training epochs (5–10 recommended for LCCC) |
| `warmup_steps` | 4000 | Noam scheduler warmup steps |
| `label_smoothing` | 0.1 | Label smoothing factor |
| `grad_clip` | 1.0 | Gradient clipping threshold |

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beam_size` | 5 | Beam Search width |
| `temperature` | 0.8 | Sampling temperature |
| `length_penalty` | 0.6 | Length penalty (<1 favors shorter outputs) |
| `max_decode_len` | 50 | Maximum decoding length |

---

## Model Architecture

### GPT Decoder-Only (LLaMA-style)

```
Token IDs → Embedding
          → DecoderLayer × N
             ├── RMSNorm → RoPE Multi-Head Self-Attention (no bias) → Residual
             ├── RMSNorm → SwiGLU Feed-Forward Network → Residual
          → RMSNorm → Linear (shared Embedding weights) → Vocab Logits
```

GPT architecture features:
- **RMSNorm** — Root Mean Square Normalization, lighter than LayerNorm
- **RoPE** (Rotary Position Embedding) — relative position + sequence extrapolation
- **SwiGLU** — modern LLM activation, better gradient properties than ReLU
- **No bias** — all linear layers are bias-free (LLaMA-style)
- **Pre-Norm** — normalization before each sub-layer for training stability
- **Speaker Tokens** — `<|user|>` / `<|assistant|>` for dialogue role distinction
- **Full-sequence loss** — model learns the complete dialogue flow
- **KV-Cache incremental decoding** — only compute new token per step at inference

### Transformer Encoder-Decoder

```
Input → Token Embedding + Positional Encoding
      → Encoder (×N layers)
         ├── Pre-LN → Multi-Head Self-Attention → Residual
         └── Pre-LN → FeedForward → Residual
      → Decoder (×N layers)
         ├── Pre-LN → Masked Multi-Head Self-Attention → Residual
         ├── Pre-LN → Multi-Head Cross-Attention → Residual
         └── Pre-LN → FeedForward → Residual
      → Linear Projection → Softmax → Output
```

---

## Training Log Example

```
Epoch  1 | Step   100 | Loss: 4.1234 | PPL: 61.7 | LR: 0.000123 | Time: 45s
Epoch  1 | Step   200 | Loss: 3.8567 | PPL: 47.3 | LR: 0.000174 | Time: 89s
...
-------------------------------------------------------------
Epoch  1/30 | Train Loss: 3.2145 | Val Loss: 3.0123 | Val PPL: 20.3 | Time: 120s
-------------------------------------------------------------
```

---

## Dataset

### Built-in Corpora

| Corpus | Identifier | Size | Format | Recommended Tokenization |
|--------|-----------|------|--------|--------------------------|
| XiaoHuangJi | `xiaohuangji` | ~500K QA pairs | `.conv` | `--fenci jieba` |
| LCCC-base | `LCCC-base-split` | ~8.9M QA pairs | `.json` | `--fenci space` |

> LCCC (Large-scale Clean Chinese Conversation) — Chinese daily conversation dataset from Tsinghua University.

### .conv Corpus Format Specification

```
M <raw Chinese text>
M <raw Chinese text>
E
```

- Each message starts with `M `, followed by **raw Chinese text** — no manual tokenization needed
- Each conversation segment ends with a single `E` line
- Adjacent `M` messages within a segment are paired as (Query, Response)
- File encoding must be UTF-8

**Example:**

```
M 你在干嘛
M 在跟你聊天呀
E
```

### JSON Corpus Format Specification (LCCC Standard)

```json
[
  ["message 1", "message 2", "message 3"],
  ["message 1", "message 2"],
  ...
]
```

- Top-level array of conversations, each containing N ≥ 2 messages
- Messages are space-pretokenized (e.g. `"我 饿 了 。"`)
- Adjacent messages pair: (msg[0], msg[1]), (msg[2], msg[3]), ...
- File encoding UTF-8

**Example:**

```json
[
  ["你 好 呀", "你 好 你 好"],
  ["吃 了 吗", "还 没 呢", "那 一 起 呀"]
]
```

### Tokenization Engine

| --fenci | Method | Speed | Use Case |
|---------|--------|-------|----------|
| `jieba` (default) | Remove spaces/`/`, then jieba word segmentation | Slow | `.conv` raw Chinese text |
| `space` | Direct whitespace split | Fast (~20×) | `.json` LCCC & pre-tokenized corpora |

### Compatibility

Legacy `.conv` files with `/` separators are still supported — `tokenize()` strips `/` automatically before jieba segmentation. `.conv` and `.json` files can coexist; the parser auto-detects by file extension.

### Corpus Directory Layout

```
data/
└── <corpus_name>/       # folder name = corpus identifier (e.g. xiaohuangji, LCCC-base-split)
    ├── *.json            # LCCC-style JSON files
    └── *.conv            # Legacy .conv files
```

---

## Corpus Management

### Single Corpus Training

1. Place your `.conv` file in `data/<corpus_name>/`
2. Set `corpora = ("<corpus_name>",)` in [config.py](config.py), or use the CLI:
   ```bash
   python train.py --corpora <corpus_name>
   ```
3. Vocabulary and checkpoints are auto-organized under `data/` and `checkpoints/`

### Multi-Corpus Joint Training

Combine multiple corpora for broader coverage:

1. Place each corpus in its own folder under `data/`:
   ```
   data/
   ├── xiaohuangji/
   │   └── xiaohuangji50w_fenciA.conv
   └── weibo/
       └── weibo.conv
   ```
2. Specify via config or CLI:
   ```bash
   python train.py --corpora xiaohuangji,weibo
   ```
3. All dialogue pairs are merged into one training set with a unified vocabulary. Output goes to `checkpoints/xiaohuangji+weibo/`.

### Adding a New Corpus

1. Create a folder under `data/` (the folder name becomes the corpus identifier)
2. Drop in `.json` or `.conv` data files

```bash
# JSON format (recommended, LCCC standard)
mkdir -p data/mycorpus
cp /path/to/data.json data/mycorpus/
python train.py --corpora mycorpus --fenci space

# .conv format (legacy)
cp /path/to/data.conv data/mycorpus/
python train.py --corpora mycorpus --fenci jieba
```

For JSON corpora with space-pretokenized text, use `--fenci space` for significantly faster vocab building. `.conv` files require no manual tokenization — jieba handles it automatically.

---

## Dependencies

- **Python** ≥ 3.10
- **PyTorch** ≥ 2.0.0
- **NumPy**
- **tqdm** (progress bars)
- **jieba** (required for `--fenci jieba`; not needed for `--fenci space`)
- **transformers** ≥ 4.30.0 (Pro/Qwen mode only)

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., NeurIPS 2017
- [Pre-LN Transformer](https://arxiv.org/abs/2002.04745) — Xiong et al.
- [Rethinking Label Smoothing](https://arxiv.org/abs/1906.02629) — Müller et al.

---

## Corpus Download

| Corpus | Link | Notes |
|--------|------|-------|
| XiaoHuangJi | [Dialog_Corpus](https://github.com/candlewill/Dialog_Corpus) | 500K Chinese QA pairs |
| LCCC-base | [CDial-GPT](https://github.com/thu-coai/CDial-GPT) | 8.9M Chinese daily conversations (Tsinghua) |

---

## License

MIT License
