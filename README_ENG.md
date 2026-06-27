# Transformer Chinese Chatbot

A Transformer chatbot built from scratch in PyTorch, trained on the XiaoHuangJi (小黄鸡) Chinese conversation dataset. Based on [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

[中文文档](README_CN.md)

---

## Features

- **Transformer from scratch** — Encoder/Decoder/Multi-Head Attention implemented by hand, zero reliance on `torch.nn.Transformer`
- **Pre-LN architecture** — Pre-LayerNorm for more stable training
- **Weight tying** — Shared weights across Encoder embedding, Decoder embedding, and output projection
- **Noam scheduler** — Learning rate warmup strategy from the original paper
- **Label smoothing** — Memory-efficient cross-entropy with label smoothing
- **Mixed precision training** — AMP support to reduce GPU memory usage
- **Multiple decoding strategies** — Beam Search / Greedy / Temperature Sampling (Top-K + Top-P)
- **Multi-corpus support** — Train on single or multiple corpora with one-line config switch; results auto-organized by corpus name
- **CLI-driven training** — Start training with `--corpora`, `--epoch`, `--batch` flags — no need to edit config files

---

## Project Structure

```
.
├── model.py           # Transformer model (Encoder/Decoder/Attention/FFN)
├── config.py          # Hyperparameters + corpus selection
├── data_loader.py     # Data preprocessing, vocabulary builder, DataLoader
├── train.py           # Training script (with CLI argument support)
├── inference.py       # Inference & interactive chat (Beam Search / Sampling)
├── requirements.txt   # Dependencies
│
├── data/              # Corpus directory (one subfolder per corpus)
│   ├── xiaohuangji/
│   │   ├── xiaohuangji50w_fenciA.conv
│   │   └── vocab.json
│   └── xiaohuangji+weibo/     ← auto-created for multi-corpus
│       └── vocab.json         ← merged vocabulary
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
# Train with default config
python train.py

# Specify corpus
python train.py --corpora xiaohuangji

# Multi-corpus joint training
python train.py --corpora xiaohuangji,weibo

# Custom training parameters
python train.py --corpora xiaohuangji --epoch 50 --batch 64

# Show all CLI options
python train.py --help
```

The training pipeline will:
- Parse the conversation corpus and build a vocabulary (saved to `data/<corpus>/vocab.json`)
- Train with Noam scheduler + label smoothing
- Validate after each epoch and save the best model to `checkpoints/<corpus>/best_model.pt`
- Record training history in `checkpoints/<corpus>/history.json`

### 3. Run Inference

After training, the model is saved at `checkpoints/<corpus>/best_model.pt`, with the vocabulary at `data/<corpus>/vocab.json`.

#### Interactive Chat

```bash
python inference.py
```

The script auto-loads the model specified by `corpora` in `config.py`. To switch corpora, change `corpora` in the config.

Chat commands:

| Command | Description |
|---------|-------------|
| `/beam` | Switch to Beam Search decoding (best quality, default) |
| `/sample` | Switch to temperature sampling (more diverse) |
| `/greedy` | Switch to greedy decoding (fastest) |
| `quit` / `exit` | Exit |

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

### Corpus Selection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `corpora` | `("xiaohuangji",)` | Corpus names (folders under `data/`). Use `("a", "b")` for joint training |

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `--corpora` | Corpus name(s), comma-separated for multi-corpus (e.g. `xiaohuangji,weibo`) |
| `--epoch` | Number of training epochs |
| `--batch` | Batch size |
| `--device` | Training device: `cuda` / `cpu` |
| `--resume` | Resume training from a checkpoint |

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 128 | Embedding / hidden dimension |
| `n_heads` | 8 | Number of attention heads |
| `n_layers` | 6 | Number of Encoder / Decoder layers |
| `d_ff` | 2048 | Feed-forward hidden dimension |
| `dropout` | 0.1 | Dropout rate |
| `max_len` | 60 | Maximum sequence length |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 128 | Batch size |
| `epochs` | 30 | Number of training epochs |
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

### Key Implementation Details

- **Positional Encoding**: Sinusoidal encoding, pre-computed and registered as a buffer
- **Multi-Head Attention**: Custom `_split_heads` / `_merge_heads`, supporting causal and padding masks
- **Pre-LN**: LayerNorm before each sub-layer — more stable than Post-LN
- **Label Smoothing Loss**: Mathematically expanded to avoid allocating `(N, V)`-sized tensors for memory efficiency

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

Trained on the XiaoHuangJi (小黄鸡) conversation corpus (`xiaohuangji50w_fenciA.conv`), containing ~500k Chinese dialogue pairs.

**Data format** — Each conversation segment ends with `E`, messages are prefixed with `M `, and tokens within a message are separated by `/`:

```
M 你/在/干嘛
M 在/跟/你/聊天/呀
E
M 今天/天气/怎么样
M 很/好/呀
E
```

Adjacent `M` messages are paired as Query-Response dialogue pairs.

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

Just create a folder under `data/` and drop in your `.conv` file. Required format:
- Conversation segments end with `E`
- Messages are prefixed with `M `
- Words are separated by `/`

The filename can be anything — the folder name becomes the corpus identifier.

---

## Dependencies

- **Python** ≥ 3.10
- **PyTorch** ≥ 2.0.0
- **NumPy**
- **tqdm**

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., NeurIPS 2017
- [Pre-LN Transformer](https://arxiv.org/abs/2002.04745) — Xiong et al.
- [Rethinking Label Smoothing](https://arxiv.org/abs/1906.02629) — Müller et al.

---

## Corpus Download

[https://github.com/candlewill/Dialog_Corpus](https://github.com/candlewill/Dialog_Corpus)

---

## License

MIT License
