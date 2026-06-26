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

---

## Project Structure

```
.
├── model.py           # Transformer model (Encoder/Decoder/Attention/FFN)
├── config.py          # Hyperparameters (model / training / inference)
├── data_loader.py     # Data preprocessing, vocabulary builder, DataLoader
├── train.py           # Training script (Noam scheduler + label smoothing loss)
├── inference.py       # Inference & interactive chat (Beam Search / Sampling)
├── vocab.json         # Vocabulary file (auto-generated)
├── checkpoints/       # Model checkpoint directory
│   ├── best_model.pt
│   └── history.json
├── xiaohuangji50w_fenciA.conv  # XiaoHuangJi conversation corpus (500k pairs)
└── requirements.txt   # Dependencies
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
python train.py
```

The training pipeline will:
- Parse the conversation corpus and build a vocabulary (saved as `vocab.json`)
- Train with Noam scheduler + label smoothing
- Validate after each epoch and save the best model to `checkpoints/best_model.pt`
- Record training history in `checkpoints/history.json`

### 3. Interactive Chat

```bash
python inference.py
```

Chat commands:

| Command | Description |
|---------|-------------|
| `/beam` | Switch to Beam Search decoding |
| `/sample` | Switch to temperature sampling |
| `/greedy` | Switch to greedy decoding |
| `quit` / `exit` | Exit |

---

## Configuration

Adjust hyperparameters in [config.py](config.py):

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

## Custom Dataset

Replace `xiaohuangji50w_fenciA.conv` with your own conversation data in the same format. Update `data_path` in [config.py](config.py) to point to your file.

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
