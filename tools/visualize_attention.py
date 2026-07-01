"""
Attention Heatmap Visualization

Generates attention weight heatmaps for all three models:
  - GPT        (Decoder-Only, Multi-Head Self-Attention + RoPE)
  - Transformer (Encoder-Decoder, Self-Attention + Cross-Attention)
  - Qwen2-1.5B  (Hand-written, Grouped Query Attention)

Captures attention weights via temporary forward hooks during a sample
forward pass, then plots per-head and summary heatmaps with matplotlib.

Output: tools/assets/  (PNG @ 200 DPI)

Usage:  python tools/visualize_attention.py
"""

import os
import sys
import math
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from models import GPT, Transformer, Qwen2Hand


ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# ── Colormap: white → yellow → orange → red ──
_CMAP_COLORS = ["#ffffff", "#fff9c4", "#ffb74d", "#ef6c00", "#b71c1c"]
ATTN_CMAP = LinearSegmentedColormap.from_list("attn", _CMAP_COLORS, N=256)

SAMPLE = "今天天气真好适合散步"


# ╔══════════════════════════════════════════════════════════════╗
# ║               GPT — capture via patched forward              ║
# ╚══════════════════════════════════════════════════════════════╝

def capture_gpt(model: GPT, input_ids: torch.Tensor) -> dict[str, np.ndarray]:
    """Monkey-patch GPTAttention.forward per layer, restore after forward."""
    captured: dict[str, np.ndarray] = {}
    saved = {}  # layer_idx -> original forward

    for i, layer in enumerate(model.layers):
        attn_module = layer.attn
        saved[i] = attn_module.forward

        # Capture layer_idx by factory-function default arg (no closure bug)
        def make_patched(mod, idx):
            _orig = mod.forward  # unbound

            def patched(self, x, mask=None, past_kv=None,
                        use_cache=False, cache_offset=0):
                # Replicate GPTAttention.forward exactly
                Q = self._split_heads(self.w_q(x))
                K = self._split_heads(self.w_k(x))
                V = self._split_heads(self.w_v(x))

                Q, K = self.rope(Q, K, offset=cache_offset)

                if past_kv is not None:
                    past_k, past_v = past_kv
                    K = torch.cat([past_k, K], dim=2).contiguous()
                    V = torch.cat([past_v, V], dim=2).contiguous()

                scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
                if mask is not None:
                    scores = scores.masked_fill(mask, float("-inf"))

                w = F.softmax(scores, dim=-1)
                captured[f"layer_{idx}"] = w[0].detach().cpu().numpy()

                w = self.dropout(w)
                out = self._merge_heads(torch.matmul(w, V))
                pkv = (K.detach(), V.detach()) if use_cache else None
                return (out, pkv) if use_cache else out
            return patched

        attn_module.forward = make_patched(attn_module, i).__get__(
            attn_module, type(attn_module))

    with torch.no_grad():
        try:
            model(input_ids)
        except Exception as exc:
            print(f"    (forward warning: {exc})")

    for i, layer in enumerate(model.layers):
        layer.attn.forward = saved[i]

    return captured


# ╔══════════════════════════════════════════════════════════════╗
# ║          Transformer — capture encoder + decoder attn        ║
# ╚══════════════════════════════════════════════════════════════╝

def _patched_mha_forward(self, q, k, v, mask, captured, key):
    """Shared MHA logic for Transformer encoder/decoder Self-Attn & Cross-Attn."""
    B, sq, dm = q.shape
    n_h, d_k = self.n_heads, self.d_k

    Q = self._split_heads(self.w_q(q))
    K = self._split_heads(self.w_k(k))
    V = self._split_heads(self.w_v(v))

    scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
    if mask is not None:
        ssz = min(scores.size(-1), mask.size(-1))
        scores = scores.masked_fill(mask[:, :, :sq, :ssz], float("-inf"))

    w = F.softmax(scores, dim=-1)
    captured[key] = w[0].detach().cpu().numpy()

    w = self.dropout(w)
    out = self._merge_heads(torch.matmul(w, V))
    return self.w_o(out)


def capture_transformer(model: Transformer, src_ids: torch.Tensor,
                        tgt_ids: torch.Tensor) -> dict[str, np.ndarray]:
    captured: dict[str, np.ndarray] = {}
    saved_enc, saved_dec_self, saved_dec_cross = {}, {}, {}

    # Patch encoder self-attn
    for i, layer in enumerate(model.encoder.layers):
        saved_enc[i] = layer.self_attn.forward
        def _enc(self, q, k, v, mask=None, past_kv=None, use_cache=False, _i=i):
            return _patched_mha_forward(self, q, k, v, mask, captured, f"enc_self_L{_i}")
        layer.self_attn.forward = _enc.__get__(layer.self_attn, type(layer.self_attn))

    # Patch decoder self-attn + cross-attn
    for i, layer in enumerate(model.decoder.layers):
        saved_dec_self[i] = layer.self_attn.forward
        saved_dec_cross[i] = layer.cross_attn.forward
        def _dself(self, q, k, v, mask=None, past_kv=None, use_cache=False, _i=i):
            return _patched_mha_forward(self, q, k, v, mask, captured, f"dec_self_L{_i}")
        def _dcross(self, q, k, v, mask=None, past_kv=None, use_cache=False, _i=i):
            return _patched_mha_forward(self, q, k, v, mask, captured, f"dec_cross_L{_i}")
        layer.self_attn.forward = _dself.__get__(layer.self_attn, type(layer.self_attn))
        layer.cross_attn.forward = _dcross.__get__(layer.cross_attn, type(layer.cross_attn))

    with torch.no_grad():
        try:
            sp = torch.zeros_like(src_ids, dtype=torch.bool)
            tp = torch.zeros_like(tgt_ids, dtype=torch.bool)
            model(src_ids, tgt_ids, sp, tp)
        except Exception as exc:
            print(f"    (forward warning: {exc})")

    for i, layer in enumerate(model.encoder.layers):
        layer.self_attn.forward = saved_enc[i]
    for i, layer in enumerate(model.decoder.layers):
        layer.self_attn.forward = saved_dec_self[i]
        layer.cross_attn.forward = saved_dec_cross[i]

    return captured


# ╔══════════════════════════════════════════════════════════════╗
# ║         Qwen2Hand — capture GQA attention                    ║
# ╚══════════════════════════════════════════════════════════════╝

def capture_qwen2(model: Qwen2Hand, input_ids: torch.Tensor) -> dict[str, np.ndarray]:
    captured: dict[str, np.ndarray] = {}
    saved = {}

    for i, block in enumerate(model.layers):
        attn_module = block.self_attn
        saved[i] = attn_module.forward

        def make_patched(mod, idx):
            def patched(self, x, mask=None, past_kv=None,
                        use_cache=False, cache_offset=0):
                B, S, _ = x.shape
                Q = self._reshape_q(self.q_proj(x))
                K = self._reshape_kv(self.k_proj(x))
                V = self._reshape_kv(self.v_proj(x))

                Q, K = self.rope(Q, K, offset=cache_offset)

                if past_kv is not None:
                    pk, pv = past_kv
                    K = torch.cat([pk, K], dim=2).contiguous()
                    V = torch.cat([pv, V], dim=2).contiguous()

                K_rep = K.repeat_interleave(self.n_rep, dim=1)
                V_rep = V.repeat_interleave(self.n_rep, dim=1)

                scores = torch.matmul(Q, K_rep.transpose(-2, -1)) / self.scale
                if mask is not None:
                    ks = K_rep.size(2)
                    scores = scores.masked_fill(mask[:, :, :S, :ks], float("-inf"))

                w = F.softmax(scores, dim=-1)
                captured[f"layer_{idx}"] = w[0].detach().cpu().numpy()

                w = self.dropout(w)
                out = torch.matmul(w, V_rep).permute(0, 2, 1, 3).contiguous().view(B, S, -1)
                out = self.o_proj(out)

                pkv = (K.detach(), V.detach()) if use_cache else None
                return (out, pkv) if use_cache else out
            return patched

        attn_module.forward = make_patched(attn_module, i).__get__(
            attn_module, type(attn_module))

    with torch.no_grad():
        try:
            model(input_ids)
        except Exception as exc:
            print(f"    (forward warning: {exc})")

    for i, block in enumerate(model.layers):
        block.self_attn.forward = saved[i]

    return captured


# ╔══════════════════════════════════════════════════════════════╗
# ║                    Plotting                                  ║
# ╚══════════════════════════════════════════════════════════════╝

def plot_head_grid(weights: dict[str, np.ndarray], title: str, fname: str,
                   n_cols: int = 8):
    """Per-head heatmap grid: rows=layers, cols=heads."""
    if not weights:
        print(f"    [skip] {fname}: empty")
        return

    items = sorted(weights.items(), key=lambda x: int(x[0].rsplit("_", 1)[-1].lstrip("L")))
    n_layers = len(items)
    n_heads = items[0][1].shape[0]
    disp_heads = min(n_heads, n_cols)

    fig, axes = plt.subplots(n_layers, disp_heads,
                              figsize=(disp_heads * 2.0, n_layers * 1.8),
                              squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    for row, (name, w) in enumerate(items):
        label = name.rsplit("_", 1)[-1]
        for col in range(disp_heads):
            ax = axes[row][col]
            hw = w[col]
            im = ax.imshow(hw, cmap=ATTN_CMAP, aspect="auto", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"H{col}", fontsize=7, pad=1)
            if col == 0:
                ax.set_ylabel(f"L{label}", fontsize=9, rotation=0, labelpad=15, va="center")

    cax = fig.add_axes([0.92, 0.06, 0.015, 0.88])
    fig.colorbar(im, cax=cax).set_label("Weight", fontsize=8)
    plt.subplots_adjust(left=0.06, right=0.90, top=0.94, bottom=0.03,
                         wspace=0.06, hspace=0.12)
    path = os.path.join(ASSETS_DIR, fname)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    -> {fname}  ({os.path.getsize(path)//1024} KB, {n_layers}L x {disp_heads}H)")


def plot_summary(weights: dict[str, np.ndarray], title: str, fname: str):
    """Single heatmap: mean over all heads, stacked layers."""
    if not weights:
        print(f"    [skip] {fname}: empty")
        return

    items = sorted(weights.items(), key=lambda x: int(x[0].rsplit("_", 1)[-1].lstrip("L")))
    n_layers = len(items)
    seq_len = items[0][1].shape[1]

    # Average over heads
    stacked = np.concatenate([w.mean(axis=0) for _, w in items], axis=0)  # (nL*seq, seq)

    fig, ax = plt.subplots(figsize=(max(6, seq_len * 0.4), max(3, n_layers * 0.5)))
    im = ax.imshow(stacked, cmap=ATTN_CMAP, aspect="auto", vmin=0, vmax=1)

    for i in range(1, n_layers):
        ax.axhline(i * seq_len - 0.5, color="#333", lw=0.6, ls="--")

    # Layer labels
    labels = [f"L{name.rsplit('_',1)[-1]}" for name, _ in items]
    ax.set_yticks([(i + 0.5) * seq_len for i in range(n_layers)])
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels([str(i) for i in range(seq_len)], fontsize=6)
    ax.set_xlabel("Key Position", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.7).set_label("Mean Weight", fontsize=8)
    path = os.path.join(ASSETS_DIR, fname)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    -> {fname}  ({os.path.getsize(path)//1024} KB)")


# ╔══════════════════════════════════════════════════════════════╗
# ║                         Main                                 ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Sample: '{SAMPLE}' ({len(SAMPLE)} chars)\n")

    seq = min(len(SAMPLE), config.max_len)

    # ═══════════ 1. GPT ═══════════
    print("[1/3] GPT  (Decoder-Only, MHA + RoPE)")
    gpt = GPT(config).to(device).eval()
    ids = torch.randint(0, config.vocab_size, (1, seq), device=device)
    w = capture_gpt(gpt, ids)
    print(f"      captured {len(w)} layers")
    plot_head_grid(w, "GPT — Multi-Head Self-Attention per Layer",
                   "heatmap_gpt_heads.png")
    plot_summary(w, "GPT — Mean Attention (averaged over heads)",
                 "heatmap_gpt_summary.png")
    del gpt

    # ═══════════ 2. Transformer ═══════════
    print("\n[2/3] Transformer  (Encoder-Decoder)")
    tf = Transformer(config).to(device).eval()
    src = torch.randint(0, config.vocab_size, (1, seq), device=device)
    tgt = torch.randint(0, config.vocab_size, (1, seq), device=device)
    w = capture_transformer(tf, src, tgt)
    print(f"      captured {len(w)} entries")

    for prefix, title, fname in [
        ("enc_self",  "Transformer Encoder — Self-Attention",
         "heatmap_transformer_enc_self"),
        ("dec_self",  "Transformer Decoder — Causal Self-Attention",
         "heatmap_transformer_dec_self"),
        ("dec_cross", "Transformer Decoder — Cross-Attention (src->tgt)",
         "heatmap_transformer_dec_cross"),
    ]:
        subset = {k: v for k, v in w.items() if k.startswith(prefix)}
        if subset:
            plot_head_grid(subset, title, f"{fname}_heads.png")
            plot_summary(subset, f"{title} (mean)", f"{fname}_summary.png")
    del tf

    # ═══════════ 3. Qwen2-1.5B Hand ═══════════
    print("\n[3/3] Qwen2-1.5B Hand-written  (GQA: 8Q / 2KV heads)")
    qw = Qwen2Hand(d_model=256, n_heads=8, n_kv_heads=2, n_layers=4,
                   d_ff=512, vocab_size=config.vocab_size,
                   max_len=config.max_len).to(device).eval()
    qw_ids = torch.randint(0, config.vocab_size, (1, min(seq, qw.max_len)), device=device)
    w = capture_qwen2(qw, qw_ids)
    print(f"      captured {len(w)} layers (GQA: 8 Q-heads, 2 KV-heads)")
    plot_head_grid(w, "Qwen2-1.5B Hand — GQA Self-Attention (8 Q-heads, 2 KV shared)",
                   "heatmap_qwen2_hand_heads.png")
    plot_summary(w, "Qwen2-1.5B Hand — Mean GQA Attention",
                 "heatmap_qwen2_hand_summary.png")
    del qw

    print(f"\nDone -> {ASSETS_DIR}/")


if __name__ == "__main__":
    main()
