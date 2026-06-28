"""
Decoder-Only GPT 模型 — 支持多轮对话记忆

与 model.py (Encoder-Decoder) 的区别:
  - 无 Encoder，无 Cross-Attention
  - 历史对话直接拼入输入序列
  - 推理时 KV-Cache 可跨多轮复用

复用 model.py 中的 PositionalEncoding / FeedForward / MultiHeadAttention / _prepare_sdpa_mask
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config, PAD_ID
from model import (
    PositionalEncoding,
    FeedForward,
    MultiHeadAttention,
    _prepare_sdpa_mask,
)


# ============================================================
#  Decoder-Only Layer
# ============================================================

class DecoderOnlyLayer(nn.Module):
    """单层 GPT Decoder: Masked Self-Attention → FFN，Pre-LN 结构。"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        """
        x: (B, seq_len, d_model)
        mask: (B, 1, seq_len, seq_len) — causal + padding, True = 遮掩
        past_kv: 可选的 (K, V) 缓存
        返回: x 或 (x, new_kv)
        """
        # Self-Attention (Pre-LN)
        residual = x
        result = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                                mask, past_kv, use_cache)
        if use_cache:
            x_attn, new_kv = result
        else:
            x_attn = result
        x = residual + self.dropout(x_attn)

        # FFN (Pre-LN)
        x = x + self.dropout(self.ffn(self.norm2(x)))

        if use_cache:
            return x, new_kv
        return x


# ============================================================
#  GPT 模型
# ============================================================

class GPT(nn.Module):
    """
    Decoder-Only GPT 模型，用于中文对话生成。

    架构:
        Embedding + PositionalEncoding
        → DecoderOnlyLayer × N
        → LayerNorm
        → Linear → Vocab

    特性:
      - Pre-LN 结构
      - 嵌入/输出权重共享
      - KV-Cache 增量解码
      - 因果 mask 自回归生成
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=PAD_ID)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_len, config.dropout)
        self.layers = nn.ModuleList([
            DecoderOnlyLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)

        # 权重共享
        self.output_proj.weight = self.embedding.weight

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ---------- Mask 工具 ----------

    @staticmethod
    def make_padding_mask(pad_mask: torch.Tensor) -> torch.Tensor:
        """(B, seq_len) bool → (B, 1, 1, seq_len)"""
        return pad_mask.unsqueeze(1).unsqueeze(2)

    @staticmethod
    def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """上三角 causal mask, (1, 1, seq_len, seq_len)"""
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    # ---------- 训练前向 ----------

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (B, seq_len) — token IDs
        pad_mask: (B, seq_len) — True = padding
        返回: (B, seq_len, vocab_size) — logits
        """
        B, seq_len = x.shape

        # Causal + padding mask
        causal = self.make_causal_mask(seq_len, x.device)       # (1, 1, seq, seq)
        if pad_mask is not None:
            padding = self.make_padding_mask(pad_mask)           # (B, 1, 1, seq)
            mask = causal | padding                              # (B, 1, seq, seq)
        else:
            mask = causal

        h = self.embedding(x) * math.sqrt(self.d_model)
        h = self.pos_encoding(h)

        for layer in self.layers:
            h = layer(h, mask)

        h = self.norm(h)
        return self.output_proj(h)  # (B, seq_len, vocab_size)

    # ---------- 增量解码（KV-Cache）----------

    def decode_step(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        past_key_values: list | None = None,
        use_cache: bool = False,
    ):
        """
        单步解码（支持 KV-Cache）。

        x: (B, seq_len) — 当前序列（首次为完整 prompt，后续为单 token）
        pad_mask: (B, seq_len) — True = padding
        past_key_values: 每层的 (K, V) 缓存列表
        use_cache: 是否返回更新后的缓存

        返回: logits 或 (logits, new_cache)
        """
        seq_len = x.size(1)

        # Causal + padding mask
        causal = self.make_causal_mask(seq_len, x.device)
        if pad_mask is not None:
            padding = self.make_padding_mask(pad_mask)
            mask = causal | padding
        else:
            mask = causal

        h = self.embedding(x) * math.sqrt(self.d_model)
        h = self.pos_encoding(h)

        new_cache: list = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            result = layer(h, mask, past_kv, use_cache)
            if use_cache:
                h, layer_kv = result
                new_cache.append(layer_kv)
            else:
                h = result

        h = self.norm(h)
        logits = self.output_proj(h)

        if use_cache:
            return logits, new_cache
        return logits


# ============================================================
#  模型测试
# ============================================================

if __name__ == "__main__":
    config = Config()
    config.vocab_size = 5000  # 测试用
    model = GPT(config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"GPT 模型参数量: {total_params:,}")

    # 测试训练前向
    B, seq_len = 4, 50
    x = torch.randint(4, config.vocab_size, (B, seq_len))
    pad = torch.zeros(B, seq_len, dtype=torch.bool)
    pad[:, -5:] = True

    logits = model(x, pad)
    print(f"训练 forward: {logits.shape}")  # (4, 50, 5000)
    assert logits.shape == (B, seq_len, config.vocab_size)

    # 测试 KV-Cache 增量解码
    prompt = torch.randint(4, config.vocab_size, (1, 10))
    ppad = torch.zeros(1, 10, dtype=torch.bool)

    # 首步：缓存全量
    logits1, cache = model.decode_step(prompt, ppad, None, use_cache=True)
    print(f"首步 (prompt=10): logits={logits1.shape}, cache_layers={len(cache)}")
    assert logits1.shape == (1, 10, config.vocab_size)

    # 后续步：只传新 token
    new_tok = torch.randint(4, config.vocab_size, (1, 1))
    npad = torch.zeros(1, 1, dtype=torch.bool)
    logits2, cache2 = model.decode_step(new_tok, npad, cache, use_cache=True)
    print(f"第二步 (1 tok + cache): logits={logits2.shape}")
    assert logits2.shape == (1, 1, config.vocab_size)

    print("\n✓ GPT 模型测试全部通过！")
