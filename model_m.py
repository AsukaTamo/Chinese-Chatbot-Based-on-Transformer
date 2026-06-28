"""
Decoder-Only GPT 模型 — 基于 RoPE 旋转位置编码

与 model_gpt.py (正弦位置编码) 的区别:
  - 使用 RoPE 替代正弦位置编码，位置信息直接注入 Q/K 向量
  - RoPE 天然支持相对位置建模和序列长度外推

与 model_gpt.GPT 保持统一接口，可互换使用:
  - GPT(config)          — 构造
  - forward(x, pad_mask) — 训练前向
  - decode_step(...)     — 增量解码 (KV-Cache)
  - make_padding_mask / make_causal_mask

复用 model.py 中的 MultiHeadAttention / FeedForward / _prepare_sdpa_mask，
仅 RotaryEmbedding + CausalSelfAttention.forward 中的 RoPE 注入为新增逻辑。
"""

import math

import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention as sdpa

from config import Config, PAD_ID
from model import FeedForward, MultiHeadAttention, _prepare_sdpa_mask


# ============================================================
#  RoPE — 旋转位置编码
# ============================================================

class RotaryEmbedding(nn.Module):
    """
    RoPE (Rotary Position Embedding) 旋转位置编码。

    原理:
      将 Q/K 向量的每两个相邻维度视为一个 2D 平面，
      按该维度对应的频率施加旋转，旋转角度与 token 绝对位置成正比。
      这样 QᵀK 中自动包含 (m-n)·θ 的相对位置信息。

    优势:
      1. 相对位置 — 注意力分数仅依赖 token 间的相对距离，
         而非绝对位置，更符合语言的平移不变性。
      2. 长度外推 — 旋转频率连续可外推；训练时未见过的更长序列
         在推理时也能较好泛化，无需重新训练。
      3. 零额外参数 — 位置编码完全由数学公式生成，
         不引入可学习参数，减小过拟合风险。
      4. 自然衰减 — 高频分量随距离增加快速衰减，
         低频分量保持长程依赖，形成「近大远小」的注意力模式。
      5. 适配 KV-Cache — 旋转按绝对位置施加，
         缓存的 K 稍后与新 Q 点积时仍保持正确相对关系。
    """

    def __init__(self, head_dim: int, max_len: int = 5000, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_len = max_len

        # 频率: 低维 → 高频(短周期), 高维 → 低频(长周期)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)                     # (max_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)              # (max_len, head_dim)
        self.register_buffer("cos_cached", emb.cos())        # cos(m·θ_i)
        self.register_buffer("sin_cached", emb.sin())        # sin(m·θ_i)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q, k: (B, n_heads, seq_len, head_dim)
        offset: KV-Cache 中已有的历史 token 数
        返回: 旋转后的 (q, k)
        """
        seq_len = q.size(2)
        positions = torch.arange(offset, offset + seq_len, device=q.device)
        cos = self.cos_cached[positions].unsqueeze(0).unsqueeze(0)   # (1, 1, T, hd)
        sin = self.sin_cached[positions].unsqueeze(0).unsqueeze(0)
        q_out = q * cos + self._rotate_half(q) * sin
        k_out = k * cos + self._rotate_half(k) * sin
        return q_out, k_out

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """将向量后半取反、与前半互换 → 2D 平面上的 90° 旋转算子。"""
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)


# ============================================================
#  多头自注意力 — 继承 model.MultiHeadAttention, 注入 RoPE
# ============================================================

class CausalSelfAttention(MultiHeadAttention):
    """
    因果自注意力，继承 model.MultiHeadAttention，在拆头后注入 RoPE。

    父类提供: 投影层 (w_q/k/v/o)、拆/合头 (_split_heads/_merge_heads)、KV-Cache、SDPA
    子类覆写: forward —— 在 Q/K 拆头后、KV-Cache 拼接前插入 RoPE 旋转

    RoPE 流程:
      x → w_q/w_k/w_v → _split_heads → rope(Q,K) → (past_kv cat) → SDPA → _merge_heads → w_o
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__(d_model, n_heads, dropout)
        self.rope = RotaryEmbedding(self.d_k)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        """
        x: (B, seq_len, d_model) — 自注意力 Q=K=V=x
        mask/past_kv/use_cache 与父类一致

        返回: output (B, seq_len, d_model) 或 (output, present_kv)
        """
        # ---- 投影 (复用父类层) ----
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # ---- 拆多头 (复用父类方法) ----
        Q = self._split_heads(Q)   # (B, n_heads, T, d_k)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # ---- ★ RoPE: model_m 相对于 model 唯一新增的步骤 ----
        offset = past_kv[0].size(2) if past_kv is not None else 0
        Q, K = self.rope(Q, K, offset=offset)

        # ---- KV-Cache ----
        if past_kv is not None:
            past_k, past_v = past_kv
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)
        present_kv = (K.detach(), V.detach()) if use_cache else None

        # ---- SDPA ----
        attn_mask = _prepare_sdpa_mask(mask, Q.dtype)
        dropout_p = self.dropout.p if self.training else 0.0
        y = sdpa(Q, K, V, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False)

        # ---- 合头 + 输出投影 (复用父类方法) ----
        y = self._merge_heads(y)
        y = self.w_o(y)

        if use_cache:
            return y, present_kv
        return y


# ============================================================
#  Decoder-Only Layer
# ============================================================

class DecoderOnlyLayer(nn.Module):
    """单层 GPT Decoder: RoPE Self-Attention → FeedForward (ReLU)，Pre-LN 残差结构。"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: tuple | None = None,
        use_cache: bool = False,
    ):
        # Self-Attention (Pre-LN)
        residual = x
        result = self.self_attn(self.norm1(x), mask, past_kv, use_cache)
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
#  GPT 模型 (RoPE 版)
# ============================================================

class GPT(nn.Module):
    """
    Decoder-Only GPT 模型 — RoPE 旋转位置编码。

    架构:
        Embedding → DecoderOnlyLayer × N → LayerNorm → Linear → Vocab

    特性:
      - RoPE 旋转位置编码（相对位置 + 长度外推）
      - KV-Cache 增量解码
      - 嵌入 / 输出投影权重共享
      - FeedForward / _prepare_sdpa_mask 复用 model.py

    与 model_gpt.GPT 接口完全一致，可在 inference.py 中直接替换导入。
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        # 注意: RoPE 不在此处定义——它内嵌在每个 attention head 中
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=PAD_ID)
        self.layers = nn.ModuleList([
            DecoderOnlyLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 权重共享：嵌入 ←→ 输出投影
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
        causal = self.make_causal_mask(seq_len, x.device)          # (1, 1, seq, seq)
        if pad_mask is not None:
            padding = self.make_padding_mask(pad_mask)              # (B, 1, 1, seq)
            mask = causal | padding                                 # (B, 1, seq, seq)
        else:
            mask = causal

        h = self.embedding(x) * math.sqrt(self.d_model)            # (B, seq, d_model)

        for layer in self.layers:
            h = layer(h, mask)

        h = self.norm(h)
        return self.output_proj(h)                                  # (B, seq, vocab_size)

    # ---------- 增量解码 (KV-Cache) ----------

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
        past_key_values: 每层的 (K, V) 缓存列表，首次传 None
        use_cache: 是否返回更新后的缓存

        返回: logits 或 (logits, new_cache)
        """
        seq_len = x.size(1)

        # 构建 attention mask
        causal = self.make_causal_mask(seq_len, x.device)          # (1, 1, seq, seq)
        if pad_mask is not None:
            padding = self.make_padding_mask(pad_mask)
            cur_mask = causal | padding                             # (B, 1, seq, seq)
        else:
            cur_mask = causal

        # 若有缓存，在 K 维度左侧补齐（允许当前 Q 关注全部历史 K）
        if past_key_values is not None and past_key_values[0] is not None:
            cache_len = past_key_values[0][0].size(2)
            history_pad = torch.zeros(1, 1, seq_len, cache_len,
                                      dtype=torch.bool, device=x.device)
            cur_mask = torch.cat([history_pad, cur_mask], dim=-1)  # (B, 1, seq, cache_len+seq)

        h = self.embedding(x) * math.sqrt(self.d_model)

        new_cache: list = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            result = layer(h, cur_mask, past_kv, use_cache)
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
    print(f"GPT (RoPE) 模型参数量: {total_params:,}")

    # ---- 测试训练前向 ----
    B, seq_len = 4, 50
    x = torch.randint(4, config.vocab_size, (B, seq_len))
    pad = torch.zeros(B, seq_len, dtype=torch.bool)
    pad[:, -5:] = True

    logits = model(x, pad)
    print(f"训练 forward: {logits.shape}")          # (4, 50, 5000)
    assert logits.shape == (B, seq_len, config.vocab_size)

    # ---- 测试 KV-Cache 增量解码 ----
    prompt = torch.randint(4, config.vocab_size, (1, 10))
    ppad = torch.zeros(1, 10, dtype=torch.bool)

    logits1, cache = model.decode_step(prompt, ppad, None, use_cache=True)
    print(f"首步 (prompt=10): logits={logits1.shape}, cache_layers={len(cache)}")
    assert logits1.shape == (1, 10, config.vocab_size)

    new_tok = torch.randint(4, config.vocab_size, (1, 1))
    npad = torch.zeros(1, 1, dtype=torch.bool)
    logits2, cache2 = model.decode_step(new_tok, npad, cache, use_cache=True)
    print(f"第二步 (1 tok + cache): logits={logits2.shape}")
    assert logits2.shape == (1, 1, config.vocab_size)

    # ---- 对比无 cache 的输出一致性 ----
    full = torch.cat([prompt, new_tok], dim=1)                # (1, 11)
    fpad = torch.zeros(1, 11, dtype=torch.bool)
    logits_full = model(full, fpad)                           # (1, 11, vocab)
    diff = (logits2[0, -1] - logits_full[0, -1]).abs().max().item()
    print(f"KV-Cache vs 全量推理 最大差异: {diff:.6f}")

    print("\n[OK] GPT (RoPE) 模型测试全部通过！")
