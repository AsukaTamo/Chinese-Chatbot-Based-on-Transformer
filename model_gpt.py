"""
Decoder-Only GPT 模型 (LLaMA-style) — 面向中文日常对话

相比初版的改进:
  - RMSNorm 替代 LayerNorm         → 更轻量，训练更稳定
  - SwiGLU FFN 替代 ReLU FFN       → 现代 LLM 标配，更好梯度特性
  - RoPE 旋转位置编码               → 相对位置感知 + 序列长度外推
  - 无 bias 线性层 (LLaMA 风格)     → 参数更高效
  - Speaker Token 区分对话角色      → <|user|> / <|assistant|>
  - 全序列 loss                     → 模型学习完整对话流
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config, PAD_ID, USER_ID, ASSISTANT_ID, EOS_ID


# ============================================================
#  RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA 风格，无 bias)。"""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ============================================================
#  RoPE (Rotary Position Embedding)
# ============================================================

class RoPE(nn.Module):
    """
    旋转位置编码 — 对 Q/K 的每对相邻维度施加 2D 旋转。

    公式: f{q,k}(x_m, m) = R_m · W_{q,k} · x_m
    其中 R_m 是按位置 m 旋转角度 m·θ_i 的旋转矩阵。

    频率: θ_i = base^(-2i/d),  i ∈ [0, d/2)

    预计算 cos/sin 表用于快速查表；当位置超出表范围（如增量解码超过
    config.max_len）时自动降级为动态计算，不影响推理正确性。
    """

    def __init__(self, d_k: int, max_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.d_k = d_k
        self.max_len = max_len
        self.base = base

        # 预计算 cos(m·θ_i), sin(m·θ_i) 表: (max_len, d_k/2)
        theta = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        positions = torch.arange(max_len).float()
        freqs = torch.outer(positions, theta)                # (max_len, d_k/2)
        self.register_buffer("cos_table", freqs.cos())       # (max_len, d_k/2)
        self.register_buffer("sin_table", freqs.sin())

    def _get_rotations(
        self, offset: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """获取位置 [offset, offset+seq_len) 的 cos/sin 旋转量 (1, 1, seq_len, d_k/2)。"""
        max_needed = offset + seq_len
        table_len = self.cos_table.size(0)

        if max_needed <= table_len:
            cos = self.cos_table[offset:max_needed, :].to(dtype)
            sin = self.sin_table[offset:max_needed, :].to(dtype)
        else:
            # 位置超出预计算表 → 动态计算（增量解码超过训练 max_len 时触发）
            theta = 1.0 / (self.base ** (torch.arange(0, self.d_k, 2, device=device).float() / self.d_k))
            positions = torch.arange(offset, max_needed, device=device).float()
            freqs = torch.outer(positions, theta)
            cos = freqs.cos().to(dtype)
            sin = freqs.sin().to(dtype)

        return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q, k: (B, n_heads, seq_len, d_k)
        offset: KV-Cache 已有长度（增量解码时使用）
        """
        cos, sin = self._get_rotations(offset, q.size(2), q.device, q.dtype)
        q_rot = _rotate_half(q, cos, sin)
        k_rot = _rotate_half(k, cos, sin)
        return q_rot, k_rot


def _rotate_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对 x 的每对相邻维度做 2D 旋转: (x1,x2) → (x1·cos - x2·sin, x1·sin + x2·cos)"""
    d = x.size(-1)
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ============================================================
#  SwiGLU Feed-Forward
# ============================================================

class SwiGLUFFN(nn.Module):
    """
    SwiGLU 前馈网络 (LLaMA 风格)。

    SwiGLU(x) = (SiLU(x·W_gate) ⊙ x·W_up) · W_down
    合并 gate/up 为一次矩阵乘以提高效率。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_gate_up = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.w_gate_up(x)                # (B, seq, 2*d_ff)
        gate, up = gate_up.chunk(2, dim=-1)       # 各 (B, seq, d_ff)
        return self.dropout(self.w_down(F.silu(gate) * up))


# ============================================================
#  RoPE-aware Multi-Head Attention
# ============================================================

class GPTAttention(nn.Module):
    """多头自注意力 + RoPE + KV-Cache（无 bias）。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rope = RoPE(self.d_k, max_len)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, seq, _ = x.shape
        x = x.view(B, seq, self.n_heads, self.d_k)
        return x.permute(0, 2, 1, 3).contiguous()       # (B, n_heads, seq, d_k)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, _, seq, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, seq, self.d_model)              # (B, seq, d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        cache_offset: int = 0,
    ):
        """
        x: (B, seq_len, d_model)
        mask: (B, 1, q_len, k_len) — True = 遮掩
        past_kv: (past_K, past_V) 用于增量解码
        cache_offset: KV-Cache 中已有的 token 数（RoPE 位置偏移）
        返回: output 或 (output, present_kv)
        """
        Q = self._split_heads(self.w_q(x))            # (B, n_heads, q_len, d_k)
        K = self._split_heads(self.w_k(x))
        V = self._split_heads(self.w_v(x))

        # RoPE（在拼接 cache 之前施加，保证位置连续）
        Q, K = self.rope(Q, K, offset=cache_offset)

        # KV-Cache 拼接
        if past_kv is not None:
            past_k, past_v = past_kv
            K = torch.cat([past_k, K], dim=2).contiguous()
            V = torch.cat([past_v, V], dim=2).contiguous()

        present_kv = (K.detach(), V.detach()) if use_cache else None

        # 缩放点积注意力
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = self._merge_heads(torch.matmul(attn_weights, V))
        return (output, present_kv) if use_cache else output


# ============================================================
#  Decoder Layer
# ============================================================

class DecoderLayer(nn.Module):
    """单层 Decoder: RMSNorm → Self-Attn(+RoPE) → RMSNorm → SwiGLU FFN（Pre-Norm 残差）。"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        self.attn = GPTAttention(d_model, n_heads, dropout, max_len)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: tuple | None = None,
        use_cache: bool = False,
        cache_offset: int = 0,
    ):
        # Self-Attention (Pre-Norm)
        residual = x
        result = self.attn(self.attn_norm(x), mask, past_kv, use_cache, cache_offset)
        if use_cache:
            h, new_kv = result
        else:
            h = result
        x = residual + self.dropout(h)

        # FFN (Pre-Norm)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))

        return (x, new_kv) if use_cache else x


# ============================================================
#  GPT 模型
# ============================================================

class GPT(nn.Module):
    """
    Decoder-Only GPT 模型，用于中文日常对话。

    架构 (LLaMA-style):
        Embedding
        → DecoderLayer × N  (RMSNorm + RoPE-Attention + SwiGLU-FFN)
        → RMSNorm
        → Linear → Vocab

    特性:
      - RoPE 旋转位置编码（相对位置 + 外推）
      - Pre-Norm + RMSNorm
      - SwiGLU 激活
      - 嵌入 / 输出权重共享
      - KV-Cache 增量解码
      - 无 bias 参数
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.max_len = config.max_len

        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=PAD_ID)
        self.layers = nn.ModuleList([
            DecoderLayer(config.d_model, config.n_heads, config.d_ff,
                         config.dropout, config.max_len)
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 权重共享：输出投影 = 词嵌入
        self.output_proj.weight = self.embedding.weight

        self._init_parameters()

    def _init_parameters(self):
        """LLaMA 风格初始化。"""
        for name, param in self.named_parameters():
            if param.dim() >= 2:
                # w_down（SwiGLU 输出投影）和 w_o（注意力输出）用小方差
                if "w_down" in name or "w_o" in name:
                    nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))
                else:
                    nn.init.normal_(param, mean=0.0, std=0.02)
            elif param.dim() == 1:
                # RMSNorm weight
                nn.init.constant_(param, 1.0)

    # ---------- Mask 工具 ----------

    @staticmethod
    def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """上三角 causal mask, (1, 1, seq_len, seq_len)。"""
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    @staticmethod
    def make_padding_mask(pad_mask: torch.Tensor) -> torch.Tensor:
        """(B, seq_len) → (B, 1, 1, seq_len)。"""
        return pad_mask.unsqueeze(1).unsqueeze(2)

    def _build_mask(self, x: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
        """构建 causal + padding 联合 mask。"""
        seq_len = x.size(1)
        causal = self.make_causal_mask(seq_len, x.device)
        if pad_mask is not None:
            return causal | self.make_padding_mask(pad_mask)
        return causal

    # ---------- 训练前向 ----------

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, seq_len) — token IDs
        pad_mask: (B, seq_len) — True = padding
        返回: (B, seq_len, vocab_size) — logits
        """
        mask = self._build_mask(x, pad_mask)
        h = self.embedding(x) * math.sqrt(self.d_model)

        for layer in self.layers:
            h = layer(h, mask)

        h = self.norm(h)
        return self.output_proj(h)

    # ---------- 增量解码 ----------

    def encode_prompt(
        self, prompt: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list]:
        """
        编码完整 prompt，返回最终 hidden state 和每层 KV-Cache。

        prompt: (1, prompt_len)
        pad_mask: (1, prompt_len)
        """
        mask = self._build_mask(prompt, pad_mask)
        h = self.embedding(prompt) * math.sqrt(self.d_model)

        cache: list = []
        for layer in self.layers:
            h, kv = layer(h, mask, use_cache=True, cache_offset=0)
            cache.append(kv)

        h = self.norm(h)
        return h, cache

    def decode_step(
        self,
        token: torch.Tensor,
        pad_mask: torch.Tensor | None,
        past_cache: list,
        cache_offset: int,
    ) -> tuple[torch.Tensor, list]:
        """
        单步增量解码。

        token: (1, 1) — 单个新 token
        past_cache: 每层的 (K, V)
        cache_offset: 当前 cache 中已有的 token 数
        返回: (logits, new_cache)
        """
        mask = self._build_mask(token, pad_mask)
        h = self.embedding(token) * math.sqrt(self.d_model)

        new_cache: list = []
        for i, layer in enumerate(self.layers):
            h, kv = layer(h, mask, past_cache[i], use_cache=True, cache_offset=cache_offset)
            new_cache.append(kv)

        h = self.norm(h)
        logits = self.output_proj(h)
        return logits, new_cache


# ============================================================
#  模型测试
# ============================================================

if __name__ == "__main__":
    from config import Config

    config = Config()
    config.vocab_size = 5000
    config.max_len = 256
    model = GPT(config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"GPT (LLaMA-style) 参数量: {total_params:,}")

    # 训练前向
    B, seq_len = 4, 50
    x = torch.randint(5, config.vocab_size, (B, seq_len))  # >4 避开特殊 token
    pad = torch.zeros(B, seq_len, dtype=torch.bool)
    pad[:, -5:] = True

    logits = model(x, pad)
    print(f"训练 forward: 输入 {x.shape} → logits {logits.shape}")
    assert logits.shape == (B, seq_len, config.vocab_size)

    # KV-Cache 增量解码
    prompt = torch.randint(5, config.vocab_size, (1, 10))
    ppad = torch.zeros(1, 10, dtype=torch.bool)

    h_last, cache = model.encode_prompt(prompt, ppad)
    print(f"encode_prompt: h {h_last.shape}, cache_layers={len(cache)}")
    assert h_last.shape == (1, 10, config.d_model)

    # 第二步
    new_tok = torch.randint(5, config.vocab_size, (1, 1))
    npad = torch.zeros(1, 1, dtype=torch.bool)
    logits2, cache2 = model.decode_step(new_tok, npad, cache, cache_offset=10)
    print(f"decode_step (offset=10): logits {logits2.shape}")
    assert logits2.shape == (1, 1, config.vocab_size)

    # 第三步
    logits3, _ = model.decode_step(new_tok, npad, cache2, cache_offset=11)
    assert logits3.shape == (1, 1, config.vocab_size)

    print("\n✓ GPT (LLaMA-style) 全部测试通过！")
