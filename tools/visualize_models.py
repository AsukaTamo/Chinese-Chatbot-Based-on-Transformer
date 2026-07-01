"""
Model Architecture Visualization (PlotNeuralNet style via Graphviz)

Generates high-resolution horizontal architecture diagrams for:
  - model_gpt.py  (GPT Decoder-Only, LLaMA-style)
  - model.py       (Transformer Encoder-Decoder)

Output: tools/assets/  (PNG @ 300 DPI + SVG)

Requirements: pip install graphviz  +  system Graphviz on PATH
Usage:          python tools/visualize_models.py
"""

import os
import sys
import graphviz

# Allow importing from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# ── PlotNeuralNet-inspired palette (slightly saturated for print) ──
CLR = {
    "embed":   "#81c784",  # green 500
    "norm":    "#ffb74d",  # orange 400
    "attn":    "#64b5f6",  # blue 400
    "rope":    "#4db6ac",  # teal 400
    "ffn":     "#f06292",  # pink 400
    "cross":   "#42a5f5",  # blue 600
    "output":  "#ba68c8",  # purple 400
    "pos":     "#a5d6a7",  # green 300
    "dropout": "#bdbdbd",  # grey 400
    "add":     "#eeeeee",
    "bg":      "#ffffff",
    "enc_bg":  "#ede7f618",
    "dec_bg":  "#e8eaf618",
    "layer_bg":"#fafafa",
    "io":      "#f5f5f5",
    "residual":"#bdbdbd",
}


# ╔══════════════════════════════════════════════════════════════╗
# ║                    Graph builder helpers                     ║
# ╚══════════════════════════════════════════════════════════════╝

def _make_graph(name: str, title: str) -> graphviz.Digraph:
    g = graphviz.Digraph(name=name, format="png", engine="dot")
    g.attr(
        rankdir="LR",
        splines="ortho",           # orthogonal lines for clean look
        nodesep="0.50",
        ranksep="0.80",
        fontname="Helvetica",
        bgcolor=CLR["bg"],
        label=title,
        labelloc="t",
        fontsize="22",
        pad="0.6",
        dpi="300",                 # high resolution
        newrank="true",
        outputorder="edgesfirst",  # edges drawn behind nodes
        overlap="false",
        sep="+12",
    )
    g.attr("node",
           shape="box",
           style="filled,rounded",
           fontname="Helvetica",
           fontsize="11",
           penwidth="1.2",
           margin="0.14,0.08",
    )
    g.attr("edge",
           arrowsize="0.8",
           penwidth="1.2",
           color="#666666",
           fontname="Helvetica",
           fontsize="9",
    )
    return g


def _make_layer_cluster(parent, prefix: str, layer_label: str):
    """Create a dashed DecoderLayer / EncoderLayer subgraph with consistent style."""
    c = graphviz.Digraph(name=f"cluster_{prefix}_layer")
    c.attr(
        label=layer_label,
        style="dashed,rounded",
        bgcolor=CLR["layer_bg"],
        pencolor="#9e9e9e",
        penwidth="1.5",
        fontsize="12",
        fontname="Helvetica",
    )
    parent.subgraph(c)
    return c


def _node_norm(parent, name: str, label: str = "RMSNorm"):
    parent.node(name, label, fillcolor=CLR["norm"])

def _node_dropout(parent, name: str):
    parent.node(name, "Dropout", fillcolor=CLR["dropout"], fontsize="9", margin="0.08,0.04")

def _node_add(parent, name: str):
    parent.node(name, "+", fillcolor=CLR["add"], shape="circle",
                width="0.25", penwidth="1.0", fontsize="12")


# ╔══════════════════════════════════════════════════════════════╗
# ║                  GPT  (LLaMA-style)                          ║
# ╚══════════════════════════════════════════════════════════════╝

def build_gpt() -> graphviz.Digraph:
    g = _make_graph("GPT", "GPT Architecture  (LLaMA-style Decoder-Only)")

    # ── Input ──
    g.node("inp", "Input Tokens\n(B, seq_len)",
           shape="ellipse", fillcolor=CLR["io"], penwidth="1.4")

    # ── Embedding ──
    g.node("emb", "Token Embedding\n x sqrt(d_model)",
           fillcolor=CLR["embed"], penwidth="1.4")

    g.edge("inp", "emb")

    # ═══════════ DecoderLayer ═══════════
    with g.subgraph(name="cluster_gpt_layer") as c:
        c.attr(
            label="DecoderLayer   (x N_layers)",
            style="dashed,rounded",
            bgcolor=CLR["layer_bg"],
            pencolor="#9e9e9e",
            penwidth="2.0",
            fontsize="13",
            fontname="Helvetica",
        )

        # ── Sub-block: GPTAttention ──
        with c.subgraph(name="cluster_gpt_attn") as ab:
            ab.attr(
                label="GPTAttention",
                style="filled,rounded",
                fillcolor="#e3f2fd28",
                pencolor=CLR["attn"],
                penwidth="1.6",
                fontsize="11",
                color=CLR["attn"],
            )
            ab.node("qkv",  "Q, K, V\nLinear (no bias)",
                    fillcolor=CLR["attn"])
            ab.node("rope", "RoPE\nRotary Pos. Enc.",
                    fillcolor=CLR["rope"])
            ab.node("kvc",  "KV-Cache\nConcat",
                    fillcolor=CLR["attn"])
            ab.node("sda",  "Scaled\nDot-Product\nAttention",
                    fillcolor=CLR["attn"])
            ab.node("mrg",  "Merge Heads\n+ Linear O",
                    fillcolor=CLR["attn"])

        # ── Sub-block: SwiGLU FFN ──
        with c.subgraph(name="cluster_gpt_ffn") as fb:
            fb.attr(
                label="SwiGLU FFN",
                style="filled,rounded",
                fillcolor="#fce4ec28",
                pencolor=CLR["ffn"],
                penwidth="1.6",
                fontsize="11",
                color=CLR["ffn"],
            )
            fb.node("gate", "Linear Gate+Up\n(d_model -> 2*d_ff)",
                    fillcolor=CLR["ffn"])
            fb.node("silu", "SiLU(Gate) * Up",
                    fillcolor=CLR["ffn"])
            fb.node("down", "Linear Down\n(d_ff -> d_model)",
                    fillcolor=CLR["ffn"])

        # ── Path nodes ──
        _node_norm(c, "anorm")
        _node_dropout(c, "do1")
        _node_add(c, "add1")
        _node_norm(c, "fnorm")
        _node_dropout(c, "do2")
        _node_add(c, "add2")

        # ── Attention forward edges ──
        c.edge("anorm", "qkv")
        c.edge("qkv",   "rope")
        c.edge("rope",  "kvc")
        c.edge("kvc",   "sda")
        c.edge("sda",   "mrg")
        c.edge("mrg",   "do1")
        c.edge("do1",   "add1")
        # Residual bypass
        c.edge("anorm", "add1",
               style="dashed", penwidth="1.2",
               color=CLR["residual"], constraint="false")

        # ── FFN forward edges ──
        c.edge("add1",  "fnorm")
        c.edge("fnorm", "gate")
        c.edge("gate",  "silu")
        c.edge("silu",  "down")
        c.edge("down",  "do2")
        c.edge("do2",   "add2")
        # Residual bypass
        c.edge("fnorm", "add2",
               style="dashed", penwidth="1.2",
               color=CLR["residual"], constraint="false")

    # ── Output ──
    g.node("fn",   "RMSNorm\n(Final)", fillcolor=CLR["norm"], penwidth="1.4")
    g.node("outp", "Linear  ->  Vocab\n(weight tied with Embedding)",
           fillcolor=CLR["output"], penwidth="1.4")
    g.node("logits", "Output Logits\n(B, seq, vocab_size)",
           shape="ellipse", fillcolor=CLR["io"], penwidth="1.4")

    g.edge("emb",   "anorm", lhead="cluster_gpt_layer")
    g.edge("add2",  "fn",    ltail="cluster_gpt_layer")
    g.edge("fn",    "outp")
    g.edge("outp",  "logits")

    return g


# ╔══════════════════════════════════════════════════════════════╗
# ║              Transformer  (Encoder-Decoder)                  ║
# ╚══════════════════════════════════════════════════════════════╝

def build_transformer() -> graphviz.Digraph:
    g = _make_graph("Transformer", "Transformer Architecture  (Encoder-Decoder)")

    # ═══════════════ Encoder (top) ═══════════════
    with g.subgraph(name="cluster_enc") as enc:
        enc.attr(
            label="Encoder",
            style="filled,rounded",
            fillcolor=CLR["enc_bg"],
            pencolor="#7e57c2",
            penwidth="2.2",
            fontsize="15",
            color="#7e57c2",
        )

        enc.node("e_inp", "Source\nTokens",
                 shape="ellipse", fillcolor=CLR["io"], penwidth="1.4")
        enc.node("e_emb", "Token\nEmbedding",
                 fillcolor=CLR["embed"], penwidth="1.4")
        enc.node("e_pos", "Positional\nEncoding (sin)",
                 fillcolor=CLR["pos"], penwidth="1.4")
        _node_add(enc, "e_add")

        with enc.subgraph(name="cluster_enc_layer") as el:
            el.attr(
                label="EncoderLayer   (x N_layers)",
                style="dashed,rounded",
                bgcolor=CLR["layer_bg"],
                pencolor="#9e9e9e",
                penwidth="1.8",
                fontsize="12",
            )

            _node_norm(el, "e_ln1", "LayerNorm")
            el.node("e_sa", "Multi-Head\nSelf-Attention",
                    fillcolor=CLR["attn"], penwidth="1.4")
            _node_dropout(el, "e_do1")
            _node_add(el, "e_ad1")
            _node_norm(el, "e_ln2", "LayerNorm")
            el.node("e_ffn", "Feed-Forward\n(ReLU)",
                    fillcolor=CLR["ffn"], penwidth="1.4")
            _node_dropout(el, "e_do2")
            _node_add(el, "e_ad2")

            el.edge("e_ln1", "e_sa")
            el.edge("e_sa",  "e_do1")
            el.edge("e_do1", "e_ad1")
            el.edge("e_ln1", "e_ad1",
                    style="dashed", penwidth="1.2",
                    color=CLR["residual"], constraint="false")
            el.edge("e_ad1", "e_ln2")
            el.edge("e_ln2", "e_ffn")
            el.edge("e_ffn", "e_do2")
            el.edge("e_do2", "e_ad2")
            el.edge("e_ln2", "e_ad2",
                    style="dashed", penwidth="1.2",
                    color=CLR["residual"], constraint="false")

        enc.node("e_fn", "LayerNorm\n(Final)",
                 fillcolor=CLR["norm"], penwidth="1.4")

        enc.edge("e_inp", "e_emb")
        enc.edge("e_emb", "e_add")
        enc.edge("e_pos", "e_add")
        enc.edge("e_add", "e_ln1", lhead="cluster_enc_layer")
        enc.edge("e_ad2", "e_fn",  ltail="cluster_enc_layer")

    # ═══════════════ Decoder (bottom) ═══════════════
    with g.subgraph(name="cluster_dec") as dec:
        dec.attr(
            label="Decoder",
            style="filled,rounded",
            fillcolor=CLR["dec_bg"],
            pencolor="#3f51b5",
            penwidth="2.2",
            fontsize="15",
            color="#3f51b5",
        )

        dec.node("d_inp", "Target\nTokens",
                 shape="ellipse", fillcolor=CLR["io"], penwidth="1.4")
        dec.node("d_emb", "Token\nEmbedding",
                 fillcolor=CLR["embed"], penwidth="1.4")
        dec.node("d_pos", "Positional\nEncoding (sin)",
                 fillcolor=CLR["pos"], penwidth="1.4")
        _node_add(dec, "d_add")

        with dec.subgraph(name="cluster_dec_layer") as dl:
            dl.attr(
                label="DecoderLayer   (x N_layers)",
                style="dashed,rounded",
                bgcolor=CLR["layer_bg"],
                pencolor="#9e9e9e",
                penwidth="1.8",
                fontsize="12",
            )

            _node_norm(dl, "d_ln1", "LayerNorm")
            dl.node("d_msa", "Masked Multi-Head\nSelf-Attention",
                    fillcolor=CLR["attn"], penwidth="1.4")
            _node_dropout(dl, "d_do1")
            _node_add(dl, "d_ad1")
            _node_norm(dl, "d_ln2", "LayerNorm")
            dl.node("d_ca",  "Multi-Head\nCross-Attention",
                    fillcolor=CLR["cross"], penwidth="1.4")
            _node_dropout(dl, "d_do2")
            _node_add(dl, "d_ad2")
            _node_norm(dl, "d_ln3", "LayerNorm")
            dl.node("d_ffn", "Feed-Forward\n(ReLU)",
                    fillcolor=CLR["ffn"], penwidth="1.4")
            _node_dropout(dl, "d_do3")
            _node_add(dl, "d_ad3")

            dl.edge("d_ln1", "d_msa")
            dl.edge("d_msa", "d_do1")
            dl.edge("d_do1", "d_ad1")
            dl.edge("d_ln1", "d_ad1",
                    style="dashed", penwidth="1.2",
                    color=CLR["residual"], constraint="false")
            dl.edge("d_ad1", "d_ln2")
            dl.edge("d_ln2", "d_ca")
            dl.edge("d_ca",  "d_do2")
            dl.edge("d_do2", "d_ad2")
            dl.edge("d_ln2", "d_ad2",
                    style="dashed", penwidth="1.2",
                    color=CLR["residual"], constraint="false")
            dl.edge("d_ad2", "d_ln3")
            dl.edge("d_ln3", "d_ffn")
            dl.edge("d_ffn", "d_do3")
            dl.edge("d_do3", "d_ad3")
            dl.edge("d_ln3", "d_ad3",
                    style="dashed", penwidth="1.2",
                    color=CLR["residual"], constraint="false")

        dec.node("d_fn", "LayerNorm\n(Final)",
                 fillcolor=CLR["norm"], penwidth="1.4")

        dec.edge("d_inp", "d_emb")
        dec.edge("d_emb", "d_add")
        dec.edge("d_pos", "d_add")
        dec.edge("d_add", "d_ln1", lhead="cluster_dec_layer")
        dec.edge("d_ad3", "d_fn",  ltail="cluster_dec_layer")

    # ── Cross-attention bridge ──
    g.edge("e_fn", "d_ca",
           style="bold", color="#1565c0", penwidth="2.5",
           xlabel="  K, V  ", fontcolor="#1565c0", fontsize="11")

    # ── Output ──
    g.node("t_out", "Linear  ->  Vocab",
           fillcolor=CLR["output"], penwidth="1.4")
    g.node("t_log", "Output Logits\n(B, tgt_len, vocab_size)",
           shape="ellipse", fillcolor=CLR["io"], penwidth="1.4")

    g.edge("d_fn",  "t_out")
    g.edge("t_out", "t_log")

    # ── Rank alignment for clean cross-attention bridge ──
    with g.subgraph() as align:
        align.attr(rank="same")
        align.node("e_fn")
        align.node("d_ca")

    return g


# ╔══════════════════════════════════════════════════════════════╗
# ║          Qwen2-1.5B  Hand-written  (GQA + RoPE)             ║
# ╚══════════════════════════════════════════════════════════════╝

def build_qwen2_hand() -> graphviz.Digraph:
    g = _make_graph("Qwen2Hand", "Qwen2-1.5B Architecture  (Hand-written, GQA Decoder-Only)")

    # ── Input ──
    g.node("qw_inp", "Input Tokens\n(B, seq_len)",
           shape="ellipse", fillcolor=CLR["io"], penwidth="1.4")

    # ── Embedding ──
    g.node("qw_emb", "Token Embedding\nvocab=151936, d_model=1536",
           fillcolor=CLR["embed"], penwidth="1.4")

    g.edge("qw_inp", "qw_emb")

    # ═══════════ Qwen2Block ═══════════
    with g.subgraph(name="cluster_qwen_block") as c:
        c.attr(
            label="Qwen2Block   (x 28 layers)",
            style="dashed,rounded",
            bgcolor=CLR["layer_bg"],
            pencolor="#9e9e9e",
            penwidth="2.0",
            fontsize="13",
            fontname="Helvetica",
        )

        # ── Sub-block: GQAttention (Grouped Query) ──
        with c.subgraph(name="cluster_qwen_gqa") as ab:
            ab.attr(
                label="GQAttention  (Grouped Query,  Q:12 / KV:2 heads)",
                style="filled,rounded",
                fillcolor="#e1f5fe28",
                pencolor="#0288d1",
                penwidth="1.6",
                fontsize="11",
                color="#0288d1",
            )
            ab.node("qw_q",  "Q Projection\n12 heads x d_k",
                    fillcolor=CLR["attn"])
            ab.node("qw_k",  "K Projection\n2 heads x d_k",
                    fillcolor=CLR["cross"])
            ab.node("qw_v",  "V Projection\n2 heads x d_k",
                    fillcolor=CLR["cross"])
            ab.node("qw_rope","RoPE\nbase=1,000,000",
                    fillcolor=CLR["rope"])
            ab.node("qw_kv",  "GQA KV-Cache\nrepeat KV x 6",
                    fillcolor=CLR["attn"])
            ab.node("qw_sda", "Scaled Dot-Product\nAttention (GQA)",
                    fillcolor=CLR["attn"])
            ab.node("qw_o",   "Output Projection\nLinear O (bias=False)",
                    fillcolor=CLR["attn"])

        # ── Sub-block: SwiGLU MLP ──
        with c.subgraph(name="cluster_qwen_ffn") as fb:
            fb.attr(
                label="SwiGLU MLP",
                style="filled,rounded",
                fillcolor="#fce4ec28",
                pencolor=CLR["ffn"],
                penwidth="1.6",
                fontsize="11",
                color=CLR["ffn"],
            )
            fb.node("qw_gate","Linear Gate+Up\n(d_model=1536 -> d_ff=8960)",
                    fillcolor=CLR["ffn"])
            fb.node("qw_silu","SiLU(Gate) * Up",
                    fillcolor=CLR["ffn"])
            fb.node("qw_down","Linear Down\n(d_ff=8960 -> d_model=1536)",
                    fillcolor=CLR["ffn"])

        # ── Path nodes ──
        _node_norm(c, "qw_anorm")
        _node_norm(c, "qw_fnorm")
        _node_add(c, "qw_add1")
        _node_add(c, "qw_add2")

        # ── Attention path edges ──
        c.edge("qw_anorm", "qw_q")
        c.edge("qw_anorm", "qw_k")
        c.edge("qw_anorm", "qw_v")
        c.edge("qw_q",    "qw_rope")
        c.edge("qw_k",    "qw_rope")
        c.edge("qw_q",    "qw_sda")
        c.edge("qw_k",    "qw_kv")
        c.edge("qw_v",    "qw_kv")
        c.edge("qw_kv",   "qw_sda")
        c.edge("qw_sda",  "qw_o")
        c.edge("qw_o",    "qw_add1")
        # Residual
        c.edge("qw_anorm","qw_add1",
               style="dashed", penwidth="1.2",
               color=CLR["residual"], constraint="false")

        # ── FFN path edges ──
        c.edge("qw_add1",  "qw_fnorm")
        c.edge("qw_fnorm", "qw_gate")
        c.edge("qw_gate",  "qw_silu")
        c.edge("qw_silu",  "qw_down")
        c.edge("qw_down",  "qw_add2")
        # Residual
        c.edge("qw_fnorm", "qw_add2",
               style="dashed", penwidth="1.2",
               color=CLR["residual"], constraint="false")

    # ── Output ──
    g.node("qw_fn",   "RMSNorm\n(Final)", fillcolor=CLR["norm"], penwidth="1.4")
    g.node("qw_outp", "LM Head  ->  Vocab\n(weight tied, bias=False)",
           fillcolor=CLR["output"], penwidth="1.4")
    g.node("qw_logits", "Output Logits\n(B, seq, vocab=151936)",
           shape="ellipse", fillcolor=CLR["io"], penwidth="1.4")

    g.edge("qw_emb",   "qw_anorm", lhead="cluster_qwen_block")
    g.edge("qw_add2",  "qw_fn",    ltail="cluster_qwen_block")
    g.edge("qw_fn",    "qw_outp")
    g.edge("qw_outp",  "qw_logits")

    return g


# ╔══════════════════════════════════════════════════════════════╗
# ║                         Main                                 ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)

    config = Config()
    gv_path = config.graphviz_bin_path
    if gv_path and os.path.isdir(gv_path) and gv_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = gv_path + os.pathsep + os.environ.get("PATH", "")
        os.add_dll_directory(gv_path)

    print("Generating high-resolution architecture diagrams (300 DPI) ...")

    for builder, name in [
        (build_gpt,           "gpt_architecture"),
        (build_transformer,   "transformer_architecture"),
        (build_qwen2_hand,    "qwen2_hand_architecture"),
    ]:
        print(f"  [{name}] ...", end=" ")
        gv = builder()
        for fmt in ("png", "svg"):
            gv.render(os.path.join(ASSETS_DIR, name), format=fmt, cleanup=True)
        size = os.path.getsize(os.path.join(ASSETS_DIR, f"{name}.png"))
        print(f"OK ({size // 1024} KB)")

    print(f"\nDone -> {ASSETS_DIR}/")


if __name__ == "__main__":
    main()
