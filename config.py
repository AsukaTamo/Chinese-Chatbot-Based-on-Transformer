"""
中文聊天机器人 — 全局配置
支持多语料库切换：在 corpora 中指定语料名称即可
"""

import os
import glob
from dataclasses import dataclass, field


@dataclass
class Config:
    #  语料库配置（修改这里切换训练语料）
    #  单语料:  corpora = ("xiaohuangji",)
    #  多语料:  corpora = ("xiaohuangji", "weibo")
    corpora: tuple = ("LCCC-base-split",)

    #  词汇表配置

    vocab_size: int = 50000        # 词汇表上限
    min_freq: int = 3              # 最低词频阈值


    #  模型选择
    #  "transformer" = Encoder-Decoder（Lite 版，手写）
    #  "gpt"         = Decoder-Only（Middle 版，手写，多轮记忆）
    #  "qwen"        = Qwen2-1.5B 预训练（Pro 版，HuggingFace）
    model_type: str = "qwen"

    #  分词模式
    #  "jieba" = jieba 分词（默认，适用于未分词的原始中文文本）
    #  "space" = 空格切分（适用于 LCCC 等已预分词的语料）
    fenci_mode: str = "jieba"

    #  预训练模型（仅 model_type="qwen" 时使用）
    qwen_model_name: str = "Qwen/Qwen2-1.5B-Instruct"
    qwen_freeze_layers: int = 6      # 微调时冻结前 N 层（0=全量微调）
    hf_cache_dir: str = ".modelcache"  # HuggingFace 模型缓存目录
    hf_endpoint: str = "https://hf-mirror.com"  # 国内镜像站


    #  模型架构参数
    d_model: int = 512             # 词向量 / 隐层维度
    n_heads: int = 8               # 多头注意力头数
    n_layers: int = 6              # Encoder / Decoder 层数
    d_ff: int = 2048               # 前馈网络隐层维度
    dropout: float = 0.1           # Dropout 比例
    max_len: int = 120             # 最大序列长度（RoPE 动态外推，解码时不受此限制）


    #  训练参数
    batch_size: int = 32
    epochs: int = 30
    warmup_steps: int = 4000       # Noam 调度器预热步数
    label_smoothing: float = 0.1   # 标签平滑
    grad_clip: float = 1.0         # 梯度裁剪阈值
    log_every: int = 100           # 每 N 步打印日志
    save_every_epoch: int = 1      # 每 N 个 epoch 保存一次


    #  推理参数
    beam_size: int = 5
    max_decode_len: int = 200
    temperature: float = 0.8
    length_penalty: float = 0.6
    max_history: int = 4            # GPT 模式保留的最近对话轮数（0=不限）


    #  硬件
    device: str = "cuda"

    #  Graphviz 可视化（tools/visualize_models.py 使用）
    #  Windows 默认路径见注释，其他系统可设为 ""（依赖 PATH）
    graphviz_bin_path: str = r"C:\Program Files\Graphviz\bin"  # Linux/macOS: ""


    #  以下字段由 __post_init__ 自动计算，无需手动配置
    data_paths: list = field(default_factory=list)
    model_save_dir: str = ""
    corpus_name: str = ""
    _initialized: bool = False

    def __post_init__(self):
        if self._initialized:
            return
        self._initialized = True

        base = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base, "data")

        # 语料库名称（多语料用 + 连接）
        self.corpus_name = "+".join(self.corpora)

        # 自动发现语料文件（优先 .json LCCC 格式 → .conv 旧格式）
        # 排除词表文件（vocab_*.json），避免被当成语料
        self.data_paths = []
        for corpus in self.corpora:
            corpus_dir = os.path.join(data_dir, corpus)
            json_files = [f for f in glob.glob(os.path.join(corpus_dir, "*.json"))
                          if not os.path.basename(f).startswith("vocab_")]
            conv_files = glob.glob(os.path.join(corpus_dir, "*.conv"))
            if json_files:
                self.data_paths.extend(json_files)
            elif conv_files:
                self.data_paths.extend(conv_files)
            else:
                raise FileNotFoundError(
                    f"语料库 '{corpus}' 目录下未找到数据文件: {corpus_dir}\n"
                    f"请放入 .json (LCCC 格式) 或 .conv (旧格式) 文件"
                )

        # 词表路径（按模型类型区分，不同架构词表不同）
        vocab_dir = os.path.join(data_dir, self.corpus_name)
        os.makedirs(vocab_dir, exist_ok=True)

        # 模型保存目录
        self.model_save_dir = os.path.join(base, "checkpoints", self.corpus_name)
        os.makedirs(self.model_save_dir, exist_ok=True)

        # 日志
        self.log_path = os.path.join(self.model_save_dir, "train.log")



    #  动态路径（随 model_type 变化自动切换）
    @property
    def vocab_path(self) -> str:
        """词表路径 — 按模型类型区分，不同架构使用不同词表文件。"""
        base = os.path.dirname(os.path.abspath(__file__))
        vocab_dir = os.path.join(base, "data", self.corpus_name)
        os.makedirs(vocab_dir, exist_ok=True)
        return os.path.join(vocab_dir, f"vocab_{self.model_type}.json")

    #  模型保存 / 加载路径辅助
    @property
    def best_model_path(self) -> str:
        return os.path.join(self.model_save_dir, f"best_model_{self.model_type}.pt")

    @property
    def history_path(self) -> str:
        return os.path.join(self.model_save_dir, f"history_{self.model_type}.json")



#  特殊 Token
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"

PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3
USER_ID = 4
ASSISTANT_ID = 5
