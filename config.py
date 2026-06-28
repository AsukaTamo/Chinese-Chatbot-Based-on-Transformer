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
    corpora: tuple = ("xiaohuangji",)

    #  词汇表配置

    vocab_size: int = 50000        # 词汇表上限
    min_freq: int = 3              # 最低词频阈值


    #  模型选择
    #  "transformer" = Encoder-Decoder（原有）
    #  "gpt"         = Decoder-Only（支持多轮记忆）
    model_type: str = "gpt"


    #  模型架构参数
    d_model: int = 512             # 词向量 / 隐层维度
    n_heads: int = 8               # 多头注意力头数
    n_layers: int = 6              # Encoder / Decoder 层数
    d_ff: int = 2048               # 前馈网络隐层维度
    dropout: float = 0.1           # Dropout 比例
    max_len: int = 120             # 最大序列长度（GPT 模式需更长以容纳多轮对话）


    #  训练参数
    batch_size: int = 128
    epochs: int = 30
    warmup_steps: int = 4000       # Noam 调度器预热步数
    label_smoothing: float = 0.1   # 标签平滑
    grad_clip: float = 1.0         # 梯度裁剪阈值
    log_every: int = 100           # 每 N 步打印日志
    save_every_epoch: int = 1      # 每 N 个 epoch 保存一次


    #  推理参数
    beam_size: int = 5
    max_decode_len: int = 50
    temperature: float = 0.8
    length_penalty: float = 0.6
    max_history: int = 4            # GPT 模式保留的最近对话轮数（0=不限）


    #  硬件
    device: str = "cuda"


    #  以下字段由 __post_init__ 自动计算，无需手动配置
    data_paths: list = field(default_factory=list)
    vocab_path: str = ""
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

        # 自动发现每个语料的 .conv 文件
        self.data_paths = []
        for corpus in self.corpora:
            corpus_dir = os.path.join(data_dir, corpus)
            conv_files = glob.glob(os.path.join(corpus_dir, "*.conv"))
            if conv_files:
                # 取第一个 .conv 文件（如有多个可在此调整逻辑）
                self.data_paths.append(conv_files[0])
            else:
                raise FileNotFoundError(
                    f"语料库 '{corpus}' 目录下未找到 .conv 文件: {corpus_dir}\n"
                    f"请将对话数据文件放入 {corpus_dir}/"
                )

        # 词表路径
        vocab_dir = os.path.join(data_dir, self.corpus_name)
        os.makedirs(vocab_dir, exist_ok=True)
        self.vocab_path = os.path.join(vocab_dir, "vocab.json")

        # 模型保存目录
        self.model_save_dir = os.path.join(base, "checkpoints", self.corpus_name)
        os.makedirs(self.model_save_dir, exist_ok=True)

        # 日志
        self.log_path = os.path.join(self.model_save_dir, "train.log")



    #  模型保存 / 加载路径辅助
    @property
    def best_model_path(self) -> str:
        return os.path.join(self.model_save_dir, "best_model.pt")

    @property
    def history_path(self) -> str:
        return os.path.join(self.model_save_dir, "history.json")



#  特殊 Token
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3
