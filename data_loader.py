import json
import os
from collections import Counter

import jieba
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import Config, PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, PAD_ID, UNK_ID, SOS_ID, EOS_ID, \
    USER_TOKEN, ASSISTANT_TOKEN, USER_ID, ASSISTANT_ID

#  通用工具

def tokenize(text: str, mode: str = "jieba") -> list[str]:
    """统一分词入口 — 根据 mode 选择分词策略。

    mode="jieba"  — jieba 词级别分词（适用于原始中文文本）
    mode="space"  — 空格切分（适用于 LCCC 等已预分词的语料）
    """
    if mode == "space":
        # 空格/空白切分（LCCC 格式）
        return text.strip().split()
    else:
        # jieba 分词：先做规范化（去除旧格式分隔符），再分词
        text = text.replace(" ", "").replace("　", "").replace("/", "")
        return jieba.lcut(text)


def normalize_text(text: str) -> str:
    """规范化原始文本：去除空格和旧格式分隔符，还原为正常中文。"""
    text = text.replace(" ", "").replace("　", "")
    text = text.replace("/", "")
    return text


def parse_lccc_json(file_path: str) -> list[tuple[str, str]]:
    """解析 LCCC JSON 对话文件，返回 (query, response) 对列表。

    JSON 格式:
        [
          ["消息1", "消息2", "消息3"],
          ["消息1", "消息2"],
          ...
        ]
    - 顶层是数组，每个元素是一次对话
    - 每次对话包含 N 条消息（N ≥ 2）
    - 相邻消息两两配对: (msg[0], msg[1]), (msg[2], msg[3]), ...
    - 文本保留原始格式（空格分隔），由 tokenize() 统一处理
    """
    pairs: list[tuple[str, str]] = []
    fname = os.path.basename(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for conv in tqdm(data, desc=f"[Parse] {fname}", unit=" conv"):
        for i in range(0, len(conv) - 1, 2):
            q, r = conv[i].strip(), conv[i + 1].strip()
            if q and r:
                pairs.append((q, r))

    return pairs


def _load_corpus_pairs(config: Config, tag: str = "Data") -> list[tuple[str, str]]:
    """加载所有语料文件的 (Q, R) 对话对。自动识别 .json / .conv 格式。"""
    all_pairs: list[tuple[str, str]] = []
    for path in config.data_paths:
        corpus_label = os.path.basename(os.path.dirname(path))
        ext = os.path.splitext(path)[1]
        print(f"[{tag}] 解析语料 [{corpus_label}]: {path}")

        if ext == ".json":
            pairs = parse_lccc_json(path)
        else:
            pairs = parse_conv_file(path)

        print(f"[{tag}]   → {len(pairs)} 个对话对")
        all_pairs.extend(pairs)
    print(f"[{tag}] 合并后总对话对: {len(all_pairs)}")
    return all_pairs



#  数据集解析
def parse_conv_file(file_path: str) -> list[tuple[str, str]]:
    """解析 .conv 对话文件，返回 (query, response) 对列表。"""
    pairs = []
    fname = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        segment_msgs: list[str] = []
        for line in tqdm(f, desc=f"[Parse] {fname}", unit=" lines"):
            line = line.strip()
            if not line:
                continue

            if line == "E":
                if segment_msgs:
                    _extract_pairs(segment_msgs, pairs)
                    segment_msgs = []
            elif line.startswith("M "):
                # 去掉 "M " 前缀，保留原始文本（规范化由 tokenize 处理）
                segment_msgs.append(line[2:].strip())

        # 文件末尾未闭合的 segment
        if segment_msgs:
            _extract_pairs(segment_msgs, pairs)

    return pairs


def _extract_pairs(msgs: list[str], pairs: list[tuple[str, str]]) -> None:

    for i in range(0, len(msgs) - 1, 2):
        query = msgs[i]
        response = msgs[i + 1]
        if query and response:
            pairs.append((query, response))



#  词汇表构建
def build_vocab(pairs: list[tuple[str, str]], config: Config) -> tuple[dict[str, int], dict[int, str]]:

    counter: Counter[str] = Counter()
    mode = getattr(config, "fenci_mode", "jieba")

    for query, response in tqdm(pairs, desc="[Vocab] 统计词频"):
        for token in tokenize(query, mode):
            counter[token] += 1
        for token in tokenize(response, mode):
            counter[token] += 1

    print(f"[Vocab] 总 token 种类数: {len(counter)}")

    # 按模型类型决定特殊 token 数量
    if config.model_type == "gpt":
        num_special = 6
        token2id: dict[str, int] = {
            PAD_TOKEN: PAD_ID,
            UNK_TOKEN: UNK_ID,
            SOS_TOKEN: SOS_ID,
            EOS_TOKEN: EOS_ID,
            USER_TOKEN: USER_ID,
            ASSISTANT_TOKEN: ASSISTANT_ID,
        }
    else:
        num_special = 4
        token2id: dict[str, int] = {
            PAD_TOKEN: PAD_ID,
            UNK_TOKEN: UNK_ID,
            SOS_TOKEN: SOS_ID,
            EOS_TOKEN: EOS_ID,
        }

    # 按词频排序，保留 top vocab_size-N（为特殊 token 留位置）
    most_common = counter.most_common(config.vocab_size - num_special)

    for token, freq in most_common:
        if freq >= config.min_freq:
            token2id[token] = len(token2id)

    id2token: dict[int, str] = {v: k for k, v in token2id.items()}
    print(f"[Vocab] 最终词汇表大小: {len(token2id)}")
    return token2id, id2token


def save_vocab(token2id: dict[str, int], id2token: dict[int, str], path: str) -> None:
    """保存词汇表到 JSON 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"token2id": token2id, "id2token": {str(k): v for k, v in id2token.items()}}, f,
                  ensure_ascii=False, indent=2)
    print(f"[Vocab] 词汇表已保存至: {path}")


def load_vocab(path: str) -> tuple[dict[str, int], dict[int, str]]:
    """从 JSON 文件加载词汇表。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    token2id = data["token2id"]
    id2token = {int(k): v for k, v in data["id2token"].items()}
    print(f"[Vocab] 词汇表已加载，大小: {len(token2id)}")
    return token2id, id2token


# ============================================================
#Transformer 数据集（Encoder-Decoder）
# ===========================================================
class ChatDataset(Dataset):

    def __init__(self, pairs: list[tuple[str, str]], token2id: dict[str, int], max_len: int,
                 fenci_mode: str = "jieba"):
        self.pairs = pairs
        self.token2id = token2id
        self.max_len = max_len
        self.fenci_mode = fenci_mode

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        query, response = self.pairs[idx]

        src_tokens = [self.token2id.get(t, UNK_ID) for t in tokenize(query, self.fenci_mode)]
        tgt_tokens = [self.token2id.get(t, UNK_ID) for t in tokenize(response, self.fenci_mode)]

        # 截断
        src_tokens = src_tokens[:self.max_len]
        tgt_tokens = tgt_tokens[:self.max_len - 1]  # 为 <SOS>/<EOS> 留空间

        # 转换为 Tensor
        src = torch.tensor(src_tokens, dtype=torch.long)
        tgt_input = torch.tensor([SOS_ID] + tgt_tokens, dtype=torch.long)
        tgt_output = torch.tensor(tgt_tokens + [EOS_ID], dtype=torch.long)

        return {"src": src, "tgt_input": tgt_input, "tgt_output": tgt_output}


def collate_fn(batch: list[dict[str, torch.Tensor]], pad_id: int = PAD_ID) -> dict[str, torch.Tensor]:
    """
    批次整理函数：将不等长序列 padding 到批次内最大长度。
    """
    # 按 src 长度降序排列（便于后续处理）
    batch = sorted(batch, key=lambda x: len(x["src"]), reverse=True)

    src_list, tgt_input_list, tgt_output_list = [], [], []
    for item in batch:
        src_list.append(item["src"])
        tgt_input_list.append(item["tgt_input"])
        tgt_output_list.append(item["tgt_output"])

    # Padding
    src_padded = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=pad_id)
    tgt_input_padded = torch.nn.utils.rnn.pad_sequence(tgt_input_list, batch_first=True, padding_value=pad_id)
    tgt_output_padded = torch.nn.utils.rnn.pad_sequence(tgt_output_list, batch_first=True, padding_value=pad_id)

    # Padding mask: True 表示 padding 位置
    src_mask = (src_padded == pad_id)         # (B, src_len)
    tgt_pad_mask = (tgt_input_padded == pad_id)  # (B, tgt_len)

    return {
        "src": src_padded,                    # (B, src_len)
        "tgt_input": tgt_input_padded,        # (B, tgt_len)
        "tgt_output": tgt_output_padded,      # (B, tgt_len)
        "src_mask": src_mask,                 # (B, src_len) — True = padding
        "tgt_pad_mask": tgt_pad_mask,         # (B, tgt_len) — True = padding
    }



#  DataLoader 构建


def create_dataloaders(
    pairs: list[tuple[str, str]],
    token2id: dict[str, int],
    config: Config,
    train_ratio: float = 0.95,
):
    """
    划分训练集 / 验证集，创建 DataLoader。
    返回: (train_loader, val_loader)
    """
    split = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]

    print(f"[Data] 训练集: {len(train_pairs)} 对, 验证集: {len(val_pairs)} 对")

    mode = getattr(config, "fenci_mode", "jieba")
    train_dataset = ChatDataset(train_pairs, token2id, config.max_len, mode)
    val_dataset = ChatDataset(val_pairs, token2id, config.max_len, mode)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,       # Windows 下多进程可能导致问题，保守设 0
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader

#  一键预处理入口
def prepare_data(config: Config):
    """
    一键完成数据预处理：解析文件 → 构建词汇表 → 保存词汇表 → 创建 DataLoader。

    支持单语料和多语料联合训练：
      - config.data_paths 包含一个或多个 .conv 文件路径
      - 多语料时合并所有对话对，构建统一词表

    返回: (train_loader, val_loader, token2id, id2token)
    """
    print("=" * 60)
    print(f"数据预处理开始 — 语料: {config.corpus_name}")
    print("=" * 60)

    all_pairs = _load_corpus_pairs(config, "Data")

    # 构建 / 加载词汇表
    if os.path.exists(config.vocab_path):
        token2id, id2token = load_vocab(config.vocab_path)
    else:
        token2id, id2token = build_vocab(all_pairs, config)
        save_vocab(token2id, id2token, config.vocab_path)

    # 创建 DataLoader
    train_loader, val_loader = create_dataloaders(all_pairs, token2id, config)

    print("=" * 60)
    print("数据预处理完成")
    print("=" * 60)
    return train_loader, val_loader, token2id, id2token


# ============================================================
#  GPT Decoder-Only 数据集

class GPTDataset(Dataset):
    """
    GPT 格式数据集：Speaker Token + 全序列自回归。

    格式: <|user|> Q_tokens <|assistant|> R_tokens <EOS>

    input_ids:  [USER, Q..., ASSISTANT, R..., EOS]
    target_ids: [Q..., ASSISTANT, R..., EOS, PAD]  (左移一位)
    loss_mask:  [1, ...1, 1, ...1, 1]              (全序列参与 loss)
    """

    def __init__(self, pairs: list[tuple[str, str]], token2id: dict[str, int], max_len: int,
                 fenci_mode: str = "jieba"):
        self.pairs = pairs
        self.token2id = token2id
        self.max_len = max_len
        self.fenci_mode = fenci_mode

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        query, response = self.pairs[idx]

        q_ids = [self.token2id.get(t, UNK_ID) for t in tokenize(query, self.fenci_mode)]
        r_ids = [self.token2id.get(t, UNK_ID) for t in tokenize(response, self.fenci_mode)]

        # 拼接: <|user|> + Q + <|assistant|> + R + <EOS>
        seq = [USER_ID] + q_ids + [ASSISTANT_ID] + r_ids + [EOS_ID]
        seq = seq[:self.max_len]
        seq_len = len(seq)

        # input_ids = 完整序列
        input_ids = torch.tensor(seq, dtype=torch.long)

        # target_ids = 左移一位（预测下一个 token）
        target_ids = torch.full((seq_len,), PAD_ID, dtype=torch.long)
        target_ids[:seq_len - 1] = torch.tensor(seq[1:], dtype=torch.long)

        # 全序列 loss：模型学习完整对话流（用户说话风格 + 助手回复方式）
        loss_mask = torch.ones(seq_len, dtype=torch.bool)

        return {"input_ids": input_ids, "target_ids": target_ids, "loss_mask": loss_mask}


def gpt_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """GPT 批次整理：padding 到批次内最大长度。"""
    batch = sorted(batch, key=lambda x: len(x["input_ids"]), reverse=True)

    input_list, target_list, mask_list = [], [], []
    for item in batch:
        input_list.append(item["input_ids"])
        target_list.append(item["target_ids"])
        mask_list.append(item["loss_mask"])

    input_padded = torch.nn.utils.rnn.pad_sequence(input_list, batch_first=True, padding_value=PAD_ID)
    target_padded = torch.nn.utils.rnn.pad_sequence(target_list, batch_first=True, padding_value=PAD_ID)
    mask_padded = torch.nn.utils.rnn.pad_sequence(mask_list, batch_first=True, padding_value=False)
    pad_mask = (input_padded == PAD_ID)

    return {
        "input_ids": input_padded,       # (B, seq_len)
        "target_ids": target_padded,     # (B, seq_len)
        "loss_mask": mask_padded,        # (B, seq_len) — True = 计入 loss
        "pad_mask": pad_mask,            # (B, seq_len) — True = padding
    }


def create_gpt_dataloaders(
    pairs: list[tuple[str, str]],
    token2id: dict[str, int],
    config: Config,
    train_ratio: float = 0.95,
):
    """创建 GPT 训练/验证 DataLoader。"""
    split = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]

    print(f"[Data-GPT] 训练集: {len(train_pairs)} 序列, 验证集: {len(val_pairs)} 序列")

    mode = getattr(config, "fenci_mode", "jieba")
    train_dataset = GPTDataset(train_pairs, token2id, config.max_len, mode)
    val_dataset = GPTDataset(val_pairs, token2id, config.max_len, mode)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=gpt_collate_fn, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=gpt_collate_fn, num_workers=0, pin_memory=True,
    )
    return train_loader, val_loader


def prepare_gpt_data(config: Config):
    """
    一键 GPT 数据预处理：解析 → 词表 → DataLoader。
    """
    print("=" * 60)
    print(f"GPT 数据预处理开始 — 语料: {config.corpus_name}")
    print("=" * 60)

    all_pairs = _load_corpus_pairs(config, "Data-GPT")

    # 构建 / 加载词汇表
    if os.path.exists(config.vocab_path):
        token2id, id2token = load_vocab(config.vocab_path)
    else:
        token2id, id2token = build_vocab(all_pairs, config)
        save_vocab(token2id, id2token, config.vocab_path)

    # 创建 DataLoader
    train_loader, val_loader = create_gpt_dataloaders(all_pairs, token2id, config)

    print("=" * 60)
    print("GPT 数据预处理完成")
    print("=" * 60)
    return train_loader, val_loader, token2id, id2token


# ============================================================
#  Qwen 预训练模型数据集（使用 Qwen 自带 Tokenizer）
# ============================================================


class QwenDataset(Dataset):
    """
    Qwen 格式数据集，使用 Qwen Tokenizer（字级别）。
    格式: Q + [SEP] + R + [SEP]
    """

    def __init__(self, pairs: list[tuple[str, str]], tokenizer, max_len: int):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        query, response = self.pairs[idx]
        # 规范化文本（去除空格和旧格式分隔符），Qwen tokenizer 需要连续文本
        query = normalize_text(query)
        response = normalize_text(response)
        text = query + self.tokenizer.sep_token + response + self.tokenizer.sep_token

        encoded = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)          # (max_len,)
        attention_mask = encoded["attention_mask"].squeeze(0)  # (max_len,)

        # labels: 同 input_ids，但 padding 位置设为 -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        # Q 部分也不计算 loss（只预测 R 部分）
        # 简化处理：将 [SEP] 之前的部分也 mask 掉
        sep_positions = (input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) >= 1:
            first_sep = sep_positions[0].item() + 1  # 第一个 [SEP] 之后开始计算 loss
            labels[:first_sep] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def qwen_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def create_qwen_dataloaders(
    pairs: list[tuple[str, str]],
    tokenizer,
    config: Config,
    train_ratio: float = 0.95,
):
    split = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]

    print(f"[Data-Qwen] 训练集: {len(train_pairs)} 序列, 验证集: {len(val_pairs)} 序列")

    train_dataset = QwenDataset(train_pairs, tokenizer, config.max_len)
    val_dataset = QwenDataset(val_pairs, tokenizer, config.max_len)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=qwen_collate_fn, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=qwen_collate_fn, num_workers=0, pin_memory=True,
    )
    return train_loader, val_loader


def prepare_qwen_data(config: Config, tokenizer):
    """Qwen 数据预处理（使用 HuggingFace tokenizer）。"""
    print("=" * 60)
    print(f"Qwen 数据预处理开始 — 语料: {config.corpus_name}")
    print("=" * 60)

    all_pairs = _load_corpus_pairs(config, "Data-Qwen")
    train_loader, val_loader = create_qwen_dataloaders(all_pairs, tokenizer, config)

    print("=" * 60)
    print("Qwen 数据预处理完成")
    print("=" * 60)
    return train_loader, val_loader


#测试脚本
if __name__ == "__main__":
    config = Config()
    # 测试 Encoder-Decoder 数据管道
    train_loader, val_loader, token2id, id2token = prepare_data(config)

    batch = next(iter(train_loader))
    print("\n--- Encoder-Decoder 批次 ---")
    print(f"src:        {batch['src'].shape}")
    print(f"tgt_input:  {batch['tgt_input'].shape}")
    print(f"tgt_output: {batch['tgt_output'].shape}")

    sample_src = batch["src"][0].tolist()
    sample_tgt = batch["tgt_output"][0].tolist()
    src_text = "".join(id2token.get(i, "<UNK>") for i in sample_src if i != PAD_ID)
    tgt_text = "".join(id2token.get(i, "<UNK>") for i in sample_tgt if i not in (PAD_ID, EOS_ID))
    print(f"Query:    {src_text}")
    print(f"Response: {tgt_text}")

    # 测试 GPT 数据管道
    print("\n--- GPT 批次 ---")
    gpt_train, gpt_val, _, _ = prepare_gpt_data(config)
    gpt_batch = next(iter(gpt_train))
    print(f"input_ids:  {gpt_batch['input_ids'].shape}")
    print(f"target_ids: {gpt_batch['target_ids'].shape}")
    print(f"loss_mask:  {gpt_batch['loss_mask'].shape}")
    print(f"pad_mask:   {gpt_batch['pad_mask'].shape}")

    # 解码一个 GPT 样本
    ids = gpt_batch["input_ids"][0].tolist()
    tids = gpt_batch["target_ids"][0].tolist()
    mask = gpt_batch["loss_mask"][0].tolist()
    seq_text = "".join(id2token.get(i, "<UNK>") for i in ids if i != PAD_ID)
    tgt_text = "".join(id2token.get(i, "<UNK>") for i in tids if i != PAD_ID)
    print(f"Seq:   {seq_text}")
    print(f"Target:{tgt_text}")
    print(f"Mask:  {''.join('^' if m else ' ' for m in mask)}")
