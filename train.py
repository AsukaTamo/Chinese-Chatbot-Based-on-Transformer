import math
import os
import time
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.amp import GradScaler, autocast

from config import Config, PAD_ID, SOS_ID, EOS_ID
from data_loader import prepare_data, prepare_gpt_data, prepare_qwen_data
from model import Transformer
from model_gpt import GPT
from model_qwen import PretrainedLM
from model_m import GPT as GPT_m


#  Noam 学习率调度器

class NoamScheduler:
    """
    "Attention is All You Need" 中的学习率调度:
        lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """

    def __init__(self, optimizer: Adam, d_model: int, warmup_steps: int):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0
        self._rate = 0.0

    def step(self):
        """更新学习率并执行 optimizer.step()。"""
        self._step += 1
        rate = self._compute_rate()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = rate
        self._rate = rate

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _compute_rate(self) -> float:
        step = self._step
        arg1 = step ** (-0.5)
        arg2 = step * (self.warmup_steps ** (-1.5))
        return (self.d_model ** (-0.5)) * min(arg1, arg2)

    @property
    def lr(self) -> float:
        return self._rate


#  标签平滑交叉熵损失

class LabelSmoothingLoss(nn.Module):

    def __init__(self, vocab_size: int, smoothing: float = 0.1, ignore_index: int = PAD_ID):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (B * tgt_len, vocab_size) 或 (B, tgt_len, vocab_size)
        target: (B * tgt_len,) 或 (B, tgt_len)

        内存高效实现：不创建 (N, V) 大小的中间张量，
        用数学展开避免 full_like 分配。
        """
        if logits.dim() == 3:
            B, T, V = logits.shape
            logits = logits.reshape(B * T, V)
            target = target.reshape(B * T)

        log_probs = F.log_softmax(logits, dim=-1)            # (N, V)

        # --- 标签平滑的数学展开 ---
        # CE = -(1-ε)·log_p[true] - Σ_{k≠true} ε/(V-1)·log_p[k]
        #    = -(1-ε)·log_p[true] - ε/(V-1)·(Σ_all log_p[k] - log_p[true])
        #    = -(1-ε - ε/(V-1))·log_p[true] - ε/(V-1)·Σ_all log_p[k]
        # 避免分配平滑分布矩阵 (N, V)，只需 (N,) 的中间量。

        epsilon = self.smoothing / (self.vocab_size - 1)
        true_weight = self.confidence - epsilon                # 标量

        nll_true = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)   # (N,)  -log_p[true]
        sum_all  = log_probs.sum(dim=-1)                                  # (N,)  Σ_all log_p[k]

        loss = -true_weight * nll_true - epsilon * sum_all      # (N,)  逐 token 损失

        # 忽略 PAD 位置
        mask = (target != self.ignore_index).float()            # (N,)
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        return loss



#  训练 / 验证循环

def train_epoch(
    model: Transformer,
    dataloader,
    criterion: LabelSmoothingLoss,
    scheduler: NoamScheduler,
    config: Config,
    epoch: int,
    scaler: GradScaler | None = None,
) -> float:
    """训练一个 epoch，返回平均损失。"""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()

    for step, batch in enumerate(dataloader):
        src = batch["src"].to(config.device)
        tgt_input = batch["tgt_input"].to(config.device)
        tgt_output = batch["tgt_output"].to(config.device)
        src_pad_mask = batch["src_mask"].to(config.device)
        tgt_pad_mask = batch["tgt_pad_mask"].to(config.device)

        scheduler.zero_grad()

        use_amp = scaler is not None and config.device == "cuda"
        if use_amp:
            with autocast("cuda"):
                logits = model(src, tgt_input, src_pad_mask, tgt_pad_mask)
                loss = criterion(logits, tgt_output)
            scaler.scale(loss).backward()
            scaler.unscale_(scheduler.optimizer)
        else:
            logits = model(src, tgt_input, src_pad_mask, tgt_pad_mask)
            loss = criterion(logits, tgt_output)
            loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        if use_amp:
            scaler.step(scheduler.optimizer)
            scaler.update()
        else:
            scheduler.optimizer.step()

        scheduler.step()  # 更新学习率

        # 统计
        batch_tokens = (tgt_output != PAD_ID).sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

        if (step + 1) % config.log_every == 0:
            elapsed = time.time() - start_time
            avg_loss = total_loss / max(total_tokens, 1)
            ppl = math.exp(min(avg_loss, 10))  # 防止溢出
            print(f"  Epoch {epoch} | Step {step+1:5d} | "
                  f"Loss: {avg_loss:.4f} | PPL: {ppl:6.1f} | "
                  f"LR: {scheduler.lr:.6f} | Time: {elapsed:.0f}s")

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss


@torch.no_grad()
def validate(
    model: Transformer,
    dataloader,
    criterion: LabelSmoothingLoss,
    config: Config,
) -> float:
    """验证，返回平均损失。"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        src = batch["src"].to(config.device)
        tgt_input = batch["tgt_input"].to(config.device)
        tgt_output = batch["tgt_output"].to(config.device)
        src_pad_mask = batch["src_mask"].to(config.device)
        tgt_pad_mask = batch["tgt_pad_mask"].to(config.device)

        logits = model(src, tgt_input, src_pad_mask, tgt_pad_mask)
        loss = criterion(logits, tgt_output)

        batch_tokens = (tgt_output != PAD_ID).sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss


def train(config: Config):

    print("=" * 60)
    print("Transformer 中文聊天机器人 — 训练")
    print("=" * 60)

    # 设备
    if config.device == "cuda" and torch.cuda.is_available():
        config.device = "cuda"
        print(f"[Device] CUDA: {torch.cuda.get_device_name(0)}")
    else:
        config.device = "cpu"
        print("[Device] 使用 CPU")

    # 数据
    train_loader, val_loader, token2id, _ = prepare_data(config)
    # 同步 vocab_size
    actual_vocab_size = len(token2id)
    config.vocab_size = actual_vocab_size

    # 模型
    model = Transformer(config).to(config.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] 参数量: {total_params:,}")

    # 损失函数
    criterion = LabelSmoothingLoss(
        vocab_size=actual_vocab_size,
        smoothing=config.label_smoothing,
        ignore_index=PAD_ID,
    )

    # 优化器
    optimizer = Adam(
        model.parameters(),
        lr=0.0,  # 由 scheduler 控制
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = NoamScheduler(optimizer, config.d_model, config.warmup_steps)

    # 混合精度
    scaler = GradScaler() if config.device == "cuda" else None

    # 创建保存目录
    os.makedirs(config.model_save_dir, exist_ok=True)

    # 训练历史
    history = {"train_loss": [], "val_loss": [], "val_ppl": []}
    best_val_loss = float("inf")

    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()

        # 训练
        train_loss = train_epoch(model, train_loader, criterion, scheduler, config, epoch, scaler)

        # 验证
        val_loss = validate(model, val_loader, criterion, config)
        val_ppl = math.exp(min(val_loss, 10))

        epoch_time = time.time() - epoch_start

        print("-" * 60)
        print(f"Epoch {epoch:2d}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val PPL: {val_ppl:6.1f} | "
              f"Time: {epoch_time:.0f}s")
        print("-" * 60)

        # 记录历史
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "model_type": "transformer",
                "d_model": config.d_model,
                "vocab_size": actual_vocab_size,
                "config": config,
                "history": history,
            }
            torch.save(checkpoint, os.path.join(config.model_save_dir, "best_model.pt"))
            print(f"  ✓ 最佳模型已保存 (Val Loss: {val_loss:.4f})")

        # 定期保存
        if epoch % config.save_every_epoch == 0:
            torch.save(checkpoint, os.path.join(config.model_save_dir, f"checkpoint_epoch{epoch}.pt"))

    # 保存训练历史
    with open(os.path.join(config.model_save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"训练完成！最佳验证 Loss: {best_val_loss:.4f} (PPL: {math.exp(min(best_val_loss, 10)):.1f})")
    print("=" * 60)


def train_gpt(config: Config, model_cls=None):
    """GPT Decoder-Only 模型训练。

    参数:
        model_cls: 模型类，默认 model_gpt.GPT；可传入 model_m.GPT 使用 RoPE。
    """
    if model_cls is None:
        model_cls = GPT  # model_gpt.GPT

    model_label = getattr(model_cls, "__name__", "GPT")
    print("=" * 60)
    print(f"GPT Decoder-Only 中文聊天机器人 — 训练 ({model_label})")
    print("=" * 60)

    # 设备
    if config.device == "cuda" and torch.cuda.is_available():
        config.device = "cuda"
        print(f"[Device] CUDA: {torch.cuda.get_device_name(0)}")
    else:
        config.device = "cpu"
        print("[Device] 使用 CPU")

    # 数据
    train_loader, val_loader, token2id, _ = prepare_gpt_data(config)
    actual_vocab_size = len(token2id)
    config.vocab_size = actual_vocab_size

    # 模型
    model = model_cls(config).to(config.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {model_label} 参数量: {total_params:,}")

    # 损失函数（标准 CE，配合 loss_mask 使用）
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="none")

    # 优化器
    optimizer = Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, config.d_model, config.warmup_steps)

    scaler = GradScaler() if config.device == "cuda" else None
    os.makedirs(config.model_save_dir, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_ppl": []}
    best_val_loss = float("inf")

    print("\n" + "=" * 60)
    print(f"开始训练 ({model_label})")
    print("=" * 60)

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()

        # ---- Train ----
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(config.device)
            target_ids = batch["target_ids"].to(config.device)
            loss_mask = batch["loss_mask"].to(config.device)
            pad_mask = batch["pad_mask"].to(config.device)

            scheduler.zero_grad()

            use_amp = scaler is not None
            if use_amp:
                with autocast("cuda"):
                    logits = model(input_ids, pad_mask)            # (B, seq, V)
                    loss = criterion(logits.permute(0, 2, 1), target_ids)  # (B, seq)
                    loss = (loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)
                scaler.scale(loss).backward()
                scaler.unscale_(scheduler.optimizer)
            else:
                logits = model(input_ids, pad_mask)
                loss = criterion(logits.permute(0, 2, 1), target_ids)
                loss = (loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            if use_amp:
                scaler.step(scheduler.optimizer)
                scaler.update()
            else:
                scheduler.optimizer.step()
            scheduler.step()

            batch_tokens = loss_mask.sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

            if (step + 1) % config.log_every == 0:
                avg_loss = total_loss / max(total_tokens, 1)
                ppl = math.exp(min(avg_loss, 10))
                print(f"  Epoch {epoch} | Step {step+1:5d} | "
                      f"Loss: {avg_loss:.4f} | PPL: {ppl:6.1f} | "
                      f"LR: {scheduler.lr:.6f} | Time: {time.time()-epoch_start:.0f}s")

        train_loss = total_loss / max(total_tokens, 1)

        # ---- Val ----
        model.eval()
        val_loss = 0.0
        val_tokens = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(config.device)
                target_ids = batch["target_ids"].to(config.device)
                loss_mask = batch["loss_mask"].to(config.device)
                pad_mask = batch["pad_mask"].to(config.device)

                logits = model(input_ids, pad_mask)
                loss = criterion(logits.permute(0, 2, 1), target_ids)
                loss = (loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)

                vt = loss_mask.sum().item()
                val_loss += loss.item() * vt
                val_tokens += vt

        val_loss = val_loss / max(val_tokens, 1)
        val_ppl = math.exp(min(val_loss, 10))
        epoch_time = time.time() - epoch_start

        print("-" * 60)
        print(f"Epoch {epoch:2d}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val PPL: {val_ppl:6.1f} | "
              f"Time: {epoch_time:.0f}s")
        print("-" * 60)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss, "val_ppl": val_ppl,
                "model_type": "gpt", "d_model": config.d_model,
                "vocab_size": actual_vocab_size,
                "config": config, "history": history,
            }
            torch.save(checkpoint, os.path.join(config.model_save_dir, "best_model.pt"))
            print(f"  + 最佳模型已保存 (Val Loss: {val_loss:.4f})")

        if epoch % config.save_every_epoch == 0:
            torch.save(checkpoint, os.path.join(config.model_save_dir, f"checkpoint_epoch{epoch}.pt"))

    with open(os.path.join(config.model_save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"{model_label} 训练完成！最佳验证 Loss: {best_val_loss:.4f} (PPL: {math.exp(min(best_val_loss, 10)):.1f})")
    print("=" * 60)


def train_qwen(config: Config):
    """Qwen 预训练模型微调。"""
    print("=" * 60)
    print("Qwen-Chinese 预训练模型微调 (Pro)")
    print("=" * 60)

    if not torch.cuda.is_available():
        config.device = "cpu"
    print(f"[Device] {config.device}")

    # 加载预训练模型
    model = PretrainedLM(model_name=config.qwen_model_name)
    if config.qwen_freeze_layers > 0:
        model.freeze_bottom_layers(config.qwen_freeze_layers)
    model.to(config.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Qwen 参数量: {total_params:,}")

    # 数据
    tokenizer = model.tokenizer
    train_loader, val_loader = prepare_qwen_data(config, tokenizer)

    # 优化器（微调用小学习率）
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5, betas=(0.9, 0.999), eps=1e-8,
    )
    scaler = GradScaler() if config.device == "cuda" else None

    # 保存目录
    qwen_save_dir = os.path.join(config.model_save_dir, "qwen_finetuned")
    os.makedirs(qwen_save_dir, exist_ok=True)

    history: dict = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    print("\n" + "=" * 60)
    print("开始微调 Qwen")
    print("=" * 60)

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()

        # ---- Train ----
        model.train()
        total_loss = 0.0
        total_steps = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["labels"].to(config.device)

            optimizer.zero_grad()

            use_amp = scaler is not None
            if use_amp:
                with autocast("cuda"):
                    out = model(input_ids, attention_mask, labels)
                    loss = out.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                out = model(input_ids, attention_mask, labels)
                loss = out.loss
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            total_loss += loss.item()
            total_steps += 1

            if (step + 1) % config.log_every == 0:
                avg_loss = total_loss / total_steps
                ppl = math.exp(min(avg_loss, 10))
                print(f"  Epoch {epoch} | Step {step+1:5d} | "
                      f"Loss: {avg_loss:.4f} | PPL: {ppl:6.1f} | "
                      f"Time: {time.time()-epoch_start:.0f}s")

        train_loss = total_loss / max(total_steps, 1)

        # ---- Val ----
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(config.device)
                attention_mask = batch["attention_mask"].to(config.device)
                labels = batch["labels"].to(config.device)
                out = model(input_ids, attention_mask, labels)
                val_loss_sum += out.loss.item()
                val_steps += 1

        val_loss = val_loss_sum / max(val_steps, 1)
        val_ppl = math.exp(min(val_loss, 10))
        epoch_time = time.time() - epoch_start

        print("-" * 60)
        print(f"Epoch {epoch:2d}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val PPL: {val_ppl:6.1f} | Time: {epoch_time:.0f}s")
        print("-" * 60)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(qwen_save_dir)
            print(f"  + 最佳模型已保存至: {qwen_save_dir}")

    with open(os.path.join(qwen_save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Qwen 微调完成！最佳 Val Loss: {best_val_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transformer 中文聊天机器人 — 训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train.py                                          # 使用 config.py 中的默认配置
  python train.py --corpora xiaohuangji                    # 单语料训练
  python train.py --corpora xiaohuangji,weibo              # 多语料联合训练
  python train.py --corpora xiaohuangji --epoch 50 --batch 64   # 自定义训练参数
        """,
    )

    # ========== 语料 & 训练参数 ==========
    parser.add_argument(
        "--corpora", type=str, default=None,
        help="语料库名称，多语料用逗号分隔（例: xiaohuangji 或 xiaohuangji,weibo）"
    )
    parser.add_argument(
        "--epoch", type=int, default=None,
        help="训练轮数（覆盖 config.py 中的 epochs）"
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help="批次大小（覆盖 config.py 中的 batch_size）"
    )
    parser.add_argument(
        "--model", type=str, default=None, choices=["transformer", "gpt", "qwen"],
        help="模型架构: transformer (Lite) | gpt (Middle, LLaMA-style+RoPE) | qwen (Pro, 预训练)"
    )
    parser.add_argument(
        "--fenci", type=str, default=None, choices=["jieba", "space"],
        help="分词模式: jieba (默认) | space (LCCC 等已预分词语料)"
    )

    # ========== 其他 ==========
    parser.add_argument(
        "--device", type=str, default=None, choices=["cuda", "cpu"],
        help="训练设备（cuda / cpu），默认使用 config.py 配置"
    )
    parser.add_argument(
        "--resume", type=str, default=None, metavar="PATH",
        help="从指定 checkpoint 恢复训练"
    )

    args = parser.parse_args()

    # ========== 加载 Config 并应用 CLI 覆盖 ==========
    config = Config()

    if args.corpora is not None:
        config.corpora = tuple(args.corpora.split(","))
        config._initialized = False
        config.__post_init__()

    if args.model is not None:
        config.model_type = args.model

    if args.epoch is not None:
        config.epochs = args.epoch
    if args.batch is not None:
        config.batch_size = args.batch
    if args.device is not None:
        config.device = args.device
    if args.fenci is not None:
        config.fenci_mode = args.fenci

    # 打印运行配置
    print(f"[Config] 模型类型: {config.model_type}")
    print(f"[Config] 语料: {config.corpus_name}")
    print(f"[Config] 数据文件: {config.data_paths}")
    print(f"[Config] 词表: {config.vocab_path}")
    print(f"[Config] 模型保存: {config.model_save_dir}")
    print(f"[Config] d_model={config.d_model}  batch={config.batch_size}  epochs={config.epochs}")
    print(f"[Config] 分词模式: {config.fenci_mode}")

    if config.model_type == "qwen":
        train_qwen(config)
    elif config.model_type == "gpt":
        train_gpt(config)
    elif config.model_type == "rope":
        train_gpt(config, model_cls=GPT_m)
    else:
        train(config)
