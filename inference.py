
import os
import math

import torch
import torch.nn.functional as F

from config import Config, PAD_ID, UNK_ID, SOS_ID, EOS_ID
from model import Transformer
from model_gpt import GPT



#  KV-Cache 工具


def _clone_cache(past_key_values: list[dict] | None) -> list[dict] | None:
    """深拷贝 KV-Cache，用于 Beam Search 分支。"""
    if past_key_values is None:
        return None
    return [
        {
            "self": (k.clone(), v.clone()),
            "cross": (ck.clone(), cv.clone()),
        }
        for layer in past_key_values
        for k, v in [layer["self"]]
        for ck, cv in [layer["cross"]]
    ]



#  Beam Search 解码器

class BeamSearchDecoder:
    """Beam Search 解码器，支持 KV-Cache、重复惩罚、N-gram 阻断。"""

    def __init__(
        self,
        model: Transformer,
        token2id: dict[str, int],
        id2token: dict[int, str],
        config: Config,
    ):
        self.model = model
        self.token2id = token2id
        self.id2token = id2token
        self.config = config

        self.sos_id = SOS_ID
        self.eos_id = EOS_ID
        self.pad_id = PAD_ID
        self.unk_id = UNK_ID

        # 重复惩罚系数 (1.0 = 不惩罚, >1.0 惩罚已出现 token)
        self.repetition_penalty: float = 1.2
        # N-gram 阻断大小 (0 = 不阻断)
        self.ngram_block: int = 3

    # 重复惩罚 

    def _apply_repetition_penalty(self, logits: torch.Tensor, generated_ids: list[int]) -> torch.Tensor:
        """对已生成的 token 施加重复惩罚。"""
        if self.repetition_penalty <= 1.0 or not generated_ids:
            return logits

        for tid in set(generated_ids):
            if tid in (self.sos_id, self.pad_id):
                continue
            if logits[tid] < 0:
                logits[tid] *= self.repetition_penalty
            else:
                logits[tid] /= self.repetition_penalty
        return logits

    # ---------- N-gram 阻断 ----------

    def _apply_ngram_block(self, logits: torch.Tensor, generated_ids: list[int]) -> torch.Tensor:
        """禁止生成与前面重复的 N-gram。"""
        if self.ngram_block <= 0 or len(generated_ids) < self.ngram_block:
            return logits

        # 取最后 (n-1) 个 token 作为前缀，查找是否出现过
        prefix = tuple(generated_ids[-(self.ngram_block - 1):])
        for i in range(len(generated_ids) - self.ngram_block + 1):
            ngram = tuple(generated_ids[i:i + self.ngram_block - 1])
            if ngram == prefix and i + self.ngram_block - 1 < len(generated_ids):
                banned = generated_ids[i + self.ngram_block - 1]
                if banned not in (self.sos_id, self.pad_id, self.eos_id):
                    logits[banned] = float("-inf")
        return logits

    #Beam Search

    @torch.no_grad()
    def beam_search(
        self,
        src_ids: list[int],
        beam_size: int | None = None,
        max_len: int | None = None,
        length_penalty: float | None = None,
    ) -> list[tuple[str, float]]:
        """
        Beam Search 解码（使用 KV-Cache 增量解码）。

        返回: [(解码文本, 分数), ...] 按分数降序排列
        """
        beam_size = beam_size or self.config.beam_size
        max_len = max_len or self.config.max_decode_len
        length_penalty = length_penalty or self.config.length_penalty
        device = next(self.model.parameters()).device

        # 编码源序列 
        src_tensor = torch.tensor([src_ids], device=device)
        src_pad_mask = torch.zeros_like(src_tensor, dtype=torch.bool)
        enc_output = self.model.encode(src_tensor, src_pad_mask)

        # 用 <SOS> 获取初始 logits =====
        first_token = torch.tensor([[self.sos_id]], device=device)
        first_pad = torch.zeros(1, 1, dtype=torch.bool, device=device)
        src_cross_pad = torch.zeros(1, src_tensor.size(1), dtype=torch.bool, device=device)

        result = self.model.decode_step(
            first_token, enc_output, first_pad, src_cross_pad,
            past_key_values=None, use_cache=True,
        )
        first_logits, first_cache = result  # (1, 1, V), cache

        log_probs = F.log_softmax(first_logits[0, 0] / self.config.temperature, dim=-1)
        top_scores, top_ids = torch.topk(log_probs, beam_size)

        # 初始化 beam: (seq, score, done, cache)
        beams: list[tuple[list[int], float, bool, list[dict] | None]] = []
        for tid, ts in zip(top_ids.tolist(), top_scores.tolist()):
            seq = [self.sos_id, tid]
            if tid == self.eos_id:
                beams.append((seq, ts, True, None))
            else:
                beams.append((seq, ts, False, first_cache))

        completed: list[tuple[list[int], float]] = []

        # KV-Cache 增量解码
        for _ in range(1, max_len):
            if not beams:
                break

            new_candidates: list[tuple[list[int], float, bool, list[dict] | None]] = []
            all_done = True

            for seq, score, is_done, cache in beams:
                if is_done:
                    new_candidates.append((seq, score, True, None))
                    continue

                all_done = False

                # 只传入最后一个 token + 缓存
                tgt_token = torch.tensor([[seq[-1]]], device=device)
                tgt_pad = torch.zeros(1, 1, dtype=torch.bool, device=device)

                result = self.model.decode_step(
                    tgt_token, enc_output, tgt_pad, src_cross_pad,
                    past_key_values=cache, use_cache=True,
                )
                logits, new_cache = result  # (1, 1, V), new_cache

                log_probs = F.log_softmax(logits[0, 0] / self.config.temperature, dim=-1)

                # 重复惩罚
                log_probs = self._apply_repetition_penalty(log_probs, seq)
                # N-gram 阻断
                log_probs = self._apply_ngram_block(log_probs, seq)

                top_k = min(beam_size * 2, log_probs.size(0))
                top_scores, top_ids = torch.topk(log_probs, top_k)

                for tid, tlp in zip(top_ids.tolist(), top_scores.tolist()):
                    new_seq = seq + [tid]
                    new_score = score + tlp
                    if tid == self.eos_id:
                        completed.append((new_seq[:-1], new_score))
                    else:
                        # 每个分支 clone 一份缓存
                        new_candidates.append((new_seq, new_score, False, _clone_cache(new_cache)))

            if all_done:
                break

            # 长度惩罚 + 剪枝
            scored = []
            for seq, sc, done, ca in new_candidates:
                if not done:
                    lp = ((5.0 + len(seq)) / 6.0) ** length_penalty
                    sc = sc / lp
                scored.append((seq, sc, done, ca))

            scored.sort(key=lambda x: x[1], reverse=True)
            beams = scored[:beam_size]

        # 收集未完成的 beam
        for seq, score, _, _ in beams:
            if len(seq) > 1:
                completed.append((seq[1:], score))

        if not completed:
            return [("", 0.0)]

        completed.sort(key=lambda x: x[1], reverse=True)
        return [(self._decode_ids(seq), sc) for seq, sc in completed]

    # ---------- 采样解码 ----------

    @torch.no_grad()
    def sample(
        self,
        src_ids: list[int],
        max_len: int | None = None,
        temperature: float | None = None,
        top_k: int = 0,
        top_p: float = 0.0,
    ) -> str:
        """
        采样解码（支持 Temperature、Top-K、Top-P、KV-Cache、重复惩罚、N-gram 阻断）。
        """
        max_len = max_len or self.config.max_decode_len
        temperature = temperature or self.config.temperature
        device = next(self.model.parameters()).device

        src_tensor = torch.tensor([src_ids], device=device)
        src_pad_mask = torch.zeros_like(src_tensor, dtype=torch.bool)
        enc_output = self.model.encode(src_tensor, src_pad_mask)
        src_cross_pad = torch.zeros(1, src_tensor.size(1), dtype=torch.bool, device=device)

        # 第 1 步
        tgt_token = torch.tensor([[self.sos_id]], device=device)
        tgt_pad = torch.zeros(1, 1, dtype=torch.bool, device=device)

        result = self.model.decode_step(
            tgt_token, enc_output, tgt_pad, src_cross_pad,
            past_key_values=None, use_cache=True,
        )
        logits, cache = result
        logits = logits[0, 0] / temperature
        next_token = self._sample_token(logits, top_k, top_p)
        if next_token == self.eos_id:
            return ""

        generated = [self.sos_id, next_token]

        # 后续步骤
        for _ in range(1, max_len):
            tgt_token = torch.tensor([[generated[-1]]], device=device)
            tgt_pad = torch.zeros(1, 1, dtype=torch.bool, device=device)

            result = self.model.decode_step(
                tgt_token, enc_output, tgt_pad, src_cross_pad,
                past_key_values=cache, use_cache=True,
            )
            logits, cache = result
            logits = logits[0, 0] / temperature

            # 重复惩罚
            logits = self._apply_repetition_penalty(logits, generated)
            # N-gram 阻断
            logits = self._apply_ngram_block(logits, generated)

            next_token = self._sample_token(logits, top_k, top_p)
            if next_token == self.eos_id:
                break
            generated.append(next_token)

        return self._decode_ids(generated[1:])

    def _sample_token(self, logits: torch.Tensor, top_k: int, top_p: float) -> int:
        """从 logits 中采样一个 token（带 Top-K / Top-P 过滤）。"""
        # Top-K
        if top_k > 0:
            topk_vals, _ = torch.topk(logits, min(top_k, logits.size(0)))
            logits[logits < topk_vals[-1]] = float("-inf")

        # Top-P
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            logits[sorted_indices[remove]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()

    # ---------- 工具 ----------

    def _decode_ids(self, token_ids: list[int]) -> str:
        """将 token ID 序列解码为文本（直接拼接，因为输入已分词）。"""
        tokens = []
        for tid in token_ids:
            if tid in (self.pad_id, self.eos_id, self.sos_id):
                continue
            token = self.id2token.get(tid, "<UNK>")
            tokens.append(token)
        return "".join(tokens)


# ============================================================
#  交互式聊天
# ============================================================

# ============================================================
#  GPT 推理（带对话记忆）
# ============================================================

class GPTDecoder:
    """GPT Decoder-Only 的 Beam Search 解码器。"""

    def __init__(self, model: GPT, token2id: dict, id2token: dict, config: Config):
        self.model = model
        self.token2id = token2id
        self.id2token = id2token
        self.config = config
        self.sos_id, self.eos_id, self.pad_id, self.unk_id = SOS_ID, EOS_ID, PAD_ID, UNK_ID
        self.repetition_penalty: float = 1.2
        self.ngram_block: int = 3

    def _apply_repetition_penalty(self, logits: torch.Tensor, generated_ids: list[int]) -> torch.Tensor:
        if self.repetition_penalty <= 1.0 or not generated_ids:
            return logits
        for tid in set(generated_ids):
            if tid in (self.sos_id, self.pad_id):
                continue
            if logits[tid] < 0:
                logits[tid] *= self.repetition_penalty
            else:
                logits[tid] /= self.repetition_penalty
        return logits

    def _apply_ngram_block(self, logits: torch.Tensor, generated_ids: list[int]) -> torch.Tensor:
        if self.ngram_block <= 0 or len(generated_ids) < self.ngram_block:
            return logits
        prefix = tuple(generated_ids[-(self.ngram_block - 1):])
        for i in range(len(generated_ids) - self.ngram_block + 1):
            ngram = tuple(generated_ids[i:i + self.ngram_block - 1])
            if ngram == prefix and i + self.ngram_block - 1 < len(generated_ids):
                banned = generated_ids[i + self.ngram_block - 1]
                if banned not in (self.sos_id, self.pad_id, self.eos_id):
                    logits[banned] = float("-inf")
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_len: int | None = None,
        temperature: float | None = None,
        beam_size: int | None = None,
        use_sample: bool = False,
    ) -> str:
        """
        GPT 自回归生成（支持 Beam Search 和采样）。

        prompt_ids: 已包含对话历史的 token 序列
        返回: 解码后的生成文本（不含 prompt）
        """
        max_len = max_len or self.config.max_decode_len
        temperature = temperature or self.config.temperature
        beam_size = beam_size or self.config.beam_size
        device = next(self.model.parameters()).device

        # 编码 prompt（一次）
        prompt_tensor = torch.tensor([prompt_ids], device=device)
        prompt_pad = torch.zeros(1, len(prompt_ids), dtype=torch.bool, device=device)
        result = self.model.decode_step(prompt_tensor, prompt_pad, None, use_cache=True)
        _, cache = result  # 只取 KV-Cache，prompt 自身 logits 不需要

        # Beam Search
        beams: list[tuple[list[int], float, bool, list | None]] = [([], 0.0, False, cache)]
        completed: list[tuple[list[int], float]] = []

        for _ in range(max_len):
            new_candidates = []
            all_done = True

            for gen_seq, score, is_done, bcache in beams:
                if is_done:
                    new_candidates.append((gen_seq, score, True, None))
                    continue
                all_done = False

                # 只传最新 token + 缓存
                if not gen_seq:
                    # 首步：取 prompt 最后一个 logit（已编码，用 dummy forward 取）
                    last_tok = torch.tensor([[prompt_ids[-1]]], device=device)
                    lpad = torch.zeros(1, 1, dtype=torch.bool, device=device)
                    r = self.model.decode_step(last_tok, lpad, bcache, use_cache=True)
                    logits, new_bcache = r
                else:
                    last_tok = torch.tensor([[gen_seq[-1]]], device=device)
                    lpad = torch.zeros(1, 1, dtype=torch.bool, device=device)
                    r = self.model.decode_step(last_tok, lpad, bcache, use_cache=True)
                    logits, new_bcache = r

                log_probs = F.log_softmax(logits[0, -1] / temperature, dim=-1)
                log_probs = self._apply_repetition_penalty(log_probs, prompt_ids + gen_seq)
                log_probs = self._apply_ngram_block(log_probs, prompt_ids + gen_seq)

                top_k = min(beam_size * 2, log_probs.size(0))
                top_scores, top_ids = torch.topk(log_probs, top_k)

                for tid, tlp in zip(top_ids.tolist(), top_scores.tolist()):
                    if tid == self.eos_id:
                        completed.append((gen_seq, score + tlp))
                    else:
                        new_candidates.append((gen_seq + [tid], score + tlp, False,
                                               _clone_gpt_cache(new_bcache)))

            if all_done:
                break

            scored = [(s, sc, d, c) for s, sc, d, c in new_candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            beams = scored[:beam_size]

        for gen_seq, score, _, _ in beams:
            if gen_seq:
                completed.append((gen_seq, score))

        if not completed:
            return ""

        completed.sort(key=lambda x: x[1], reverse=True)

        # Beam Search: 返回最佳
        best = completed[0][0]
        return self._decode_ids(best)

    def _decode_ids(self, token_ids: list[int]) -> str:
        tokens = [self.id2token.get(t, "<UNK>") for t in token_ids
                  if t not in (self.pad_id, self.sos_id)]
        return "".join(tokens)


def _clone_gpt_cache(cache: list | None) -> list | None:
    """深拷贝 GPT KV-Cache（每层只有一个 (K,V) 对，无 cross-attn）。"""
    if cache is None:
        return None
    return [(k.clone(), v.clone()) for k, v in cache]


class GPTChatBot:
    """GPT 聊天机器人 — 带短期对话记忆。"""

    def __init__(self, model_path: str, config: Config):
        self.config = config

        import data_loader
        self.token2id, self.id2token = data_loader.load_vocab(config.vocab_path)
        config.vocab_size = len(self.token2id)

        self.model = GPT(config)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(config.device)
        self.model.eval()

        self.decoder = GPTDecoder(self.model, self.token2id, self.id2token, config)

        # 短期记忆: [(Q, R), ...]
        self.history: list[tuple[str, str]] = []

        print(f"[GPTChatBot] 模型已加载: {model_path}")
        print(f"[GPTChatBot] 词汇表大小: {len(self.token2id)}")
        val_ppl = checkpoint.get("val_ppl", "N/A")
        print(f"[GPTChatBot] 验证 PPL: {val_ppl}")

    def _preprocess(self, text: str) -> list[int]:
        if "/" in text:
            tokens = text.split("/")
        else:
            tokens = list(text)
        tokens = [t.strip() for t in tokens if t.strip()]
        return [self.token2id.get(t, UNK_ID) for t in tokens]

    def _build_prompt(self, user_input: str) -> list[int]:
        """构建完整 prompt：历史对话 + 当前输入。"""
        prompt: list[int] = []

        # 限制历史轮数
        max_hist = self.config.max_history
        recent = self.history[-max_hist:] if max_hist > 0 else self.history

        for q, r in recent:
            prompt.append(SOS_ID)
            prompt.extend(self._preprocess(q))
            prompt.append(EOS_ID)
            prompt.extend(self._preprocess(r))
            prompt.append(EOS_ID)

        # 当前输入
        prompt.append(SOS_ID)
        prompt.extend(self._preprocess(user_input))
        prompt.append(EOS_ID)

        return prompt[:self.config.max_len]

    def reply(self, text: str, use_beam: bool = True, use_sample: bool = False) -> str:
        prompt_ids = self._build_prompt(text)
        if len(prompt_ids) < 2:
            return "请输入有效内容~"

        response = self.decoder.generate(
            prompt_ids,
            max_len=self.config.max_decode_len,
            temperature=self.config.temperature,
            beam_size=self.config.beam_size if use_beam else 1,
            use_sample=use_sample,
        )

        # 存入记忆
        self.history.append((text, response))
        return response

    def clear_history(self):
        """清空对话记忆。"""
        self.history = []
        print("[记忆] 已清空")

    def chat(self):
        """启动交互式聊天（带记忆）。"""
        print("\n" + "=" * 60)
        print("  GPT 中文聊天机器人 (带短期记忆)")
        print("  输入 'quit' 或 'exit' 退出")
        print("  输入 '/clear' 清空对话历史")
        print("  输入 '/history' 查看当前记忆")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见~")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("再见~")
                break
            if user_input == "/clear":
                self.clear_history()
                continue
            if user_input == "/history":
                if not self.history:
                    print("[记忆] 暂无对话历史")
                else:
                    for i, (q, r) in enumerate(self.history, 1):
                        print(f"  {i}. 你: {q}")
                        print(f"     小黄鸡: {r}")
                continue

            response = self.reply(user_input, use_beam=True)
            print(f"小黄鸡: {response}\n")


# ============================================================
#  原有 Encoder-Decoder ChatBot
# ============================================================

class ChatBot:
    """交互式中文聊天机器人。"""

    def __init__(self, model_path: str, config: Config):
        self.config = config

        import data_loader
        self.token2id, self.id2token = data_loader.load_vocab(config.vocab_path)

        config.vocab_size = len(self.token2id)

        self.model = Transformer(config)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(config.device)
        self.model.eval()

        self.decoder = BeamSearchDecoder(self.model, self.token2id, self.id2token, config)

        print(f"[ChatBot] 模型已加载: {model_path}")
        print(f"[ChatBot] 词汇表大小: {len(self.token2id)}")
        val_ppl = checkpoint.get("val_ppl", "N/A")
        print(f"[ChatBot] 验证 PPL: {val_ppl}")

    def _preprocess(self, text: str) -> list[int]:
        if "/" in text:
            tokens = text.split("/")
        else:
            tokens = list(text)
        tokens = [t.strip() for t in tokens if t.strip()]
        ids = [self.token2id.get(t, UNK_ID) for t in tokens]
        return ids[: self.config.max_len]

    def reply(self, text: str, use_beam: bool = True, use_sample: bool = False) -> str:
        src_ids = self._preprocess(text)
        if not src_ids:
            return "请输入有效内容~"

        if use_sample:
            return self.decoder.sample(
                src_ids,
                temperature=self.config.temperature,
                top_k=50,
                top_p=0.9,
            )
        elif use_beam:
            results = self.decoder.beam_search(src_ids)
            return results[0][0] if results else "我不知道怎么回...T_T"
        else:
            results = self.decoder.beam_search(src_ids, beam_size=1)
            return results[0][0] if results else "我不知道怎么回...T_T"

    def chat(self):
        """启动交互式聊天循环"""
        print("\n" + "=" * 60)
        print("  🐤 小黄鸡 — Transformer 中文聊天机器人")
        print("  输入 'quit' 或 'exit' 退出")
        print("  输入 '/beam' 切换 Beam Search")
        print("  输入 '/sample' 切换随机采样")
        print("=" * 60 + "\n")

        mode = "beam"
        while True:
            try:
                user_input = input("你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见~ 🐤")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("再见~ 🐤")
                break
            if user_input == "/beam":
                mode = "beam"; print("[模式] Beam Search 解码"); continue
            if user_input == "/sample":
                mode = "sample"; print("[模式] 随机采样解码"); continue
            if user_input == "/greedy":
                mode = "greedy"; print("[模式] 贪心解码"); continue

            if mode == "beam":
                response = self.reply(user_input, use_beam=True, use_sample=False)
            elif mode == "sample":
                response = self.reply(user_input, use_beam=False, use_sample=True)
            else:
                response = self.reply(user_input, use_beam=False, use_sample=False)

            print(f"小黄鸡: {response}\n")



#  命令行入口

def main():
    config = Config()

    if config.device == "cuda" and not torch.cuda.is_available():
        print("[Info] CUDA 不可用，回退到 CPU。")
        config.device = "cpu"

    model_path = os.path.join(config.model_save_dir, "best_model.pt")
    if not os.path.exists(model_path):
        if os.path.exists(config.model_save_dir):
            ckpts = [f for f in os.listdir(config.model_save_dir) if f.endswith(".pt")]
            if ckpts:
                ckpts.sort()
                model_path = os.path.join(config.model_save_dir, ckpts[0])
                print(f"[Warning] 未找到 best_model.pt，使用: {ckpts[0]}")
            else:
                print(f"[Error] 未找到模型文件在 {config.model_save_dir}")
                print("请先运行 train.py 训练模型。")
                return
        else:
            print(f"[Error] 模型目录不存在: {config.model_save_dir}")
            print("请先运行 train.py 训练模型。")
            return

    if not os.path.exists(config.vocab_path):
        print(f"[Error] 词汇表不存在: {config.vocab_path}")
        return

    if config.model_type == "gpt":
        bot = GPTChatBot(model_path, config)
    else:
        bot = ChatBot(model_path, config)
    bot.chat()


if __name__ == "__main__":
    main()
