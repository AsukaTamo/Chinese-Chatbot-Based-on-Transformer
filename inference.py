import os
import math
import json

import torch
import torch.nn.functional as F

from config import Config, PAD_ID, UNK_ID, SOS_ID, EOS_ID
from model import Transformer



#  Beam Search 解码器

class BeamSearchDecoder:


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

    @torch.no_grad()
    def beam_search(
        self,
        src_ids: list[int],
        beam_size: int | None = None,
        max_len: int | None = None,
        length_penalty: float | None = None,
    ) -> list[tuple[str, float]]:
        """
        Beam Search 解码。

        参数:
            src_ids: 源序列 token ID 列表
            beam_size: Beam 宽度
            max_len: 最大解码长度
            length_penalty: 长度惩罚系数（<1 鼓励短句，>1 鼓励长句）

        返回: [(解码文本, 分数), ...] 按分数降序排列
        """
        beam_size = beam_size or self.config.beam_size
        max_len = max_len or self.config.max_decode_len
        length_penalty = length_penalty or self.config.length_penalty

        device = next(self.model.parameters()).device

        # 编码源序列
        src_tensor = torch.tensor([src_ids], device=device)           # (1, src_len)
        src_pad_mask = torch.zeros_like(src_tensor, dtype=torch.bool) # (1, src_len)
        enc_output = self.model.encode(src_tensor, src_pad_mask)      # (1, src_len, d_model)

        # 初始化 Beam: 每个候选为 (token_ids, log_prob_sum, is_done)
        beams: list[tuple[list[int], float, bool]] = [([self.sos_id], 0.0, False)]
        completed: list[tuple[list[int], float]] = []

        for _ in range(max_len):
            if not beams:
                break

            new_candidates: list[tuple[list[int], float, bool]] = []
            all_done = True

            for seq, score, is_done in beams:
                if is_done:
                    new_candidates.append((seq, score, True))
                    continue

                all_done = False

                # 传入完整已生成序列（含 SOS），自注意力需要看到全部历史
                tgt_token = torch.tensor([seq], device=device)          # (1, len(seq))
                tgt_pad_mask = torch.zeros(1, len(seq), dtype=torch.bool, device=device)
                src_mask_for_cross = torch.zeros(1, src_tensor.size(1), dtype=torch.bool, device=device)

                logits = self.model.decode_step(
                    tgt_token, enc_output, tgt_pad_mask, src_mask_for_cross
                )  # (1, len(seq), vocab_size)

                # 取最后一个位置的 logits 作为下一 token 的预测
                log_probs = F.log_softmax(logits[0, -1] / self.config.temperature, dim=-1)

                # 取 top-k
                top_k = min(beam_size * 2, log_probs.size(0))
                top_scores, top_ids = torch.topk(log_probs, top_k)

                for token_id, token_log_prob in zip(top_ids.tolist(), top_scores.tolist()):
                    new_seq = seq + [token_id]
                    new_score = score + token_log_prob

                    if token_id == self.eos_id:
                        completed.append((new_seq[:-1], new_score))  # 去掉 EOS
                    else:
                        new_candidates.append((new_seq, new_score, False))

            if all_done:
                break

            # 长度惩罚 + 剪枝
            scored_candidates = []
            for seq, score, is_done in new_candidates:
                if not is_done:
                    lp = ((5.0 + len(seq)) / 6.0) ** length_penalty
                    score = score / lp
                scored_candidates.append((seq, score, is_done))

            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = scored_candidates[:beam_size]

        # 收集所有完成序列
        for seq, score, is_done in beams:
            if not is_done and len(seq) > 1:
                completed.append((seq[1:], score))  # 去掉 SOS

        if not completed:
            # 如果没有完成的，取当前 beam 的第一条
            if beams:
                seq = beams[0][0][1:]  # 去掉 SOS
                completed = [(seq, beams[0][1])]
            else:
                completed = [([], 0.0)]

        # 按分数排序
        completed.sort(key=lambda x: x[1], reverse=True)

        # 解码为文本
        results = []
        for seq, score in completed:
            text = self._decode_ids(seq)
            results.append((text, score))

        return results

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
        采样解码（支持 Temperature、Top-K、Top-P）。

        返回: 解码文本
        """
        max_len = max_len or self.config.max_decode_len
        temperature = temperature or self.config.temperature

        device = next(self.model.parameters()).device

        src_tensor = torch.tensor([src_ids], device=device)
        src_pad_mask = torch.zeros_like(src_tensor, dtype=torch.bool)
        enc_output = self.model.encode(src_tensor, src_pad_mask)

        generated = [self.sos_id]
        for _ in range(max_len):
            # 传入完整已生成序列，自注意力需要看到全部历史
            tgt_token = torch.tensor([generated], device=device)
            tgt_pad_mask = torch.zeros(1, len(generated), dtype=torch.bool, device=device)
            src_mask_for_cross = torch.zeros(1, src_tensor.size(1), dtype=torch.bool, device=device)

            logits = self.model.decode_step(tgt_token, enc_output, tgt_pad_mask, src_mask_for_cross)
            # 取最后一个位置的 logits 作为下一 token 的预测
            logits = logits[0, -1] / temperature

            # Top-K 过滤
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(0)))
                min_val = top_k_vals[-1]
                logits[logits < min_val] = float("-inf")

            # Top-P (Nucleus) 过滤
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            if next_token == self.eos_id:
                break
            generated.append(next_token)

        return self._decode_ids(generated[1:])  # 去掉 SOS

    def _decode_ids(self, token_ids: list[int]) -> str:
        """将 token ID 序列解码为文本（直接拼接，因为输入已分词）。"""
        tokens = []
        for tid in token_ids:
            if tid in (self.pad_id, self.eos_id, self.sos_id):
                continue
            token = self.id2token.get(tid, "<UNK>")
            tokens.append(token)
        return "".join(tokens)  # 中文分词后直接拼接



#  交互式聊天


class ChatBot:
    """交互式中文聊天机器人。"""

    def __init__(self, model_path: str, config: Config):
        self.config = config

        # 加载词汇表
        import data_loader
        self.token2id, self.id2token = data_loader.load_vocab(config.vocab_path)

        # 同步 vocab_size
        config.vocab_size = len(self.token2id)

        # 加载模型
        self.model = Transformer(config)
        # 先加载到 CPU，然后移到目标设备（避免 CUDA 不可用时反序列化失败）
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(config.device)
        self.model.eval()

        # 解码器
        self.decoder = BeamSearchDecoder(self.model, self.token2id, self.id2token, config)

        print(f"[ChatBot] 模型已加载: {model_path}")
        print(f"[ChatBot] 词汇表大小: {len(self.token2id)}")
        val_ppl = checkpoint.get("val_ppl", "N/A")
        print(f"[ChatBot] 验证 PPL: {val_ppl}")

    def _preprocess(self, text: str) -> list[int]:
        """
        预处理用户输入。
        如果输入未分词，按字符切分；否则按 / 切分。
        """
        # 检测是否已分词（包含 /）
        if "/" in text:
            tokens = text.split("/")
        else:
            # 按字符切分（简单中文分词）
            tokens = list(text)

        # 过滤空 token
        tokens = [t.strip() for t in tokens if t.strip()]

        # 转为 ID
        ids = [self.token2id.get(t, UNK_ID) for t in tokens]
        return ids[: self.config.max_len]

    def reply(self, text: str, use_beam: bool = True, use_sample: bool = False) -> str:
        """
        生成回复。

        参数:
            text: 用户输入
            use_beam: 使用 Beam Search（否则用贪心）
            use_sample: 使用温度采样
        """
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
            # 贪心解码 (beam_size=1)
            results = self.decoder.beam_search(src_ids, beam_size=1)
            return results[0][0] if results else "我不知道怎么回...T_T"

    def chat(self):
        """启动交互式聊天循环"""
        print("\n" + "=" * 60)
        print(" Transformer 中文聊天机器人")
        print("  输入 'quit' 或 'exit' 退出")
        print("  输入 '/beam' 切换 Beam Search")
        print("  输入 '/sample' 切换随机采样")
        print("=" * 60 + "\n")

        mode = "beam"
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

            if user_input == "/beam":
                mode = "beam"
                print("[模式切换] Beam Search 解码")
                continue

            if user_input == "/sample":
                mode = "sample"
                print("[模式切换] 随机采样解码")
                continue

            if user_input == "/greedy":
                mode = "greedy"
                print("[模式切换] 贪心解码")
                continue

            # 生成回复
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

    # 自动检测设备：CUDA 不可用时回退到 CPU
    if config.device == "cuda" and not torch.cuda.is_available():
        print("[Info] CUDA 不可用，回退到 CPU。")
        config.device = "cpu"

    # 确定模型路径
    model_path = os.path.join(config.model_save_dir, "best_model.pt")
    if not os.path.exists(model_path):
        # 尝试找其他 checkpoint
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
        print("请先运行 data_loader.py 或 train.py 生成词汇表。")
        return

    bot = ChatBot(model_path, config)
    bot.chat()


if __name__ == "__main__":
    main()
