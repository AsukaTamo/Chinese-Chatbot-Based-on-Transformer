"""
HuggingFace 预训练模型封装 (Pro 版本)

支持任意 Causal LM:
  - Qwen-Chinese:   uer/qwen-chinese-cluecorpussmall (100M)
  - Qwen2-1.5B:     Qwen/Qwen2-1.5B-Instruct   (1.5B, 对话模型)

通过 config.qwen_model_name 切换。
首次运行自动下载权重，缓存至 ~/.cache/huggingface/
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class PretrainedLM(nn.Module):
    """通用 HuggingFace Causal LM 封装，兼容训练/推理接口。"""

    def __init__(self, model_name: str = "uer/qwen-chinese-cluecorpussmall"):
        super().__init__()
        self.model_name = model_name

        print(f"[HF] 加载预训练模型: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )

        # pad_token 处理
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.config = self.model.config
        self.d_model = getattr(self.config, "hidden_size",
                               getattr(self.config, "n_embd", 768))
        self._freeze_layers: int = 0

    @property
    def device(self):
        return next(self.model.parameters()).device

    def freeze_bottom_layers(self, n: int):
        """冻结前 n 层，仅微调顶层 + LM head。"""
        self._freeze_layers = n
        # 获取 transformer layers（兼容 Qwen 和 Qwen2）
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layers = self.model.transformer.h  # Qwen
            for p in self.model.transformer.wte.parameters():
                p.requires_grad = False
            for p in self.model.transformer.wpe.parameters():
                p.requires_grad = False
        elif hasattr(self.model.model, "layers"):
            layers = self.model.model.layers  # Qwen2 / LLaMA
            if hasattr(self.model.model, "embed_tokens"):
                for p in self.model.model.embed_tokens.parameters():
                    p.requires_grad = False
        else:
            print(f"[HF] 无法识别层结构，跳过冻结")
            return

        for i, block in enumerate(layers):
            if i < n:
                for p in block.parameters():
                    p.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[HF] 冻结前 {n} 层，可训练参数: {trainable:,} / {total:,}")

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, prompt_ids, **kwargs):
        return self.model.generate(input_ids=prompt_ids, **kwargs)

    def save_pretrained(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"[HF] 模型已保存至: {path}")

    @classmethod
    def from_pretrained(cls, path: str) -> "PretrainedLM":
        print(f"[HF] 从本地加载: {path}")
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.model_name = path
        instance.model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True,
        )
        instance.tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True,
        )
        if instance.tokenizer.pad_token is None:
            instance.tokenizer.pad_token = instance.tokenizer.eos_token
        instance.model.config.pad_token_id = instance.tokenizer.pad_token_id
        instance.config = instance.model.config
        instance.d_model = getattr(instance.config, "hidden_size",
                                   getattr(instance.config, "n_embd", 768))
        instance._freeze_layers = 0
        return instance

    def apply_chat_template(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        """应用模型自带的 chat template（Qwen2 等有，Qwen 没有则回退到简单格式）。"""
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        # Qwen 没有 chat template，回退
        parts = []
        for m in messages:
            if m["role"] == "user":
                parts.append(f"甲：{m['content']}")
            else:
                parts.append(f"乙：{m['content']}")
        parts.append("乙：")
        return "\n".join(parts)
QwenChinese = PretrainedLM


# ============================================================
#  测试
# ============================================================

if __name__ == "__main__":
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "uer/qwen-chinese-cluecorpussmall"

    print(f"测试: 加载 {model_name} ...\n")
    model = PretrainedLM(model_name)
    print(f"d_model: {model.d_model}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试 tokenizer
    text = "你好，今天天气怎么样？"
    tokens = model.tokenizer.encode(text)
    decoded = model.tokenizer.decode(tokens, skip_special_tokens=True)
    print(f"\n原文:   {text}")
    print(f"Token IDs ({len(tokens)}): {tokens[:20]}...")
    print(f"解码:   {decoded}")

    # 测试 chat template
    msgs = [{"role": "user", "content": "你好"}]
    prompt = model.apply_chat_template(msgs)
    print(f"\nChat prompt:\n{prompt}")

    input_ids = model.tokenizer.encode(prompt, return_tensors="pt")
    gen = model.generate(
        input_ids,
        max_length=100,
        num_beams=3,
        repetition_penalty=1.2,
        eos_token_id=model.tokenizer.eos_token_id,
        pad_token_id=model.tokenizer.pad_token_id,
    )
    reply = model.tokenizer.decode(gen[0], skip_special_tokens=True)
    reply = reply.replace(prompt.strip(), "").strip()
    print(f"\n生成回复: {reply}")

    print("\n✓ 测试通过!")
