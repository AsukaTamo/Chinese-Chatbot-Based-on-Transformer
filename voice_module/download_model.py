"""
声频模型下载脚本
===============
将 faster-whisper 模型下载到本地 voice_module/model_voice/ 目录，
供 model_voice.py 直接使用，无需再次联网下载或依赖缓存目录。

用法:
    # 下载默认模型 (medium，推荐中文使用)
    python voice_module/download_model.py

    # 下载指定大小的模型
    python voice_module/download_model.py --model medium
    python voice_module/download_model.py --model large-v3
    python voice_module/download_model.py --model small

可用模型: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en,
          large-v1, large-v2, large-v3, large-v3-turbo, turbo
"""

import os
import sys
import argparse

# 确保能找到项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from faster_whisper.utils import download_model
from faster_whisper import available_models


def get_model_dir(model_size: str) -> str:
    """返回模型在 model_voice 目录下的存储路径"""
    voice_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(voice_dir, "model_voice", model_size)


def main():
    parser = argparse.ArgumentParser(
        description="下载 faster-whisper 声频模型到本地 model_voice 目录"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="medium",
        help="模型大小 (默认: medium，推荐中文使用)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用模型",
    )
    args = parser.parse_args()

    if args.list:
        print("可用的 faster-whisper 模型:")
        for m in available_models():
            print(f"  - {m}")
        return

    model_size = args.model
    output_dir = get_model_dir(model_size)

    print(f"{'='*60}")
    print(f"声频模型下载工具")
    print(f"{'='*60}")
    print(f"模型: {model_size}")
    print(f"下载目录: {output_dir}")
    print(f"{'='*60}")

    # 检查是否已下载
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        print(f"\n[提示] 模型目录已存在且非空: {output_dir}")
        response = input("是否重新下载? (y/N): ").strip().lower()
        if response != "y":
            print("已取消下载。")
            return
        print("将重新下载...")

    os.makedirs(output_dir, exist_ok=True)

    try:
        print(f"\n正在从 Hugging Face Hub 下载模型 '{model_size}' ...")
        print("(首次下载可能需要几分钟，取决于网络速度)\n")

        model_path = download_model(
            size_or_id=model_size,
            output_dir=output_dir,
        )

        print(f"\n{'='*60}")
        print(f"✓ 下载完成!")
        print(f"模型路径: {model_path}")
        print(f"{'='*60}")
        print(f"\n现在可以在 model_voice.py 中使用本地模型了。")

    except Exception as e:
        print(f"\n✗ 下载失败: {e}", file=sys.stderr)
        print("提示:", file=sys.stderr)
        print("  1. 检查网络连接是否正常", file=sys.stderr)
        print("  2. 如在国内，可能需要设置 Hugging Face 镜像:", file=sys.stderr)
        print('     export HF_ENDPOINT=https://hf-mirror.com', file=sys.stderr)
        print(f"  3. 模型名称是否正确: {model_size}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
