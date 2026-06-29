"""
声频识别模块 (Voice to Words / VTW)
=====================================
基于 faster-whisper 的语音转文字模块，使用本地预下载模型，
无需联网查找或下载到缓存目录。

使用前请先运行下载脚本将模型下载到本地:
    python voice_module/download_model.py --model medium

可用模型: tiny, base, small, medium, large-v3 等
"""

import os
import sys

from faster_whisper import WhisperModel

# --- 本地模型路径配置 ---
_VOICE_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_MODEL_ROOT = os.path.join(_VOICE_DIR, "model_voice")

# 默认使用的模型大小
_DEFAULT_MODEL_SIZE = "medium"


def _get_local_model_path(model_size: str = _DEFAULT_MODEL_SIZE) -> str:
    """
    获取本地模型路径。

    参数:
        model_size: 模型大小名称 (tiny, base, small, medium, large-v3 等)

    返回:
        本地模型目录的完整路径
    """
    return os.path.join(_LOCAL_MODEL_ROOT, model_size)


def is_model_downloaded(model_size: str = _DEFAULT_MODEL_SIZE) -> bool:
    """检查指定模型是否已下载到本地 model_voice 目录"""
    model_path = _get_local_model_path(model_size)
    # faster-whisper 模型目录包含 model.bin 和 config.json
    return (
        os.path.isdir(model_path)
        and os.path.isfile(os.path.join(model_path, "model.bin"))
        and os.path.isfile(os.path.join(model_path, "config.json"))
    )


def voice_to_words(voice_path: str, model_size: str = _DEFAULT_MODEL_SIZE):
    """
    语音转文字。

    参数:
        voice_path: 音频文件路径
        model_size: 模型大小，默认 "medium"

    返回:
        (segments, info) — 识别片段列表和音频信息
    """
    model_path = _get_local_model_path(model_size)

    if not is_model_downloaded(model_size):
        raise FileNotFoundError(
            f"本地模型未找到: {model_path}\n"
            f"请先运行下载脚本:\n"
            f"    python voice_module/download_model.py --model {model_size}\n"
            f"或使用其他模型大小，如:\n"
            f"    python voice_module/download_model.py --model small"
        )

    print(f"{'='*60}")
    print(f"VTW: 加载本地模型 -> {model_path}")
    print(f"{'='*60}")

    # 直接使用本地模型路径，不会联网下载
    model = WhisperModel(
        model_path,                      # 本地模型路径
        device="cuda",
        compute_type="float16",
        local_files_only=True,           # 确保不联网
        download_root=_LOCAL_MODEL_ROOT,  # 模型根目录
    )

    print(f"{'='*60}")
    print("VTW: 模型加载完成")
    print(f"{'='*60}")
    print("识别中...")

    segments, info = model.transcribe(
        voice_path,
        beam_size=5,
        language="zh",
        initial_prompt="以下是普通话话语。",
    )

    print("识别完成 /^-^/")

    return segments, info


# --- 测试入口 ---
if __name__ == "__main__":
    model_size = "medium"  # 可改为 small / large-v3 等
    test_audio = os.path.join(_VOICE_DIR, "你好我是中国人.wav")

    if not os.path.isfile(test_audio):
        print(f"测试音频文件不存在: {test_audio}")
        sys.exit(1)

    segments, info = voice_to_words(test_audio, model_size=model_size)

    print(f"\n检测到语种: {info.language} (置信度: {info.language_probability:.2f})")
    print("-" * 40)

    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
