# 模型大小可选: tiny, base, small, medium, large-v3
# 推荐中文使用 'small' 或 'medium' 平衡速度与效果；追求极致准确用 'large-v3'
from faster_whisper import WhisperModel


def voice_to_words(voice_path:str,model_size:str):
    
    print("="*5+"vtw:正在初始化，初次使用可能需要一点时间下载模型参数"+"="*5)

    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    print("="*5+"vtw:初始化完成"+"="*5)
    print("识别中...")

    segments, info = model.transcribe(voice_path, beam_size=5, language="zh", initial_prompt="以下是普通话话语。")
    
    print("识别完成/^-^/")

    return segments,info


if __name__ == "__main__":

    model_size = "medium"

    # 如果使用 CPU，把 device 改为 "cpu", compute_type 改为 "int8"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # initial_prompt 传入繁体/简体提示，有助于固定输出简体中文
    segments, info = model.transcribe("voice_module\你好我是中国人.wav", beam_size=5, language="zh", initial_prompt="以下是普通话话语。")

    print(f"检测到语种: {info.language} (置信度: {info.language_probability:.2f})")

    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")