import os
import sys
import torch
import torchaudio
import torch_npu
from torch_npu.contrib import transfer_to_npu
from cosyvoice.cli.cosyvoice import AutoModel

# 1. 环境与硬件初始化
def init_env(model_path):
    # 关闭 JIT 编译以提高初次运行速度或兼容性
    torch_npu.npu.set_compile_mode(jit_compile=False)
    
    # 打印 NPU 状态
    is_npu = torch_npu.npu.is_available()
    print(f"--- NPU Available: {is_npu} ---")
    device = "npu" if is_npu else "cpu"
    print(f"--- Using device: {device} ---")

    # 动态添加第三方库路径
    matcha_path = os.path.join(os.getcwd(), 'third_party/Matcha-TTS')
    if matcha_path not in sys.path:
        sys.path.append(matcha_path)

    # 初始化 CosyVoice 模型
    # load_vllm=True 会使用 vLLM 加速，确保你之前的 pydantic 依赖已解决
    cosyvoice = AutoModel(model_dir=model_path, load_vllm=True)
    return cosyvoice, device

# 2. 推理并保存音频的通用函数
def run_and_save(inference_gen, prefix, sample_rate):
    """
    处理生成器并保存音频文件
    """
    for i, output in enumerate(inference_gen):
        save_path = f'{prefix}_{i}.wav'
        # 核心：必须转回到 CPU 才能由 torchaudio 保存
        audio_data = output['tts_speech'].cpu()
        torchaudio.save(save_path, audio_data, sample_rate)
        print(f"Saved: {save_path}")

# 3. 主逻辑
def main():
    MODEL_DIR = '/home/Fun-CosyVoice3-0.5B-2512/'
    PROMPT_WAV = '/home/cosyvoice-npu/asset/zero_shot_prompt.wav'
    
    cosyvoice, device = init_env(MODEL_DIR)

    # --- 任务 1: Zero-shot (零样本克隆) ---
    print("\nRunning Zero-shot...")
    gen_zero = cosyvoice.inference_zero_shot(
        '八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。',
        'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
        PROMPT_WAV, stream=False
    )
    run_and_save(gen_zero, 'zero_shot', cosyvoice.sample_rate)

    # --- 任务 2: Cross-lingual (跨语言/细粒度控制) ---
    print("\nRunning Cross-lingual...")
    gen_cross = cosyvoice.inference_cross_lingual(
        'You are a helpful assistant.<|endofprompt|>因为他们那一辈人在乡里面住的要习惯一点，[breath]邻居都很活络，[breath]嗯，都很熟悉.',
        PROMPT_WAV, stream=False
    )
    run_and_save(gen_cross, 'fine_grained', cosyvoice.sample_rate)

    # --- 任务 3: Instruct (指令控制 - 粤语) ---
    print("\nRunning Instruct (Cantonese)...")
    gen_ins1 = cosyvoice.inference_instruct2(
        '好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。',
        'You are a helpful assistant. 请用广东话表达。<|endofprompt|>',
        PROMPT_WAV, stream=False
    )
    run_and_save(gen_ins1, 'instruct_cantonese', cosyvoice.sample_rate)

    # --- 任务 4: Instruct (指令控制 - 语速) ---
    print("\nRunning Instruct (Fast Speed)...")
    gen_ins2 = cosyvoice.inference_instruct2(
        '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
        'You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>',
        PROMPT_WAV, stream=False
    )
    run_and_save(gen_ins2, 'instruct_fast', cosyvoice.sample_rate)

    # --- 任务 5: Hotfix (多音字/纠错) ---
    print("\nRunning Hotfix...")
    gen_hotfix = cosyvoice.inference_zero_shot(
        '高管也通过电话、短信、微信等方式对报道[j][ǐ]予好评。',
        'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
        PROMPT_WAV, stream=False
    )
    run_and_save(gen_hotfix, 'hotfix', cosyvoice.sample_rate)

if __name__ == "__main__":
    main()