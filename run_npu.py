"""CosyVoice2/3 inference runner for Ascend NPU.

Applies the torch_npu transfer_to_npu shim (maps torch.cuda.* onto NPU) before
importing CosyVoice, then runs a single zero-shot synthesis.

Usage:
    python3 run_npu.py [model_dir] [output_wav] [--stream] [--bistream]

    --stream    stream audio output (chunk by chunk)
    --bistream  feed text as a Generator (text-in streaming, exercises
                Qwen2LM.inference_bistream). Combine with --stream for full
                bi-streaming (text-in + audio-out).
"""
import argparse
import os
import sys

import torch  # noqa: F401
import torch_npu  # noqa: F401
import torch_npu.contrib.transfer_to_npu  # noqa: F401  maps cuda -> npu

sys.path.append('third_party/Matcha-TTS')

import torchaudio  # noqa: E402
from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402

TEXT = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
PROMPT_WAV = './asset/zero_shot_prompt.wav'


def is_v3(model_dir):
    return os.path.exists(os.path.join(model_dir, 'cosyvoice3.yaml'))


def enable_bf16_autocast():
    """Force torch.npu.amp.autocast (mapped from torch.cuda.amp.autocast) to use
    bfloat16 instead of its default float16. CosyVoice enables autocast via the
    fp16 flag: torch.cuda.amp.autocast(self.fp16)."""
    from torch_npu.npu.amp.autocast_mode import autocast as npu_autocast

    def _autocast(enabled=True, dtype=None, cache_enabled=True):
        return npu_autocast(enabled=enabled, dtype=torch.bfloat16, cache_enabled=cache_enabled)

    torch.cuda.amp.autocast = _autocast


def text_generator(text, chunk_size=8):
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if chunk:
            yield chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', nargs='?', default='pretrained_models/CosyVoice2-0.5B')
    parser.add_argument('out_wav', nargs='?', default='outputs/zero_shot.wav')
    parser.add_argument('--stream', action='store_true', help='stream audio output')
    parser.add_argument('--bistream', action='store_true', help='feed text as a Generator')
    parser.add_argument('--bf16', action='store_true', help='enable bfloat16 autocast inference')
    parser.add_argument('--vllm', action='store_true', help='use vLLM for the LLM stage')
    args = parser.parse_args()

    model_dir, out_wav = args.model_dir, args.out_wav
    os.makedirs(os.path.dirname(out_wav) or '.', exist_ok=True)

    if args.bf16:
        enable_bf16_autocast()

    if args.vllm:
        # NPU: torch_npu cannot be re-initialized in a forked subprocess, so run
        # the V1 engine core in-process. Must be set before importing vllm.
        os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
        # This image has both a (mapped) CUDA driver and the Ascend driver, so
        # vllm sees 2 active triton drivers and disables triton -> vllm_ascend
        # triton kernels break ('function' object is not subscriptable). Mark
        # the nvidia triton backend inactive so only ascend remains active.
        # Must run before the first `import vllm`.
        try:
            from triton.backends import backends
            _nb = backends.get('nvidia')
            if _nb is not None and getattr(_nb, 'driver', None) is not None:
                _nb.driver.is_active = lambda: False
        except Exception:
            pass
        # custom architecture must be registered before the vLLM engine is built
        from vllm import ModelRegistry
        from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
        ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

    kwargs = dict(model_dir=model_dir, load_trt=False, load_vllm=args.vllm, fp16=args.bf16)
    if not is_v3(model_dir):
        # CosyVoice3.__init__ does not accept load_jit
        kwargs['load_jit'] = False
    cosyvoice = AutoModel(**kwargs)

    if is_v3(model_dir):
        prompt_text = 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。'
    else:
        prompt_text = '希望你以后能够做的比我还好呦。'

    text = text_generator(TEXT) if args.bistream else TEXT
    chunks = []
    for j in cosyvoice.inference_zero_shot(text, prompt_text, PROMPT_WAV, stream=args.stream):
        chunks.append(j['tts_speech'])
    speech = torch.cat(chunks, dim=-1) if len(chunks) > 1 else chunks[0]
    torchaudio.save(out_wav, speech, cosyvoice.sample_rate)
    duration = speech.shape[-1] / cosyvoice.sample_rate
    print('saved {} (chunks={}, dur={:.2f}s)'.format(out_wav, len(chunks), duration))


if __name__ == '__main__':
    main()
