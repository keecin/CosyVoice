# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
from concurrent import futures
import argparse
import cosyvoice_pb2
import cosyvoice_pb2_grpc
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import grpc
import torch
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

print("[INFO] Initializing NPU environment...")
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    # 设置编译模式
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    print("[INFO] NPU initialized successfully.")
except ImportError:
    print("[WARNING] NPU libraries not found. Using CPU.")

# taohouchao
import threading
import time
import uuid

# --- Ascend + vLLM 0.30 runtime fixes (MUST run before `import vllm`) ---------
# Engine mode depends on the exported vLLM model architecture:
#  - CosyVoice2ForCausalLM (v2, use_bias=True): custom class -> V1 engine must run
#    in-process so the registration is visible.
#  - Qwen2ForCausalLM (v3, use_bias=False): stock Qwen2, no registration ->
#    use MindIE-style multi-process V1 (spawn) for parity/perf.


def _vllm_exported_is_custom(model_dir):
    try:
        import json as _json
        cfg = os.path.join(model_dir, 'vllm', 'config.json')
        archs = _json.load(open(cfg, encoding='utf-8')).get('architectures') or []
        return 'CosyVoice2ForCausalLM' in archs
    except Exception:
        return False


def _model_dir_from_argv(argv):
    for i, a in enumerate(argv):
        if a == '--model_dir' and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith('--model_dir='):
            return a.split('=', 1)[1]
    return ''


_CUSTOM_ARCH = _vllm_exported_is_custom(_model_dir_from_argv(sys.argv))
if _CUSTOM_ARCH:
    os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
else:
    # MindIE-style multi-process V1; spawn is required on NPU (no fork re-init).
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
# Equivalent to the image's torch TMA patch (without editing torch): make
# torch.cuda.get_device_capability() non-None on NPU so dynamo's
# _device_supports_tma() does not crash on `None >= (9,0)`. Format major.minor.
os.environ.setdefault('TORCH_NPU_DEVICE_CAPABILITY', '8.0')
# 2) this image reports both nvidia+ascend triton drivers -> vllm disables
#    triton -> vllm_ascend triton kernels break. Mark nvidia backend inactive.
try:
    from triton.backends import backends
    _nb = backends.get('nvidia')
    if _nb is not None and getattr(_nb, 'driver', None) is not None:
        _nb.driver.is_active = lambda: False
except Exception:
    pass
# -----------------------------------------------------------------------------

import vllm
if _CUSTOM_ARCH:
    # v2: register the custom architecture (only effective for in-process engine)
    try:
        from vllm import ModelRegistry
        from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
        ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
    except Exception as _e:
        logging.warning(f"register CosyVoice2ForCausalLM failed: {_e}")

import queue
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
import datetime
import json
import gc
from typing import Optional
from enum import Enum
from dataclasses import dataclass, field

from hyperpyyaml import load_hyperpyyaml
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.cli.frontend import CosyVoiceFrontEnd


class ThcSectionStatus(str, Enum):
    LLM_DOING = 'llm_doing'
    T2W_WAITING = 't2w_waiting'
    T2W_DOING = 't2w_doing'
    DONE = 'done'

@dataclass
class ThcSection:
    idx: tuple
    mode: str
    spk_id: str
    first: bool
    finalize: bool
    tokens: list
    token_begin_pos: int
    token_end_pos: int
    token_begin_pos_in_batch: Optional[int] = None
    token_end_pos_in_batch: Optional[int] = None
    flow_begin_pos_in_batch: Optional[int] = None
    flow_end_pos_in_batch: Optional[int] = None
    flow_cache: Optional[torch.Tensor] = None
    speech_begin_pos_in_batch: Optional[int] = None
    speech_end_pos_in_batch: Optional[int] = None
    speech: Optional[torch.Tensor] = None
    speech_cpu: Optional[np.array] = None
    speech_dt: Optional[float] = None
    status: ThcSectionStatus = ThcSectionStatus.LLM_DOING

class ThcMetric:
    def __init__(self, begin=None):
        self.begin: Optional[datetime.datetime] = begin
        self.end: Optional[datetime.datetime] = None
        self.dt: Optional[float] = None

    def start(self):
        self.begin = self.begin or datetime.datetime.now()

    def stop(self):
        self.end = self.end or datetime.datetime.now()
        self.dt = (self.end - self.begin).total_seconds()

@dataclass
class ThcAllMetric:
    e2e: ThcMetric = field(default_factory=ThcMetric)
    llm: ThcMetric = field(default_factory=ThcMetric)
    t2w: list = field(default_factory=list)

@dataclass
class ThcRequestInfo:
    idx: int
    request_id: str
    mode: str
    spk_id: str
    first: bool = False
    finalize: bool = False
    done: bool = False
    llm_output: list = field(default_factory=list)
    llm_done: bool = False
    section_queue: queue.Queue = field(default_factory=queue.Queue)
    flow_cache: Optional[torch.Tensor] = None
    speech: Optional[torch.Tensor] = None
    speech_cum_dt: float = 0.0
    dynamic_rtf: Optional[float] = None
    dynamic_slack: Optional[float] = None
    release_dt: Optional[float] = None
    sections: list = field(default_factory=list)
    release: queue.Queue = field(default_factory=queue.Queue)
    metric: ThcAllMetric = field(default_factory=ThcAllMetric)

def thc_print(*args, **kwargs):
    time_str = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print(f"thc: [{time_str}],", *args, **kwargs)


class Token2Wave():
    def __init__(self,
                 device,
                 fp16: bool = False,
                 model_dir: str = None,
                 speaker_info_dir: str = None,
                 graph_mode: bool = False):
        hyper_yaml_path = os.path.join(model_dir, 'cosyvoice3.yaml')
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})

        self.device = device
        self.flow_len_per_token = 2
        self.speech_len_per_token = 960
        self.speech_sample_rate = configs['sample_rate']
        self.token_sample_rate = configs['sample_rate'] / self.speech_len_per_token

        self.model_dir = model_dir
        self.graph_mode = graph_mode

        frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                     configs['feat_extractor'],
                                     os.path.join(model_dir, 'campplus.onnx'),
                                     os.path.join(model_dir, 'speech_tokenizer_v3.onnx'),
                                     os.path.join(model_dir, 'spk2info.pt'),
                                     configs['allowed_special'])
        sample_rate = configs['sample_rate']

        flow_ckpt_file = os.path.join(model_dir, 'flow.pt')
        hift_ckpt_file = os.path.join(model_dir, 'hift.pt')

        self.flow = configs['flow'].to(self.device)
        self.flow.load_state_dict(torch.load(flow_ckpt_file, map_location=self.device, weights_only=True), strict=True)
        self.flow.eval()
        # in case hift_model is a hifigan model
        self.hift = configs['hift'].to(self.device)
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_ckpt_file, map_location=self.device, weights_only=True).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.eval()

        self.prompt_wave_info = None

        self.lookahead_token_len = self.flow.pre_lookahead_len
        self.pre_gap_token_len = self.flow.pre_lookahead_len * 4
        self.post_gap_token_len = self.flow.pre_lookahead_len * 1

        self.silent_tokens = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]

        self.perfs = []

        self.batches_requests = queue.Queue()
        self.inference_event = threading.Event()
        self.lock = threading.Lock()

        del configs

        if graph_mode:
            import torchair as tng
            from torchair.configs.compiler_config import CompilerConfig
            config = CompilerConfig()
            # config.mode = "reduce-overhead"
            #config.experimental_config.tiling_schedule_optimize = True
            #config.experimental_config.frozen_parameter = True
            npu_backend = tng.get_npu_backend(compiler_config=config)
            self.flow.decoder.estimator.forward = torch.compile(
                self.flow.decoder.estimator.forward,
                dynamic=True,
                fullgraph=True,
                backend=npu_backend
            )
            thc_print("Token2Wave: compile graph success")

    def add_requests(self, batch_requests):
        self.batches_requests.put(batch_requests)
        self.inference_event.set()

    def prepare_batch_data(self, batch_requests):
        merged_tokens = []
        mode = None
        spk_id = None
        pos = 0
        for request in batch_requests:
            mode = mode or request.mode
            assert request.mode == mode
            spk_id = spk_id or request.spk_id
            assert request.spk_id == spk_id

            merged_tokens.extend(request.tokens)
            request.token_begin_pos_in_batch = pos + request.token_begin_pos
            request.token_end_pos_in_batch = pos + request.token_end_pos
            pos += request.token_end_pos
            pos += self.post_gap_token_len
            request.flow_begin_pos_in_batch = int(request.token_begin_pos_in_batch * self.flow_len_per_token)
            request.flow_end_pos_in_batch = int(request.token_end_pos_in_batch * self.flow_len_per_token)
            request.status = ThcSectionStatus.T2W_DOING
            # thc_print(f"Token2Wave.prepare_batch_data: {request.idx}, "
            #           f"merged_tokens = {len(merged_tokens)}, "
            #           f"start_pos_in_batch = {request.token_begin_pos_in_batch}, end_pos_in_batch = {request.token_end_pos_in_batch}, "
            #           f"pos = {pos}")
        assert pos == len(merged_tokens), f"pos = {pos}, len(merged_tokens) = {len(merged_tokens)}"

        tokens = torch.tensor(merged_tokens, dtype=torch.int32, device=self.device).unsqueeze(dim=0)
        token_len = torch.tensor([tokens.shape[1]], dtype=torch.int32, device=self.device)

        if mode == 'zero_shot_by_id':
            flow_prompt_speech_token = self.prompt_wave_info[spk_id]['flow_prompt_speech_token']
            flow_prompt_speech_token_len = self.prompt_wave_info[spk_id]['flow_prompt_speech_token_len']
            prompt_speech_feat = self.prompt_wave_info[spk_id]['prompt_speech_feat']
            prompt_speech_feat_len = self.prompt_wave_info[spk_id]['prompt_speech_feat_len']
        elif mode == 'sft':
            flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32, device=self.device)
            flow_prompt_speech_token_len = torch.tensor([flow_prompt_speech_token.shape[1]], dtype=torch.int32, device=self.device)
            prompt_speech_feat = torch.zeros(1, 0, 80, device=self.device)
            prompt_speech_feat_len = torch.tensor([prompt_speech_feat.shape[1]], dtype=torch.int32, device=self.device)

        flow_embedding = self.prompt_wave_info[spk_id]['flow_embedding']
        stream = True
        finalize = False
        # perf case 1310304325630799872: 首段 8 步固定音质基线，后续段 5 步（原 10/5）
        n_timesteps = 8 if any([request.first for request in batch_requests]) else 5

        return batch_requests, tokens, token_len, flow_prompt_speech_token, flow_prompt_speech_token_len, prompt_speech_feat, prompt_speech_feat_len, flow_embedding, stream, finalize, n_timesteps

    def step(self):
        # modify if enable multi_threads
        batch_requests = self.batches_requests.get_nowait()

        metric = ThcMetric()
        metric.start()
        # thc_print(f"Token2Wave.infer: start: {t0:%H:%M:%S.%f}")

        batch_requests, tokens, token_len, flow_prompt_speech_token, flow_prompt_speech_token_len, prompt_speech_feat, prompt_speech_feat_len, flow_embedding, stream, finalize, n_timesteps = self.prepare_batch_data(batch_requests)

        tts_mel, _ = self.flow.inference(token=tokens,
                                         token_len=token_len,
                                         prompt_token=flow_prompt_speech_token,
                                         prompt_token_len=flow_prompt_speech_token_len,
                                         prompt_feat=prompt_speech_feat,
                                         prompt_feat_len=prompt_speech_feat_len,
                                         embedding=flow_embedding,
                                         streaming=stream,
                                         finalize=finalize,
                                         n_timesteps=n_timesteps)

        # thc_print(f"tts_mel = {tts_mel.shape}")
        assert tts_mel.shape[-1] == (tokens.shape[-1] - self.lookahead_token_len) * 2

        # thc_print(f"self.hift.f0_predictor.condnet[0].causal_padding = {self.hift.f0_predictor.condnet[0].causal_padding}")
        prefix_flow_len = 16 * self.flow_len_per_token
        token_offset = 4

        speech_pos = 0
        merged_tts_mel = []
        for request in batch_requests:
            request_tts_mel = tts_mel[..., request.flow_begin_pos_in_batch : request.flow_end_pos_in_batch]
            speech_len = (request.token_end_pos - request.token_begin_pos) * self.speech_len_per_token
            assert request.first == (request.flow_cache is None)
            if request.first:
                flow_prefix = request_tts_mel.new_zeros([*request_tts_mel.shape[: -1], prefix_flow_len])
                request.flow_cache = request_tts_mel
                request.speech_begin_pos_in_batch = speech_pos + flow_prefix.shape[-1] // self.flow_len_per_token * self.speech_len_per_token
                speech_len -= token_offset * self.speech_len_per_token
            else:
                flow_prefix = request.flow_cache[..., -prefix_flow_len :]
                request.flow_cache = torch.cat([request.flow_cache, request_tts_mel], dim=-1)
                assert flow_prefix.shape[-1] >= token_offset * self.flow_len_per_token, f"request.flow_cache = {request.flow_cache.shape}, flow_prefix = {flow_prefix.shape}"
                request.speech_begin_pos_in_batch = speech_pos + (flow_prefix.shape[-1] // self.flow_len_per_token - token_offset) * self.speech_len_per_token

            if request.finalize:
                flow_postfix = request_tts_mel.new_zeros([*request_tts_mel.shape[: -1], prefix_flow_len])
                request_tts_mel = torch.cat([flow_prefix, request_tts_mel, flow_postfix], dim=-1)
                speech_len += token_offset * self.speech_len_per_token
            else:
                request_tts_mel = torch.cat([flow_prefix, request_tts_mel], dim=-1)
            request.speech_end_pos_in_batch = request.speech_begin_pos_in_batch + speech_len
            merged_tts_mel.append(request_tts_mel)
            speech_pos += request_tts_mel.shape[-1] // self.flow_len_per_token * self.speech_len_per_token
            # thc_print(f"Token2Wave.step: {request.idx}, finalize = {request.finalize}, "
            #           f"speech_begin_pos_in_batch = {request.speech_begin_pos_in_batch}, speech_end_pos_in_batch = {request.speech_end_pos_in_batch}, "
            #           f"speech_pos = {speech_pos}")
        merged_tts_mel = torch.cat(merged_tts_mel, dim=-1)

        tts_speech, _ = self.hift.inference(speech_feat=merged_tts_mel, finalize=True)
        tts_speech_cpu = (tts_speech * (2 ** 15)).to(torch.int16).view(-1).cpu().numpy()
        # thc_print(f"merged_tts_mel = {merged_tts_mel.shape}, tts_speech_cpu = {tts_speech_cpu.shape}")

        for request in batch_requests:
            request.speech = tts_speech[:, request.speech_begin_pos_in_batch : request.speech_end_pos_in_batch]
            request.speech_cpu = tts_speech_cpu[request.speech_begin_pos_in_batch : request.speech_end_pos_in_batch]
            assert request.speech_cpu.shape[0] == request.speech_end_pos_in_batch - request.speech_begin_pos_in_batch, f"{request.speech_cpu.shape[0]} != {request.speech_end_pos_in_batch} - {request.speech_begin_pos_in_batch}"
            request.speech_dt = request.speech_cpu.shape[0] / self.speech_sample_rate
            request.status = ThcSectionStatus.DONE

        metric.stop()

        perf = {'token_len': token_len[0].item(), 'tts_speech_shape': list(tts_speech.shape), 'n_timesteps': n_timesteps, 'metric': metric}
        self.perfs.append(perf)
        # thc_print(f"Token2Wave.infer: end: token_len = {perf['token_len']}, tts_speech_shape = {perf['tts_speech_shape']}, dt = {metric.dt:.3f}")

        return batch_requests

    def infer(self):
        while True:
            self.inference_event.wait()
            self.inference_event.clear()

            # NOTE this thread is not started by default (the service drives
            # add_requests()/step() itself); fixed the previous
            # `self.inputs` NameError here for anyone re-enabling it.
            if self.batches_requests.empty():
                continue

            self.step()


class CosyVoiceServiceImpl(cosyvoice_pb2_grpc.CosyVoiceServicer):
    def __init__(self, args):
        self.args = args  # 保存args引用
        # 注意：AutoModel(load_vllm=True) 会在此处启动 vLLM 子进程
        self.cosyvoice = AutoModel(
            model_dir=args.model_dir,
            load_vllm=True,
            speaker_info_dir=args.speaker_info_dir,
            graph_mode=args.graph_mode
        )
        logging.info(f'grpc service initialized, graph_mode={args.graph_mode}')

        # taohouchao
        self.prompt_emb = {}
        for spk_id, spk_info in self.cosyvoice.frontend.spk2info.items():
            if spk_id not in self.cosyvoice.frontend.spk2info_sft:
                self.cosyvoice.add_sft_spk(spk_id)
            model_input = self.cosyvoice.frontend.spk2info_sft[spk_id]
            assert self.prompt_emb.get(spk_id) is None
            self.prompt_emb[spk_id] = {
                'sos_emb': self.cosyvoice.model.llm.speech_embedding.weight[self.cosyvoice.model.llm.sos].reshape(1, 1, -1),
                'task_id_emb': self.cosyvoice.model.llm.speech_embedding.weight[self.cosyvoice.model.llm.task_id].reshape(1, 1, -1),
            }

        for spk_id, wave_info in self.cosyvoice.promote_wave_info.items():
            if spk_id not in self.cosyvoice.frontend.spk2info:
                self.cosyvoice.add_zero_shot_spk(wave_info['prompt_text'], wave_info['prompt_wav'], spk_id)
            model_input = self.cosyvoice.frontend.spk2info[spk_id]
            prompt_text = model_input['prompt_text']
            llm_prompt_speech_token = model_input['llm_prompt_speech_token']
            assert self.prompt_emb.get(spk_id) is None
            self.prompt_emb[spk_id] = {
                'sos_emb': self.cosyvoice.model.llm.speech_embedding.weight[self.cosyvoice.model.llm.sos].reshape(1, 1, -1),
                'task_id_emb': self.cosyvoice.model.llm.speech_embedding.weight[self.cosyvoice.model.llm.task_id].reshape(1, 1, -1),
                'prompt_text_emb': self.cosyvoice.model.llm.llm.model.model.embed_tokens(prompt_text),
                'prompt_speech_token_emb': self.cosyvoice.model.llm.speech_embedding(llm_prompt_speech_token),
            }

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.requests_info = {}
        self.requests_id = [None for _ in range(args.max_conc)]

        self.lock = threading.Lock()
        self.t2w = Token2Wave(device=self.device,
                              fp16=False,
                              model_dir=args.model_dir,
                              speaker_info_dir=args.speaker_info_dir,
                              graph_mode=args.graph_mode)
        self.t2w.prompt_wave_info = {**self.cosyvoice.frontend.spk2info, **self.cosyvoice.frontend.spk2info_sft}
        self.t2w_first_token_len = 8 + self.t2w.post_gap_token_len  # perf case: 20+gap -> 8+gap, Flow starts earlier (TTFT)
        self.t2w_min_token_len = 8  # perf case: 25 -> 8, more frequent Flow triggers, less LLM-side idle
        self.t2w_total_token_len = {True: 800, False: 700}  # perf repo HEAD: 300/800 -> 700/800 (case seal used 1400/1600, rejected: throughput -7%)
        self.t2w_delay = 0.2
        self.release_timeout = 1
        self.t2w_timeout = 1
        self.prev_release_time = None

        # self.t2w_inference_work_thread = threading.Thread(target=self.t2w.inference, daemon=True)
        # self.t2w_inference_work_thread.start()

        self.inference_event = threading.Event()
        self.inference_work_thread = threading.Thread(target=self.infer, daemon=True)
        self.inference_work_thread.start()

        # 方案1：在 gRPC 线程启动前，先跑一遍完整流水线，触发 CANN TBE
        # 知识库(repository_manager)与 Flow/HiFi 算子的懒加载/编译，使其在
        # “没有 gRPC 线程”的环境里完成初始化（避免多线程里 fork 崩溃）。
        # 跳过：--no_prewarm / graph_mode（Token2Wave 已在 init 编译图，预热无必要且会卡）/ 无 speaker。
        _skip_prewarm = None
        if getattr(args, 'no_prewarm', False):
            _skip_prewarm = '--no_prewarm'
        elif getattr(args, 'graph_mode', False):
            _skip_prewarm = 'graph_mode'
        elif not getattr(self, 'prompt_emb', None):
            _skip_prewarm = 'no speaker available'
        if _skip_prewarm:
            thc_print(f"prewarm (pre-grpc): skipped ({_skip_prewarm})")
        else:
            thc_print("prewarm (pre-grpc): start")
            _prewarm_timeout = getattr(args, 'prewarm_timeout', 300)
            _pw_err = {}

            def _pw_run():
                try:
                    self.warmup()
                except BaseException as _e:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    _pw_err['e'] = _e

            _pw = threading.Thread(target=_pw_run, daemon=True)
            _pw.start()
            _pw.join(timeout=_prewarm_timeout)
            if _pw.is_alive():
                raise RuntimeError(
                    f"prewarm did not finish within {_prewarm_timeout}s; refusing to start. "
                    "Operator compilation would then fork inside the multithreaded gRPC "
                    "context and crash (see docs/bugfixes/04). Increase --prewarm_timeout, "
                    "or pass --no_prewarm only if the operator cache is already warm.")
            if 'e' in _pw_err:
                raise RuntimeError(f"prewarm failed: {_pw_err['e']!r}; refusing to start")
            thc_print("prewarm (pre-grpc): done")

        # 启动vllm profile（通过参数控制）
        if args.enable_vllm_profile:
            assert False
            try:
                self.cosyvoice.model.llm.vllm.start_profile()
                logging.info('vLLM profile started')
            except Exception as e:
                logging.warning(f'Failed to start vLLM profile: {e}')

    def postprocess_vllm_outputs(self, llm_step_outputs, now):
        with self.lock:
            abort_requests_id = []
            for output in llm_step_outputs:
                request_id = output.request_id
                token_ids = output.outputs[0].token_ids
                request_info = self.requests_info.get(request_id)
                if request_info is None:
                    abort_requests_id.append(request_id)
                    continue
                request_info.llm_output = token_ids[: -1] if token_ids[-1] in self.cosyvoice.model.llm.stop_token_ids else token_ids
                request_info.llm_done = output.finished
                if request_info.llm_done:
                    request_info.metric.llm.stop()

            if len(abort_requests_id) > 0:
                thc_print(f"postprocess_vllm_outputs: abort_requests_id = {abort_requests_id}")
                self.cosyvoice.model.llm.vllm.abort_request(abort_requests_id)

    def schedule_nonfirst_requests_t2w(self, now):
        requests_history_token_len = {}
        requests_remain_token_len = {}
        requests_slack = {}

        for request_id, request_info in self.requests_info.items():
            if len(request_info.sections) == 0 or any([section.status != ThcSectionStatus.DONE for section in request_info.sections]):
                continue
            history_token_len = sum([section.token_end_pos - section.token_begin_pos for section in request_info.sections])
            requests_history_token_len[request_id] = history_token_len
            new_token_len = len(request_info.llm_output) - history_token_len
            if new_token_len == 0:
                continue
            requests_remain_token_len[request_id] = new_token_len
            calc_cum_dt = (now - request_info.metric.t2w[0].end).total_seconds()
            requests_slack[request_id] = request_info.speech_cum_dt - calc_cum_dt

        if len(requests_remain_token_len) == 0:
            # no pending non-first tokens for any request -> nothing to schedule
            return {}

        flag = [len(request_info.sections) >= 1 for request_info in self.requests_info.values()]
        available_token_len = self.t2w_total_token_len[len(flag) == args.max_conc and all(flag)]
        t2w_min_token_len = max(self.t2w_min_token_len, available_token_len // len(requests_remain_token_len) - (self.t2w.pre_gap_token_len + self.t2w.post_gap_token_len))

        # flag = any([token_len >= t2w_min_token_len or self.requests_info[request_id].llm_done for request_id, token_len in requests_remain_token_len.items()])
        flag = [token_len >= t2w_min_token_len or self.requests_info[request_id].llm_done for request_id, token_len in requests_remain_token_len.items()]
        if len(flag) == 0 or all(flag) == False:
            return {}

        requests_priority = {}
        requests_t2w_sched = {}
        for request_id in requests_slack.keys():
            requests_priority[request_id] = int((max(requests_slack.values()) - requests_slack[request_id]) * self.t2w.token_sample_rate)
            requests_t2w_sched[request_id] = {'history_len': requests_history_token_len[request_id], 'alloc_len': 0, 'full': None}

        if 0:
            estimate_t2w_calc_dt = lambda v: int(212 * (v ** 2) + 646.7 * v - 153.5)
            expected_t2w_calc_dt = min(1.0, min(requests_slack.values()))
            available_token_len = max(100, estimate_t2w_calc_dt(expected_t2w_calc_dt))

        brief = {self.requests_info[request_id].idx: {'slack': round(requests_slack[request_id], 3), 'remain': requests_remain_token_len[request_id], 'sections': len(self.requests_info[request_id].sections)} for request_id in requests_slack.keys()}
        # thc_print(f"available_token_len = {available_token_len}, {brief}")
        thc_print(f"schedule_nonfirst_requests_t2w: available_token_len = {available_token_len}")

        if 0:
            for request_id in requests_remain_token_len.keys():
                request_info = self.requests_info[request_id]
                # if requests_slack[request_id] < 0.8:
                if (now - request_info.metric.t2w[-1].end).total_seconds() > self.t2w_timeout:
                    assert requests_t2w_sched[request_id]['alloc_len'] == 0
                    alloc_token_len = min(requests_remain_token_len[request_id], t2w_min_token_len)
                    alloc_token_len_with_overhead = alloc_token_len + self.t2w.pre_gap_token_len + self.t2w.post_gap_token_len
                    available_token_len -= alloc_token_len_with_overhead
                    requests_remain_token_len[request_id] -= alloc_token_len
                    requests_priority[request_id] -= alloc_token_len
                    requests_t2w_sched[request_id]['alloc_len'] += alloc_token_len
                    requests_t2w_sched[request_id]['full'] = requests_remain_token_len[request_id] == 0
                    assert requests_remain_token_len[request_id] >= 0
        else:
            for request_id in requests_remain_token_len.keys():
                request_info = self.requests_info[request_id]
                assert requests_t2w_sched[request_id]['alloc_len'] == 0
                alloc_token_len = min(requests_remain_token_len[request_id], t2w_min_token_len)
                alloc_token_len_with_overhead = alloc_token_len + self.t2w.pre_gap_token_len + self.t2w.post_gap_token_len
                available_token_len -= alloc_token_len_with_overhead
                requests_remain_token_len[request_id] -= alloc_token_len
                requests_priority[request_id] -= alloc_token_len
                requests_t2w_sched[request_id]['alloc_len'] += alloc_token_len
                requests_t2w_sched[request_id]['full'] = requests_remain_token_len[request_id] == 0
                assert requests_remain_token_len[request_id] >= 0

        while available_token_len > 0 and max(requests_remain_token_len.values()) > 0:
            requests_remain_token_len = {request_id: token_len for request_id, token_len in requests_remain_token_len.items() if token_len > 0}
            requests_priority = {request_id: requests_priority[request_id] for request_id in requests_remain_token_len.keys()}
            request_id = max(requests_priority.items(), key=lambda v: v[1])[0]
            assert requests_t2w_sched[request_id]['alloc_len'] > 0
            if requests_t2w_sched[request_id]['alloc_len'] == 0:
                alloc_token_len = min(requests_remain_token_len[request_id], t2w_min_token_len)
                alloc_token_len_with_overhead = alloc_token_len + self.t2w.pre_gap_token_len + self.t2w.post_gap_token_len
            else:
                alloc_token_len = 1
                alloc_token_len_with_overhead = 1
            available_token_len -= alloc_token_len_with_overhead
            requests_remain_token_len[request_id] -= alloc_token_len
            requests_priority[request_id] -= alloc_token_len
            requests_t2w_sched[request_id]['alloc_len'] += alloc_token_len
            requests_t2w_sched[request_id]['full'] = requests_remain_token_len[request_id] == 0
            assert requests_remain_token_len[request_id] >= 0

        requests_t2w_sched = {request_id: sched for request_id, sched in requests_t2w_sched.items() if sched['alloc_len'] > 0}

        return requests_t2w_sched

    def maybe_t2w_infer(self, now):
        with self.lock:
            first_flag = []
            for request_id, request_info in self.requests_info.items():
                if len(request_info.sections) == 0:
                    request_info.first = (len(request_info.llm_output) >= self.t2w_first_token_len) or request_info.llm_done
                    first_flag.append(request_info.first)
                else:
                    request_info.first = False

            # thc_print(f"maybe_t2w_infer: first_flag = {first_flag}, non_first_flag = {non_first_flag}")

            if len(first_flag) > 0:
                if all(first_flag):
                    requests_t2w_sched = {request_id: {'history_len': 0,
                                                       'alloc_len': min(self.t2w_first_token_len, len(request_info.llm_output)),
                                                       'full': len(request_info.llm_output) <= self.t2w_first_token_len} \
                                          for request_id, request_info in self.requests_info.items() if request_info.first}
                else:
                    requests_t2w_sched = {}
            else:
                requests_t2w_sched = self.schedule_nonfirst_requests_t2w(now)
            if len(requests_t2w_sched) == 0:
                return None

            t2w_requests = []
            for request_id, sched in requests_t2w_sched.items():
                request_info = self.requests_info[request_id]
                t2w_finalize = request_info.llm_done and sched['full']

                valid_tokens = request_info.llm_output[sched['history_len'] : sched['history_len'] + sched['alloc_len']]

                if self.t2w.pre_gap_token_len > 0:
                    pre_gap = request_info.llm_output[: sched['history_len']][-self.t2w.pre_gap_token_len :]
                    pre_gap = [self.t2w.silent_tokens[0]] * (self.t2w.pre_gap_token_len - len(pre_gap)) + pre_gap
                else:
                    pre_gap = []

                if t2w_finalize:
                    post_gap = [self.t2w.silent_tokens[0]] * self.t2w.post_gap_token_len
                else:
                    post_gap = []

                tokens = pre_gap + valid_tokens + post_gap
                rel_begin_pos = len(pre_gap)
                rel_end_pos = rel_begin_pos + len(valid_tokens) - (0 if t2w_finalize else self.t2w.post_gap_token_len)

                t2w_request = ThcSection(idx=(request_info.idx, request_id, len(request_info.sections)),
                                         mode=request_info.mode,
                                         spk_id=request_info.spk_id,
                                         first=request_info.first,
                                         finalize=t2w_finalize,
                                         tokens=tokens,
                                         token_begin_pos=rel_begin_pos,
                                         token_end_pos=rel_end_pos,
                                         flow_cache=request_info.flow_cache,
                                         status=ThcSectionStatus.T2W_DOING)
                request_info.sections.append(t2w_request)
                t2w_requests.append(t2w_request)
                request_info.metric.t2w.append(ThcMetric(begin=now))
                # thc_print(f"t2w_infer: {request_info.idx}: t2w_request: {request_info.metric.t2w[-1].begin:%H:%M:%S.%f}, "
                #           f"abs_begin_pos = {abs_begin_pos}, abs_end_pos = {abs_end_pos}, "
                #           f"rel_begin_pos = {rel_begin_pos}, rel_end_pos = {rel_end_pos}, "
                #           f"first = {t2w_request.first}, finalize = {t2w_request.finalize}")

        assert len(t2w_requests) > 0
        sections_brief = {f'{t2w_request.idx[0]}-{t2w_request.idx[2]}': len(t2w_request.tokens) for t2w_request in t2w_requests}
        thc_print(f"t2w_infer: sections: {sections_brief}")
        self.t2w.add_requests(t2w_requests)
        outputs = self.t2w.step()

        return outputs

    def postprocess_t2w_outputs(self, t2w_step_outputs):
        with self.lock:
            for output in t2w_step_outputs:
                request_id = output.idx[1]
                section_idx = output.idx[2]
                assert isinstance(request_id, str)
                request_info = self.requests_info.get(request_id, None)
                if request_info is None:
                    continue
                request_info.flow_cache = output.flow_cache
                if request_info.speech is None:
                    request_info.speech = output.speech
                else:
                    request_info.speech = torch.cat([request_info.speech, output.speech], dim=-1)
                request_info.speech_cum_dt += output.speech_dt
                request_info.metric.t2w[section_idx].stop()
                request_info.done = any([section.finalize for section in request_info.sections])
                request_info.section_queue.put(output)

            now = datetime.datetime.now()
            for request_info in self.requests_info.values():
                if request_info.done:
                    calc_cum_dt = (now - request_info.metric.e2e.begin).total_seconds()
                    request_info.dynamic_rtf = calc_cum_dt / request_info.speech_cum_dt
                    request_info.dynamic_slack = request_info.speech_cum_dt - calc_cum_dt
                    request_info.release_dt = (now - request_info.metric.t2w[-1].end).total_seconds()
                    # thc_print(f"infer: {request_info.idx}: "
                    #       f"begin = {request_info.metric.e2e.begin:%H:%M:%S.%f}, "
                    #       f"now = {now:%H:%M:%S.%f}, "
                    #       f"done = {request_info.done}, "
                    #       f"calc_cum_dt = {calc_cum_dt:.3f}, "
                    #       f"speech_cum_dt = {request_info.speech_cum_dt:.3f}, "
                    #       f"dynamic_rtf = {request_info.dynamic_rtf:.3f}, "
                    #       f"dynamic_slack = {request_info.dynamic_slack:.3f}")

            # time.sleep(0.1)
            if 1:
                can_release_flag = []
                must_release_flag = []
                for request_info in self.requests_info.values():
                    can_release_flag.append(request_info.done)
                    if request_info.done:
                        if len(request_info.sections) == 1:
                            flag = True
                        else:
                            dt = (now - request_info.metric.t2w[-2].end).total_seconds()
                            # flag = dt > self.release_timeout or request_info.speech_cum_dt < 5 or request_info.dynamic_rtf > 0.6
                            flag = dt > self.release_timeout
                        must_release_flag.append(flag)
                    else:
                        must_release_flag.append(False)

                if all(can_release_flag) or any(must_release_flag):
                    self.prev_release_time = now
                    requests_id = list(self.requests_info.keys())
                    for request_id in requests_id:
                        request_info = self.requests_info[request_id]
                        if request_info.done:
                            self.requests_info.pop(request_id)
                            self.requests_id[request_info.idx] = None
                            request_info.release.put(True)
                            # thc_print(f"infer: remove: request_idx: {request_info.idx}, dynamic_rtf = {request_info.dynamic_rtf:.3f}, dynamic_slack = {request_info.dynamic_slack:.3f}")
                else:
                    self.prev_release_time = None
            else:
                requests_id = list(self.requests_info.keys())
                for request_id in requests_id:
                    request_info = self.requests_info[request_id]
                    if request_info.done:
                        self.prev_release_time = now
                        self.requests_info.pop(request_id)
                        self.requests_id[request_info.idx] = None
                        request_info.release.put(True)
                        # thc_print(f"infer: remove: request_idx: {request_info.idx}, dynamic_rtf = {request_info.dynamic_rtf:.3f}, dynamic_slack = {request_info.dynamic_slack:.3f}")


    def infer(self):
        llm = self.cosyvoice.model.llm
        print_wait_flag = True
        empty_cache_flag = True

        thc_print("infer: start")

        while True:
            if print_wait_flag:
                thc_print("infer: wait")
                print_wait_flag = False

            if not self.inference_event.wait(timeout=30):
                if empty_cache_flag:
                    thc_print("infer: empty_cache")
                    gc.collect()
                    if self.device.type == 'npu':
                        torch_npu.npu.empty_cache()
                    empty_cache_flag = False
                continue

            if llm.vllm.get_num_unfinished_requests() == 0:
                continue

            print_wait_flag = True
            empty_cache_flag = True

            llm_section_begin = datetime.datetime.now()
            self.prev_release_time = None
            n_step = 0
            # thc_print(f"infer: llm.vllm.get_num_unfinished_requests() = {llm.vllm.get_num_unfinished_requests()}")

            while llm.vllm.has_unfinished_requests() or len(self.requests_info) > 0:
                self.inference_event.clear()

                if llm.vllm.has_unfinished_requests():
                    llm_step_outputs = llm.vllm.step()

                    now = datetime.datetime.now()
                    llm_section_end = now
                    # thc_print(f"infer: n_step = {n_step}, len(llm_step_outputs) = {len(llm_step_outputs)}, "
                    #           f"{[len(request_output.outputs) for request_output in llm_step_outputs]}, "
                    #           f"{[len(request_output.outputs[0].token_ids) for request_output in llm_step_outputs]}, "
                    #           f"{[request_output.outputs[0].token_ids[-1] for request_output in llm_step_outputs]}, "
                    #           f"{[request_output.finished for request_output in llm_step_outputs]}, "
                    #           f"{[request_output.outputs[0].finish_reason for request_output in llm_step_outputs]}")
                    n_step += 1

                    self.postprocess_vllm_outputs(llm_step_outputs, now)

                if self.prev_release_time is None or \
                   (datetime.datetime.now() - self.prev_release_time).total_seconds() >= self.t2w_delay or \
                   len(self.requests_info) == args.max_conc:
                    t2w_step_outputs = self.maybe_t2w_infer(now)
                    if t2w_step_outputs is not None:
                        llm_section_dt = (llm_section_end - llm_section_begin).total_seconds()
                        t2w_perf = self.t2w.perfs[-1]
                        # thc_print(f"infer: llm_dt = {llm_section_dt:.3f}, t2w_token_len = {t2w_perf['token_len']}, t2w_dt = {t2w_perf['metric'].dt:.3f}")
                        llm_section_begin = datetime.datetime.now()
                        self.postprocess_t2w_outputs(t2w_step_outputs)
                elif llm.vllm.has_unfinished_requests():
                    pass
                else:
                    time.sleep(0.01)

    def add_request(self, mode, tts_text, spk_id, stream_flag):
        if stream_flag != True:
            raise NotImplementedError("only support stream mode")

        cosyvoice = self.cosyvoice
        frontend = cosyvoice.frontend
        model = cosyvoice.model
        llm = model.llm

        request_id = str(uuid.uuid1())

        spk_prompt_emb = self.prompt_emb[spk_id]
        sos_emb = spk_prompt_emb['sos_emb']
        task_id_emb = spk_prompt_emb['task_id_emb']

        # todo iter text
        # tts_text = frontend.text_normalize(tts_text, split=True, text_frontend=frontend.text_frontend)[0]
        tts_texts = frontend.text_normalize(tts_text, split=True, text_frontend=frontend.text_frontend)
        tts_text = ''.join(tts_texts)
        if mode == 'zero_shot_by_id':
            model_input = frontend.frontend_zero_shot(tts_text, None, None, None, spk_id)
            text = model_input['text']
            text_len = model_input['text_len']
            text_emb = llm.llm.model.model.embed_tokens(text)
            prompt_text_emb = spk_prompt_emb['prompt_text_emb']
            prompt_speech_token_emb = spk_prompt_emb['prompt_speech_token_emb']
            lm_input = torch.concat([sos_emb, prompt_text_emb, text_emb, task_id_emb, prompt_speech_token_emb], dim=1)
        elif mode == 'sft':
            tts_text = '<|endofprompt|>' + tts_text
            model_input = frontend.frontend_sft(tts_text, spk_id)
            text = model_input['text']
            text_len = model_input['text_len']
            text_emb = llm.llm.model.model.embed_tokens(text)
            lm_input = torch.concat([sos_emb, text_emb, task_id_emb], dim=1)

        # thc_print(f"add_request: lm_input: [{lm_input.dtype}, lm_input.shape: {lm_input.shape}]")

        min_token_text_ratio = 2
        max_token_text_ratio = 20
        min_len = int(text_len * min_token_text_ratio)
        max_len = int(text_len * max_token_text_ratio)

        # self.llm.inference_wrapper(lm_input, sampling, min_len, max_len, uuid)
        sampling = 25
        sampling_params = vllm.SamplingParams(top_k=sampling,
                                              stop_token_ids=llm.stop_token_ids,
                                              min_tokens=min_len,
                                              max_tokens=max_len)
        # thc_print(f"add_request: sampling_params: {sampling_params}")
        now = datetime.datetime.now()

        with self.lock:
            for n in range(len(self.requests_id)):
                other_request_id = self.requests_id[n]
                other_requests_info = self.requests_info.get(other_request_id, None)
                if other_requests_info is not None and (now - other_requests_info.metric.e2e.begin).total_seconds() > 60:
                    thc_print(f"add_request: request_idx: {n}, {other_request_id}, timeout")
                    self.requests_info.pop(other_request_id)
                    self.requests_id[n] = None

            request_idx = [n for n in range(args.max_conc) if self.requests_id[n] is None]
            assert len(request_idx) > 0, f"requests_id = {self.requests_id}"
            request_idx = request_idx[0]
            request_info = ThcRequestInfo(
                idx=request_idx,
                request_id=request_id,
                mode=mode,
                spk_id=spk_id,
            )
            request_info.metric.e2e.start()
            request_info.metric.llm.start()

            self.requests_id[request_idx] = request_id
            self.requests_info[request_id] = request_info

            llm.vllm.add_request(request_id, {'prompt_embeds': lm_input.squeeze(0).to(torch.bfloat16)}, sampling_params)

        self.inference_event.set()

        thc_print(f"add_request: request_idx: {request_idx}, {request_id}, mode = {mode}, tts_text = {tts_text}, spk_id = {spk_id}")

        return request_info

    def check_context_and_get_queue(self, request_info, context, q, timeout, q_name):
        request_idx = request_info.idx
        request_id = request_info.request_id
        unit = 0.2

        for n in range(max(1, int(timeout / unit))):
            if (context is not None) and (not context.is_active()):
                thc_print(f"check_context: request_idx: {request_idx}, {request_id}, client disconnected")
                return None
            try:
                data = q.get(timeout=unit)
                return data
            except queue.Empty:
                continue
        else:
            thc_print(f"check_queue: request_idx: {request_idx}, {request_id}, get {q_name} timeout")
            return None

    def postprocess_request(self, request_info):
        request_idx = request_info.idx
        request_id = request_info.request_id

        # thc_print(f"process_request: {request_idx}: submit: {request_info["submit_time"]:%H:%M:%S.%f}, "
        #       f"llm: sections_len: {request_info["llm"]["sections_len"]}, speechs_dt: {request_info.speechs_dt}, "
        #       f"calc_dt: {(request_info["llm"]["times"]["end"] - request_info["llm"]["times"]["start"]).total_seconds():.3f}")
        worst_slack = 999
        for n in range(1, len(request_info.sections)):
            speech_cum_dt = sum([section.speech_dt for section in request_info.sections[: n]])
            calc_cum_dt = (request_info.metric.t2w[n].end - request_info.metric.t2w[0].end).total_seconds()
            worst_slack = min(worst_slack, speech_cum_dt - calc_cum_dt)

        # thc_print(f"request_info.llm_output = {request_info.llm_output}")
        # import joblib
        # joblib.dump(request_info.flow_cache.cpu().numpy(), 'flow_cache.pkl')
        # joblib.dump(request_info.speech.cpu().numpy(), 'speech.pkl')

        request_info.metric.e2e.stop()
        first_latency = (request_info.metric.t2w[0].end - request_info.metric.e2e.begin).total_seconds()
        rtf = request_info.metric.e2e.dt / request_info.speech_cum_dt
        thc_print(f"complete_request: request_idx: {request_idx}, {request_id}, "
                    f"submit_time = {request_info.metric.e2e.begin:%H:%M:%S.%f}, "
                    f"e2e = {request_info.metric.e2e.dt:.3f}, "
                    f"slack = {worst_slack:.3f}, "
                    f"first_latency = {first_latency:.3f}, "
                    f"rtf = {rtf:.3f}, "
                    f"speech_dt = {request_info.speech_cum_dt:.3f}")
        if False:# worst_slack < 0:
            for n in range(len(request_info.sections)):
                speech_dt = request_info.sections[n].speech_dt
                speech_cum_dt = sum([section.speech_dt for section in request_info.sections[: n]])
                calc_cum_dt = (request_info.metric.t2w[n].end - request_info.metric.t2w[0].end).total_seconds()
                rtf = calc_cum_dt / speech_cum_dt if n > 0 else 0.0
                if n == 0:
                    calc_dt = (request_info.metric.t2w[n].end - request_info.metric.e2e.begin).total_seconds()
                    slack = 9.999
                else:
                    calc_dt = (request_info.metric.t2w[n].end - request_info.metric.t2w[n - 1].end).total_seconds()
                    slack = speech_cum_dt - calc_cum_dt
                thc_print(f"complete_request: request_idx: {request_idx}, {request_id}, "
                            f'section = {n}, '
                            f"speech_dt = {speech_dt:.3f}, "
                            f"t2w_calc_dt = {request_info.metric.t2w[n].dt:.3f}, "
                            f"calc_dt = {calc_dt:.3f}, "
                            f"speech_cum_dt = {speech_cum_dt:.3f}, "
                            f"calc_cum_dt = {calc_cum_dt:.3f}, "
                            f"slack = {slack:.3f}, "
                            f"rtf = {rtf:.3f}")

    def process_request(self, mode, tts_text, spk_id, stream_flag, context):
        # thc_print(f"process_request: tts_text = {tts_text}")
        request_info = self.add_request(mode, tts_text, spk_id, stream_flag)
        request_idx = request_info.idx
        request_id = request_info.request_id

        n_section = 0
        while True:
            section = self.check_context_and_get_queue(request_info, context, request_info.section_queue, timeout=10, q_name='section')
            if section is None:
                with self.lock:
                    self.requests_info.pop(request_id)
                    self.requests_id[request_idx] = None
                return
            assert section.idx[2] == n_section, f"section.idx[2] = {section.idx[2]}, n_section = {n_section}"
            n_section += 1

            # thc_print(f"{request_id}, {section.speech_cpu}, {section.speech_dt}")

            if section.finalize and request_info.done:
                release = self.check_context_and_get_queue(request_info, context, request_info.release, timeout=30, q_name='release')
                if release is None:
                    with self.lock:
                        self.requests_info.pop(request_id)
                        self.requests_id[request_idx] = None
                    return

                self.postprocess_request(request_info)

            yield section.speech_cpu, section.finalize and request_info.done

            if request_info.done:
                break

    def warmup(self):
        tts_text = '1,2,3,4,5,6,7,8,9,10'
        # pick an available speaker instead of hardcoding 'spk1'
        spk_id = next(iter(self.prompt_emb), None)
        if spk_id is None:
            logging.warning('warmup skipped: no speaker available')
            return
        stream_flag = True

        for n_req in range(args.max_conc):
            request_info = self.add_request('zero_shot_by_id', tts_text, spk_id, stream_flag)

        while True:
            time.sleep(1)

            with self.lock:
                request_num = len(self.requests_info)

            if request_num == 0:
                break

    def Inference(self, request, context):
        # thc_print(f"Inference: context = {context}, PID = {os.getpid()}, TID = {threading.get_ident()}")
        # thc_print(f"Inference: tts_text = {request.zero_shot_by_id_request.tts_text}, spk_id = {request.zero_shot_by_id_request.spk_id}")

        if 1:
            if request.HasField('sft_request'):
                mode = 'sft'
                tts_text = request.sft_request.tts_text
                spk_id = request.sft_request.spk_id
                stream_flag = request.sft_request.stream
            elif request.HasField('zero_shot_by_id_request'):
                mode = 'zero_shot_by_id'
                tts_text = request.zero_shot_by_id_request.tts_text
                spk_id = request.zero_shot_by_id_request.spk_id
                stream_flag = request.zero_shot_by_id_request.stream
            else:
                raise NotImplementedError('Mode not implemented!')

            for tts_speech, done_flag in self.process_request(mode, tts_text, spk_id, stream_flag, context):
                response = cosyvoice_pb2.Response()
                response.tts_audio = tts_speech.tobytes()
                yield response
                if done_flag:
                    break

        else:
            if request.HasField('zero_shot_by_id_request'):
                logging.info('get zero_shot_by_id inference request')
                model_output = self.cosyvoice.inference_zero_shot_by_id(
                    request.zero_shot_by_id_request.tts_text,
                    request.zero_shot_by_id_request.spk_id,
                    stream=request.zero_shot_by_id_request.stream
                )
            else:
                assert False
                logging.info('get instruct inference request')
                model_output = self.cosyvoice.inference_instruct(request.instruct_request.tts_text,
                                                                request.instruct_request.spk_id,
                                                                request.instruct_request.instruct_text)

            logging.info('send inference response')

            count=0
            for i in model_output:
                response = cosyvoice_pb2.Response()
                response.tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
                yield response
                count+=1
                if count==4 and self.args.enable_vllm_profile:
                    assert False
                    try:
                        self.cosyvoice.model.llm.vllm.stop_profile()
                        logging.info('vLLM profile stopped')
                    except Exception as e:
                        logging.warning(f'Failed to stop vLLM profile: {e}')


def _resolve_speaker_info_dir(args):
    """Resolve --speaker_info_dir: None/empty -> <model_dir>/speaker_info.
    If the requested dir has no speaker_info.json, fall back to
    <model_dir>/speaker_info (and warn). Never fatal."""
    model_dir = args.model_dir
    d = args.speaker_info_dir
    default_dir = os.path.join(model_dir, 'speaker_info') if os.path.isdir(model_dir) else None
    if d and os.path.exists(os.path.join(d, 'speaker_info.json')):
        return
    if default_dir and os.path.exists(os.path.join(default_dir, 'speaker_info.json')):
        if d:
            logging.warning(f"speaker_info.json not found in {d}, fall back to {default_dir}")
        args.speaker_info_dir = default_dir
        return
    if d:
        logging.warning(f"speaker_info.json not found in {d} nor <model_dir>/speaker_info; "
                        "zero_shot_by_id may be unavailable (sft spk still works)")
    else:
        logging.warning("speaker_info.json not found in <model_dir>/speaker_info; "
                        "zero_shot_by_id may be unavailable (sft spk still works)")


def main():
    _resolve_speaker_info_dir(args)
    service = CosyVoiceServiceImpl(args)
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_conc), maximum_concurrent_rpcs=args.max_conc)
    cosyvoice_pb2_grpc.add_CosyVoiceServicer_to_server(service, grpcServer)
    grpcServer.add_insecure_port('0.0.0.0:{}'.format(args.port))
    grpcServer.start()

    # Non-graph: pre-warm already ran in __init__ (before gRPC threads) to avoid
    # CANN TBE fork-in-multithread crashes. graph_mode: MindIE-style warmup
    # runs here AFTER gRPC start (t2w graph compiled at Token2Wave init).
    if getattr(args, 'graph_mode', False):
        thc_print("warmup (post-grpc, graph_mode): start")
        service.warmup()
        thc_print("warmup (post-grpc, graph_mode): done")

    logging.info("server listening on 0.0.0.0:{}".format(args.port))
    grpcServer.wait_for_termination()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--max_conc',
                        type=int,
                        default=4)
    parser.add_argument('--model_dir',
                        type=str,
                        default='iic/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    parser.add_argument('--graph_mode',
                        action='store_true',
                        help='whether to use graph mode')
    parser.add_argument('--speaker_info_dir',
                        type=str,
                        default=None,
                        help='directory containing speaker_info.json (default: <model_dir>/speaker_info)')
    parser.add_argument('--enable_vllm_profile',
                        action='store_true',
                        help='whether to enable vLLM profiling (default: False)')
    parser.add_argument('--prewarm_timeout',
                        type=int,
                        default=300,
                        help='seconds to wait for the pre-gRPC warmup before continuing (default 300)')
    parser.add_argument('--no_prewarm',
                        action='store_true',
                        help='skip the pre-gRPC warmup (not recommended: operator compilation may fork in multithreaded gRPC context)')
    args = parser.parse_args()
    logging.info(f"args：{args}")
    main()
