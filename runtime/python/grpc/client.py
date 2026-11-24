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
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
import logging
import argparse
import torchaudio
import cosyvoice_pb2
import cosyvoice_pb2_grpc
import grpc
import torch
import numpy as np
from cosyvoice.utils.file_utils import load_wav
import time


def main():
    with grpc.insecure_channel("{}:{}".format(args.host, args.port)) as channel:
        stub = cosyvoice_pb2_grpc.CosyVoiceStub(channel)
        request = cosyvoice_pb2.Request()
        if args.mode == 'sft':
            logging.info('send sft request')
            sft_request = cosyvoice_pb2.sftRequest()
            sft_request.spk_id = args.spk_id
            sft_request.tts_text = args.tts_text
            request.sft_request.CopyFrom(sft_request)
        elif args.mode == 'zero_shot':
            logging.info('send zero_shot request')
            zero_shot_request = cosyvoice_pb2.zeroshotRequest()
            zero_shot_request.tts_text = args.tts_text
            zero_shot_request.prompt_text = args.prompt_text
            prompt_speech = load_wav(args.prompt_wav, 16000)
            zero_shot_request.prompt_audio = (prompt_speech.numpy() * (2**15)).astype(np.int16).tobytes()
            # 关键：需要定义 proto 中 zeroshotRequest 有 stream 字段
            zero_shot_request.stream = args.stream
            request.zero_shot_request.CopyFrom(zero_shot_request)
        elif args.mode == 'cross_lingual':
            logging.info('send cross_lingual request')
            cross_lingual_request = cosyvoice_pb2.crosslingualRequest()
            cross_lingual_request.tts_text = args.tts_text
            prompt_speech = load_wav(args.prompt_wav, 16000)
            cross_lingual_request.prompt_audio = (prompt_speech.numpy() * (2**15)).astype(np.int16).tobytes()
            request.cross_lingual_request.CopyFrom(cross_lingual_request)
        elif args.mode == 'zero_shot_by_id':
            logging.info('send zero_shot_by_id request')
            zero_shot_by_id_request = cosyvoice_pb2.zeroshotByIdRequest()
            zero_shot_by_id_request.tts_text = args.tts_text
            zero_shot_by_id_request.spk_id = args.spk_id
            zero_shot_by_id_request.stream = args.stream
            request.zero_shot_by_id_request.CopyFrom(zero_shot_by_id_request)
        else:
            logging.info('send instruct request')
            instruct_request = cosyvoice_pb2.instructRequest()
            instruct_request.tts_text = args.tts_text
            instruct_request.spk_id = args.spk_id
            instruct_request.instruct_text = args.instruct_text
            request.instruct_request.CopyFrom(instruct_request)

        # 开始统计时间
        start_request_time = time.time()
        response = stub.Inference(request)
        
        first_packet_time = None
        all_audio_chunks = []
        
        logging.info('Waiting for response...')
        
        for r in response:
            if first_packet_time is None:
                first_packet_time = time.time()
                first_fix_latency = (first_packet_time - start_request_time) * 1000
                # logging.info(f'[METRIC] First Fix Latency (首包时延): {first_fix_latency:.2f} ms')
            
            chunk = np.frombuffer(r.tts_audio, dtype=np.int16)
            all_audio_chunks.append(chunk)

        end_inference_time = time.time()
        
        if not all_audio_chunks:
            logging.error("No audio received from server.")
            return

        # 计算统计数据
        full_audio = np.concatenate(all_audio_chunks)
        
        # 音频总时长 (秒) = 总采样点数 / 采样率
        audio_duration = len(full_audio) / target_sr
        
        # 总推理耗时 (秒)
        total_inference_time = end_inference_time - start_request_time
        
        # RTF (Real Time Factor) = 总推理耗时 / 音频时长
        # RTF < 1 表示推理速度快于播放速度
        rtf = total_inference_time / audio_duration if audio_duration > 0 else 0

        # 打印性能指标
        print("-" * 30)
        print(f"推理模式: {'流式 (Stream)' if args.stream else '非流式 (Batch)'}")
        print(f"首包时延: {first_fix_latency:.0f} ms")
        print(f"音频时长: {audio_duration:.1f} s")
        print(f"总推理耗时: {total_inference_time:.1f} s")
        print(f"RTF (实时率): {rtf:.2f}")
        print("-" * 30)

        # 转换为 Tensor 准备保存
        tts_speech = torch.from_numpy(full_audio).unsqueeze(dim=0).float() / (2**15)
        logging.info('save response to {}'.format(args.tts_wav))
        torchaudio.save(args.tts_wav, tts_speech, target_sr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',
                        type=str,
                        default='0.0.0.0')
    parser.add_argument('--port',
                        type=int,
                        default='50000')
    parser.add_argument('--mode',
                        default='sft',
                        choices=['sft', 'zero_shot', 'cross_lingual', 'instruct', 'zero_shot_by_id'],
                        help='request mode')
    parser.add_argument('--tts_text',
                        type=str,
                        default='旅游是一种很好的放松方式，可以让我们暂时远离繁忙的工作，享受大自然的美好与宁静。')
    parser.add_argument('--spk_id',
                        type=str,
                        default='spk1')
    parser.add_argument('--prompt_text',
                        type=str,
                        default='You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。')
    parser.add_argument('--prompt_wav',
                        type=str,
                        default='/data/code/CosyVoice/asset/zero_shot_prompt.wav')
    parser.add_argument('--instruct_text',
                        type=str,
                        default='Theo \'Crimson\', is a fiery, passionate rebel leader. \
                                 Fights with fervor for justice, but struggles with impulsiveness.')
    parser.add_argument('--tts_wav',
                        type=str,
                        default='demo.wav')
    # 新增 stream 开关
    parser.add_argument('--stream', 
                        action='store_true', 
                        help='whether to use streaming inference')
    args = parser.parse_args()
    prompt_sr, target_sr = 16000, 24000
    main()
    # python client.py --mode zero_shot_by_id --tts_text “欢迎使用CosyVoice！”
