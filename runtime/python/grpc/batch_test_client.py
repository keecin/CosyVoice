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
import time
import datetime

from cosyvoice.utils.file_utils import load_wav
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Barrier, Lock

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

TARGET_SR = 24000


def percentile(data, p):
    """计算分位数"""
    if len(data) == 0:
        return 0
    return np.percentile(data, p)


def build_request(args, prompt_audio_bytes, tts_text):

    request = cosyvoice_pb2.Request()

    if args.mode == 'zero_shot':

        zero_shot_request = cosyvoice_pb2.zeroshotRequest()

        zero_shot_request.tts_text = tts_text
        zero_shot_request.prompt_text = args.prompt_text
        zero_shot_request.prompt_audio = prompt_audio_bytes
        zero_shot_request.stream = args.stream

        request.zero_shot_request.CopyFrom(zero_shot_request)

    elif args.mode == 'zero_shot_by_id':

        req = cosyvoice_pb2.zeroshotByIdRequest()

        req.tts_text = tts_text
        req.spk_id = args.spk_id
        req.stream = args.stream

        request.zero_shot_by_id_request.CopyFrom(req)

    else:

        sft_request = cosyvoice_pb2.sftRequest(
            spk_id=args.spk_id,
            tts_text=tts_text,
            stream=args.stream
        )

        request.sft_request.CopyFrom(sft_request)

    return request


def single_request(task_id, args, stub, barrier, prompt_audio_bytes, tts_text, use_barrier):

    try:

        request = build_request(args, prompt_audio_bytes, tts_text)

        # 只有启用同步模式时才使用barrier
        if use_barrier and barrier is not None:
            barrier.wait()

        start_t = time.time()

        response = stub.Inference(request)

        first_packet_time = None
        chunks = []

        for r in response:

            if first_packet_time is None:
                first_packet_time = time.time()

            chunks.append(np.frombuffer(r.tts_audio, dtype=np.int16))

        end_t = time.time()

        if len(chunks) == 0:
            return {"success": False}

        full_audio = np.concatenate(chunks)

        audio_dur = len(full_audio) / TARGET_SR

        total_time = end_t - start_t

        first_fix = (first_packet_time - start_t) * 1000 if first_packet_time else 0

        rtf = total_time / audio_dur if audio_dur > 0 else 0

        print(f"start = {datetime.datetime.fromtimestamp(start_t):%H:%M:%S.%f}, "
              f"first = {datetime.datetime.fromtimestamp(first_packet_time):%H:%M:%S.%f}, "
              f"end = {datetime.datetime.fromtimestamp(end_t):%H:%M:%S.%f}, "
              f"total_time = {total_time:.3f}, "
              f"audio_dur = {audio_dur:.3f}, "
              f"first = {first_fix:.3f}, "
              f"rtf = {rtf:.3f}")

        if args.save_task_id is not None and (task_id == args.save_task_id or args.save_task_id == -1):

            os.makedirs(args.save_audio_dir, exist_ok=True)

            save_path = os.path.join(
                args.save_audio_dir,
                f"task_{task_id}.wav"
            )

            torchaudio.save(
                save_path,
                torch.from_numpy(full_audio).unsqueeze(0),
                TARGET_SR
            )

            logging.info(f"Audio saved: {save_path}")

        return {
            "success": True,
            "first_fix": first_fix,
            "rtf": rtf,
            "audio_dur": audio_dur
        }

    except Exception as e:

        logging.error(f"Task {task_id} failed: {str(e)}")

        return {"success": False}


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=50000)

    parser.add_argument('--mode',
                        default='sft',
                        choices=['sft', 'zero_shot', 'zero_shot_by_id'])

    parser.add_argument('--tts_text', type=str,
                        default='旅游是一种很好的放松方式，可以让我们暂时远离繁忙的工作，享受大自然的美好与宁静。')
    
    #parser.add_argument('--tts_text', type=str,
    #                    default='旅游是一种很好的放松方式，可以让我们暂时远离繁忙的工作。')

    parser.add_argument('--spk_id', type=str, default='spk1')

    parser.add_argument('--prompt_text', type=str,
                        default='You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。')
    
    #parser.add_argument('--prompt_text', type=str,
    #                    default='You are a helpful assistant.')

    parser.add_argument('--prompt_wav', type=str,
                        default='/data/code/CosyVoice/asset/zero_shot_prompt.wav')

    parser.add_argument('--stream', action='store_true')

    parser.add_argument('--total_req', type=int, default=100)

    parser.add_argument('--concurrency', type=int, default=8)

    parser.add_argument('--save_task_id', type=int, default=None)

    parser.add_argument('--save_audio_dir', type=str, default='./saved_audio')

    parser.add_argument('--text_file', type=str, default=None,
                        help='文本文件路径，每行一个句子，用于多样化测试')

    parser.add_argument('--sync_start', action='store_true',
                        help='启用同步启动模式：等待所有并发线程就绪后同时发起请求（会降低吞吐量）')

    args = parser.parse_args()

    logging.info(f"Start benchmark total={args.total_req} concurrency={args.concurrency}")

    # 准备测试文本列表
    text_list = []
    if args.text_file and os.path.exists(args.text_file):
        # 从文件读取多个测试文本
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text_list = [line.strip() for line in f if line.strip()]
        logging.info(f"Loaded {len(text_list)} texts from {args.text_file}")

    if not text_list:
        # 如果没有提供文本文件，使用默认文本并添加序号使其多样化
        text_list = [args.tts_text]
        logging.info("Using default text with task_id suffix for diversity")

    prompt_audio_bytes = None

    if args.mode == "zero_shot":

        prompt_speech = load_wav(args.prompt_wav, 16000)

        prompt_audio_bytes = (
            (prompt_speech.numpy() * (2**15))
            .astype(np.int16)
            .tobytes()
        )

    '''options = [
        ("grpc.keepalive_time_ms", 30000),
        ("grpc.keepalive_timeout_ms", 10000),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.enable_http_proxy", 0),
    ]'''

    options = [
        ("grpc.keepalive_time_ms", 120000),
        ("grpc.keepalive_timeout_ms", 60000),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.enable_http_proxy", 0),
    ]

    channel = grpc.insecure_channel(
        f"{args.host}:{args.port}",
        options=options
    )

    stub = cosyvoice_pb2_grpc.CosyVoiceStub(channel)

    # 只有启用同步模式时才创建barrier
    barrier = Barrier(args.concurrency) if args.sync_start else None

    if args.sync_start:
        logging.info("同步启动模式已启用：每批次线程将同时开始请求")
    else:
        logging.info("连续并发模式：线程完成后立即开始下一个请求（推荐用于吞吐量测试）")

    results = []

    progress = 0
    progress_lock = Lock()

    start_bench = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:

        futures = []
        for i in range(args.total_req):
            # 为每个请求分配不同的文本
            if len(text_list) == 1:
                # 只有一个文本时，添加序号使其不同
                tts_text = f"{text_list[0]}第{i+1}句。"
            else:
                # 有多个文本时，循环使用
                tts_text = text_list[i % len(text_list)]

            future = executor.submit(
                single_request,
                i,
                args,
                stub,
                barrier,
                prompt_audio_bytes,
                tts_text,
                args.sync_start
            )
            futures.append(future)

        for future in as_completed(futures):

            r = future.result()

            results.append(r)

            with progress_lock:

                progress += 1

                percent = progress / args.total_req * 100

                print(f"Progress: {progress}/{args.total_req} ({percent:.1f}%)")

    end_bench = time.time()

    success_results = [r for r in results if r["success"]]

    if not success_results:
        print("All requests failed.")
        return

    first_fix_list = [r["first_fix"] for r in success_results]
    rtf_list = [r["rtf"] for r in success_results]
    audio_dur_list = [r["audio_dur"] for r in success_results]

    total_audio_dur = sum(audio_dur_list)

    throughput = total_audio_dur / (end_bench - start_bench)

    avg_first_fix = sum(r['first_fix']
                            for r in success_results) / len(success_results)
    avg_rtf = sum(r['rtf']
                      for r in success_results) / len(success_results)

    print("\n" + "="*20)
    print("并发测试统计报告")
    print("="*20)

    print(f"成功率: {len(success_results)}/{args.total_req}")

    print(f"总运行时间: {end_bench-start_bench:.2f}s")

    print("\n首包时延统计(ms)")

    print(f"平均首包时延: {avg_first_fix:.0f}")
    print(f"P50: {percentile(first_fix_list,50):.0f}")
    print(f"P70: {percentile(first_fix_list,70):.0f}")
    print(f"P80: {percentile(first_fix_list,80):.0f}")
    print(f"P90: {percentile(first_fix_list,90):.0f}")
    print(f"P99: {percentile(first_fix_list,99):.0f}")

    print("\nRTF统计")

    print(f"平均RTF: {avg_rtf:.2f}")
    print(f"P50: {percentile(rtf_list,50):.2f}")
    print(f"P70: {percentile(rtf_list,70):.2f}")
    print(f"P80: {percentile(rtf_list,80):.2f}")
    print(f"P90: {percentile(rtf_list,90):.2f}")
    print(f"P99: {percentile(rtf_list,99):.2f}")

    print(f"\n系统吞吐量(audio_sec / wall_sec): {throughput:.2f}x")

    print("="*20)


if __name__ == "__main__":
    main()