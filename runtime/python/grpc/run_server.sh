#!/bin/bash

# 基础环境变量配置
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES=2
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
#export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
#export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_ASCEND_ENABLE_MLP_OPTIMIZE=1
#export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
#export VLLM_ASCEND_ENABLE_FLASHCOMM=1
#export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION=1
#export VLLM_ASCEND_TRACE_RECOMPILES=0
#export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:256"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
# 日志调试配置：查看图断点和重新编译原因
#export TORCH_LOGS="graph_breaks,recompiles"

# 配置参数
MODEL_DIR="/home/Fun-CosyVoice3-0.5B-2512/"
MAX_CONC=8


# -graph_mode --use_process_pool
# python server.py --port 50099 --model_dir "$MODEL_DIR" --max_conc "$MAX_CONC" --num_worker=8 --use_process_pool --graph_mode
# python server.py --port 50099 --model_dir "$MODEL_DIR" --max_conc "$MAX_CONC" --num_worker=8 --use_process_pool
# python server.py --port 50099 --model_dir "$MODEL_DIR" --max_conc "$MAX_CONC" --graph_mode
# python server.py --port 50099 --model_dir "$MODEL_DIR" --max_conc "$MAX_CONC"
python server.py --port 50099 --model_dir "$MODEL_DIR" --max_conc "$MAX_CONC" --graph_mode
# --enable_vllm_profile

# --- 逻辑修改部分 ---
# 使用环境变量或脚本参数来控制是否开启 graph_mode
# 运行示例：GRAPH_MODE=true bash start.sh
# if [ "$GRAPH_MODE" = "true" ]; then
#     echo "Starting server in GRAPH MODE,MAX_CONC=$MAX_CONC..."
#     python server.py \
#         --model_dir "$MODEL_DIR" \
#         --max_conc "$MAX_CONC" \
#         --graph_mode
# else
#     echo "Starting server in default mode,MAX_CONC=$MAX_CONC..."
#     python server.py \
#         --model_dir "$MODEL_DIR" \
#         --max_conc "$MAX_CONC"
# fi
