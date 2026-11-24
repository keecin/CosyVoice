export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES=0
# npu-smi info -t topo 查询npu情况
taskset -c 192-223 python infer.py
