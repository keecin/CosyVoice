# 测试方法
1. 进入目录`CosyVoice-npu\runtime\python\grpc`
1. 启动server: `bash run_server.sh`，相关参数修改脚本（模型参数model_dir，grpc支持并发数max_conc，进程数num_worker）
1. 启动批量测试: `python batch_test_client.py --stream --mode zero_shot_by_id --save_task_id 555 --save_audio_dir ./`my_saved_audio  `--total_req 24 --concurrency 8`
