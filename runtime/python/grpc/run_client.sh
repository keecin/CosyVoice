python client.py --stream --mode zero_shot_by_id 
python batch_test_client.py --port 50099 --stream --mode zero_shot_by_id  --total_req 8 --concurrency 8
python batch_test_client.py --port 50099 --stream --mode zero_shot_by_id --save_task_id 5 --save_audio_dir ./my_saved_audio  --total_req 24 --concurrency 8

python batch_test_client.py --port 50099 --stream --mode zero_shot_by_id  --total_req 8 --concurrency 8 --text_file test.txt
python batch_test_client.py --port 50099 --stream --mode zero_shot_by_id --total_req 8 --concurrency 8 --text_file test.txt --save_task_id 5 --save_audio_dir ./my_saved_audio

python batch_test_client.py --port 50099 --stream --mode sft --spk_id 001 --total_req 8 --concurrency 8 --text_file test.txt
python batch_test_client.py --port 50099 --stream --mode sft --spk_id 001 --total_req 8 --concurrency 8 --text_file test.txt --save_task_id 5 --save_audio_dir ./my_saved_audio

