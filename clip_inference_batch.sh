export DECORD_EOF_RETRY_MAX=20480
python clip_inference_batch.py \
    --result_folder="/path/to/your/result_folder" \
    --model_path="/path/to/your/model_checkpoint" \
    --test_data_path="/path/to/your/test_data.json"