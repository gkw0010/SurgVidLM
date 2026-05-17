# SurgVidLM

## Setup

```bash
git clone https://github.com/gkw0010/SurgVidLM.git && cd SurgVidLM

conda create -n surgvidlm python=3.10 
conda activate surgvidlm

# install LLaMA-Factory and  transformers adapted for SurgVidLM
cd LLaMA-Factory && pip install -e ".[torch,metrics]" && cd .. 
cd transformers && pip install . && cd ..

pip install decord
pip install flash-attn==2.8.3 --no-build-isolation
pip install qwen-vl-utils==0.0.9
```

## Dataset

Download the dataset from [here](https://pan.quark.cn/s/0e6f74ff02fc#/list/share). For the convenience of uploading, we have split the video of the test set into two parts. After downloading, please put these two parts together.

Note that the test set of our SVU-31K is available, the training set will be publicly available after the paper is accepted.

After downloading the data, put all the `json` files into `./LLaMA-Factory/data` and change the videos path to your local path.

## Finetune
Due to the large amount of data,  the tokenization process requires a long time to complete. We recommend you to save the tokenizer before training (already implemented in the training config).
### stage 1
For stage 1 fine-tuning, use the following command: 
```Shell
sh train_stage1.sh
llamafactory-cli export examples/merge_lora/surgvidlm_lora_sft_stage1.yaml
```
You can change the training config at examples/train_lora/surgvidlm_lora_sft_stage1.yaml if needed.

### stage 2
For stage 2 fine-tuning:
1. You first need to run inference on stage 1 training dataset to obtain full video description. Use the following command: 
```Shell
python full_video_inference_batch.py \
    --result_folder="/path/to/your/result_folder" \
    --model_path="/path/to/your/stage1_checkpoint" \
    --data_path="/path/to/your/stage1_training_data.json"
```
2. Use the following command to construct stage 2 fine-tuning dataset:
```Shell
python gen_dataset_stage2.py
```
3. Use the following command for stage 2 fine-tuning:
```Shell
sh train_stage2.sh
llamafactory-cli export examples/merge_lora/surgvidlm_lora_sft_stage2.yaml
```

### stage 2
## Inference
You can download the checkpoint after fine-tuning on our data from [here](https://pan.quark.cn/s/550ae982845e) and do inference.

For single-sample inference, use the following command:
```bash
# Stage 1 inference
python full_video_inference.py
# Stage 2 inference
python clip_inference.py
```

For batch inference, use the following command:
```bash
# Stage 1 inference
python full_video_inference_batch.py \
    --result_folder="/path/to/your/result_folder" \
    --model_path="/path/to/your/model_checkpoint" \
    --data_path="/path/to/your/test_data.json"

# Stage 2 inference
python clip_inference_batch.py \
    --result_folder="/path/to/your/result_folder" \
    --model_path="/path/to/your/model_checkpoint" \
    --data_path="/path/to/your/test_data.json"
```

## Evaluation
For evaluation, run the following command:
```bash
sh evalute_metrics.sh
```