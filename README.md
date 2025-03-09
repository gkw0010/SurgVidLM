# SurgVidLM

## Setup

```bash
git clone https://github.com/gkw0010/SurgVidLM.git --recursive && cd SurgVidLM

conda create -n surgvidlm python=3.10 
conda activate surgvidlm

# install LLaMA-Factory and  transformers adapted for SurgVidLM
cd LLaMA-Factory && pip install -e ".[torch,metrics]" && cd .. 
cd transformers && pip install . && cd ..

pip install decord
pip install flash-attn --no-build-isolation
```

## Dataset

Download the dataset from [here](xxx)

## Train

Before training, please download the pre-trained checkpoint of `Qwen2-VL-7B-Instruct` following the instruction in [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL).

We fine-tune our model based on LLaMA-Factory, ensure the `model_name_or_path` in the config file points to the pre-trained model's directory.

```bash
cd LLaMA-Factory
# for stage 1 training (full video understanding)
llamafactory-cli train examples/train_lora/surgvidlm_lora_sft_stage1.yaml
# after finish training, export the checkpoint of stage 1 for stage 2 training
llamafactory-cli export examples/merge_lora/surgvidlm_lora_sft_stage1.yaml
# for stage 2 training (video perception and temporal reasoning)
llamafactory-cli train examples/train_lora/surgvidlm_lora_sft_stage2.yaml
# after finish training, export the checkpoint
llamafactory-cli export examples/merge_lora/surgvidlm_lora_sft_stage2.yaml
```


## Inference

You can download the checkpoint after fine-tuning on our data from [here](xxx) and do inference.

```bash
# Stage 1 inference
CUDA_VISIBLE_DEVICES=0,1 python full_video_inference.py
# Stage 2 inference
CUDA_VISIBLE_DEVICES=0,1 python clip_inference.py
```

