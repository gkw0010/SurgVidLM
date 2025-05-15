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
pip install flash-attn --no-build-isolation
```

## Dataset

Download the dataset from [here](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155180074_link_cuhk_edu_hk/ESXzk2TtCmpKjiDo3USdKbIBsWLbrYy17ocy9Z8eyr-RXQ?e=uD5Yoi). Note that the test set of our SVU-31K is available, the training set will be publicly available soon.

After downloading the data, put all the `json` files into `./LLaMA-Factory/data` and change the videos path to your local path.

## Train

Before training, please download the pre-trained checkpoint of `Qwen2-VL-7B-Instruct` following the instruction in [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL).

We fine-tune our model based on LLaMA-Factory, ensure the `model_name_or_path` in the config file points to the pre-trained model's directory.

For stage 1 training(full video understanding), run 
```bash
sh stage1_script.sh
```
For stage 2 training(video perception and temporal reasoning), run 
```bash
sh stage2_script.sh
```

## Inference

```bash
# Stage 1 inference
CUDA_VISIBLE_DEVICES=0,1 python full_video_inference.py
# Stage 2 inference
CUDA_VISIBLE_DEVICES=0,1 python clip_inference.py
```

