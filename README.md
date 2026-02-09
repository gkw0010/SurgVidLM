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


## Inference
You can download the checkpoint after fine-tuning on our data from [here](https://pan.quark.cn/s/550ae982845e) and do inference.
```bash
# full video inference
sh full_video_inference_batch.sh
# video clip inference
sh clip_inference_batch.sh
```

