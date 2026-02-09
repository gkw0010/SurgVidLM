from datasets import load_from_disk

dataset = load_from_disk("/data2/wgk/SurgVidLM/LLaMA-Factory/saves/tokenizer_video_clip_stage2_2fps")
print(dataset)  # 查看数据集元数据