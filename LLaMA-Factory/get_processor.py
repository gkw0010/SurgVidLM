from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("/data2/wgk/SurgVidLM/LLaMA-Factory/saves/model/merged/qwen2-vl-full-video")
print(processor)