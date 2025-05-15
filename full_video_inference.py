from PIL.Image import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from vision_process import process_vision_info
import os
import torch

# NOTE: put the export checkpoint path of stage 1 here!!
path=r"CKPT_PATH"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    path, torch_dtype=torch.bfloat16, device_map="auto",
    attn_implementation="flash_attention_2"
)

processor = AutoProcessor.from_pretrained(path)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "VIDEO_PATH",
            },
            {"type": "text", "text": "Describe the video in detail."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs, clip_inputs = process_vision_info(messages)
assert clip_inputs is None, 'clip inputs should be None in stage 1 inference'
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    clips=clip_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

