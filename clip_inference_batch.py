import argparse
import json
from datetime import datetime
import os
from PIL.Image import Image
import torch
from transformers import  Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from vision_process import process_vision_info

#读模型
# result_folder="/mnt/data2/wgk/test/surgvidlm_ablation/result/surgvidlm_wo_stage2_open"
# model_path="/mnt/data2/wgk/surgvidlm_project/LLaMA-Factory/saves/model/surgvidlm-clip-stage2_1fps/ckpt-1423" #check
# test_data_path="/mnt/data2/wgk/test/surgvidlm_ablation/clip_test_set_stage2_change_prompt.json"


def inference_batch(result_folder, model_path, test_data_path):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    result_path = f"{result_folder}/clip_result.json"  #check
    print("result_path",result_path)


    device = "cuda:1" #check
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
    )
    print(model.device)
    processor = AutoProcessor.from_pretrained(model_path)

    with open(test_data_path, 'r', encoding='utf-8') as file:
        test = json.load(file)
    print(len(test))

    #exclude broken videos
    test=[item for item in test if "lt03enkeller002" not in item['videos'][0] and "vd01en3812" not in item['videos'][0]]
    print(len(test)) 
    print("start-end:",test[0]['id'],test[-1]['id'])
    print(f"test_model:{model_path}")
    print("start inference")

    if not os.path.exists(result_path):
        results = []
    else:
        with open(result_path, 'r', encoding='utf-8') as file:
            results = json.load(file)

    print(len(results))

    tested_ids=[item['id'] for item in results]
    print(tested_ids)
    print(f"already tested:{len(tested_ids)}")

    for i,item in enumerate(test):  
        # if i==10:
        #     break
        if test[i]['id'] not in tested_ids:  
            print(test[i]['id'])
            video_path=test[i]['videos'][0]
            question=item['messages'][0]['content'].replace("<video>","")
            timecode=test[i]['timecodes']
            #开始测
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": f"{video_path}",
                            "max_pixels": 224*224,
                            "fps": 2,  #check
                            "timecode": timecode
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]
            
            with torch.inference_mode():
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs, full_video_inputs = process_vision_info(messages)
                # print("clip shape:",video_inputs[0].shape, " video shape: ",full_video_inputs[0].shape)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    full_videos=full_video_inputs,
                    # fps=2.0,
                    padding=True,
                    return_tensors="pt",
                    # **video_kwargs,
                )
                inputs = inputs.to(device)

                # Inference
                generated_ids = model.generate(**inputs, max_new_tokens=1024)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

            result_json = {
                "id":item['id'],
                # "old_id": item['id'],
                "video": item['videos'][0],
                "question": question,
                "answer": output_text[0],
                "answer_gt": item['messages'][1]['content'],
                "tag":item['tag'],
            }
            results.append(result_json)

            if i%1 ==0 or i==len(test)-1:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
                print(f"[--------------------------------{timestamp}]:inferencing:{i}/{len(test)}--------------------------------")
                print(result_json['id'],result_json['video'],result_json['tag'])
                # print(result_json)
                with open(result_path, "w") as f:
                    json.dump(results, f, indent=4)
            #
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)

    print("result saved in ",result_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full video inference batch")
    parser.add_argument("--result_folder", type=str, required=True, help="Folder to store results")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test JSON file")

    args = parser.parse_args()

    result_folder=args.result_folder
    model_path=args.model_path
    test_data_path=args.test_data_path
    inference_batch(args.result_folder, args.model_path, args.test_data_path)

