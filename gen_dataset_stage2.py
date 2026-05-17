import json

# path to full video inference result and stage2 original dataset
file1 = '/mnt/data2/wgk/surg_data/clip_train_set_no_abstract_final.json'
file2 = '/mnt/data2/wgk/test/result/full_video_train/inference_result.json'


with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
    clip_train = json.load(f1)
    full_video_description = json.load(f2)

print(len(clip_train),len(full_video_description))


print(isinstance(full_video_description, list))
entries = full_video_description

# # construct video-to-answer mapping
vid_to_answer = {entry['video']: entry['answer'] for entry in entries}


prompt_prefix="You are given a video segment and the background information of the full video: "
prompt_postfix=" Use the background information as context and focus on the video segment to provide a fine-grained reasoning answer to the following question: "
for rec in clip_train:
    vid = rec['videos'][0]
    answer = vid_to_answer.get(vid)
    print(vid)
    if answer is None:
        print(rec['id'])
        raise ValueError("answer does not exist")
    
    else:
        rec['messages'][0]['content']= "<video>"+prompt_prefix + "\n" +answer + "\n"+ prompt_postfix + rec['messages'][0]['content'].replace("<video>","") #重新构建问题，增加提示语

# 输出到新文件
output_path = '/mnt/data2/wgk/test/surgvidlm_ablation/clip_test_set_stage2.json' # test

with open(output_path, 'w', encoding='utf-8') as fout:
    json.dump(clip_train, fout, ensure_ascii=False, indent=2)

print(f"generation complete，saved to：{output_path}")
