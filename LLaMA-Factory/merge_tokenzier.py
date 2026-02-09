from datasets import load_from_disk, DatasetDict

train_ds = load_from_disk("/mnt/data2/wgk/surgvidlm_project/LLaMA-Factory/saves/tokenizer_video_clip_stage2_1fps_change_prompt")["train"]   # 你原来的缓存
eval_ds  = load_from_disk("/mnt/data2/wgk/surgvidlm_project/LLaMA-Factory/saves/tokenizer_video_clip_stage2_1fps_change_prompt_eval")["validation_svu-31k-video-clip-eval"]    # 刚生成的评估缓存

merged = DatasetDict({"train": train_ds, "validation_svu-31k-video-clip-eval": eval_ds})
merged.save_to_disk("/mnt/data2/wgk/surgvidlm_project/LLaMA-Factory/saves/tokenizer_video_clip_stage2_1fps_change_prompt_weval")  # 建议新目录
print(merged)   