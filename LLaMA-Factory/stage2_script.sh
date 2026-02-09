cd /mnt/data2/wgk/surgvidlm_project/LLaMA-Factory

export DECORD_EOF_RETRY_MAX=20480
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DISABLE_VERSION_CHECK=1 \
llamafactory-cli train /mnt/data2/wgk/surgvidlm_project/LLaMA-Factory/examples/train_lora/surgvidlm_lora_sft_stage2.yaml 
# 2>&1 | tee logs/train_clip_2fps_change_prompt_new_attention.log

# llamafactory-cli train /mnt/data2/wgk/surgvidlm_project/LLaMA-Factory/examples/train_lora/surgvidlm_lora_sft_stage2.yaml \
# 2>&1 | tee logs/output_clip_4fps_change_prompt_new_attention.log
# #check

# DISABLE_VERSION_CHECK=1 llamafactory-cli export examples/merge_lora/surgvidlm_lora_sft_stage2.yaml

# cd ..
