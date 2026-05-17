cd /mnt/data2/wgk/surgvidlm_project/surgvidlm_open/LLaMA-Factory

export DECORD_EOF_RETRY_MAX=20480
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1,2,3 DISABLE_VERSION_CHECK=1 \
llamafactory-cli train /mnt/data2/wgk/surgvidlm_project/surgvidlm_open/LLaMA-Factory/examples/train_lora/surgvidlm_lora_sft_stage2.yaml \


