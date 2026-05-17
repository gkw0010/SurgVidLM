cd /mnt/data2/wgk/surgvidlm_project/opensource/SurgVidLM/LLaMA-Factory

export DECORD_EOF_RETRY_MAX=20480
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=7 DISABLE_VERSION_CHECK=1 \
llamafactory-cli train examples/train_lora/surgvidlm_lora_sft_stage2.yaml \


