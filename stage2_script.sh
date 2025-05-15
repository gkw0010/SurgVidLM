cd LLaMA-Factory

CUDA_VISIBLE_DEVICES=0 DISABLE_VERSION_CHECK=1 llamafactory-cli train examples/train_lora/surgvidlm_lora_sft_stage2.yaml

DISABLE_VERSION_CHECK=1 llamafactory-cli export examples/merge_lora/surgvidlm_lora_sft_stage2.yaml

cd ..
