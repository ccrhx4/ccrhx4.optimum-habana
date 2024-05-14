#!/bin/bash

export DEEPSPEED_HPU_ZERO3_SYNC_MARK_STEP_REQUIRED=1
export num_cards=8
export USE_CUDA=1

accelerate launch --config_file "deepspeed_config_z3_qlora.yaml" run_lora_clm.py \
	--model_name_or_path "mistralai/Mixtral-8x7B-v0.1" \
	--dataset_name tatsu-lab/alpaca \
	--bf16 True \
	--dataset_concatenation \
	--per_device_train_batch_size 12 \
    	--per_device_eval_batch_size 8 \
    	--gradient_accumulation_steps 1 \
	--gradient_checkpointing True\
	--use_reentrant True \
	--max_seq_len 2048 \
    	--do_train \
     	--learning_rate 1e-4 \
	--logging_steps 1 \
	--overwrite_output_dir \
	--log_level info \
	--output_dir ./mixtral_peft_finetuned_model \
	--peft lora \
	--lora_target_modules q_proj v_proj \
     	--lora_rank 4 \
	--use_cache False \
	--use_flash_attn True \
	--use_4bit_quantization True \
        --use_nested_quant True \
        --bnb_4bit_compute_dtype "bfloat16" \
        --bnb_4bit_quant_storage_dtype "bfloat16"

