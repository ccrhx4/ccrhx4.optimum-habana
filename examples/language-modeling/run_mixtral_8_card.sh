#!/bin/bash

export DEEPSPEED_HPU_ZERO3_SYNC_MARK_STEP_REQUIRED=1
export num_cards=8

python ../gaudi_spawn.py --use_deepspeed --world_size $num_cards run_lora_clm.py \
	--model_name_or_path "mistralai/Mixtral-8x7B-v0.1" \
	--bf16 True \
	--dataset_name tatsu-lab/alpaca \
	--dataset_concatenation \
	--per_device_train_batch_size 4 \
    	--per_device_eval_batch_size 8 \
    	--gradient_accumulation_steps 4 \
    	--do_train \
     	--learning_rate 1e-4 \
	--logging_steps 1 \
	--overwrite_output_dir \
	--log_level info \
	--output_dir ./mixtral_peft_finetuned_model \
	--peft lora \
	--lora_target_modules q_proj v_proj \
     	--lora_rank 4 \
	--use_fast_tokenizer True \
	--use_habana \
	--use_lazy_mode \
	--throughput_warmup_steps 3 \
	--max_steps 50 \
	--deepspeed llama2_ds_zero3_config.json

