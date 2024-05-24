#!/bin/bash

export USE_CUDA=0
export CUDA_VISIBLE_DEVICES="0"

python  run_lora_clm.py \
	--model_name_or_path "mistralai/Mixtral-8x7B-v0.1" \
	--bf16 True \
	--dataset_name tatsu-lab/alpaca \
	--dataset_concatenation \
	--per_device_train_batch_size 24 \
    	--per_device_eval_batch_size 8 \
    	--gradient_accumulation_steps 1 \
	--max_seq_length 1024 \
	--gradient_checkpointing \
    	--do_train \
     	--learning_rate 1e-4 \
   	--num_train_epochs 3 \
	--logging_steps 10 \
       	--save_total_limit 2 \
	--overwrite_output_dir \
	--log_level info \
	--save_strategy "no" \
	--output_dir ./mixtral_peft_finetuned_model \
	--peft lora \
	--lora_target_modules q_proj k_proj v_proj \
     	--lora_rank 4 \
	--use_fast_tokenizer True \
	--low_cpu_mem_usage True \
	--max_steps 10 \
	--use_lazy_mode \
	--use_habana \
	--use_cache False \
	--use_hpu_graphs_for_training \
	--num_hidden_layers 5 \
	--dataloader_num_workers 1 \
	--profiling_steps 0 \
	--profiling_record_shapes False \
	--profiling_warmup_steps 5

