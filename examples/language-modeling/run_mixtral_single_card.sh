#!/bin/bash

python     run_lora_clm.py     --model_name_or_path "mistralai/Mixtral-8x7B-v0.1"     --bf16 True     --dataset_name tatsu-lab/alpaca     --dataset_concatenation     --per_device_train_batch_size 2     --per_device_eval_batch_size 8     --gradient_accumulation_steps 4     --do_train     --learning_rate 1e-4     --num_train_epochs 3     --logging_steps 10     --save_total_limit 2     --overwrite_output_dir     --log_level info     --save_strategy epoch     --output_dir ./mixtral_peft_finetuned_model     --peft lora     --lora_target_modules q_proj k_proj v_proj    --lora_rank 64     --lora_alpha 16     --use_fast_tokenizer True    --use_habana     --use_lazy_mode    --num_hidden_layers 16

