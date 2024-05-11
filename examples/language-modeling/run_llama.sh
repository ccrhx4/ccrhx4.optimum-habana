PT_HPU_MAX_COMPOUND_OP_SIZE=10 DEEPSPEED_HPU_ZERO3_SYNC_MARK_STEP_REQUIRED=1 \
python3 ../gaudi_spawn.py --use_deepspeed  --world_size 8  run_lora_clm.py \
  --model_name_or_path meta-llama/Llama-2-70b-hf \
  --deepspeed llama2_ds_zero3_config.json \
  --dataset_name tatsu-lab/alpaca \
  --bf16 True \
  --output_dir ./lora_out \
  --num_train_epochs 2 \
  --max_seq_len 2048 \
  --per_device_train_batch_size 10 \
  --per_device_eval_batch_size 10 \
  --evaluation_strategy epoch \
  --eval_delay 2 \
  --save_strategy no \
  --learning_rate 0.0018 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --dataset_concatenation \
  --attn_softmax_bf16 True \
  --do_train \
  --do_eval \
  --use_habana \
  --use_lazy_mode \
  --throughput_warmup_steps 3 \
  --lora_rank 4 \
  --lora_target_modules "q_proj" "v_proj" "k_proj" "o_proj" \
  --validation_split_percentage 4 \
  --use_flash_attention True \
  --flash_attention_causal_mask True \
  --max_steps 10 

  
