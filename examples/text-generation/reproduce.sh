python oh_inference.py --model_name_or_path Qwen/Qwen2-7B-Instruct --use_hpu_graphs --use_kv_cache --max_new_tokens 512 --prompt "Here is my prompt" --bf16 --top_p 1.0 --top_k 1 --penalty_alpha 1.0
