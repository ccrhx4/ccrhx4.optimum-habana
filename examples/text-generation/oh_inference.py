#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Conditional text generation on Habana Gaudi/Gaudi2.
"""

import argparse
import json
import logging
import math
import os
import time
from itertools import cycle
from pathlib import Path

import torch
from utils import adjust_batch, count_hpu_graphs, finalize_quantization, initialize_model

from optimum.habana.utils import get_hpu_memory_stats


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def setup_parser(parser):
    # Arguments management
    parser.add_argument("--device", "-d", type=str, choices=["hpu"], help="Device to run", default="hpu")
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model (on the HF Hub or locally).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to perform generation in bf16 precision.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate.")
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=0,
        help="If > 0 then pad and truncate the input sequences to this specified length of tokens. \
            if == 0, then truncate to 16 (original default) \
            if < 0, then do not truncate, use full input prompt",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Input batch size.")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations for benchmarking.")
    parser.add_argument("--n_iterations", type=int, default=5, help="Number of inference iterations for benchmarking.")
    parser.add_argument("--local_rank", type=int, default=0, metavar="N", help="Local process rank.")
    parser.add_argument(
        "--use_kv_cache",
        action="store_true",
        help="Whether to use the key/value cache for decoding. It should speed up generation.",
    )
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        type=str,
        help="Optional argument if you want to assess your model on a given dataset of the HF Hub.",
    )
    parser.add_argument(
        "--column_name",
        default=None,
        type=str,
        help="If `--dataset_name` was given, this will be the name of the column to use as prompts for generation.",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling for generation.",
    )
    parser.add_argument(
        "--num_beams",
        default=1,
        type=int,
        help="Number of beams used for beam search generation. 1 means greedy search will be performed.",
    )
    parser.add_argument(
        "--top_k",
        default=None,
        type=int,
        help="Size of candidate set used for re-ranking in contrastive search. top_k > 1 enables contrastive search.",
    )
    parser.add_argument(
        "--penalty_alpha",
        default=None,
        type=float,
        help="Degeneration penalty for contrastive search. penalty_alpha > 0 enables contrastive search.",
    )
    parser.add_argument(
        "--trim_logits",
        action="store_true",
        help="Calculate logits only for the last token to save memory in the first step.",
    )
    parser.add_argument(
        "--seed",
        default=27,
        type=int,
        help="Seed to use for random generation. Useful to reproduce your runs with `--do_sample`.",
    )
    parser.add_argument(
        "--profiling_warmup_steps",
        default=0,
        type=int,
        help="Number of steps to ignore for profiling.",
    )
    parser.add_argument(
        "--profiling_steps",
        default=0,
        type=int,
        help="Number of steps to capture for profiling.",
    )
    parser.add_argument(
        "--profiling_record_shapes",
        action="store_true",
        help="Record shapes when enabling profiling.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        type=str,
        nargs="*",
        help='Optional argument to give a prompt of your choice as input. Can be a single string (eg: --prompt "Hello world"), or a list of space-separated strings (eg: --prompt "Hello world" "How are you?")',
    )
    parser.add_argument(
        "--bad_words",
        default=None,
        type=str,
        nargs="+",
        help="Optional argument list of words that are not allowed to be generated.",
    )
    parser.add_argument(
        "--force_words",
        default=None,
        type=str,
        nargs="+",
        help="Optional argument list of words that must be generated.",
    )
    parser.add_argument(
        "--assistant_model",
        default=None,
        type=str,
        help="Optional argument to give a path to a draft/assistant model for assisted decoding.",
    )
    parser.add_argument(
        "--peft_model",
        default=None,
        type=str,
        help="Optional argument to give a path to a PEFT model.",
    )
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument(
        "--token",
        default=None,
        type=str,
        help="The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
        "generated when running `huggingface-cli login` (stored in `~/.huggingface`).",
    )
    parser.add_argument(
        "--model_revision",
        default="main",
        type=str,
        help="The specific model version to use (can be a branch name, tag name or commit id).",
    )
    parser.add_argument(
        "--attn_softmax_bf16",
        action="store_true",
        help="Whether to run attention softmax layer in lower precision provided that the model supports it and "
        "is also running in lower precision.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Output directory to store results in.",
    )
    parser.add_argument(
        "--bucket_size",
        default=-1,
        type=int,
        help="Bucket size to maintain static shapes. If this number is negative (default is -1) \
            then we use `shape = prompt_length + max_new_tokens`. If a positive number is passed \
            we increase the bucket in steps of `bucket_size` instead of allocating to max (`prompt_length + max_new_tokens`).",
    )
    parser.add_argument(
        "--bucket_internal",
        action="store_true",
        help="Split kv sequence into buckets in decode phase. It improves throughput when max_new_tokens is large.",
    )
    parser.add_argument(
        "--dataset_max_samples",
        default=-1,
        type=int,
        help="If a negative number is passed (default = -1) perform inference on the whole dataset, else use only `dataset_max_samples` samples.",
    )
    parser.add_argument(
        "--limit_hpu_graphs",
        action="store_true",
        help="Skip HPU Graph usage for first token to save memory",
    )
    parser.add_argument(
        "--show_graphs_count",
        action="store_true",
        help="Show statistics of HPU graph compilation.",
    )
    parser.add_argument(
        "--reuse_cache",
        action="store_true",
        help="Whether to reuse key/value cache for decoding. It should save memory.",
    )
    parser.add_argument("--verbose_workers", action="store_true", help="Enable output from non-master workers")
    parser.add_argument(
        "--simulate_dyn_prompt",
        default=None,
        type=int,
        nargs="*",
        help="If empty, static prompt is used. If a comma separated list of integers is passed, we warmup and use those shapes for prompt length.",
    )
    parser.add_argument(
        "--reduce_recompile",
        action="store_true",
        help="Preprocess on cpu, and some other optimizations. Useful to prevent recompilations when using dynamic prompts (simulate_dyn_prompt)",
    )

    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Whether to enable Habana Flash Attention, provided that the model supports it.",
    )
    parser.add_argument(
        "--flash_attention_recompute",
        action="store_true",
        help="Whether to enable Habana Flash Attention in recompute mode on first token generation. This gives an opportunity of splitting graph internally which helps reduce memory consumption.",
    )
    parser.add_argument(
        "--flash_attention_causal_mask",
        action="store_true",
        help="Whether to enable Habana Flash Attention in causal mode on first token generation.",
    )
    parser.add_argument(
        "--flash_attention_fast_softmax",
        action="store_true",
        help="Whether to enable Habana Flash Attention in fast softmax mode.",
    )
    parser.add_argument(
        "--book_source",
        action="store_true",
        help="Whether to use project Guttenberg books data as input. Usefull for testing large sequence lenghts.",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Whether to use torch compiled model or not.",
    )
    parser.add_argument(
        "--ignore_eos",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to disable stopping with eos token when calling `generate`. --no-ignore_eos to disable it",
    )
    parser.add_argument("--temperature", default=1.0, type=float, help="Temperature value for text generation")
    parser.add_argument("--top_p", default=1.0, type=float, help="Top_p value for generating text via sampling")
    parser.add_argument(
        "--const_serialization_path",
        "--csp",
        type=str,
        help="Path to serialize const params. Const params will be held on disk memory instead of being allocated on host memory.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust the execution of code from datasets/models defined on the Hub. This option should only be set to `True` for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine.",
    )
    parser.add_argument(
        "--parallel_strategy",
        type=str,
        choices=["tp", "none"],  # Add other strategies as needed
        default="none",
        help="Run multi card with the specified parallel strategy. Choices are 'tp' for Tensor Parallel Strategy or 'none'.",
    )
    parser.add_argument(
        "--input_embeds",
        action="store_true",
        help="Whether to enable inputs_embeds or not.",
    )
    parser.add_argument(
        "--run_partial_dataset",
        action="store_true",
        help="Run the inference with dataset for specified --n_iterations(default:5)",
    )

    quant_parser_group = parser.add_mutually_exclusive_group()
    quant_parser_group.add_argument(
        "--load_quantized_model_with_autogptq",
        action="store_true",
        help="Load an AutoGPTQ quantized checkpoint using AutoGPTQ.",
    )
    quant_parser_group.add_argument(
        "--disk_offload",
        action="store_true",
        help="Whether to enable device map auto. In case no space left on cpu, weights will be offloaded to disk.",
    )
    quant_parser_group.add_argument(
        "--load_quantized_model_with_inc",
        action="store_true",
        help="Load a Huggingface quantized checkpoint using INC.",
    )
    quant_parser_group.add_argument(
        "--local_quantized_inc_model_path",
        type=str,
        default=None,
        help="Path to neural-compressor quantized model, if set, the checkpoint will be loaded.",
    )

    args = parser.parse_args()

    if args.torch_compile:
        args.use_hpu_graphs = False

    if not args.use_hpu_graphs:
        args.limit_hpu_graphs = False

    if args.use_flash_attention and not args.flash_attention_fast_softmax:
        args.flash_attention_fast_softmax = True

    args.quant_config = os.getenv("QUANT_CONFIG", "")
    if args.quant_config and args.load_quantized_model_with_autogptq:
        raise RuntimeError("Setting both quant_config and load_quantized_model_with_autogptq is unsupported. ")

    if args.quant_config == "" and args.disk_offload:
        logger.warning(
            "`--disk_offload` was tested only with fp8, it may not work with full precision. If error raises try to remove the --disk_offload flag."
        )
    return args


def prepare_generation_embedding(model, model_name, input_tokens):
    batch_size = input_tokens["input_ids"].size(0)

    inputs_embeds = model.get_input_embeddings()(input_tokens["input_ids"])

    if inputs_embeds.size(0) != batch_size:
        inputs_embeds = inputs_embeds.expand(batch_size, -1, -1)

    attention_mask = input_tokens["attention_mask"]
    return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}


def main():
    parser = argparse.ArgumentParser()
    args = setup_parser(parser)
    model, assistant_model, tokenizer, generation_config = initialize_model(args, logger)

    use_lazy_mode = True
    if args.torch_compile:
        use_lazy_mode = False

    import habana_frameworks.torch.hpu as torch_hpu

    if args.dataset_name is None:
        # Benchmark over the prompts below
        if args.prompt:
            input_sentences = args.prompt
        elif args.book_source:

            def download_book(book_id):
                import os

                import requests

                url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
                response = requests.get(url)
                if response.status_code == 200:
                    pid = os.getpid()
                    save_path = f"/tmp/{book_id}_{pid}.txt"
                    with open(save_path, "wb") as file:
                        file.write(response.content)
                    print(f"Book downloaded and saved to: {save_path}")
                    return save_path
                else:
                    print("Failed to download book! Exiting...")
                    import sys

                    sys.exit()

            def assemble_prompt(prompt_size, book_path):
                prompt = ""
                counter = 0
                book_lines = open(book_path).readlines()
                for line in book_lines:
                    for word in line.split():
                        counter += 1
                        prompt += word + " "
                        if counter == prompt_size:
                            return [prompt] * args.batch_size

            book_ids = [
                2701,  # Moby Dick; Or, The Whale
                1513,  # Romeo and Juliet
                1342,  # Pride and Prejudice
            ]
            input_sentences = assemble_prompt(prompt_size=args.max_input_tokens, book_path=download_book(book_ids[0]))
        else:
            input_sentences = [
                "DeepSpeed is a machine learning framework",
                "He is working on",
                "He has a",
                "He got all",
                "Everyone is happy and I can",
                "The new movie that got Oscar this year",
                "In the far far distance from our galaxy,",
                "Peace is the only way",
            ]

        if args.batch_size > len(input_sentences):
            # Dynamically extends to support larger batch sizes
            num_sentences_to_add = args.batch_size - len(input_sentences)
            for i in range(num_sentences_to_add):
                input_sentences.append(input_sentences[i % len(input_sentences)])
        elif args.batch_size < len(input_sentences):
            input_sentences = input_sentences[: args.batch_size]

        def generate(size=None, reduce_recompile=False):
            """Generates sequences from the input sentences and returns them."""
            encode_t0 = time.perf_counter()
            # Tokenization
            if args.max_input_tokens > 0:
                input_tokens = tokenizer.batch_encode_plus(
                    input_sentences,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=args.max_input_tokens,
                    truncation=True,
                )

                def compute_valid_sequence_lengths_tensor(input_tokens):
                    attn_mask = input_tokens["attention_mask"]
                    return torch.sum(attn_mask, dim=1)

                valid_sequence_lengths = compute_valid_sequence_lengths_tensor(input_tokens).to(args.device)
                generation_config.valid_sequence_lengths = valid_sequence_lengths
            else:
                input_tokens = tokenizer.batch_encode_plus(input_sentences, return_tensors="pt", padding=True)
            encode_duration = time.perf_counter() - encode_t0

            if size is not None:
                input_tokens = adjust_batch(input_tokens, size)
            if not reduce_recompile:
                # Move inputs to target device(s)
                for t in input_tokens:
                    if torch.is_tensor(input_tokens[t]):
                        input_tokens[t] = input_tokens[t].to(args.device)

            input_data = {}
            if args.input_embeds:
                inputs_embeds = prepare_generation_embedding(model, args.model_name_or_path, input_tokens)
                if inputs_embeds is not None:
                    input_data.update(inputs_embeds)
                    input_data.update(input_tokens)
                else:
                    args.input_embeds = False
                    input_data.update(input_tokens)
            else:
                input_data.update(input_tokens)

            iteration_times = []
            input_tokens = [785, 2701, 525, 5248, 5754, 4755, 320, 4197, 11253, 8, 911, 29803, 13, 21149, 3019, 553, 3019, 323, 1221, 6248, 697, 4226, 448, 330, 1782, 4226, 374, 320, 55, 9940, 1380, 1599, 374, 279, 4396, 6524, 5754, 624, 14582, 510, 53544, 279, 1372, 315, 5128, 304, 279, 468, 6480, 19745, 315, 264, 6291, 315, 220, 16, 18, 34, 31801, 78474, 17855, 320, 16, 18, 2149, 18, 6667, 701, 25538, 279, 5128, 653, 537, 27248, 624, 3798, 510, 32, 13, 220, 16, 15, 198, 33, 13, 220, 23, 198, 34, 13, 220, 19, 198, 35, 13, 220, 17, 15, 198, 36, 13, 220, 16, 17, 198, 37, 13, 220, 18, 198, 38, 13, 220, 16, 21, 198, 39, 13, 220, 20, 198, 40, 13, 220, 17, 19, 198, 41, 13, 220, 21, 198, 16141, 25, 6771, 594, 1744, 3019, 553, 3019, 13, 576, 16715, 1685, 38000, 56981, 19745, 686, 387, 6718, 553, 1378, 7586, 315, 21880, 13, 576, 1156, 374, 279, 17071, 62057, 16230, 448, 279, 220, 16, 18, 34, 320, 77, 9637, 12616, 400, 40, 284, 715, 59, 37018, 90, 16, 15170, 17, 31716, 8, 892, 686, 6718, 279, 19745, 1119, 220, 17, 5128, 13, 1096, 686, 387, 4623, 6718, 1119, 220, 19, 5128, 553, 279, 16230, 448, 2326, 13578, 220, 16, 39, 96092, 13, 576, 2790, 1372, 315, 5128, 374, 8916, 400, 17, 1124, 50853, 220, 19, 284, 220, 23, 12947, 576, 4226, 374, 320, 33, 3593, 14582, 510, 23085, 315, 279, 2701, 11469, 279, 6275, 67, 18245, 315, 1874, 12, 16, 19, 5424, 304, 1973, 315, 28387, 19753, 11, 504, 15457, 311, 8426, 5267, 3798, 510, 32, 13, 4229, 39, 19, 366, 13059, 39, 19, 366, 97354, 39, 19, 366, 11832, 39, 19, 366, 6826, 19, 198, 33, 13, 11832, 39, 19, 366, 4229, 39, 19, 366, 13059, 39, 19, 366, 97354, 39, 19, 366, 6826, 19, 198, 34, 13, 97354, 39, 19, 366, 6826, 19, 366, 13059, 39, 19, 366, 4229, 39, 19, 366, 11832, 39, 19, 198, 35, 13, 97354, 39, 19, 366, 13059, 39, 19, 366, 6826, 19, 366, 4229, 39, 19, 366, 11832, 39, 19, 198, 36, 13, 13059, 39, 19, 366, 4229, 39, 19, 366, 11832, 39, 19, 366, 97354, 39, 19, 366, 6826, 19, 198, 37, 13, 6826, 19, 366, 4229, 39, 19, 366, 13059, 39, 19, 366, 97354, 39, 19, 366, 11832, 39, 19, 198, 38, 13, 11832, 39, 19, 366, 13059, 39, 19, 366, 97354, 39, 19, 366, 4229, 39, 19, 366, 6826, 19, 198, 39, 13, 6826, 19, 366, 11832, 39, 19, 366, 4229, 39, 19, 366, 13059, 39, 19, 366, 97354, 39, 19, 198, 40, 13, 6826, 19, 366, 97354, 39, 19, 366, 4229, 39, 19, 366, 13059, 39, 19, 366, 11832, 39, 19, 198, 41, 13, 97354, 39, 19, 366, 13059, 39, 19, 366, 4229, 39, 19, 366, 11832, 39, 19, 366, 6826, 19, 198, 16141, 25, 6771, 594, 1744, 3019, 553, 3019, 13, 576, 28387, 19753, 315, 1874, 12, 16, 19, 6275, 67, 18245, 42054, 438, 582, 3271, 504, 279, 1909, 315, 1874, 220, 16, 19, 311, 279, 5622, 13, 576, 1973, 315, 5424, 304, 279, 1874, 504, 1909, 311, 5622, 374, 356, 11, 11832, 11, 4229, 11, 13059, 11, 97354, 13, 15277, 304, 1973, 315, 7703, 28387, 19753, 582, 614, 97354, 39, 19, 11, 13059, 39, 19, 11, 4229, 39, 19, 11, 11832, 39, 19, 11, 323, 6826, 19, 11, 476, 4226, 320, 41, 568, 576, 4226, 374, 320, 41, 3593, 14582, 510, 23085, 315, 279, 2701, 374, 6509, 458, 13621, 458, 8503, 67, 1399, 5267, 3798, 510, 32, 13, 472, 17, 13880, 18, 198, 33, 13, 12812, 5066, 198, 34, 13, 6826, 19, 198, 35, 13, 472, 8996, 18, 198, 36, 13, 5627, 17, 198, 37, 13, 1674, 77927, 18, 8, 18, 198, 38, 13, 14413, 8281, 18, 198, 39, 13, 472, 17, 46, 198, 40, 13, 472, 5066, 198, 41, 13, 451, 10360, 198, 16141, 25, 6771, 594, 1744, 3019, 553, 3019, 13, 1527, 13621, 458, 8503, 67, 1399, 374, 264, 23628, 429, 374, 14257, 553, 17592, 3015, 504, 458, 13621, 13, 576, 11483, 14806, 369, 3015, 374, 472, 17, 46, 11, 892, 3363, 429, 582, 1184, 311, 8253, 892, 315, 1493, 2606, 11, 979, 10856, 448, 472, 17, 46, 11, 7586, 458, 13621, 13, 5627, 17, 11, 476, 328, 14308, 324, 39489, 11, 979, 10856, 448, 472, 17, 46, 11, 3643, 472, 17, 13880, 19, 11, 476, 71491, 292, 13621, 13, 576, 4226, 374, 320, 36, 3593, 14582, 510, 32, 501, 23628, 374, 91006, 323, 1730, 311, 387, 264, 1615, 453, 4640, 292, 13621, 448, 264, 296, 7417, 3072, 315, 220, 17, 19, 23, 342, 38871, 13, 3197, 220, 15, 13, 15, 15, 20, 15, 21609, 315, 419, 13621, 525, 55667, 304, 220, 15, 13, 20, 15, 15, 444, 315, 3015, 11, 279, 36043, 374, 16878, 438, 220, 18, 13, 23, 24, 13, 3555, 374, 279, 281, 82968, 315, 419, 13621, 5267, 3798, 510, 32, 13, 220, 20, 13, 22, 23, 198, 33, 13, 220, 19, 13, 22, 23, 198, 34, 13, 220, 19, 13, 20, 21, 198, 35, 13, 220, 21, 13, 23, 24, 198, 36, 13, 220, 22, 13, 22, 23, 198, 37, 13, 220, 18, 13, 23, 24, 198, 38, 13, 220, 16, 13, 17, 18, 198, 39, 13, 220, 17, 13, 23, 24, 198, 40, 13, 220, 17, 13, 18, 18, 198, 41, 13, 220, 20, 13, 18, 18, 198, 16141, 25, 6771, 594, 1744, 3019, 553, 3019, 13, 79540, 429, 400, 58, 32, 60, 284, 508, 39, 47822, 10, 25439, 12947, 5692, 11, 419, 374, 6144, 311, 26107, 16, 15, 87210, 18, 13, 23, 24, 92, 12947, 5005, 582, 614, 400, 42, 15159, 64, 92, 284, 24437, 59, 37018, 90, 58, 39, 47822, 10, 92, 1457, 32, 87210, 92, 13989, 90, 58, 17020, 13989, 284, 715, 59, 37018, 90, 16, 15, 87210, 18, 13, 23, 24, 92, 1124, 50853, 220, 16, 15, 87210, 18, 13, 23, 24, 3417, 90, 16, 15, 87210, 17, 3417, 13, 576, 12942, 27690, 374, 400, 12, 18, 13, 23, 24, 488, 10293, 18, 13, 23, 24, 8, 481, 10293, 17, 8, 284, 220, 20, 13, 22, 23, 54876, 8916, 400, 42, 4306, 284, 220, 16, 15, 87210, 20, 13, 22, 23, 92, 12947, 576, 400, 79, 42, 4306, 3, 374, 279, 8225, 1487, 315, 400, 42, 4306, 54876, 892, 374, 6144, 311, 400, 20, 13, 22, 23, 12947, 576, 4226, 374, 320, 32, 3593, 14582, 510, 32, 6291, 5610, 220, 17, 13, 15, 15, 34651, 315, 1613, 5298, 13621, 11, 6826, 18, 8281, 46761, 11, 323, 220, 16, 13, 15, 15, 34651, 315, 34619, 64702, 349, 11, 14413, 82934, 18, 8281, 46, 8, 17, 13, 576, 6291, 374, 2952, 311, 22106, 279, 5256, 315, 264, 2613, 3311, 315, 3746, 13621, 476, 3746, 2331, 448, 1172, 8922, 4344, 304, 279, 36043, 315, 279, 6291, 13, 80808, 32676, 315, 3746, 13621, 476, 3746, 2331, 646, 5240, 264, 5089, 2297, 304, 36043, 13, 2585, 1657, 4544, 642, 315, 24691, 2216, 13621, 11, 472, 8996, 18, 11, 1231, 387, 3694, 1573, 279, 36043, 12033, 311, 2297, 11941, 5267, 3798, 510, 32, 13, 220, 15, 13, 17, 20, 15, 34651, 198, 33, 13, 220, 15, 13, 20, 15, 15, 34651, 198, 34, 13, 220, 18, 13, 15, 15, 34651, 198, 35, 13, 220, 16, 13, 15, 15, 34651, 198, 36, 13, 220, 18, 13, 20, 15, 34651, 198, 37, 13, 220, 16, 13, 20, 15, 34651, 198, 38, 13, 220, 17, 13, 20, 15, 34651, 198, 39, 13, 220, 19, 13, 15, 15, 34651, 198, 40, 13, 220, 15, 13, 22, 20, 15, 34651, 198, 41, 13, 220, 17, 13, 15, 15, 34651, 198, 16141, 25, 6771, 594, 1744, 3019, 553, 3019, 13, 1205, 1035, 1075, 311, 12564, 279, 4147, 8654, 315, 419, 6291, 13, 5512, 582, 3270, 279, 23606, 369, 279, 27672, 2022, 315, 279, 7469, 13621, 11, 304, 419, 1142, 315, 1613, 5298, 13621, 13, 400, 2149, 15159, 18, 92, 8281, 46761, 320, 36306, 8, 488, 472, 15159, 17, 92, 46, 715, 491, 6044, 472, 15159, 18, 92, 46, 47822, 10, 92, 488, 6826, 18, 8281, 46, 87210, 92, 12947, 576, 63280, 349, 2331, 374, 8916, 279, 64702, 349, 27672, 13, 576, 3694, 3746, 13621, 11, 49516, 2216, 13621, 11, 686, 13767, 448, 279, 63280, 349, 2331, 13, 15277, 279, 7192, 3311, 315, 13621, 429, 646, 387, 3694, 686, 387, 6144, 311, 279, 3311, 315, 64702, 349, 27672, 11, 476, 220, 17, 4544, 642, 13, 576, 4226, 374, 320, 41, 3593, 14582, 510, 4340, 1657, 2544, 3664, 388, 315, 220, 15, 13, 17, 20, 15, 386, 37512, 39, 1558, 432, 1896, 311, 20628, 551, 6587, 220, 20, 15, 13, 15, 64070, 315, 220, 15, 13, 16, 20, 15, 386, 472, 18, 2045, 19, 5267, 3798, 510, 32, 13, 220, 22, 20, 13, 15, 64070, 198, 33, 13, 220, 24, 15, 13, 15, 64070, 198, 34, 13, 220, 21, 15, 13, 15, 64070, 198, 35, 13, 220, 16, 17, 15, 64070, 198, 36, 13, 220, 18, 15, 13, 15, 64070, 198, 37, 13, 220, 16, 23, 15, 64070, 198, 38, 13, 220, 17, 22, 15, 64070, 198, 39, 13, 220, 16, 15, 15, 64070, 198, 40, 13, 220, 17, 22, 64070, 198, 41, 13, 220, 16, 20, 15, 64070, 198, 16141, 25, 6771, 594, 1744, 3019, 553, 3019, 13]

            tokens = []
            tokens.append(input_tokens)
            seq = torch.LongTensor(tokens).to("hpu")

            outputs = model.generate(
                seq,
                generation_config=generation_config,
                assistant_model=assistant_model,
                lazy_mode=use_lazy_mode,
                hpu_graphs=args.use_hpu_graphs,
                profiling_steps=args.profiling_steps,
                profiling_warmup_steps=args.profiling_warmup_steps,
                ignore_eos=args.ignore_eos,
                iteration_times=iteration_times,
                profiling_record_shapes=args.profiling_record_shapes,
            ).cpu()
            first_token_time = iteration_times[0] + encode_duration
            logger.info(f"Time to first token = {first_token_time*1000}ms")
            return tokenizer.batch_decode(outputs, skip_special_tokens=True)

        from optimum.habana.utils import HabanaProfile

        # compilation stage disable profiling
        HabanaProfile.disable()
        # Compilation
        logger.info("Graph compilation...")
        dyn_prompt_lens = args.simulate_dyn_prompt
        t0 = time.perf_counter()
        # The first three iterations take longer because of graph compilation
        if dyn_prompt_lens is None or len(set(dyn_prompt_lens)) == 1:
            for i in range(args.warmup):
                if dyn_prompt_lens is None:
                    print(f"Warming up iteration {i+1}/{args.warmup}", flush=True)
                    generate(None, args.reduce_recompile)
                else:
                    print(f"Warming up for shape {dyn_prompt_lens[0]} iteration {i+1}/{args.warmup}", flush=True)
                    generate(dyn_prompt_lens[0], args.reduce_recompile)
        else:
            if args.bucket_size > 0:
                mn = min(dyn_prompt_lens)
                mx = max(dyn_prompt_lens)

                def rounder(x):
                    return int(math.ceil(x / args.bucket_size) * args.bucket_size)

                min_prompt_len = rounder(mn)
                max_sentence_len = rounder(mx)
                for i in range(args.warmup):
                    lst = list(range(min_prompt_len, max_sentence_len + 1, args.bucket_size))
                    for sz in lst:
                        print(f"Warming up for shape {sz - 1} iteration {i+1}/{args.warmup}", flush=True)
                        generate(sz - 1, args.reduce_recompile)
        torch_hpu.synchronize()
        compilation_duration = time.perf_counter() - t0
        HabanaProfile.enable()
        total_new_tokens_generated = 0
        logger.info("Running generate...")
        t0 = time.perf_counter()
        # Benchmark over n_iterations iterations
        if dyn_prompt_lens is None:
            for i in range(args.n_iterations):
                generated = generate(None, args.reduce_recompile)
        else:
            repeated_prompt_len = cycle(dyn_prompt_lens)
            for i in range(args.n_iterations):
                prompt_len = next(repeated_prompt_len)
                print("Generating for shape,", prompt_len)
                generated = generate(prompt_len, args.reduce_recompile)
        duration = time.perf_counter() - t0
        total_new_tokens_generated = args.n_iterations * args.batch_size * args.max_new_tokens
        throughput = total_new_tokens_generated / duration

        print()
        print("Input/outputs:")
        for i, input_sentence in enumerate(zip(input_sentences)):
            print(f"input {i+1}: {input_sentence}")
            for j, output in enumerate(
                zip(generated[args.num_return_sequences * i : args.num_return_sequences * (i + 1)])
            ):
                print(f"output {j+1}: {output}")
            print()

        # Store results if necessary
        if args.output_dir is not None and args.global_rank == 0:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            results = {
                "throughput": throughput,
                "output": output,
            }
            with (output_dir / "results.json").open("w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

        stats = "Input embeds" if args.input_embeds else "Input tokens"
        stats = stats + f"\nThroughput (including tokenization) = {throughput} tokens/second"
        if args.show_graphs_count:
            stats = stats + f"\nNumber of HPU graphs                = {count_hpu_graphs()}"
        separator = "-" * len(stats)
        print()
        print("Stats:")
        print(separator)
        print(stats)
        mem = get_hpu_memory_stats()
        for k, v in mem.items():
            print("{:35} = {} GB".format(k[:-5].replace("_", " ").capitalize(), v))
        print(f"Graph compilation duration          = {compilation_duration} seconds")
        print(separator)
        print()
    else:
        # Downloading and loading a dataset from the hub.
        from datasets import load_dataset
        from torch.utils.data import DataLoader

        assert not args.simulate_dyn_prompt, "Both dataset_name and simulate_dyn_prompt are set"

        raw_dataset = load_dataset(args.dataset_name)
        if "test" in raw_dataset:
            split = "test"
        elif "validation" in raw_dataset:
            split = "validation"
        else:
            split = "train"
        raw_dataset = (
            raw_dataset[split]
            .shuffle()
            .select(range(args.dataset_max_samples if args.dataset_max_samples > 0 else (raw_dataset[split]).num_rows))
        )

        if args.column_name is None:
            # If no column name is given, take the first column that has strings
            column_name = [key for key in raw_dataset.features.keys() if raw_dataset.features[key].dtype == "string"][
                0
            ]
            logger.info(
                f"No column name was given so automatically choosing '{column_name}' for prompts. If you would like to use another column of the dataset, you can set the argument `--column_name`."
            )
        else:
            column_name = args.column_name

        # Remove unused columns
        raw_dataset = raw_dataset.remove_columns([name for name in raw_dataset.column_names if name != column_name])

        # Set the prompt length to args.max_input_tokens if > 0 else (if 0 truncate to 16, otherwise use full length)
        prompt_length = args.max_input_tokens if args.max_input_tokens > 0 else (-1, 16)[args.max_input_tokens == 0]

        def preprocess_function(examples):
            # Tokenize the texts
            return tokenizer(
                examples[column_name],
                padding="max_length",
                max_length=prompt_length if prompt_length > 0 else None,
                truncation=prompt_length > 0,
            )

        raw_dataset = raw_dataset.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on dataset",
        )
        # After tokenization, we can remove the column of interest
        raw_dataset = raw_dataset.remove_columns([column_name])
        raw_dataset.set_format(type="torch")

        if prompt_length <= 0:
            # Todo please check if this collate function is suitable for your model
            # This has been tested for OPT, llama, and Bloom
            assert model.config.model_type in ["opt", "bloom", "llama"]

            def collate_fn(data):
                collect = {k: [dt[k] for dt in data] for k in data[0]}
                result = {}
                for k in collect:
                    tensors = collect[k]
                    max_shape = max([item.shape[0] for item in tensors])
                    result[k] = torch.stack(
                        [torch.cat((torch.zeros(max_shape - t.shape[0], dtype=t.dtype), t)) for t in tensors], 0
                    )
                return result

        else:
            collate_fn = None

        dataloader = DataLoader(raw_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

        def generate_dataset(batch):
            prompt = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            # Move inputs to target device(s)
            for t in batch:
                if torch.is_tensor(batch[t]):
                    batch[t] = batch[t].to(args.device)
            # Generate new sequences
            outputs = model.generate(
                **batch,
                generation_config=generation_config,
                lazy_mode=use_lazy_mode,
                hpu_graphs=args.use_hpu_graphs,
                profiling_steps=args.profiling_steps,
                profiling_warmup_steps=args.profiling_warmup_steps,
                ignore_eos=args.ignore_eos,
                profiling_record_shapes=args.profiling_record_shapes,
            ).cpu()
            return prompt, outputs

        # warmup
        if prompt_length > 0:
            from optimum.habana.utils import HabanaProfile

            # compilation stage disable profiling
            HabanaProfile.disable()
            # Compilation
            logger.info("Graph compilation...")
            t0 = time.perf_counter()
            for i, batch in enumerate(dataloader):
                generate_dataset(batch)
                # The first three iterations take longer because of graph compilation
                if (i + 1) == 3:
                    break
            torch_hpu.synchronize()
            compilation_duration = time.perf_counter() - t0
            HabanaProfile.enable()

        total_new_tokens_generated = 0
        duration = 0
        separator = "-" * 50
        logger.info("Running generate dataset...")
        t_start = time.time()
        for i, batch in enumerate(dataloader):
            t0 = time.perf_counter()
            prompt, outputs = generate_dataset(batch)
            duration += time.perf_counter() - t0
            total_new_tokens_generated += args.batch_size * args.max_new_tokens
            print(separator)
            print(f"Batch n°{i+1}")
            print(f"Input: {prompt[:args.batch_size]}")
            print(
                f"Output: {tokenizer.batch_decode(outputs, skip_special_tokens=True)[:args.batch_size*args.num_return_sequences]}"
            )
            print(separator)
            if args.run_partial_dataset and args.n_iterations == i + 1:
                break
        t_end = time.time()

        throughput = total_new_tokens_generated / duration
        # Print Stats

        stats = f"Throughput (including tokenization) = {throughput} tokens/second"
        separator = "-" * len(stats)
        print()
        print("Stats:")
        print(separator)
        print(stats)
        print("Total runtime for dataset:", t_end - t_start)
        mem = get_hpu_memory_stats()
        for k, v in mem.items():
            print("{:35} = {} GB".format(k[:-5].replace("_", " ").capitalize(), v))
        if prompt_length > 0:
            print(f"Graph compilation duration          = {compilation_duration} seconds")
        print(separator)
    if args.quant_config:
        finalize_quantization(model)
    if args.const_serialization_path and os.path.isdir(args.const_serialization_path):
        import shutil

        shutil.rmtree(args.const_serialization_path)


if __name__ == "__main__":
    main()
