# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

from __future__ import print_function

from torch.nn.utils.rnn import pad_sequence
import random
import os
import sys
import argparse
import configparser
import timeit
import torch
import torch.distributed as dist
import numpy as np
from transformers import AutoTokenizer
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.pytorch.gptneox.utils.gptneox import GptNeoX

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_len', type=int, default=32,
                        help='output sequence length to generate.')
    parser.add_argument('--beam_width', type=int, default=1,
                        help='beam width for beam search. Using sampling when beam width is 1.')
    parser.add_argument('--top_k', type=int, default=1,
                        help='top k candidate num')
    parser.add_argument('--top_p', type=float, default=0.,
                        help='top p probability threshold')
    parser.add_argument('--temperature', type=float, default=1.,
                        help='temperature')
    parser.add_argument('--len_penalty', type=float, default=0.,
                        help='len_penalty')
    parser.add_argument('--beam_search_diversity_rate', type=float, default=0.,
                        help='beam_search_diversity_rate')
    parser.add_argument('--tensor_para_size', type=int, default=1,
                        help='tensor parallel size')
    parser.add_argument('--pipeline_para_size', type=int, default=1,
                        help='pipeline parallel size')
    parser.add_argument('--ckpt_path', type=str, default='../models/gptneox/c-model/NeoX-1.3B/1-gpu',
                        help='path to the checkpoint file.')
    parser.add_argument('--tokenizer_path', type=str, default='../models/gptneox/model/NeoX-1.3B',
                        help='directory where the tokenizer file is located.')
    parser.add_argument('--lib_path', type=str, default='./lib/libth_transformer.so',
                        help='path to the pyt_fastertransformer dynamic lib file.')
    parser.add_argument('--sample_input_file', type=str,
                        help='path to the sample input file.')
    parser.add_argument('--max_batch_size', type=int, default=8,
                        help='max batch size.')
    parser.add_argument('--repetition_penalty', type=float, default=1.,
                        help='repetition penalty')
    parser.add_argument('--max_seq_len', type=int, default=1024,
                        help='max sequence length for position embedding table.')
    parser.add_argument('--inference_data_type', '--data_type', type=str, choices=['fp32', 'fp16'], default='fp16')
    parser.add_argument('--time', action='store_true',
                        help='whether or not to measure time elapsed.')
    parser.add_argument('--enable_random_seed', action='store_true',
                        help='is enable the random seed.')

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(os.path.join(args.ckpt_path, "config.ini"))
    head_num = int(config.get('gptneox', 'head_num'))
    size_per_head = int(config.get('gptneox', 'size_per_head'))
    vocab_size = int(config.get('gptneox', 'vocab_size'))
    layer_num = int(config.get('gptneox', 'num_layer'))
    rotary_embedding = int(config.get('gptneox', 'rotary_embedding'))
    start_id = int(config.get('gptneox', 'start_id'))
    end_id = int(config.get('gptneox', 'end_id'))
    use_gptj_residual = (config.get('gptneox', 'use_gptj_residual') == "1")
    weight_data_type = config.get('gptneox', 'weight_data_type')

    ckpt_path = args.ckpt_path
    tokenizer_path = args.tokenizer_path
    lib_path = args.lib_path
    output_len = args.output_len
    beam_width = args.beam_width
    top_k = args.top_k
    top_p = args.top_p
    temperature = args.temperature
    len_penalty = args.len_penalty
    beam_search_diversity_rate = args.beam_search_diversity_rate
    tensor_para_size = args.tensor_para_size
    pipeline_para_size = args.pipeline_para_size
    max_batch_size = args.max_batch_size
    max_seq_len = args.max_seq_len
    repetition_penalty = args.repetition_penalty
    inference_data_type = args.inference_data_type

    print("\n=============== Arguments ===============")
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))
    print("=========================================\n")

    if tensor_para_size * pipeline_para_size > 1:
        dist.init_process_group(backend=dist.Backend.MPI)
    rank = dist.get_rank() if dist.is_initialized() else 0
    device_count = dist.get_world_size() if dist.is_initialized() else 1
    device = rank % device_count
    torch.cuda.set_device(device)
    device = torch.cuda.current_device()

    # sentencepiece needed
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Inputs
    contexts = []
    if args.sample_input_file:  # conditional case
        with open(args.sample_input_file, "r") as f:
            contexts = f.read().splitlines()
            batch_size = min(len(contexts), max_batch_size)
        contexts = contexts[:batch_size]
        start_ids = [torch.tensor(tokenizer.encode(c), dtype=torch.int32, device=device) for c in contexts]
    else:  # unconditional case
        batch_size = max_batch_size
        contexts = ['<|endoftext|>'] * batch_size
        start_ids = [torch.IntTensor([end_id])] * batch_size

    print("[INFO] batch size: {}".format(batch_size))

    start_lengths = [len(ids) for ids in start_ids]

    start_ids = pad_sequence(start_ids, batch_first=True, padding_value=end_id)
    start_lengths = torch.IntTensor(start_lengths)

    if args.enable_random_seed == True:
        random_seed_tensor = torch.randint(0, 10000, size=[batch_size], dtype=torch.int64)
    else:
        random_seed_tensor = torch.zeros([batch_size], dtype=torch.int64)

    # Prepare model.
    gpt = GptNeoX(head_num, size_per_head, vocab_size, rotary_embedding,
                  start_id, end_id, layer_num, max_seq_len, 
                  tensor_para_size, pipeline_para_size, 
                  use_gptj_residual, lib_path, 
                  inference_data_type=inference_data_type, 
                  weights_data_type=weight_data_type)
    if not gpt.load(ckpt_path=ckpt_path):
        print("[WARNING] Checkpoint file not found. Model loading is skipped.")

    with torch.no_grad():
        tokens_batch = gpt(
            start_ids=start_ids,
            start_lengths=start_lengths,
            output_len=output_len,
            beam_width=beam_width,
            top_k=top_k * torch.ones(size=[batch_size], dtype=torch.int32),
            top_p=top_p * torch.ones(size=[batch_size], dtype=torch.float32),
            beam_search_diversity_rate=beam_search_diversity_rate * torch.ones(size=[batch_size], dtype=torch.float32),
            temperature=temperature * torch.ones(size=[batch_size], dtype=torch.float32),
            len_penalty=len_penalty * torch.ones(size=[batch_size], dtype=torch.float32),
            repetition_penalty=repetition_penalty * torch.ones(size=[batch_size], dtype=torch.float32),
            random_seed=random_seed_tensor,
            return_output_length=False,
            return_cum_log_probs=0)
        if tokens_batch is not None and rank == 0:
            tokens_batch = tokens_batch.cpu().numpy()
            for i, (context, tokens) in enumerate(zip(contexts, tokens_batch)):
                for beam_id in range(beam_width):
                    token = tokens[beam_id][start_lengths[i]:]  # exclude context input from the output
                    output = tokenizer.decode(token)
                    print(f'[INFO] batch {i}, beam {beam_id}:\n[Context]\n{context}\n\n[Output]\n{output}\n')

        # Measure inference time.
        if args.time:
            iterations = 10
            # warmup
            for i in range(iterations):
                tokens_batch = gpt(
                    start_ids=start_ids,
                    start_lengths=start_lengths,
                    output_len=output_len,
                    beam_width=beam_width,
                    top_k=top_k * torch.ones(size=[batch_size], dtype=torch.int32),
                    top_p=top_p * torch.ones(size=[batch_size], dtype=torch.float32),
                    beam_search_diversity_rate=beam_search_diversity_rate * torch.ones(size=[batch_size], dtype=torch.float32),
                    temperature=temperature * torch.ones(size=[batch_size], dtype=torch.float32),
                    len_penalty=len_penalty * torch.ones(size=[batch_size], dtype=torch.float32),
                    repetition_penalty=repetition_penalty * torch.ones(size=[batch_size], dtype=torch.float32),
                    random_seed=random_seed_tensor,
                    return_output_length=False,
                    return_cum_log_probs=0)

            batch_num = 0
            token_num = 0
            time = timeit.default_timer()
            for i in range(iterations):
                tokens_batch = gpt(
                    start_ids=start_ids,
                    start_lengths=start_lengths,
                    output_len=output_len,
                    beam_width=beam_width,
                    top_k=top_k * torch.ones(size=[batch_size], dtype=torch.int32),
                    top_p=top_p * torch.ones(size=[batch_size], dtype=torch.float32),
                    beam_search_diversity_rate=beam_search_diversity_rate * torch.ones(size=[batch_size], dtype=torch.float32),
                    temperature=temperature * torch.ones(size=[batch_size], dtype=torch.float32),
                    len_penalty=len_penalty * torch.ones(size=[batch_size], dtype=torch.float32),
                    repetition_penalty=repetition_penalty * torch.ones(size=[batch_size], dtype=torch.float32),
                    random_seed=random_seed_tensor,
                    return_output_length=False,
                    return_cum_log_probs=0)
                batch_num += 1
                for j, tokens in enumerate(tokens_batch):
                    token_num += tokens.shape[-1] - start_lengths[j]
            time_elapsed = timeit.default_timer() - time
            throughput = token_num / time_elapsed
            print(f"[INFO] FT-GPT generates {batch_num} batches, taking {time_elapsed:0.3f} secs "
                  f"to generate {token_num} tokens, {throughput:0.3f} tokens/sec.")


if __name__ == '__main__':
    main()
