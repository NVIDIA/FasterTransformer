# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import argparse
import configparser
import json
import numpy as np
import os
import sys
import torch
import torch.distributed as dist
from datetime import datetime
from datasets import load_dataset, load_metric
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.pytorch.gpt.utils.parallel_gpt import ParallelGPT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ft_model_location', type=str,
                        default='/data2/byshiue/models/huggingface-gpt/gpt2-xl/c-models')
    parser.add_argument('--hf_model_location', type=str,
                        default='/data2/byshiue/models/huggingface-gpt/gpt2-xl/gpt2-xl')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--test_hf', action='store_true')
    parser.add_argument('--data_type', type=str, choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument("--cache_path", type=str, default="/workdir/datasets/ccdv/")
    parser.add_argument("--max_ite", type=int, default=20)
    parser.add_argument("--ft_use_hf_config", action="store_true",
                        help="use the hyper-parameters from the hf model")
    parser.add_argument('--lib_path', type=str, default='./lib/libth_parallel_gpt.so',
                        help='path to the pyt_fastertransformer dynamic lib file.')
    parser.add_argument('--tensor_para_size', type=int, default=1,
                        help='tensor parallel size')
    parser.add_argument('--pipeline_para_size', type=int, default=1,
                        help='pipeline parallel size')

    args = parser.parse_args()

    try:
        dist.init_process_group(backend='mpi')
    except:
        print("[INFO] WARNING: Have initalize the process group")
    rank = dist.get_rank()

    summarize = args.summarize
    test_hf = args.test_hf
    ft_model_location = args.ft_model_location
    hf_model_location = args.hf_model_location

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    dataset_cnn = load_dataset("ccdv/cnn_dailymail", '3.0.0', cache_dir=args.cache_path)

    hf_config = json.load(open(os.path.join(hf_model_location, 'config.json'), 'r'))
    ft_config = None

    head_num = hf_config['n_head']
    layer_num = hf_config['n_layer']
    start_id = hf_config['bos_token_id']
    end_id = hf_config['eos_token_id']
    size_per_head = hf_config['n_embd'] // head_num

    if not args.ft_use_hf_config:
        ft_config = configparser.ConfigParser()
        ft_config.read(os.path.join(ft_model_location, '1-gpu/config.ini'))

        head_num = ft_config.getint('gpt', 'num_attention_heads')
        layer_num = ft_config.getint('gpt', 'num_layers')
        start_id = 50256  # TODO: get this from the tokenizer
        end_id = 50256  # TODO: get this from the tokenizer
        size_per_head = ft_config.getint('gpt', 'hidden_size') // head_num

    if summarize:
        top_k = 2
        output_len = 100
    else:
        top_k = 1
        output_len = 256
    top_p = 0.0
    random_seed = 5
    temperature = 1
    max_seq_len = hf_config['n_ctx'] if args.ft_use_hf_config else ft_config.getint('gpt', 'max_position_embeddings')
    max_batch_size = 5
    repetition_penalty = 1
    vocab_size = 50257
    tensor_para_size = args.tensor_para_size
    pipeline_para_size = args.pipeline_para_size
    lib_path = args.lib_path
    ckpt_path = os.path.join(ft_model_location, f'{tensor_para_size}-gpu')

    print(f"top_k: {top_k}")
    print(f"top_p: {top_p}")
    print(f"random_seed: {random_seed}")
    print(f"temperature: {temperature}")
    print(f"max_seq_len: {max_seq_len}")
    print(f"max_batch_size: {max_batch_size}")
    print(f"repetition_penalty: {repetition_penalty}")
    print(f"vocab_size: {vocab_size}")
    print(f"tensor_para_size: {tensor_para_size}")
    print(f"pipeline_para_size: {pipeline_para_size}")
    print(f"lib_path: {lib_path}")
    print(f"ckpt_path: {ckpt_path}")
    print(f"hf_config: {hf_config}")

    gpt = ParallelGPT(head_num, size_per_head, vocab_size, start_id, end_id, layer_num,
                      max_seq_len, tensor_para_size, pipeline_para_size, lib_path=lib_path, int8_mode=0)

    if not gpt.load(ckpt_path=ckpt_path):
        print("[WARNING] Checkpoint file not found. Model loading is skipped.")

    if (test_hf and summarize) or not summarize:
        model = GPT2LMHeadModel.from_pretrained(hf_model_location)
        # device_hf = 'cuda:1'
        # model.to(device_hf)
        model.cuda()
        if args.data_type == 'fp16':
            model.half()
        elif args.data_type == 'bf16':
            model.bfloat16()

    if args.data_type == 'fp16':
        gpt.half()
    elif args.data_type == 'bf16':
        gpt.bfloat16()

    def summarize_ft(datapoint):
        if summarize:
            line = datapoint['article'] + ' TL;DR: '
        else:
            line = datapoint['article']
        line = line.strip()
        line = line.replace(" n't", "n't")

        line_encoded = tokenizer.encode(line, return_tensors='pt')
        if summarize:
            line_encoded = line_encoded[:, -923:]
        else:
            line_encoded = line_encoded[:, -768:]
        line_encoded = line_encoded.type(torch.int32)

        with torch.no_grad():
            output, ft_output_len = gpt(line_encoded, torch.IntTensor([len(line_encoded[0])]),
                                        output_len,
                                        1,
                                        top_k,
                                        top_p,
                                        0.0,
                                        temperature,
                                        1.0,
                                        repetition_penalty,
                                        random_seed,
                                        True)

        tokens = output[0][0][len(line_encoded[0]):ft_output_len[0]].cpu().numpy()

        output_lines = tokenizer.decode(output[0][0][len(line_encoded[0]):ft_output_len[0]])
        output_lines = ".".join(output_lines.split('.')[:4]) + "."
        return output_lines, tokens

    def summarize_hf(datapoint):
        if summarize:
            line = datapoint['article'] + ' TL;DR: '
        else:
            line = datapoint['article']
        line = line.strip()
        line = line.replace(" n't", "n't")

        line_encoded = tokenizer.encode(line, return_tensors='pt')
        if summarize:
            line_encoded = line_encoded[:, -923:]
        else:
            line_encoded = line_encoded[:, -768:]
        # line_encoded = line_encoded.to(device_hf)
        line_encoded = line_encoded.cuda()

        with torch.no_grad():
            output = model.generate(line_encoded,
                                    max_length=len(line_encoded[0]) + output_len,
                                    k=top_k,
                                    temprature=temperature,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id)

        tokens = output[0][len(line_encoded[0]):].cpu().numpy()
        output_lines = tokenizer.decode(output[0][len(line_encoded[0]):])
        output_lines = ".".join(output_lines.split('.')[:4]) + "."
        return output_lines, tokens

    if summarize:
        datapoint = dataset_cnn['test'][0]
        summary, _ = summarize_ft(datapoint)
        print('---------------------------------------------------------')
        print('FT Generated : ')
        print(' Article : ', datapoint['article'])
        print('\n Highlights : ', datapoint['highlights'])
        print('\n Summary : ', summary)
        print('---------------------------------------------------------')

        if test_hf:
            summary, _ = summarize_hf(datapoint)
            print('---------------------------------------------------------')
            print('HF Generated : ')
            print(' Article : ', datapoint['article'])
            print('\n Highlights : ', datapoint['highlights'])
            print('\n Summary : ', summary)
            print('---------------------------------------------------------')

    if summarize:
        metric_ft = load_metric("rouge")
        metric_hf = load_metric("rouge")
    else:
        tokens = []

    ft_time = 0.0
    hf_time = 0.0
    for data_point_idx in tqdm(range(1, 11490, int(11490 / args.max_ite))):
        try:
            datapoint = dataset_cnn['test'][data_point_idx]

            start_time = datetime.now()
            summary_ft, tokens_ft = summarize_ft(datapoint)
            stop_time = datetime.now()
            ft_time += (stop_time - start_time).total_seconds()
            if (test_hf and summarize) or not summarize:
                start_time = datetime.now()
                summary_hf, tokens_hf = summarize_hf(datapoint)
                stop_time = datetime.now()
                hf_time += (stop_time - start_time).total_seconds()

            if rank == 0:
                if summarize:
                    metric_ft.add_batch(predictions=[summary_ft], references=[datapoint['highlights']])
                    if test_hf:
                        metric_hf.add_batch(predictions=[summary_hf], references=[datapoint['highlights']])
                else:
                    tokens.append((tokens_ft, tokens_hf))
        except:
            print('Error with datapoint : ', data_point_idx)

    def compute_exact_match(tokens, n_tokens=[1, 10, 25, 50, 100, 150, 200, 250]):
        em_metrics = []
        for t in n_tokens:
            errors = 0
            total = 0
            for ft_tokens, hf_tokens in tokens:
                if len(ft_tokens) > t and len(hf_tokens) > t:
                    total = total + 1
                    if not np.array_equal(ft_tokens[:t], hf_tokens[:t]):
                        errors = errors + 1

            if total > 0:
                print(f"{t}-token exact match acc: {100*(1-errors/total):.2f}")
                em_metrics.append(1 - errors / total)
            else:
                em_metrics.append(np.nan)

        return em_metrics

    if rank == 0:
        if summarize:
            computed_metrics_ft = metric_ft.compute()

            if test_hf:
                computed_metrics_hf = metric_hf.compute()

                print(f'Hugging Face (total latency: {hf_time} sec)')
                for key in computed_metrics_hf.keys():
                    print(f'{key} : {computed_metrics_hf[key].mid[2]*100}')

            print(f'Faster Transformers (total latency: {ft_time} sec)')
            for key in computed_metrics_ft.keys():
                print(f'{key} : {computed_metrics_ft[key].mid[2]*100}')
        else:
            em_metrics = compute_exact_match(tokens)


if __name__ == '__main__':
    main()
