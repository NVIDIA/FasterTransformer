# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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
import numpy as np
import os
import sys
import torch
import torch.distributed as dist
from datetime import datetime
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm

from utils import gpt_decoder

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.pytorch.gpt.utils.parallel_gpt import ParallelGPT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ft_model_location', type=str,
                        default='/models/GPT/HF/gpt2-xl/c-models')
    parser.add_argument('--hf_model_name', type=str,
                        default='facebook/opt-350m')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--test_hf', action='store_true')
    parser.add_argument('--data_type', type=str, choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument("--cache_path", type=str, default="/workdir/datasets/ccdv/")
    parser.add_argument("--max_ite", type=int, default=20)
    parser.add_argument('--lib_path', type=str, default='./lib/libth_transformer.so',
                        help='path to the pyt_fastertransformer dynamic lib file.')
    parser.add_argument('--tensor_para_size', type=int, default=1,
                        help='tensor parallel size')
    parser.add_argument('--pipeline_para_size', type=int, default=1,
                        help='pipeline parallel size')
    parser.add_argument(
        '--weights_data_type',
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help='Data type of FT checkpoint weights',
    )
    parser.add_argument(
        '--int8_mode', type=int, default=0, choices=[0, 1],
        help='The level of quantization to perform.'
             ' 0: No quantization. All computation in data_type'
             ' 1: Quantize weights to int8, all compute occurs in fp16/bf16. Not supported when data_type is fp32')
    parser.add_argument(
        '--use_gpt_decoder_ops', action='store_true',
        help='Use separate decoder FT operators instead of end-to-end model op.')
    parser.add_argument(
        '--use_fp32_to_compute_logit', action='store_true',
        help='Use FP32 data type for computing logit values when using gpt decoder ops. '
             'FT end-to-end GPT op always uses FP32 data type when computing logit.')
    parser.add_argument(
        '--rougeLsum_threshold', type=float, default=None,
        help='Threshold of FT rougeLsum score')
    parser.add_argument(
        '--verbose', action='store_true', help='Print all summary result.')

    args = parser.parse_args()
    np.random.seed(1) # rouge score use sampling to compute the score

    try:
        dist.init_process_group(backend='mpi')
    except:
        print("[INFO] WARNING: Have initialized the process group")
    rank = dist.get_rank()

    summarize = args.summarize
    test_hf = args.test_hf
    ft_model_location = args.ft_model_location
    hf_model_name = args.hf_model_name

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset_cnn = load_dataset("ccdv/cnn_dailymail", '3.0.0', cache_dir=args.cache_path)

    hf_config = vars(AutoConfig.from_pretrained(hf_model_name))

    head_num = hf_config['num_attention_heads']
    layer_num = hf_config['num_hidden_layers']
    start_id = hf_config['bos_token_id']
    end_id = hf_config['eos_token_id']
    size_per_head = hf_config['hidden_size'] // head_num

    # opt specific params: some are fixed
    layernorm_eps = 1e-5
    layernorm_type = 'pre_layernorm' if hf_config['do_layer_norm_before'] else 'post_layernorm'
    activation_type = 'Relu' if hf_config['activation_function'] == 'relu' else 'Gelu'
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L498
    # has post decoder layernorm when layernorm_type is pre layernorm
    has_post_decoder_layernorm = layernorm_type == 'pre_layernorm'

    if summarize:
        top_k = 2
        output_len = 100
    else:
        top_k = 1
        output_len = 256
    top_p = 0.0
    temperature = 1
    max_seq_len = hf_config['max_position_embeddings']
    max_batch_size = 1
    repetition_penalty = 1
    random_seed = 0
    vocab_size = hf_config['vocab_size']
    tensor_para_size = args.tensor_para_size
    pipeline_para_size = args.pipeline_para_size
    lib_path = args.lib_path
    ckpt_path = os.path.join(ft_model_location, f'{tensor_para_size}-gpu')

    print(f"top_k: {top_k}")
    print(f"top_p: {top_p}")
    print(f"int8_mode: {args.int8_mode}")
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

    infer_decode_args = dict(
        beam_width=1,
        top_k=top_k * torch.ones(max_batch_size, dtype=torch.int32),
        top_p=top_p * torch.ones(max_batch_size, dtype=torch.float32),
        temperature=temperature * torch.ones(max_batch_size, dtype=torch.float32),
        repetition_penalty=repetition_penalty * torch.ones(max_batch_size, dtype=torch.float32),
        random_seed=random_seed * torch.ones(max_batch_size, dtype=torch.int64)
    )

    if not args.use_gpt_decoder_ops:
        gpt = ParallelGPT(head_num, size_per_head, vocab_size, start_id, end_id, layer_num,
                          max_seq_len, tensor_para_size, pipeline_para_size, lib_path,
                          inference_data_type=args.data_type,
                          layernorm_eps=layernorm_eps,
                          layernorm_type=layernorm_type,
                          activation_type=activation_type,
                          has_post_decoder_layernorm=has_post_decoder_layernorm,
                          int8_mode=args.int8_mode,
                          weights_data_type=args.weights_data_type)
        if not gpt.load(ckpt_path=ckpt_path):
            print("[WARNING] Checkpoint file not found. Model loading is skipped.")
    else:
        gpt = gpt_decoder.Gpt(
            num_heads=head_num,
            size_per_head=size_per_head,
            num_layers=layer_num,
            vocab_size=vocab_size,
            start_id=start_id,
            end_id=end_id,
            tensor_para_size=tensor_para_size,
            pipeline_para_size=pipeline_para_size,
            lib_path=lib_path,
            max_seq_len=max_seq_len,
            layernorm_eps=layernorm_eps,
            layernorm_type=layernorm_type,
            activation_type=activation_type,
            has_post_decoder_layernorm=has_post_decoder_layernorm,
            int8_mode=args.int8_mode,
            inference_data_type=args.data_type,
            weights_data_type=args.weights_data_type,
            use_fp32_to_compute_logit=args.use_fp32_to_compute_logit)
        gpt.load(ckpt_path, args.data_type)

    if (test_hf and summarize) or not summarize:
        model = AutoModelForCausalLM.from_pretrained(hf_model_name)
        model.cuda()
        if args.data_type == 'fp16':
            model.half()
        elif args.data_type == 'bf16':
            model.bfloat16()

    def summarize_ft_e2e(datapoint):
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
                                        return_output_length=True,
                                        **infer_decode_args)

        tokens = output[0][0][len(line_encoded[0]):ft_output_len[0]].cpu().numpy()

        output_lines = tokenizer.decode(output[0][0][len(line_encoded[0]):ft_output_len[0]])
        output_lines = ".".join(output_lines.split('.')[:4]) + "."
        return output_lines, tokens

    def summarize_ft_sep(datapoint):
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
        line_encoded = line_encoded.type(torch.int32).to(gpt.device)
        input_lengths = torch.tensor([len(line_encoded[0])], dtype=torch.int32, device=gpt.device)

        with torch.no_grad():
            output_dict = gpt.generate(input_token_ids=line_encoded,
                                       input_lengths=input_lengths,
                                       gen_length=output_len,
                                       eos_token_id=tokenizer.eos_token_id,
                                       return_output_length=True,
                                       **infer_decode_args)

        output_token_ids = output_dict['output_token_ids']
        output_lengths = output_dict['output_lengths']
        tokens = output_token_ids[0, 0, input_lengths[0]:output_lengths[0]]
        output_lines = tokenizer.decode(tokens)
        output_lines = ".".join(output_lines.split('.')[:4]) + "."
        return output_lines, tokens.cpu().numpy()

    summarize_ft = summarize_ft_e2e if not args.use_gpt_decoder_ops else summarize_ft_sep

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
                                    top_k=top_k,
                                    temperature=temperature,
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
                if args.verbose:
                    print('-' * 100)
                    print('FT Summary:', summary_ft)
                    if test_hf:
                        print('HF Summary:', summary_hf)
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
            if args.rougeLsum_threshold is not None:
                assert computed_metrics_ft["rougeLsum"].mid[2]*100 >= args.rougeLsum_threshold, "[INFO] TEST FAIL !"
                print(f"[INFO] TEST PASS !")
        else:
            em_metrics = compute_exact_match(tokens)


if __name__ == '__main__':
    main()
