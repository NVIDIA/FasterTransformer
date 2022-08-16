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

'''
This example is used to verify the correctess on summarization task. So, we don't
put benchmark testing in this example.
'''

from __future__ import print_function
import argparse
import json
import numpy as np
import os
import sys
import torch
import torch.distributed as dist
from datasets import load_dataset, load_metric
# dir_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(dir_path + "/../../../3rdparty/transformers/src/")

from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Config
from tqdm import tqdm
import configparser
import math
import datetime

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.pytorch.t5.utils.ft_decoding import FTT5DecodingWeight, FTT5Decoding, FTT5
from examples.pytorch.t5.utils.ft_encoder import FTT5EncoderWeight, FTT5Encoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ft_model_location', type=str,
                        default='/models/T5/HF/t5-base/c-models/')
    parser.add_argument('--hf_model_location', type=str,
                        default='/models/T5/HF/t5-base/')
    parser.add_argument('--disable_summarize', action='store_true')
    parser.add_argument('--test_hf', action='store_true')
    parser.add_argument('--test_ft', action='store_true')
    parser.add_argument('--data_type', type=str, choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument("--cache_path", type=str, default="/workdir/datasets/ccdv/")
    parser.add_argument("--max_ite", type=int, default=20)
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--ft_use_hf_config", action="store_true",
                        help="use the hyper-parameters from the hf model")
    parser.add_argument('--lib_path', type=str, default='./lib/libth_t5.so',
                        help='path to the pyt_fastertransformer dynamic lib file.')
    parser.add_argument('--tensor_para_size', type=int, default=1,
                        help='tensor parallel size')
    parser.add_argument('--pipeline_para_size', type=int, default=1,
                        help='pipeline parallel size')
    parser.add_argument('--rougeLsum_threshold', type=float,
                        help='Threshold of FT rougeLsum score')

    args = parser.parse_args()

    if dist.is_mpi_available():
        try:
            dist.init_process_group(backend='mpi')
            rank = dist.get_rank()
        except:
            rank = dist.get_rank()
    else:
        rank = 0

    disable_summarize = args.disable_summarize
    test_hf = args.test_hf
    test_ft = args.test_ft

    tensor_para_size = args.tensor_para_size
    pipeline_para_size = args.pipeline_para_size
    ft_model_location = args.ft_model_location + f"/{tensor_para_size}-gpu/"
    hf_model_location = args.hf_model_location

    tokenizer = AutoTokenizer.from_pretrained(hf_model_location)
    tokenizer.pad_token = tokenizer.eos_token
    dataset_cnn = load_dataset("ccdv/cnn_dailymail", '3.0.0', cache_dir=args.cache_path)

    if rank == 0 and test_hf:
        start_time = datetime.datetime.now()
        if args.data_type == "fp32":
            model = T5ForConditionalGeneration.from_pretrained(hf_model_location, torch_dtype=torch.float32).cuda()
        elif args.data_type == "fp16":
            model = T5ForConditionalGeneration.from_pretrained(hf_model_location, torch_dtype=torch.float16).cuda()
        elif args.data_type == "bf16":
            model = T5ForConditionalGeneration.from_pretrained(hf_model_location, torch_dtype=torch.bfloat16).cuda()
        stop_time = datetime.datetime.now()
        print(f"[INFO] load HF model spend {(stop_time - start_time).total_seconds()} sec")

    if test_ft:
        ckpt_config = configparser.ConfigParser()

        ckpt_config_path = os.path.join(ft_model_location, 'config.ini')
        if os.path.isfile(ckpt_config_path):
            ckpt_config.read(ckpt_config_path)
        else:
            assert False, "[ERROR] This example only support loading model with FT format directly."

        weight_data_type = np.float32
        weight_data_type = {"fp16": np.float16, "fp32": np.float32}[ckpt_config.get("encoder", "weight_data_type")]
        relative_attention_max_distance = 128
        encoder_config = T5Config(vocab_size=ckpt_config.getint("encoder", "vocab_size"),
                                  d_model=ckpt_config.getint("encoder", "d_model"),
                                  d_kv=ckpt_config.getint("encoder", "d_kv"),
                                  d_ff=ckpt_config.getint("encoder", "d_ff"),
                                  num_layers=ckpt_config.getint("encoder", "num_layers"),
                                  num_decoder_layers=ckpt_config.getint("encoder", "num_decoder_layers"),
                                  num_heads=ckpt_config.getint("encoder", "num_heads"),
                                  relative_attention_num_buckets=ckpt_config.getint(
                                      "encoder", "relative_attention_num_buckets_or_max_pos_seq_len"),
                                  feed_forward_proj=ckpt_config.get("encoder", "feed_forward_proj"),
                                  pad_token_id=ckpt_config.getint("encoder", "pad_token_id"),
                                  eos_token_id=ckpt_config.getint("encoder", "eos_token_id"),
                                  is_gated_act=ckpt_config.getboolean("encoder", "is_gated_act", fallback=0),
                                  )
        decoder_config = T5Config(vocab_size=ckpt_config.getint("decoder", "vocab_size"),
                                  d_model=ckpt_config.getint("decoder", "d_model"),
                                  d_kv=ckpt_config.getint("decoder", "d_kv"),
                                  d_ff=ckpt_config.getint("decoder", "d_ff"),
                                  num_layers=ckpt_config.getint("decoder", "num_layers"),
                                  num_decoder_layers=ckpt_config.getint("decoder", "num_decoder_layers"),
                                  num_heads=ckpt_config.getint("decoder", "num_heads"),
                                  relative_attention_num_buckets=ckpt_config.getint(
                                      "decoder", "relative_attention_num_buckets_or_max_pos_seq_len"),
                                  feed_forward_proj=ckpt_config.get("decoder", "feed_forward_proj"),
                                  pad_token_id=ckpt_config.getint("decoder", "pad_token_id"),
                                  eos_token_id=ckpt_config.getint("decoder", "eos_token_id"),
                                  decoder_start_token_id=ckpt_config.getint("decoder", "decoder_start_token_id"),
                                  is_gated_act=ckpt_config.getboolean("decoder", "is_gated_act", fallback=0),
                                  )
        assert decoder_config.feed_forward_proj == encoder_config.feed_forward_proj
        assert decoder_config.feed_forward_proj == encoder_config.feed_forward_proj

        t5_with_bias = ckpt_config.getboolean("structure", "t5_with_bias")
        use_gated_activation = encoder_config.is_gated_act
        position_embedding_type = 0 if ckpt_config.get('structure', 'position_embedding_type') == 'relative' else 1
        activation_type = encoder_config.feed_forward_proj

        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L1660
        # if tie_word_embeddings == True, scale the decoder output by sequence_output = sequence_output * (self.model_dim**-0.5)
        tie_word_embeddings = ckpt_config.getboolean("decoder", "tie_word_embeddings")
        ft_encoder_weight = FTT5EncoderWeight(
            encoder_config,
            tensor_para_size,
            pipeline_para_size,
            t5_with_bias=t5_with_bias,
            use_gated_activation=use_gated_activation,
            position_embedding_type=position_embedding_type,
            weight_data_type=weight_data_type
        )
        ft_decoding_weight = FTT5DecodingWeight(
            decoder_config,
            tensor_para_size,
            pipeline_para_size,
            t5_with_bias=t5_with_bias,
            use_gated_activation=use_gated_activation,
            position_embedding_type=position_embedding_type,
            weight_data_type=weight_data_type,
        )

        start_time = datetime.datetime.now()
        ft_encoder_weight.load_from_bin(ft_model_location)
        stop_time = datetime.datetime.now()
        print(f"[INFO] load FT encoder model spend {(stop_time - start_time).total_seconds()} sec")
        start_time = datetime.datetime.now()
        ft_decoding_weight.load_from_bin(ft_model_location)
        stop_time = datetime.datetime.now()
        print(f"[INFO] load FT decoding model spend {(stop_time - start_time).total_seconds()} sec")
        if args.data_type == "fp32":
            ft_encoder_weight.to_float()
            ft_decoding_weight.to_float()
        elif args.data_type == "fp16":
            ft_encoder_weight.to_half()
            ft_decoding_weight.to_half()
        elif args.data_type == "bf16":
            ft_encoder_weight.to_bfloat16()
            ft_decoding_weight.to_bfloat16()

        ft_encoder_weight.to_cuda()
        ft_decoding_weight.to_cuda()

        q_scaling = 1.0 / (math.sqrt(encoder_config.d_kv))
        remove_padding = True
        ft_encoder = FTT5Encoder(ft_encoder_weight.w, args.lib_path, encoder_config.num_heads,
                                 encoder_config.d_kv, encoder_config.d_ff,
                                 encoder_config.d_model, remove_padding, encoder_config.num_layers,
                                 encoder_config.relative_attention_num_buckets,
                                 relative_attention_max_distance, False, q_scaling, tensor_para_size,
                                 pipeline_para_size, t5_with_bias,
                                 position_embedding_type, activation_type=activation_type)

        ft_decoding = FTT5Decoding(ft_decoding_weight.w, args.lib_path,
                                   decoder_config.num_heads, decoder_config.d_kv,
                                   decoder_config.d_ff, encoder_config.d_model,
                                   decoder_config.d_model, decoder_config.num_layers,
                                   decoder_config.decoder_start_token_id, decoder_config.eos_token_id,
                                   decoder_config.vocab_size, q_scaling,
                                   decoder_config.relative_attention_num_buckets, max_distance=relative_attention_max_distance,
                                   tensor_para_size=tensor_para_size, pipeline_para_size=pipeline_para_size,
                                   t5_with_bias=t5_with_bias, position_embedding_type=position_embedding_type,
                                   activation_type=activation_type, tie_word_embeddings=tie_word_embeddings)

        ft_t5 = FTT5(ft_encoder, ft_decoding)

    if disable_summarize:
        top_k = 1
        output_len = args.max_seq_len
    else:
        top_k = 1
        output_len = args.max_seq_len

    def summarize_ft(datapoint):
        if not disable_summarize:
            line = "summarize: " + datapoint['article']
        else:
            line = datapoint['article']
        line = line.strip()
        line = line.replace(" n't", "n't")

        line_tokens = tokenizer(line, return_tensors='pt')

        with torch.no_grad():
            output, ft_output_len = ft_t5(line_tokens,
                                          None,
                                          1,
                                          output_len,
                                          top_k,
                                          0.0,
                                          beam_search_diversity_rate=0.0,
                                          is_return_output_log_probs=False,
                                          is_return_cum_log_probs=False)
        tokens = output[0][0]

        output_lines = tokenizer.decode(output[0][0][:ft_output_len[0][0]])
        output_lines = ".".join(output_lines.split('.')[:4]) + "."
        return output_lines, tokens

    def summarize_hf(datapoint):
        if not disable_summarize:
            line = "summarize: " + datapoint['article']
        else:
            line = datapoint['article']
        line = line.strip()
        line = line.replace(" n't", "n't")

        line_encoded = tokenizer.encode(line, return_tensors='pt')
        line_encoded = line_encoded.cuda()

        with torch.no_grad():
            output = model.generate(line_encoded,
                                    max_length=output_len + 1,
                                    top_k=top_k,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id)

        tokens = output[0].cpu().numpy()
        output_lines = tokenizer.decode(output[0])
        output_lines = ".".join(output_lines.split('.')[:4]) + "."
        return output_lines, tokens

    if disable_summarize:
        tokens = []
    else:
        metric_ft = load_metric("rouge")
        metric_hf = load_metric("rouge")

    if not disable_summarize:
        datapoint = dataset_cnn['test'][0]
        if test_ft:
            summary_ft, _ = summarize_ft(datapoint)
            if rank == 0:
                print('---------------------------------------------------------')
                print('FT Generated : ')
                print(' Article : ', datapoint['article'])
                print('\n Highlights : ', datapoint['highlights'])
                print('\n Summary : ', summary_ft)
                print('---------------------------------------------------------')
                metric_ft.add_batch(predictions=[summary_ft], references=[datapoint['highlights']])

        if test_hf and rank == 0:
            summary_hf, _ = summarize_hf(datapoint)
            print('---------------------------------------------------------')
            print('HF Generated : ')
            print(' Article : ', datapoint['article'])
            print('\n Highlights : ', datapoint['highlights'])
            print('\n Summary : ', summary_hf)
            print('---------------------------------------------------------')
            metric_hf.add_batch(predictions=[summary_hf], references=[datapoint['highlights']])

    ft_time = 0.0
    hf_time = 0.0
    for data_point_idx in tqdm(range(1, 11490, int(11490 / args.max_ite))):
        try:
            datapoint = dataset_cnn['test'][data_point_idx]

            start_time = datetime.datetime.now()
            if test_ft:
                summary_ft, tokens_ft = summarize_ft(datapoint)
            stop_time = datetime.datetime.now()
            ft_time += (stop_time - start_time).total_seconds()

            if rank == 0 and ((test_hf and not disable_summarize) or disable_summarize):
                start_time = datetime.datetime.now()
                summary_hf, tokens_hf = summarize_hf(datapoint)
                stop_time = datetime.datetime.now()
                hf_time += (stop_time - start_time).total_seconds()

            if rank == 0:
                if not disable_summarize:
                    if test_ft:
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
        if not disable_summarize:
            if test_ft:
                computed_metrics_ft = metric_ft.compute()

            if test_hf:
                computed_metrics_hf = metric_hf.compute()

                print(f'Hugging Face (total latency: {hf_time} sec)')
                for key in computed_metrics_hf.keys():
                    print(f'{key} : {computed_metrics_hf[key].mid[2]*100}')

            if test_ft:
                print(f'Faster Transformers (total latency: {ft_time} sec)')
                for key in computed_metrics_ft.keys():
                    print(f'{key} : {computed_metrics_ft[key].mid[2]*100}')
                if args.rougeLsum_threshold != None:
                    assert computed_metrics_ft["rougeLsum"].mid[2] * \
                        100 >= args.rougeLsum_threshold, "[INFO] TEST FAIL !"
                    print(f"[INFO] TEST PASS !")
        else:
            em_metrics = compute_exact_match(tokens)


if __name__ == '__main__':
    main()
