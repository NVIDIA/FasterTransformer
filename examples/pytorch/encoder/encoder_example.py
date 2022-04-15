# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
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

import os
import argparse
import timeit
import torch
import torch.cuda.nvtx as nvtx
import time
import sys
import numpy as np
import random
from onmt.utils.misc import sequence_mask
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.pytorch.encoder.utils.ft_encoder import EncoderWeights
from examples.pytorch.encoder.utils.ft_encoder import CustomEncoder
from examples.pytorch.encoder.utils.ft_encoder import ONMTEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_size', type=int,
                        help='batch size')
    parser.add_argument('layer_num', type=int,
                        help='number of layers')
    parser.add_argument('seq_len', type=int,
                        help='sequence length')
    parser.add_argument('head_num', type=int,
                        help='head number')
    parser.add_argument('head_size', type=int,
                        help='size per head')
    parser.add_argument('--fp16', action='store_true',
                        help='is fp16')
    parser.add_argument('--time', action='store_true',
                        help='test the time or not.')
    parser.add_argument('--avg_seq_len', type=int, default=-1, metavar='NUMBER',
                        help='average sequence length (default: -1)')
    parser.add_argument('--remove_padding', action='store_true',
                        help='Remove the padding of sentences of encoder.')
    parser.add_argument('--allow_gemm_test', action='store_true',
                        help='Whether allow gemm test inside FT.')
    parser.add_argument('--ths_path', type=str, default='./lib/libth_encoder.so',
                        help='path of the pyt_fastertransformer dynamic lib file')    
    args = parser.parse_args()
    encoder_example(vars(args))

def encoder_example(args):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    batch_size = args['batch_size']
    seq_len = args['seq_len']
    layer_num = args['layer_num']
    head_num = args['head_num']
    head_size = args['head_size']
    hidden_dim = head_num * head_size

    print("\n=============== Argument ===============")
    for key in args:
        print("{}: {}".format(key, args[key]))
    print("========================================\n")

    inp = torch.empty(batch_size, seq_len, hidden_dim).cuda()
    torch.nn.init.normal_(inp, -0.02, 0.02)
    mem_seq_lens = torch.randint(1, seq_len+1, (batch_size,), dtype=torch.int32).cuda()
    if args['remove_padding']:
        if args['avg_seq_len'] > 0:
            mem_seq_lens = torch.ones((batch_size,)) * args['avg_seq_len']
            mem_seq_lens = mem_seq_lens.to(torch.int32).cuda()
        elif args['avg_seq_len'] == -1:
            mem_seq_lens = torch.ones((batch_size,)) * seq_len / 2
            mem_seq_lens = mem_seq_lens.to(torch.int32).cuda()
        else:
            raise ValueError("wrong avg_seq_len")

    mask = ~sequence_mask(mem_seq_lens, seq_len).unsqueeze(1)
    if args['fp16']:
        inp = inp.half()

    weights = EncoderWeights(layer_num, hidden_dim)

    onmt_encoder = ONMTEncoder(layer_num, hidden_dim, head_num, 4 * hidden_dim, weights)
    onmt_encoder.cuda()
    if args['fp16']:
        onmt_encoder.half()
    onmt_encoder.eval()
    onmt_encoder = torch.jit.trace(onmt_encoder, (inp, mask))

    if args['fp16']:
        weights.to_half()
    weights.to_cuda()
    custom_encoder = CustomEncoder(layer_num, head_num, head_size, weights,
                                    remove_padding=False, allow_gemm_test=args['allow_gemm_test'],
                                    path=args['ths_path'])
    custom_encoder = torch.jit.script(custom_encoder)

    eff_custom_encoder = CustomEncoder(layer_num, head_num, head_size, weights,
                                    remove_padding=True, allow_gemm_test=args['allow_gemm_test'],
                                    path=args['ths_path'])
    eff_custom_encoder = torch.jit.script(eff_custom_encoder)

    with torch.no_grad():
        output_mask = sequence_mask(mem_seq_lens, args['seq_len']).to(mask.dtype).unsqueeze(-1)
        onmt_output = onmt_encoder(inp, mask) * output_mask
        print(onmt_output)
        print(onmt_output.size())

        ft_output = custom_encoder(inp, mem_seq_lens) * output_mask
        print(ft_output)
        print(ft_output.size())

        eff_ft_output = eff_custom_encoder(inp, mem_seq_lens) * output_mask
        print(eff_ft_output)
        print(eff_ft_output.size())

        FT_diff = torch.abs(onmt_output - ft_output)
        print('FT Mean diff: {}'.format(torch.mean(FT_diff)))
        print('FT Max diff:  {}'.format(torch.max(FT_diff)))
        print('FT Min diff:  {}'.format(torch.min(FT_diff)))

        EFF_diff = torch.abs(onmt_output - eff_ft_output)
        print('EFF-FT Mean diff: {}'.format(torch.mean(EFF_diff)))
        print('EFF-FT Max diff:  {}'.format(torch.max(EFF_diff)))
        print('EFF-FT Min diff:  {}'.format(torch.min(EFF_diff)))

        if args['time']:
            iterations = 100

            for i in range(iterations):
                output = onmt_encoder(inp, mask)
            t10 = timeit.default_timer()
            # nvtx.range_push("hf")
            for i in range(iterations):
                # nvtx.range_push("hf"+str(i))
                output = onmt_encoder(inp, mask)
                # nvtx.range_pop()
            # nvtx.range_pop()
            t1 = timeit.default_timer() - t10
            # time.sleep(60)

            for i in range(iterations):
                output = custom_encoder(inp, mem_seq_lens)
            t20 = timeit.default_timer()
            # nvtx.range_push("ext")
            for i in range(iterations):
                # nvtx.range_push("ext"+str(i))
                output = custom_encoder(inp, mem_seq_lens)
                # nvtx.range_pop()
            # nvtx.range_pop()
            t2 = timeit.default_timer() - t20
            # time.sleep(60)

            for i in range(iterations):
                output = eff_custom_encoder(inp, mem_seq_lens)
            t30 = timeit.default_timer()
            # nvtx.range_push("eff_ext")
            for i in range(iterations):
                # nvtx.range_push("eff_ext"+str(i))
                output = eff_custom_encoder(inp, mem_seq_lens)
                # nvtx.range_pop()
            # nvtx.range_pop()
            t3 = timeit.default_timer() - t30
            # time.sleep(60)
            print("[INFO] ONMTEnocder time costs:    {:.2f} ms".format(t1*1000/iterations))
            print("[INFO] FasterTransformer time costs:     {:.2f} ms".format(t2*1000/iterations))
            print("[INFO] EFF-FasterTransformer time costs: {:.2f} ms".format(t3*1000/iterations))
        
        return max(torch.mean(FT_diff), torch.mean(EFF_diff))


if __name__ == '__main__':
    main()
