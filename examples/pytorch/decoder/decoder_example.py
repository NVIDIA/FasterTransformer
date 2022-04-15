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

import argparse
import numpy as np
import os
import random
import sys
import timeit
import torch

from onmt.utils.misc import sequence_mask
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.pytorch.decoder.utils.decoder import ONMTDecoder, init_op_cache, init_onmt_cache
from examples.pytorch.decoder.utils.ft_decoder import FTDecoder, FtDecoderWeights
from examples.pytorch.decoding.utils.decoding import DecodingWeights

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
    parser.add_argument('--step', type=int, default=0,
                        help='decoding step number')
    parser.add_argument('--decoder_ths_path', type=str, default='./lib/libth_decoder.so',
                        help='path of the pyt_fastertransformer dynamic lib file')
    parser.add_argument('--time', action='store_true',
                        help='test the time or not.')
    parser.add_argument('--ths_path', type=str, default='./lib/libpyt_fastertransformer.so',
                        help='path of the pyt_fastertransformer dynamic lib file')
    parser.add_argument('-d', '--data_type', type=str, default="fp32", metavar='STRING',
                        help='data type (default: fp32)', choices=['fp32', 'fp16'])

    args = parser.parse_args()

    hidden_dim = args.head_num * args.head_size

    if args.step <= 0:
        step = args.seq_len
    else:
        step = args.step

    print("\n=============== Argument ===============")
    print('batch_size: ' + str(args.batch_size))
    print('layer_num: ' + str(args.layer_num))
    print('seq_len: ' + str(args.seq_len))
    print('head_num: ' + str(args.head_num))
    print('head_size: ' + str(args.head_size))
    print('hidden_dim: ' + str(hidden_dim))
    print('step: ' + str(step))
    print('data_type: ' + str(args.data_type))
    print('test_time: ' + str(args.time))
    print("========================================\n")
    
    np.random.seed(1)
    torch.manual_seed(0)
    random.seed(0)

    inp = torch.empty(args.batch_size, 1, hidden_dim).cuda()
    mem = torch.empty(args.batch_size, args.seq_len, hidden_dim).cuda() # We assume mem_hidden_dim = hidden_dim
    torch.nn.init.uniform_(inp, -0.5, 0.5)
    torch.nn.init.uniform_(mem, -0.5, 0.5)
    if args.data_type == 'fp16':
        inp = inp.half()
        mem = mem.half()
    mem_seq_lens = torch.randint(1, args.seq_len+1, (args.batch_size,), dtype=torch.int32).cuda()
    src_pad_mask = ~sequence_mask(mem_seq_lens, args.seq_len).unsqueeze(1)

    weights = DecodingWeights(args.layer_num, hidden_dim, 30000)
    ft_weights = FtDecoderWeights(args.layer_num, hidden_dim, weights.w)
    
    onmt_decoder = ONMTDecoder(args.layer_num, args.head_num, args.head_size, weights)
    onmt_decoder.cuda()
    if args.data_type == 'fp16':
        onmt_decoder.half()
    onmt_decoder.eval()

    ft_weights.to_cuda()
    weights.to_cuda()
    if args.data_type == 'fp16':
        weights.to_half()
        ft_weights.to_half()
    custom_decoder = FTDecoder(args.head_num, args.head_size, hidden_dim, args.layer_num, ft_weights, args)

    with torch.no_grad():
        self_cache, mem_cache = init_op_cache(args.layer_num, args.batch_size, 1, args.seq_len, \
                                              args.seq_len, args.head_num, args.head_size, hidden_dim, args.data_type == 'fp16')
        cache = init_onmt_cache(args.layer_num, mem)
        output1 = inp
        output2 = inp

        for i in range(step):
            output1 = onmt_decoder(output1, mem, src_pad_mask, cache, i)
            output2, self_cache, mem_cache = custom_decoder(output2, mem, mem_seq_lens, self_cache, mem_cache, torch.ones(args.batch_size, dtype=torch.int32).cuda() * i, i)
            epsilon = 1e-6
            if args.data_type == 'fp16':
                epsilon = 1e-3
            diff = torch.abs((output1 - output2) / (output1 + epsilon))
            
            print('step: {}     Mean relative diff: {}     Max relative diff: {}     Min relative diff: {}'.format(
                i, torch.mean(diff), torch.max(diff), torch.min(diff)))
            output2 = output1

        if args.time:
            iterations = 10

            for i in range(iterations):
                cache = init_onmt_cache(args.layer_num, mem)
                output1 = inp
                for i in range(step):
                    output1 = onmt_decoder(output1, mem, src_pad_mask, cache, 0)
            t10 = timeit.default_timer()
            for i in range(iterations):
                cache = init_onmt_cache(args.layer_num, mem)
                output1 = inp
                for i in range(step):
                    output1 = onmt_decoder(output1, mem, src_pad_mask, cache, 0)
            t1 = timeit.default_timer() - t10

            for i in range(iterations):
                self_cache, mem_cache = init_op_cache(args.layer_num, args.batch_size, 1, args.seq_len, \
                                                      args.seq_len, args.head_num, args.head_size, hidden_dim, args.data_type == 'fp16')
                output2 = inp
                for i in range(step):
                    output2, self_cache, mem_cache = custom_decoder(output2, mem, mem_seq_lens, self_cache, mem_cache, torch.ones(args.batch_size, dtype=torch.int32).cuda() * i, i)
            t20 = timeit.default_timer()
            for i in range(iterations):
                self_cache, mem_cache = init_op_cache(args.layer_num, args.batch_size, 1, args.seq_len, \
                                                      args.seq_len, args.head_num, args.head_size, hidden_dim, args.data_type == 'fp16')
                output2 = inp
                for i in range(step):
                    output2, self_cache, mem_cache = custom_decoder(output2, mem, mem_seq_lens, self_cache, mem_cache, torch.ones(args.batch_size, dtype=torch.int32).cuda() * i, i)
            t2 = timeit.default_timer() - t20
            print("[INFO] ONMTDecoder time costs: {:.2f} ms".format(t1*1000/iterations))
            print("[INFO] FTDecoder time costs: {:.2f} ms".format(t2*1000/iterations))


if __name__ == '__main__':
    main()