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
import random
import numpy as np
# import torch.cuda.nvtx as nvtx
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")

from examples.pytorch.decoding.utils.decoding import DecodingWeights, TorchDecoding, ArgHelper
from examples.pytorch.decoding.utils.ft_decoding import FtDecodingWeights, CustomDecoding

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
    parser.add_argument('-inter_size', '--inter_size', type=int, default=0, metavar='NUMBER',
                        help='inter_size (default: 0)')
    parser.add_argument('-mem_hidden', '--memory_hidden_dim', type=int, default=512, metavar='NUMBER',
                        help='memory hidden dim (default: 512)')
    parser.add_argument('beam_size', type=int,
                        help='beam size')
    parser.add_argument('vocab_size', type=int,
                        help='vocab size')
    parser.add_argument('--fp16', action='store_true',
                        help='is fp16')
    parser.add_argument('--time', action='store_true',
                        help='test the time or not.')
    parser.add_argument('--use_pretrained', action='store_true',
                        help='use pretrained weights or not.')
    parser.add_argument('--decoding_ths_path', type=str, default='./lib/libth_decoding.so',
                        help='path of the pyt_fastertransformer dynamic lib file')
    parser.add_argument('--decoder_ths_path', type=str, default='./lib/libth_decoder.so',
                        help='path of the pyt_fastertransformer dynamic lib file')
    parser.add_argument('-diversity_rate', '--beam_search_diversity_rate', type=float, default=0.0, metavar='NUMBER',
                        help='deviersity rate of beam search. default is 0. When diversity rate = 0, it is equivalent to the naive beams earch.')
    parser.add_argument('-topk', '--sampling_topk', type=int, default=1, metavar='NUMBER',
                        help='Candidate (k) value of top k sampling in decoding. Default is 1.')
    parser.add_argument('-topp', '--sampling_topp', type=float, default=0.0, metavar='NUMBER',
                        help='Probability (p) value of top p sampling in decoding. Default is 0.0. ')

    args = parser.parse_args()
    
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    if args.use_pretrained:
        layer_num = 6
        head_num = 8
        head_size = 64
        inter_size = head_num * head_size * 4
        vocab_size = 31538
    else:
        layer_num = args.layer_num
        head_num = args.head_num
        head_size = args.head_size
        inter_size = args.inter_size
        if inter_size == 0:
            inter_size = 4 * head_num * head_size
        vocab_size = args.vocab_size
    hidden_dim = head_num * head_size
    start_id = 2
    end_id = 3

    print("\n=============== Argument ===============")
    for key in vars(args):
        print("{}: {}".format(key, vars(args)[key]))
    print("========================================")

    decodingargs1 = ArgHelper('torch_decoding', 'fp16' if args.fp16 else 'fp32', os.path.abspath(args.decoder_ths_path), os.path.abspath(args.decoding_ths_path))
    decodingargs2 = ArgHelper('torch_decoding_with_decoder_ext', 'fp16' if args.fp16 else 'fp32', os.path.abspath(args.decoder_ths_path), os.path.abspath(args.decoding_ths_path))

    mem = torch.empty(args.batch_size, args.seq_len, args.memory_hidden_dim).cuda()
    torch.nn.init.uniform_(mem, -1, 1)
    if args.fp16:
        mem = mem.half()
    mem_seq_lens = torch.randint(1, args.seq_len+1, (args.batch_size,), dtype=torch.int32).cuda()

    if args.use_pretrained:
        ckpt = torch.load('./pytorch/translation/models/averaged-10-epoch.pt')
        import re
        def fix_key(s):
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                        r'\1.layer_norm\2.bias', s)
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                        r'\1.layer_norm\2.weight', s)
            return s
        ckpt['model'] = {fix_key(k): v for k, v in ckpt['model'].items()}
        weights = DecodingWeights(layer_num, hidden_dim, vocab_size, ckpt)
    else:
        weights = DecodingWeights(layer_num, hidden_dim, vocab_size)
    ft_weights = FtDecodingWeights(layer_num, hidden_dim, weights.w)
    
    # TODO(bhsueh) Add decoder op
    torch_decoding = TorchDecoding(layer_num, head_num, head_size, vocab_size, start_id, end_id, weights, args=decodingargs1)
    torch_decoding_with_decoder_ext = TorchDecoding(layer_num, head_num, head_size, vocab_size, start_id, end_id, weights, args=decodingargs2)
    torch_decoding.cuda()
    torch_decoding_with_decoder_ext.cuda()
    if args.fp16:
        torch_decoding.half()
        torch_decoding_with_decoder_ext.half()
    torch_decoding.eval()
    torch_decoding_with_decoder_ext.eval()
    ft_weights.to_cuda()
    if args.fp16:
        ft_weights.to_half()
    custom_decoding = CustomDecoding(head_num, head_size,
                                    inter_size, args.memory_hidden_dim, layer_num, vocab_size,
                                    start_id, end_id, args.beam_search_diversity_rate,
                                    args.sampling_topk, args.sampling_topp, 1.0,
                                    1.0, 1.0, ft_weights, args=decodingargs1)

    with torch.no_grad():
        output0, lens0 = torch_decoding(args.batch_size, args.beam_size, args.seq_len, mem, mem_seq_lens)
        print(output0)
        print(lens0)
        # return
        output1, lens1 = torch_decoding_with_decoder_ext(args.batch_size, args.beam_size, args.seq_len, mem, mem_seq_lens)
        print(output1)
        print(lens1)
        output2, lens2 = custom_decoding(args.batch_size, args.beam_size, args.seq_len, mem, mem_seq_lens)
        print(output2)
        print(lens2)
        
        diff = torch.abs((output0 - output1) / output0)
        print('FT Decoder Mean relative diff: {}     Max relative diff: {}     Min relative diff: {}'.format(
            torch.mean(diff), torch.max(diff), torch.min(diff)))
        diff = torch.abs((output0 - output2) / output0)
        print('FT Decoding Mean relative diff: {}     Max relative diff: {}     Min relative diff: {}'.format(
            torch.mean(diff), torch.max(diff), torch.min(diff)))

        if args.time:
            iterations = 10

            for i in range(iterations):
                output, lens = torch_decoding(args.batch_size, args.beam_size, args.seq_len, mem, mem_seq_lens)
            t00 = timeit.default_timer()
            for i in range(iterations):
                output, lens = torch_decoding(args.batch_size, args.beam_size, args.seq_len, mem, mem_seq_lens)
            t0 = timeit.default_timer() - t00

            # for i in range(iterations):
            #     output, lens = torch_decoding_with_decoder_ext(args.batch_size, args.beam_size, args.seq_len, mem, mem_seq_lens)
            # t10 = timeit.default_timer()
            # for i in range(iterations):
            #     output, lens = torch_decoding_with_decoder_ext(args.batch_size, args.beam_size, args.seq_len, mem, mem_seq_lens)
            # t1 = timeit.default_timer() - t10

            for i in range(iterations):
                output2, lens2 = custom_decoding(args.batch_size, args.beam_size, args.seq_len, mem, mem_seq_lens)
            t20 = timeit.default_timer()
            for i in range(iterations):
                output2, lens2 = custom_decoding(args.batch_size, args.beam_size, args.seq_len, mem, mem_seq_lens)
            t2 = timeit.default_timer() - t20
            print("[INFO] TorchDecoding time costs: {:.2f} ms".format(t0*1000/iterations))
            # print("[INFO] TorchDecoding (with FTDecoder) time costs: {:.2f} ms".format(t1*1000/iterations))
            print("[INFO] FTDecoding time costs: {:.2f} ms".format(t2*1000/iterations))


if __name__ == '__main__':
    main()
