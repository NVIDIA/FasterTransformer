# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import numpy as np
import utils.gpt_token_encoder as encoder
from torch.nn.utils.rnn import pad_sequence


from utils.gpt import GPT, GPTWeights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_num', type=int, default=24,
                        help='number of layers')
    parser.add_argument('--output_len', type=int, default=32,
                        help='output sequence length to generate.')
    parser.add_argument('--head_num', type=int, default=16,
                        help='head number')
    parser.add_argument('--size_per_head', type=int, default=64,
                        help='size per head')
    parser.add_argument('--vocab_size', type=int, default=50304,
                        help='vocab size')
    parser.add_argument('--top_k', type=int, default=1,
                        help='top k candidate num')
    parser.add_argument('--top_p', type=float, default=0.,
                        help='top p probability threshold')
    parser.add_argument('--temperature', type=float, default=1.,
                        help='temperature')
    parser.add_argument('--is_fuse_QKV', type=bool, default=True,
                        help='whether or not to fuse QKV')
    parser.add_argument('--tensor_para_size', type=int, default=1,
                        help='tensor parallel size')
    parser.add_argument('--layer_para_size', type=int, default=1,
                        help='layer parallel size')
    parser.add_argument('--layer_para_batch_size', type=int, default=1,
                        help='local batch size for pipeline parallel')
    parser.add_argument('--ckpt_path', type=str, default='../models/megatron-models/c-model/345m/1-gpu',
                        help='path to the checkpoint file.')
    parser.add_argument('--lib_path', type=str, default='./lib/libpyt_fastertransformer.so',
                        help='path to the pyt_fastertransformer dynamic lib file.')
    parser.add_argument('--vocab_file', type=str, default="../models/gpt2-vocab.json",
                        help='vocabulary file.')
    parser.add_argument('--merges_file', type=str, default="../models/gpt2-merges.txt",
                        help='merges file.')
    parser.add_argument('--start_id', type=int, default=50256,
                        help='start token id.')
    parser.add_argument('--end_id', type=int, default=50256,
                        help='end token id.')
    parser.add_argument('--max_batch_size', type=int, default=8,
                        help='max batch size.')
    parser.add_argument('--max_seq_len', type=int, default=128,
                        help='max sequence length.')
    parser.add_argument('--fp16', action='store_true',
                        help='whether or not to run in fp16')
    parser.add_argument('--time', action='store_true',
                        help='whether or not to measure time elapsed.')
    parser.add_argument('--sample_input_file', type=str, default=None,
                        help='path to sample input file. If not set, it runs with no context inputs.')
    parser.add_argument('--sample_output_file', type=str, default=None,
                        help='path to sample output file.')
                        
    args = parser.parse_args()

    layer_num = args.layer_num
    output_len = args.output_len
    head_num = args.head_num
    size_per_head = args.size_per_head
    vocab_size = args.vocab_size
    top_k = args.top_k
    top_p = args.top_p
    temperature = args.temperature
    is_fuse_QKV = args.is_fuse_QKV
    tensor_para_size = args.tensor_para_size
    layer_para_size = args.layer_para_size
    layer_para_batch_size = args.layer_para_batch_size
    start_id = args.start_id
    end_id = args.end_id
    max_batch_size = args.max_batch_size
    max_seq_len = args.max_seq_len

    print("\n=============== Arguments ===============")
    for arg in vars(args):
        print ("{}: {}".format(arg, getattr(args, arg)))
    print("=========================================\n")

    enc = encoder.get_encoder(args.vocab_file, args.merges_file)

    # Inputs
    contexts = []
    if args.sample_input_file:  # conditional case
        with open(args.sample_input_file, "r") as f:
            contexts = f.read().splitlines()
            batch_size = min(len(contexts), max_batch_size)
        contexts = contexts[:batch_size]
        start_ids = [torch.IntTensor(enc.encode(c)) for c in contexts]
    else:  # unconditional case
        batch_size = max_batch_size
        contexts = ['<|endoftext|>'] * batch_size
        start_ids = [torch.IntTensor([end_id])] * batch_size

    print("[INFO] batch size: {}".format(batch_size))

    start_lengths = [len(ids) for ids in start_ids]
    input_len = min(start_lengths)

    start_ids = pad_sequence(start_ids, batch_first=True, padding_value=end_id)
    start_lengths = torch.IntTensor(start_lengths)
    attn_mask = torch.ones((batch_size, input_len, input_len)).tril()

    # Prepare model.
    gpt = GPT(head_num, size_per_head, vocab_size, start_id, end_id,
              layer_num, top_k, top_p, temperature, output_len, max_seq_len, 
              tensor_para_size, layer_para_size, layer_para_batch_size, 
              is_fuse_QKV, max_batch_size, lib_path=args.lib_path)
    gpt.load(ckpt_path=args.ckpt_path)
    if args.fp16:
        gpt.half()
    gpt.cuda()

    with torch.no_grad():
        # Generate tokens.
        tokens_batch = gpt(start_ids, start_lengths, attn_mask)
        if tokens_batch is not None:  # only a thread (rank 0) gets the output, while the others are supposed to return None.
            outputs = []
            tokens_batch = tokens_batch.cpu().numpy()
            for i, (context, tokens) in enumerate(zip(contexts, tokens_batch)):
                token = tokens[start_lengths[i]:]  # exclude context input from the output
                output = enc.decode(tokens[start_lengths[i]:])
                outputs.append(output)
                print("[INFO] batch {}: \n[Context]\n{}\n\n[Output]\n{}".format(i, context, output))

            if args.sample_output_file:
                with open(args.sample_output_file, "w+") as f:
                    outputs = [o.replace("\n","\\n") for o in outputs]
                    f.writelines("\n".join(outputs))

        # Measure inference time.
        if args.time:
            iterations = 10
            for i in range(iterations):
                tokens_batch = gpt(start_ids, start_lengths, attn_mask)
            
            time = timeit.default_timer()
            for i in range(iterations):
                tokens_batch = gpt(start_ids, start_lengths, attn_mask)
            time_elapsed = timeit.default_timer() - time
            print("[INFO] GPT time costs: {:.2f} ms".format(time_elapsed*1000/iterations))

if __name__ == '__main__':
    main()
