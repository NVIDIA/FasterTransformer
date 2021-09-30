# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import timeit
import torch
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.pytorch.gpt.utils.parallel_gpt import ParallelGPT
import examples.pytorch.gpt.utils.gpt_token_encoder as encoder


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
    parser.add_argument('--tensor_para_size', type=int, default=1,
                        help='tensor parallel size')
    parser.add_argument('--pipeline_para_size', type=int, default=1,
                        help='pipeline parallel size')
    parser.add_argument('--ckpt_path', type=str, default='../models/megatron-models/c-model/345m/1-gpu',
                        help='path to the checkpoint file.')
    parser.add_argument('--lib_path', type=str, default='./lib/libth_parallel_gpt.so',
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
    parser.add_argument('--repetition_penalty', type=float, default=1.,
                        help='repetition penalty')
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
    parser.add_argument('--is_fix_random_seed', type=bool, default=True,
                        help='is fixing the random seed.')

    args = parser.parse_args()

    layer_num = args.layer_num
    output_len = args.output_len
    head_num = args.head_num
    size_per_head = args.size_per_head
    vocab_size = args.vocab_size
    top_k = args.top_k
    top_p = args.top_p
    temperature = args.temperature
    tensor_para_size = args.tensor_para_size
    pipeline_para_size = args.pipeline_para_size
    start_id = args.start_id
    end_id = args.end_id
    max_batch_size = args.max_batch_size
    max_seq_len = args.max_seq_len
    repetition_penalty = args.repetition_penalty

    print("\n=============== Arguments ===============")
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))
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

    start_lengths = [len(ids) for ids in start_ids]
    input_len = max(start_lengths)

    start_ids = pad_sequence(start_ids, batch_first=True, padding_value=end_id)
    start_lengths = torch.IntTensor(start_lengths)
    
    if args.is_fix_random_seed == True:
        random_seed = 0
    else:
        random_seed = random.randint(0, 100000)

    # Prepare model.
    gpt = ParallelGPT(head_num, size_per_head, vocab_size, start_id, end_id,
              layer_num, top_k, top_p, random_seed, temperature, output_len, max_seq_len,
              tensor_para_size, pipeline_para_size, max_batch_size, repetition_penalty, lib_path=args.lib_path)
    if not gpt.load(ckpt_path=args.ckpt_path):
        print("[WARNING] Checkpoint file not found. Model loading is skipped.")
    if args.fp16:
        gpt.half()

    with torch.no_grad():
        # Generate tokens.
        tokens_batch = gpt(start_ids, start_lengths)
        # only a thread (rank 0) gets the output, while the others are supposed to return None.
        if tokens_batch is not None:
            outputs = []
            tokens_batch = tokens_batch.cpu().numpy()
            for i, (context, tokens) in enumerate(zip(contexts, tokens_batch)):
                token = tokens[start_lengths[i]:]  # exclude context input from the output
                output = enc.decode(tokens[input_len:])
                outputs.append(output)
                print("[INFO] batch {}: \n[Context]\n{}\n\n[Output]\n{}".format(i, context, output))

            if args.sample_output_file:
                with open(args.sample_output_file, "w+") as f:
                    outputs = [o.replace("\n", "\\n") for o in outputs]
                    f.writelines("\n".join(outputs))

        # Measure inference time.
        if args.time:
            iterations = 10
            for i in range(iterations):
                tokens_batch = gpt(start_ids, start_lengths)

            time = timeit.default_timer()
            for i in range(iterations):
                tokens_batch = gpt(start_ids, start_lengths)
            time_elapsed = timeit.default_timer() - time
            print("[INFO] GPT time costs: {:.2f} ms".format(time_elapsed * 1000 / iterations))


if __name__ == '__main__':
    main()
