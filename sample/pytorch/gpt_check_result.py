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

'''
This is a sample code compare the difference of results of FasterTransformer-gpt
and Megatron-LM by computing the bleu score with some conditioned contexts. 
'''

from __future__ import print_function
import copy
import numpy as np
import argparse
import os
import sys
import json
import random
sys.path.append("../sample")
from pytorch.utils.convert_gpt_token import convert_token
import pytorch.utils.gpt_token_encoder as encoder
from pytorch.utils.generate_gpt_config import generate_gpt_config
import sacrebleu

def gpt_sample(args_dict):
    print("\n=============== Argument ===============")
    for key in args_dict:
        print("{}: {}".format(key, args_dict[key]))
    print("========================================")

    np.random.seed(1)
    random.seed(1)

    refs = []
    sys = []

    MEGATRON_OUTPUT_FILENAME = ".megatron_output.json"
    VOCAB_FILEPATH = "./models/gpt2-vocab.json"
    MERGE_FILEPATH = "./models/gpt2-merges.txt"

    if os.path.exists(MEGATRON_OUTPUT_FILENAME) == False:
        """Megatron program."""
        os.system(" python Megatron-LM/tools/generate_samples_gpt.py \
                    --num-layers {} \
                    --hidden-size {} \
                    --num-attention-heads {} \
                    --seq-length 1024 \
                    --max-position-embeddings 1024 \
                    --micro-batch-size 4 \
                    --global-batch-size 8 \
                    --vocab-file {} \
                    --merge-file {} \
                    --load ./models/megatron-models/345m \
                    --out-seq-length {} \
                    --temperature 1.0 \
                    --genfile {} \
                    --num-samples {} \
                    --top_k {} \
                    --top_p {} \
                    --recompute ".format(args_dict['num_layer'], args_dict['head_number'] * args_dict['size_per_head'], 
                                        args_dict['head_number'], VOCAB_FILEPATH, MERGE_FILEPATH,
                                        args_dict['request_input_len'] + args_dict['request_output_len'], MEGATRON_OUTPUT_FILENAME,
                                        args_dict['request_batch_size'], args_dict['sampling_topk'], args_dict['sampling_topp']))

    with open(MEGATRON_OUTPUT_FILENAME, 'r') as megatron_file:
        for i, line in enumerate(megatron_file.readlines()):
            line_j = json.loads(line)
            refs.append(line_j['text'])
            i += 1
            if i == args_dict['request_batch_size']:
                break

    """ megatron finish """

    enc = encoder.get_encoder(VOCAB_FILEPATH, MERGE_FILEPATH)
    
    with open("../sample/cpp/start_ids.csv", "w") as f:
        for tokens in refs:
            id_list = enc.encode(tokens)
            id_str = "{}\n".format(id_list[:args_dict['request_input_len']])
            id_str = id_str.replace("[", "").replace("]", "")
            f.write(id_str)

    generate_gpt_config(args_dict)
    os.system("rm out")
    os.system("mpirun -n {} --allow-run-as-root ./bin/gpt_sample .tmp.config.ini".format(args_dict['gpu_num']))
    tokens_batch = np.loadtxt("out", dtype=np.int32).T

    for tokens in tokens_batch:
        sys.append(enc.decode(tokens))

    bleu = sacrebleu.corpus_bleu(sys, [refs])
    print("[INFO] bleu score: {}".format(bleu.score))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-max_batch_size', '--max_batch_size', type=int, default=8, metavar='NUMBER',
                        help='batch size (default: 8)')
    parser.add_argument('-max_seq_len', '--max_seq_len', type=int, default=64, metavar='NUMBER',
                        help='max sequence length (default: 64)')
    parser.add_argument('-diversity_rate', '--beam_search_diversity_rate', type=float, default=0.0, metavar='NUMBER',
                        help='deviersity rate of beam search. default is 0. When diversity rate = 0, it is equivalent to the naive beams earch.')
    parser.add_argument('-n', '--head_number', type=int, default=16, metavar='NUMBER',
                        help='head number (default: 16)')
    parser.add_argument('-size', '--size_per_head', type=int, default=64, metavar='NUMBER',
                        help='size per head (default: 64)')
    parser.add_argument('-l', '--num_layer', type=int, default=24, metavar='NUMBER',
                        help='number of layers (default: 24)')
    parser.add_argument('-v', '--vocab_size', type=int, default=50304, metavar='BOOL',
                        help='vocabulary size. (default: 50304).')
    parser.add_argument('-d', '--data_type', type=str, default="fp16", metavar='STRING',
                        help='data type (default: fp16)', choices=['fp32', 'fp16'])
    parser.add_argument('-topk', '--sampling_topk', type=int, default=4, metavar='NUMBER',
                        help='Candidate (k) value of top k sampling in decoding. Default is 4.')
    parser.add_argument('-topp', '--sampling_topp', type=float, default=0.0, metavar='NUMBER',
                        help='Probability (p) value of top p sampling in decoding. Default is 0.0.')
    parser.add_argument('-tensor_para_size', '--tensor_para_size', type=int, default=1, metavar='NUMBER',
                        help='tensor parallelism size. Default is 1.')
    parser.add_argument('-layer_para_size', '--layer_para_size', type=int, default=1, metavar='NUMBER',
                        help='layer parallelism size. Default is 1.')
    parser.add_argument('-g', '--gpu_num', type=int, default=1, metavar='NUMBER',
                        help='Number of total gpu. Default is 1.')
    parser.add_argument('-local_batch', '--local_batch_size', type=int, default=8, metavar='NUMBER',
                        help='local batch size for layer parallelism. Default is 8.')
    parser.add_argument('--model_path_prefix', type=str, default="./models/megatron-models/c-model/345m/", metavar='STRING',
                        help='Model path prfix. Default is "./models/megatron-models/c-model/345m/".')
    parser.add_argument('-temperature', '--temperature', type=float, default=1.0, metavar='NUMBER',
                        help='temperature of penalty. Default is 1.0.')
    parser.add_argument('-request_batch_size', '--request_batch_size', type=int, default=8, metavar='NUMBER',
                        help='batch size (default: 8)')
    parser.add_argument('-request_input_len', '--request_input_len', type=int, default=8, metavar='NUMBER',
                        help='input length (default: 8)')
    parser.add_argument('-request_output_len', '--request_output_len', type=int, default=32, metavar='NUMBER',
                        help='output length (default: 32)')

    args = parser.parse_args()
    gpt_sample(vars(args))
