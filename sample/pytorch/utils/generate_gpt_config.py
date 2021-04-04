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

import argparse
import configparser
import os
import sys

def generate_gpt_config(args):

    is_half = 1
    if args['data_type'] == "fp32":
        is_half = 0

    config = configparser.ConfigParser()
    config["ft_instance_hyperparameter"] = {
        "max_batch_size": "{}".format(args['max_batch_size']),
        "max_seq_len": "{}".format(args['max_seq_len']),
        "layer_para_batch_size": "{}".format(args['local_batch_size']),
        "candidate_num": "{}".format(args['sampling_topk']),
        "probability_threshold": "{}".format(args['sampling_topp']),
        "temperature": "{}".format(args['temperature']),
        "tensor_para_size": "{}".format(args['tensor_para_size']),
        "layer_para_size": "{}".format(args['layer_para_size']),
        "is_half": "{}".format(is_half),
        "is_fuse_QKV": "{}".format(1),
        "model_name": "tmp_model",
        "model_path_prefix": "{}".format(args['model_path_prefix']),
    }

    config["request"] = {
        "request_batch_size": "{}".format(args['request_batch_size']),
        "request_input_len": "{}".format(args['request_input_len']),
        "request_output_len": "{}".format(args['request_output_len']),
    }

    config["tmp_model"] = {
        "head_num": "{}".format(args['head_number']),
        "size_per_head": "{}".format(args['size_per_head']),
        "vocab_size": "{}".format(args['vocab_size']),
        "decoder_layers": "{}".format(args['num_layer']),
    }

    with open('.tmp.config.ini', 'w') as configfile:
        config.write(configfile)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-max_batch_size', '--max_batch_size', type=int, default=8, metavar='NUMBER',
                        help='batch size (default: 8)')
    parser.add_argument('-max_seq_len', '--max_seq_len', type=int, default=32, metavar='NUMBER',
                        help='max sequence length (default: 32)')
    parser.add_argument('-n', '--head_number', type=int, default=32, metavar='NUMBER',
                        help='head number (default: 32)')
    parser.add_argument('-size', '--size_per_head', type=int, default=128, metavar='NUMBER',
                        help='size per head (default: 128)')
    parser.add_argument('-l', '--num_layer', type=int, default=12, metavar='NUMBER',
                        help='number of layers (default: 12)')
    parser.add_argument('-v', '--vocab_size', type=int, default=50257, metavar='BOOL',
                        help='vocabulary size. (default: 50257).')
    parser.add_argument('-d', '--data_type', type=str, default="fp32", metavar='STRING',
                        help='data type (default: fp32)', choices=['fp32', 'fp16'])
    parser.add_argument('-topk', '--sampling_topk', type=int, default=1, metavar='NUMBER',
                        help='Candidate (k) value of top k sampling in decoding. Default is 1.')
    parser.add_argument('-topp', '--sampling_topp', type=float, default=0.0, metavar='NUMBER',
                        help='Probability (p) value of top p sampling in decoding. Default is 0.0.')
    parser.add_argument('-tensor_para_size', '--tensor_para_size', type=int, default=1, metavar='NUMBER',
                        help='tensor parallelism size. Default is 1.')
    parser.add_argument('-layer_para_size', '--layer_para_size', type=int, default=1, metavar='NUMBER',
                        help='layer parallelism size. Default is 1.')
    parser.add_argument('-local_batch', '--local_batch_size', type=int, default=8, metavar='NUMBER',
                        help='local batch size for layer parallelism. Default is 8.')
    parser.add_argument('--model_path_prefix', type=str, default="./models/", metavar='STRING',
                        help='Model path prfix. Default is "./models".')
    parser.add_argument('-temperature', '--temperature', type=float, default=1.0, metavar='NUMBER',
                        help='temperature of penalty. Default is 1.0.')
    parser.add_argument('-request_batch_size', '--request_batch_size', type=int, default=8, metavar='NUMBER',
                        help='batch size (default: 8)')
    parser.add_argument('-request_input_len', '--request_input_len', type=int, default=8, metavar='NUMBER',
                        help='input length (default: 8)')
    parser.add_argument('-request_output_len', '--request_output_len', type=int, default=32, metavar='NUMBER',
                        help='output length (default: 32)')

    args = parser.parse_args()
    generate_gpt_config(vars(args))