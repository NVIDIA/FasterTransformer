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


import argparse
import configparser
import numpy as np
from pathlib import Path
import torch 

import os
# import sys
# dir_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(dir_path + "/../../../..")

from transformers import BertModel

def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    else:
        assert False, f"Invalid weight data type {data_type}"

def split_and_convert(args):
    assert args.infer_gpu_num == 1, "only support args.infer_gpu_num == 1 now"

    saved_dir = args.saved_dir + "/%d-gpu/" % args.infer_gpu_num

    if(os.path.exists(saved_dir) == False):
        os.makedirs(saved_dir)
    ckpt_name = args.in_file

    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bert_model = BertModel.from_pretrained(args.in_file).to(torch_device)
    
    try:
        config = configparser.ConfigParser()
        config["bert"] = {}
        for key in vars(args):
            config["bert"][key] = f"{vars(args)[key]}"
        for k, v in vars(bert_model.config).items():
            config["bert"][k] = f"{v}"
        config["bert"]["weight_data_type"] = args.weight_data_type
        with open((Path(saved_dir) / f"config.ini").as_posix(), 'w') as configfile:
            config.write(configfile)
    except Exception as e:
        print(f"Fail to save the config in config.ini. due to {e}")
    np_weight_data_type = get_weight_data_type(args.weight_data_type)

    '''
    huggingface_model_name_pattern = [
        "attention.output.add_local_input_quantizer._amax",
        "attention.output.add_residual_input_quantizer._amax",
        "attention.output.dense.bias",
        "attention.output.dense._input_quantizer._amax",
        "attention.output.dense.weight",
        "attention.output.dense._weight_quantizer._amax",
        "attention.output.LayerNorm.bias",
        "attention.output.LayerNorm.weight",
        "attention.self.key.bias",
        "attention.self.key._input_quantizer._amax",
        "attention.self.key.weight",
        "attention.self.key._weight_quantizer._amax",
        "attention.self.matmul_a_input_quantizer._amax",
        "attention.self.matmul_k_input_quantizer._amax",
        "attention.self.matmul_q_input_quantizer._amax",
        "attention.self.matmul_v_input_quantizer._amax",
        "attention.self.query.bias",
        "attention.self.query._input_quantizer._amax",
        "attention.self.query.weight",
        "attention.self.query._weight_quantizer._amax",
        "attention.self.value.bias",
        "attention.self.value._input_quantizer._amax",
        "attention.self.value.weight",
        "attention.self.value._weight_quantizer._amax",
        "intermediate.dense.bias",
        "intermediate.dense._input_quantizer._amax",
        "intermediate.dense.weight",
        "intermediate.dense._weight_quantizer._amax",
        "intermediate.intermediate_act_fn_input_quantizer._amax",
        "output.add_local_input_quantizer._amax",
        "output.add_residual_input_quantizer._amax",
        "output.dense.bias",
        "output.dense._input_quantizer._amax",
        "output.dense.weight",
        "output.dense._weight_quantizer._amax",
        "output.LayerNorm.bias",
        "output.LayerNorm.weight",
    ]
    '''

    model = {}
    for key, param in bert_model.named_parameters():
        model[key] = param

    for key in model:
        if key == "bert.embeddings.word_embeddings.weight" or \
                key == "bert.embeddings.position_embeddings.weight" or \
                key == "bert.embeddings.token_type_embeddings.weight":
            weight = model[key]
            weight.detach().cpu().numpy().astype(np_weight_data_type).tofile(f"{saved_dir}/bert.{key}.bin")
            print(f"convert {key}")
        elif key.find("self.query") == -1 and key.find("self.key") == -1 and key.find("self.value") == -1:
            #  If not query, key and values, we don't do concat or other operations. Convert them directly.
            weight = model[key]
            if weight.dim() == 2:
                weight = weight.transpose(1, 0)
            weight.detach().cpu().numpy().astype(np_weight_data_type).tofile(f"{saved_dir}/bert.{key}.bin")
            print(f"convert {key}")
        elif key.find("self.query.bias") != -1:
            q_name = key
            k_name = key.replace("query", "key")
            v_name = key.replace("query", "value")
            q_bias = model[q_name]
            k_bias = model[k_name]
            v_bias = model[v_name]
            qkv_bias = torch.cat([q_bias, k_bias, v_bias])
            new_name = key.replace("query", "query_key_value")
            qkv_bias.detach().cpu().numpy().astype(np_weight_data_type).tofile(f"{saved_dir}/bert.{new_name}.bin")
            print(f"convert {new_name}")
        elif key.find("self.query._input_quantizer") != -1:
            new_name = key.replace("query", "query_key_value")
            model[key].detach().cpu().numpy().astype(np_weight_data_type).tofile(f"{saved_dir}/bert.{new_name}.bin")
            print(f"convert {new_name}")
        elif key.find("self.query.weight") != -1:
            q_name = key
            k_name = key.replace("query", "key")
            v_name = key.replace("query", "value")
            q_weight = model[q_name].transpose(1, 0)
            k_weight = model[k_name].transpose(1, 0)
            v_weight = model[v_name].transpose(1, 0)
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], axis=-1)
            new_name = key.replace("query", "query_key_value")
            qkv_weight.detach().cpu().numpy().astype(np_weight_data_type).tofile(f"{saved_dir}/bert.{new_name}.bin")
            print(f"convert {new_name}")
        elif key.find("self.query._weight_quantizer") != -1:
            # PER CHANNEL
            '''
            q_name = key
            k_name = key.replace("query", "key")
            v_name = key.replace("query", "value")
            q_quantizer = model[q_name]
            k_quantizer = model[k_name]
            v_quantizer = model[v_name]
            qkv_quantizer = torch.cat([q_quantizer, k_quantizer, v_quantizer])
            new_name = key.replace("query", "query_key_value")
            qkv_quantizer.detach().cpu().numpy().astype(np_weight_data_type).tofile(f"{saved_dir}/{new_name}.bin")
            # print(f"name: {new_name}, {qkv_quantizer.shape}")
            '''

            # PER TENSOR, the checkpoint has separate q, k, v quantizers, need to choose max one
            q_name = key
            k_name = key.replace("query", "key")
            v_name = key.replace("query", "value")
            q_quantizer = model[q_name].view(1)
            k_quantizer = model[k_name].view(1)
            v_quantizer = model[v_name].view(1)
            qkv_quantizer = torch.max(torch.cat([q_quantizer, k_quantizer, v_quantizer]))
            new_name = key.replace("query", "query_key_value")
            qkv_quantizer.detach().cpu().numpy().astype(np_weight_data_type).tofile(f"{saved_dir}/bert.{new_name}.bin")
            print(f"convert {new_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-saved_dir', '-o', type=str, help='file name of output file', required=True)
    parser.add_argument('-in_file', '-i', type=str, help='file name of input checkpoint file', required=True)
    # parser.add_argument('-trained_gpu_num', '-t_g', type=int, help='How many gpus for inference', default=1)
    parser.add_argument('-infer_gpu_num', '-i_g', type=int, help='How many gpus for inference', default=1)
    # parser.add_argument("-processes", "-p", type=int, help="How many processes to spawn for conversion (default: 4)", default=4)
    parser.add_argument("-weight_data_type", type=str, default="fp32", choices=["fp32", "fp16"])

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print("{}: {}".format(key, vars(args)[key]))
    print("========================================")

    split_and_convert(args)
