# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

import os
from transformers import LlamaForCausalLM

# using numpy extension: https://github.com/GreenWaves-Technologies/bfloat16
# install the library with `pip install bfloat16`
from bfloat16 import bfloat16

def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    elif data_type == "bf16":
        return bfloat16
    else:
        assert False, f"Invalid weight data type {data_type}"


def split_and_convert_process(saved_dir, factor, key, val):
    if key.find("input_layernorm.weight") != -1 or key.find("post_attention_layernorm.weight") != -1:
        # shared weights, only need to convert the weights of rank 0
        saved_path = saved_dir + "/" + key + ".bin"
        val.tofile(saved_path)
    elif key.find("attention.dense.weight") != -1 or key.find("mlp.down_proj.weight") != -1:
        split_vals = np.split(val, factor, axis=0)
        for j in range(factor):
            saved_path = saved_dir + "/" + key + ".%d.bin" % j
            split_vals[j].tofile(saved_path)
    elif key.find("mlp.gate_proj.weight") != -1 or key.find("mlp.up_proj.weight") != -1:
        split_vals = np.split(val, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir + "/" + key + ".%d.bin" % j
            split_vals[j].tofile(saved_path)
    elif key.find("attention.query_key_value.weight") != -1:
        split_vals = np.split(val, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir + "/" + key + ".%d.bin" % j
            split_vals[j].tofile(saved_path)
    else:
        print("[ERROR] cannot find key '{}'".format(key))

def split_and_convert(args):
    saved_dir = args.saved_dir + "/%d-gpu/" % args.infer_gpu_num

    if(os.path.exists(saved_dir) == False):
        os.makedirs(saved_dir)

    t_gpu_num = args.trained_gpu_num
    i_gpu_num = args.infer_gpu_num
    assert(i_gpu_num % t_gpu_num == 0)

    factor = (int)(i_gpu_num / t_gpu_num)

    # load position_embedding from rank 0
    # model = torch.load(ckpt_name)
    model = LlamaForCausalLM.from_pretrained(args.in_file)
    hf_config = vars(model.config)
    print(f"hf_config: {hf_config}")

    print("named parameters:")
    for name, param in model.named_parameters():
        print(f"- {name}")

    hidden_size = hf_config["hidden_size"]
    head_num = hf_config["num_attention_heads"]
    head_size = hidden_size // head_num
    num_layers = hf_config["num_hidden_layers"]


    np_weight_data_type = get_weight_data_type(args.weight_data_type)

    try:
        model_name = args.model_name
        config = configparser.ConfigParser()
        config['llama'] = {}
        config['llama']['model_name'] = model_name
        config['llama']["head_num"] = str(head_num)
        config['llama']["size_per_head"] = str(head_size)
        config['llama']["inter_size"] = str(hf_config["intermediate_size"])
        config['llama']["num_layer"] = str(num_layers)
        config['llama']["rotary_embedding"] = str(head_size)
        config['llama']['layernorm_eps'] = str(hf_config["rms_norm_eps"])
        config['llama']["vocab_size"] = str(hf_config["vocab_size"])
        config['llama']["start_id"] = str(hf_config["bos_token_id"])
        config['llama']["end_id"] = str(hf_config["eos_token_id"])
        config['llama']["weight_data_type"] = args.weight_data_type

        with open((Path(saved_dir) / f"config.ini").as_posix(), 'w') as configfile:
            config.write(configfile)
    except Exception as e:
        print(f"Fail to save the config in config.ini.")
        print(e)

    param_to_weights = lambda param: param.detach().cpu().numpy().astype(np_weight_data_type)

    # layer-wise weights, example:
    #   - model.layers.0.self_attn.q_proj.weight
    #   - model.layers.0.self_attn.k_proj.weight
    #   - model.layers.0.self_attn.v_proj.weight
    #   - model.layers.0.self_attn.o_proj.weight
    #   - model.layers.0.mlp.gate_proj.weight
    #   - model.layers.0.mlp.down_proj.weight
    #   - model.layers.0.mlp.up_proj.weight
    #   - model.layers.0.input_layernorm.weight
    #   - model.layers.0.post_attention_layernorm.weight
    for l in range(num_layers):
        print(f"converting layer {l}")
        # first merge QKV into a single weight
        # concat direct to FT shape: [hidden_size, 3, head_num, head_size]
        # copied from huggingface_gptj_ckpt_convert.py
        qkv_weights = np.stack([
            param_to_weights(model.state_dict()[f'model.layers.{l}.self_attn.q_proj.weight']),
            param_to_weights(model.state_dict()[f'model.layers.{l}.self_attn.k_proj.weight']),
            param_to_weights(model.state_dict()[f'model.layers.{l}.self_attn.v_proj.weight']),
        ])
        qkv_weights = np.transpose(qkv_weights, (2, 0, 1))
        qkv_weights_base_name = f'model.layers.{l}.attention.query_key_value.weight'
        split_and_convert_process(saved_dir, factor, qkv_weights_base_name, qkv_weights)

        # attention dense
        o_weight = param_to_weights(model.state_dict()[f'model.layers.{l}.self_attn.o_proj.weight']).T
        o_weight_base_name = f'model.layers.{l}.attention.dense.weight'
        split_and_convert_process(saved_dir, factor, o_weight_base_name, o_weight)

        # MLP
        mlp_down_weight = param_to_weights(model.state_dict()[f'model.layers.{l}.mlp.down_proj.weight']).T
        mlp_down_base_name = f'model.layers.{l}.mlp.down_proj.weight'
        split_and_convert_process(saved_dir, factor, mlp_down_base_name, mlp_down_weight)

        mlp_gate_weight = param_to_weights(model.state_dict()[f'model.layers.{l}.mlp.gate_proj.weight']).T
        mlp_gate_base_name = f'model.layers.{l}.mlp.gate_proj.weight'
        split_and_convert_process(saved_dir, factor, mlp_gate_base_name, mlp_gate_weight)

        mlp_up_weight = param_to_weights(model.state_dict()[f'model.layers.{l}.mlp.up_proj.weight']).T
        mlp_up_base_name = f'model.layers.{l}.mlp.up_proj.weight'
        split_and_convert_process(saved_dir, factor, mlp_up_base_name, mlp_up_weight)

        # LayerNorm
        input_ln_weight = param_to_weights(model.state_dict()[f'model.layers.{l}.input_layernorm.weight'])
        input_ln_base_name = f'model.layers.{l}.input_layernorm.weight'
        split_and_convert_process(saved_dir, factor, input_ln_base_name, input_ln_weight)

        post_attn_ln_weight = param_to_weights(model.state_dict()[f'model.layers.{l}.post_attention_layernorm.weight'])
        post_attn_ln_base_name = f'model.layers.{l}.post_attention_layernorm.weight'
        split_and_convert_process(saved_dir, factor, post_attn_ln_base_name, post_attn_ln_weight)

        print(f"done layer {l}")


    # final common weights
    for name, param in model.named_parameters():
        if name == 'model.embed_tokens.weight':
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(saved_dir + "model.wte.weight.bin")
        elif name == 'model.norm.weight':
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(saved_dir + "model.final_layernorm.weight.bin")
        elif name == 'lm_head.weight':
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(saved_dir + "model.lm_head.weight.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-saved_dir', '-o', type=str, help='file name of output file', required=True)
    parser.add_argument('-in_file', '-i', type=str, help='file name of input checkpoint file', required=True)
    parser.add_argument('-trained_gpu_num', '-t_g', type=int, help='How many gpus for inference', default=1)
    parser.add_argument('-infer_gpu_num', '-i_g', type=int, help='How many gpus for inference', required=True)
    parser.add_argument("-weight_data_type", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument('-model_name', '-m_n', type=str, help='model name', required=True)

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print("{}: {}".format(key, vars(args)[key]))
    print("========================================")

    split_and_convert(args)
