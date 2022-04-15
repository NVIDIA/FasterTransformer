# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
from datetime import datetime
from pathlib import Path

import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../../3rdparty/transformers/src/")

from transformers import T5ForConditionalGeneration # transformers-4.10.0-py3

import numpy as np
import torch  # pytype: disable=import-error

rename_mapping={"relative_attention_num_buckets":"relative_attention_num_buckets_or_max_pos_seq_len"}
new_configs={"structure":{"t5_with_bias":"0", "position_embedding_type":"0"}}

def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    else:
        assert False, f"Invalid weight data type {data_type}"

def fuse_decoder_qkv(model, factor, saved_dir, np_weight_data_type):
    model_dict = {}
    for name, param in model.named_parameters():
        if name.find("decoder") != -1 and name.find("SelfAttention") != -1:
            model_dict[name] = param
    
    for i in range(model.decoder.config.num_layers):
        shape = model_dict[f"decoder.block.{i}.layer.0.SelfAttention.q.weight"].T.shape
        qkv = torch.cat([model_dict[f"decoder.block.{i}.layer.0.SelfAttention.q.weight"].T,
                         model_dict[f"decoder.block.{i}.layer.0.SelfAttention.k.weight"].T,
                         model_dict[f"decoder.block.{i}.layer.0.SelfAttention.v.weight"].T], dim=-1)

        qkv = qkv.reshape([shape[0], 3, shape[1]])
        qkv = qkv.cpu().detach().numpy().astype(np_weight_data_type)
        
        split_vals = np.split(qkv, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir / f"decoder.block.{i}.layer.0.SelfAttention.qkv.weight.{j}.bin"
            split_vals[j].tofile(saved_path.as_posix())
 
def split_and_convert_process(key, val, factor, saved_dir, np_weight_data_type):
    val = val.T.cpu().detach().numpy().astype(np_weight_data_type)
    saved_key = key

    if key.find("shared.weight") != -1:
        # shared weights, only need to convert the weights of rank 0
        saved_path = saved_dir / f"{saved_key}.bin"
        val.tofile(saved_path.as_posix())
        
        saved_path = saved_dir / f"{saved_key}_T.bin"
        val.T.tofile(saved_path.as_posix())
    elif key.find("layer_norm.weight") != -1:
        # shared weights, only need to convert the weights of rank 0
        saved_path = saved_dir / f"{saved_key}.bin"
        val.tofile(saved_path.as_posix())

    elif (
        key.find("SelfAttention.o.weight") != -1
        or key.find("EncDecAttention.o.weight") != -1 
        or key.find("DenseReluDense.wo.weight") != -1
        ):
        split_vals = np.split(val, factor, axis=0)
        for j in range(factor):
            saved_path = saved_dir / f"{saved_key}.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())

    elif (
        key.find("DenseReluDense.wi.weight") != -1 
        or (key.find("encoder") != -1 and (
            key.find("SelfAttention.q.weight") != -1
            or key.find("SelfAttention.k.weight") != -1
            or key.find("SelfAttention.v.weight") != -1
            )
            )
        or key.find("EncDecAttention.q.weight") != -1 
        or key.find("EncDecAttention.k.weight") != -1 
        or key.find("EncDecAttention.v.weight") != -1 
        ):
        split_vals = np.split(val, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir / f"{saved_key}.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())
    elif key.find("relative_attention_bias") != -1:
        split_vals = np.split(val, factor, axis=0)
        for j in range(factor):
            saved_path = saved_dir / f"{saved_key}.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())
    elif (
        key.find("decoder") != -1 and 
        (
            key.find("SelfAttention.q.weight") != -1
            or key.find("SelfAttention.k.weight") != -1
            or key.find("SelfAttention.v.weight") != -1
        )
        ):
        pass
    else:
        print(f"[ERROR] cannot find key '{key}'")

def convert_checkpoint(args):
    saved_dir = Path(args.saved_dir) / f"{args.infer_gpu_num:d}-gpu"
    saved_dir.mkdir(parents=True, exist_ok=True)

    t5_model = T5ForConditionalGeneration.from_pretrained(args.in_file)
    config = configparser.ConfigParser()
    
    config["encoder"] = {}
    for key, val in t5_model.encoder.config.to_dict().items():
        config["encoder"][key] = f"{val}"
    config["encoder"]["weight_data_type"] = args.weight_data_type
    config["decoder"] = {}
    for key, val in t5_model.decoder.config.to_dict().items():
        config["decoder"][key] = f"{val}"
    config["decoder"]["weight_data_type"] = args.weight_data_type
    for key, val in rename_mapping.items():
        config['encoder'][val] = config['encoder'].pop(key)
        config['decoder'][val] = config['decoder'].pop(key)
    for key, val in new_configs.items():
        config[key] = {}
        for val_key, val_val in val.items():
            config[key][val_key] = val_val
    with open((saved_dir / f"config.ini").as_posix(), 'w') as configfile:
        config.write(configfile)
    np_weight_data_type = get_weight_data_type(args.weight_data_type)
    
    i_gpu_num = args.infer_gpu_num
    
    for name, param in t5_model.named_parameters():
        split_and_convert_process(name, param, i_gpu_num, saved_dir, np_weight_data_type)
    fuse_decoder_qkv(t5_model, i_gpu_num, saved_dir, np_weight_data_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-saved_dir", "-o", type=str, help="file name of output file", required=True)
    parser.add_argument("-in_file", "-i", type=str, help="file name of input checkpoint file", required=True)
    parser.add_argument("-infer_gpu_num", "-i_g", type=int, help="How many gpus for inference", required=True)
    parser.add_argument("-weight_data_type", type=str, default="fp32", choices=["fp32", "fp16"])
    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    print("========================================")

    start_time = datetime.now()    
    convert_checkpoint(args)
    stop_time = datetime.now()
    run_time = (stop_time - start_time)
    print("[INFO] Spend {} (h:m:s) to convert the model".format(run_time))
