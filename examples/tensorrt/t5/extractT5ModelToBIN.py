# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import os
import configparser
import numpy as np
import torch
from transformers import T5ForConditionalGeneration
from pathlib import Path

rename_mapping={"relative_attention_num_buckets":"relative_attention_num_buckets_or_max_pos_seq_len"}
new_configs={"structure":{"t5_with_bias":"false", "use_gated_activation":"false", "position_embedding_type":"relative"}}

def fuse_decoder_qkv(model, factor, saved_dir):
    model_dict = {}
    for name, param in model.named_parameters():
        if name.find("decoder") != -1 and name.find("SelfAttention") != -1:
            model_dict[name] = param

    for i in range(model.decoder.config.num_layers):
        shape = model_dict[f"decoder.block.{i}.layer.0.SelfAttention.q.weight"].transpose(1, 0).shape
        qkv = torch.cat([model_dict[f"decoder.block.{i}.layer.0.SelfAttention.q.weight"].transpose(1, 0),
                         model_dict[f"decoder.block.{i}.layer.0.SelfAttention.k.weight"].transpose(1, 0),
                         model_dict[f"decoder.block.{i}.layer.0.SelfAttention.v.weight"].transpose(1, 0)], dim=-1)

        qkv = qkv.reshape([shape[0], 3, shape[1]])
        qkv = qkv.float().cpu().detach().numpy()

        split_vals = np.split(qkv, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir / f"decoder.block.{i}.layer.0.SelfAttention.qkv.weight.{j}.bin"
            split_vals[j].tofile(saved_path)

def split_and_convert_process(key, val, factor, saved_dir):
    if val.dim() == 2:
        val = val.transpose(1, 0)
    val = val.detach().numpy()
    saved_key = key

    if key.find("shared.weight") != -1:
        # shared weights, only need to convert the weights of rank 0
        saved_path = saved_dir / f"{saved_key}.bin"
        val.tofile(saved_path)

        saved_path = saved_dir / f"{saved_key}_T.bin"
        val.transpose(1, 0).tofile(saved_path)
    elif key.find("layer_norm.weight") != -1:
        # shared weights, only need to convert the weights of rank 0
        saved_path = saved_dir / f"{saved_key}.bin"
        val.tofile(saved_path)

    elif (
        key.find("SelfAttention.o.weight") != -1
        or key.find("EncDecAttention.o.weight") != -1
        or key.find("DenseReluDense.wo.weight") != -1
        ):
        split_vals = np.split(val, factor, axis=0)
        for j in range(factor):
            saved_path = saved_dir / f"{saved_key}.{j:d}.bin"
            split_vals[j].tofile(saved_path)

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
            split_vals[j].tofile(saved_path)
    elif key.find("relative_attention_bias") != -1:
        split_vals = np.split(val, factor, axis=0)
        for j in range(factor):
            saved_path = saved_dir / f"{saved_key}.{j:d}.bin"
            split_vals[j].tofile(saved_path)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-saved_dir", "-o", type=str, help="file name of output file", required=True)
    parser.add_argument("-in_file", "-i", type=str, help="file name of input checkpoint file. Using model name like 't5-small' is also ok.", required=True)
    args = parser.parse_args()

    saved_dir = Path(args.saved_dir) / f"1-gpu"
    saved_dir.mkdir(parents=True, exist_ok=True)

    t5_model = T5ForConditionalGeneration.from_pretrained(args.in_file)
    config = configparser.ConfigParser()
    
    config["encoder"] = {}
    for key, val in t5_model.encoder.config.to_dict().items():
        config["encoder"][key] = f"{val}"
    config["encoder"]["weight_data_type"] = "fp32"
    config["decoder"] = {}
    for key, val in t5_model.decoder.config.to_dict().items():
        config["decoder"][key] = f"{val}"
    config["decoder"]["weight_data_type"] = "fp32"
    for key, val in rename_mapping.items():
        config['encoder'][val] = config['encoder'].pop(key)
        config['decoder'][val] = config['decoder'].pop(key)
    for key, val in new_configs.items():
        config[key] = {}
        for val_key, val_val in val.items():
            config[key][val_key] = val_val
    with open(f"{saved_dir}/config.ini", 'w') as configfile:
        config.write(configfile)
    for name, param in t5_model.named_parameters():
        split_and_convert_process(name, param, 1, saved_dir)
    fuse_decoder_qkv(t5_model, 1, saved_dir)

    print("extract T5 model weight finish!")

