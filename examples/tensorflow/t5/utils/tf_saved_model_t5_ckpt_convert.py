#! /usr/bin/env python3
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
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorflow_text

from datetime import datetime
from pathlib import Path

'''
TensorFlow's saved_model T5 -> FasterTransformer
'''

def write_config_file(save_dir, **kwargs):
    file_template = """
[encoder]
vocab_size = {vocab_size:d}
d_model = {d_model:d}
d_kv = {d_kv:d}
d_ff = {d_ff:d}
num_layers = {num_encoder_layers:d}
num_decoder_layers = {num_encoder_layers:d}
num_heads = {num_heads:d}
is_gated_act = {decoder_is_gated_act!s}
weight_data_type = {data_type}

[decoder]
vocab_size = {vocab_size:d}
d_model = {d_model:d}
d_kv = {d_kv:d}
d_ff = {d_ff:d}
num_layers = {num_decoder_layers:d}
num_decoder_layers = {num_decoder_layers:d}
num_heads = {num_heads:d}
is_gated_act = {encoder_is_gated_act!s}
weight_data_type = {data_type}
    """

    with open(Path(save_dir) / "config.ini", "w") as f:
        f.write(file_template.format(**kwargs))


def export_key_name(key):
    export_key = ""
    split = key.split("__")

    prefix = split[0]
    is_layer = len(split) > 1 and "layers" in split[1]

    if is_layer:
        block_id = split[1].split("_")[-1]
        name = ""

        if prefix == "encoder":
            if "attention__key" in key:
                name = "0.SelfAttention.k.weight"
            elif "attention__out" in key:
                name = "0.SelfAttention.o.weight"
            elif "attention__query" in key:
                name = "0.SelfAttention.q.weight"
            elif "attention__value" in key:
                name = "0.SelfAttention.v.weight"
            elif "pre_attention_layer_norm" in key:
                name = "0.layer_norm.weight"

            elif "mlp__wi_0" in key:
                name = "1.DenseReluDense.wi.weight"
            elif "mlp__wi_1" in key:
                name = "1.DenseReluDense.wi2.weight"
            elif "mlp__wo" in key:
                name = "1.DenseReluDense.wo.weight"
            elif "pre_mlp_layer_norm" in key:
                name = "1.layer_norm.weight"

        elif prefix == "decoder":
            if "self_attention__key" in key:
                name = "0.SelfAttention.k.weight"
            elif "self_attention__out" in key:
                name = "0.SelfAttention.o.weight"
            elif "self_attention__query" in key:
                name = "0.SelfAttention.q.weight"
            elif "self_attention__value" in key:
                name = "0.SelfAttention.v.weight"
            elif "pre_self_attention_layer_norm" in key:
                name = "0.layer_norm.weight"

            elif "encoder_decoder_attention__key" in key:
                name = "1.EncDecAttention.k.weight"
            elif "encoder_decoder_attention__out" in key:
                name = "1.EncDecAttention.o.weight"
            elif "encoder_decoder_attention__query" in key:
                name = "1.EncDecAttention.q.weight"
            elif "encoder_decoder_attention__value" in key:
                name = "1.EncDecAttention.v.weight"
            elif "pre_cross_attention_layer_norm" in key:
                name = "1.layer_norm.weight"

            elif "mlp__wi_0" in key:
                name = "2.DenseReluDense.wi.weight"
            elif "mlp__wi_1" in key:
                name = "2.DenseReluDense.wi2.weight"
            elif "mlp__wo" in key:
                name = "2.DenseReluDense.wo.weight"
            elif "pre_mlp_layer_norm" in key:
                name = "2.layer_norm.weight"

        export_key = f"{prefix}.block.{block_id}.layer.{name}"

    elif "decoder__relpos_bias" in key:
        export_key = "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    elif "decoder_norm" in key:
        export_key = "decoder.final_layer_norm.weight"

    elif "encoder__relpos_bias" in key:
        export_key = "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    elif "encoder_norm" in key:
        export_key = "encoder.final_layer_norm.weight"

    elif "logits_dense" in key:
        export_key = "lm_head.weight"
    elif "token_embedder" in key:
        export_key = "shared.weight"

    return export_key


def handle_layer(key, value, dtype, saved_dir, tensor_para):
    print(f"Handling {key} with shape {value.shape}")
    val = value.astype(dtype)
    factor = tensor_para

    if key.find("shared.weight") != -1:
        # shared weights, only need to convert the weights of rank 0
        saved_path = saved_dir / f"{key}.bin"
        val.tofile(saved_path.as_posix())

        saved_path = saved_dir / f"{key}_T.bin"
        val.T.tofile(saved_path.as_posix())
    elif key.find("lm_head.weight") != -1:
        # lm_head weights, only need to convert the weights of rank 0
        val = val.transpose(1, 0) # For lm_head, we use TN gemm to compute, so we don't need to transpose
        saved_path = saved_dir / f"{key}.bin"
        val.tofile(saved_path.as_posix())
        
    elif key.find("layer_norm.weight") != -1:
        # shared weights, only need to convert the weights of rank 0
        saved_path = saved_dir / f"{key}.bin"
        val.tofile(saved_path.as_posix())

    elif (
        key.find("SelfAttention.o.weight") != -1
        or key.find("EncDecAttention.o.weight") != -1 
        or key.find("DenseReluDense.wo.weight") != -1
        ):
        split_vals = np.split(val, factor, axis=0)
        for j in range(factor):
            saved_path = saved_dir / f"{key}.{j:d}.bin"
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
            saved_path = saved_dir / f"{key}.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())
    elif (
        key.find("DenseReluDense.wi.weight") != -1 
        or key.find("DenseReluDense.wi2.weight") != -1
        ):
        # For gated activation.
        split_vals = np.split(val, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir / f"{key}.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())
    elif key.find("relative_attention_bias") != -1:
        split_vals = np.split(val, factor, axis=0)
        for j in range(factor):
            saved_path = saved_dir / f"{key}.{j:d}.bin"
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


def fuse_decoder_qkv(model_dict, num_layers, dtype, saved_dir, tensor_para):
    factor = tensor_para

    for i in range(num_layers):
        shape = model_dict[f"decoder.block.{i}.layer.0.SelfAttention.q.weight"].T.shape
        qkv = np.concatenate([model_dict[f"decoder.block.{i}.layer.0.SelfAttention.q.weight"].T,
                              model_dict[f"decoder.block.{i}.layer.0.SelfAttention.k.weight"].T,
                              model_dict[f"decoder.block.{i}.layer.0.SelfAttention.v.weight"].T], axis=-1)

        qkv = qkv.reshape([shape[0], 3, shape[1]])
        qkv = qkv.astype(dtype)

        split_vals = np.split(qkv, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir / f"decoder.block.{i}.layer.0.SelfAttention.qkv.weight.{j}.bin"
            split_vals[j].tofile(saved_path.as_posix())
 

def convert_checkpoint(args):
    base_dir = Path(args.checkpoint_dir)

    model = tf.saved_model.load(base_dir)

    layers = {layer._handle_name: np.array(layer)
            for layer in model.signatures['serving_default'].variables}

    vocab_size = layers["decoder__logits_dense__kernel:0"].shape[1]
    d_model, d_ff = layers["encoder__layers_0__mlp__wi_0__kernel:0"].shape
    num_heads = layers["decoder__relpos_bias__rel_embedding:0"].shape[0]
    d_kv = d_model // num_heads

    decoder_is_gated_act = "decoder__layers_0__mlp_wi_1__kernel" in layers
    encoder_is_gated_act = "encoder__layers_0__mlp_wi_1__kernel" in layers

    num_decoder_layers = 0
    num_encoder_layers = 0

    for key in layers:
        layer = key.split("__")[1]
        num = int(layer.split("_")[-1]) if "layers" in layer else 0

        if "encoder" in key:
            num_encoder_layers = max(num, num_encoder_layers)
        elif "decoder" in key:
            num_decoder_layers = max(num, num_decoder_layers)

    tp_source = 1
    tp_target = args.tensor_parallelism
    print(f"Converting from {tp_source} to {tp_target} GPUs")

    save_dir = Path(args.save_dir) / f"{tp_target:d}-gpu"
    save_dir.mkdir(parents=True, exist_ok=True)

    layers_export = {export_key_name(layer[0]): layer[1] for layer in layers.items()}
    for item in layers_export.items():
        handle_layer(*item, np.float32, save_dir, 1)
    fuse_decoder_qkv(layers_export, num_decoder_layers, np.float32, save_dir, 1)

    write_config_file(args.save_dir + f"/{args.tensor_parallelism}-gpu",
            vocab_size=vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            d_kv=d_kv,
            num_heads=num_heads,
            num_decoder_layers=num_decoder_layers,
            decoder_is_gated_act=decoder_is_gated_act,
            num_encoder_layers=num_encoder_layers,
            encoder_is_gated_act=encoder_is_gated_act,
            data_type="fp32",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", metavar="checkpoint-dir",
            help="directory where resides the source model.")
    parser.add_argument("save_dir", metavar="save-dir",
            help="where to store the FT model")
    parser.add_argument("--tensor-parallelism", "-t", type=int, default=1,
            help="level of tensor parallelism used for inference")
    parser.add_argument("--jobs", "-j", type=int, default=None,
            help="how many processes to spawn for conversion (default: cpu_count)")
    args = parser.parse_args()

    start_time = datetime.now()
    convert_checkpoint(args)
    stop_time = datetime.now()
    run_time = (stop_time - start_time)
    print("[INFO] Spend {} (h:m:s) to convert the model".format(run_time))
