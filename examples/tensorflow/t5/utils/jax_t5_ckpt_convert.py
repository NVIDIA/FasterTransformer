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
import zarr as zr
import re

from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from string import Template

'''
TensorFlow's saved_model T5 -> FasterTransformer
'''

def write_config_file(save_dir, **kwargs):
    script_dir = Path(__file__).parent.resolve()
    with open(script_dir / "ul2_config.template") as f:
        config_template = Template(f.read())

    with open(Path(save_dir) / "config.ini", "w") as f:
        f.write(config_template.substitute(**kwargs))


def export_key_name(key):
    export_key = ""
    split = key.split(".")

    prefix = split[1]
    is_layer = "layers" in split[2]

    if is_layer:
        block_id = split[2].split("_")[-1]
        name = ""

        if prefix == "encoder":
            if "attention.key" in key:
                name = "0.SelfAttention.k.weight"
            elif "attention.out" in key:
                name = "0.SelfAttention.o.weight"
            elif "attention.query" in key:
                name = "0.SelfAttention.q.weight"
            elif "attention.value" in key:
                name = "0.SelfAttention.v.weight"
            elif "pre_attention_layer_norm" in key:
                name = "0.layer_norm.weight"

            elif "mlp.wi_0" in key:
                name = "1.DenseReluDense.wi.weight"
            elif "mlp.wi_1" in key:
                name = "1.DenseReluDense.wi2.weight"
            elif "mlp.wo" in key:
                name = "1.DenseReluDense.wo.weight"
            elif "pre_mlp_layer_norm" in key:
                name = "1.layer_norm.weight"

        elif prefix == "decoder":
            if "self_attention.key" in key:
                name = "0.SelfAttention.k.weight"
            elif "self_attention.out" in key:
                name = "0.SelfAttention.o.weight"
            elif "self_attention.query" in key:
                name = "0.SelfAttention.q.weight"
            elif "self_attention.value" in key:
                name = "0.SelfAttention.v.weight"
            elif "pre_self_attention_layer_norm" in key:
                name = "0.layer_norm.weight"

            elif "encoder_decoder_attention.key" in key:
                name = "1.EncDecAttention.k.weight"
            elif "encoder_decoder_attention.out" in key:
                name = "1.EncDecAttention.o.weight"
            elif "encoder_decoder_attention.query" in key:
                name = "1.EncDecAttention.q.weight"
            elif "encoder_decoder_attention.value" in key:
                name = "1.EncDecAttention.v.weight"
            elif "pre_cross_attention_layer_norm" in key:
                name = "1.layer_norm.weight"

            elif "mlp.wi_0" in key:
                name = "2.DenseReluDense.wi.weight"
            elif "mlp.wi_1" in key:
                name = "2.DenseReluDense.wi2.weight"
            elif "mlp.wo" in key:
                name = "2.DenseReluDense.wo.weight"
            elif "pre_mlp_layer_norm" in key:
                name = "2.layer_norm.weight"

        export_key = f"{prefix}.block.{block_id}.layer.{name}"

    elif "decoder.relpos_bias" in key:
        export_key = "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    elif "decoder_norm" in key:
        export_key = "decoder.final_layer_norm.weight"

    elif "encoder.relpos_bias" in key:
        export_key = "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    elif "encoder_norm" in key:
        export_key = "encoder.final_layer_norm.weight"

    elif "token_embedder" in key:
        export_key = "shared.weight"

    return export_key


def handle_layer(key, file_name, dtype, saved_dir, tensor_para):
    val = np.array(zr.load(file_name)).astype(dtype)
    print(f"Processing {key} with shape {val.shape}")
    factor = tensor_para

    if key.find("shared.weight") != -1:
        # shared weights, only need to convert the weights of rank 0
        saved_path = saved_dir / f"{key}.bin"
        val.T.tofile(saved_path.as_posix())

        saved_path = saved_dir / f"{key}_T.bin"
        val.tofile(saved_path.as_posix())
        val.tofile(saved_dir / "lm_head.weight.bin")
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


def fuse_decoder_qkv(model_dict, layer_num, dtype, saved_dir, tensor_para):
    print(f"Processing decoder qkv SelfAttention merge block {layer_num}")
    factor = tensor_para

    q = np.array(zr.load(model_dict[f"decoder.block.{layer_num}.layer.0.SelfAttention.q.weight"]))
    k = np.array(zr.load(model_dict[f"decoder.block.{layer_num}.layer.0.SelfAttention.k.weight"]))
    v = np.array(zr.load(model_dict[f"decoder.block.{layer_num}.layer.0.SelfAttention.v.weight"]))
    shape = q.shape
    qkv = np.concatenate([q, k, v], axis=-1)

    qkv = qkv.reshape([shape[0], 3, shape[1]])
    qkv = qkv.astype(dtype)

    split_vals = np.split(qkv, factor, axis=-1)
    for j in range(factor):
        saved_path = saved_dir / f"decoder.block.{layer_num}.layer.0.SelfAttention.qkv.weight.{j}.bin"
        split_vals[j].tofile(saved_path.as_posix())


def read_config(gin_file):
    with open(gin_file) as f:
        data = f.read()

    config = {}
    config["num_embeddings"] = int(re.search(r"NUM_EMBEDDINGS = (\d+)", data).group(1))
    config["embed_dim"] = int(re.search(r"EMBED_DIM = (\d+)", data).group(1))
    config["head_dim"] = int(re.search(r"HEAD_DIM = (\d+)", data).group(1))
    config["mlp_dim"] = int(re.search(r"MLP_DIM = (\d+)", data).group(1))
    config["num_decoder_layers"] = int(re.search(r"NUM_DECODER_LAYERS = (\d+)", data).group(1))
    config["num_encoder_layers"] = int(re.search(r"NUM_ENCODER_LAYERS = (\d+)", data).group(1))
    config["num_heads"] = int(re.search(r"NUM_HEADS = (\d+)", data).group(1))

    return config


def convert_checkpoint(args):
    base_dir = Path(args.checkpoint_dir)

    config = read_config(base_dir / "config.gin")
    print(config)

    checkpoint_dir = list(base_dir.glob("checkpoint_*"))[0]
    print(f"[INFO] Reading checkpoint dir {checkpoint_dir}")

    layers = {}
    for file in checkpoint_dir.iterdir():
        if not file.is_dir():
            continue
        weight_name = file.parts[-1]
        layers[weight_name] = str(file.resolve())

    tp_source = 1
    tp_target = args.tensor_parallelism
    save_dtype = np.float32
    print(f"Converting from {tp_source} to {tp_target} GPUs")

    save_dir = Path(args.save_dir) / f"{tp_target:d}-gpu"
    save_dir.mkdir(parents=True, exist_ok=True)

    layers_export = {export_key_name(layer[0]): layer[1] for layer in layers.items() if "target" in layer[0]}

    final_layernorm_key = "decoder.final_layer_norm.weight"
    if "decoder.final_layer_norm.weight" not in layers_export:
        print("[WARNING] Decoder final LayerNorm not found. Generate tensor of ones as a replacement.")
        np.ones(config["embed_dim"], dtype=save_dtype).tofile(str(save_dir / (final_layernorm_key + ".bin")))

    with Pool(processes=args.jobs) as pool:
        pool.starmap(handle_layer,
                ((*item, save_dtype, save_dir, tp_target) for item in layers_export.items()))
        pool.starmap(fuse_decoder_qkv,
                ((layers_export, i, save_dtype, save_dir, tp_target) for i in range(config["num_decoder_layers"])))

    write_config_file(args.save_dir + f"/{args.tensor_parallelism}-gpu",
            vocab_size=config["num_embeddings"],
            d_model=config["embed_dim"],
            d_ff=config["mlp_dim"],
            d_kv=config["head_dim"],
            num_heads=config["num_heads"],
            num_decoder_layers=config["num_decoder_layers"],
            num_encoder_layers=config["num_encoder_layers"],
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
