#! /usr/bin/env python3
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
import multiprocessing
import numpy as np
import torch  # pytype: disable=import-error
import yaml

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import List

'''
GPT-NeoX 20B model
Download by wget --cut-dirs=5 -nH -r --no-parent --reject "index.html*" https://mystic.the-eye.eu/public/AI/models/GPT-NeoX-20B/slim_weights/ -P 20B_checkpoints

layer_00-model_00-model_states.pt
    word_embeddings.weight: embedding table, split by tensor parallelism
layer_02-model_00-model_states.pt ~ layer_45-model_01-model_states.pt: 
    input_layernorm.weight
    input_layernorm.bias
    attention.query_key_value.weight
    attention.query_key_value.bias
    attention.rotary_emb.inv_freq
    attention.dense.weight
    attention.dense.bias
    post_attention_layernorm.weight
    post_attention_layernorm.bias
    mlp.dense_h_to_4h.weight
    mlp.dense_h_to_4h.bias
    mlp.dense_4h_to_h.weight
    mlp.dense_4h_to_h.bias

layer_47-model_00-model_states.pt:
    finally layernorm. model_00 and model_01 have same weights. Using one of them is enough.
layer_48-model_00-model_states.pt
    final_linear.weight. It should be the logit gemm weight.

mp_rank_xx_model_states.pt:
    some training states, useless in inference
'''

weights_skip_tensor_split = ["input_layernorm.bias",
                             "input_layernorm.weight",
                             "attention.dense.bias",
                             "mlp.dense_4h_to_h.bias",
                             "post_attention_layernorm.bias",
                             "post_attention_layernorm.weight"]

def write_config_file(save_dir):
    file_template = """
[gptneox]
model_name=gptneox_20B
head_num=64
size_per_head=96
vocab_size=50432
num_layer=44
rotary_embedding=24
start_id=0
end_id=2
inter_size=24576
use_gptj_residual=1
weight_data_type=fp32
    """

    with open(Path(save_dir) / "config.ini", "w") as f:
        f.write(file_template)

@dataclass
class KeyHandler:
    outname: str
    gather: str = ""
    scatter: str = "copy"
    reshape: List = field(default_factory=lambda: [])
    transpose: List = field(default_factory=lambda: [])


def on_cpu(storage, loc):
    return storage.cpu()


def handle_layer(chkpt_dir, in_filename, key_mapping, save_dir,
        in_range, out_range, whole_range=None):

    def read_layers(filename, range):
        if range is not None:
            filename = [filename.format(i) for i in range]
        else:
            filename = [filename]
        return [torch.load(chkpt_dir / fn, map_location=on_cpu) for fn in filename]

    layers = read_layers(in_filename, in_range)
    layer_keys = set(layers[0].keys())

    for key, value in key_mapping.items():
        key_templ, gather, scatter = value.outname, value.gather, value.scatter
        reshape, transpose = value.reshape, value.transpose
        layer_keys.remove(key)
        if key_templ == "":
            continue

        # Preprocess tensors
        tensors = [np.array(layer[key], dtype=np.float32) for layer in layers]
        if reshape:
            tensors = [ten.reshape(reshape) for ten in tensors]
        if transpose:
            tensors = [ten.transpose(transpose) for ten in tensors]

        # Gather tensors
        if len(tensors) == 1:
            gather_tensor = tensors[0]
        else:
            if "join" in gather:
                axis = int(gather.partition("_")[2])
                gather_tensor = np.concatenate(tensors, axis=axis)
            elif gather == "mean":
                gather_tensor = np.sum(tensors, axis=0) / len(tensors)
            elif gather == "sum":
                gather_tensor = np.sum(tensors, axis=0)
            else:
                raise NotImplementedError(f"Gather strategy {gather} is not supported")

        # Scatter tensors
        if len(out_range) == 1:
            scatter_tensors = [gather_tensor]
        else:
            if scatter == "copy":
                scatter_tensors = [gather_tensor for i in out_range]
            elif "split" in scatter:
                axis = int(scatter.partition("_")[2])
                if gather_tensor.shape[axis] % out_range != 0:
                    raise ValueError(f"{key} cannot be divided in {len(out_range)} along axis {axis}")

                scatter_tensors = np.split(gather_tensor, len(out_range), axis=axis)
            elif scatter == "divide":
                scatter_tensors = [gather_tensor / len(out_range) for i in out_range]
            else:
                raise NotImplementedError(f"Scatter strategy {scatter} is not supported")

        for tensor, idx in zip(scatter_tensors, out_range):
            output_name = key_templ.format(out_range[0])
            for weight_name in weights_skip_tensor_split:
                if weight_name in output_name:
                    output_name = output_name.split('.')
                    del output_name[-1]
                    output_name = '.'.join(output_name)
            tensor.tofile(save_dir / ("model." + output_name + ".bin"))

    if len(layer_keys) > 0:
        print("[Warning] Remaining keys:", layer_keys)


def convert_checkpoint(args):
    base_dir = Path(args.checkpoint_dir)

    with open(base_dir / "latest") as f:
        chkpt_dir = f.readline().rstrip()
    chkpt_dir = base_dir / chkpt_dir

    with open(base_dir / "configs/20B.yml") as f:
        model_args = yaml.safe_load(f)

    hidden_dim = model_args["hidden-size"]
    n_layers = model_args["num-layers"]
    n_heads = model_args["num-attention-heads"]
    hidden_per_head = hidden_dim // n_heads

    tp_source = model_args["model-parallel-size"]
    tp_target = args.tensor_parallelism
    print(f"Converting from {tp_source} to {tp_target} GPUs")

    save_dir = Path(args.save_dir) / f"{tp_target:d}-gpu"
    save_dir.mkdir(parents=True, exist_ok=True)

    handle_layer_args = []
    handle_layer_args.append((
            chkpt_dir,
            "layer_00-model_{:02d}-model_states.pt",
            {"word_embeddings.weight": KeyHandler("wte", "join_0")},
            save_dir,
            range(tp_source),
            [0],
    ))
    handle_layer_args.append((
            chkpt_dir,
            "layer_47-model_{:02d}-model_states.pt",
            {
                "norm.weight": KeyHandler("final_layernorm.weight", "mean"),
                "norm.bias":   KeyHandler("final_layernorm.bias", "mean"),
            },
            save_dir,
            range(tp_source),
            [0],
    ))
    handle_layer_args.append((
            chkpt_dir,
            "layer_48-model_{:02d}-model_states.pt",
            {
                "final_linear.weight": KeyHandler("lm_head.weight", "join_0"),
            },
            save_dir,
            range(tp_source),
            [0],
    ))

    gcd = np.gcd(tp_source, tp_target)
    print(f"Strategy: group {tp_source//gcd} source gpu(s) into {tp_target//gcd} out gpu(s).\n")

    in_indices = np.split(np.arange(tp_source), gcd)
    out_indices = np.split(np.arange(tp_target), gcd)

    for layer_id in range(model_args["num-layers"]):
        for in_idx, out_idx in zip(in_indices, out_indices):
            def make_fn_out(fn):
                return f"layers.{layer_id}." + fn + ".{:d}"

            handle_layer_args.append((
                    chkpt_dir,
                    f"layer_{layer_id+2:02d}" + "-model_{:02d}-model_states.pt",
                    {
                        "attention.rotary_emb.inv_freq": KeyHandler(""),
                        "attention.dense.weight": KeyHandler(
                            make_fn_out("attention.dense.weight"),
                            "join_0", "split_0",
                            transpose=[1, 0]),
                        "attention.dense.bias": KeyHandler(
                            make_fn_out("attention.dense.bias"), "sum", "divide"),
                        "attention.query_key_value.weight": KeyHandler(
                            make_fn_out("attention.query_key_value.weight"),
                            "join_2", "split_2",
                            reshape=[n_heads // tp_source, 3, hidden_per_head, hidden_dim],
                            transpose=[3, 1, 0, 2]),
                        "attention.query_key_value.bias": KeyHandler(
                            make_fn_out("attention.query_key_value.bias"),
                            "join_1", "split_1",
                            reshape=[n_heads // tp_source, 3, hidden_per_head],
                            transpose=[1, 0, 2]),
                        "input_layernorm.weight": KeyHandler(
                            make_fn_out("input_layernorm.weight"), "mean"),
                        "input_layernorm.bias": KeyHandler(
                            make_fn_out("input_layernorm.bias"), "mean"),
                        "mlp.dense_4h_to_h.weight": KeyHandler(
                            make_fn_out("mlp.dense_4h_to_h.weight"),
                            "join_0", "split_0",
                            transpose=[1, 0]),
                        "mlp.dense_4h_to_h.bias": KeyHandler(
                            make_fn_out("mlp.dense_4h_to_h.bias"), "sum", "divide"),
                        "mlp.dense_h_to_4h.weight": KeyHandler(
                            make_fn_out("mlp.dense_h_to_4h.weight"),
                            "join_1", "split_1",
                            transpose=[1, 0]),
                        "mlp.dense_h_to_4h.bias": KeyHandler(
                            make_fn_out("mlp.dense_h_to_4h.bias"), "join_0", "split_0"),
                        "post_attention_layernorm.weight": KeyHandler(
                            make_fn_out("post_attention_layernorm.weight"), "mean"),
                        "post_attention_layernorm.bias": KeyHandler(
                            make_fn_out("post_attention_layernorm.bias"), "mean"),
                    },
                    save_dir,
                    in_idx,
                    out_idx,
            ))

    torch.multiprocessing.set_start_method("spawn")
    with multiprocessing.Pool(args.jobs) as pool:
        pool.starmap(handle_layer, handle_layer_args)

    # Post-process biases and lm_head (TODO: remove this)
    for layer_idx in range(model_args["num-layers"]):
        attn_bias = np.fromfile(save_dir / f"model.layers.{layer_idx}.attention.dense.bias.bin", dtype=np.float32)
        mlp_bias =  np.fromfile(save_dir / f"model.layers.{layer_idx}.mlp.dense_4h_to_h.bias.bin", dtype=np.float32)

        (attn_bias + mlp_bias).tofile(save_dir / f"model.layers.{layer_idx}.mlp.attention.bias.sum.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", metavar="checkpoint-dir",
            help="directory where resides the source model. Must contain a \"latest\" file.")
    parser.add_argument("save_dir", metavar="save-dir",
            help="where to store the FT model")
    parser.add_argument("--tensor-parallelism", "-t", type=int, default=1,
            help="level of tensor parallelism used for inference")
    parser.add_argument("--jobs", "-j", type=int, default=None,
            help="how many processes to spawn for conversion (default: cpu_count)")
    args = parser.parse_args()

    start_time = datetime.now()
    convert_checkpoint(args)
    write_config_file(args.save_dir + f"/{args.tensor_parallelism}-gpu")
    stop_time = datetime.now()
    run_time = (stop_time - start_time)
    print("[INFO] Spend {} (h:m:s) to convert the model".format(run_time))
