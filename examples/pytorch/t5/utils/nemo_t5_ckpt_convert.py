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
import concurrent.futures
import configparser
import datetime
import logging
import pathlib
import shutil
import sys
import tempfile
import typing

import numpy as np
import torch  # pytype: disable=import-error

# verify if root package is in PYTHONPATH
__root_package_path__ = pathlib.Path(__file__).parent.parent.parent.parent.parent.absolute().as_posix()
if __root_package_path__ not in sys.path:
    print(
        f"[ERROR] add project root directory to your PYTHONPATH with "
        f"'export PYTHONPATH={__root_package_path__}:${{PYTHONPATH}}'"
    )

from examples.pytorch.nemo import unpack_nemo_ckpt, UnpackedNemoCheckpointDir, extract_layers_with_prefix
from examples.pytorch.utils import gpu_map_location, WEIGHT2DTYPE, torch2np, cpu_map_location, safe_transpose


LOGGER = logging.getLogger(__name__)


shared_mapping = {
    "wte": "shared.weight",
    "wte_T": "shared.weight_T",
    "ape": "shared.ape",
    "encoder_rpe": "block.0.layer.0.SelfAttention.relative_attention_bias",
    "decoder_rpe": "block.0.layer.0.SelfAttention.relative_attention_bias",
}

encoder_mapping = {
    "input_layernorm": "layer.0.layer_norm",
    "self_attention.query_key_value": ["layer.0.SelfAttention.q", "layer.0.SelfAttention.k", "layer.0.SelfAttention.v"],
    "self_attention.dense": "layer.0.SelfAttention.o",
    "post_attention_layernorm": "layer.1.layer_norm",
    "mlp.dense_h_to_4h": "layer.1.DenseReluDense.wi",
    "mlp.dense_h_to_4h_2": "layer.1.DenseReluDense.wi2",  ## gated activation
    "mlp.dense_4h_to_h": "layer.1.DenseReluDense.wo",
    "final_layernorm": "final_layer_norm",
}

decoder_mapping = {
    "input_layernorm": "layer.0.layer_norm",
    "self_attention.query_key_value": ["layer.0.SelfAttention.qkv"],
    "self_attention.dense": "layer.0.SelfAttention.o",
    "post_attention_layernorm": "layer.1.layer_norm",
    "inter_attention.query": ["layer.1.EncDecAttention.q"],
    "inter_attention.key_value": ["layer.1.EncDecAttention.k", "layer.1.EncDecAttention.v"],
    "inter_attention.dense": "layer.1.EncDecAttention.o",
    "post_inter_attention_layernorm": "layer.2.layer_norm",
    "mlp.dense_h_to_4h": "layer.2.DenseReluDense.wi",
    "mlp.dense_h_to_4h_2": "layer.2.DenseReluDense.wi2",
    "mlp.dense_4h_to_h": "layer.2.DenseReluDense.wo",
    "final_layernorm": "final_layer_norm",
}

megatron_HF_name_mapping = {"shared": shared_mapping, "encoder": encoder_mapping, "decoder": decoder_mapping}

encoder_config_mapping = {
    "num_attention_heads": "num_heads",
    "hidden_size": "d_model",
    "kv_channels": "d_kv",
    "ffn_hidden_size": "d_ff",
    "num_layers": "num_layers",
    "max_position_embeddings": "relative_attention_num_buckets_or_max_pos_seq_len",
    "relative_position_num_buckets": "relative_attention_num_buckets_or_max_pos_seq_len",
    "activation": "feed_forward_proj",
}

decoder_config_mapping = {
    "num_attention_heads": "num_heads",
    "hidden_size": "d_model",
    "kv_channels": "d_kv",
    "ffn_hidden_size": "d_ff",
    "num_layers": "num_layers",
    "max_position_embeddings": "relative_attention_num_buckets_or_max_pos_seq_len",
    "relative_position_num_buckets": "relative_attention_num_buckets_or_max_pos_seq_len",
}


def megatron2hf_name(saved_key, name_mapping):
    saved_key = saved_key.replace("layers", "block")
    mapping_key = saved_key.rsplit(sep=".", maxsplit=1)[0]
    mapping_key_no_num = mapping_key[mapping_key.find(".", 6) + 1 :]
    block_num = mapping_key[: mapping_key.find(".", 6) + 1]
    weight_or_bias = saved_key.rsplit(sep=".", maxsplit=1)[1]

    mapping_vals_no_num = name_mapping[mapping_key_no_num]
    if not isinstance(mapping_vals_no_num, list):
        mapping_vals_no_num = [mapping_vals_no_num]

    saved_keys = [block_num + val + "." + weight_or_bias for val in mapping_vals_no_num]
    return saved_keys


# This tool is used to support the new megatron model trained by pipeline parallel + tensor parallel
def merge_and_convert_process(
    model_type,
    tensor_para_rank,
    pipeline_para_rank,
    saved_dir,
    factor,
    key,
    nemo_model_config,
    models_list,
    np_weight_data_type,
):
    assert model_type == "encoder" or model_type == "decoder"
    prefix = model_type
    pipeline_para_size = nemo_model_config["pipeline_model_parallel_size"]
    pipeline_model_parallel_split_rank = nemo_model_config.get("pipeline_model_parallel_split_rank", 0)
    num_layers = nemo_model_config["num_layers"]

    major_device = models_list[0][key].device

    name_mapping = megatron_HF_name_mapping[model_type]
    saved_dir = pathlib.Path(saved_dir)
    if key.find("layers.") != -1:
        layer_index = int(key[7 : key.find(".", 7)])
        encoder_num_pipeline_stages = pipeline_model_parallel_split_rank
        decoder_num_pipeline_stages = pipeline_para_size - pipeline_model_parallel_split_rank
        offset = 0
        if model_type == "encoder" and pipeline_para_size > 1:
            offset = pipeline_para_rank * (num_layers // encoder_num_pipeline_stages)
        elif model_type == "decoder" and pipeline_para_size > 1:
            offset = (pipeline_para_rank - pipeline_model_parallel_split_rank) * (
                num_layers // decoder_num_pipeline_stages
            )
        saved_key = key.replace(f"layers.{layer_index}.", f"layers.{layer_index + offset}.")
    else:
        saved_key = key

    if (
        key.find("input_layernorm.weight") != -1
        or key.find("input_layernorm.bias") != -1
        or key.find("self_attention.dense.bias") != -1
        or key.find("inter_attention.dense.bias") != -1
        or key.find("post_attention_layernorm.weight") != -1
        or key.find("post_inter_attention_layernorm.weight") != -1
        or key.find("post_attention_layernorm.bias") != -1
        or key.find("post_inter_attention_layernorm.bias") != -1
        or key.find("mlp.dense_4h_to_h.bias") != -1
        or key.find("final_layernorm.weight") != -1
        or key.find("final_layernorm.bias") != -1
    ):
        # shared weights, only need to convert the weights of rank 0
        if tensor_para_rank == 0:
            saved_keys = megatron2hf_name(saved_key, name_mapping)
            saved_path = saved_dir / f"{prefix}.{saved_keys[0]}.bin"
            val = safe_transpose(models_list[0][key])
            val = torch2np(val, np_weight_data_type)
            val = np.squeeze(val)
            LOGGER.debug(
                "merge for pp_rank=%d tp_rank=%d only for tp_rank=0 src_key=%s filename=%s shape=%s dtype=%s",
                pipeline_para_rank,
                tensor_para_rank,
                key,
                saved_path.name,
                val.shape,
                val.dtype,
            )
            val.tofile(saved_path)

    elif (
        key.find("self_attention.dense.weight") != -1
        or key.find("inter_attention.dense.weight") != -1
        or key.find("mlp.dense_4h_to_h.weight") != -1
    ):
        vals = []
        for k in range(factor):
            val = safe_transpose(models_list[k][key])
            val = val.float().to(major_device)
            vals.append(val)
        saved_keys = megatron2hf_name(saved_key, name_mapping)
        saved_path = saved_dir / f"{prefix}.{saved_keys[0]}.{tensor_para_rank:d}.bin"
        val = torch.cat(vals, dim=0)
        val = torch2np(val, np_weight_data_type)
        LOGGER.debug(
            "merge for pp_rank=%d tp_rank=%d factor=%d src_key=%s filename=%s shape=%s dtype=%s",
            pipeline_para_rank,
            tensor_para_rank,
            factor,
            key,
            saved_path.name,
            val.shape,
            val.dtype,
        )
        val.tofile(saved_path)

    elif (
        key.find("mlp.dense_h_to_4h.weight") != -1
        or key.find("mlp.dense_h_to_4h.bias") != -1
        or key.find("mlp.dense_h_to_4h_2.weight") != -1
        or key.find("mlp.dense_h_to_4h_2.bias") != -1
    ):
        vals = []
        for k in range(factor):
            val = safe_transpose(models_list[k][key])
            val = val.float().to(major_device)
            vals.append(val)
        saved_keys = megatron2hf_name(saved_key, name_mapping)
        saved_path = saved_dir / f"{prefix}.{saved_keys[0]}.{tensor_para_rank:d}.bin"
        val = torch.cat(vals, dim=-1)
        val = torch2np(val, np_weight_data_type)
        LOGGER.debug(
            "merge for pp_rank=%d tp_rank=%d factor=%d src_key=%s filename=%s shape=%s dtype=%s",
            pipeline_para_rank,
            tensor_para_rank,
            factor,
            key,
            saved_path.name,
            val.shape,
            val.dtype,
        )
        val.tofile(saved_path)

    elif (
        key.find("self_attention.query_key_value.bias") != -1
        or key.find("inter_attention.query.bias") != -1
        or key.find("inter_attention.key_value.bias") != -1
    ):
        num_splits = 3
        if key.find("inter_attention.key_value.bias") != -1:
            num_splits = 2
        if key.find("inter_attention.query.bias") != -1:
            num_splits = 1

        vals = []
        for k in range(factor):
            val = safe_transpose(models_list[k][key])
            val = val.float()
            local_dim = int(val.shape[-1] / num_splits)
            head_num = nemo_model_config["num_attention_heads"] // nemo_model_config["tensor_model_parallel_size"]
            # t5 kv_channels may not be equal to hidden_size // head_num
            size_per_head = nemo_model_config["kv_channels"]
            val = val.reshape(head_num, num_splits, size_per_head)
            val = val.permute(1, 0, 2)
            val = val.reshape(num_splits, local_dim)
            vals.append(val.to(major_device))

        saved_vals = torch.cat(vals, dim=-1)
        saved_keys = megatron2hf_name(saved_key, name_mapping)
        if len(saved_keys) == 1:
            saved_path = saved_dir / f"{prefix}.{saved_keys[0]}.{tensor_para_rank:d}.bin"
            val = torch2np(saved_vals, np_weight_data_type)
            LOGGER.debug(
                "merge for pp_rank=%d tp_rank=%d factor=%d src_key=%s filename=%s shape=%s dtype=%s",
                pipeline_para_rank,
                tensor_para_rank,
                factor,
                key,
                saved_path.name,
                val.shape,
                val.dtype,
            )
            val.tofile(saved_path)
        else:
            for index in range(len(saved_keys)):
                saved_path = saved_dir / f"{prefix}.{saved_keys[index]}.{tensor_para_rank:d}.bin"
                val = torch2np(saved_vals[index, ...], np_weight_data_type)
                LOGGER.debug(
                    "merge for pp_rank=%d tp_rank=%d factor=%d src_key=%s filename=%s shape=%s dtype=%s",
                    pipeline_para_rank,
                    tensor_para_rank,
                    factor,
                    key,
                    saved_path.name,
                    val.shape,
                    val.dtype,
                )
                val.tofile(saved_path)

    elif (
        key.find("self_attention.query_key_value.weight") != -1
        or key.find("inter_attention.query.weight") != -1
        or key.find("inter_attention.key_value.weight") != -1
    ):
        num_splits = 3
        if key.find("inter_attention.key_value.weight") != -1:
            num_splits = 2
        if key.find("inter_attention.query.weight") != -1:
            num_splits = 1

        vals = []
        for k in range(factor):
            val = safe_transpose(models_list[k][key])
            val = val.float()
            hidden_dim = val.shape[0]
            local_dim = int(val.shape[-1] / num_splits)
            head_num = nemo_model_config["num_attention_heads"]
            # t5 kv_channels may not be equal to hidden_size // head_num
            size_per_head = nemo_model_config["kv_channels"]
            head_num = head_num // nemo_model_config["tensor_model_parallel_size"]
            val = val.reshape(hidden_dim, head_num, num_splits, size_per_head)
            val = val.permute(0, 2, 1, 3)
            val = val.reshape(hidden_dim, num_splits, local_dim)
            vals.append(val.to(major_device))

        saved_vals = torch.cat(vals, dim=-1)
        saved_keys = megatron2hf_name(saved_key, name_mapping)
        if len(saved_keys) == 1:
            saved_path = saved_dir / f"{prefix}.{saved_keys[0]}.{tensor_para_rank:d}.bin"
            val = torch2np(saved_vals, np_weight_data_type)
            LOGGER.debug(
                "merge for pp_rank=%d tp_rank=%d factor=%d src_key=%s filename=%s shape=%s dtype=%s",
                pipeline_para_rank,
                tensor_para_rank,
                factor,
                key,
                saved_path.name,
                val.shape,
                val.dtype,
            )
            val.tofile(saved_path)
        else:
            for index in range(len(saved_keys)):
                saved_path = saved_dir / f"{prefix}.{saved_keys[index]}.{tensor_para_rank:d}.bin"
                val = torch2np(saved_vals[:, index, ...], np_weight_data_type)
                LOGGER.debug(
                    "merge for pp_rank=%d tp_rank=%d factor=%d src_key=%s filename=%s shape=%s dtype=%s",
                    pipeline_para_rank,
                    tensor_para_rank,
                    factor,
                    key,
                    saved_path.name,
                    val.shape,
                    val.dtype,
                )
                val.tofile(saved_path)
    else:
        LOGGER.error(f"cannot find key '{key}'")


def split_and_convert_process(
    model_type,
    tensor_para_rank,
    pipeline_para_rank,
    saved_dir,
    factor,
    key,
    nemo_model_config,
    models_list,
    np_weight_data_type,
):
    assert model_type == "encoder" or model_type == "decoder"
    prefix = model_type
    num_layers = nemo_model_config["num_layers"]
    pipeline_para_size = nemo_model_config["pipeline_model_parallel_size"]
    pipeline_model_parallel_split_rank = nemo_model_config.get("pipeline_model_parallel_split_rank", 0)

    name_mapping = megatron_HF_name_mapping[model_type]
    val = safe_transpose(models_list[0][key])
    val = torch2np(val, np_weight_data_type)

    if key.find("layers.") != -1:
        layer_index = int(key[7 : key.find(".", 7)])
        encoder_num_pipeline_stages = pipeline_model_parallel_split_rank
        decoder_num_pipeline_stages = pipeline_para_size - pipeline_model_parallel_split_rank
        offset = 0
        if model_type == "encoder" and pipeline_para_size > 1:
            offset = pipeline_para_rank * (num_layers // encoder_num_pipeline_stages)
        elif model_type == "decoder" and pipeline_para_size > 1:
            offset = (pipeline_para_rank - pipeline_model_parallel_split_rank) * (
                num_layers // decoder_num_pipeline_stages
            )
        saved_key = key.replace(f"layers.{layer_index}.", f"layers.{layer_index + offset}.")
    else:
        saved_key = key

    if (
        key.find("input_layernorm.weight") != -1
        or key.find("input_layernorm.bias") != -1
        or key.find("self_attention.dense.bias") != -1
        or key.find("inter_attention.dense.bias") != -1
        or key.find("post_attention_layernorm.weight") != -1
        or key.find("post_inter_attention_layernorm.weight") != -1
        or key.find("post_attention_layernorm.bias") != -1
        or key.find("post_inter_attention_layernorm.bias") != -1
        or key.find("mlp.dense_4h_to_h.bias") != -1
        or key.find("final_layernorm.weight") != -1
        or key.find("final_layernorm.bias") != -1
    ):
        # shared weights, only need to convert the weights of rank 0
        if tensor_para_rank == 0:
            saved_keys = megatron2hf_name(saved_key, name_mapping)
            saved_path = saved_dir / f"{prefix}.{saved_keys[0]}.bin"
            LOGGER.debug(
                "split for pp_rank=%d tp_rank=%d only for tp_rank=0 src_key=%s filename=%s "
                "shape=%s (same as original) dtype=%s",
                pipeline_para_rank,
                tensor_para_rank,
                key,
                saved_path.name,
                val.shape,
                val.dtype,
            )
            val.tofile(saved_path)

    elif (
        key.find("self_attention.dense.weight") != -1
        or key.find("inter_attention.dense.weight") != -1
        or key.find("mlp.dense_4h_to_h.weight") != -1
    ):
        split_vals = np.split(val, factor, axis=0)
        saved_keys = megatron2hf_name(saved_key, name_mapping)
        for j in range(factor):
            saved_path = saved_dir / f"{prefix}.{saved_keys[0]}.{tensor_para_rank * factor + j:d}.bin"
            LOGGER.debug(
                "split for pp_rank=%d tp_rank=%d factor=%d src_key=%s filename=%s original_shape=%s shape=%s dtype=%s",
                pipeline_para_rank,
                tensor_para_rank,
                factor,
                key,
                saved_path.name,
                val.shape,
                split_vals[j].shape,
                split_vals[j].dtype,
            )
            split_vals[j].tofile(saved_path)

    elif (
        key.find("mlp.dense_h_to_4h.weight") != -1
        or key.find("mlp.dense_h_to_4h.bias") != -1
        or key.find("mlp.dense_h_to_4h_2.weight") != -1
        or key.find("mlp.dense_h_to_4h_2.bias") != -1
    ):
        split_vals = np.split(val, factor, axis=-1)
        saved_keys = megatron2hf_name(saved_key, name_mapping)
        for j in range(factor):
            saved_path = saved_dir / f"{prefix}.{saved_keys[0]}.{tensor_para_rank * factor + j:d}.bin"
            LOGGER.debug(
                "split for pp_rank=%d tp_rank=%d factor=%d src_key=%s filename=%s original_shape=%s shape=%s dtype=%s",
                pipeline_para_rank,
                tensor_para_rank,
                factor,
                key,
                saved_path.name,
                val.shape,
                split_vals[j].shape,
                split_vals[j].dtype,
            )
            split_vals[j].tofile(saved_path)

    elif (
        key.find("self_attention.query_key_value.bias") != -1
        or key.find("inter_attention.query.bias") != -1
        or key.find("inter_attention.key_value.bias") != -1
    ):
        num_splits = 3
        if key.find("inter_attention.key_value.bias") != -1:
            num_splits = 2
        if key.find("inter_attention.query.bias") != -1:
            num_splits = 1
        local_dim = int(val.shape[-1] / num_splits)
        head_num = nemo_model_config["num_attention_heads"] // nemo_model_config["tensor_model_parallel_size"]
        # t5 kv_channels may not be equal to hidden_size // head_num
        size_per_head = nemo_model_config["kv_channels"]
        val = val.reshape(head_num, num_splits, size_per_head)
        val = val.transpose(1, 0, 2)

        val = val.reshape(num_splits, local_dim)
        split_vals = np.split(val, factor, axis=-1)
        saved_keys = megatron2hf_name(saved_key, name_mapping)
        for j in range(factor):
            if len(saved_keys) == 1:
                saved_path = saved_dir / f"{prefix}.{saved_keys[0]}.{tensor_para_rank * factor + j:d}.bin"
                LOGGER.debug(
                    "split for pp_rank=%d tp_rank=%d factor=%d src_key=%s filename=%s "
                    "preprocessed_shape=%s shape=%s dtype=%s",
                    pipeline_para_rank,
                    tensor_para_rank,
                    factor,
                    key,
                    saved_path.name,
                    val.shape,
                    split_vals[j].shape,
                    split_vals[j].dtype,
                )
                split_vals[j].tofile(saved_path)
                continue
            for index in range(len(saved_keys)):
                saved_path = saved_dir / f"{prefix}.{saved_keys[index]}.{tensor_para_rank * factor + j:d}.bin"
                split_val_idxed = split_vals[j][index, ...]
                LOGGER.debug(
                    "split for pp_rank=%d tp_rank=%d factor=%d src_key=%s filename=%s "
                    "preprocessed_shape=%s shape=%s dtype=%s",
                    pipeline_para_rank,
                    tensor_para_rank,
                    factor,
                    key,
                    saved_path.name,
                    val.shape,
                    split_val_idxed.shape,
                    split_val_idxed.dtype,
                )
                split_val_idxed.tofile(saved_path)

    elif (
        key.find("self_attention.query_key_value.weight") != -1
        or key.find("inter_attention.query.weight") != -1
        or key.find("inter_attention.key_value.weight") != -1
    ):
        num_splits = 3
        if key.find("inter_attention.key_value.weight") != -1:
            num_splits = 2
        if key.find("inter_attention.query.weight") != -1:
            num_splits = 1
        hidden_dim = val.shape[0]
        local_dim = int(val.shape[-1] / num_splits)

        head_num = nemo_model_config["num_attention_heads"]
        # t5 kv_channels may not be equal to hidden_size // head_num
        size_per_head = nemo_model_config["kv_channels"]
        head_num = head_num // nemo_model_config["tensor_model_parallel_size"]
        val = val.reshape(hidden_dim, head_num, num_splits, size_per_head)
        val = val.transpose(0, 2, 1, 3)

        val = val.reshape(hidden_dim, num_splits, local_dim)
        split_vals = np.split(val, factor, axis=-1)

        saved_keys = megatron2hf_name(saved_key, name_mapping)
        for j in range(factor):
            if len(saved_keys) == 1:
                saved_path = saved_dir / f"{prefix}.{saved_keys[0]}.{tensor_para_rank * factor + j:d}.bin"
                LOGGER.debug(
                    "split for pp_rank=%d tp_rank=%d factor=%d src_key=%s filename=%s "
                    "preprocessed_shape=%s shape=%s dtype=%s",
                    pipeline_para_rank,
                    tensor_para_rank,
                    factor,
                    key,
                    saved_path.name,
                    val.shape,
                    split_vals[j].shape,
                    split_vals[j].dtype,
                )
                split_vals[j].tofile(saved_path)
                continue
            for index in range(len(saved_keys)):
                saved_path = saved_dir / f"{prefix}.{saved_keys[index]}.{tensor_para_rank * factor + j:d}.bin"
                split_val_idxed = split_vals[j][:, index, ...]
                LOGGER.debug(
                    "split for pp_rank=%d tp_rank=%d factor=%d src_key=%s filename=%s "
                    "preprocessed_shape=%s shape=%s dtype=%s",
                    pipeline_para_rank,
                    tensor_para_rank,
                    factor,
                    key,
                    saved_path.name,
                    val.shape,
                    split_val_idxed.shape,
                    split_val_idxed.dtype,
                )
                split_val_idxed.tofile(saved_path)
    else:
        LOGGER.error(f"cannot find key '{key}'")


def convert_checkpoint(unpacked_checkpoints_dir: UnpackedNemoCheckpointDir, args: argparse.Namespace):
    nemo_model_config = unpacked_checkpoints_dir.model_config

    if nemo_model_config.get("kv_channels", None) is None:
        nemo_model_config["kv_channels"] = nemo_model_config["hidden_size"] // nemo_model_config["num_attention_heads"]

    inference_tensor_para_size = args.infer_gpu_num

    # if checkpoints files could be found - start preparing output dir
    saved_dir = pathlib.Path(args.saved_dir) / f"{inference_tensor_para_size:d}-gpu"
    if saved_dir.exists():
        LOGGER.error("Remove %s target directory before running conversion", saved_dir)
        for file_path in saved_dir.rglob("*"):
            LOGGER.debug("  %s", file_path.relative_to(saved_dir))
        sys.exit(1)

    saved_dir.mkdir(parents=True)

    checkpoints_paths = unpacked_checkpoints_dir.get_checkpoints_paths(
        nemo_model_config.get("tensor_model_parallel_size", 1),
        nemo_model_config.get("pipeline_model_parallel_size", 1),
    )

    LOGGER.debug("Expecting checkpoints paths in:")
    for tp_rank_checkpoints_paths in checkpoints_paths:
        for checkpoint_path in tp_rank_checkpoints_paths:
            LOGGER.debug("  %s", checkpoint_path)

    map_location_fn = cpu_map_location if bool(args.load_checkpoints_to_cpu) else gpu_map_location
    np_weight_data_type = WEIGHT2DTYPE[args.weight_data_type]

    has_gated_activations = False

    for pipeline_rank in range(len(checkpoints_paths[0])):
        model_from_selected_pipeline = torch.load(checkpoints_paths[0][pipeline_rank], map_location=map_location_fn)
        model_from_selected_pipeline = model_from_selected_pipeline.get("state_dict", model_from_selected_pipeline)

        LOGGER.debug(f"Existent pipeline_rank={pipeline_rank} keys:")
        for key in model_from_selected_pipeline.keys():
            LOGGER.debug("  %s", key)

        encoder_ape_key = "enc_dec_model.encoder_embedding.position_embeddings.weight"
        if encoder_ape_key in model_from_selected_pipeline.keys():
            saved_path = saved_dir / "shared.ape.bin"
            # not weight, do not need to transpose
            val = model_from_selected_pipeline[encoder_ape_key]
            val = torch2np(val, np_weight_data_type)
            LOGGER.debug(
                "save for pp_rank=%d src_key=%s saved_keys=%s shape=%s dtype=%s",
                pipeline_rank,
                encoder_ape_key,
                saved_path.name,
                val.shape,
                val.dtype,
            )
            val.tofile(saved_path)

        has_gated_activations |= any("mlp.dense_h_to_4h_2" in key for key in model_from_selected_pipeline.keys())

        def _split(src_key, dst_filename_fn):
            if src_key in model_from_selected_pipeline.keys():
                _val = model_from_selected_pipeline[src_key]
                _val = torch2np(_val, np_weight_data_type)
                _val = np.split(_val, inference_tensor_para_size, axis=0)
                for tensor_idx in range(inference_tensor_para_size):
                    saved_path = saved_dir / dst_filename_fn(tensor_idx)
                    LOGGER.debug(
                        "save for pp_rank=%d src_key=%s filename=%s shape=%s dtype=%s",
                        pipeline_rank,
                        src_key,
                        saved_path.name,
                        val.shape,
                        val.dtype,
                    )
                    _val[tensor_idx].tofile(saved_path)
                del _val

        # split rpe into tensor parallel ranks
        _split(
            "enc_dec_model.encoder_embedding.encoder_relative_position_embedding.weight",
            lambda idx: f"encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight.{idx}.bin",
        )
        _split(
            "enc_dec_model.decoder_embedding.decoder_relative_position_embedding.weight",
            lambda idx: f"decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight.{idx}.bin",
        )

        del model_from_selected_pipeline

    nemo_position_embedding_type = nemo_model_config.get("position_embedding_type", "absolute")
    nenemo_position_embedding_type = (
        "absolute" if nemo_position_embedding_type == "learned_absolute" else nemo_position_embedding_type
    )
    model_new_config = {
        "structure": {
            "t5_with_bias": str(True),
            "position_embedding_type": nenemo_position_embedding_type,
            "use_gated_activation": str(has_gated_activations),
        }
    }

    training_tensor_para_size = nemo_model_config.get("tensor_model_parallel_size", 1)
    training_pipeline_para_size = nemo_model_config.get("pipeline_model_parallel_size", 1)

    if training_tensor_para_size > inference_tensor_para_size:
        assert training_tensor_para_size % inference_tensor_para_size == 0
        is_merge_ckpt = True
        factor = int(training_tensor_para_size / inference_tensor_para_size)
    else:
        assert inference_tensor_para_size % training_tensor_para_size == 0
        is_merge_ckpt = False
        factor = int(inference_tensor_para_size / training_tensor_para_size)

    assert nemo_model_config["ffn_hidden_size"] % inference_tensor_para_size == 0
    assert nemo_model_config["num_attention_heads"] % inference_tensor_para_size == 0

    main_loop = min(training_tensor_para_size, inference_tensor_para_size)

    w_e_list = []
    lm_head_list = []

    torch.multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy("file_system")
    with concurrent.futures.ProcessPoolExecutor(args.processes) as pool:
        for tp_idx in range(main_loop):
            for pp_idx in range(training_pipeline_para_size):

                encoder_models = []
                decoder_models = []

                word_embedding_key = "enc_dec_model.encoder_embedding.word_embeddings.weight"
                lm_head_bias_key = "enc_dec_model.tokens_head.bias"
                if is_merge_ckpt:
                    for k in range(factor):
                        rank_weights = checkpoints_paths[tp_idx * factor + k][pp_idx]
                        model = torch.load(rank_weights, map_location=map_location_fn)

                        if pp_idx == 0:
                            w_e_val = model.get("state_dict", model)[word_embedding_key]
                            w_e_val = torch2np(w_e_val, np_weight_data_type)
                            w_e_list.append(w_e_val)
                        if pp_idx == training_pipeline_para_size - 1:
                            lm_head_val = model.get("state_dict", model)[lm_head_bias_key]
                            lm_head_val = torch2np(lm_head_val, np_weight_data_type)
                            lm_head_list.append(lm_head_val)

                        encoder_models.append(
                            extract_layers_with_prefix(model, "enc_dec_model.enc_dec_model.encoder.model.")
                        )
                        decoder_models.append(
                            extract_layers_with_prefix(model, "enc_dec_model.enc_dec_model.decoder.model.")
                        )
                        LOGGER.debug(
                            "For pp_idx=%d tp_id=%d merging weights from %s extracted:", pp_idx, tp_idx, rank_weights
                        )
                        LOGGER.debug("  encoder layers")
                        for name in encoder_models[-1]:
                            LOGGER.debug("    %s", name)
                        LOGGER.debug("  decoder layers")
                        for name in decoder_models[-1]:
                            LOGGER.debug("    %s", name)
                else:
                    rank_weights = checkpoints_paths[tp_idx][pp_idx]
                    model = torch.load(rank_weights, map_location=map_location_fn)

                    if pp_idx == 0:
                        w_e_val = model.get("state_dict", model)[word_embedding_key]
                        w_e_val = torch2np(w_e_val, np_weight_data_type)
                        w_e_list.append(w_e_val)
                    if pp_idx == training_pipeline_para_size - 1:
                        lm_head_val = model.get("state_dict", model)[lm_head_bias_key]
                        lm_head_val = torch2np(lm_head_val, np_weight_data_type)
                        lm_head_list.append(lm_head_val)

                    encoder_models.append(
                        extract_layers_with_prefix(model, "enc_dec_model.enc_dec_model.encoder.model.")
                    )
                    decoder_models.append(
                        extract_layers_with_prefix(model, "enc_dec_model.enc_dec_model.decoder.model.")
                    )
                    LOGGER.debug(
                        "For pp_idx=%d tp_id=%d copy/splitting weights from %s extracted:", pp_idx, tp_idx, rank_weights
                    )
                    LOGGER.debug("  encoder layers")
                    for name in encoder_models[-1]:
                        LOGGER.debug("    %s", name)
                    LOGGER.debug("  decoder layers")
                    for name in decoder_models[-1]:
                        LOGGER.debug("    %s", name)

                process_fn = merge_and_convert_process if is_merge_ckpt else split_and_convert_process

                for key in encoder_models[0]:
                    pool.submit(
                        process_fn,
                        "encoder",
                        tp_idx,  # tp_rank
                        pp_idx,  # pp_rank
                        saved_dir,
                        factor,
                        key,
                        nemo_model_config,
                        encoder_models,
                        np_weight_data_type,
                    )

                for key in decoder_models[0]:
                    pool.submit(
                        process_fn,
                        "decoder",
                        tp_idx,  # tp_rank
                        pp_idx,  # pp_rank
                        saved_dir,
                        factor,
                        key,
                        nemo_model_config,
                        decoder_models,
                        np_weight_data_type,
                    )

    w_e_saved_path = saved_dir / "shared.weight_T.bin"
    lm_head_weight_saved_path = saved_dir / "lm_head.weight.bin"
    lm_head_saved_path = saved_dir / "shared.bias.bin"
    w_e_val = np.concatenate(w_e_list, axis=0)
    lm_head_val = np.concatenate(lm_head_list, axis=0)
    LOGGER.debug(
        "save for src_key=%s filename=%s shape=%s dtype=%s",
        word_embedding_key,
        w_e_saved_path.name,
        w_e_val.shape,
        w_e_val.dtype,
    )
    LOGGER.debug(
        "save for src_key=%s filename=%s shape=%s dtype=%s",
        lm_head_bias_key,
        lm_head_saved_path.name,
        lm_head_val.shape,
        lm_head_val.dtype,
    )
    w_e_val.tofile(w_e_saved_path)
    w_e_val.tofile(lm_head_weight_saved_path)
    lm_head_val.tofile(lm_head_saved_path)

    vocab_size = w_e_val.shape[0]

    config = configparser.ConfigParser()

    config["encoder"] = {
        **{
            "_name_or_path": args.model_name,
            "model_type": "T5",
            "weight_data_type": args.weight_data_type,
            "tensor_para_size": str(inference_tensor_para_size),
            "vocab_size": str(vocab_size),
        },
        **{
            encoder_config_mapping[key]: str(val)
            for key, val in nemo_model_config.items()
            if key in encoder_config_mapping
        },
    }

    tokenizer_config = nemo_model_config["tokenizer"]
    tokenizer_config = _update_tokenizer_config(tokenizer_config, unpacked_checkpoints_dir)
    if args.tokenizer_model_path:
        LOGGER.debug("Use tokenizer model passed from CLI: %s", args.tokenizer_model_path)
        tokenizer_config["model"] = args.tokenizer_model_path
    if args.vocab_path:
        LOGGER.debug("Use tokenizer vocab passed from CLI: %s", args.vocab_path)
        tokenizer_config["vocab_file"] = args.vocab_path
    if args.merges_path:
        LOGGER.debug("Use tokenizer merge passed from CLI: %s", args.merges_path)
        tokenizer_config["merge_file"] = args.merges_path

    _copy_tokenizer_file_if_defined("model", tokenizer_config["model"], saved_dir)
    _copy_tokenizer_file_if_defined("vocab_file", tokenizer_config["vocab_file"], saved_dir)
    _copy_tokenizer_file_if_defined("merge_file", tokenizer_config["merge_file"], saved_dir)

    bos_id, eos_id = _get_special_tokens_ids(tokenizer_config)

    config["decoder"] = {
        **{
            "_name_or_path": args.model_name,
            "model_type": "T5",
            "weight_data_type": args.weight_data_type,
            "tensor_para_size": str(inference_tensor_para_size),
            "vocab_size": str(vocab_size),
            "decoder_start_token_id": str(bos_id),
            "eos_token_id": str(eos_id),
        },
        **{
            decoder_config_mapping[key]: str(val)
            for key, val in nemo_model_config.items()
            if key in decoder_config_mapping
        },
    }

    for section, section_dict in model_new_config.items():
        config[section] = {k: str(v) for k, v in section_dict.items()}

    with (saved_dir / f"config.ini").open("w") as configfile:
        config.write(configfile)


def _update_tokenizer_config(tokenizer_config: typing.Dict, unpacked_checkpoints_dir):
    def _update_config_entry(key, file_pattern):
        old_file_path = tokenizer_config[key]
        if old_file_path:
            LOGGER.debug("tokenizer %s %s type %s", key, old_file_path, type(old_file_path))
            old_file_path = pathlib.Path(old_file_path)
            new_file_path = unpacked_checkpoints_dir.get_tokenizer_file_path("tokenizer", key, file_pattern)
            if new_file_path:
                LOGGER.debug("Update tokenizer %s %s -> %s", key, old_file_path, new_file_path)
                tokenizer_config[key] = new_file_path.as_posix()
            elif not old_file_path.exists():
                LOGGER.warning("Because tokenizer %s %s does not exists - set it as None", key, old_file_path)
                tokenizer_config[key] = None

    _update_config_entry("model", "*.model")
    _update_config_entry("vocab_file", "*vocab*")
    _update_config_entry("merge_file", "*merge*.txt")

    return tokenizer_config


def _copy_tokenizer_file_if_defined(key_name, tokenizer_file_path, saved_dir):
    if tokenizer_file_path:
        tokenizer_file_path = pathlib.Path(tokenizer_file_path)
        if tokenizer_file_path.exists():
            tokenizer_basename = {
                "model": "tokenizer",
                "vocab_file": "vocab",
                "merge_file": "merges",
            }[key_name]
            dst_path = saved_dir / f"{tokenizer_basename}{tokenizer_file_path.suffix}"
            LOGGER.debug("Copy of %s %s file as %s", tokenizer_file_path, key_name, dst_path)
            shutil.copy(tokenizer_file_path.as_posix(), dst_path.as_posix())
        else:
            LOGGER.debug("%s %s file does not exists", tokenizer_file_path, key_name)


def _get_special_tokens_ids(tokenizer_config: typing.Dict):
    from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
    from examples.pytorch.tokenizer import add_special_tokens_to_tokenizer

    logging.getLogger("git.cmd").setLevel(logging.INFO)
    logging.getLogger("h5py._conv").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.INFO)
    logging.getLogger("matplotlib.pyplot").setLevel(logging.INFO)

    tokenizer = get_nmt_tokenizer(
        library=tokenizer_config["library"],
        model_name=tokenizer_config["type"],
        tokenizer_model=tokenizer_config["model"],
        vocab_file=tokenizer_config["vocab_file"],
        merges_file=tokenizer_config["merge_file"],
        legacy=True,
    )

    if tokenizer_config["library"] == "sentencepiece":
        add_special_tokens_to_tokenizer(tokenizer)

    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id

    LOGGER.debug("for %s obtained tokenizer tokens ids bos_id=%d eos_id=%d", tokenizer_config, bos_id, eos_id)

    return bos_id, eos_id


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--saved-dir",
        "-saved_dir",
        "-o",
        help="folder name of output files",
        required=True,
    )
    parser.add_argument(
        "--in-file",
        "-in_file",
        "-i",
        help="file name of .nemo checkpoint file or checkpoint dir",
        required=True,
    )
    parser.add_argument(
        "--infer-gpu-num",
        "-infer_gpu_num",
        "-i_g",
        type=int,
        help="How many gpus for inference",
        required=True,
    )
    parser.add_argument(
        "--processes",
        "-processes",
        "-p",
        type=int,
        default=64,
        help="How many processes to spawn for conversion",
    )
    parser.add_argument(
        "--weight-data-type",
        "-weight_data_type",
        choices=["fp32", "fp16"],
        default="fp32",
        help="Data type of results weights",
    )
    parser.add_argument(
        "--model-name",
        "-model_name",
        "-m",
        help="model name",
        required=True,
    )
    parser.add_argument(
        "--vocab-path",
        help="Path to vocabulary file to embed in FasterTransformer checkpoint",
        required=False,
    )
    parser.add_argument(
        "--merges-path",
        help="Path to merges file to embed in FasterTransformer checkpoint",
        required=False,
    )
    parser.add_argument(
        "--tokenizer-model-path",
        help="Path to tokenizer model file to embed in FasterTransformer checkpoint",
        required=False,
    )
    parser.add_argument(
        "--load-checkpoints-to-cpu",
        "-load_checkpoints_to_cpu",
        "-cpu",
        type=int,
        choices=[0, 1],
        default=1,
        help="Whether to load model weights to CPU",
    )
    parser.add_argument("--verbose", action="store_true", help="Provide verbose messages")
    args = parser.parse_args()

    log_format = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=log_format)

    print("\n=============== Argument ===============")
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    print("========================================")

    input_path = pathlib.Path(args.in_file)
    if not input_path.exists():
        LOGGER.error("%s does not exists", input_path)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = pathlib.Path(temp_dir)

        # unpack if needed
        if input_path.is_file():
            checkpoint_dir_path = temp_dir / "unpacked"
            start_time = datetime.datetime.now()
            unpacked_checkpoint_dir = UnpackedNemoCheckpointDir(
                unpack_nemo_ckpt(args.in_file, checkpoint_dir_path),
                load_checkpoints_to_cpu=bool(args.load_checkpoints_to_cpu),
            )
            LOGGER.info("Spent %s (h:m:s) to unpack NeMo archive", datetime.datetime.now() - start_time)
        else:
            unpacked_checkpoint_dir = UnpackedNemoCheckpointDir(
                input_path, load_checkpoints_to_cpu=bool(args.load_checkpoints_to_cpu)
            )

        LOGGER.debug("Unpacked NeMo checkpoint contains:")
        for file_path in unpacked_checkpoint_dir.checkpoints_dir.rglob("*"):
            LOGGER.debug("  %s", file_path)

        start_time = datetime.datetime.now()
        convert_checkpoint(unpacked_checkpoint_dir, args)
        LOGGER.info("Spent %s (h:m:s) to convert the model", datetime.datetime.now() - start_time)


if __name__ == "__main__":
    main()
