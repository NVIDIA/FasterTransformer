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
import datetime
import json
import pathlib
import shutil
import sys
import os

import numpy as np
import torch  # pytype: disable=import-error

# verify if root package is in PYTHONPATH
__root_package_path__ = pathlib.Path(__file__).parent.parent.parent.parent.parent.absolute().as_posix()
if __root_package_path__ not in sys.path:
    print(
        f"[ERROR] add project root directory to your PYTHONPATH with "
        f"'export PYTHONPATH={__root_package_path__}:${{PYTHONPATH}}'"
    )

from examples.pytorch.gpt.utils.gpt import DEFAULT_START_TAG, DEFAULT_END_TAG, OPENAI_GPT2_START_ID, OPENAI_GPT2_END_ID
from examples.pytorch.utils import torch2np, safe_transpose, cpu_map_location, gpu_map_location, WEIGHT2DTYPE


def save_dense_split(model_states_list, factor_dense, model_key, megatron_gpt_key, np_weight_data_type, saved_dir,
        ckpt_ver, model_training_args):
    training_pipeline_para_dense_size = len(model_states_list[0])
    step_layer_pp = model_training_args.num_layers // training_pipeline_para_dense_size
    model_list = [
        [
            model_states[model_key]["language_model"][megatron_gpt_key]
            for model_states in sub_model_states_list
        ]
        for sub_model_states_list in model_states_list
    ]
    has_adapters = any("adaptor" in key for key in model_list[0][0].keys())
    moe_layers = []
    for idx_tp, sub_model_list in enumerate(model_list):
        save_offset = idx_tp * factor_dense

        for idx_pp, model in enumerate(sub_model_list):
            for key, val in model.items():
                val = safe_transpose(val)
                val = torch2np(val, np_weight_data_type)
                saved_key = key
                if key.find("layers.") != -1:
                    key_split = key.split('.')
                    layer_index = (int)(key_split[1]) + idx_pp * step_layer_pp
                    saved_key = '.'.join(key_split[:1] + [str(layer_index)] + key_split[2:])
                    if saved_key.find("self_attention") != -1:
                        saved_key = saved_key.replace("self_attention", "attention")
                    if saved_key.find("adaptor1") != -1:
                        saved_key = saved_key.replace("adaptor1", "after_attention_adapter")
                    if saved_key.find("adaptor2") != -1:
                        saved_key = saved_key.replace("adaptor2", "after_ffn_adapter")

                if (
                    key.find("input_layernorm.weight") != -1
                    or key.find("input_layernorm.bias") != -1
                    or key.find("attention.dense.bias") != -1
                    or key.find("post_attention_layernorm.weight") != -1
                    or key.find("post_attention_layernorm.bias") != -1
                    or key.find("mlp.dense_4h_to_h.bias") != -1
                    or key.find("adaptor1.dense_4h_to_h.bias") != -1
                    or key.find("adaptor2.dense_4h_to_h.bias") != -1
                    or key.find("final_layernorm.weight") != -1
                    or key.find("final_layernorm.bias") != -1
                ):
                    # shared weights, only need to convert the weights of rank 0
                    if idx_tp == 0:
                        saved_path = saved_dir / f"model.{saved_key}.bin"
                        val.tofile(saved_path.as_posix())

                elif (key.find("attention.dense.weight") != -1
                    or key.find("mlp.dense_4h_to_h.weight") != -1
                    or key.find("adaptor1.dense_4h_to_h.weight") != -1
                    or key.find("adaptor2.dense_4h_to_h.weight") != -1):
                    split_vals = np.split(val, factor_dense, axis=0)
                    for j in range(factor_dense):
                        saved_path = saved_dir / f"model.{saved_key}.{save_offset + j:d}.bin"
                        split_vals[j].tofile(saved_path.as_posix())

                elif (key.find("mlp.dense_h_to_4h.weight") != -1
                    or key.find("adaptor1.dense_h_to_4h.weight") != -1
                    or key.find("adaptor2.dense_h_to_4h.weight") != -1
                    or key.find("mlp.dense_h_to_4h.bias") != -1
                    or key.find("adaptor1.dense_h_to_4h.bias") != -1
                    or key.find("adaptor2.dense_h_to_4h.bias") != -1):
                    split_vals = np.split(val, factor_dense, axis=-1)
                    for j in range(factor_dense):
                        saved_path = saved_dir / f"model.{saved_key}.{save_offset + j:d}.bin"
                        split_vals[j].tofile(saved_path.as_posix())

                elif key.find("attention.query_key_value.bias") != -1:
                    local_dim = int(val.shape[-1] / 3)

                    if ckpt_ver == 3:
                        num_splits = 3
                        head_num = model_training_args.num_attention_heads // model_training_args.tensor_model_parallel_size
                        size_per_head = local_dim // head_num
                        val = val.reshape(head_num, num_splits, size_per_head)
                        val = val.transpose(1, 0, 2)

                    val = val.reshape(3, local_dim)
                    split_vals = np.split(val, factor_dense, axis=-1)

                    for j in range(factor_dense):
                        saved_path = saved_dir / f"model.{saved_key}.{save_offset + j:d}.bin"
                        split_vals[j].tofile(saved_path.as_posix())

                elif key.find("attention.query_key_value.weight") != -1:
                    hidden_dim = val.shape[0]
                    assert val.shape[-1] % 3 == 0
                    local_dim = val.shape[-1] // 3

                    if ckpt_ver == 3:
                        num_splits = 3
                        head_num = model_training_args.num_attention_heads
                        assert hidden_dim % head_num == 0
                        size_per_head = hidden_dim // head_num
                        assert head_num % model_training_args.tensor_model_parallel_size == 0
                        head_num = head_num // model_training_args.tensor_model_parallel_size
                        val = val.reshape(hidden_dim, head_num, num_splits, size_per_head)
                        val = val.transpose(0, 2, 1, 3)

                    val = val.reshape(hidden_dim, 3, local_dim)
                    split_vals = np.split(val, factor_dense, axis=-1)

                    for j in range(factor_dense):
                        saved_path = saved_dir / f"model.{saved_key}.{save_offset + j:d}.bin"
                        split_vals[j].tofile(saved_path.as_posix())

                elif key.find('experts') == -1:
                    if key.find('deepspeed_moe.gate') != -1 or key.find('megatron_moe.gate') != -1:
                        layer_index = (int)(key.split('.')[1]) + idx_pp * step_layer_pp
                        if idx_tp == 0:
                            moe_layers.append(layer_index)
                    prefix = key.replace('deepspeed_moe', 'moe').replace('megatron_moe', 'moe')
                    if key.find('layernorm') != -1 or key.find('gate') != -1 or key.find("attention.dense.bias") != -1 \
                            or key.find("dense_4h_to_h.bias") != -1:
                        if idx_tp == 0:
                            file_name = os.path.join(saved_dir, "model." + prefix + ".bin")
                            saved_path = file_name
                            # print(f"Saving '{prefix}' to '{file_name}'")
                            # print(f"Shape: '{val.shape}'")
                            val.tofile(file_name)
                    else:
                        val_tensor_para = []
                        print(key, val.shape)
                        if key.find("attention.dense.weight") != -1 or key.find("dense_4h_to_h.weight") != -1 \
                                or key.find("dense_h_to_4h.bias") != -1:
                            val_tensor_para = np.split(val, factor_dense, axis=0)
                        elif key.find("dense_h_to_4h.weight") != -1:
                            val_tensor_para = np.split(val, factor_dense, axis=1)
                        else:
                            print(f"[ERROR] cannot find experts key '{key}'")
                            sys.exit(1)

                        for j in range(factor_dense):
                            file_name = os.path.join(saved_dir, "model." + prefix + "." + str(save_offset + j) + ".bin")
                            saved_path = file_name
                            # print(f"Saving '{j}' '{prefix}' to '{file_name}'")
                            val_to_save = val_tensor_para[j]
                            # print(f"Shape: '{val_to_save.shape}'")
                            val_to_save.tofile(file_name)

                else:
                    print(f"[ERROR] cannot find key '{key}'")
                    sys.exit(1)
                print('{} {} {} {}'.format(idx_tp, idx_pp, key, saved_path))
                print(val.shape)
    return moe_layers, has_adapters


def save_dense_concat(model_states_list, inference_tensor_para_size, factor_dense, model_key, megatron_gpt_key,
        np_weight_data_type, saved_dir, ckpt_ver, model_training_args):
    def convert_val(x):
        x = safe_transpose(x)
        x = torch2np(x, np_weight_data_type)
        return x

    training_pipeline_para_dense_size = len(model_states_list[0])
    step_layer_pp = model_training_args.num_layers // training_pipeline_para_dense_size
    model_list = [
        [
            model_states[model_key]["language_model"][megatron_gpt_key]
            for model_states in sub_model_states_list
        ]
        for sub_model_states_list in model_states_list
    ]
    has_adapters = any("adaptor" in key for key in model_list[0][0].keys())
    moe_layers = []
    for idx_tp in range(inference_tensor_para_size):
        load_offset = idx_tp * factor_dense

        for idx_pp in range(training_pipeline_para_dense_size):
            for key in model_list[0][0]:
                saved_key = key
                if key.find("layers.") != -1:
                    key_split = key.split('.')
                    layer_index = (int)(key_split[1]) + idx_pp * step_layer_pp
                    saved_key = '.'.join(key_split[:1] + [str(layer_index)] + key_split[2:])
                    if saved_key.find("self_attention") != -1:
                        saved_key = saved_key.replace("self_attention", "attention")
                    if saved_key.find("adaptor1") != -1:
                        saved_key = saved_key.replace("adaptor1", "after_attention_adapter")
                    if saved_key.find("adaptor2") != -1:
                        saved_key = saved_key.replace("adaptor2", "after_ffn_adapter")

                if (
                    key.find("input_layernorm.weight") != -1
                    or key.find("input_layernorm.bias") != -1
                    or key.find("attention.dense.bias") != -1
                    or key.find("post_attention_layernorm.weight") != -1
                    or key.find("post_attention_layernorm.bias") != -1
                    or key.find("mlp.dense_4h_to_h.bias") != -1
                    or key.find("adaptor1.dense_4h_to_h.bias") != -1
                    or key.find("adaptor2.dense_4h_to_h.bias") != -1
                    or key.find("final_layernorm.weight") != -1
                    or key.find("final_layernorm.bias") != -1
                ):
                    # shared weights, only need to convert the weights of rank 0
                    if idx_tp == 0:
                        concat_val = convert_val(model_list[0][idx_pp][key])
                        saved_path = saved_dir / f"model.{saved_key}.bin"
                        concat_val.tofile(saved_path.as_posix())

                elif (key.find("attention.dense.weight") != -1
                    or key.find("mlp.dense_4h_to_h.weight") != -1
                    or key.find("adaptor1.dense_4h_to_h.weight") != -1
                    or key.find("adaptor2.dense_4h_to_h.weight") != -1):
                    val_list = [convert_val(model_list[load_offset + j][idx_pp][key]) for j in range(factor_dense)]
                    concat_val = np.concatenate(val_list, axis=0)
                    saved_path = saved_dir / f"model.{saved_key}.{idx_tp:d}.bin"
                    concat_val.tofile(saved_path.as_posix())

                elif (key.find("mlp.dense_h_to_4h.weight") != -1
                    or key.find("adaptor1.dense_h_to_4h.weight") != -1
                    or key.find("adaptor2.dense_h_to_4h.weight") != -1
                    or key.find("mlp.dense_h_to_4h.bias") != -1
                    or key.find("adaptor1.dense_h_to_4h.bias") != -1
                    or key.find("adaptor2.dense_h_to_4h.bias") != -1):
                    val_list = [convert_val(model_list[load_offset + j][idx_pp][key]) for j in range(factor_dense)]
                    concat_val = np.concatenate(val_list, axis=-1)
                    saved_path = saved_dir / f"model.{saved_key}.{idx_tp:d}.bin"
                    concat_val.tofile(saved_path.as_posix())

                elif key.find("attention.query_key_value.bias") != -1:
                    val_list = []
                    for j in range(factor_dense):
                        val = convert_val(model_list[load_offset + j][idx_pp][key])
                        assert val.shape[-1] % 3 == 0
                        local_dim = val.shape[-1] // 3
                        if ckpt_ver == 3:
                            num_splits = 3
                            num_attention_heads = model_training_args.num_attention_heads
                            tensor_model_parallel_size = model_training_args.tensor_model_parallel_size
                            assert num_attention_heads % tensor_model_parallel_size == 0
                            head_num = num_attention_heads // tensor_model_parallel_size
                            assert local_dim % head_num == 0
                            size_per_head = local_dim // head_num
                            val = val.reshape(head_num, num_splits, size_per_head)
                            val = val.transpose(1, 0, 2)
                        val = val.reshape(3, local_dim)
                        val_list.append(val)
                    concat_val = np.concatenate(val_list, axis=-1)
                    saved_path = saved_dir / f"model.{saved_key}.{idx_tp:d}.bin"
                    concat_val.tofile(saved_path.as_posix())

                elif key.find("attention.query_key_value.weight") != -1:
                    val_list = []
                    for j in range(factor_dense):
                        val = convert_val(model_list[load_offset + j][idx_pp][key])
                        hidden_dim = val.shape[0]
                        local_dim = int(val.shape[-1] / 3)
                        if ckpt_ver == 3:
                            num_splits = 3
                            head_num = model_training_args.num_attention_heads
                            size_per_head = hidden_dim // head_num
                            head_num = head_num // model_training_args.tensor_model_parallel_size
                            val = val.reshape(hidden_dim, head_num, num_splits, size_per_head)
                            val = val.transpose(0, 2, 1, 3)
                        val = val.reshape(hidden_dim, 3, local_dim)
                        val_list.append(val)
                    concat_val = np.concatenate(val_list, axis=-1)
                    saved_path = saved_dir / f"model.{saved_key}.{idx_tp:d}.bin"
                    concat_val.tofile(saved_path.as_posix())

                elif key.find('experts') == -1:
                    if key.find('deepspeed_moe.gate') != -1 or key.find('megatron_moe.gate') != -1:
                        layer_index = (int)(key.split('.')[1]) + idx_pp * step_layer_pp
                        if idx_tp == 0:
                            moe_layers.append(layer_index)
                    prefix = key.replace('deepspeed_moe', 'moe').replace('megatron_moe', 'moe')
                    if key.find('layernorm') != -1 or key.find('gate') != -1 or key.find("attention.dense.bias") != -1 or key.find("dense_4h_to_h.bias") != -1:
                        if idx_tp == 0:
                            concat_val = convert_val(model_list[0][idx_pp][key])
                            file_name = os.path.join(saved_dir, "model." + prefix + ".bin")
                            saved_path = file_name
                            # print(f"Saving '{prefix}' to '{file_name}'")
                            # print(f"Shape: '{val.shape}'")
                            concat_val.tofile(file_name)
                    else:
                        if key.find("attention.dense.weight") != -1 or key.find("dense_4h_to_h.weight") != -1 or key.find("dense_h_to_4h.bias") != -1:
                            concat_axis = 0
                        elif key.find("dense_h_to_4h.weight") != -1:
                            concat_axis = 1
                        else:
                            print(f"[ERROR] cannot find experts key '{key}'")
                            sys.exit(1)
                        val_list = []
                        for j in range(factor_dense):
                            val = convert_val(model_list[load_offset + j][idx_pp][key])
                            val_list.append(val)
                        concat_val = np.concatenate(val_list, axis=concat_axis)
                        file_name = os.path.join(saved_dir, "model." + prefix + "." + str(idx_tp) + ".bin")
                        saved_path = file_name
                        concat_val.tofile(file_name)

                else:
                    print(f"[ERROR] cannot find key '{key}'")
                    sys.exit(1)
                print('{} {} {} {}'.format(idx_tp, idx_pp, key, saved_path))
                print(concat_val.shape)
    return moe_layers, has_adapters


def save_experts_split(moe_layers, num_experts, training_tensor_para_dense_size, training_tensor_para_expert_size,
        factor_expert, args, map_location_fn, is_deepspeed, np_weight_data_type, saved_dir,
        training_pipeline_para_expert_size):
    def get_file_name(idx_moe, idx_expert, idx_rank):
        if training_tensor_para_expert_size > 1:
            assert len(moe_layers) % training_pipeline_para_expert_size == 0
            step_layer_pp = len(moe_layers) // training_pipeline_para_expert_size
            file_name = 'layer_{}_expert_{}_mp_rank_{:02}_{:03}_model_states.pt'.format(
                idx_moe % step_layer_pp, idx_expert, idx_rank, idx_moe // step_layer_pp)
        else:
            file_name = 'layer_{}_expert_{}_mp_rank_{:02}_model_states.pt'.format(idx_moe, idx_expert, idx_rank)
        return file_name
    assert training_tensor_para_expert_size == 1 or training_tensor_para_expert_size == training_tensor_para_dense_size
    # Saving experts weight
    print(f"The number of moe layers is '{len(moe_layers)}'")
    for idx_tp in range(training_tensor_para_expert_size):
        save_offset = idx_tp * factor_expert
        for n, idx_layer in enumerate(moe_layers):
            fc1_weight = []
            fc1_bias = []
            fc2_weight = []
            fc2_bias = []
            prefix = None
            for e in range(num_experts):
                if training_tensor_para_expert_size == training_tensor_para_dense_size:
                    file_name = get_file_name(n, e, idx_tp)
                    file_path = os.path.join(args.input_dir, file_name)
                else:
                    for idx_rank in range(training_tensor_para_dense_size):
                        file_name = get_file_name(n, e, idx_rank)
                        file_path = os.path.join(args.input_dir, file_name)
                        if os.path.exists(file_path):
                            break
                    else:
                        raise FileNotFoundError
                expert_dict = torch.load(file_path, map_location=map_location_fn)
                for k, v in expert_dict.items():
                    if k.find('dense_h_to_4h.weight') != -1:
                        if prefix is None:
                            prefix = k
                        fc1_weight.append(v)
                    elif k.find('dense_h_to_4h.bias') != -1:
                        fc1_bias.append(v)
                    elif k.find('dense_4h_to_h.weight') != -1:
                        fc2_weight.append(v)
                    elif k.find('dense_4h_to_h.bias') != -1:
                        fc2_bias.append(v)
                    else:
                        print(f"[ERROR] cannot find expert_dict key '{k}'")
                        sys.exit(1)
            if is_deepspeed:
                prefix_list = ['model'] + prefix.split('.')[2:5] + ['moe', 'experts']
            else:
                prefix_list = ['model'] + prefix.split('.')[4:7] + ['moe', 'experts']
            prefix = '.'.join(prefix_list) + '.'
            prefix_split = prefix.split('.')
            prefix_split[1] = 'layers'
            prefix_split[3] = 'mlp'
            prefix = '.'.join(prefix_split[:2] + [str(idx_layer)] + prefix_split[3:])

            stacked_fc1_weight = torch.stack(fc1_weight, 0).transpose(-1, -2).contiguous()
            # val = stacked_fc1_weight.float().cpu().numpy()  # (num_experts, d_model, d_ff)
            val = torch2np(stacked_fc1_weight, np_weight_data_type)
            val_tensor_para = np.split(val, factor_expert, axis=2)
            for i in range(factor_expert):
                file_name = os.path.join(saved_dir, prefix + "dense_h_to_4h.weight." + str(save_offset + i) + ".bin")
                print(f"Saving '{i}' '{prefix}' to '{file_name}'")
                val_to_save = val_tensor_para[i]
                print(f"Shape: '{val_to_save.shape}'")
                val_to_save.tofile(file_name)

            stacked_fc1_bias = torch.stack(fc1_bias, 0).contiguous()
            # val = stacked_fc1_bias.float().cpu().numpy() # (num_experts, d_ff)
            val = torch2np(stacked_fc1_bias, np_weight_data_type)
            val_tensor_para = np.split(val, factor_expert, axis=1)
            for i in range(factor_expert):
                file_name = os.path.join(saved_dir, prefix + "dense_h_to_4h.bias." + str(save_offset + i) + ".bin")
                print(f"Saving '{i}' '{prefix}' to '{file_name}'")
                val_to_save = val_tensor_para[i]
                print(f"Shape: '{val_to_save.shape}'")
                val_to_save.tofile(file_name)

            stacked_fc2_weight = torch.stack(fc2_weight, 0).transpose(-1, -2).contiguous()
            # val = stacked_fc2_weight.float().cpu().numpy() # (num_experts, d_ff, d_model)
            val = torch2np(stacked_fc2_weight, np_weight_data_type)
            val_tensor_para = np.split(val, factor_expert, axis=1)
            for i in range(factor_expert):
                file_name = os.path.join(saved_dir, prefix + "dense_4h_to_h.weight." + str(save_offset + i) + ".bin")
                print(f"Saving '{i}' '{prefix}' to '{file_name}'")
                val_to_save = val_tensor_para[i]
                print(f"Shape: '{val_to_save.shape}'")
                val_to_save.tofile(file_name)

            if idx_tp == 0:
                stacked_fc2_bias = torch.stack(fc2_bias, 0)
                # val = stacked_fc2_bias.float().cpu().numpy()
                val = torch2np(stacked_fc2_bias, np_weight_data_type)
                file_name = os.path.join(saved_dir, prefix + "dense_4h_to_h.bias.bin")
                print(f"Saving '{i}' '{prefix}' to '{file_name}'")
                print(f"Shape: '{val.shape}'")
                val.tofile(file_name)
    return


def convert_checkpoint(args):
    saved_dir = pathlib.Path(args.saved_dir) / f"{args.infer_gpu_num:d}-gpu"
    if saved_dir.exists():
        shutil.rmtree(saved_dir)
    saved_dir.mkdir(parents=True)

    if args.vocab_path:
        shutil.copy(args.vocab_path, (saved_dir / "vocab.json").as_posix())
    if args.merges_path:
        shutil.copy(args.merges_path, (saved_dir / "merges.txt").as_posix())

    load_checkpoints_to_cpu = bool(args.load_checkpoints_to_cpu)
    map_location_fn = cpu_map_location if load_checkpoints_to_cpu else gpu_map_location

    config_model_states_list = [
        # Pattern 1
        {
            'pattern': 'mp_rank_{:02}/model_states.pt',
            'use_dense_pp': False,
            'is_deepspeed': True,
        },
        {
            'pattern': 'mp_rank_{:02}_{:03}/model_states.pt',
            'use_dense_pp': True,
            'is_deepspeed': True,
        },
        {
            'pattern': 'tp_rank_{:02}_pp_rank_{:03}/model_states.pt',
            'use_dense_pp': True,
            'is_deepspeed': True,
        },
        # Pattern 2
        {
            'pattern': 'mp_rank_{:02}_model_states.pt',
            'use_dense_pp': False,
            'is_deepspeed': True,
        },
        {
            'pattern': 'mp_rank_{:02}_{:03}_model_states.pt',
            'use_dense_pp': True,
            'is_deepspeed': True,
        },
        {
            'pattern': 'tp_rank_{:02}_pp_rank_{:03}_model_states.pt',
            'use_dense_pp': True,
            'is_deepspeed': True,
        },
        # Pattern 3
        {
            'pattern': 'mp_rank_{:02}/model_rng.pt',
            'use_dense_pp': False,
            'is_deepspeed': False,
        },
        {
            'pattern': 'mp_rank_{:02}_{:03}/model_rng.pt',
            'use_dense_pp': True,
            'is_deepspeed': False,
        },
        {
            'pattern': 'tp_rank_{:02}_pp_rank_{:03}/model_rng.pt',
            'use_dense_pp': True,
            'is_deepspeed': False,
        },
    ]
    for config_model_states in config_model_states_list:
        pattern = config_model_states['pattern']
        use_dense_pp = config_model_states['use_dense_pp']
        is_deepspeed = config_model_states['is_deepspeed']
        if use_dense_pp:
            path_model_states = os.path.join(args.input_dir, pattern.format(0, 0))
        else:
            path_model_states = os.path.join(args.input_dir, pattern.format(0))
        if os.path.exists(path_model_states):
            break
    else:
        raise FileNotFoundError("'path_model_states' not found")
    model_states_00 = torch.load(path_model_states, map_location=map_location_fn)
    for model_key in ['model', 'module']:
        if model_key in model_states_00:
            break
    else:
        raise KeyError("'model_key' not found")
    ckpt_ver = model_states_00["checkpoint_version"]
    assert ckpt_ver == 3
    megatron_gpt_key = "encoder"
    model_training_args = model_states_00["args"]
    training_tensor_para_dense_size = model_training_args.tensor_model_parallel_size
    training_tensor_para_expert_size = 1
    training_pipeline_para_dense_size = model_training_args.pipeline_model_parallel_size
    training_pipeline_para_expert_size = 1
    inference_tensor_para_size = args.infer_gpu_num
    assert use_dense_pp == (training_pipeline_para_dense_size > 1)
    assert model_training_args.num_layers % training_pipeline_para_dense_size == 0
    assert model_training_args.num_layers % training_pipeline_para_expert_size == 0
    if use_dense_pp:
        model_states_list = [
            [
                torch.load(os.path.join(args.input_dir, pattern.format(idx_tp, idx_pp)), map_location=map_location_fn)
                for idx_pp in range(training_pipeline_para_dense_size)
            ]
            for idx_tp in range(training_tensor_para_dense_size)
        ]
    else:
        model_states_list = [
            [torch.load(os.path.join(args.input_dir, pattern.format(idx_tp)), map_location=map_location_fn)]
            for idx_tp in range(training_tensor_para_dense_size)
        ]

    with (saved_dir / "args.txt").open("w") as training_args_file:
        for k, v in vars(model_training_args).items():
            training_args_file.write(f"{k}:{v}\n")

    np_weight_data_type = WEIGHT2DTYPE[args.weight_data_type]

    val = model_states_00[model_key]["language_model"]["embedding"]["position_embeddings"]["weight"]
    val = torch2np(val, np_weight_data_type)
    val.tofile((saved_dir / "model.wpe.bin").as_posix())  # not weight, do not need to transpose

    val_list = [
        torch2np(sub_model_states_list[0][model_key]["language_model"]["embedding"]["word_embeddings"]["weight"],
            np_weight_data_type)
        for sub_model_states_list in model_states_list
    ]
    val = np.concatenate(val_list, axis=0)
    vocab_size = val.shape[0]
    val.tofile((saved_dir / "model.wte.bin").as_posix())
    # save vocab_size
    if not hasattr(model_training_args, "padded_vocab_size"):
        model_training_args.padded_vocab_size = vocab_size

    structure_config = {
        "gpt_with_moe": 0,
        "expert_num": 0,
        "moe_layers": [],
    }
    model_training_args_vars = vars(model_training_args)
    num_experts = 0
    if 'num_experts' in model_training_args_vars.keys():
        num_experts = model_training_args_vars['num_experts'][0]
    if num_experts != 0:
        structure_config["gpt_with_moe"] = 1
    structure_config['expert_num'] = num_experts

    if inference_tensor_para_size >= training_tensor_para_dense_size:
        assert inference_tensor_para_size % training_tensor_para_dense_size == 0
        factor_dense = inference_tensor_para_size // training_tensor_para_dense_size
        moe_layers, has_adapters = save_dense_split(model_states_list, factor_dense, model_key, megatron_gpt_key,
                np_weight_data_type, saved_dir, ckpt_ver, model_training_args)
    else:
        assert training_tensor_para_dense_size % inference_tensor_para_size == 0
        factor_dense = training_tensor_para_dense_size // inference_tensor_para_size
        moe_layers, has_adapters = save_dense_concat(model_states_list, inference_tensor_para_size, factor_dense,
                model_key, megatron_gpt_key, np_weight_data_type, saved_dir, ckpt_ver, model_training_args)

    if inference_tensor_para_size >= training_tensor_para_expert_size:
        assert inference_tensor_para_size % training_tensor_para_expert_size == 0
        factor_expert = inference_tensor_para_size // training_tensor_para_expert_size
        save_experts_split(moe_layers, num_experts, training_tensor_para_dense_size, training_tensor_para_expert_size,
                factor_expert, args, map_location_fn, is_deepspeed, np_weight_data_type, saved_dir,
                training_pipeline_para_expert_size)
    else:
        raise NotImplementedError

    torch.cuda.synchronize()

    structure_config['moe_layers'] = moe_layers

    # Configuration for the model (load by triton backends)
    config = configparser.ConfigParser()
    config["gpt"] = {}

    if args.vocab_path:
        vocab_path = pathlib.Path(args.vocab_path)
        with vocab_path.open("r") as vocab_file:
            vocab = json.load(vocab_file)
        start_id, end_id = vocab[DEFAULT_START_TAG], vocab[DEFAULT_END_TAG]
    else:
        # hard coded values from english gpt_vocab.json file
        start_id, end_id = str(OPENAI_GPT2_START_ID), str(OPENAI_GPT2_END_ID)
    try:
        config["gpt"]["model_name"] = "gpt"
        config["gpt"]["head_num"] = str(model_training_args.num_attention_heads)
        config["gpt"]["size_per_head"] = str(model_training_args.hidden_size // model_training_args.num_attention_heads)
        config["gpt"]["inter_size"] = str(model_training_args.ffn_hidden_size)
        config["gpt"]["num_layer"] = str(model_training_args.num_layers)
        config["gpt"]["max_pos_seq_len"] = str(model_training_args.max_position_embeddings)
        config["gpt"]["vocab_size"] = str(model_training_args.padded_vocab_size)
        config["gpt"]["has_adapters"] = str(has_adapters)
        config['gpt']['adapter_inter_size'] = str(model_training_args.project_size) if has_adapters else str(0)
        config["gpt"]["layernorm_eps"] = str(model_training_args.layernorm_epsilon)
        config["gpt"]["start_id"] = str(start_id)
        config["gpt"]["end_id"] = str(end_id)
        config["gpt"]["weight_data_type"] = args.weight_data_type
        config["gpt"]["tensor_para_size"] = str(args.infer_gpu_num)
        config["structure"] = structure_config
        with open((saved_dir / f"config.ini").as_posix(), 'w') as configfile:
            config.write(configfile)
    except Exception as e:
        print(f"Fail to save the config in config.ini: {e}")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input-dir", "-input_dir", "-i", help="folder name of checkpoint files", required=True)
    parser.add_argument("--saved-dir", "-saved_dir", "-o", help="folder name of output files", required=True)
    parser.add_argument(
        "--infer-gpu-num", "-infer_gpu_num", "-i_g", type=int, help="How many gpus for inference", required=True
    )
    parser.add_argument(
        "--weight-data-type", "-weight_data_type", choices=["fp32", "fp16"], default="fp16", help=""
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
    parser.add_argument(
        "--vocab-path",
        type=str,
        help="Path to vocabulary file to embed in FasterTransformer checkpoint",
        required=False,
    )
    parser.add_argument(
        "--merges-path", type=str, help="Path to merges file to embed in FasterTransformer checkpoint", required=False
    )

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    print("========================================")

    start_time = datetime.datetime.now()
    convert_checkpoint(args)
    run_time = datetime.datetime.now() - start_time
    print(f"[INFO] Spent {run_time} (h:m:s) to convert the model")


if __name__ == "__main__":
    main()
