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
import multiprocessing
import pathlib
import re
import shutil
import sys

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


def _inject_model_parallel_rank(
    filepath,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    tensor_model_parallel_rank=0,
    pipeline_model_parallel_rank=0,
):
    """
    Injects tensor/pipeline model parallel ranks into the filepath.
    Does nothing if not using model parallelism.
    """
    filepath = pathlib.Path(filepath)
    if tensor_model_parallel_size > 1 or pipeline_model_parallel_size > 1:
        # filepath needs to be updated to include mp_rank
        if pipeline_model_parallel_size is None or pipeline_model_parallel_size == 1:
            filepath = filepath.parent / f"mp_rank_{tensor_model_parallel_rank:02d}" / filepath.name
        else:
            filepath = (
                    filepath.parent /
                    f"mp_rank_{tensor_model_parallel_rank:02d}_{pipeline_model_parallel_rank:03d}" /
                    filepath.name
            )
            if not filepath.exists():
                filepath = (
                    filepath.parent /
                    f"tp_rank_{tensor_model_parallel_rank:02d}_pp_rank_{pipeline_model_parallel_rank:03d}" /
                    filepath.name
                )
        return filepath
    else:
        if filepath.exists():
            return filepath
        else:
            return filepath.parent / "mp_rank_00" / filepath.name


def _create_model_training_args_for_checkpoint_version_0(args, model_00):
    model_training_args = argparse.Namespace()
    if args.head_num is None or args.trained_tensor_parallel_size is None:
        raise ValueError(
            "Provided checkpoint have missing training args. "
            "Thus it is required to provide -head_num and -trained_tensor_parallel_size CLI arguments"
        )
    model_training_args.num_attention_heads = args.head_num
    model_training_args.tensor_model_parallel_size = args.trained_tensor_parallel_size
    # megatron ckpt_ver=0 only supports pipeline_parallel_size = 1
    model_training_args.pipeline_model_parallel_size = 1
    model_training_args.max_position_embeddings = \
        model_00["model"]["language_model"]["embedding"]["position_embeddings"]["weight"].shape[0]
    model_training_args.hidden_size = \
        model_00["model"]["language_model"]["embedding"]["position_embeddings"]["weight"].shape[1]
    model_training_args.ffn_hidden_size = 4 * model_training_args.hidden_size

    def get_layer_num_from_weights(model_keys):
        layer_num = 1
        for key in model_keys:
            if re.search(r'\d+', key) is not None:
                layer_num = max(int(re.search(r'\d+', key).group()), layer_num)
        return layer_num + 1

    model_training_args.num_layers = \
        get_layer_num_from_weights(model_00["model"]["language_model"]['transformer'].keys())

    model_training_args.layernorm_epsilon = 1e-6

    return model_training_args


# This tool is used to support the new megatron model trained by pipeline parallel + tensor parallel
def merge_and_convert_process(i, pipeline_para_rank, saved_dir, factor, key, model_training_args, transformer_model_list, ckpt_ver, np_weight_data_type):
    saved_dir = pathlib.Path(saved_dir)
    if key.find("layers.") != -1:
        layer_index = (int)(key[7 : key.find(".", 7)])
        saved_key = key.replace(
            "layers.%d." % layer_index,
            "layers.%d." % (layer_index + pipeline_para_rank * model_training_args.num_layers // model_training_args.pipeline_model_parallel_size))

        if saved_key.find("self_attention") != -1:
            saved_key = saved_key.replace("self_attention", "attention")
        if saved_key.find("adaptor1") != -1:
            saved_key = saved_key.replace("adaptor1", "after_attention_adapter")
        if saved_key.find("adaptor2") != -1:
            saved_key = saved_key.replace("adaptor2", "after_ffn_adapter")
    else:
        saved_key = key
    major_device = transformer_model_list[0][key].device

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
        or key.find("final_layernorm.bias") != -1):

        # shared weights, only need to convert the weights of rank 0
        if i == 0:
            saved_path = saved_dir / f"model.{saved_key}.bin"
            val = safe_transpose(transformer_model_list[0][key])
            val = torch2np(val, np_weight_data_type)
            val = np.squeeze(val)
            val.tofile(saved_path)

    elif (key.find("attention.dense.weight") != -1
        or key.find("mlp.dense_4h_to_h.weight") != -1
        or key.find("adaptor1.dense_4h_to_h.weight") != -1
        or key.find("adaptor2.dense_4h_to_h.weight") != -1):
        vals = [
            safe_transpose(transformer_model_list[k][key]).float().to(major_device)
            for k in range(factor)
        ]
        val = torch.cat(vals, dim=0)
        val = torch2np(val, np_weight_data_type)
        saved_path = saved_dir / f"model.{saved_key}.{i:d}.bin"
        val.tofile(saved_path)

    elif (key.find("mlp.dense_h_to_4h.weight") != -1
        or key.find("adaptor1.dense_h_to_4h.weight") != -1
        or key.find("adaptor2.dense_h_to_4h.weight") != -1
        or key.find("mlp.dense_h_to_4h.bias") != -1
        or key.find("adaptor1.dense_h_to_4h.bias") != -1
        or key.find("adaptor2.dense_h_to_4h.bias") != -1):
        vals = [
            safe_transpose(transformer_model_list[k][key]).float().to(major_device)
            for k in range(factor)
        ]
        val = torch.cat(vals, dim=-1)
        val = torch2np(val, np_weight_data_type)
        saved_path = saved_dir / f"model.{saved_key}.{i:d}.bin"
        val.tofile(saved_path)

    elif key.find("attention.query_key_value.bias") != -1:
        vals = []
        for k in range(factor):
            val = safe_transpose(transformer_model_list[k][key]).float()
            local_dim = int(val.shape[-1] / 3)
            if ckpt_ver == 3:
                num_splits = 3
                head_num = model_training_args.num_attention_heads // model_training_args.tensor_model_parallel_size
                size_per_head = local_dim // head_num
                val = val.reshape(head_num, num_splits, size_per_head)
                val = val.permute(1, 0, 2)
            val = val.reshape(3, local_dim)
            vals.append(val.to(major_device))
        val = torch.cat(vals, dim=-1)
        val = torch2np(val, np_weight_data_type)
        saved_path = saved_dir / f"model.{saved_key}.{i:d}.bin"
        val.tofile(saved_path)

    elif key.find("attention.query_key_value.weight") != -1:
        vals = []
        for k in range(factor):
            val = safe_transpose(transformer_model_list[k][key]).float()
            hidden_dim = val.shape[0]
            local_dim = int(val.shape[-1] / 3)
            if ckpt_ver == 3:
                num_splits = 3
                head_num = model_training_args.num_attention_heads
                size_per_head = hidden_dim // head_num
                head_num = head_num // model_training_args.tensor_model_parallel_size
                val = val.reshape(hidden_dim, head_num, num_splits, size_per_head)
                val = val.permute(0, 2, 1, 3)
            val = val.reshape(hidden_dim, 3, local_dim)
            vals.append(val.to(major_device))
        val = torch.cat(vals, dim=-1)
        val = torch2np(val, np_weight_data_type)
        saved_path = saved_dir / f"model.{saved_key}.{i:d}.bin"
        val.tofile(saved_path)
        
    else:
        print(f"[ERROR] cannot find key '{key}'")


def split_and_convert_process(i, pipeline_para_rank, saved_dir, factor, key, model_training_args, transformer_model_list, ckpt_ver, np_weight_data_type):
    val = safe_transpose(transformer_model_list[0][key])
    val = torch2np(val, np_weight_data_type)
    if key.find("layers.") != -1:
        layer_index = (int)(key[7 : key.find(".", 7)])
        saved_key = key.replace(
            "layers.%d." % layer_index,
            "layers.%d." % (layer_index + pipeline_para_rank * model_training_args.num_layers // model_training_args.pipeline_model_parallel_size))

        if saved_key.find("self_attention") != -1:
            saved_key = saved_key.replace("self_attention", "attention")
        if saved_key.find("adaptor1") != -1:
            saved_key = saved_key.replace("adaptor1", "after_attention_adapter")
        if saved_key.find("adaptor2") != -1:
            saved_key = saved_key.replace("adaptor2", "after_ffn_adapter")
    else:
        saved_key = key

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
        if i == 0:
            saved_path = saved_dir / f"model.{saved_key}.bin"
            val.tofile(saved_path.as_posix())

    elif (key.find("attention.dense.weight") != -1
        or key.find("mlp.dense_4h_to_h.weight") != -1
        or key.find("adaptor1.dense_4h_to_h.weight") != -1
        or key.find("adaptor2.dense_4h_to_h.weight") != -1):
        split_vals = np.split(val, factor, axis=0)
        for j in range(factor):
            saved_path = saved_dir / f"model.{saved_key}.{i * factor + j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())

    elif (key.find("mlp.dense_h_to_4h.weight") != -1
        or key.find("adaptor1.dense_h_to_4h.weight") != -1
        or key.find("adaptor2.dense_h_to_4h.weight") != -1
        or key.find("mlp.dense_h_to_4h.bias") != -1
        or key.find("adaptor1.dense_h_to_4h.bias") != -1
        or key.find("adaptor2.dense_h_to_4h.bias") != -1):
        split_vals = np.split(val, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir / f"model.{saved_key}.{i * factor + j:d}.bin"
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
        split_vals = np.split(val, factor, axis=-1)

        for j in range(factor):
            saved_path = saved_dir / f"model.{saved_key}.{i * factor + j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())

    elif key.find("attention.query_key_value.weight") != -1:
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
        split_vals = np.split(val, factor, axis=-1)

        for j in range(factor):
            saved_path = saved_dir / f"model.{saved_key}.{i * factor + j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())

    else:
        print(f"[ERROR] cannot find key '{key}'")


def _get_checkpoint_name(checkpoint_dir):

    checkpoint_dir = pathlib.Path(checkpoint_dir)
    patterns = [
        "model_optim_rng.pt",  # older megatron checkpoints
        "*last.ckpt",  # newer format of checkpoints
    ]
    for pattern in patterns:
        model_files = sorted(list(checkpoint_dir.rglob(pattern)))
        if model_files:
            return model_files[0].name

    raise ValueError(f"Could not find checkpoint files in {checkpoint_dir}")


def convert_checkpoint(args):
    saved_dir = pathlib.Path(args.saved_dir) / f"{args.infer_gpu_num:d}-gpu"
    if saved_dir.exists():
        print(f"[ERROR] Remove {saved_dir} target directory before running conversion")
        sys.exit(1)
    saved_dir.mkdir(parents=True)

    if args.vocab_path:
        shutil.copy(args.vocab_path, (saved_dir / "vocab.json").as_posix())
    if args.merges_path:
        shutil.copy(args.merges_path, (saved_dir / "merges.txt").as_posix())

    load_checkpoints_to_cpu = bool(args.load_checkpoints_to_cpu)
    map_location_fn = cpu_map_location if load_checkpoints_to_cpu else gpu_map_location

    checkpoints_dir = pathlib.Path(args.in_file)
    checkpoint_name = _get_checkpoint_name(checkpoints_dir)

    # load position_embedding from rank 0
    checkpoints_paths = sorted(checkpoints_dir.rglob(checkpoint_name))
    if not checkpoints_paths:
        print(f"[ERROR] Cannot find checkpoint in {checkpoints_dir}.")
        exit(1)
    model_00 = torch.load(checkpoints_paths[0].as_posix(), map_location=map_location_fn)

    if "hyper_parameters" in list(model_00.keys()):
        print("Use nemo_ckpt_converter.py script for conversion of this checkpoint")
        exit(1)
    elif "args" in list(model_00.keys()):
        checkpoint_version = model_00["checkpoint_version"]
        model_training_args = model_00["args"]
        megatron_gpt_key = "encoder"
    else:
        checkpoint_version = 0
        model_training_args = _create_model_training_args_for_checkpoint_version_0(args, model_00)
        megatron_gpt_key = "transformer"

    with (saved_dir / "args.txt").open("w") as training_args_file:
        for k, v in vars(model_training_args).items():
            training_args_file.write(f"{k}:{v}\n")

    np_weight_data_type = WEIGHT2DTYPE[args.weight_data_type]

    val = model_00["model"]["language_model"]["embedding"]["position_embeddings"]["weight"]
    val = torch2np(val, np_weight_data_type)
    val.tofile((saved_dir / "model.wpe.bin").as_posix())  # not weight, do not need to transpose

    del model_00
    w_e_list = []

    training_tensor_para_size = model_training_args.tensor_model_parallel_size
    training_pipeline_para_size = model_training_args.pipeline_model_parallel_size
    inference_tensor_para_size = args.infer_gpu_num

    model_weights_paths = [
        [
            _inject_model_parallel_rank(
                checkpoints_dir / checkpoint_name,
                tensor_model_parallel_size=training_tensor_para_size,
                pipeline_model_parallel_size=training_pipeline_para_size,
                tensor_model_parallel_rank=tp_rank,
                pipeline_model_parallel_rank=pp_rank,
            )
            for pp_rank in range(training_pipeline_para_size)
        ]
        for tp_rank in range(training_tensor_para_size)
    ]

    if training_tensor_para_size > inference_tensor_para_size:
        assert training_tensor_para_size % inference_tensor_para_size == 0
        is_merge_ckpt = True
        factor = int(training_tensor_para_size / inference_tensor_para_size)
    else:
        assert inference_tensor_para_size % training_tensor_para_size == 0
        is_merge_ckpt = False
        factor = int(inference_tensor_para_size / training_tensor_para_size)

    main_loop = min(training_tensor_para_size, inference_tensor_para_size)
    vocab_size_list = [0 for i in range(main_loop)]
    
    torch.multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy("file_system")
    pool = multiprocessing.Pool(args.processes)
    has_adapters = False
    for i in range(main_loop):
        for j in range(training_pipeline_para_size):
            
            transformer_models = []
            if is_merge_ckpt:
                for k in range(factor):
                    m = torch.load(model_weights_paths[i * factor + k][j].as_posix(), map_location=map_location_fn)
                    if not has_adapters:
                        has_adapters = any("adaptor" in key for key in m['model']['language_model'][megatron_gpt_key].keys())
                    transformer_models.append(m["model"]["language_model"][megatron_gpt_key])

                    if j == 0:
                        vocab_size_list[i] = m["model"]["language_model"]["embedding"]["word_embeddings"]["weight"].shape[0]
                        w_e_list.append(torch2np(m["model"]["language_model"]["embedding"]["word_embeddings"]["weight"], np_weight_data_type))
            else:
                m = torch.load(model_weights_paths[i][j].as_posix(), map_location=map_location_fn)
                if not has_adapters:
                    has_adapters = any("adaptor" in key for key in m['model']['language_model'][megatron_gpt_key].keys())
            
                if j == 0:
                    vocab_size_list[i] = m["model"]["language_model"]["embedding"]["word_embeddings"]["weight"].shape[0]
                    w_e_list.append(torch2np(
                        m["model"]["language_model"]["embedding"]["word_embeddings"]["weight"],
                        np_weight_data_type
                    ))
                transformer_models.append(m["model"]["language_model"][megatron_gpt_key])

            pool.starmap(
                merge_and_convert_process if is_merge_ckpt else split_and_convert_process,
                [
                    (
                        i,
                        j,
                        saved_dir,
                        factor,
                        k,
                        model_training_args,
                        transformer_models,
                        checkpoint_version,
                        np_weight_data_type,
                    )
                    for (k, v) in transformer_models[0].items()
                ],
            )

    pool.close()
    pool.join()

    torch.cuda.synchronize()

    np.concatenate(w_e_list, axis=0).tofile((saved_dir / "model.wte.bin").as_posix())

    # save vocab_size
    full_vocab_size = sum(vocab_size_list)
    if not hasattr(model_training_args, "padded_vocab_size"):
        model_training_args.padded_vocab_size = full_vocab_size

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
        with open((saved_dir / f"config.ini").as_posix(), 'w') as configfile:
            config.write(configfile)
    except Exception as e:
        print(f"Fail to save the config in config.ini: {e}")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--saved-dir", "-saved_dir", "-o", help="folder name of output files", required=True)
    parser.add_argument(
        "--in-file", "-in_file", "-i", help="file name of input checkpoint file", required=True
    )
    parser.add_argument(
        "--infer-gpu-num", "-infer_gpu_num", "-i_g", type=int, help="How many gpus for inference", required=True
    )
    # -h_n and -t_g are needed when megatron_ckpt_version = 0, for example the public megatron 345M gpt model
    parser.add_argument(
        "--head-num",
        "-head_num",
        "-h_n",
        type=int,
        help="The number of heads, only needed when weight doesn't contain structure hyperparameters"
    )
    parser.add_argument(
        "--trained-tensor-parallel-size",
        "-trained_tensor_parallel_size",
        "-t_g",
        type=int,
        help="the tensor parallel size for training"
    )
    parser.add_argument(
        "--processes",
        "-processes",
        "-p",
        type=int,
        default=16,
        help="How many processes to spawn for conversion",
    )
    parser.add_argument(
        "--weight-data-type", "-weight_data_type", choices=["fp32", "fp16"], default="fp32", help=""
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
