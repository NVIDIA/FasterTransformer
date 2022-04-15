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

import os
import argparse
import configparser
import multiprocessing
from pathlib import Path
import tarfile
import tempfile
import numpy as np
import torch  # pytype: disable=import-error
import yaml

def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    else:
        assert False, f"Invalid weight data type {data_type}"

def unpack_nemo_ckpt(nemo_ckpt_path, out_folder):
    """
    .nemo file is an archive (tar.gz) with the following:
        model_config.yaml - model configuration in .yaml format. You can deserialize this into cfg argument for model's constructor
        model_wights.chpt - model checkpoint
    """
    if not os.path.exists(nemo_ckpt_path):
        raise FileNotFoundError(f"{nemo_ckpt_path} does not exist")
    tar = tarfile.open(nemo_ckpt_path, "r:gz")
    tar.extractall(path=out_folder)
    tar.close()
    return out_folder


def _cpu_map_location(storage, loc):
    return storage.cpu()


def _gpu_map_location(storage, loc):
    if loc.startswith("cuda"):
        training_gpu_idx = int(loc.split(":")[1])
        inference_gpu_idx = training_gpu_idx % torch.cuda.device_count()
        return storage.cuda(inference_gpu_idx)
    elif loc.startswith("cpu"):
        return storage.cpu()
    else:
        raise NotImplementedError(f"Not handled {loc}")


# more to less. e.g., trained by 8 gpus, infer by 2 gpus
def merge_and_convert(
    args, model_config, weight_files, *, load_checkpoints_to_cpu: bool = False
):  # noqa: C901 too complex
    saved_dir = Path(args.saved_dir)
    if args.fused_qkv == 1:
        saved_dir = saved_dir / f"{args.infer_gpu_num:d}-gpu/"
    else:
        saved_dir = saved_dir / f"unfusedQKV-{args.infer_gpu_num:d}-gpu"

    saved_dir.mkdir(parents=True, exist_ok=True)

    config = configparser.ConfigParser()
    config["gpt"] = {}

    try:
        for key in vars(args):
            config["gpt"][key] = f"{vars(args)[key]}"
        for k, v in model_config.items():
            config["gpt"][k] = f"{v}"
        config["gpt"]["weight_data_type"] = args.weight_data_type
        with open((saved_dir / f"config.ini").as_posix(), 'w') as configfile:
            config.write(configfile)
    except:
        print(f"Fail to save the config in config.ini.")

    np_weight_data_type = get_weight_data_type(args.weight_data_type)

    prefix = Path(args.in_file)
    i_gpu_num = args.infer_gpu_num

    t_gpu_num = model_config["tensor_model_parallel_size"]
    num_attention_heads = model_config["num_attention_heads"]

    assert t_gpu_num % i_gpu_num == 0
    factor = int(t_gpu_num / i_gpu_num)

    num_checkpoints_per_convert = max(factor, 1)
    if num_checkpoints_per_convert > torch.cuda.device_count():
        print(
            f"[WARNING] Need to load #{num_checkpoints_per_convert} checkpoints at once "
            f"while having {torch.cuda.device_count()} GPUs. Force load checkpoints on CPU"
        )
        load_checkpoints_to_cpu = True

    map_location_fn = _cpu_map_location if load_checkpoints_to_cpu else _gpu_map_location

    # load position_embedding from rank 0
    model_00 = torch.load(weight_files[0], map_location=map_location_fn)
    model_00["model.language_model.embedding.position_embeddings.weight"].float().cpu().numpy().astype(
        np_weight_data_type
    ).tofile(
        (saved_dir / "model.wpe.bin").as_posix()
    )  # not weight, do not need transpose

    del model_00
    w_e_list = []
    for i in range(i_gpu_num):
        transformer_models = []
        for j in range(factor):
            model = torch.load(weight_files[i * factor + j], map_location=map_location_fn)

            w_e_list.append(
                model["model.language_model.embedding.word_embeddings.weight"]
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np_weight_data_type)
            )

            prefix = "model.language_model.encoder"
            model["model"] = {}
            model["model"]["language_model"] = {}
            model["model"]["language_model"]["encoder"] = {}
            model["model"]["language_model"]["embedding"] = {}
            model["model"]["language_model"]["embedding"]["word_embeddings"] = {}
            model["model"]["language_model"]["embedding"]["position_embeddings"] = {}
            model["model"]["language_model"]["embedding"]["word_embeddings"]["weight"] = model[
                "model.language_model.embedding.word_embeddings.weight"]
            model["model"]["language_model"]["embedding"]["position_embeddings"]["weight"] = model[
                "model.language_model.embedding.position_embeddings.weight"]
            for key in model.keys():
                if prefix in key:
                    first = key[:len(prefix)]
                    second = key[len(prefix) + 1:]
                    model["model"]["language_model"]["encoder"][second] = model[key]

            # print(model["model"]["language_model"]["encoder"].keys())

            # this model should be able to load into megatron
            # torch.save(model, "model.pt")

            transformer_models.append(model["model"]["language_model"]["encoder"])

        for key in transformer_models[0]:
            if (
                    key.find("input_layernorm.weight") != -1
                    or key.find("input_layernorm.bias") != -1
                    or key.find("attention.dense.bias") != -1
                    or key.find("post_attention_layernorm.weight") != -1
                    or key.find("post_attention_layernorm.bias") != -1
                    or key.find("mlp.dense_4h_to_h.bias") != -1
                    or key.find("final_layernorm.weight") != -1
                    or key.find("final_layernorm.bias") != -1
            ):
                # shared weights, only need to convert the weights of rank 0
                if i == 0:
                    val = transformer_models[0][key].T.float().cpu().numpy()
                    key = key.replace("self_attention", "attention")
                    saved_path = saved_dir / f"model.{key}.bin"
                    np.squeeze(val).astype(np_weight_data_type).tofile(saved_path.as_posix())

            elif key.find("attention.dense.weight") != -1 or key.find("mlp.dense_4h_to_h.weight") != -1:
                vals = []
                for k in range(factor):
                    vals.append(transformer_models[k][key].T.float().cpu().numpy())
                key = key.replace("self_attention", "attention")
                saved_path = saved_dir / f"model.{key}.{i}.bin"
                np.concatenate(vals, axis=0).astype(np_weight_data_type).tofile(saved_path.as_posix())

            elif key.find("mlp.dense_h_to_4h.weight") != -1 or key.find("mlp.dense_h_to_4h.bias") != -1:

                vals = []
                for k in range(factor):
                    vals.append(transformer_models[k][key].T.float().cpu().numpy())
                saved_path = saved_dir / f"model.{key}.{i}.bin"
                np.concatenate(vals, axis=-1).astype(np_weight_data_type).tofile(saved_path.as_posix())

            elif key.find("attention.query_key_value.bias") != -1:
                vals = []
                for k in range(factor):
                    val = transformer_models[k][key].T.float().cpu().numpy()
                    local_dim = (int)(val.shape[-1] / 3)
                    num_splits = 3
                    head_num = num_attention_heads // t_gpu_num
                    size_per_head = local_dim // head_num
                    val = val.reshape(head_num, num_splits, size_per_head)
                    val = val.transpose(1, 0, 2)
                    val = val.reshape(3, local_dim)
                    vals.append(val)

                key = key.replace("self_attention", "attention")
                saved_path = saved_dir / f"model.{key}.{i}.bin"
                np.concatenate(vals, axis=-1).astype(np_weight_data_type).tofile(saved_path.as_posix())

            elif key.find("attention.query_key_value.weight") != -1:
                vals = []
                for k in range(factor):
                    val = transformer_models[k][key].T.float().cpu().numpy()
                    hidden_dim = val.shape[0]
                    local_dim = (int)(val.shape[-1] / 3)
                    num_splits = 3
                    head_num = num_attention_heads // t_gpu_num
                    size_per_head = local_dim // head_num
                    val = val.reshape(hidden_dim, head_num, num_splits, size_per_head)
                    val = val.transpose(0, 2, 1, 3)
                    val = val.reshape(hidden_dim, 3, local_dim)
                    vals.append(val)

                key = key.replace("self_attention", "attention")
                saved_path = saved_dir / f"model.{key}.{i}.bin"
                if args.fused_qkv == 1:
                    np.concatenate(vals, axis=-1).astype(np_weight_data_type).tofile(saved_path.as_posix())
                elif args.fused_qkv == 0:
                    np.concatenate(vals, axis=-1).transpose(1, 0, 2).astype(np_weight_data_type).tofile(saved_path.as_posix())

            else:
                print(f"[ERROR] cannot find key '{key}'")

    np.concatenate(w_e_list, axis=0).tofile((saved_dir / "model.wte.bin").as_posix())


def split_and_convert(args, model_config, weight_files, *, load_checkpoints_to_cpu: bool = False):
    saved_dir = Path(args.saved_dir)
    if args.fused_qkv == 1:
        saved_dir = saved_dir / f"{args.infer_gpu_num:d}-gpu/"
    else:
        saved_dir = saved_dir / f"unfusedQKV-{args.infer_gpu_num:d}-gpu"

    saved_dir.mkdir(parents=True, exist_ok=True)
    
    config = configparser.ConfigParser()
    config["gpt"] = {} 
    try:
        for key in vars(args):
            config["gpt"][key] = f"{vars(args)[key]}"
        for k, v in model_config.items():
            config["gpt"][k] = f"{v}"
        config["gpt"]["weight_data_type"] = args.weight_data_type
        with open((saved_dir / f"config.ini").as_posix(), 'w') as configfile:
            config.write(configfile)
    except:
        print(f"Fail to save the config in config.ini.")

    np_weight_data_type = get_weight_data_type(args.weight_data_type)
    prefix = Path(args.in_file)

    i_gpu_num = args.infer_gpu_num
    t_gpu_num = model_config["tensor_model_parallel_size"]
    num_attention_heads = model_config["num_attention_heads"]

    assert i_gpu_num % t_gpu_num == 0
    factor = int(i_gpu_num / t_gpu_num)

    num_checkpoints_per_convert = max(int(1 / (i_gpu_num / t_gpu_num)), 1)
    if num_checkpoints_per_convert > torch.cuda.device_count():
        print(
            f"[WARNING] Need to load #{num_checkpoints_per_convert} checkpoints at once "
            f"while having {torch.cuda.device_count()} GPUs. Force load checkpoints on CPU"
        )
        load_checkpoints_to_cpu = True

    map_location_fn = _cpu_map_location if load_checkpoints_to_cpu else _gpu_map_location

    # load position_embedding from rank 0
    model_00 = torch.load(weight_files[0], map_location=map_location_fn)
    model_00["model.language_model.embedding.position_embeddings.weight"].float().cpu().numpy().astype(
        np_weight_data_type
    ).tofile(
        (saved_dir / "model.wpe.bin").as_posix()
    )  # not weight, do not need transpose
    del model_00

    w_e_list = []

    # main_loop = min(t_gpu_num, i_gpu_num)
    for i in range(t_gpu_num):
        model = torch.load(weight_files[i], map_location=map_location_fn)

        w_e_list.append(
            model["model.language_model.embedding.word_embeddings.weight"]
                .float()
                .cpu()
                .numpy()
                .astype(np_weight_data_type)
        )

        prefix = "model.language_model.encoder"
        # Build dictionary
        model["model"] = {}
        model["model"]["language_model"] = {}
        model["model"]["language_model"]["encoder"] = {}
        model["model"]["language_model"]["embedding"] = {}
        model["model"]["language_model"]["embedding"]["word_embeddings"] = {}
        model["model"]["language_model"]["embedding"]["position_embeddings"] = {}
        model["model"]["language_model"]["embedding"]["word_embeddings"]["weight"] = model[
            "model.language_model.embedding.word_embeddings.weight"]
        model["model"]["language_model"]["embedding"]["position_embeddings"]["weight"] = model[
            "model.language_model.embedding.position_embeddings.weight"]

        for key in model.keys():
            if prefix in key:
                first = key[:len(prefix)]
                second = key[len(prefix) + 1:]
                model["model"]["language_model"]["encoder"][second] = model[key]

        transformer_model = model["model"]["language_model"]["encoder"]

        for key in transformer_model:
            val = transformer_model[key].T.float().cpu().numpy().astype(np_weight_data_type)
            if key.find("layers.") != -1:
                layer_index = (int)(key[7: key.find(".", 7)])
                saved_key = key
                # saved_key = key.replace(
                #     "layers.%d." % layer_index,
                #     "layers.%d." % (layer_index + pipeline_para_rank * model_args.num_layers // model_args.pipeline_model_parallel_size))

                if saved_key.find("self_attention") != -1:
                    saved_key = saved_key.replace("self_attention", "attention")
            else:
                saved_key = key

            if (
                    key.find("input_layernorm.weight") != -1
                    or key.find("input_layernorm.bias") != -1
                    or key.find("attention.dense.bias") != -1
                    or key.find("post_attention_layernorm.weight") != -1
                    or key.find("post_attention_layernorm.bias") != -1
                    or key.find("mlp.dense_4h_to_h.bias") != -1
                    or key.find("final_layernorm.weight") != -1
                    or key.find("final_layernorm.bias") != -1
            ):
                # shared weights, only need to convert the weights of rank 0
                if i == 0:
                    saved_path = saved_dir / f"model.{saved_key}.bin"
                    val.tofile(saved_path.as_posix())

            elif key.find("attention.dense.weight") != -1 or key.find("mlp.dense_4h_to_h.weight") != -1:
                split_vals = np.split(val, factor, axis=0)
                for j in range(factor):
                    saved_path = saved_dir / f"model.{saved_key}.{i * factor + j:d}.bin"
                    split_vals[j].tofile(saved_path.as_posix())

            elif key.find("mlp.dense_h_to_4h.weight") != -1 or key.find("mlp.dense_h_to_4h.bias") != -1:
                split_vals = np.split(val, factor, axis=-1)
                for j in range(factor):
                    saved_path = saved_dir / f"model.{saved_key}.{i * factor + j:d}.bin"
                    split_vals[j].tofile(saved_path.as_posix())

            elif key.find("attention.query_key_value.bias") != -1:
                local_dim = int(val.shape[-1] / 3)

                # ckpt_ver == 3
                num_splits = 3
                head_num = num_attention_heads // t_gpu_num
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

                # ckpt_ver == 3:
                num_splits = 3
                head_num = num_attention_heads
                size_per_head = hidden_dim // head_num
                head_num = head_num // t_gpu_num
                val = val.reshape(hidden_dim, head_num, num_splits, size_per_head)
                val = val.transpose(0, 2, 1, 3)

                val = val.reshape(hidden_dim, 3, local_dim)
                split_vals = np.split(val, factor, axis=-1)

                for j in range(factor):
                    saved_path = saved_dir / f"model.{saved_key}.{i * factor + j:d}.bin"
                    split_vals[j].tofile(saved_path.as_posix())
                    # print(split_vals[j].shape)

            else:
                print(f"[ERROR] cannot find key '{key}'")

    np.concatenate(w_e_list, axis=0).tofile((saved_dir / "model.wte.bin").as_posix())
    # print(torch.from_numpy(np.fromfile((saved_dir / "model.wte.bin").as_posix(), dtype=np.single)).size())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-saved_dir", "-o", type=str, help="folder name of output files", required=True)
    parser.add_argument("-in_file", "-i", type=str, help="file name of .nemo checkpoint file", required=True)
    parser.add_argument("-infer_gpu_num", "-i_g", type=int, help="How many gpus for inference", required=True)
    parser.add_argument(
        "-fused_qkv",
        "-fused_qkv",
        type=int,
        default=1,
        help="Fuse the qkv weights or not. Default is true (1)",
        choices=[0, 1],
    )
    parser.add_argument("-weight_data_type", type=str, default="fp32", choices=["fp32", "fp16"])
    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    print("========================================")

    model_config_yaml = "model_config.yaml"
    model_weights_ckpt = "model_weights.ckpt"
    config_yaml = os.path.join(args.saved_dir, model_config_yaml)

    unpack_nemo_ckpt(args.in_file, args.saved_dir)

    with open(config_yaml) as f:
        model_config = yaml.full_load(f)

    t_gpu_num = model_config["tensor_model_parallel_size"]
    if t_gpu_num == 1:
        model_weights = [os.path.join(args.saved_dir, model_weights_ckpt)]
    else:
        model_weights = [os.path.join(args.saved_dir, f"mp_rank_{i:02d}", model_weights_ckpt) for i in range(t_gpu_num)]

    print(model_config)
    if t_gpu_num > args.infer_gpu_num:
        merge_and_convert(args, model_config, model_weights)
    else:
        split_and_convert(args, model_config, model_weights)
