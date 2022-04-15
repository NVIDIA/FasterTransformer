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
import multiprocessing
from pathlib import Path

import numpy as np
import torch  # pytype: disable=import-error

def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    else:
        assert False, f"Invalid weight data type {data_type}"

# more to less. e.g., trained by 8 gpus, infer by 2 gpus
def merge_and_convert(args):  # noqa: C901 too complex
    saved_dir = Path(args.saved_dir)
    if args.fused_qkv == 1:
        saved_dir = saved_dir / f"{args.infer_gpu_num:d}-gpu/"
    else:
        saved_dir = saved_dir / f"unfusedQKV-{args.infer_gpu_num:d}-gpu"
    ckpt_ver = args.checkpoint_version

    saved_dir.mkdir(parents=True, exist_ok=True)
    
    config = configparser.ConfigParser()
    config["gpt"] = {}
    for key in vars(args):
        config["gpt"][key] = f"{vars(args)[key]}"
    config["gpt"]["weight_data_type"] = args.weight_data_type
    with open((saved_dir / f"config.ini").as_posix(), 'w') as configfile:
        config.write(configfile)

    np_weight_data_type = get_weight_data_type(args.weight_data_type)
        
    prefix = Path(args.in_file)
    ckpt_name = "model_optim_rng.pt"
    t_gpu_num = args.trained_gpu_num
    i_gpu_num = args.infer_gpu_num

    assert t_gpu_num % i_gpu_num == 0
    factor = int(t_gpu_num / i_gpu_num)

    # load position_embedding from rank 0
    model_00 = torch.load((prefix / "mp_rank_00" / ckpt_name).as_posix())
    model_00["model"]["language_model"]["embedding"]["position_embeddings"]["weight"].cpu().numpy().astype(
        np_weight_data_type
    ).tofile(
        (saved_dir / "model.wpe.bin").as_posix()
    )  # not weight, do not need transpose

    del model_00
    w_e_list = []
    for i in range(i_gpu_num):
        transformer_models = []
        for j in range(factor):
            model = torch.load(prefix / f"mp_rank_{i * factor + j:02d}" / ckpt_name)
            w_e_list.append(
                model["model"]["language_model"]["embedding"]["word_embeddings"]["weight"]
                .cpu()
                .numpy()
                .astype(np_weight_data_type)
            )
            if ckpt_ver == 3:
                transformer_models.append(model["model"]["language_model"]["encoder"])
            else:                
                transformer_models.append(model["model"]["language_model"]["transformer"])

        for key in transformer_models[0]:
            print(key, transformer_models[0][key].shape)

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
                    val = transformer_models[0][key].T.cpu().numpy()
                    saved_path = saved_dir / f"model.{key}.bin"
                    np.squeeze(val).astype(np_weight_data_type).tofile(saved_path.as_posix())

            elif key.find("attention.dense.weight") != -1 or key.find("mlp.dense_4h_to_h.weight") != -1:
                vals = []
                for k in range(factor):
                    vals.append(transformer_models[k][key].T.cpu().numpy())
                saved_path = saved_dir / f"model.{key}.{i}.bin"
                np.concatenate(vals, axis=0).astype(np_weight_data_type).tofile(saved_path.as_posix())

            elif key.find("mlp.dense_h_to_4h.weight") != -1 or key.find("mlp.dense_h_to_4h.bias") != -1:

                vals = []
                for k in range(factor):
                    vals.append(transformer_models[k][key].T.cpu().numpy())
                saved_path = saved_dir / f"model.{key}.{i}.bin"
                np.concatenate(vals, axis=-1).astype(np_weight_data_type).tofile(saved_path.as_posix())

            elif key.find("attention.query_key_value.bias") != -1:
                vals = []
                for k in range(factor):
                    val = transformer_models[k][key].T.cpu().numpy()
                    local_dim = (int)(val.shape[-1] / 3)
                    if ckpt_ver == 3:
                        num_splits = 3
                        head_num = args.head_num // args.trained_gpu_num
                        size_per_head = local_dim // head_num
                        val = val.reshape(head_num, num_splits, size_per_head)
                        val = val.transpose(1, 0, 2)
                    val = val.reshape(3, local_dim)
                    vals.append(val)

                saved_path = saved_dir / f"model.{key}.{i}.bin"
                np.concatenate(vals, axis=-1).astype(np_weight_data_type).tofile(saved_path.as_posix())

            elif key.find("attention.query_key_value.weight") != -1:
                vals = []
                for k in range(factor):
                    val = transformer_models[k][key].T.cpu().numpy()
                    hidden_dim = val.shape[0]
                    local_dim = (int)(val.shape[-1] / 3)
                    if ckpt_ver == 3:
                        num_splits = 3
                        head_num = args.head_num
                        size_per_head = hidden_dim // head_num
                        head_num = head_num // args.trained_gpu_num
                        val = val.reshape(hidden_dim, head_num, num_splits, size_per_head)
                        val = val.transpose(0, 2, 1, 3)
                    val = val.reshape(hidden_dim, 3, local_dim)
                    vals.append(val)

                saved_path = saved_dir / f"model.{key}.{i}.bin"
                if args.fused_qkv == 1:
                    np.concatenate(vals, axis=-1).astype(np_weight_data_type).tofile(saved_path.as_posix())
                elif args.fused_qkv == 0:
                    np.concatenate(vals, axis=-1).transpose(1, 0, 2).astype(np_weight_data_type).tofile(saved_path.as_posix())

            else:
                print(f"[ERROR] cannot find key '{key}'")

    np.concatenate(w_e_list, axis=0).tofile((saved_dir / "model.wte.bin").as_posix())


def split_and_convert_process(i, saved_dir, factor, key, args, val, ckpt_ver):
    saved_dir = Path(saved_dir)
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
            saved_path = saved_dir / f"model.{key}.bin"
            val.tofile(saved_path.as_posix())

    elif key.find("attention.dense.weight") != -1 or key.find("mlp.dense_4h_to_h.weight") != -1:
        split_vals = np.split(val, factor, axis=0)
        for j in range(factor):
            saved_path = saved_dir / f"model.{key}.{i * factor + j}.bin"
            split_vals[j].tofile(saved_path.as_posix())

    elif key.find("mlp.dense_h_to_4h.weight") != -1 or key.find("mlp.dense_h_to_4h.bias") != -1:

        split_vals = np.split(val, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir / f"model.{key}.{i * factor + j}.bin"
            split_vals[j].tofile(saved_path.as_posix())

    elif key.find("attention.query_key_value.bias") != -1:
        local_dim = (int)(val.shape[-1] / 3)

        if ckpt_ver == 3:
            num_splits = 3
            head_num = args.head_num // args.trained_gpu_num
            size_per_head = local_dim // head_num

            val = val.reshape(head_num, num_splits, size_per_head)
            val = val.transpose(1, 0, 2)

        val = val.reshape(3, local_dim)
        split_vals = np.split(val, factor, axis=-1)

        for j in range(factor):
            saved_path = saved_dir / f"model.{key}.{i * factor + j}.bin"
            split_vals[j].tofile(saved_path.as_posix())

    elif key.find("attention.query_key_value.weight") != -1:
        hidden_dim = val.shape[0]
        local_dim = (int)(val.shape[-1] / 3)

        if ckpt_ver == 3:
            num_splits = 3
            head_num = args.head_num
            size_per_head = hidden_dim // head_num
            head_num = head_num // args.trained_gpu_num

            val = val.reshape(hidden_dim, head_num, num_splits, size_per_head)
            val = val.transpose(0, 2, 1, 3)

        if args.fused_qkv == 1:
            val = val.reshape(hidden_dim, 3, local_dim)
        elif args.fused_qkv == 0:
            val = val.reshape(hidden_dim, 3, local_dim).transpose(1, 0, 2)
        split_vals = np.split(val, factor, axis=-1)

        for j in range(factor):
            saved_path = saved_dir / f"model.{key}.{i * factor + j}.bin"
            split_vals[j].tofile(saved_path.as_posix())

    else:
        print(f"[ERROR] cannot find key '{key}'")


# less to more. e.g., trained by 2 gpus, infer by 8 gpus
def split_and_convert(args):
    saved_dir = Path(args.saved_dir)
    if args.fused_qkv == 1:
        saved_dir = saved_dir / f"{args.infer_gpu_num}-gpu"
    else:
        saved_dir = saved_dir / f"unfusedQKV-{args.infer_gpu_num}-gpu/"

    saved_dir.mkdir(parents=True, exist_ok=True)
     
    config = configparser.ConfigParser()
    config["gpt"] = {}
    for key in vars(args):
        config["gpt"][key] = f"{vars(args)[key]}"
    config["gpt"]["weight_data_type"] = args.weight_data_type
    with open((saved_dir / f"config.ini").as_posix(), 'w') as configfile:
        config.write(configfile)

    np_weight_data_type = get_weight_data_type(args.weight_data_type)
    prefix = Path(args.in_file)
    ckpt_name = "model_optim_rng.pt"
    t_gpu_num = args.trained_gpu_num
    i_gpu_num = args.infer_gpu_num
    assert i_gpu_num % t_gpu_num == 0

    factor = int(i_gpu_num / t_gpu_num)

    # load position_embedding from rank 0
    model_00 = torch.load((prefix / "mp_rank_00" / ckpt_name).as_posix())
    model_00["model"]["language_model"]["embedding"]["position_embeddings"]["weight"].cpu().numpy().astype(
        np_weight_data_type
    ).tofile(
        (saved_dir / "model.wpe.bin").as_posix()
    )  # not weight, do not need transpose

    del model_00
    w_e_list = []

    pool = multiprocessing.Pool(8)
    for i in range(t_gpu_num):
        m = torch.load(prefix / f"mp_rank_{i:02d}" / ckpt_name)
        if args.checkpoint_version == 3:
            transformer_model = m["model"]["language_model"]["encoder"]
        else:
            transformer_model = m["model"]["language_model"]["transformer"]

        w_e_list.append(
            m["model"]["language_model"]["embedding"]["word_embeddings"]["weight"].cpu().numpy().astype(np_weight_data_type)
        )

        pool.starmap(
            split_and_convert_process,
            [
                (
                    i,
                    saved_dir,
                    factor,
                    k,
                    args,
                    transformer_model[k].T.cpu().numpy().astype(np_weight_data_type),
                    args.checkpoint_version,
                )
                for (k, v) in transformer_model.items()
            ],
        )

    pool.close()
    pool.join()

    np.concatenate(w_e_list, axis=0).tofile((saved_dir / "model.wte.bin").as_posix())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-saved_dir", "-o", type=str, help="file name of output file", required=True)
    parser.add_argument("-in_file", "-i", type=str, help="file name of input checkpoint file", required=True)
    parser.add_argument("-trained_gpu_num", "-t_g", type=int, help="How many gpus for inference", required=True)
    parser.add_argument("-infer_gpu_num", "-i_g", type=int, help="How many gpus for inference", required=True)
    parser.add_argument(
        "-fused_qkv",
        "-fused_qkv",
        type=int,
        default=1,
        help="Fuse the qkv weights or not. Default is true (1)",
        choices=[0, 1],
    )
    parser.add_argument("-head_num", "-h_n", type=int, help="Number of heads", required=True)
    parser.add_argument("-checkpoint_version", type=int, default=0, help="Checkpoint version of Megatron-LM")
    parser.add_argument("-weight_data_type", type=str, default="fp32", choices=["fp32", "fp16"])

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    print("========================================")

    if args.trained_gpu_num > args.infer_gpu_num:
        merge_and_convert(args)
    else:
        split_and_convert(args)

