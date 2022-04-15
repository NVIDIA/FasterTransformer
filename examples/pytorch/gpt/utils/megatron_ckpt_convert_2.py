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
import multiprocessing
from pathlib import Path

import numpy as np
import torch  # pytype: disable=import-error

def _gpu_map_location(storage, loc):
    if loc.startswith("cuda"):
        training_gpu_idx = int(loc.split(":")[1])
        inference_gpu_idx = training_gpu_idx % torch.cuda.device_count()
        return storage.cuda(inference_gpu_idx)
    elif loc.startswith("cpu"):
        return storage.cpu()
    else:
        raise NotImplementedError(f"Not handled {loc}")

def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    else:
        assert False, f"Invalid weight data type {data_type}"

# This tool is used to support the new megatron model trained by pipeline parallel + tensor parallel
def merge_and_convert_process(i, pipeline_para_rank, saved_dir, factor, key, model_args, transformer_model_list, ckpt_ver, np_weight_data_type):
    saved_dir = Path(saved_dir)
    if key.find("layers.") != -1:
        layer_index = (int)(key[7 : key.find(".", 7)])
        saved_key = key.replace(
            "layers.%d." % layer_index,
            "layers.%d." % (layer_index + pipeline_para_rank * model_args.num_layers // model_args.pipeline_model_parallel_size))

        if saved_key.find("self_attention") != -1:
            saved_key = saved_key.replace("self_attention", "attention")
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
        or key.find("final_layernorm.weight") != -1 
        or key.find("final_layernorm.bias") != -1):

        # shared weights, only need to convert the weights of rank 0
        if i == 0:
            val = transformer_model_list[0][key].T.float().cpu().numpy()
            saved_path = saved_dir / f"model.{saved_key}.bin"
            np.squeeze(val).astype(np_weight_data_type).tofile(saved_path)

    elif key.find("attention.dense.weight") != -1 or key.find("mlp.dense_4h_to_h.weight") != -1:
        vals = []
        for k in range(factor):
            vals.append(transformer_model_list[k][key].T.float().to(major_device))
        saved_path = saved_dir / f"model.{saved_key}.{i:d}.bin"
        torch.cat(vals, dim=0).cpu().numpy().astype(np_weight_data_type).tofile(saved_path)

    elif key.find("mlp.dense_h_to_4h.weight") != -1 or key.find("mlp.dense_h_to_4h.bias") != -1:
        vals = []
        for k in range(factor):
            vals.append(transformer_model_list[k][key].T.float().to(major_device))
        saved_path = saved_dir / f"model.{saved_key}.{i:d}.bin"
        torch.cat(vals, dim=-1).cpu().numpy().astype(np_weight_data_type).tofile(saved_path)

    elif key.find("attention.query_key_value.bias") != -1:
        vals = []
        for k in range(factor):
            val = transformer_model_list[k][key].T.float()
            local_dim = int(val.shape[-1] / 3)
            if ckpt_ver == 3:
                num_splits = 3
                head_num = model_args.num_attention_heads // model_args.tensor_model_parallel_size
                size_per_head = local_dim // head_num
                val = val.reshape(head_num, num_splits, size_per_head)
                val = val.permute(1, 0, 2)
            val = val.reshape(3, local_dim)
            vals.append(val.to(major_device))

        saved_path = saved_dir / f"model.{saved_key}.{i:d}.bin"
        torch.cat(vals, dim=-1).cpu().numpy().astype(np_weight_data_type).tofile(saved_path)

    elif key.find("attention.query_key_value.weight") != -1:
        vals = []
        for k in range(factor):
            val = transformer_model_list[k][key].T.float()
            hidden_dim = val.shape[0]
            local_dim = int(val.shape[-1] / 3)
            if ckpt_ver == 3:
                num_splits = 3
                head_num = model_args.num_attention_heads
                size_per_head = hidden_dim // head_num
                head_num = head_num // model_args.tensor_model_parallel_size
                val = val.reshape(hidden_dim, head_num, num_splits, size_per_head)
                val = val.permute(0, 2, 1, 3)
            val = val.reshape(hidden_dim, 3, local_dim)
            vals.append(val.to(major_device))

        saved_path = saved_dir / f"model.{saved_key}.{i:d}.bin"
        torch.cat(vals, dim=-1).cpu().numpy().astype(np_weight_data_type).tofile(saved_path)
        
    else:
        print(f"[ERROR] cannot find key '{key}'")
        
def split_and_convert_process(i, pipeline_para_rank, saved_dir, factor, key, model_args, transformer_model_list, ckpt_ver, np_weight_data_type):
    val = transformer_model_list[0][key].T.float().cpu().numpy().astype(np_weight_data_type)
    if key.find("layers.") != -1:
        layer_index = (int)(key[7 : key.find(".", 7)])
        saved_key = key.replace(
            "layers.%d." % layer_index,
            "layers.%d." % (layer_index + pipeline_para_rank * model_args.num_layers // model_args.pipeline_model_parallel_size))

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

        if ckpt_ver == 3:
            num_splits = 3
            head_num = model_args.num_attention_heads // model_args.tensor_model_parallel_size
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
            head_num = model_args.num_attention_heads
            size_per_head = hidden_dim // head_num
            head_num = head_num // model_args.tensor_model_parallel_size
            val = val.reshape(hidden_dim, head_num, num_splits, size_per_head)
            val = val.transpose(0, 2, 1, 3)

        val = val.reshape(hidden_dim, 3, local_dim)
        split_vals = np.split(val, factor, axis=-1)

        for j in range(factor):
            saved_path = saved_dir / f"model.{saved_key}.{i * factor + j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())

    else:
        print(f"[ERROR] cannot find key '{key}'")

def convert_checkpoint(args):
    saved_dir = Path(args.saved_dir) / f"{args.infer_gpu_num:d}-gpu"
    saved_dir.mkdir(parents=True, exist_ok=True)

    prefix = Path(args.in_file)
    ckpt_name = "model_optim_rng.pt"

    # load position_embedding from rank 0
    if (prefix / "mp_rank_00").is_dir():
        model_00 = torch.load((prefix / "mp_rank_00" / ckpt_name).as_posix(), map_location=_gpu_map_location)
    elif (prefix / "mp_rank_00_000").is_dir():
        model_00 = torch.load((prefix / "mp_rank_00_000" / ckpt_name).as_posix(), map_location=_gpu_map_location)
    else:
        print(f"[ERROR] Cannot find checkpoint in {prefix}.")
        exit(1)

    model_args = model_00["args"]
    with open((saved_dir / "args.txt").as_posix(), "w") as f:
        for k, v in vars(model_args).items():
            f.write("{}:{} \n".format(k, v))
    
    config = configparser.ConfigParser()
    config["gpt"] = {}
    for key in vars(args):
        config["gpt"][key] = f"{vars(args)[key]}"
    for k, v in vars(model_args).items():
        config["gpt"][k] = f"{v}"
    config["gpt"]["weight_data_type"] = args.weight_data_type
    with open((saved_dir / f"config.ini").as_posix(), 'w') as configfile:
        config.write(configfile)
    np_weight_data_type = get_weight_data_type(args.weight_data_type)

    model_00["model"]["language_model"]["embedding"]["position_embeddings"]["weight"].float().cpu().numpy().astype(
        np_weight_data_type
    ).tofile(
        (saved_dir / "model.wpe.bin").as_posix()
    )  # not weight, do not need transpose
    del model_00
    w_e_list = []
    
    t_gpu_num = model_args.tensor_model_parallel_size
    i_gpu_num = args.infer_gpu_num

    if t_gpu_num > i_gpu_num:
        assert t_gpu_num % i_gpu_num == 0
        is_merge_ckpt = True
        factor = int(t_gpu_num / i_gpu_num)
    else:
        assert i_gpu_num % t_gpu_num == 0
        is_merge_ckpt = False
        factor = int(i_gpu_num / t_gpu_num)

    main_loop = min(t_gpu_num, i_gpu_num)
    
    torch.multiprocessing.set_start_method("spawn")
    pool = multiprocessing.Pool(args.processes)
    for i in range(main_loop):
        for j in range(model_args.pipeline_model_parallel_size):
            if model_args.pipeline_model_parallel_size == 1:
                layer_rank_num = ""
            else:
                layer_rank_num = f"_{j:03d}"
            
            transformer_models = []
            if is_merge_ckpt == True:
                for k in range(factor):
                    m = torch.load((prefix / f"mp_rank_{i * factor + k:02d}{layer_rank_num}" / ckpt_name).as_posix(), map_location=_gpu_map_location)
                    transformer_models.append(m["model"]["language_model"]["encoder"])

                    if j == 0:
                        w_e_list.append(m["model"]["language_model"]["embedding"]["word_embeddings"]["weight"].float().cpu().numpy().astype(np_weight_data_type))
            else:
                m = torch.load(prefix / f"mp_rank_{i:02d}{layer_rank_num}/" / ckpt_name, map_location=_gpu_map_location)
            
                if j == 0:
                    w_e_list.append(
                        m["model"]["language_model"]["embedding"]["word_embeddings"]["weight"]
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np_weight_data_type)
                    )
                transformer_models.append(m["model"]["language_model"]["encoder"])

            pool.starmap(
                merge_and_convert_process if is_merge_ckpt == True else split_and_convert_process,
                [
                    (
                        i,
                        j,
                        saved_dir,
                        factor,
                        k,
                        model_args,
                        transformer_models,
                        m["checkpoint_version"],
                        np_weight_data_type,
                    )
                    for (k, v) in transformer_models[0].items()
                ],
            )

    pool.close()
    pool.join()

    np.concatenate(w_e_list, axis=0).tofile((saved_dir / "model.wte.bin").as_posix())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-saved_dir", "-o", type=str, help="file name of output file", required=True)
    parser.add_argument("-in_file", "-i", type=str, help="file name of input checkpoint file", required=True)
    parser.add_argument("-infer_gpu_num", "-i_g", type=int, help="How many gpus for inference", required=True)
    parser.add_argument("-processes", "-p", type=int, help="How many processes to spawn for conversion (default: 64)", default=64)
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
