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
import configparser
from datetime import datetime
import multiprocessing
from pathlib import Path

import numpy as np
import torch  # pytype: disable=import-error
import sys

sys.path.append("/workdir/megatron-lm")

shared_mapping = {
    "wte":"shared.weight",
    "wte_T":"shared.weight_T",
    "ape":"shared.ape",
    "encoder_rpe":"block.0.layer.0.SelfAttention.relative_attention_bias",
    "decoder_rpe":"block.0.layer.0.SelfAttention.relative_attention_bias"
}

encoder_mapping = {
    "input_layernorm":"layer.0.layer_norm",
    "self_attention.query_key_value":["layer.0.SelfAttention.q", "layer.0.SelfAttention.k", "layer.0.SelfAttention.v"],
    "self_attention.dense":"layer.0.SelfAttention.o",
    "post_attention_layernorm":"layer.1.layer_norm",
    "mlp.dense_h_to_4h":"layer.1.DenseReluDense.wi",
    "mlp.dense_4h_to_h":"layer.1.DenseReluDense.wo",
    "final_layernorm":"final_layer_norm"
}

decoder_mapping = {
    "input_layernorm":"layer.0.layer_norm",
    "self_attention.query_key_value":["layer.0.SelfAttention.qkv"],
    "self_attention.dense":"layer.0.SelfAttention.o",
    "post_attention_layernorm":"layer.1.layer_norm",
    "inter_attention.query":["layer.1.EncDecAttention.q"],
    "inter_attention.key_value":["layer.1.EncDecAttention.k","layer.1.EncDecAttention.v"],
    "inter_attention.dense":"layer.1.EncDecAttention.o",
    "post_inter_attention_layernorm":"layer.2.layer_norm",
    "mlp.dense_h_to_4h":"layer.2.DenseReluDense.wi",
    "mlp.dense_4h_to_h":"layer.2.DenseReluDense.wo",
    "final_layernorm":"final_layer_norm"
}

megatron_HF_name_mapping = {
    "shared":shared_mapping,
    "encoder":encoder_mapping,
    "decoder":decoder_mapping
}

encoder_config_mapping = {
    "num_attention_heads":"num_heads",
    "kv_channels":"d_kv",
    "hidden_size":"d_model",
    "ffn_hidden_size":"d_ff",
    "num_layers":"num_layers",
    "padded_vocab_size":"vocab_size",
    "max_position_embeddings":"relative_attention_num_buckets_or_max_pos_seq_len",
    "relative_position_num_buckets":"relative_attention_num_buckets_or_max_pos_seq_len"
}

decoder_config_mapping = {
    "num_attention_heads":"num_heads",
    "kv_channels":"d_kv",
    "hidden_size":"d_model",
    "ffn_hidden_size":"d_ff",
    "num_layers":"num_layers",
    "padded_vocab_size":"vocab_size",
    "max_position_embeddings":"relative_attention_num_buckets_or_max_pos_seq_len",
    "relative_position_num_buckets":"relative_attention_num_buckets_or_max_pos_seq_len"
}

decoder_new_config = {
    "decoder_start_token_id":30522, ## need to adjust
    "eos_token_id":30523 ## need to adjust
}

model_new_config = {"structure":{"t5_with_bias":1, "position_embedding_type":0}}

def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16 
    else:
        assert False, f"Invalid weight data type {data_type}"

def convert_megatron_to_HF_naming_style_single(saved_key, name_mapping):
    saved_key = saved_key.replace("layers","block")
    mapping_key = saved_key.rsplit(sep=".", maxsplit=1)[0]
    mapping_key_no_num = mapping_key[mapping_key.find(".", 6) + 1 :]
    block_num = mapping_key[: mapping_key.find(".", 6) + 1]
    weight_or_bias = saved_key.rsplit(sep=".", maxsplit=1)[1]
    saved_key = block_num + name_mapping[mapping_key_no_num] + "." + weight_or_bias
    return saved_key

def convert_megatron_to_HF_naming_style_multiple(saved_key, name_mapping):
    saved_key = saved_key.replace("layers","block")
    mapping_key = saved_key.rsplit(sep=".", maxsplit=1)[0]
    mapping_key_no_num = mapping_key[mapping_key.find(".", 6) + 1 :]
    mapping_vals_no_num = name_mapping[mapping_key_no_num]
    block_num = mapping_key[: mapping_key.find(".", 6) + 1]
    weight_or_bias = saved_key.rsplit(sep=".", maxsplit=1)[1]
    saved_keys = [block_num + val + "." + weight_or_bias for val in mapping_vals_no_num]
    return saved_keys

def _gpu_map_location(storage, loc):
    if loc.startswith("cuda"):
        training_gpu_idx = int(loc.split(":")[1])
        inference_gpu_idx = training_gpu_idx % torch.cuda.device_count()
        return storage.cuda(inference_gpu_idx)
    elif loc.startswith("cpu"):
        return storage.cpu()
    else:
        raise NotImplementedError(f"Not handled {loc}")

# This tool is used to support the new megatron model trained by pipeline parallel + tensor parallel
def merge_and_convert_process(model_type, i, pipeline_para_rank, saved_dir, factor, key, model_args, transformer_model_list, ckpt_ver, np_weight_data_type):
    prefix = model_type
    name_mapping = megatron_HF_name_mapping[model_type]
    saved_dir = Path(saved_dir)
    if key.find("layers.") != -1:
        layer_index = (int)(key[7 : key.find(".", 7)])
        saved_key = key.replace(
            "layers.%d." % layer_index,
            "layers.%d." % (layer_index + pipeline_para_rank * model_args.num_layers // model_args.pipeline_model_parallel_size))
    else:
        saved_key = key
    major_device = transformer_model_list[0][key].device

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
        or key.find("final_layernorm.bias") != -1):

        # shared weights, only need to convert the weights of rank 0
        if i == 0:
            val = transformer_model_list[0][key].T.float().cpu().numpy()
            saved_key = convert_megatron_to_HF_naming_style_single(saved_key, name_mapping)
            saved_path = saved_dir / f"{prefix}.{saved_key}.bin"
            np.squeeze(val).astype(np_weight_data_type).tofile(saved_path)

    elif (key.find("self_attention.dense.weight") != -1
         or key.find("inter_attention.dense.weight") != -1
         or key.find("mlp.dense_4h_to_h.weight") != -1):
        vals = []
        for k in range(factor):
            vals.append(transformer_model_list[k][key].T.float().to(major_device))
        saved_key = convert_megatron_to_HF_naming_style_single(saved_key, name_mapping)
        torch.cat(vals, dim=0).cpu().numpy().astype(np_weight_data_type).tofile(saved_path)

    elif key.find("mlp.dense_h_to_4h.weight") != -1 or key.find("mlp.dense_h_to_4h.bias") != -1:
        vals = []
        for k in range(factor):
            vals.append(transformer_model_list[k][key].T.float().to(major_device))
        saved_key = convert_megatron_to_HF_naming_style_single(saved_key, name_mapping)
        saved_path = saved_dir / f"{prefix}.{saved_key}.{i:d}.bin"
        torch.cat(vals, dim=-1).cpu().numpy().astype(np_weight_data_type).tofile(saved_path)

    elif (key.find("self_attention.query_key_value.bias") != -1
          or key.find("inter_attention.query.bias") != -1
          or key.find("inter_attention.key_value.bias") != -1):
        num_splits = 3
        if key.find("inter_attention.key_value.bias") != -1:
            num_splits = 2
        if key.find("inter_attention.query.bias") != -1:
            num_splits = 1
        vals = []
        for k in range(factor):
            val = transformer_model_list[k][key].T.float()
            local_dim = int(val.shape[-1] / num_splits)
            if ckpt_ver == 3:
                head_num = model_args.num_attention_heads // model_args.tensor_model_parallel_size
                size_per_head = model_args.kv_channels
                val = val.reshape(head_num, num_splits, size_per_head)
                val = val.permute(1, 0, 2)
            val = val.reshape(num_splits, local_dim)
            vals.append(val.to(major_device))

        saved_vals = torch.cat(vals, dim=-1)
        saved_keys = convert_megatron_to_HF_naming_style_multiple(saved_key, name_mapping)
        if len(saved_keys) == 1:
            saved_path = saved_dir / f"{prefix}.{saved_keys[0]}.{i:d}.bin"
            saved_vals.cpu().numpy().astype(np_weight_data_type).tofile(saved_path)
            return
        for index in range(len(saved_keys)):
            saved_path = saved_dir / f"{prefix}.{saved_keys[index]}.{i:d}.bin"
            saved_vals[index,...].cpu().numpy().astype(np_weight_data_type).tofile(saved_path)

    elif (key.find("self_attention.query_key_value.weight") != -1
          or key.find("inter_attention.query.weight") != -1
          or key.find("inter_attention.key_value.weight") != -1):
        num_splits = 3
        if key.find("inter_attention.key_value.weight") != -1:
            num_splits = 2
        if key.find("inter_attention.query.weight") != -1:
            num_splits = 1
        vals = []
        for k in range(factor):
            val = transformer_model_list[k][key].T.float()
            hidden_dim = val.shape[0]
            local_dim = int(val.shape[-1] / num_splits)
            if ckpt_ver == 3:
                head_num = model_args.num_attention_heads
                size_per_head = model_args.kv_channels
                head_num = head_num // model_args.tensor_model_parallel_size
                val = val.reshape(hidden_dim, head_num, num_splits, size_per_head)
                val = val.permute(0, 2, 1, 3)
            val = val.reshape(hidden_dim, num_splits, local_dim)
            vals.append(val.to(major_device))

        saved_vals = torch.cat(vals, dim=-1)
        saved_keys = convert_megatron_to_HF_naming_style_multiple(saved_key, name_mapping)
        if len(saved_keys) == 1:
            saved_path = saved_dir / f"{prefix}.{saved_keys[0]}.{i:d}.bin"
            saved_vals.cpu().numpy().astype(np_weight_data_type).tofile(saved_path)
            return
        for index in range(len(saved_keys)):
            saved_path = saved_dir / f"{prefix}.{saved_keys[index]}.{i:d}.bin"
            saved_vals[:, index, ...].cpu().numpy().astype(np_weight_data_type).tofile(saved_path)
    else:
        print(f"[ERROR] cannot find key '{key}'")

def split_and_convert_process(model_type, i, pipeline_para_rank, saved_dir, factor, key, model_args, transformer_model_list, ckpt_ver, np_weight_data_type):
    prefix = model_type
    name_mapping = megatron_HF_name_mapping[model_type]
    val = transformer_model_list[0][key].T.float().cpu().numpy().astype(np_weight_data_type)
    del transformer_model_list[0][key]

    if key.find("layers.") != -1:
        layer_index = (int)(key[7 : key.find(".", 7)])
        saved_key = key.replace(
            "layers.%d." % layer_index,
            "layers.%d." % (layer_index + pipeline_para_rank * model_args.num_layers // model_args.pipeline_model_parallel_size))
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
        or key.find("final_layernorm.bias") != -1):
        # shared weights, only need to convert the weights of rank 0
        if i == 0:
            saved_key = convert_megatron_to_HF_naming_style_single(saved_key, name_mapping)
            saved_path = saved_dir / f"{prefix}.{saved_key}.bin"
            val.tofile(saved_path.as_posix())

    elif (key.find("self_attention.dense.weight") != -1
         or key.find("inter_attention.dense.weight") != -1
         or key.find("mlp.dense_4h_to_h.weight") != -1):
        split_vals = np.split(val, factor, axis=0)
        saved_key = convert_megatron_to_HF_naming_style_single(saved_key, name_mapping)
        for j in range(factor):
            saved_path = saved_dir / f"{prefix}.{saved_key}.{i * factor + j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())

    elif key.find("mlp.dense_h_to_4h.weight") != -1 or key.find("mlp.dense_h_to_4h.bias") != -1:
        split_vals = np.split(val, factor, axis=-1)
        saved_key = convert_megatron_to_HF_naming_style_single(saved_key, name_mapping)
        for j in range(factor):
            saved_path = saved_dir / f"{prefix}.{saved_key}.{i * factor + j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())

    elif (key.find("self_attention.query_key_value.bias") != -1
          or key.find("inter_attention.query.bias") != -1
          or key.find("inter_attention.key_value.bias") != -1):
        num_splits = 3
        if key.find("inter_attention.key_value.bias") != -1:
            num_splits = 2
        if key.find("inter_attention.query.bias") != -1:
            num_splits = 1
        local_dim = int(val.shape[-1] / num_splits)
        if ckpt_ver == 3:
            head_num = model_args.num_attention_heads // model_args.tensor_model_parallel_size
            size_per_head = model_args.kv_channels
            val = val.reshape(head_num, num_splits, size_per_head)
            val = val.transpose(1, 0, 2)

        val = val.reshape(num_splits, local_dim)
        split_vals = np.split(val, factor, axis=-1)
        saved_keys = convert_megatron_to_HF_naming_style_multiple(saved_key, name_mapping)
        for j in range(factor):
            if len(saved_keys) == 1:
                saved_path = saved_dir / f"{prefix}.{saved_keys[0]}.{i * factor + j:d}.bin"
                split_vals[j].tofile(saved_path.as_posix())
                continue
            for index in range(len(saved_keys)):
                saved_path = saved_dir / f"{prefix}.{saved_keys[index]}.{i * factor + j:d}.bin"
                split_vals[j][index, ...].tofile(saved_path.as_posix())

    elif (key.find("self_attention.query_key_value.weight") != -1
          or key.find("inter_attention.query.weight") != -1
          or key.find("inter_attention.key_value.weight") != -1):
        num_splits = 3
        if key.find("inter_attention.key_value.weight") != -1:
            num_splits = 2
        if key.find("inter_attention.query.weight") != -1:
            num_splits = 1
        hidden_dim = val.shape[0]
        local_dim = int(val.shape[-1] / num_splits)

        if ckpt_ver == 3:
            head_num = model_args.num_attention_heads
            size_per_head = model_args.kv_channels
            head_num = head_num // model_args.tensor_model_parallel_size
            val = val.reshape(hidden_dim, head_num, num_splits, size_per_head)
            val = val.transpose(0, 2, 1, 3)

        val = val.reshape(hidden_dim, num_splits, local_dim)
        split_vals = np.split(val, factor, axis=-1)

        saved_keys = convert_megatron_to_HF_naming_style_multiple(saved_key, name_mapping)
        for j in range(factor):
            if len(saved_keys) == 1:
                saved_path = saved_dir / f"{prefix}.{saved_keys[0]}.{i * factor + j:d}.bin"
                split_vals[j].tofile(saved_path.as_posix())
                continue
            for index in range(len(saved_keys)):
                saved_path = saved_dir / f"{prefix}.{saved_keys[index]}.{i * factor + j:d}.bin"
                split_vals[j][:, index, ...].tofile(saved_path.as_posix())
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

    # update model structure config
    if not hasattr(model_args, 'position_embedding_type') or model_args.position_embedding_type == "absolute":
        model_new_config["structure"]["position_embedding_type"] = 1
    config = configparser.ConfigParser()

    config["encoder"] = {}
    config["decoder"] = {}
    for key, val in vars(model_args).items():
        if key in encoder_config_mapping.keys():
            config["encoder"][encoder_config_mapping[key]] = f"{val}"
        if key in decoder_config_mapping.keys():
            config["decoder"][decoder_config_mapping[key]] = f"{val}"
    for key, val in decoder_new_config.items():
        config["decoder"][key] = f"{val}"
    for key, val in model_new_config.items():
        config[key] = {}
        for val_key, val_val in val.items():
            config[key][val_key] = f"{val_val}"
    
    ## add weight data type
    np_weight_data_type = get_weight_data_type(args.weight_data_type)
    config['encoder']['weight_data_type'] = args.weight_data_type
    config['decoder']['weight_data_type'] = args.weight_data_type

    # add model name
    config["encoder"]["_name_or_path"] = args.model_name
    config["decoder"]["_name_or_path"] = args.model_name

    # add weight data type
    config["encoder"]["weight_data_type"] = args.weight_data_type
    config["decoder"]["weight_data_type"] = args.weight_data_type

    np_weight_data_type = get_weight_data_type(args.weight_data_type)

    with open((saved_dir / f"config.ini").as_posix(), 'w') as configfile:
        config.write(configfile)

    if "position_embeddings" in model_00["model"]["language_model"]["embedding"].keys():
        model_00["model"]["language_model"]["embedding"]["position_embeddings"]["weight"] \
                                    .float().cpu().numpy().astype(np_weight_data_type) \
                                    .tofile((saved_dir / "shared.ape.bin").as_posix())

    # inference factor calculation
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

    # split rpe into tensor parallel ranks
    if "encoder_relative_position_embedding" in model_00["model"]["language_model"].keys():
        encoder_relative_position_embedding = model_00["model"]["language_model"]["encoder_relative_position_embedding"]["weight"] \
                                    .T.float().cpu().numpy().astype(np_weight_data_type)
        encoder_relative_position_embedding_split = np.split(encoder_relative_position_embedding, i_gpu_num, axis=0)
        for i in range(i_gpu_num):
            saved_path = saved_dir / f"encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight.{i}.bin"
            encoder_relative_position_embedding_split[i].tofile(saved_path.as_posix())

        del encoder_relative_position_embedding, encoder_relative_position_embedding_split

    if "decoder_relative_position_embedding" in model_00["model"]["language_model"].keys():
        decoder_relative_position_embedding = model_00["model"]["language_model"]["decoder_relative_position_embedding"]["weight"] \
                                    .T.float().cpu().numpy().astype(np_weight_data_type)
        decoder_relative_position_embedding_split = np.split(decoder_relative_position_embedding, i_gpu_num, axis=0)
        for i in range(i_gpu_num):
            saved_path = saved_dir / f"decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight.{i}.bin"
            decoder_relative_position_embedding_split[i].tofile(saved_path.as_posix())

        del decoder_relative_position_embedding, decoder_relative_position_embedding_split

    del model_00
    w_e_list = []

    torch.multiprocessing.set_start_method("spawn")
    pool = multiprocessing.Pool(args.processes)
    for i in range(main_loop):
        for j in range(model_args.pipeline_model_parallel_size):
            if model_args.pipeline_model_parallel_size == 1:
                layer_rank_num = ""
            else:
                layer_rank_num = f"_{j:03d}"

            encoder_models = []
            decoder_models = []

            if is_merge_ckpt == True:
                for k in range(factor):
                    m = torch.load((prefix / f"mp_rank_{i * factor + k:02d}{layer_rank_num}" / ckpt_name).as_posix(), map_location=_gpu_map_location)
                    encoder_models.append(m["model"]["language_model"]["encoder"])
                    decoder_models.append(m["model"]["language_model"]["decoder"])

                    if j == 0:
                        w_e_list.append(m["model"]["language_model"]["embedding"]["word_embeddings"]["weight"].float().cpu().numpy().astype(np_weight_data_type))
            else:
                m = torch.load((prefix / f"mp_rank_{i:02d}{layer_rank_num}" / ckpt_name).as_posix(), map_location=_gpu_map_location)

                if j == 0:
                    w_e_list.append(
                        m["model"]["language_model"]["embedding"]["word_embeddings"]["weight"]
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np_weight_data_type)
                    )
                encoder_models.append(m["model"]["language_model"]["encoder"])
                decoder_models.append(m["model"]["language_model"]["decoder"])

            pool.starmap(
                merge_and_convert_process if is_merge_ckpt == True else split_and_convert_process,
                [
                    (
                        "encoder",
                        i,
                        j,
                        saved_dir,
                        factor,
                        k,
                        model_args,
                        encoder_models,
                        m["checkpoint_version"],
                        np_weight_data_type
                    )
                    for k in encoder_models[0].keys()
                ],
            )
            pool.starmap(
                merge_and_convert_process if is_merge_ckpt == True else split_and_convert_process,
                [
                    (
                        "decoder",
                        i,
                        j,
                        saved_dir,
                        factor,
                        k,
                        model_args,
                        decoder_models,
                        m["checkpoint_version"],
                        np_weight_data_type
                    )
                    for k in decoder_models[0].keys()
                ],
            )

    pool.close()
    pool.join()

    np.concatenate(w_e_list, axis=0).tofile((saved_dir / "shared.weight_T.bin").as_posix())
    m["model"]["lm_head"]["bias"].float().cpu().numpy().tofile((saved_dir / "shared.bias.bin").as_posix())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-saved_dir", "-o", type=str, help="file name of output file", required=True)
    parser.add_argument("-in_file", "-i", type=str, help="file name of input checkpoint file", required=True)
    parser.add_argument("-infer_gpu_num", "-i_g", type=int, help="How many gpus for inference", required=True)
    parser.add_argument("-processes", "-p", type=int, help="How many processes to spawn for conversion (default: 64)", default=64)
    parser.add_argument("-weight_data_type", type=str, default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("-model_name", "-m", type=str, help="model name", required=True)
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
