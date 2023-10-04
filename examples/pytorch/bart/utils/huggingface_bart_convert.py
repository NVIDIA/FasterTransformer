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
import multiprocessing
from datetime import datetime
import logging
from pathlib import Path

from transformers import BartForConditionalGeneration, MBartForConditionalGeneration, AutoConfig

import numpy as np
import torch  # pytype: disable=import-error

LOGGER = logging.getLogger(__name__)


def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    else:
        assert False, f"Invalid weight data type {data_type}"


def fuse_decoder_qkv(model, factor, saved_dir, np_weight_data_type):
    model_dict = {}
    for name, param in model.named_parameters():
        if name.find("self_attn") == -1 or name.find("decoder.layers") == -1:
            continue
        if name.find(".q_proj.") != -1 or name.find(".k_proj.") != -1 or name.find(".v_proj.") != -1:
            model_dict[name] = param

    for i in range(model.config.decoder_layers):
        shape = model_dict[f"model.decoder.layers.{i}.self_attn.q_proj.weight"].T.shape
        qkv = torch.cat([model_dict[f"model.decoder.layers.{i}.self_attn.q_proj.weight"].T,
                         model_dict[f"model.decoder.layers.{i}.self_attn.k_proj.weight"].T,
                         model_dict[f"model.decoder.layers.{i}.self_attn.v_proj.weight"].T], dim=-1)

        qkv = qkv.reshape([shape[0], 3, shape[1]])
        qkv = qkv.cpu().detach().numpy().astype(np_weight_data_type)

        split_vals = np.split(qkv, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir / f"decoder.{i}.layer.SelfAttention.qkv.weight.{j}.bin"
            split_vals[j].tofile(saved_path.as_posix())

    for i in range(model.config.decoder_layers):
        shape = model_dict[f"model.decoder.layers.{i}.self_attn.q_proj.bias"].shape
        qkv = torch.cat([model_dict[f"model.decoder.layers.{i}.self_attn.q_proj.bias"],
                         model_dict[f"model.decoder.layers.{i}.self_attn.k_proj.bias"],
                         model_dict[f"model.decoder.layers.{i}.self_attn.v_proj.bias"]], dim=-1)
        qkv = qkv.cpu().detach().numpy().astype(np_weight_data_type)

        split_vals = np.split(qkv, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir / f"decoder.{i}.layer.SelfAttention.qkv.bias.{j}.bin"
            split_vals[j].tofile(saved_path.as_posix())


def get_encoder_or_decoder(key):
    return "encoder" if key.find("encoder") != -1 else "decoder"


def get_fc(key):
    return "fc1" if key.find("fc1.") != -1 else "fc2"


def split_and_convert_process(key, val, factor, saved_dir, scale):
    if val.ndim == 2:
        val = val.transpose(1, 0)

    if key.find(".embed_positions.weight") != -1:
        prefix = get_encoder_or_decoder(key)
        saved_path = saved_dir / f"{prefix}.embed_positions.weight.bin"
        val[:, 2:].T.tofile(saved_path.as_posix())
    elif key.find(".embed_tokens.weight") != -1:
        prefix = get_encoder_or_decoder(key)
        saved_path = saved_dir / f"{prefix}.embed_tokens.weight.bin"
        val = val * scale
        val.T.tofile(saved_path.as_posix())
    elif key.find(".layernorm_embedding.weight") != -1:
        prefix = get_encoder_or_decoder(key)
        saved_path = saved_dir / f"{prefix}.final_layer_norm.weight.bin"
        val.tofile(saved_path.as_posix())
    elif key.find(".layernorm_embedding.bias") != -1:
        prefix = get_encoder_or_decoder(key)
        saved_path = saved_dir / f"{prefix}.final_layer_norm.bias.bin"
        val.tofile(saved_path.as_posix())
    elif key.find(".layer_norm.weight") != -1:
        prefix = get_encoder_or_decoder(key)
        saved_path = saved_dir / f"{prefix}.layer_norm.weight.bin"
        val.tofile(saved_path.as_posix())
    elif key.find(".layer_norm.bias") != -1:
        prefix = get_encoder_or_decoder(key)
        saved_path = saved_dir / f"{prefix}.layer_norm.bias.bin"
        val.tofile(saved_path.as_posix())
    elif (
        key.find("self_attn.k_proj.weight") != -1
        or key.find("self_attn.v_proj.weight") != -1
        or key.find("self_attn.q_proj.weight") != -1
    ):
        split_vals = np.split(val, factor, axis=0)
        prefix = get_encoder_or_decoder(key)
        if prefix == "decoder":
            # will be handled in fuse_decoder_qkv instead
            return
        layer = int(key.split('layers.')[1].split('.self_attn')[0])
        qkv = key.split('self_attn.')[1][:1]
        for j in range(factor):
            saved_path = saved_dir / f"{prefix}.{layer}.layer.SelfAttention.{qkv}.weight.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())
    elif (
        key.find("self_attn.k_proj.bias") != -1
        or key.find("self_attn.v_proj.bias") != -1
        or key.find("self_attn.q_proj.bias") != -1
    ):
        split_vals = np.split(val, factor, axis=0)
        prefix = get_encoder_or_decoder(key)
        if prefix == "decoder":
            # will be handled in fuse_decoder_qkv instead
            return
        layer = int(key.split('layers.')[1].split('.self_attn')[0])
        qkv = key.split('self_attn.')[1][:1]
        for j in range(factor):
            saved_path = saved_dir / f"{prefix}.{layer}.layer.SelfAttention.{qkv}.bias.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())
    elif key.find("self_attn.out_proj.weight") != -1:
        split_vals = np.split(val, factor, axis=0)
        prefix = get_encoder_or_decoder(key)
        layer = int(key.split('layers.')[1].split('.self_attn')[0])
        for j in range(factor):
            saved_path = saved_dir / f"{prefix}.{layer}.layer.SelfAttention.out_proj.weight.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())
    elif key.find("self_attn.out_proj.bias") != -1:
        split_vals = np.split(val, factor, axis=0)
        prefix = get_encoder_or_decoder(key)
        layer = int(key.split('layers.')[1].split('.self_attn')[0])
        for j in range(factor):
            saved_path = saved_dir / f"{prefix}.{layer}.layer.SelfAttention.out_proj.bias.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())
    elif key.find("self_attn_layer_norm.weight") != -1:
        prefix = get_encoder_or_decoder(key)
        layer = int(key.split('layers.')[1].split('.self_attn')[0])
        saved_path = saved_dir / f"{prefix}.{layer}.layer.SelfAttention.attn_layer_norm.weight.bin"
        val.tofile(saved_path.as_posix())
    elif key.find("self_attn_layer_norm.bias") != -1:
        prefix = get_encoder_or_decoder(key)
        layer = int(key.split('layers.')[1].split('.self_attn')[0])
        saved_path = saved_dir / f"{prefix}.{layer}.layer.SelfAttention.attn_layer_norm.bias.bin"
        val.tofile(saved_path.as_posix())
    elif (
        key.find("encoder_attn.k_proj.weight") != -1
        or key.find("encoder_attn.v_proj.weight") != -1
        or key.find("encoder_attn.q_proj.weight") != -1
    ):
        split_vals = np.split(val, factor, axis=0)
        layer = int(key.split('layers.')[1].split('.encoder_attn')[0])
        qkv = key.split('encoder_attn.')[1][:1]
        for j in range(factor):
            saved_path = saved_dir / f"decoder.{layer}.layer.CrossAttention.{qkv}.weight.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())
    elif (
        key.find("encoder_attn.k_proj.bias") != -1
        or key.find("encoder_attn.v_proj.bias") != -1
        or key.find("encoder_attn.q_proj.bias") != -1
    ):
        split_vals = np.split(val, factor, axis=0)
        layer = int(key.split('layers.')[1].split('.encoder_attn')[0])
        qkv = key.split('encoder_attn.')[1][:1]
        for j in range(factor):
            saved_path = saved_dir / f"decoder.{layer}.layer.CrossAttention.{qkv}.bias.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())
    elif key.find("encoder_attn.out_proj.weight") != -1:
        split_vals = np.split(val, factor, axis=0)
        layer = int(key.split('layers.')[1].split('.encoder_attn')[0])
        for j in range(factor):
            saved_path = saved_dir / f"decoder.{layer}.layer.CrossAttention.out_proj.weight.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())
    elif key.find("encoder_attn.out_proj.bias") != -1:
        split_vals = np.split(val, factor, axis=0)
        layer = int(key.split('layers.')[1].split('.encoder_attn')[0])
        for j in range(factor):
            saved_path = saved_dir / f"decoder.{layer}.layer.CrossAttention.out_proj.bias.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())
    elif key.find("encoder_attn_layer_norm.weight") != -1:
        layer = int(key.split('layers.')[1].split('.encoder_attn')[0])
        saved_path = saved_dir / f"decoder.{layer}.layer.CrossAttention.attn_layer_norm.weight.bin"
        val.tofile(saved_path.as_posix())
    elif key.find("encoder_attn_layer_norm.bias") != -1:
        layer = int(key.split('layers.')[1].split('.encoder_attn')[0])
        saved_path = saved_dir / f"decoder.{layer}.layer.CrossAttention.attn_layer_norm.bias.bin"
        val.tofile(saved_path.as_posix())
    elif key.find("fc1.weight") != -1 or key.find("fc2.weight") != -1:
        prefix = get_encoder_or_decoder(key)
        split_vals = np.split(val, factor, axis=0)
        fc = get_fc(key)
        layer = int(key.split('layers.')[1].split(f'.{fc}.')[0])
        for j in range(factor):
            saved_path = saved_dir / f"{prefix}.{layer}.layer.SelfAttention.{fc}.weight.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())
    elif key.find("fc1.bias") != -1 or key.find("fc2.bias") != -1:
        prefix = get_encoder_or_decoder(key)
        fc = get_fc(key)
        layer = int(key.split('layers.')[1].split(f'.{fc}.')[0])
        split_vals = np.split(val, factor, axis=0)
        for j in range(factor):
            saved_path = saved_dir / f"{prefix}.{layer}.layer.SelfAttention.{fc}.bias.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())
    elif key.find("final_layer_norm.weight") != -1:
        prefix = get_encoder_or_decoder(key)
        layer = int(key.split('layers.')[1].split('.final_layer_norm.')[0])
        saved_path = saved_dir / f"{prefix}.{layer}.layer.SelfAttention.final_layer_norm.weight.bin"
        val.tofile(saved_path.as_posix())
    elif key.find("final_layer_norm.bias") != -1:
        prefix = get_encoder_or_decoder(key)
        layer = int(key.split('layers.')[1].split('.final_layer_norm.')[0])
        saved_path = saved_dir / f"{prefix}.{layer}.layer.SelfAttention.final_layer_norm.bias.bin"
        val.tofile(saved_path.as_posix())
    elif key.find("lm_head.weight") != -1:
        saved_path = saved_dir / "decoder.lm_head.weight.bin"
        val.T.tofile(saved_path.as_posix())
    elif key.find("final_logits_bias") != -1:
        saved_path = saved_dir / "decoder.final_logits_bias.bin"
        val.tofile(saved_path.as_posix())
    elif key.find("encoder.embed_tokens.weight") != -1 or \
            key.find("decoder.embed_tokens.weight") != -1:
        LOGGER.warning(f"Not save {key}, using shared.weight directly.")
    else:
        LOGGER.warning(f"Not save '{key}' with shape {val.shape}")


def convert_checkpoint(args):
    saved_dir = Path(args.saved_dir) / f"{args.inference_tensor_para_size:d}-gpu"
    saved_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(args.in_file)
    mbart = "false"
    scale = 1.0
    if config.model_type == 'mbart':
        mbart = "true"
        model = MBartForConditionalGeneration.from_pretrained(args.in_file)
    else:
        model = BartForConditionalGeneration.from_pretrained(args.in_file)
    if config.scale_embedding:
        scale = np.sqrt(config.d_model)
    hf_config = vars(model.config)
    config = configparser.ConfigParser()

    config["encoder"] = {}
    config["encoder"]["model_name"] = "bart"
    config["encoder"]["num_heads"] = str(hf_config["encoder_attention_heads"])
    config["encoder"]["d_kv"] = str(hf_config["d_model"] // hf_config["encoder_attention_heads"])
    config["encoder"]["d_model"] = str(hf_config["d_model"])
    config["encoder"]["d_ff"] = str(hf_config["encoder_ffn_dim"])
    config["encoder"]["num_layers"] = str(hf_config["encoder_layers"])
    config["encoder"]["vocab_size"] = str(hf_config["vocab_size"])
    config["encoder"]["max_pos_seq_len"] = str(hf_config["max_position_embeddings"])
    config["encoder"]["feed_forward_proj"] = str(hf_config["activation_function"])
    config["encoder"]["weight_data_type"] = args.weight_data_type
    config["encoder"]["mbart"] = mbart

    config["decoder"] = {}
    config["decoder"]["num_heads"] = str(hf_config["decoder_attention_heads"])
    config["decoder"]["d_kv"] = str(hf_config["d_model"] // hf_config["decoder_attention_heads"])
    config["decoder"]["d_model"] = str(hf_config["d_model"])
    config["decoder"]["d_ff"] = str(hf_config["decoder_ffn_dim"])
    config["decoder"]["num_layers"] = str(hf_config["decoder_layers"])
    config["decoder"]["vocab_size"] = str(hf_config["vocab_size"])
    config["decoder"]["max_pos_seq_len"] = str(hf_config["max_position_embeddings"])
    config["decoder"]["decoder_start_token_id"] = str(hf_config["decoder_start_token_id"])
    config["decoder"]["eos_token_id"] = str(hf_config["eos_token_id"])
    config["decoder"]["weight_data_type"] = args.weight_data_type

    with open((saved_dir / "config.ini").as_posix(), 'w') as configfile:
        config.write(configfile)
    np_weight_data_type = get_weight_data_type(args.weight_data_type)

    i_gpu_num = args.inference_tensor_para_size

    for name, param in model.state_dict().items():
        split_and_convert_process(name, param.cpu().detach().numpy().astype(np_weight_data_type), i_gpu_num, saved_dir, scale)
    # pool = multiprocessing.Pool(args.processes)
    # pool.starmap_async(split_and_convert_process,
    #                    [(name, param.cpu().detach().numpy().astype(np_weight_data_type), i_gpu_num, saved_dir)
    #                     for name, param in model.state_dict().items()])

    # pool.close()
    # pool.join()

    fuse_decoder_qkv(model, i_gpu_num, saved_dir, np_weight_data_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-saved_dir", "-o", type=str, help="file name of output file", required=True)
    parser.add_argument("-in_file", "-i", type=str, help="file name of input checkpoint file", required=True)
    parser.add_argument("-inference_tensor_para_size", "-i_g", type=int, help="How many gpus for inference",
                        required=True)
    parser.add_argument("-processes", "-p", type=int, help="How many processes to spawn for conversion (default: 4)",
                        default=4)
    parser.add_argument("-weight_data_type", type=str, default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--verbose", action="store_true", help="Provide verbose messages")
    args = parser.parse_args()
    log_format = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=log_format)
    LOGGER.info("\n=============== Argument ===============")
    for key in vars(args):
        LOGGER.info(f"{key}: {vars(args)[key]}")
    LOGGER.info("========================================")

    start_time = datetime.now()
    convert_checkpoint(args)
    stop_time = datetime.now()
    run_time = (stop_time - start_time)
    LOGGER.info("Spend {} (h:m:s) to convert the model".format(run_time))
