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
import dataclasses
import datetime
import logging
import multiprocessing
import os
import pathlib
import shutil
import sys
import tempfile
import typing

import numpy as np
import torch  # pytype: disable=import-error
import yaml

# verify if root package is in PYTHONPATH
__root_package_path__ = pathlib.Path(__file__).parent.parent.parent.parent.parent.absolute().as_posix()
if __root_package_path__ not in sys.path:
    print(
        f"[ERROR] add project root directory to your PYTHONPATH with "
        f"'export PYTHONPATH={__root_package_path__}:${{PYTHONPATH}}'"
    )

from examples.pytorch.gpt.utils.gpt import GptModelConfig
from examples.pytorch.nemo import (
    UnpackedNemoCheckpointDir,
    unpack_nemo_ckpt,
    extract_layers_with_prefix,
)
from examples.pytorch.utils import (
    torch2np,
    safe_transpose,
    cpu_map_location,
    gpu_map_location,
    WEIGHT2DTYPE,
)


LOGGER = logging.getLogger(__name__)


# This tool is used to support the new NeMo megatron model trained by pipeline parallel + tensor parallel
def merge_and_convert_process(
    tp_rank: int,
    pp_rank: int,
    saved_dir: typing.Union[str, pathlib.Path],
    factor: int,
    key: str,
    nemo_model_config: typing.Dict[str, typing.Any],
    transformer_model_list: typing.List,
    np_weight_data_type,
    args: argparse.Namespace,
):
    # Config params
    num_layers = nemo_model_config["num_layers"]
    num_attention_heads = nemo_model_config["num_attention_heads"]
    tensor_model_parallel_size = nemo_model_config.get("tensor_model_parallel_size", 1)
    pipeline_model_parallel_size = nemo_model_config.get("pipeline_model_parallel_size", 1)

    if key.find("layers.") != -1:
        layer_index = int(key[7 : key.find(".", 7)])
        saved_key = key.replace(
            "layers.%d." % layer_index,
            "layers.%d." % (layer_index + pp_rank * num_layers // pipeline_model_parallel_size),
        )

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
        if tp_rank == 0:
            val = safe_transpose(transformer_model_list[0][key])
            val = torch2np(val, np_weight_data_type)
            saved_path = saved_dir / f"model.{saved_key}.bin"
            np.squeeze(val).tofile(saved_path)

    elif key.find("attention.dense.weight") != -1 or key.find("mlp.dense_4h_to_h.weight") != -1:
        vals = []
        for k in range(factor):
            val = safe_transpose(transformer_model_list[k][key])
            val = torch2np(val, np_weight_data_type)
            vals.append(val)
        saved_path = saved_dir / f"model.{saved_key}.{tp_rank:d}.bin"
        np.concatenate(vals, axis=0).tofile(saved_path)

    elif key.find("mlp.dense_h_to_4h.weight") != -1 or key.find("mlp.dense_h_to_4h.bias") != -1:
        vals = []
        for k in range(factor):
            val = safe_transpose(transformer_model_list[k][key])
            val = torch2np(val, np_weight_data_type)
            vals.append(val)
        saved_path = saved_dir / f"model.{saved_key}.{tp_rank:d}.bin"
        np.concatenate(vals, axis=-1).tofile(saved_path)

    elif key.find("attention.query_key_value.bias") != -1:
        vals = []
        for k in range(factor):
            val = safe_transpose(transformer_model_list[k][key])
            val = torch2np(val, np_weight_data_type)
            local_dim = int(val.shape[-1] / 3)
            num_splits = 3
            head_num = num_attention_heads // tensor_model_parallel_size
            size_per_head = local_dim // head_num
            val = val.reshape(head_num, num_splits, size_per_head)
            val = val.transpose(1, 0, 2)
            val = val.reshape(3, local_dim)
            vals.append(val)

        saved_path = saved_dir / f"model.{saved_key}.{tp_rank:d}.bin"
        np.concatenate(vals, axis=-1).tofile(saved_path)

    elif key.find("attention.query_key_value.weight") != -1:
        vals = []
        for k in range(factor):
            val = safe_transpose(transformer_model_list[k][key])
            val = torch2np(val, np_weight_data_type)
            hidden_dim = val.shape[0]
            local_dim = int(val.shape[-1] / 3)
            num_splits = 3
            head_num = num_attention_heads // tensor_model_parallel_size
            size_per_head = local_dim // head_num
            val = val.reshape(hidden_dim, head_num, num_splits, size_per_head)
            val = val.transpose(0, 2, 1, 3)
            val = val.reshape(hidden_dim, 3, local_dim)
            vals.append(val)

        saved_path = saved_dir / f"model.{saved_key}.{tp_rank:d}.bin"
        if args.fused_qkv == 1:
            np.concatenate(vals, axis=-1).tofile(saved_path)
        elif args.fused_qkv == 0:
            np.concatenate(vals, axis=-1).transpose(1, 0, 2).tofile(saved_path)
    else:
        LOGGER.error("cannot find key '%s'", key)


def split_and_convert_process(
    tp_rank: int,
    pp_rank: int,
    saved_dir: typing.Union[str, pathlib.Path],
    factor: int,
    key: str,
    nemo_model_config: typing.Dict[str, typing.Any],
    transformer_model_list: typing.List,
    np_weight_data_type,
    args: argparse.Namespace,
):

    # Config params
    num_layers = nemo_model_config["num_layers"]
    num_attention_heads = nemo_model_config["num_attention_heads"]
    tensor_model_parallel_size = nemo_model_config.get("tensor_model_parallel_size", 1)
    pipeline_model_parallel_size = nemo_model_config.get("pipeline_model_parallel_size", 1)

    # Handle model[key] weights
    transformer_model = transformer_model_list[0]
    val = safe_transpose(transformer_model[key])
    val = torch2np(val, np_weight_data_type)
    if key.find("layers.") != -1:
        layer_index = (int)(key[7 : key.find(".", 7)])
        saved_key = key.replace(
            "layers.%d." % layer_index,
            "layers.%d." % (layer_index + pp_rank * num_layers // pipeline_model_parallel_size),
        )

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
        if tp_rank == 0:
            saved_path = saved_dir / f"model.{saved_key}.bin"
            val.tofile(saved_path)

    elif key.find("attention.dense.weight") != -1 or key.find("mlp.dense_4h_to_h.weight") != -1:
        split_vals = np.split(val, factor, axis=0)
        for j in range(factor):
            saved_path = saved_dir / f"model.{saved_key}.{tp_rank * factor + j:d}.bin"
            split_vals[j].tofile(saved_path)

    elif key.find("mlp.dense_h_to_4h.weight") != -1 or key.find("mlp.dense_h_to_4h.bias") != -1:
        split_vals = np.split(val, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir / f"model.{saved_key}.{tp_rank * factor + j:d}.bin"
            split_vals[j].tofile(saved_path)

    elif key.find("attention.query_key_value.bias") != -1:
        local_dim = int(val.shape[-1] / 3)

        num_splits = 3
        head_num = num_attention_heads // tensor_model_parallel_size
        size_per_head = local_dim // head_num
        val = val.reshape(head_num, num_splits, size_per_head)
        val = val.transpose(1, 0, 2)

        val = val.reshape(3, local_dim)
        split_vals = np.split(val, factor, axis=-1)

        for j in range(factor):
            saved_path = saved_dir / f"model.{saved_key}.{tp_rank * factor + j:d}.bin"
            split_vals[j].tofile(saved_path)

    elif key.find("attention.query_key_value.weight") != -1:
        hidden_dim = val.shape[0]
        local_dim = int(val.shape[-1] / 3)

        num_splits = 3
        head_num = num_attention_heads
        size_per_head = hidden_dim // head_num
        head_num = head_num // tensor_model_parallel_size
        val = val.reshape(hidden_dim, head_num, num_splits, size_per_head)
        val = val.transpose(0, 2, 1, 3)

        val = val.reshape(hidden_dim, 3, local_dim)
        split_vals = np.split(val, factor, axis=-1)

        for j in range(factor):
            saved_path = saved_dir / f"model.{saved_key}.{tp_rank * factor + j:d}.bin"
            split_vals[j].tofile(saved_path)

    else:
        LOGGER.error("cannot find key '%s'", key)


def convert_checkpoint(unpacked_checkpoints_dir: UnpackedNemoCheckpointDir, args):
    nemo_model_config = unpacked_checkpoints_dir.model_config

    checkpoints_paths = unpacked_checkpoints_dir.get_checkpoints_paths(
        nemo_model_config.get("tensor_model_parallel_size", 1),
        nemo_model_config.get("pipeline_model_parallel_size", 1),
    )

    # if checkpoints files could be found - start preparing output dir
    saved_dir = _prepare_saved_dir(args)

    map_location_fn = cpu_map_location if bool(args.load_checkpoints_to_cpu) else gpu_map_location
    np_weight_data_type = WEIGHT2DTYPE[args.weight_data_type]

    # load position_embedding from rank 0
    model_00 = torch.load(checkpoints_paths[0][0], map_location=map_location_fn)
    val = model_00.get("state_dict", model_00)["model.language_model.embedding.position_embeddings.weight"]
    # not weight, do not need to transpose
    val = torch2np(val, np_weight_data_type)
    val.tofile(saved_dir / "model.wpe.bin")
    del model_00

    w_e_list = []

    training_tensor_para_size = nemo_model_config.get("tensor_model_parallel_size", 1)
    training_pipeline_para_size = nemo_model_config.get("pipeline_model_parallel_size", 1)
    inference_tensor_para_size = args.infer_gpu_num

    if training_tensor_para_size > inference_tensor_para_size:
        assert training_tensor_para_size % inference_tensor_para_size == 0
        is_merge_ckpt = True
        factor = int(training_tensor_para_size / inference_tensor_para_size)
    else:
        assert inference_tensor_para_size % training_tensor_para_size == 0
        is_merge_ckpt = False
        factor = int(inference_tensor_para_size / training_tensor_para_size)

    main_loop = min(training_tensor_para_size, inference_tensor_para_size)

    torch.multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy("file_system")
    pool = multiprocessing.Pool(args.processes)
    for i in range(main_loop):
        for j in range(training_pipeline_para_size):

            transformer_models = []
            if is_merge_ckpt:
                for k in range(factor):
                    rank_weights = checkpoints_paths[i * factor + k][j]
                    model = torch.load(rank_weights, map_location=map_location_fn)
                    if j == 0:
                        val = model.get("state_dict", model)["model.language_model.embedding.word_embeddings.weight"]
                        val = torch2np(val, np_weight_data_type)
                        w_e_list.append(val)
                    layers = extract_layers_with_prefix(model, "model.language_model.encoder.")
                    transformer_models.append(layers)
            else:
                rank_weights = checkpoints_paths[i][j]
                model = torch.load(rank_weights, map_location=map_location_fn)

                if j == 0:
                    val = model.get("state_dict", model)["model.language_model.embedding.word_embeddings.weight"]
                    val = torch2np(val, np_weight_data_type)
                    w_e_list.append(val)
                layers = extract_layers_with_prefix(model, "model.language_model.encoder.")
                transformer_models.append(layers)

            pool.starmap(
                merge_and_convert_process if is_merge_ckpt else split_and_convert_process,
                [
                    (
                        i,  # tp_rank
                        j,  # pp_rank
                        saved_dir,
                        factor,
                        key,
                        nemo_model_config,
                        transformer_models,
                        np_weight_data_type,
                        args,
                    )
                    for key in transformer_models[0]
                ],
            )

    pool.close()
    pool.join()

    val = np.concatenate(w_e_list, axis=0)
    val.tofile(saved_dir / "model.wte.bin")

    vocab_size = val.shape[0]

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

    gpt_model_config = GptModelConfig.from_nemo_package(
        args=args,
        nemo_model_config=nemo_model_config,
        vocab_size=vocab_size,
        bos_id=bos_id,
        eos_id=eos_id,
    )

    # Configuration for the model (load by triton backends)
    config = configparser.ConfigParser()
    config["gpt"] = {k: str(v) for k, v in dataclasses.asdict(gpt_model_config).items()}
    try:
        config_path = saved_dir / "config.ini"
        with config_path.open("w") as config_file:
            config.write(config_file)
    except Exception as e:
        LOGGER.error("Fail to save the config; %s", e)


def _prepare_saved_dir(args):
    saved_dir = pathlib.Path(args.saved_dir)
    if args.fused_qkv == 1:
        saved_dir = saved_dir / f"{args.infer_gpu_num:d}-gpu/"
    else:
        saved_dir = saved_dir / f"unfusedQKV-{args.infer_gpu_num:d}-gpu"
    if saved_dir.exists():
        LOGGER.error(f"Remove %s target directory before running conversion", saved_dir)
        sys.exit(1)
    saved_dir.mkdir(parents=True)
    return saved_dir


def prompt_convert(args, prompt_config, prompt_weights):

    prompt_templates = prompt_config["task_templates"]

    # model config save dir
    config_saved_dir = _prepare_saved_dir(args)

    # Configuration for the model (load by triton backends)
    config_path = config_saved_dir / "config.ini"
    config = configparser.ConfigParser()
    with config_path.open("r") as config_file:
        config.read_file(config_file)

    num_tasks = len(prompt_templates)
    prompt_learning_type = 3  # p_prompt_tuning
    prompt_learning_start_id = 50257  # hard code here
    config["gpt"]["num_tasks"] = str(num_tasks)
    config["gpt"]["prompt_learning_start_id"] = str(prompt_learning_start_id)
    config["gpt"]["prompt_learning_type"] = str(prompt_learning_type)

    for task_name_id, prompt_task in enumerate(prompt_templates):
        prompt_task_name = prompt_task["taskname"]
        prompt_length = int(prompt_task["total_virtual_tokens"])
        config[f"task_{task_name_id:d}"] = {}
        config[f"task_{task_name_id:d}"]["task_name"] = prompt_task_name
        config[f"task_{task_name_id:d}"]["prompt_length"] = str(prompt_length)
        prompt_task_weights = prompt_weights["prompt_table"][
            f"prompt_table.{prompt_task_name}.prompt_embeddings.weight"
        ]
        # put converted prompts weights to the model weights saved dir
        prompt_task_weights_output_path = config_saved_dir / f"model.prompt_table.{prompt_task_name}.weight.bin"
        val = torch2np(prompt_task_weights)
        val.tofile(prompt_task_weights_output_path)

    with config_path.open("w") as config_file:
        config.write(config_file)

    LOGGER.info(">>>>>>>>>>>>>>>> model saved config")
    LOGGER.info(config_path.read_text())


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
        help="file name of .nemo checkpoint file",
        required=True,
    )
    parser.add_argument(
        "--prompt-in-file",
        "-prompt_in_file",
        "-p_i",
        help="file name of .nemo prompt checkpoint file",
    )
    parser.add_argument(
        "--prompt-saved-dir",
        "-prompt_saved_dir",
        "-p_o",
        help="folder name of prompt checkpoint output files",
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
        "--fused-qkv",
        "-fused_qkv",
        type=int,
        choices=[0, 1],
        default=1,
        help="Fuse the qkv weights or not",
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
        "--weight-data-type",
        "-weight_data_type",
        choices=["fp32", "fp16"],
        default="fp32",
        help="Data type of results weights",
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

        start_time = datetime.datetime.now()
        convert_checkpoint(unpacked_checkpoint_dir, args)
        LOGGER.info("Spent %s (h:m:s) to convert the model", datetime.datetime.now() - start_time)

    map_location_fn = cpu_map_location if bool(args.load_checkpoints_to_cpu) else gpu_map_location
    # prompt checkpoint converting
    if args.prompt_in_file is not None:
        start_time = datetime.datetime.now()
        assert args.prompt_saved_dir is not None
        unpack_nemo_ckpt(args.prompt_in_file, args.prompt_saved_dir)
        LOGGER.info("Spent %s (h:m:s) to unpack NeMo prompt archive", datetime.datetime.now() - start_time)

        model_config_yaml = "model_config.yaml"
        model_weights_ckpt = "model_weights.ckpt"
        prompt_config_file = open(os.path.join(args.prompt_saved_dir, model_config_yaml), "r")
        prompt_config = yaml.full_load(prompt_config_file)
        LOGGER.info(prompt_config)

        start_time = datetime.datetime.now()
        prompt_weights = torch.load(
            os.path.join(args.prompt_saved_dir, model_weights_ckpt),
            map_location=map_location_fn,
        )
        prompt_convert(args, prompt_config, prompt_weights)
        LOGGER.info(f"Spent %s (h:m:s) to unpack convert prompt model", datetime.datetime.now() - start_time)


if __name__ == "__main__":
    main()
