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

from collections import defaultdict
from examples.pytorch.nemo import unpack_nemo_ckpt, UnpackedNemoCheckpointDir, extract_layers_with_prefix
from examples.pytorch.utils import gpu_map_location, WEIGHT2DTYPE, torch2np, cpu_map_location, safe_transpose
from functools import partial
from multiprocessing import Pool

LOGGER = logging.getLogger(__name__)

def merge_adapters(models):
    model_fused = {}
    for key in models[0].keys():
        model_fused[key] = {}
        model_fused[key]["scalers"] = torch.cat([models[i][key]["scalers"] for i in range(len(models))])
    return model_fused


def convert_checkpoint(unpacked_checkpoints_dir: UnpackedNemoCheckpointDir, args: argparse.Namespace):
    nemo_model_config = unpacked_checkpoints_dir.model_config
    infer_tp = args.infer_gpu_num
    saved_dir = pathlib.Path(args.saved_dir) / f"{infer_tp:d}-gpu"
    saved_config_file = saved_dir / "config.ini"

    if not saved_dir.is_dir():
        LOGGER.error("No models found at " + str(saved_dir) + ". Run Nemo converter on base model first")
        sys.exit(1)
    if not saved_config_file.is_file():
        LOGGER.error("No model config at " + str(saved_config_file) + ". Run Nemo converter on base model first")
        sys.exit(1)

    saved_config = configparser.ConfigParser()
    with open(saved_config_file) as f:
        saved_config.read_file(f)

    if ("structure" in saved_config
            and saved_config["structure"].get("ia3_adapted", "False") == "True"):
        LOGGER.error("Model is already ia3-adapted. Refusing to go further")
        sys.exit(1)

    if ("structure" in saved_config
            and int(saved_config["structure"].get("ia3_num_tasks", "0")) > 0
            and args.in_place):
        LOGGER.error("Model already has ia3 weights. Refusing to adapt it in-place.")
        sys.exit(1)

    train_tp = nemo_model_config.get("tensor_model_parallel_size", 1)
    train_pp = nemo_model_config.get("pipeline_model_parallel_size", 1)

    checkpoint_paths = unpacked_checkpoints_dir.get_checkpoints_paths(train_tp, train_pp)
    checkpoint_paths_T = [[checkpoint_paths[i][j] for i in range(train_tp)] for j in range(train_pp)]

    if "encoder" in saved_config and "weight_data_type" in saved_config["encoder"]:
        weight_dt = WEIGHT2DTYPE[saved_config["encoder"]["weight_data_type"]]
    elif "decoder" in saved_config and "weight_data_type" in saved_config["decoder"]:
        weight_dt = WEIGHT2DTYPE[saved_config["decoder"]["weight_data_type"]]
    else:
        LOGGER.info("Could not find existing model data type. Using fp32")
        weight_dt = np.float32

    for ckpt_tp in checkpoint_paths_T:
        model = merge_adapters([torch.load(ckpt, map_location=cpu_map_location) for ckpt in ckpt_tp])

        if args.in_place:
            grouped_layers = defaultdict(list)
            for layer_name, layer in model.items():
                target_file = str(saved_dir / corresponding_ft_name(layer_name))
                grouped_layers[target_file].append((layer_name, layer))

            args_list = grouped_layers.items()
        else:
            args_list = list(model.items())

        with Pool() as p:
            call_fn = multiply_weights if args.in_place else add_ia3_task
            call_fn = partial(call_fn, saved_dir, saved_config, weight_dt, infer_tp)
            ret = p.starmap(call_fn, args_list)


    if args.in_place:
        saved_config["structure"]["ia3_adapted"] = "True"
    else:
        ia3_num_tasks = int(saved_config["structure"].get("ia3_num_tasks", "0")) + 1
        saved_config["structure"]["ia3_num_tasks"] = str(ia3_num_tasks)
        LOGGER.info("Model now has {} IA3 task(s)".format(ia3_num_tasks))

    with open(saved_config_file, "w") as f:
        saved_config.write(f)


def add_ia3_task(saved_dir, saved_config, weight_dt, infer_tp, layer_name, layer):
    ia3_weights_tp = np.array(layer["scalers"], dtype=weight_dt)

    for tp, ia3_weights in zip(range(infer_tp), np.split(ia3_weights_tp, infer_tp)):
        ia3_name = corresponding_ft_name(layer_name, ia3_name=True).format(tp=tp)
        enc_dec = "encoder" if "encoder" in ia3_name else "decoder"
        ia3_filename = saved_dir / ia3_name

        if ia3_filename.is_file():
            previous_weights = np.fromfile(ia3_filename, dtype=weight_dt)
            if "DenseReluDense" in ia3_name:
                hidden_dim = int(saved_config[enc_dec]["d_ff"])
            else:
                hidden_dim = int(saved_config[enc_dec]["d_model"])
            previous_weights = previous_weights.reshape((-1, hidden_dim))

            ia3_weights = np.concatenate((previous_weights, ia3_weights[None, :]))

        ia3_weights.tofile(ia3_filename)


def corresponding_ft_name(ia3_weight, ia3_name=False):
    ia3_suffix = ".ia3" if ia3_name else ""
    name = ""

    is_decoder = "decoder" in ia3_weight
    if is_decoder:
        name += "decoder."
    else:
        name += "encoder."

    layer_id = ia3_weight.split(".")[-1].split(":")[0]
    name += "block." + layer_id + ".layer."

    if "mlp_infused_adapter" in ia3_weight:
        name += ("2" if is_decoder else "1") + ".DenseReluDense"
        name += (ia3_suffix if ia3_name else ".wo") + ".weight.{tp}.bin"
    else:
        is_cross_attention = "inter" in ia3_weight

        rank = "1" if is_cross_attention else "0"
        base_layer = "EncDecAttention" if is_cross_attention else "SelfAttention"
        features = "k" if "key" in ia3_weight else "v"
        features = "qkv" if (is_decoder and not is_cross_attention and not ia3_name) else features
        features += ia3_suffix

        name += ".".join((rank, base_layer, features)) + ".weight.{tp}.bin"

    return name


def reshape(config, name, array):
    enc_dec = "encoder" if "encoder" in name else "decoder"

    if "DenseReluDense" in name:
        dims = int(config[enc_dec]["d_ff"]), int(config[enc_dec]["d_model"])
    elif enc_dec == "decoder" and "SelfAttention.qkv" in name:
        dims = (3, int(config[enc_dec]["d_model"]), int(config[enc_dec]["d_model"]))
    elif "SelfAttention" in name or "EncDecAttention" in name:
        dims = int(config[enc_dec]["d_model"]), int(config[enc_dec]["d_model"])

    return array.reshape(dims)


def multiply_weights(saved_dir, saved_config, weight_dt, infer_tp, weight_file, layers):
    for tp in range(infer_tp):
        weight_file = weight_file.format(tp=tp)
        weights = reshape(saved_config, weight_file, np.fromfile(weight_file, dtype=weight_dt))

        if len(layers) == 1:
            ia3_weights = np.split(np.array(layers[0][1]['scalers'], dtype=weight_dt), infer_tp)[tp]

            if "DenseReluDense" in weight_file:
                ia3_weights = ia3_weights[:, None] # column-wise broadcast
            else:
                ia3_weights = ia3_weights[None, :] # row-wise broadcast
        else:
            if "key_infused_adapter" in layers[0][0]:
                key, value = layers[0][1], layers[1][1]
            else:
                key, value = layers[1][1], layers[0][1]

            key, value = np.array(key['scalers'], dtype=weight_dt), np.array(value['scalers'], dtype=weight_dt)
            key, value = np.split(key, infer_tp)[tp], np.split(value, infer_tp)[tp]

            query = np.ones_like(key)
            ia3_weights = np.stack((query, key, value))[:, None, :]

        ia3_adapted = weights * ia3_weights

        ia3_adapted.tofile(weight_file)


def main():
    """ Enhance your model IA3 features.

    The converter has two modes:
        - Out-of-place: use dedicated IA3 weight files to apply the converters at run-time (default). This allows using multiple IA3 tasks with a single base model. Running this script multiple times with the same output directory and different IA3 tasks will stack IA3 adapters.
        - In-place: pre-process existing model by multiplying weights with IA3 adapters. With this scheme, only one fine-tuned task is supported.
    """

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--in-place",
        dest="in_place",
        action="store_true",
        help="multiply model weights directly"
    )
    parser.add_argument(
        "--saved-dir",
        "-o",
        dest="saved_dir",
        help="folder name of output files",
        required=True,
    )
    parser.add_argument(
        "--in-file",
        "-i",
        dest="in_file",
        help="file name of .nemo IA3 checkpoint file",
        required=True,
    )
    parser.add_argument(
        "--clean-tasks",
        "-c",
        dest="clean_tasks",
        action="store_true",
        help="in Out-of-place mode, clean previous IA3_tasks"
    )
    parser.add_argument(
        "--infer-gpu-num",
        "-i_g",
        dest="infer_gpu_num",
        type=int,
        help="how many gpus for inference",
        required=True,
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="increase verbosity")
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
                load_checkpoints_to_cpu=True,
            )
            LOGGER.info("Spent %s (h:m:s) to unpack NeMo archive", datetime.datetime.now() - start_time)
        else:
            unpacked_checkpoint_dir = UnpackedNemoCheckpointDir(
                input_path, load_checkpoints_to_cpu=True,
            )

        LOGGER.debug("Unpacked NeMo checkpoint contains:")
        for file_path in unpacked_checkpoint_dir.checkpoints_dir.rglob("*"):
            LOGGER.debug("  %s", file_path)

        start_time = datetime.datetime.now()
        convert_checkpoint(unpacked_checkpoint_dir, args)
        LOGGER.info("Spent %s (h:m:s) to convert the model", datetime.datetime.now() - start_time)


if __name__ == "__main__":
    main()
