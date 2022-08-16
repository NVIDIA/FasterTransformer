#!/usr/bin/env python3
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
import pathlib


def main():
    parser = argparse.ArgumentParser(
        description="Script updating GPT config.ini hyper-parameters and requests parameters"
    )
    # config.ini path
    parser.add_argument("--config-ini-path", required=True, help="Path to config.ini file to be updated")

    # FT hyperparameters
    parser.add_argument("--model-dir", type=str, required=True, help="Model path prefix")
    parser.add_argument("--tensor-para-size", type=int, required=True, help="tensor parallelism size")
    parser.add_argument("--pipeline-para-size", type=int, required=True, help="layer parallelism size")
    parser.add_argument("--data-type", type=str, default="fp32", help="data type", choices=["fp32", "fp16", "bf16"])
    # request
    parser.add_argument("--request-batch-size", type=int, default=8, help="batch size")
    parser.add_argument("--request-seq-len", type=int, default=32, help="output length")

    args = parser.parse_args()

    config_path = pathlib.Path(args.config_ini_path)

    config = configparser.ConfigParser()
    config.read(config_path)

    config["ft_instance_hyperparameter"] = {
        "tensor_para_size": args.tensor_para_size,
        "pipeline_para_size": args.pipeline_para_size,
        "data_type": args.data_type,
        "is_sparse": 0,
        "is_remove_padding": 1,
        "int8_mode": 0,
        "enable_custom_all_reduce": 0,
        "model_name": "bert_base",
        "model_dir": args.model_dir,
    }

    config["request"] = {
        "request_batch_size": args.request_batch_size,
        "request_seq_len": args.request_seq_len,
    }

    with config_path.open("w") as config_file:
        config.write(config_file)

if __name__ == "__main__":
    main()
