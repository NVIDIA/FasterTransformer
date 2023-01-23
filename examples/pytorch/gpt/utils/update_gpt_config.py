#!/usr/bin/env python3
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
    parser.add_argument("--max-batch-size", type=int, default=8, help="batch size")
    parser.add_argument("--max-seq-len", type=int, default=256, help="max sequence length")
    parser.add_argument("--beam-width", type=int, default=1, help="beam width for beam search")
    parser.add_argument("--data-type", type=str, default="fp32", help="data type", choices=["fp32", "fp16", "bf16"])
    parser.add_argument(
        "--sampling-top-k",
        type=int,
        default=1,
        help="Candidate (k) value of top k sampling in decoding",
    )
    parser.add_argument(
        "--sampling-top-p",
        type=float,
        default=0.0,
        help="Probability (p) value of top p sampling in decoding",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature of penalty")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="repetition_penalty")
    parser.add_argument("--len-penalty", type=float, default=0.0, help="len_penalty")
    parser.add_argument("--beam-search-diversity-rate", type=float, default=0.0, help="beam_search_diversity_rate")

    # request
    parser.add_argument("--request-batch-size", type=int, default=8, help="batch size")
    parser.add_argument("--request-output-len", type=int, default=32, help="output length")
    parser.add_argument("--model-name", type=str, default="gpt", help="model-name for testing")

    args = parser.parse_args()

    config_path = pathlib.Path(args.config_ini_path)

    config = configparser.ConfigParser()
    config.read(config_path)

    config["ft_instance_hyperparameter"] = {
        "max_batch_size": args.max_batch_size,
        "max_seq_len": args.max_seq_len,
        "beam_width": args.beam_width,
        "top_k": args.sampling_top_k,
        "top_p": args.sampling_top_p,
        "temperature": args.temperature,
        "tensor_para_size": args.tensor_para_size,
        "pipeline_para_size": args.pipeline_para_size,
        "data_type": args.data_type,
        "sparse": 0,
        "int8_mode": 0,
        "enable_custom_all_reduce": 0,
        "model_name": args.model_name,
        "model_dir": args.model_dir,
        "repetition_penalty": args.repetition_penalty,
        "len_penalty": args.len_penalty,
        "beam_search_diversity_rate": args.beam_search_diversity_rate,
    }

    config["request"] = {
        "request_batch_size": args.request_batch_size,
        "request_output_len": args.request_output_len,
        "return_log_probs": "false",
        "context_log_probs": "false",
        "beam_width": args.beam_width,
        "top_k": args.sampling_top_k,
        "top_p": args.sampling_top_p,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "len_penalty": args.len_penalty,
        "beam_search_diversity_rate": args.beam_search_diversity_rate,
    }

    with config_path.open("w") as config_file:
        config.write(config_file)


if __name__ == "__main__":
    main()
