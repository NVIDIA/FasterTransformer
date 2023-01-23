# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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
import json
import pathlib
import typing

import numpy as np
import torch
import transformers

from utils.gpt import GptInitModelParameters, GptRuntimeModelParameters
from utils.parallel_gpt import ParallelGPT

class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super().default(obj)

class LambadaDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, seq_len):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        with open(path, "r") as f:
            texts = [json.loads(line)["text"] for line in f.readlines()]

            # this whitespace preprocessing (additional space and stripping) is required
            labels = [" " + text.split()[-1] for text in texts]
            inputs = [text[: text.rfind(label)].strip() for text, label in zip(texts, labels)]
            self.encodings = self.tokenizer(
                inputs,
                labels,
                padding="max_length",
                max_length=self.seq_len,
                return_token_type_ids=True,
                return_tensors="pt",
            )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings["token_type_ids"][idx],
        }


@dataclasses.dataclass
class Metric:
    acc: float


@dataclasses.dataclass
class RequestAndResult:
    prompt: str
    model_answer: str
    target: str
    input_ids: typing.List[int]
    input_len: int
    output_len: int
    init_model_parameters: GptInitModelParameters
    runtime_model_parameters: GptRuntimeModelParameters
    output_ids: typing.List[int]
    metrics: Metric


def _read_config_ini(args, checkpoint_path):
    config_reader = configparser.ConfigParser()
    config_ini_files_in_checkpoint_dir = list(checkpoint_path.rglob("config.ini"))
    if args.config_ini_path is None and not config_ini_files_in_checkpoint_dir:
        raise RuntimeError(
            f"Missing config.ini file in {checkpoint_path}. Use --config-ini-path to point config.ini to load"
        )
    config_ini_path = pathlib.Path(args.config_ini_path or config_ini_files_in_checkpoint_dir[0])
    if not config_ini_path.is_file():
        raise FileNotFoundError(f"Missing {config_ini_path}")
    else:
        config_reader.read(config_ini_path.as_posix())
    return config_reader


def _get_model(args, config_reader):
    init_parameters = GptInitModelParameters.from_args(args, config_reader)
    print("\n=============== GPT params ===============")
    for key, value in dataclasses.asdict(init_parameters).items():
        print(f"{key}: {value}")
    print(f"lib_path: {args.lib_path}")
    print("========================================")

    gpt_params = init_parameters.gpt_init_kwargs()
    gpt = ParallelGPT(**gpt_params, lib_path=args.lib_path)

    if not gpt.load(ckpt_path=args.checkpoint_path):
        print("[WARNING] Checkpoint file not found. Model loading is skipped.")

    if init_parameters.sparse:
        gpt.sparse()

    gpt.eval()

    return gpt


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to FasterTransformer checkpoint dir")
    parser.add_argument("--lib-path", type=str, required=True, help="Path of FasterTransformer PyTorch GPT op library")
    parser.add_argument(
        "--config-ini-path",
        type=str,
        help="Path to config.ini file. If not provided <checkpoint_path>/config.ini will be used.",
    )
    parser.add_argument("--lambada-path", type=str, help="LAMBADA task data path")
    parser.add_argument("--output-path", type=str, help="Path to sample output file.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")

    GptInitModelParameters.update_argparser(parser)
    GptRuntimeModelParameters.update_argparser(parser)

    args = parser.parse_args()

    print("\n============== Arguments ===============")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("========================================")

    checkpoint_path = pathlib.Path(args.checkpoint_path)

    config_reader = _read_config_ini(args, checkpoint_path)

    gpt = _get_model(args, config_reader)

    vocab_path = checkpoint_path / "vocab.json"
    merges_path = checkpoint_path / "merges.txt"
    max_seq_len = config_reader.getint("ft_instance_hyperparameter", "max_seq_len")

    tokenizer = transformers.GPT2TokenizerFast(vocab_path.as_posix(), merges_path.as_posix())
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    if args.lambada_path:
        dataset = LambadaDataset(args.lambada_path, tokenizer=tokenizer, seq_len=max_seq_len)
    else:
        from datasets import load_dataset
        dataset = load_dataset("lambada", split="validation")

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    runtime_parameters = GptRuntimeModelParameters.from_args(args, config_reader)
    inference_parameters_dict = dataclasses.asdict(runtime_parameters)
    print("\n=========== Inference params ===========")
    for key, value in inference_parameters_dict.items():
        print(f"{key}: {value}")
    print("========================================")

    beam_idx = 0  # use only 1st beam result

    requested_num = 0
    correct_num = 0
    results = {"output": {"lambada": []}, "results": {"lambada": {}}}
    with torch.no_grad():

        for entries in data_loader:
            inputs_tokens_batch = [
                input_ids[(attention_mask == 1) & (token_type_ids == 0)]
                for input_ids, attention_mask, token_type_ids in zip(
                    entries["input_ids"], entries["attention_mask"], entries["token_type_ids"]
                )
            ]
            labels_tokens_batch = [
                input_ids[(attention_mask == 1) & (token_type_ids == 1)]
                for input_ids, attention_mask, token_type_ids in zip(
                    entries["input_ids"], entries["attention_mask"], entries["token_type_ids"]
                )
            ]

            inputs_tokens_batch_padded = [
                torch.nn.functional.pad(
                    input_tokens,
                    pad=[0, (max_seq_len - input_tokens.shape[0])],
                    mode="constant",
                    value=tokenizer.pad_token_id,
                )
                for input_tokens in inputs_tokens_batch
            ]

            input_tokens_lengths = [input_tokens.shape[0] for input_tokens in inputs_tokens_batch]
            # max is required due to scalar is used for output_seq_len input
            expected_tokens_lengths = max([label_tokens.shape[0] for label_tokens in labels_tokens_batch])

            start_ids = torch.stack(inputs_tokens_batch_padded)  # shape=(batch_size, max_seq_len)
            runtime_parameters = GptRuntimeModelParameters.from_args(args, config_reader, start_ids.shape[0])
            inference_parameters_dict = dataclasses.asdict(runtime_parameters)

            start_ids = start_ids.to(torch.int32)
            result_all_tokens_batch = gpt(
                start_ids,
                torch.IntTensor(input_tokens_lengths),
                expected_tokens_lengths,
                **inference_parameters_dict,
            )

            results_idxes = [
                torch.nonzero(token_type_ids, as_tuple=True)[0] for token_type_ids in entries["token_type_ids"]
            ]
            results_tokens_batch = [
                result_tokens_ids[beam_idx][result_idxes].cpu()
                for result_tokens_ids, result_idxes in zip(result_all_tokens_batch, results_idxes)
            ]

            labels_tokens_batch = [tokens.cpu() for tokens in labels_tokens_batch]
            results_tokens_batch = [tokens.cpu() for tokens in results_tokens_batch]

            result_text_batch = tokenizer.batch_decode(results_tokens_batch)
            input_text_batch = tokenizer.batch_decode(inputs_tokens_batch)
            label_text_batch = tokenizer.batch_decode(labels_tokens_batch)

            for idx in range(len(inputs_tokens_batch)):
                is_correct_answer = torch.all(labels_tokens_batch[idx] == results_tokens_batch[idx])
                correct_num += int(is_correct_answer)
                result = RequestAndResult(
                    prompt=input_text_batch[idx],
                    model_answer=result_text_batch[idx],
                    target=label_text_batch[idx],
                    input_ids=list(map(int, inputs_tokens_batch[idx])),
                    input_len=int(input_tokens_lengths[idx]),
                    output_len=expected_tokens_lengths,
                    init_model_parameters=GptInitModelParameters.from_args(args, config_reader),
                    runtime_model_parameters=runtime_parameters.slice_args(idx),
                    output_ids=list(map(int, result_all_tokens_batch[idx][beam_idx])),
                    metrics=Metric(acc=float(is_correct_answer)),
                )
                results["output"]["lambada"].append(dataclasses.asdict(result))

            requested_num += len(inputs_tokens_batch)

    accuracy = correct_num * 100 / requested_num
    print(f"[INFO] accuracy: {accuracy:0.4f}% (total : {requested_num})")

    # Dump prediction json
    results["results"]["lambada"]["acc"] = accuracy
    if args.output_path:
        output_json_path = pathlib.Path(args.output_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with output_json_path.open(mode="w") as json_file:
            json.dump(results, json_file, indent=2, cls=TensorEncoder)
        print(f"[INFO] Detailed test results saved to {output_json_path}")


if __name__ == "__main__":
    main()
