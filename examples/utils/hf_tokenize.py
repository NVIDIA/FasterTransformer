#! /usr/bin/env python3
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
import json
from argparse import ArgumentParser
from transformers import AutoTokenizer

def main(in_file, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    tokens = []
    with open(in_file) as f:
        for line in f:
            tokens.append(json.loads(line))
    tokens = tokenizer(tokens, padding=True)["input_ids"]

    token_string = "\n".join((",".join(str(token) for token in token_line)) for token_line in tokens)

    print(token_string)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("in_file")
    parser.add_argument("model_name", nargs="?", default="gpt2")

    main(**vars(parser.parse_args()))
