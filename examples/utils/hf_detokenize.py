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

    sentences = []
    with open(in_file) as f:
        for line in f:
            sentences.append(json.dumps(tokenizer.decode([int(tok) for tok in line.split(" ") if tok != "\n"])))

    for sentence in sentences:
        print(sentence)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("in_file")
    parser.add_argument("model_name", nargs="?", default="gpt2")

    main(**vars(parser.parse_args()))

