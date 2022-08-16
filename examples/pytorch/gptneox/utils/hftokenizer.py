#! /usr/bin/env python3

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

from argparse import ArgumentParser
from tokenizers import Tokenizer
from typing import List, Union


class HFTokenizer:
    def __init__(self, vocab_file):
        self.tokenizer = Tokenizer.from_file(vocab_file)

    def tokenize(self, text: str):
        return self.tokenizer.encode(text).ids

    def tokenize_batch(self, text_batch: Union[List[str], str]):
        return self.tokenizer.encode_batch(text_batch)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)


def handle_args():
    parser = ArgumentParser()
    parser.add_argument("in_file")
    parser.add_argument("--out-file")
    parser.add_argument("--tokenizer", default="../models/20B_tokenizer.json")
    parser.add_argument("--action", choices=["tokenize", "detokenize", "auto"], default="auto")

    return parser.parse_args()


def main(in_file, tokenizer, out_file, action):
    tokenizer = HFTokenizer(tokenizer)

    with open(in_file) as f:
        lines = f.read().split('\n')

    in_lines = None
    do = None
    if action != "tokenize":
        if in_lines is None:
            try:
                in_lines = [[int(tok) for tok in line.split(' ') if tok] for line in lines if line]
                do = "detokenize"
            except ValueError:
                pass
        if in_lines is None:
            try:
                in_lines = [[int(tok) for tok in line.split(', ') if tok] for line in lines if line]
                do = "detokenize"
            except ValueError:
                pass

    if action != "detokenize":
        if in_lines is None:
            try:
                in_lines = [line for line in lines if line]
                do = "tokenize"
            except ValueError:
                pass

    if do is not None:
        if do == "detokenize":
            output = [tokenizer.detokenize(token_list) for token_list in in_lines]
        else:
            output = [tokenizer.tokenize(line) for line in in_lines]
            output = [",".join(str(tok) for tok in tok_seq) for tok_seq in output]

        if args.out_file:
            with open(out_file, "w") as f:
                f.write("\n".join(output))
        else:
            print("\n---\n".join(output))


if __name__ == "__main__":
    args = handle_args()
    main(args.in_file, args.tokenizer, args.out_file, args.action)
