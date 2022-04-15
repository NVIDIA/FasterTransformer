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

import csv
import numpy as np
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
base_path = dir_path + "/../../../.."
sys.path.append(base_path)

from examples.pytorch.gpt.utils import gpt_token_encoder as encoder


def get_tokenizer(vocab_file=None, bpe_file=None):
    vocab_file = vocab_file if vocab_file is not None else base_path + "/models/gpt2-vocab.json"
    bpe_file = bpe_file if bpe_file is not None else base_path + "/models/gpt2-merges.txt"

    tokenizer = encoder.get_encoder(vocab_file, bpe_file)

    return tokenizer


def to_word_list_format(word_dict):
    tokenizer = get_tokenizer()

    flat_ids = []
    offsets = []
    for word_dict_item in word_dict:
        item_flat_ids = []
        item_offsets = []

        words = list(csv.reader(word_dict_item))[0]
        for word in words:
            ids = tokenizer.encode(word)

            if len(ids) == 0:
                continue

            item_flat_ids += ids
            item_offsets.append(len(ids))

        flat_ids.append(np.array(item_flat_ids))
        offsets.append(np.cumsum(np.array(item_offsets)))

    pad_to = max(1, max(len(ids) for ids in flat_ids))

    for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
        flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
        offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

    return np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))


def save_word_list(filename, word_list):
    with open(filename, "w") as f:
        writer = csv.writer(f)
        for word_list_item in word_list:
            writer.writerow(word_list_item[0].tolist())
            writer.writerow(word_list_item[1].tolist())


def load_word_list(filename):
    with open(filename, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
    data = np.array(data, dtype=np.int32)
    batch_size_x2, list_len = data.shape

    return data.reshape((batch_size_x2 // 2, 2, list_len))


def test_csv_read_write():
    filename = sys.argv[1]

    test_words = [["one,two,three, one, two, three, one two three"], ["four"]]
    word_list = to_word_list_format(test_words)

    save_word_list(filename, word_list)
    read_word_list = load_word_list(filename)

    assert np.all(word_list == read_word_list)

if __name__ == "__main__":
    test_csv_read_write()
