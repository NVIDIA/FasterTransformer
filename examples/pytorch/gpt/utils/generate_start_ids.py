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
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-max_batch_size', '--max_batch_size', type=int, required=True, metavar='NUMBER',
                        help='batch size')
    parser.add_argument('-max_input_length', '--max_input_length', type=int, required=True, metavar='NUMBER',
                        help='max input length')
    parser.add_argument('--destination', type=str, default="../examples/cpp/multi_gpu_gpt/start_ids.csv", metavar='STRING',
                        help='Configuration save file. Default is "../examples/cpp/multi_gpu_gpt/start_ids.csv".')

    args = parser.parse_args()
    args_dict = vars(args)

    batch_size = args_dict["max_batch_size"]
    max_input_length = args_dict["max_input_length"]
    path = f"../examples/cpp/multi_gpu_gpt/start_ids.csv"

    with open(args_dict["destination"], 'w') as f:
        ids = ""
        for i in range(batch_size):
            for j in range(max_input_length):
                if j == 0:
                    ids = f"{ids}{random.randint(1, 100)}"
                else:
                    ids = f"{ids}, {random.randint(1, 100)}"
            ids = f"{ids}\n"
        f.write(ids)
