#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

data_dir=./data/
if [[ ! -e $data_dir ]]; then
    mkdir $data_dir
fi

wget https://storage.googleapis.com/xlnet/released_models/cased_L-12_H-768_A-12.zip
unzip cased_L-12_H-768_A-12.zip
mv xlnet_cased_L-12_H-768_A-12 ${data_dir}
mv cased_L-12_H-768_A-12.zip ${data_dir}

wget https://dl.fbaipublicfiles.com/glue/data/STS-B.zip
unzip STS-B.zip
mv STS-B ${data_dir}
mv STS-B.zip ${data_dir}

