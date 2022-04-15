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

from __future__ import print_function

import torch
from examples.pytorch.gpt.utils.gpt import GPT

class ParallelGPT(GPT):
    def __init__(self, head_num, size_per_head, vocab_size, start_id, end_id, layer_num, max_seq_len,
                 tensor_para_size, pipeline_para_size, lib_path, int8_mode):
        super().__init__(head_num, size_per_head, vocab_size, start_id, end_id, layer_num, max_seq_len,
                         tensor_para_size, pipeline_para_size, lib_path, int8_mode)

    def cuda(self):
        self.weights._map(lambda w: w.cuda(self.device))
        if self.int8_mode != 0:
            self.weights._map_int8(lambda w: w.cuda(self.device))

        if self.build_model == True:
            del self.model
            self.build_model = False
        self.model = torch.classes.FasterTransformer.ParallelGptOp(self.head_num, self.size_per_head, 4 * self.head_num * self.size_per_head,
                                                                   self.layer_num, self.vocab_size, self.start_id, self.end_id,
                                                                   self.tensor_para_size, self.pipeline_para_size, self.int8_mode,
                                                                   self.weights.w, self.weights.int8_w, self.weights.scale)
        self.build_model = True
