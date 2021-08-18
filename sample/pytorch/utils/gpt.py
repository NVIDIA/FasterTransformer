# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import os
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist


class GPTWeights(object):
    def __init__(self, head_num, size_per_head, layer_num, vocab_size, max_seq_len, tensor_para_size, layer_para_size):
        assert(head_num % tensor_para_size == 0)

        self.head_num = head_num
        self.size_per_head = size_per_head
        self.layer_num = layer_num
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.tensor_para_size = tensor_para_size
        self.layer_para_size = layer_para_size
        self.layers_per_device = layer_num // layer_para_size
        
        local_head_num = head_num // tensor_para_size
        global_head_num = head_num
        local_hidden_units = local_head_num * size_per_head
        global_hidden_units = global_head_num * size_per_head
        local_inner_size = local_hidden_units * 4

        self.local_head_num = local_head_num
        self.global_head_num = global_head_num
        self.local_hidden_units = local_hidden_units
        self.global_hidden_units = global_hidden_units
        self.local_inner_size = local_inner_size

        self.w = []
        # Before Transformer blocks
        self.w.append(torch.zeros(vocab_size, global_hidden_units))   # embedding_table
        self.w.append(torch.zeros(max_seq_len, global_hidden_units))   # position_encoding_table
        # Transformer blocks
        self.w.append([torch.zeros(global_hidden_units)] * layer_num)   # self_layernorm_gamma
        self.w.append([torch.zeros(global_hidden_units)] * layer_num)   # self_layernorm_beta
        self.w.append([torch.zeros(global_hidden_units, local_hidden_units * 3)] * layer_num)   # self_kernel
        self.w.append([torch.zeros(local_hidden_units * 3)] * layer_num)   # self_bias
        self.w.append([torch.zeros(local_hidden_units, global_hidden_units)] * layer_num)   # self_output_kernel
        self.w.append([torch.zeros(global_hidden_units)] * layer_num)   # self_output_bias
        self.w.append([torch.zeros(global_hidden_units)] * layer_num)   # ffn_layernorm_gamma
        self.w.append([torch.zeros(global_hidden_units)] * layer_num)   # ffn_layernorm_beta
        self.w.append([torch.zeros(global_hidden_units, local_inner_size)] * layer_num)   # ffn_kernel1
        self.w.append([torch.zeros(local_inner_size, global_hidden_units)] * layer_num)   # ffn_kernel2
        self.w.append([torch.zeros(local_inner_size)] * layer_num)   # ffn_bias1
        self.w.append([torch.zeros(global_hidden_units)] * layer_num)   # ffn_bias2
        # After Transformer blocks
        self.w.append(torch.zeros(global_hidden_units))   # layernorm_gamma
        self.w.append(torch.zeros(global_hidden_units))   # layernorm_beta

        # Initialization
        self._map(lambda w : torch.nn.init.normal_(w, mean=0., std=1.))

    def __getitem__(self, idx):
        return self.w[idx]

    def __setitem__(self, idx, val):
        self.w[idx] = val

    def __len__(self):
        return len(self.w)

    def _map(self, func):
        for i in range(len(self.w)):
            if isinstance(self.w[i], list):
                for j in range(len(self.w[i])):
                    self.w[i][j] = func(self.w[i][j])
            else:
                self.w[i] = func(self.w[i])

    def load(self, ckpt_path, tensor_para_rank, layer_para_rank):
        w = []

        # Load
        w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.wte.bin", dtype=np.single)))

        wpe = torch.from_numpy(np.fromfile(ckpt_path + "/model.wpe.bin", dtype=np.single)).reshape(-1, self.global_hidden_units)
        assert self.max_seq_len <= wpe.size(0), "max_seq_len must not exceed the value of maximum sequence length during traning."
        wpe = wpe[:self.max_seq_len]  # excludes weights not to really use.
        w.append(wpe)

        is_load = lambda i: i >= self.layers_per_device * layer_para_rank and i < self.layers_per_device * (layer_para_rank + 1)
        w.append([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.input_layernorm.weight.bin".format(i), dtype=np.single)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.append([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.input_layernorm.bias.bin".format(i), dtype=np.single)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.append([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.attention.query_key_value.weight.{}.bin".format(i, tensor_para_rank), dtype=np.single)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.append([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.attention.query_key_value.bias.{}.bin".format(i, tensor_para_rank), dtype=np.single)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.append([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.attention.dense.weight.{}.bin".format(i, tensor_para_rank), dtype=np.single)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.append([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.attention.dense.bias.bin".format(i), dtype=np.single)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.append([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.post_attention_layernorm.weight.bin".format(i), dtype=np.single)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.append([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.post_attention_layernorm.bias.bin".format(i), dtype=np.single)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.append([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.mlp.dense_h_to_4h.weight.{}.bin".format(i, tensor_para_rank), dtype=np.single)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.append([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.mlp.dense_4h_to_h.weight.{}.bin".format(i, tensor_para_rank), dtype=np.single)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.append([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.mlp.dense_h_to_4h.bias.{}.bin".format(i, tensor_para_rank), dtype=np.single)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.append([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.mlp.dense_4h_to_h.bias.bin".format(i), dtype=np.single)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.final_layernorm.weight.bin", dtype=np.single)))
        w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.final_layernorm.bias.bin", dtype=np.single)))

        # Reshape
        try:
            for i in range(len(w)):
                if isinstance(w[i], list):
                    for j in range(len(w[i])):
                        self.w[i][j] = w[i][j].reshape(self.w[i][j].shape) if w[i][j].nelement() > 0 else self.w[i][j]
                else:
                    self.w[i] = w[i].reshape(self.w[i].shape)

        except RuntimeError:
            raise RuntimeError("head_num, size_per_head, vocab_size, and max_seq_len must be the same as the ones during training.")

class GPT(nn.Module):
    def __init__(self, head_num, size_per_head, vocab_size, start_id, end_id, layer_num, top_k, top_p, temperature,
                 output_len, max_seq_len, tensor_para_size, layer_para_size, layer_para_batch_size, max_batch_size, lib_path):
        super().__init__()
        self.head_num = head_num
        self.size_per_head = size_per_head
        self.vocab_size = vocab_size
        self.start_id = start_id
        self.end_id = end_id
        self.layer_num = layer_num
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.output_len = output_len
        self.max_seq_len = max_seq_len
        self.tensor_para_size = tensor_para_size
        self.layer_para_size = layer_para_size
        self.layer_para_batch_size = layer_para_batch_size
        self.max_batch_size = max_batch_size

        assert torch.cuda.is_available(), "CUDA is required for this model."

        assert head_num % tensor_para_size == 0, "head_num must be a multiple of tensor_para_size."
        assert layer_num % layer_para_size == 0, "layer_num must be a multiple of layer_para_size."

        # Prepare weights
        self.weights = GPTWeights(head_num, size_per_head, layer_num, vocab_size, max_seq_len, tensor_para_size, layer_para_size)

        # Prepare for tensor/pipeline parallel
        dist.init_process_group(backend='mpi')
        self.rank = dist.get_rank()
        self.device_count = torch.cuda.device_count()
        self.device = self.rank % self.device_count
        torch.cuda.set_device(self.device)

        world_size = dist.get_world_size()
        assert world_size == tensor_para_size * layer_para_size, "tensor_para_size * layer_para_size must be equal to world_size."

        self.tensor_para_rank = self.rank % self.tensor_para_size
        self.layer_para_rank = self.rank // self.tensor_para_size

        # Load the C++ model into Pytorch model.
        torch.classes.load_library(os.path.abspath(lib_path))

        self.model = None

    def load(self, ckpt_path):
        self.weights.load(ckpt_path, tensor_para_rank=self.tensor_para_rank, layer_para_rank=self.layer_para_rank)


    def half(self):
        self.weights._map(lambda w : w.half())
        if self.model is not None:
            self.cuda()


    def cuda(self):
        self.weights._map(lambda w : w.cuda(self.device))
        self.model = torch.classes.FasterTransformer.GPT(self.head_num, self.size_per_head, self.vocab_size,
                                                        self.start_id, self.end_id, self.layer_num, self.top_k, self.top_p, self.temperature, self.max_seq_len,
                                                        self.tensor_para_size, self.layer_para_size, self.layer_para_batch_size, 
                                                        True, self.max_batch_size, 1.0, *self.weights.w)


    def forward(self, start_ids, start_lengths, attn_mask, batch_first=True):
        batch_size = start_ids.size(0)
        assert batch_size <= self.max_batch_size, "batch_size must not exceed max_batch_size."
        assert batch_size >= self.layer_para_batch_size, "batch_size must be equal to or larger than layer_para_batch_size."
        assert batch_size % self.layer_para_batch_size == 0, "batch_size must be a multiple of layer_para_batch_size."

        input_len = min(start_lengths)
        assert input_len > 0, "input_len must be larger than zero. For an unconditional case, use start_id as the first token."
        assert input_len + self.output_len <= self.max_seq_len, "input_len + output_len must not exceed max_seq_len."

        # Inputs to device
        start_ids = start_ids.cuda(self.device)
        start_lengths = start_lengths.cuda(self.device)
        attn_mask = attn_mask.cuda(self.device)

        assert self.model is not None, "The model must be copied to the device(s) through cuda()."

        output_ids, = self.model.forward(start_ids, start_lengths, attn_mask, self.output_len)
        if batch_first:
            output_ids = output_ids.T

        if self.rank == 0:
            return output_ids
