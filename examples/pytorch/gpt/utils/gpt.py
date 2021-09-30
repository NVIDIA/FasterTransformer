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
    def __init__(self, head_num, size_per_head, layer_num, vocab_size, max_seq_len, tensor_para_size, pipeline_para_size):
        assert(head_num % tensor_para_size == 0)

        self.head_num = head_num
        self.size_per_head = size_per_head
        self.layer_num = layer_num
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size
        self.layers_per_device = layer_num // pipeline_para_size

        local_head_num = head_num // tensor_para_size
        global_head_num = head_num
        local_hidden_units = local_head_num * size_per_head
        global_hidden_units = global_head_num * size_per_head
        local_inter_size = local_hidden_units * 4

        self.local_head_num = local_head_num
        self.global_head_num = global_head_num
        self.local_hidden_units = local_hidden_units
        self.global_hidden_units = global_hidden_units
        self.local_inter_size = local_inter_size

        self.w = []
        # Transformer blocks
        self.w.extend([torch.zeros(global_hidden_units)] * layer_num)   # self_layernorm_gamma
        self.w.extend([torch.zeros(global_hidden_units)] * layer_num)   # self_layernorm_beta
        self.w.extend([torch.zeros(global_hidden_units, local_hidden_units * 3)] * layer_num)   # self_kernel
        self.w.extend([torch.zeros(local_hidden_units * 3)] * layer_num)   # self_bias
        self.w.extend([torch.zeros(local_hidden_units, global_hidden_units)] * layer_num)   # self_output_kernel
        self.w.extend([torch.zeros(global_hidden_units)] * layer_num)   # self_output_bias
        self.w.extend([torch.zeros(global_hidden_units)] * layer_num)   # ffn_layernorm_gamma
        self.w.extend([torch.zeros(global_hidden_units)] * layer_num)   # ffn_layernorm_beta
        self.w.extend([torch.zeros(global_hidden_units, local_inter_size)] * layer_num)   # ffn_kernel1
        self.w.extend([torch.zeros(local_inter_size)] * layer_num)   # ffn_bias1
        self.w.extend([torch.zeros(local_inter_size, global_hidden_units)] * layer_num)   # ffn_kernel2
        self.w.extend([torch.zeros(global_hidden_units)] * layer_num)   # ffn_bias2
        # After Transformer blocks
        self.w.append(torch.zeros(global_hidden_units))   # layernorm_gamma
        self.w.append(torch.zeros(global_hidden_units))   # layernorm_beta
        self.w.append(torch.zeros(max_seq_len, global_hidden_units))   # position_encoding_table
        self.w.append(torch.zeros(vocab_size, global_hidden_units))   # embedding_table
        self.w.append(torch.zeros(vocab_size, global_hidden_units))   # embedding_kernel

        # Initialization
        self._map(lambda w: torch.nn.init.normal_(w, mean=0., std=1.))

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

    def load(self, ckpt_path, tensor_para_rank, pipeline_para_rank):
        if not os.path.exists(ckpt_path):
            return False
        w = []

        # Load
        def is_load(i): return i >= self.layers_per_device * \
            pipeline_para_rank and i < self.layers_per_device * (pipeline_para_rank + 1)
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.input_layernorm.weight.bin".format(i),
                                               dtype=np.single)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.input_layernorm.bias.bin".format(i),
                                               dtype=np.single)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.attention.query_key_value.weight.{}.bin".format(i,
                                                                                                                             tensor_para_rank), dtype=np.single)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.attention.query_key_value.bias.{}.bin".format(i,
                                                                                                                           tensor_para_rank), dtype=np.single)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.attention.dense.weight.{}.bin".format(i,
                                                                                                                   tensor_para_rank), dtype=np.single)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.attention.dense.bias.bin".format(i),
                                               dtype=np.single)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.post_attention_layernorm.weight.bin".format(i),
                                               dtype=np.single)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.post_attention_layernorm.bias.bin".format(i),
                                               dtype=np.single)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.mlp.dense_h_to_4h.weight.{}.bin".format(i,
                                                                                                                     tensor_para_rank), dtype=np.single)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.mlp.dense_h_to_4h.bias.{}.bin".format(i,
                                                                                                                   tensor_para_rank), dtype=np.single)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.mlp.dense_4h_to_h.weight.{}.bin".format(i,
                                                                                                                     tensor_para_rank), dtype=np.single)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.mlp.dense_4h_to_h.bias.bin".format(i),
                                               dtype=np.single)) if is_load(i) else torch.empty(0) for i in range(self.layer_num)])

        w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.final_layernorm.weight.bin", dtype=np.single)))
        w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.final_layernorm.bias.bin", dtype=np.single)))

        wpe = torch.from_numpy(np.fromfile(ckpt_path + "/model.wpe.bin", dtype=np.single)
                               ).reshape(-1, self.global_hidden_units)
        assert self.max_seq_len <= wpe.size(
            0), "max_seq_len must not exceed the value of maximum sequence length during traning."
        wpe = wpe[:self.max_seq_len]  # excludes weights not to really use.
        w.append(wpe)
        w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.wte.bin", dtype=np.single)))
        w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.wte.bin", dtype=np.single)))

        # Reshape
        try:
            for i in range(len(w)):
                if w[i].nelement() > 0:
                    self.w[i] = w[i].reshape(self.w[i].shape)

        except RuntimeError:
            raise RuntimeError(
                "head_num, size_per_head, vocab_size, and max_seq_len must be the same as the ones during training.")

        return True


class GPT(nn.Module):
    def __init__(self, head_num, size_per_head, vocab_size, start_id, end_id, layer_num, top_k, top_p, random_seed, temperature,
                 output_len, max_seq_len, tensor_para_size, pipeline_para_size, max_batch_size, repetition_penalty, lib_path):
        super().__init__()
        self.beam_width = 1
        self.head_num = head_num
        self.size_per_head = size_per_head
        self.vocab_size = vocab_size
        self.start_id = start_id
        self.end_id = end_id
        self.layer_num = layer_num
        self.beam_search_diversity_rate = 0.0
        self.top_k = top_k
        self.top_p = top_p
        self.random_seed = random_seed
        self.temperature = temperature
        self.output_len = output_len
        self.max_seq_len = max_seq_len
        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size
        self.max_batch_size = max_batch_size
        self.repetition_penalty = repetition_penalty
        self.build_model = False

        assert torch.cuda.is_available(), "CUDA is required for this model."

        assert head_num % tensor_para_size == 0, "head_num must be a multiple of tensor_para_size."
        assert layer_num % pipeline_para_size == 0, "layer_num must be a multiple of pipeline_para_size."

        # Prepare weights
        self.weights = GPTWeights(head_num, size_per_head, layer_num, vocab_size,
                                  max_seq_len, tensor_para_size, pipeline_para_size)

        # Prepare for tensor/pipeline parallel
        try:
            dist.init_process_group(backend='mpi')
        except:
            print("[INFO] WARNING: Have initalize the process group")
        self.rank = dist.get_rank()
        self.device_count = torch.cuda.device_count()
        self.device = self.rank % self.device_count
        torch.cuda.set_device(self.device)

        world_size = dist.get_world_size()
        assert world_size == tensor_para_size * pipeline_para_size, "tensor_para_size * pipeline_para_size must be equal to world_size."

        self.tensor_para_rank = self.rank % self.tensor_para_size
        self.pipeline_para_rank = self.rank // self.tensor_para_size

        # Load the C++ model into Pytorch model.
        torch.classes.load_library(os.path.abspath(lib_path))

        # Create and copy model to the device.
        self.cuda()

    def load(self, ckpt_path):
        is_load = self.weights.load(ckpt_path, tensor_para_rank=self.tensor_para_rank,
                                    pipeline_para_rank=self.pipeline_para_rank)
        self.cuda()
        return is_load

    def half(self):
        self.weights._map(lambda w: w.half())
        self.cuda()

    def cuda(self):
        self.weights._map(lambda w: w.cuda(self.device))
        if self.build_model == True:
            del self.model
            self.build_model = False
        self.model = torch.classes.FasterTransformer.GptOp(self.max_batch_size,
                                                           self.max_seq_len, self.beam_width,
                                                           self.head_num, self.size_per_head, 4 * self.head_num * self.size_per_head,
                                                           self.layer_num, self.vocab_size, self.start_id, self.end_id,
                                                           self.beam_search_diversity_rate, self.top_k, self.top_p, self.random_seed, self.temperature, 1.0,
                                                           self.repetition_penalty, self.weights.w)
        self.build_model = True
    def forward(self, start_ids, start_lengths, batch_first=True):
        if self.build_model == False:
            self.cuda()
        batch_size = start_ids.size(0)
        assert batch_size <= self.max_batch_size, "batch_size must not exceed max_batch_size."

        input_len = start_ids.size(1)
        assert input_len > 0, "input len must be larger than zero. For an unconditional case, use start_id as the first token."
        assert input_len + self.output_len <= self.max_seq_len, "input_len + output_len must not exceed max_seq_len."

        # Inputs to device
        start_ids = start_ids.cuda(self.device)
        start_lengths = start_lengths.cuda(self.device)

        output_ids, output_lengths = self.model.forward(start_ids, start_lengths, self.output_len)
        output_ids = output_ids.reshape((batch_size * self.beam_width, -1))
        if self.rank == 0:
            return output_ids

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor
