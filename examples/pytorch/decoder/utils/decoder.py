# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
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
from onmt.decoders.transformer import TransformerDecoderLayer

USE_CACHE_BATCH_MAJOR_ATTENTION = True

def get_op_cache_config(size_per_head, is_fp16):
    x = 8 if is_fp16 else 4
    use_batch_major_op_cache = True if USE_CACHE_BATCH_MAJOR_ATTENTION == True and \
                                       size_per_head % x == 0 \
                                    else False
    x = x if use_batch_major_op_cache else 1
    return use_batch_major_op_cache, x

def init_op_cache(layer_num, batch_size, beam_width, max_seq_len, \
                  decoding_max_seq_len, head_num, size_per_head, hidden_dim, is_fp16):
    use_batch_major_op_cache, x = get_op_cache_config(size_per_head, is_fp16)
    dtype = torch.half if is_fp16 else torch.float32
    if use_batch_major_op_cache == True:
        self_cache = [ torch.zeros(layer_num, batch_size * beam_width, head_num, size_per_head // x, 
                                   decoding_max_seq_len, x, dtype=dtype, device='cuda'),
                       torch.zeros(layer_num, batch_size * beam_width, head_num, 
                                   decoding_max_seq_len, size_per_head, dtype=dtype, device='cuda') ]
    else:
        self_cache = [ torch.zeros(layer_num, 0, batch_size * beam_width, hidden_dim, dtype=dtype, device='cuda'),
                       torch.zeros(layer_num, 0, batch_size * beam_width, hidden_dim, dtype=dtype, device='cuda') ]
    
    # always use old format for cross attention for now
    mem_cache = torch.zeros(2, layer_num, batch_size * beam_width, max_seq_len, hidden_dim, dtype=dtype, device='cuda')

    return self_cache, mem_cache

def init_onmt_cache(layer_num, memory_bank):
    cache = {}
    for i in range(layer_num):
        layer_cache = {"memory_keys": None, "memory_values": None}
        layer_cache["self_keys"] = None
        layer_cache["self_values"] = None
        cache[i] = layer_cache
    return cache

class ONMTDecoder(torch.nn.Module):
    def __init__(self, layer_num, head_num, head_size, weights):
        super().__init__()
        self.layer_num = layer_num
        self.hidden_dim = head_num * head_size
        self.decoders = torch.nn.ModuleList()
        for i in range(layer_num):
            self.decoders.append(TransformerDecoderLayer(self.hidden_dim, head_num, 4 * self.hidden_dim, 0, 0))
        for i in range(layer_num):
            prefix = 'decoder.transformer_layers.' + str(i)
            self.decoders[i].layer_norm_1.weight.data = weights.w['model'][prefix + '.layer_norm_1.weight']
            self.decoders[i].layer_norm_1.bias.data = weights.w['model'][prefix + '.layer_norm_1.bias']
            self.decoders[i].self_attn.linear_query.weight.data = weights.w['model'][prefix + '.self_attn.linear_query.weight']
            self.decoders[i].self_attn.linear_keys.weight.data = weights.w['model'][prefix + '.self_attn.linear_keys.weight']
            self.decoders[i].self_attn.linear_values.weight.data = weights.w['model'][prefix + '.self_attn.linear_values.weight']
            self.decoders[i].self_attn.linear_query.bias.data = weights.w['model'][prefix + '.self_attn.linear_query.bias']
            self.decoders[i].self_attn.linear_keys.bias.data = weights.w['model'][prefix + '.self_attn.linear_keys.bias']
            self.decoders[i].self_attn.linear_values.bias.data = weights.w['model'][prefix + '.self_attn.linear_values.bias']
            self.decoders[i].self_attn.final_linear.weight.data = weights.w['model'][prefix + '.self_attn.final_linear.weight']
            self.decoders[i].self_attn.final_linear.bias.data = weights.w['model'][prefix + '.self_attn.final_linear.bias']
            self.decoders[i].layer_norm_2.weight.data = weights.w['model'][prefix + '.layer_norm_2.weight']
            self.decoders[i].layer_norm_2.bias.data = weights.w['model'][prefix + '.layer_norm_2.bias']
            self.decoders[i].context_attn.linear_query.weight.data = weights.w['model'][prefix + '.context_attn.linear_query.weight']
            self.decoders[i].context_attn.linear_keys.weight.data = weights.w['model'][prefix + '.context_attn.linear_keys.weight']
            self.decoders[i].context_attn.linear_values.weight.data = weights.w['model'][prefix + '.context_attn.linear_values.weight']
            self.decoders[i].context_attn.linear_query.bias.data = weights.w['model'][prefix + '.context_attn.linear_query.bias']
            self.decoders[i].context_attn.linear_keys.bias.data = weights.w['model'][prefix + '.context_attn.linear_keys.bias']
            self.decoders[i].context_attn.linear_values.bias.data = weights.w['model'][prefix + '.context_attn.linear_values.bias']
            self.decoders[i].context_attn.final_linear.weight.data = weights.w['model'][prefix + '.context_attn.final_linear.weight']
            self.decoders[i].context_attn.final_linear.bias.data = weights.w['model'][prefix + '.context_attn.final_linear.bias']
            self.decoders[i].feed_forward.layer_norm.weight.data = weights.w['model'][prefix + '.feed_forward.layer_norm.weight']
            self.decoders[i].feed_forward.layer_norm.bias.data = weights.w['model'][prefix + '.feed_forward.layer_norm.bias']
            self.decoders[i].feed_forward.w_1.weight.data = weights.w['model'][prefix + '.feed_forward.w_1.weight']
            self.decoders[i].feed_forward.w_1.bias.data = weights.w['model'][prefix + '.feed_forward.w_1.bias']
            self.decoders[i].feed_forward.w_2.weight.data = weights.w['model'][prefix + '.feed_forward.w_2.weight']
            self.decoders[i].feed_forward.w_2.bias.data = weights.w['model'][prefix + '.feed_forward.w_2.bias']

    def forward(self, inputs, memory, src_pad_msk, cache, step):
        output = inputs
        for i in range(self.layer_num):
            output, _, _ = self.decoders[i](output, memory, src_pad_msk, None, cache[i], step)
        return output