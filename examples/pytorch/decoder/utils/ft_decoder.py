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
import torch.nn as nn

USE_CACHE_BATCH_MAJOR_ATTENTION = True

def get_op_cache_config(size_per_head, is_fp16):
    x = 8 if is_fp16 else 4
    use_batch_major_op_cache = True if USE_CACHE_BATCH_MAJOR_ATTENTION == True and \
                                       size_per_head % x == 0 \
                                    else False
    x = x if use_batch_major_op_cache else 1
    return use_batch_major_op_cache, x

class FtDecoderWeights(object):
    def __init__(self, layer_num, hidden_dim, onmtcheckpoint, max_step_for_pe=2048):
        self.max_step_for_pe = max_step_for_pe
        self.hidden_dim = hidden_dim
        self.w = []
        prefix = 'decoder.transformer_layers.'
        self.w.append(torch.stack(
            [onmtcheckpoint['model'][prefix + str(i) + '.layer_norm_1.weight'] for i in range(layer_num)],
            0).contiguous())
        self.w.append(torch.stack(
            [onmtcheckpoint['model'][prefix + str(i) + '.layer_norm_1.bias'] for i in range(layer_num)],
            0).contiguous())
        self.w.append(torch.stack(
            [torch.stack([onmtcheckpoint['model'][prefix + str(i) + '.self_attn.linear_query.weight'].transpose(-1, -2),
                        onmtcheckpoint['model'][prefix + str(i) + '.self_attn.linear_keys.weight'].transpose(-1, -2),
                        onmtcheckpoint['model'][prefix + str(i) + '.self_attn.linear_values.weight'].transpose(-1, -2)], -2)
                        for i in range(layer_num)], 0).contiguous())
        self.w.append(torch.stack(
            [torch.stack([onmtcheckpoint['model'][prefix + str(i) + '.self_attn.linear_query.bias'],
                        onmtcheckpoint['model'][prefix + str(i) + '.self_attn.linear_keys.bias'],
                        onmtcheckpoint['model'][prefix + str(i) + '.self_attn.linear_values.bias']], -2) for i in range(layer_num)],
            0).contiguous())
        self.w.append(torch.stack(
            [onmtcheckpoint['model'][prefix + str(i) + '.self_attn.final_linear.weight'].transpose(-1, -2) for i in range(layer_num)],
            0).contiguous())
        self.w.append(torch.stack(
            [onmtcheckpoint['model'][prefix + str(i) + '.self_attn.final_linear.bias'] for i in range(layer_num)],
            0).contiguous())
        self.w.append(torch.stack(
            [onmtcheckpoint['model'][prefix + str(i) + '.layer_norm_2.weight'] for i in range(layer_num)],
            0).contiguous())
        self.w.append(torch.stack(
            [onmtcheckpoint['model'][prefix + str(i) + '.layer_norm_2.bias'] for i in range(layer_num)],
            0).contiguous())
        self.w.append(torch.stack(
            [onmtcheckpoint['model'][prefix + str(i) + '.context_attn.linear_query.weight'].transpose(-1, -2) for i in range(layer_num)],
            0).contiguous())
        self.w.append(torch.stack(
            [onmtcheckpoint['model'][prefix + str(i) + '.context_attn.linear_keys.weight'].transpose(-1, -2) for i in range(layer_num)],
            0).contiguous())
        self.w.append(torch.stack(
            [onmtcheckpoint['model'][prefix + str(i) + '.context_attn.linear_values.weight'].transpose(-1, -2) for i in range(layer_num)],
            0).contiguous())
        self.w.append(torch.stack(
            [onmtcheckpoint['model'][prefix + str(i) + '.context_attn.linear_query.bias'] for i in range(layer_num)],
            0).contiguous())
        self.w.append(torch.stack(
            [onmtcheckpoint['model'][prefix + str(i) + '.context_attn.linear_keys.bias'] for i in range(layer_num)],
            0).contiguous())
        self.w.append(torch.stack(
            [onmtcheckpoint['model'][prefix + str(i) + '.context_attn.linear_values.bias'] for i in range(layer_num)],
            0).contiguous())
        self.w.append(torch.stack(
            [onmtcheckpoint['model'][prefix + str(i) + '.context_attn.final_linear.weight'].transpose(-1, -2) for i in range(layer_num)],
            0).contiguous())
        self.w.append(torch.stack(
            [onmtcheckpoint['model'][prefix + str(i) + '.context_attn.final_linear.bias'] for i in range(layer_num)],
            0).contiguous())
        self.w.append(torch.stack(
            [onmtcheckpoint['model'][prefix + str(i) + '.feed_forward.layer_norm.weight'] for i in range(layer_num)],
            0).contiguous())
        self.w.append(torch.stack(
            [onmtcheckpoint['model'][prefix + str(i) + '.feed_forward.layer_norm.bias'] for i in range(layer_num)],
            0).contiguous())
        self.w.append(torch.stack(
            [onmtcheckpoint['model'][prefix + str(i) + '.feed_forward.w_1.weight'].transpose(-1, -2) for i in range(layer_num)],
            0).contiguous())
        self.w.append(torch.stack(
            [onmtcheckpoint['model'][prefix + str(i) + '.feed_forward.w_1.bias'] for i in range(layer_num)],
            0).contiguous())
        self.w.append(torch.stack(
            [onmtcheckpoint['model'][prefix + str(i) + '.feed_forward.w_2.weight'].transpose(-1, -2) for i in range(layer_num)],
            0).contiguous())
        self.w.append(torch.stack(
            [onmtcheckpoint['model'][prefix + str(i) + '.feed_forward.w_2.bias'] for i in range(layer_num)],
            0).contiguous())
        
    def to_cuda(self):
        for i in range(len(self.w)):
            self.w[i] = self.w[i].cuda()

    def to_half(self):
        for i in range(len(self.w)):
            self.w[i] = self.w[i].half()

class FTDecoder(nn.Module):
    def __init__(self, head_num, head_size, mem_hidden_dim, layer_num, weights, args):
        super().__init__()
        self.args = args
        self.is_fp16 = True if self.args.data_type == 'fp16' else False
        self.layer_num = layer_num
        self.use_batch_major_op_cache, self.op_cache_dim_x = get_op_cache_config(head_size, self.is_fp16)
        
        torch.classes.load_library(args.decoder_ths_path)
        try:
            self.dec_layer = torch.classes.FasterTransformer.Decoder(*weights.w, head_num, head_size, head_num * head_size * 4, layer_num, mem_hidden_dim)
        except:
            # legacy ths for 20.03 image
            self.dec_layer = torch.classes.FasterTransformerDecoder(*weights.w, head_num, head_size, head_num * head_size * 4, layer_num, mem_hidden_dim)
    
    def forward(self, inputs, memory, memory_seq_lens, self_cache, mem_cache, sequence_lengths, step):
        dtype = torch.half if self.is_fp16 else torch.float32
        inputs_shape = inputs.shape
        inputs = inputs.reshape([-1, inputs.shape[-1]])
        output, self_key_cache, self_val_cache, mem_key_cache, mem_val_cache = \
                self.dec_layer.forward(step, inputs, memory, memory_seq_lens, sequence_lengths,
                                       self_cache[0], self_cache[1], mem_cache[0], mem_cache[1])
        output = output.reshape(inputs_shape)

        return output, [self_key_cache, self_val_cache], [mem_key_cache, mem_val_cache]


class ArgHelper(object):
    def __init__(self, model_type=None, data_type=None, ths_path=None):
        self.model_type = model_type
        self.data_type = data_type
        self.ths_path = ths_path

