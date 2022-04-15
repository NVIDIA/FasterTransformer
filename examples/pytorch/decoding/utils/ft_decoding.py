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

import sys
import os
import math
import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx

from onmt.utils.misc import tile

class FtDecodingWeights(object):
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
        self.w.append(onmtcheckpoint['model']['decoder.layer_norm.weight'])
        self.w.append(onmtcheckpoint['model']['decoder.layer_norm.bias'])
        self.w.append(onmtcheckpoint['model']['decoder.embeddings.make_embedding.emb_luts.0.weight'])
        self.w.append(self._get_position_encoding()) # pe_encoding
        self.w.append(onmtcheckpoint['generator']['0.weight'].transpose(-1, -2).contiguous())
        self.w.append(onmtcheckpoint['generator']['0.bias'])
    
    def to_cuda(self):
        for i in range(len(self.w)):
            self.w[i] = self.w[i].cuda()

    def to_half(self):
        for i in range(len(self.w)):
            self.w[i] = self.w[i].half()
            
    def _get_position_encoding(self):
        pe = torch.zeros(self.max_step_for_pe, self.hidden_dim)
        position = torch.arange(0, self.max_step_for_pe).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, self.hidden_dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / self.hidden_dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        return pe.cuda().contiguous()

class CustomDecoding(nn.Module):
    def __init__(self, head_num, head_size,
                inter_size, mem_hidden_dim, layer_num, vocab_size, start_id, end_id,
                beam_search_diversity_rate, top_k, top_p, temperature,
                len_penalty, repetition_penalty, weights, args=None):
        super().__init__()
        self.end_id = end_id
        self.args = args
        torch.classes.load_library(os.path.abspath(args.decoding_ths_path))
        try:
            self.decoding = torch.classes.FasterTransformer.Decoding(head_num, head_size,
                                                                    inter_size, mem_hidden_dim, layer_num, vocab_size, start_id, end_id,
                                                                    beam_search_diversity_rate, top_k, top_p, temperature,
                                                                    len_penalty, repetition_penalty, *weights.w)
        except:
            # legacy ths for 20.03 image
            self.decoding = torch.classes.FasterTransformerDecoding(head_num, head_size,
                                                                    inter_size, mem_hidden_dim, layer_num, vocab_size, start_id, end_id,
                                                                    beam_search_diversity_rate, top_k, top_p, temperature,
                                                                    len_penalty, repetition_penalty, *weights.w)
        self.is_clean_cache = False
    def forward(self, batch_size, beam_size, seq_len, memory, memory_seq_lens):
        if self.is_clean_cache == False:
            torch.cuda.empty_cache()
            self.is_clean_cache = True

        extended_memory = tile(memory, beam_size)
        extended_memory_seq_lens = tile(memory_seq_lens, beam_size)
        output_ids, parent_ids, out_seq_lens = self.decoding.forward(beam_size, seq_len, extended_memory, extended_memory_seq_lens)
        output_ids = output_ids.reshape([seq_len, memory.size(0), beam_size])
        output_ids = output_ids.permute(1, 2, 0)
        return output_ids, out_seq_lens

class ArgHelper(object):
    def __init__(self, model_type=None, data_type=None, ths_path=None):
        self.model_type = model_type
        self.data_type = data_type
        self.ths_path = ths_path

