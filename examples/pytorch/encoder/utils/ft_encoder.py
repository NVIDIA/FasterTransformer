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
import torch
import torch.nn as nn

from onmt.encoders.transformer import TransformerEncoderLayer

class EncoderWeights(object):
    def __init__(self, layer_num, hidden_dim, weights=None):
        """weights need be a state_dict of bert model"""
        self.layer_num = layer_num
        self.hidden_dim = hidden_dim
        self.weights = {}
        if weights is None:
            self.weights['encoder.layer_norm.weight'] = torch.zeros(hidden_dim)
            self.weights['encoder.layer_norm.bias'] = torch.zeros(hidden_dim)
            # self.weights['encoder.embeddings.make_embedding.emb_luts.0.weight']
            # self.weights['encoder.embeddings.make_embedding.pe.pe']
            for i in range(layer_num):
                pre = 'encoder.transformer.' + str(i) + '.'
                self.weights[pre + 'layer_norm.weight'] = torch.zeros(hidden_dim)
                self.weights[pre + 'layer_norm.bias'] = torch.zeros(hidden_dim)
                self.weights[pre + 'self_attn.linear_query.weight'] = torch.zeros(hidden_dim, hidden_dim)
                self.weights[pre + 'self_attn.linear_query.bias'] = torch.zeros(hidden_dim)
                self.weights[pre + 'self_attn.linear_keys.weight'] = torch.zeros(hidden_dim, hidden_dim)
                self.weights[pre + 'self_attn.linear_keys.bias'] = torch.zeros(hidden_dim)
                self.weights[pre + 'self_attn.linear_values.weight'] = torch.zeros(hidden_dim, hidden_dim)
                self.weights[pre + 'self_attn.linear_values.bias'] = torch.zeros(hidden_dim)
                self.weights[pre + 'self_attn.final_linear.weight'] = torch.zeros(hidden_dim, hidden_dim)
                self.weights[pre + 'self_attn.final_linear.bias'] = torch.zeros(hidden_dim)
                self.weights[pre + 'feed_forward.layer_norm.weight'] = torch.zeros(hidden_dim)
                self.weights[pre + 'feed_forward.layer_norm.bias'] = torch.zeros(hidden_dim)
                self.weights[pre + 'feed_forward.w_1.weight'] = torch.zeros(4 * hidden_dim, hidden_dim)
                self.weights[pre + 'feed_forward.w_1.bias'] = torch.zeros(4 * hidden_dim)
                self.weights[pre + 'feed_forward.w_2.weight'] = torch.zeros(hidden_dim, 4 * hidden_dim)
                self.weights[pre + 'feed_forward.w_2.bias'] = torch.zeros(hidden_dim)
            for k, v in self.weights.items():
                self.weights[k] = torch.nn.init.uniform_(v, -1, 1)
        else:
            self.weights = weights

    def listed_weights(self):
        ret = []
        ret.append(torch.stack([self.weights['encoder.transformer.' + str(layer_idx) + '.' + 'layer_norm.weight'] for layer_idx in range(self.layer_num)], 0).contiguous())
        ret.append(torch.stack([self.weights['encoder.transformer.' + str(layer_idx) + '.' + 'layer_norm.bias'] for layer_idx in range(self.layer_num)], 0).contiguous())
        ret.append(torch.stack([self.weights['encoder.transformer.' + str(layer_idx) + '.' + 'self_attn.linear_query.weight'].transpose(-1, -2) for layer_idx in range(self.layer_num)], 0).contiguous())
        ret.append(torch.stack([self.weights['encoder.transformer.' + str(layer_idx) + '.' + 'self_attn.linear_query.bias'] for layer_idx in range(self.layer_num)], 0).contiguous())
        ret.append(torch.stack([self.weights['encoder.transformer.' + str(layer_idx) + '.' + 'self_attn.linear_keys.weight'].transpose(-1, -2) for layer_idx in range(self.layer_num)], 0).contiguous())
        ret.append(torch.stack([self.weights['encoder.transformer.' + str(layer_idx) + '.' + 'self_attn.linear_keys.bias'] for layer_idx in range(self.layer_num)], 0).contiguous())
        ret.append(torch.stack([self.weights['encoder.transformer.' + str(layer_idx) + '.' + 'self_attn.linear_values.weight'].transpose(-1, -2) for layer_idx in range(self.layer_num)], 0).contiguous())
        ret.append(torch.stack([self.weights['encoder.transformer.' + str(layer_idx) + '.' + 'self_attn.linear_values.bias'] for layer_idx in range(self.layer_num)], 0).contiguous())
        ret.append(torch.stack([self.weights['encoder.transformer.' + str(layer_idx) + '.' + 'self_attn.final_linear.weight'].transpose(-1, -2) for layer_idx in range(self.layer_num)], 0).contiguous())
        ret.append(torch.stack([self.weights['encoder.transformer.' + str(layer_idx) + '.' + 'self_attn.final_linear.bias'] for layer_idx in range(self.layer_num)], 0).contiguous())
        ret.append(torch.stack([self.weights['encoder.transformer.' + str(layer_idx) + '.' + 'feed_forward.layer_norm.weight'] for layer_idx in range(self.layer_num)], 0).contiguous())
        ret.append(torch.stack([self.weights['encoder.transformer.' + str(layer_idx) + '.' + 'feed_forward.layer_norm.bias'] for layer_idx in range(self.layer_num)], 0).contiguous())
        ret.append(torch.stack([self.weights['encoder.transformer.' + str(layer_idx) + '.' + 'feed_forward.w_1.weight'].transpose(-1, -2) for layer_idx in range(self.layer_num)], 0).contiguous())
        ret.append(torch.stack([self.weights['encoder.transformer.' + str(layer_idx) + '.' + 'feed_forward.w_1.bias'] for layer_idx in range(self.layer_num)], 0).contiguous())
        ret.append(torch.stack([self.weights['encoder.transformer.' + str(layer_idx) + '.' + 'feed_forward.w_2.weight'].transpose(-1, -2) for layer_idx in range(self.layer_num)], 0).contiguous())
        ret.append(torch.stack([self.weights['encoder.transformer.' + str(layer_idx) + '.' + 'feed_forward.w_2.bias'] for layer_idx in range(self.layer_num)], 0).contiguous())
        ret.append(self.weights['encoder.layer_norm.weight'].contiguous())
        ret.append(self.weights['encoder.layer_norm.bias'].contiguous())
        return ret

    def to_cuda(self):
        for k, v in self.weights.items():
            self.weights[k] = v.cuda()

    def to_half(self):
        for k, v in self.weights.items():
            self.weights[k] = v.half()


class CustomEncoder(torch.nn.Module):
    def __init__(self, layer_num, head_num, head_size, weights,
                 remove_padding=False, allow_gemm_test=False, path='./lib/libth_encoder.so', embedding=None):
        super().__init__()
        self.layer_num = layer_num
        self.remove_padding = remove_padding
        self.embedding = embedding
        torch.classes.load_library(path)

        weights_ = weights.listed_weights()
        assert len(weights_) == 18
        try:
            self.encoders = torch.classes.FasterTransformer.Encoder(
                    *weights_,
                    head_num, head_size, 4 * head_num * head_size, remove_padding, layer_num, allow_gemm_test, False, 1.0)
        except:
            # legacy ths for 20.03 image
            self.encoders = torch.classes.FasterTransformerEncoder(
                    *weights_,
                    head_num, head_size, 4 * head_num * head_size, remove_padding, layer_num, allow_gemm_test, False, 1.0)

    def forward(self, inputs, lengths):
        if self.embedding is not None:
            emb = self.embedding(inputs)
            inputs = emb.transpose(0, 1).contiguous()
        hidden_states = self.encoders.forward(inputs, lengths)
        if self.embedding is not None:
            return emb, hidden_states.transpose(0, 1).contiguous(), lengths
        else:
            return hidden_states


class ONMTEncoder(torch.nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff, weights):
        super(ONMTEncoder, self).__init__()
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, 0.0, 0.0,
                max_relative_positions=0)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm.weight.data = weights.weights['encoder.layer_norm.weight']
        self.layer_norm.bias.data = weights.weights['encoder.layer_norm.bias']
        w = {}
        for k, v in weights.weights.items():
            if k.startswith('encoder.transformer'):
                w[k[20:]] = v
        self.transformer.load_state_dict(w)

    def forward(self, src, mask):
        out = src
        for layer in self.transformer:
            out = layer(out, mask)
        out = self.layer_norm(out)
        return out
