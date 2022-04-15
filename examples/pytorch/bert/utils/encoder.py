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

from transformers import BertConfig
from transformers.modeling_bert import BertEncoder
from .checkpoint_quantization import checkpoint_quantization

class EncoderWeights(object):
    def __init__(self, layer_num, hidden_dim, weights=None, sparse=False):
        """weights need be a state_dict of bert model"""
        self.layer_num = layer_num
        self.int8 = False
        self.hidden_dim = hidden_dim
        self.weights = {}
        if weights is None:
            self._generated_weights = True
            for i in range(layer_num):
                pre = 'bert.encoder.layer.' + str(i) + '.'
                self.weights[pre + 'attention.self.query.weight'] = torch.zeros(hidden_dim, hidden_dim)
                self.weights[pre + 'attention.self.query.bias'] = torch.zeros(hidden_dim)
                self.weights[pre + 'attention.self.key.weight'] = torch.zeros(hidden_dim, hidden_dim)
                self.weights[pre + 'attention.self.key.bias'] = torch.zeros(hidden_dim)
                self.weights[pre + 'attention.self.value.weight'] = torch.zeros(hidden_dim, hidden_dim)
                self.weights[pre + 'attention.self.value.bias'] = torch.zeros(hidden_dim)
                self.weights[pre + 'attention.output.dense.weight'] = torch.zeros(hidden_dim, hidden_dim)
                self.weights[pre + 'attention.output.dense.bias'] = torch.zeros(hidden_dim)
                self.weights[pre + 'attention.output.LayerNorm.weight'] = torch.zeros(hidden_dim)
                self.weights[pre + 'attention.output.LayerNorm.bias'] = torch.zeros(hidden_dim)
                self.weights[pre + 'intermediate.dense.weight'] = torch.zeros(4 * hidden_dim, hidden_dim)
                self.weights[pre + 'intermediate.dense.bias'] = torch.zeros(4 * hidden_dim)
                self.weights[pre + 'output.dense.weight'] = torch.zeros(hidden_dim, 4 * hidden_dim)
                self.weights[pre + 'output.dense.bias'] = torch.zeros(hidden_dim)
                self.weights[pre + 'output.LayerNorm.weight'] = torch.zeros(hidden_dim)
                self.weights[pre + 'output.LayerNorm.bias'] = torch.zeros(hidden_dim)
            for k, v in self.weights.items():
                if not k.endswith('_amax'):
                    self.weights[k] = torch.nn.init.uniform_(v, -1, 1)
            if sparse:
                for k, v in self.weights.items():
                    if 'query.weight' in k or 'key.weight' in k or 'value.weight' in k or 'dense.weight' in k:
                        v_shape = v.shape
                        v = v.view(-1, 4)
                        _, indices = torch.topk(torch.abs(v), 2, dim=-1, largest=False)
                        v.scatter_(1, indices, 0)
                        self.weights[k] = v.view(v_shape)
        else:
            self._generated_weights = False
            for k, v in weights.items():
                ks = k.split('.')
                if ks[-2] == 'LayerNorm':
                    if ks[-1] == 'gamma':
                        ks[-1] = 'weight'
                    elif ks[-1] == 'beta':
                        ks[-1] = 'bias'
                self.weights['.'.join(ks)] = v

    def listed_weights(self):
        ret = []
        if not self.int8:
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'attention.self.query.weight'].transpose(-1, -2) for layer_idx in range(self.layer_num)], 0).contiguous())       # 0
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'attention.self.query.bias'] for layer_idx in range(self.layer_num)], 0).contiguous())
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'attention.self.key.weight'].transpose(-1, -2) for layer_idx in range(self.layer_num)], 0).contiguous())         # 2
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'attention.self.key.bias'] for layer_idx in range(self.layer_num)], 0).contiguous())
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'attention.self.value.weight'].transpose(-1, -2) for layer_idx in range(self.layer_num)], 0).contiguous())       # 4
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'attention.self.value.bias'] for layer_idx in range(self.layer_num)], 0).contiguous())
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'attention.output.dense.weight'].transpose(-1, -2) for layer_idx in range(self.layer_num)], 0).contiguous())     # 6
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'attention.output.dense.bias'] for layer_idx in range(self.layer_num)], 0).contiguous())
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'attention.output.LayerNorm.weight'] for layer_idx in range(self.layer_num)], 0).contiguous())
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'attention.output.LayerNorm.bias'] for layer_idx in range(self.layer_num)], 0).contiguous())
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'intermediate.dense.weight'].transpose(-1, -2) for layer_idx in range(self.layer_num)], 0).contiguous())         # 10
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'intermediate.dense.bias'] for layer_idx in range(self.layer_num)], 0).contiguous())
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'output.dense.weight'].transpose(-1, -2) for layer_idx in range(self.layer_num)], 0).contiguous())               # 12
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'output.dense.bias'] for layer_idx in range(self.layer_num)], 0).contiguous())
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'output.LayerNorm.weight'] for layer_idx in range(self.layer_num)], 0).contiguous())
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'output.LayerNorm.bias'] for layer_idx in range(self.layer_num)], 0).contiguous())
        else:
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'attention.self.query.weight'] for layer_idx in range(self.layer_num)], 0).contiguous())       # 0
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'attention.self.query.bias'] for layer_idx in range(self.layer_num)], 0).contiguous())
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'attention.self.key.weight'] for layer_idx in range(self.layer_num)], 0).contiguous())         # 2
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'attention.self.key.bias'] for layer_idx in range(self.layer_num)], 0).contiguous())
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'attention.self.value.weight'] for layer_idx in range(self.layer_num)], 0).contiguous())       # 4
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'attention.self.value.bias'] for layer_idx in range(self.layer_num)], 0).contiguous())
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'attention.output.dense.weight'] for layer_idx in range(self.layer_num)], 0).contiguous())     # 6
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'attention.output.dense.bias'] for layer_idx in range(self.layer_num)], 0).contiguous())
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'attention.output.LayerNorm.weight'] for layer_idx in range(self.layer_num)], 0).contiguous())
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'attention.output.LayerNorm.bias'] for layer_idx in range(self.layer_num)], 0).contiguous())
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'intermediate.dense.weight'] for layer_idx in range(self.layer_num)], 0).contiguous())         # 10
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'intermediate.dense.bias'] for layer_idx in range(self.layer_num)], 0).contiguous())
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'output.dense.weight'] for layer_idx in range(self.layer_num)], 0).contiguous())               # 12
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'output.dense.bias'] for layer_idx in range(self.layer_num)], 0).contiguous())
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'output.LayerNorm.weight'] for layer_idx in range(self.layer_num)], 0).contiguous())
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'output.LayerNorm.bias'] for layer_idx in range(self.layer_num)], 0).contiguous())
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'amaxList'] for layer_idx in range(self.layer_num)], 0).contiguous())
            ret.append(torch.stack([self.weights['bert.encoder.layer.' + str(layer_idx) + '.' + 'h_amaxList'] for layer_idx in range(self.layer_num)], 0).contiguous())
        return ret

    def to_cuda(self):
        if not self.int8:
            for k, v in self.weights.items():
                self.weights[k] = v.cuda()
        else:
            h_scale_list = {}
            for k, v in self.weights.items():
                if "amaxList" in k:
                    k_h = k.replace("amaxList", "h_amaxList")
                    h_scale_list[k_h] = v
                self.weights[k] = v.cuda() 
            for k, v in h_scale_list.items():
                self.weights[k] = v

    def to_half(self):
        if self.int8:
            raise RuntimeError("Cannot cast to half if the weights have been casted to int8.")
        for k, v in self.weights.items():
            self.weights[k] = v.half()

    def to_int8(self, sparse=False, ths_path='./lib/libth_bert.so'):
        if self._generated_weights:
            amax_tensor_1 = torch.Tensor(self.hidden_dim).fill_(127.)
            amax_tensor_2 = torch.Tensor(self.hidden_dim * 4).fill_(127.)
            for i in range(self.layer_num):
                pre = 'bert.encoder.layer.' + str(i) + '.'
                self.weights[pre + 'attention.self.query._input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.query._weight_quantizer._amax'] = amax_tensor_1
                self.weights[pre + 'attention.self.query._aftergemm_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.key._input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.key._weight_quantizer._amax'] = amax_tensor_1
                self.weights[pre + 'attention.self.key._aftergemm_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.value._input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.value._weight_quantizer._amax'] = amax_tensor_1
                self.weights[pre + 'attention.self.value._aftergemm_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.matmul_q_input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.matmul_k_input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.matmul_v_input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.matmul_a_input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.softmax_input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.output.dense._input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.output.dense._weight_quantizer._amax'] = amax_tensor_1
                self.weights[pre + 'attention.output.dense._aftergemm_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.output.add_local_input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.output.add_residual_input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'intermediate.dense._input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'intermediate.dense._weight_quantizer._amax'] = amax_tensor_2
                self.weights[pre + 'intermediate.dense._aftergemm_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'output.dense._input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'output.dense._weight_quantizer._amax'] = amax_tensor_1
                self.weights[pre + 'output.dense._aftergemm_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'output.add_local_input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'output.add_residual_input_quantizer._amax'] = torch.tensor(127.)
        if 'bert.encoder.layer.0.attention.self.query._input_quantizer._amax' not in self.weights:
            raise RuntimeError("There is no quantization node in the checkpoint, cannot be quantized to int8.")
        if self.int8:
            return
        self.int8 = True
        for k, v in self.weights.items():
            if k.endswith('bias') or k.endswith('LayerNorm.weight'):
                self.weights[k] = v.half()
            elif k.endswith('weight'):
                self.weights[k] = v.float().cuda()
            else:
                self.weights[k] = v.float().cpu()
        self.weights = checkpoint_quantization(self.weights, sparse, ths_path, verbose=False)


class CustomEncoder(torch.nn.Module):
    def __init__(self, layer_num, head_num, head_size, weights,
                 int8_mode=0, remove_padding=False,sparse=False,
                 path='./lib/libth_bert.so'):
        super().__init__()
        self.layer_num = layer_num
        self.remove_padding = remove_padding
        self.int8_mode = int8_mode
        torch.classes.load_library(path)

        weights_ = weights.listed_weights()
        if int8_mode == 0:
            assert len(weights_) == 16
            try:
                self.encoders = torch.classes.FasterTransformer.Bert(
                        *weights_,
                        head_num, head_size, 4 * head_num * head_size, remove_padding, layer_num, sparse, 1.0)
            except:
                # legacy ths for 20.03 image
                self.encoders = torch.classes.FasterTransformerBert(
                        *weights_,
                        head_num, head_size, 4 * head_num * head_size, remove_padding, layer_num, sparse, 1.0)
        else:
            assert len(weights_) == 18
            try:
                self.encoders = torch.classes.FasterTransformer.INT8Bert(
                        *weights_,
                        head_num, head_size, remove_padding, layer_num, int8_mode, sparse, 1.0)
            except:
                # legacy ths for 20.03 image
                self.encoders = torch.classes.FasterTransformerINT8Bert(
                        *weights_,
                        head_num, head_size, remove_padding, layer_num, int8_mode, sparse, 1.0)

    def forward(self, hidden_states, attention_mask, sequence_lengths):
        hidden_states = self.encoders.forward(hidden_states, sequence_lengths)
        return (hidden_states,)


class HuggingFaceEncoder(torch.nn.Module):
    def __init__(self, layer_num, head_num, head_size, weights=None):
        super().__init__()
        hidden_dim = head_num * head_size
        # TODO(bhsueh) The implementation of hidden_act='gelu' is different to FT's (and google BERT) implementation
        # FT's implementation is equivalent to hidden_act='gelu_new', but there are some issues for int8 sparse under gelu_new
        conf = BertConfig(hidden_size=hidden_dim, intermediate_size=4*hidden_dim, num_attention_heads=head_num, num_hidden_layers=layer_num, hidden_act='gelu')
        self.encoder = BertEncoder(conf)
        w = {}
        for k, v in weights.weights.items():
            if k.startswith('bert.encoder') and not k.endswith('_amax'):
                w[k[13:]] = weights.weights[k]
        self.encoder.load_state_dict(w)
        self.head_mask = [None] * layer_num

    def forward(self, hidden_states, attention_mask):
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        output = self.encoder(hidden_states, extended_attention_mask, self.head_mask)
        return output
