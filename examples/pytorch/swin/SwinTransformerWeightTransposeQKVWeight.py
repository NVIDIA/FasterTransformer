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

import torch
import math

class SwinTransformerWeightTransposeQKVWeight(object):
    def __init__(self, layer_num, window_size, depths, num_heads, ths_path, weights=None):
        """weights need be a state_dict of swin transformer model"""
        block_weight_suffixes = ['attn.qkv.weight',
                                 'attn.qkv.bias',
                                 'attn.proj.weight',
                                 'attn.proj.bias',
                                 'mlp.fc1.weight',
                                 'mlp.fc1.bias',
                                 'mlp.fc2.weight',
                                 'mlp.fc2.bias',
                                 'norm1.weight',
                                 'norm1.bias',
                                 'norm2.weight',
                                 'norm2.bias']
        layer_weight_suffixes = ['downsample.norm.weight',
                                 'downsample.norm.bias',
                                 'downsample.reduction.weight']
        sw_weight_suffixes = ['patch_embed.proj.weight',
                              'patch_embed.proj.bias',
                              'patch_embed.norm.weight',
                              'patch_embed.norm.bias',
                              'norm.weight',
                              'norm.bias']
        self.layer_num = layer_num
        self.depths = depths
        self.weights = []
        torch.classes.load_library(ths_path)
        gen_relative_pos_bias = torch.ops.fastertransformer.gen_relative_pos_bias
        if weights is None:
            print("[ERROR][SwinTransformerWeights::__init__] weights should not be empty!")
            exit(-1)
        else:
            self._generated_weights = False
            #loop over layers
            for layer_idx in range(layer_num):
                ##loop over blocks
                for block_idx in range(depths[layer_idx]):
                    ###block_weight_suffixes
                    for block_weight_suffix in block_weight_suffixes:
                        weight_name = 'layers.{}.blocks.{}.{}'.format(layer_idx, block_idx, block_weight_suffix)
                        if weight_name in weights:
                            #transpose qkv weight [3*head*size, k] --> [k, head*3*size]
                            if "attn.qkv.weight" in weight_name:
                                shape = weights[weight_name].shape
                                #in case we flatten this weight
                                if len(shape) == 1:
                                    dim = int(math.sqrt(shape[0]/3))
                                    weights[weight_name] = weights[weight_name].reshape([3*dim, dim])
                                    shape = weights[weight_name].shape
                                weights[weight_name] = weights[weight_name].reshape([3, num_heads[layer_idx], int(shape[0]/3/num_heads[layer_idx]), -1]).permute(3, 1, 0, 2).reshape(shape[1], -1)
                            #transpose qkv bias
                            if "attn.qkv.bias" in weight_name:
                                shape = weights[weight_name].shape
                                weights[weight_name] = weights[weight_name].reshape([3, num_heads[layer_idx], int(shape[0]/3/num_heads[layer_idx])]).permute(1, 0, 2).reshape(-1)
                            self.weights.append(weights[weight_name])
                        else:
                            print("[ERROR][SwinTransformerWeights::__init__] missing weight {}.".format(weight_name))
                            exit(-1)
                    ###get relative position bias
                    index_name = 'layers.{}.blocks.{}.attn.relative_position_index'.format(layer_idx, block_idx)
                    table_name = 'layers.{}.blocks.{}.attn.relative_position_bias_table'.format(layer_idx, block_idx)
                    if index_name in weights and table_name in weights:
                        relative_position_bias = gen_relative_pos_bias(weights[table_name], weights[index_name], window_size, num_heads[layer_idx])
                        self.weights.append(relative_position_bias)
                    else:
                        print("[ERROR][SwinTransformerWeights::__init__] missing weight {} or {}.".format(index_name, table_name))
                        exit(-1)
                ##deal with layer weights
                ###loop over layer_weight_suffixes
                for layer_weight_suffix in layer_weight_suffixes:
                    weight_name = 'layers.{}.{}'.format(layer_idx, layer_weight_suffix)
                    if weight_name in weights:
                        self.weights.append(weights[weight_name])
                    else:
                        ####the last layer has not dowmsample weight
                        if layer_idx == layer_num - 1:
                            self.weights.append(torch.Tensor())
                        else:
                            print("[ERROR][SwinTransformerWeights::__init__] missing weight {}.".format(weight_name))
                            exit(-1)
                ###get attn_mask (same for each layer, some layer may not has one)
                attn_mask_name = 'layers.{}.blocks.1.attn_mask'.format(layer_idx)
                if attn_mask_name in weights:
                    self.weights.append(weights[attn_mask_name])
                else:
                    self.weights.append(torch.Tensor())
            #deal with sw weights
            for sw_weight_suffix in sw_weight_suffixes:
                weight_name = '{}'.format(sw_weight_suffix)
                if weight_name in weights:
                    self.weights.append(weights[weight_name])
                else:
                    print("[ERROR][SwinTransformerWeights::__init__] missing weight {}.".format(weight_name))
                    exit(-1)


    def to_cuda(self):
        for idx, v in enumerate(self.weights):
            self.weights[idx] = v.cuda()

    def to_half(self):
        for idx, v in enumerate(self.weights):
            self.weights[idx] = v.half()

