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
from checkpoint_quantization import extract_amaxlist


class SwinTransformerINT8Weight(object):
    def __init__(self, layer_num, window_size, depths, num_heads, ths_path, weights=None, int8=True):
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
                                 'norm2.bias',
                                 'amaxList',
                                 'h_amaxList']
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
        self.int8 = int8
        torch.classes.load_library(ths_path)
        gen_relative_pos_bias = torch.ops.fastertransformer.gen_relative_pos_bias
        if weights is None:
            print("[ERROR][SwinTransformerWeights::__init__] weights should not be empty!")
            exit(-1)

        # print(weights.keys())
        if self.int8:
            if 'layers.0.blocks.0.attn.qkv._input_quantizer._amax' not in weights.keys():
                raise RuntimeError("There is no quantization node in the checkpoint, cannot be quantized to int8.")
            for k, v in weights.items():
                if k.endswith('bias') or k.endswith('weight'):
                    weights[k] = v.half()
                else:
                    weights[k] = v.cpu()
            weights = extract_amaxlist(weights, depths, ths_path=ths_path, verbose=False)
            h_scale_list = {}
            for k, v in weights.items():
                if "amaxList" in k:
                    k_h = k.replace("amaxList", "h_amaxList")
                    h_scale_list[k_h] = v
                weights[k] = v.cuda() 
            for k, v in h_scale_list.items():
                weights[k] = v
        else:
            for idx, v in enumerate(self.weights):
                weights[idx] = v.cuda()
        
        self._generated_weights = False
        #loop over layers
        for layer_idx in range(layer_num):
            ##loop over blocks
            for block_idx in range(depths[layer_idx]):
                ###block_weight_suffixes
                for block_weight_suffix in block_weight_suffixes:
                    weight_name = 'layers.{}.blocks.{}.{}'.format(layer_idx, block_idx, block_weight_suffix)
                    if weight_name in weights:
                        self.weights.append(weights[weight_name])
                    else:
                        print("[ERROR][SwinTransformerWeights::__init__] missing weight {}.".format(weight_name))
                        exit(-1)
                ###get relative position bias
                index_name = 'layers.{}.blocks.{}.attn.relative_position_index'.format(layer_idx, block_idx)
                table_name = 'layers.{}.blocks.{}.attn.relative_position_bias_table'.format(layer_idx, block_idx)
                if index_name in weights and table_name in weights:
                    relative_position_bias = gen_relative_pos_bias(weights[table_name], weights[index_name], window_size, num_heads[layer_idx])
                    self.weights.append(relative_position_bias.half())
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
                    ####the last layer do not have downsample weight
                    if layer_idx == layer_num - 1:
                        self.weights.append(torch.Tensor())
                    else:
                        print("[ERROR][SwinTransformerWeights::__init__] missing weight {}.".format(weight_name))
                        exit(-1)
            ###get attn_mask (same for each layer, some layer may not has one)
            attn_mask_name = 'layers.{}.blocks.1.attn_mask'.format(layer_idx)
            if attn_mask_name in weights:
                self.weights.append(weights[attn_mask_name].half())
            else:
                self.weights.append(torch.HalfTensor())
        #deal with sw weights
        for sw_weight_suffix in sw_weight_suffixes:
            weight_name = sw_weight_suffix
            if weight_name in weights:
                self.weights.append(weights[weight_name])
            else:
                print("[ERROR][SwinTransformerWeights::__init__] missing weight {}.".format(weight_name))
                exit(-1)
        # for w in self.weights:
        #     print(w.type())


    def to_half(self):
        if self.int8:
            return
        for idx, v in enumerate(self.weights):
            self.weights[idx] = v.half()
        

