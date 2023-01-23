# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
from checkpoint_quantization import extract_amaxlist


class SwinTransformerINT8Weight(object):
    def __init__(self, layer_num, window_size, depths, num_heads, ths_path, weights=None, version=1):
        """weights need be a state_dict of swin transformer model"""
        block_weight_suffixes = ['attn.proj.weight',
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
        torch.classes.load_library(ths_path)
        gen_relative_pos_bias = torch.ops.fastertransformer.gen_relative_pos_bias
        transform_trt_mask = torch.ops.fastertransformer.transform_trt_mask
        if weights is None:
            print("[ERROR][SwinTransformerWeights::__init__] weights should not be empty!")
            exit(-1)

        if 'layers.0.blocks.0.attn.qkv._input_quantizer._amax' not in weights.keys():
            raise RuntimeError("There is no quantization node in the checkpoint, cannot be quantized to int8.")
        for k, v in weights.items():
            if k.endswith('_amax'):
                weights[k] = v.cpu()
            elif k.endswith('relative_position_index'):
                weights[k] = v
            else:
                weights[k] = v.half()
                
        # exit(0)
        weights = extract_amaxlist(weights, depths, version=version, ths_path=ths_path, verbose=False)
        h_scale_list = {}
        for k, v in weights.items():
            if "amaxList" in k:
                k_h = k.replace("amaxList", "h_amaxList")
                h_scale_list[k_h] = v
            weights[k] = v.cuda() 
        for k, v in h_scale_list.items():
            weights[k] = v
        
        #calculate size_per_head
            qkv_weight_name = "layers.0.blocks.0.attn.qkv.weight"
            shape = weights[qkv_weight_name].shape
            #in case we flatten this weight
            if len(shape) == 1:
                dim = int(math.sqrt(shape[0]/3))
                weights[qkv_weight_name] = weights[qkv_weight_name].reshape([3*dim, dim])
                shape = weights[qkv_weight_name].shape
            size_per_head = int(shape[0]/3/num_heads[0])

        #loop over layers
        for layer_idx in range(layer_num):
            ##loop over blocks
            for block_idx in range(depths[layer_idx]):
                # deal with attn.qkv.weight
                qkv_weight_name = "layers.{}.blocks.{}.attn.qkv.weight".format(layer_idx, block_idx)
                if qkv_weight_name in weights:
                    self.weights.append(weights[qkv_weight_name])
                else:
                    print("[ERROR][SwinTransformerWeights::__init__] missing weight {}.".format(qkv_weight_name))
                    exit(-1)
                # deal with attn.qkv.bias
                if version == 1:
                    qkv_bias_name = "layers.{}.blocks.{}.attn.qkv.bias".format(layer_idx, block_idx)
                    if qkv_bias_name in weights:
                        self.weights.append(weights[qkv_bias_name])
                    else:
                        print("[ERROR][SwinTransformerWeights::__init__] missing weight {}.".format(qkv_bias_name))
                        exit(-1)
                elif version == 2:
                    q_bias_name = "layers.{}.blocks.{}.attn.q_bias".format(layer_idx, block_idx)
                    v_bias_name = "layers.{}.blocks.{}.attn.v_bias".format(layer_idx, block_idx)
                    if q_bias_name in weights and v_bias_name in weights:
                        qkv_weights = torch.cat((weights[q_bias_name], torch.zeros_like(weights[v_bias_name], requires_grad=False), weights[v_bias_name]))
                        self.weights.append(qkv_weights)
                    else:
                        print("[ERROR][SwinTransformerWeights::__init__] missing weight {} or {}.".format(q_bias_name, v_bias_name))
                        exit(-1)
                ###block_weight_suffixes
                for block_weight_suffix in block_weight_suffixes:
                    weight_name = 'layers.{}.blocks.{}.{}'.format(layer_idx, block_idx, block_weight_suffix)
                    if weight_name in weights:
                        self.weights.append(weights[weight_name])
                    else:
                        print("[ERROR][SwinTransformerWeights::__init__] missing weight {}.".format(weight_name))
                        exit(-1)
                ###get relative position bias
                ###Notice : for some model, like (img_size, window_size) = (224, 16), 
                ###the window_size_in_use of last layer may changes.
                if version == 1:
                    index_name = 'layers.{}.blocks.{}.attn.relative_position_index'.format(layer_idx, block_idx)
                    table_name = 'layers.{}.blocks.{}.attn.relative_position_bias_table'.format(layer_idx, block_idx)
                    if index_name in weights and table_name in weights:
                        window_size_in_use = int(math.sqrt(weights[index_name].shape[0]))
                        relative_position_bias = gen_relative_pos_bias(weights[table_name], weights[index_name], window_size_in_use, num_heads[layer_idx], weights[table_name], weights[table_name], weights[table_name], version)
                        self.weights.append(relative_position_bias.half())
                        if relative_position_bias.shape[1] <= 256 and size_per_head == 32:
                            trt_relative_position_bias = transform_trt_mask(relative_position_bias.half(), relative_position_bias.shape[0], relative_position_bias.shape[1], True)
                            self.weights.append(trt_relative_position_bias.half())
                        else:
                            self.weights.append(torch.HalfTensor())
                    else:
                        print("[ERROR][SwinTransformerWeights::__init__] missing weight {} or {}.".format(index_name, table_name))
                        exit(-1)
                elif version == 2:
                    index_name = 'layers.{}.blocks.{}.attn.relative_position_index'.format(layer_idx, block_idx)
                    table_name = 'layers.{}.blocks.{}.attn.relative_coords_table'.format(layer_idx, block_idx)
                    cpb_mlp_weight1_name = 'layers.{}.blocks.{}.attn.cpb_mlp.0.weight'.format(layer_idx, block_idx)
                    cpb_mlp_bias1_name = 'layers.{}.blocks.{}.attn.cpb_mlp.0.bias'.format(layer_idx, block_idx)
                    cpb_mlp_weight2_name = 'layers.{}.blocks.{}.attn.cpb_mlp.2.weight'.format(layer_idx, block_idx)
                    if index_name in weights and table_name in weights and cpb_mlp_weight1_name in weights and cpb_mlp_bias1_name in weights and cpb_mlp_weight2_name in weights:
                        window_size_in_use = int(math.sqrt(weights[index_name].shape[0]))
                        relative_position_bias = gen_relative_pos_bias(weights[table_name], weights[index_name], window_size_in_use, num_heads[layer_idx], weights[cpb_mlp_weight1_name], weights[cpb_mlp_bias1_name], weights[cpb_mlp_weight2_name], version)
                        self.weights.append(relative_position_bias)
                        if relative_position_bias.shape[1] <= 256 and size_per_head == 32:
                            trt_relative_position_bias = transform_trt_mask(relative_position_bias.half(), relative_position_bias.shape[0], relative_position_bias.shape[1], True)
                            self.weights.append(trt_relative_position_bias.half())
                        else:
                            self.weights.append(torch.HalfTensor())
                    else:
                        print("[ERROR][SwinTransformerWeights::__init__] missing weight {} or {} or {} or {} or {}.".format(index_name, table_name, cpb_mlp_weight1_name, cpb_mlp_bias1_name, cpb_mlp_weight2_name))
                        exit(-1)
                print('relative_position_bias', self.weights[-2].shape, '=>', self.weights[-1].shape)
                ##process attn.logit_scale for version 2
                if version == 2:
                    logit_scale_name = 'layers.{}.blocks.{}.attn.logit_scale'.format(layer_idx, block_idx)
                    if logit_scale_name in weights:
                        self.weights.append(torch.clamp(weights[logit_scale_name], max=torch.log(torch.tensor(1. / 0.01))).exp())
                    else:
                        print("[ERROR][SwinTransformerWeights::__init__] missing weight {}.".format(logit_scale_name))
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
                if weights[attn_mask_name].shape[1] <= 256 and size_per_head == 32:
                    trt_attn_mask = transform_trt_mask(weights[attn_mask_name].cuda().half(), weights[attn_mask_name].shape[0], weights[attn_mask_name].shape[1], True)
                    self.weights.append(trt_attn_mask.half())
                else:
                    self.weights.append(torch.HalfTensor())
            else:
                self.weights.append(torch.HalfTensor())
                self.weights.append(torch.HalfTensor())
            print('attn_mask', self.weights[-2].shape, '=>', self.weights[-1].shape)
        #deal with sw weights
        for sw_weight_suffix in sw_weight_suffixes:
            weight_name = sw_weight_suffix
            if weight_name in weights:
                self.weights.append(weights[weight_name])
            else:
                print("[ERROR][SwinTransformerWeights::__init__] missing weight {}.".format(weight_name))
                exit(-1)


    def to_half(self):
        if self.int8:
            return
        for idx, v in enumerate(self.weights):
            self.weights[idx] = v.half()
        

