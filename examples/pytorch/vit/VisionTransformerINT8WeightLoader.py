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

import torch as th
import math
import numpy as np
import os
from scipy import ndimage

from checkpoint_quantization import checkpoint_quantization


L_ROOT   = 'Transformer/encoderblock_{}'
ATT_Q    = 'MultiHeadDotProductAttention_1/query'
ATT_K    = 'MultiHeadDotProductAttention_1/key'
ATT_V    = 'MultiHeadDotProductAttention_1/value'
ATT_OUT  = 'MultiHeadDotProductAttention_1/out'
ATT_NORM = 'LayerNorm_0'
FFN_NORM = 'LayerNorm_2'
FFN_IN   = 'MlpBlock_3/Dense_0'
FFN_OUT  = 'MlpBlock_3/Dense_1'

def np2th(weights, is_conv=False):
    if is_conv:
        """ convert HWIO to OIHW."""
        weights = weights.transpose([3, 2, 0, 1])
    return th.from_numpy(weights).contiguous()

class ViTINT8WeightLoader(object):
    def __init__(self, layer_num, img_size, patch_size, weight_dict=None, classifier='token'):
        """weights need be a pytorch state_dict of swin transformer model"""
            
        pre_layer_weight_names = [
                                 'transformer.embeddings.patch_embeddings.weight',
                                 'transformer.embeddings.patch_embeddings.bias',
                                 'transformer.embeddings.cls_token',
                                 'transformer.embeddings.position_embeddings'
        ]
        self.layer_num = layer_num
        self.weights = []
        self.int8 = False
        if weight_dict is None:
            print("[ERROR][SwinTransformerWeights::__init__] weights should not be empty!")
            exit(-1)

        self._generated_weights = False

        for name in pre_layer_weight_names:
            if name not in weight_dict.keys():
                print("Unsupport weight file: Missing weights %s" % name)

            th_weight = weight_dict[name]
            if name.split('.')[-1] == "pos_embedding":
                posemb_new_size = pow(img_size//patch_size, 2) + 1
                if th_weight.size(1) != posemb_new_size:
                    print("load_pretrained: resized variant: %s to %s" % (th_weight.size(1), posemb_new_size))
                    ntok_new = posemb_new_size

                    if classifier == "token":
                        posemb_tok, posemb_grid = th_weight[:, :1], th_weight[0, 1:]
                        ntok_new -= 1
                    else:
                        posemb_tok, posemb_grid = th_weight[:, :0], th_weight[0]

                    gs_old = int(np.sqrt(len(posemb_grid)))
                    gs_new = int(np.sqrt(ntok_new))
                    print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                    zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                    posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                    posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                    th_weight = np2th(posemb)
                    weight_dict[name] = th_weight

        self.weights = weight_dict
    
    # def load_weights(self, weight_path:str):
    #     suffix = weight_path.split('.')[-1]
    #     if suffix != 'pth':
    #         print("Unsupport weight file: Unrecognized format %s " % suffix)
    #         exit(-1)
    #     return th.load(weight_path)

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
        for k, v in self.weights.items():
            self.weights[k] = v.half()

    def listed_weights(self):
        ret = []
        for k, v in self.weights.items():
            if k.split('.')[-1] == '_amax' or k.endswith('amaxList'):
                continue
            if k.split('.')[0] == 'head':
                continue
            # print(k, v.type())
            ret.append(v)
        
        for i in range(self.layer_num):
            name = 'transformer.encoder.layer.{}.amaxList'.format(i)
            ret.append(self.weights[name])
            # print(name, self.weights[name].type())
            name = 'transformer.encoder.layer.{}.h_amaxList'.format(i)
            ret.append(self.weights[name])
            # print(name, self.weights[name].type())
        
        return ret

    def to_int8(self, ths_path='../../../lib/libpyt_vit.so'):
        # print(self.weights.keys())
        if 'transformer.encoder.layer.0.attn.query._input_quantizer._amax' not in self.weights:
            raise RuntimeError("There is no quantization node in the checkpoint, cannot be quantized to int8.")
        if self.int8:
            return
        self.int8 = True
        for k, v in self.weights.items():
            if k.endswith('bias') or k.endswith('norm.weight') or 'embeddings' in k:
                self.weights[k] = v.half()
            elif k.endswith('weight'):
                self.weights[k] = v.float().cuda()
            else:
                self.weights[k] = v.float().cpu()
        self.weights = checkpoint_quantization(self.weights, ths_path, verbose=False)
        # print(self.weights.keys())
