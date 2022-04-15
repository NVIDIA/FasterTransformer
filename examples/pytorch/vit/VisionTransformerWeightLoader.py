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

class ViTWeightLoader(object):
    def __init__(self, layer_num, img_size, patch_size, weight_path=None, classifier='token'):
        """weights need be a state_dict of swin transformer model"""
        layer_weight_names = [
                                os.path.join(L_ROOT, ATT_NORM, 'scale'  ),
                                os.path.join(L_ROOT, ATT_NORM, 'bias'   ),
                                os.path.join(L_ROOT, ATT_Q,    'kernel' ),
                                os.path.join(L_ROOT, ATT_Q,    'bias'   ),
                                os.path.join(L_ROOT, ATT_K,    'kernel' ),
                                os.path.join(L_ROOT, ATT_K,    'bias'   ),
                                os.path.join(L_ROOT, ATT_V,    'kernel' ),
                                os.path.join(L_ROOT, ATT_V,    'bias'   ),
                                os.path.join(L_ROOT, ATT_OUT,  'kernel' ),
                                os.path.join(L_ROOT, ATT_OUT,  'bias'   ),
                                os.path.join(L_ROOT, FFN_NORM, 'scale'  ),
                                os.path.join(L_ROOT, FFN_NORM, 'bias'   ),
                                os.path.join(L_ROOT, FFN_IN,   'kernel' ),
                                os.path.join(L_ROOT, FFN_IN,   'bias'   ),
                                os.path.join(L_ROOT, FFN_OUT,  'kernel' ),
                                os.path.join(L_ROOT, FFN_OUT,  'bias'   )
        ]
            
        pre_layer_weight_names = [
                                 'embedding/kernel',
                                 'embedding/bias',
                                 'cls',
                                 'Transformer/posembed_input/pos_embedding'
        ]
        post_layer_weight_names = [
                                 'Transformer/encoder_norm/scale', 
                                 'Transformer/encoder_norm/bias' 
        ]
        self.layer_num = layer_num
        self.weights = []
        if weight_path is None:
            print("[ERROR][SwinTransformerWeights::__init__] weights should not be empty!")
            exit(-1)
        else:
            self._generated_weights = False
            weight_dict = self.load_weights(weight_path)

            for name in pre_layer_weight_names:
                if name not in weight_dict.files:
                    print("Unsupport weight file: Missing weights %s" % name)
                is_conv = name == 'embedding/kernel'

                if classifier != 'token' and name == 'cls':
                    continue

                th_weight = np2th(weight_dict[name], is_conv)
                if name.split('/')[-1] == "pos_embedding":
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

                self.weights.append(th_weight)
            #loop over layers
            for layer_idx in range(layer_num):
                for name in layer_weight_names:
                    w_name = name.format(layer_idx)
                    if w_name not in weight_dict.files:
                        print("Unsupport weight file: Missing weights %s" % w_name)
                    th_weight = np2th(weight_dict[w_name])
                    self.weights.append(th_weight)

            for name in post_layer_weight_names:
                if name not in weight_dict.files:
                    print("Unsupport weight file: Missing weights %s" % name)
                th_weight = np2th(weight_dict[name])
                self.weights.append(th_weight)
    
    def load_weights(self, weight_path:str):
        suffix = weight_path.split('.')[-1]
        if suffix != 'npz':
            print("Unsupport weight file: Unrecognized format %s " % suffix)
            exit(-1)
        return np.load(weight_path)

    def to_cuda(self):
        for idx, v in enumerate(self.weights):
            self.weights[idx] = v.cuda()

    def to_half(self):
        for idx, v in enumerate(self.weights):
            self.weights[idx] = v.half()

