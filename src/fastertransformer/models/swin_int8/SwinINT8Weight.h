/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/FfnINT8Weight.h"
#include "src/fastertransformer/layers/attention_layers_int8/AttentionINT8Weight.h"
#include <cuda_fp16.h>
#include <cudnn.h>
#include <vector>
using namespace std;

namespace fastertransformer {

template<typename T>
class SwinTransformerINT8BlockWeight {
public:
    AttentionINT8Weight<T> attention_weights;
    FfnINT8Weight<T> ffn_weights;
    LayerNormWeight<T> attn_layernorm_weights;
    LayerNormWeight<T> ffn_layernorm_weights;
    const T* attention_relative_pos_bias = nullptr;
    ScaleList scalelist;
};  // SwinTransformerINT8BlockWeight

template<typename T>
class SwinTransformerINT8BasicLayerWeight {
public:
    LayerNormWeight<T> merge_layernorm_weights;
    DenseWeight<T> merge_linear_weights;
    const T* attn_mask = nullptr;
    vector<SwinTransformerINT8BlockWeight<T>> block_weight_list;
};  // SwinTransformerINT8BasicLayerWeight

template<typename T>
class SwinTransformerINT8Weight {
public:
    DenseWeight<T> patchEmbed_linear_weights;
    LayerNormWeight<T> patchEmbed_norm_weights;
    LayerNormWeight<T> norm_weights;
    vector<SwinTransformerINT8BasicLayerWeight<T>> basic_layer_weight_list;
};  // class SwinTransformerINT8Weight

}  // namespace fastertransformer