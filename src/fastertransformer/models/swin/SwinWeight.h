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
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
#include <cuda_fp16.h>

namespace fastertransformer {

template<typename T>
class SwinTransformerBlockWeight {
public:
    AttentionWeight<T> attention_weights;
    FfnWeight<T> ffn_weights;
    LayerNormWeight<T> attn_layernorm_weights;
    LayerNormWeight<T> ffn_layernorm_weights;
    const T* attention_relative_pos_bias = nullptr;
};  // SwinTransformerBlockWeight

template<typename T>
class SwinTransformerBasicLayerWeight {
public:
    LayerNormWeight<T> merge_layernorm_weights;
    DenseWeight<T> merge_linear_weights;
    const T* attn_mask = nullptr;
    std::vector<SwinTransformerBlockWeight<T>> block_weight_list;
};  // SwinTransformerBasicLayerWeight

template<typename T>
class SwinTransformerWeight {
public:
    DenseWeight<T> patchEmbed_linear_weights;
    LayerNormWeight<T> patchEmbed_norm_weights;
    LayerNormWeight<T> norm_weights;
    std::vector<SwinTransformerBasicLayerWeight<T>> basic_layer_weight_list;
};  // class SwinTransformerWeight

}  // namespace fastertransformer