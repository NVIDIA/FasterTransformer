/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include <string>

#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"


namespace fastertransformer {

template<typename T>
struct GptDecoderLayerWeight {
public:
    GptDecoderLayerWeight() = default;
    GptDecoderLayerWeight(const int hidden_units, const int inter_size);
    ~GptDecoderLayerWeight();
    GptDecoderLayerWeight(const GptDecoderLayerWeight& other);
    GptDecoderLayerWeight& operator=(const GptDecoderLayerWeight& other);
    void loadModel(std::string dir_path);
#ifdef SPARSITY_ENABLED
    void compress_weights(cublasMMWrapper& cublas_wrapper, int hidden_dim);
#endif
    LayerNormWeight<T> pre_layernorm_weights;
    AttentionWeight<T> self_attention_weights;
    LayerNormWeight<T> self_attn_layernorm_weights;
    FfnWeight<T> ffn_weights;

private:
    int hidden_units_;
    int inter_size_;
    bool is_maintain_buffer = false;
    T* weights_ptr[12];
#ifdef SPARSITY_ENABLED
    T* sp_weights_ptr[4];
    bool is_maintain_sp_buffer = false;
#endif

    void setWeightPtr();
    void mallocWeights();
};

}  // namespace fastertransformer
