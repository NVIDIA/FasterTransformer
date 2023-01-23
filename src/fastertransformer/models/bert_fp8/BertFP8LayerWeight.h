/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <stdint.h>
#include <string>
#include <vector>

#include "src/fastertransformer/kernels/layernorm_fp8_kernels.h"
#include "src/fastertransformer/layers/FfnFP8Weight.h"
#include "src/fastertransformer/layers/attention_layers_fp8/AttentionFP8Weight.h"
#include "src/fastertransformer/models/bert_fp8/serialize.hpp"
#include "src/fastertransformer/utils/cuda_fp8_utils.h"

namespace fastertransformer {

template<typename T1, typename T2>
struct BertFP8LayerWeight {

    BertFP8LayerWeight() = default;
    BertFP8LayerWeight(const size_t d_model,
                       const size_t head_num,
                       const size_t size_per_head,
                       const size_t inter_size,
                       const size_t tensor_para_size,
                       const size_t pipeline_para_size,
                       const int    fp8_mode,
                       bool         is_load_model,
                       bool         is_fused_qkv_gemm);
    ~BertFP8LayerWeight();
    BertFP8LayerWeight(const BertFP8LayerWeight& other);
    BertFP8LayerWeight& operator=(const BertFP8LayerWeight& other) = delete;
    void                transposeWeight();
    void                quantizeWeights();
    void                loadModel(std::string dir_path);
    void                serialize(uint8_t*& buffer) const;
    void                deserialize(const uint8_t*& buffer);
    int32_t             getSerializationSize() const;

    AttentionFP8Weight<T1, T2> attention_weights;
    LayerNormWeight<T2>        attn_layernorm_weights;
    FfnFP8Weight<T1, T2>       ffn_weights;
    LayerNormWeight<T2>        ffn_layernorm_weights;

private:
    void setWeightPtr();
    void mallocWeights();

    size_t d_model_              = 0;
    size_t head_num_             = 0;
    size_t size_per_head_        = 0;
    size_t hidden_units_         = 0;
    size_t inter_size_           = 0;
    size_t tensor_para_size_     = 1;
    size_t pipeline_para_size_   = 1;
    bool   is_maintain_buffer    = false;
    bool   is_maintain_sp_buffer = false;
    // mode 0: no fp8. Should use original bert directly
    // mode 1: per tensor scale for activation, per channel scale for weight
    // mode 2: per tensor scale for activation and weight
    int  fp8_mode_;
    bool is_fused_qkv_gemm_bias_;

    std::vector<std::pair<size_t, T1*>>    weights_ptr;
    std::vector<std::pair<size_t, T2*>>    vec_ptr;  // containing bias, gamma and beta
    std::vector<std::pair<size_t, T1*>>    sp_weights_ptr;
    std::vector<std::pair<size_t, float*>> scale_ptr_;
    std::vector<float*>                    scale_h_ptr_;
};

}  // namespace fastertransformer
