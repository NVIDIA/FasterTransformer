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

#include <string>
#include <vector>

#include "src/fastertransformer/kernels/layernorm_fp8_kernels.h"
#include "src/fastertransformer/layers/FfnFP8Weight.h"
#include "src/fastertransformer/layers/attention_layers_fp8/AttentionFP8Weight.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/cuda_fp8_utils.h"

namespace fastertransformer {

template<typename T1, typename T2>
struct GptFP8DecoderLayerWeight {
public:
    GptFP8DecoderLayerWeight() = default;
    GptFP8DecoderLayerWeight(const int hidden_units,
                             const int inter_size,
                             const int tensor_para_size,
                             const int tensor_para_rank);
    ~GptFP8DecoderLayerWeight();
    GptFP8DecoderLayerWeight(const GptFP8DecoderLayerWeight& other);
    GptFP8DecoderLayerWeight& operator=(const GptFP8DecoderLayerWeight& other);
    void                      loadModel(std::string dir_path);
    // #ifdef SPARSITY_ENABLED
    //     void compress_weights(cublasMMWrapper& cublas_wrapper, int hidden_dim);
    // #endif
    void transposeWeight();

    LayerNormWeight<T2>        pre_layernorm_weights;
    AttentionFP8Weight<T1, T2> self_attention_weights;
    LayerNormWeight<T2>        self_attn_layernorm_weights;
    FfnFP8Weight<T1, T2>       ffn_weights;
    float*                     identity_scale;
    float*                     identity_h_scale;

    T1* fp8_qkv_bias;

private:
    void setWeightPtr();
    void mallocWeights();

protected:
    size_t                                hidden_units_;
    size_t                                inter_size_;
    size_t                                tensor_para_size_  = 1;
    size_t                                tensor_para_rank_  = 0;
    bool                                  is_maintain_buffer = false;
    std::vector<std::pair<uint32_t, T1*>> weights_ptr;
    std::vector<std::pair<uint32_t, T2*>> vec_ptr;  // containing bias, gamma and beta

    std::vector<T1*> trans_ptr;

    std::vector<std::pair<uint32_t, float*>> scale_ptr;
    std::vector<float*>                      scale_h_ptr_;
    cudaStream_t                             stream_   = 0;
    const int                                fp8_mode_ = 2;

#ifdef SPARSITY_ENABLED
    std::pair<uint32_t, T1*> sp_weights_ptr[4];
    bool                     is_maintain_sp_buffer = false;
#endif
};

}  // namespace fastertransformer
