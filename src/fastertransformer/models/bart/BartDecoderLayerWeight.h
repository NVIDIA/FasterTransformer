/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "src/fastertransformer/utils/cuda_utils.h"
#include <string>

namespace fastertransformer {

template<typename T>
struct BartDecoderLayerWeight {

    BartDecoderLayerWeight() = default;
    BartDecoderLayerWeight(const size_t head_num,
                           const size_t size_per_head,
                           const size_t d_model,
                           const size_t inter_size,
                           const size_t mem_d_model,
                           const size_t tensor_para_size,
                           const size_t tensor_para_rank,
                           const bool   bart_with_bias       = true,
                           const bool   use_gated_activation = false);
    ~BartDecoderLayerWeight();
    BartDecoderLayerWeight(const BartDecoderLayerWeight& other);
    BartDecoderLayerWeight& operator=(const BartDecoderLayerWeight& other);

    LayerNormWeight<T> layernorm_weights;
    AttentionWeight<T> self_attention_weights;
    LayerNormWeight<T> self_attn_layernorm_weights;
    AttentionWeight<T> cross_attention_weights;
    LayerNormWeight<T> cross_attn_layernorm_weights;
    FfnWeight<T>       ffn_weights;
    bool               bart_with_bias_;
    bool               use_gated_activation_;

    void loadModel(std::string dir_path, FtCudaDataType model_file_type);

    void setBartWithBias(bool bart_with_bias_para, bool use_gated_activation_para);

private:
    void setWeightPtr();
    void mallocWeights();
    void initialize();

    size_t head_num_;
    size_t size_per_head_;
    size_t d_model_;
    size_t inter_size_;
    size_t mem_d_model_;
    size_t tensor_para_size_;
    size_t tensor_para_rank_;
    bool   is_maintain_buffer_ = false;

    int real_weights_num_;

    // Assume bias added, and gated activation used
    const static int weights_num_ = 24;
    T*               weights_ptr[weights_num_];
    size_t           weights_size[weights_num_];
};

}  // namespace fastertransformer
