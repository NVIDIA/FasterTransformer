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
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
struct BartEncoderLayerWeight {

    BartEncoderLayerWeight() = default;
    BartEncoderLayerWeight(const size_t head_num,
                           const size_t size_per_head,
                           const size_t d_model,
                           const size_t inter_size,
                           const size_t tensor_para_size,
                           const size_t tensor_para_rank,
                           const bool   bart_with_bias       = true,
                           const bool   use_gated_activation = false);
    ~BartEncoderLayerWeight();
    BartEncoderLayerWeight(const BartEncoderLayerWeight& other);
    BartEncoderLayerWeight& operator=(const BartEncoderLayerWeight& other);

#ifdef SPARSITY_ENABLED
    void compress_weights(cublasMMWrapper& cublas_wrapper, int hidden_dim);
#endif

    AttentionWeight<T> attention_weights_;
    LayerNormWeight<T> attn_layernorm_weights_;
    FfnWeight<T>       ffn_weights_;
    LayerNormWeight<T> ffn_layernorm_weights_;
    bool               bart_with_bias_;
    bool               use_gated_activation_;

    void loadModel(std::string dir_path, FtCudaDataType model_file_type);
    void setBartWithBias(bool bart_with_bias_para, bool use_gated_activation_para);

private:
    void initialize();    /* compute weight shapes */
    void mallocWeights(); /* malloc internal byte buffers */
    void setWeightPtr();  /* weight class interface */

    size_t head_num_;
    size_t size_per_head_;
    size_t d_model_;
    size_t inter_size_;
    size_t tensor_para_size_;
    size_t tensor_para_rank_;

    bool is_maintain_buffer_ = false;

    // Assume bias added, and gated activation used
    int              real_weights_num_;
    const static int weights_num_ = 18;
    T*               weights_ptr_[weights_num_];
    size_t           weights_size_[weights_num_];

    T*   sp_weights_ptr_[6];
    bool is_maintain_sp_buffer_ = false;
};

}  // namespace fastertransformer
