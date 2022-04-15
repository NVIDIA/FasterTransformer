/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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
struct T5EncoderLayerWeight {

    T5EncoderLayerWeight() = default;
    T5EncoderLayerWeight(const size_t head_num,
                         const size_t size_per_head,
                         const size_t d_model,
                         const size_t inter_size,
                         const size_t tensor_para_size,
                         const size_t tensor_para_rank,
                         const bool t5_with_bias);
    ~T5EncoderLayerWeight();
    T5EncoderLayerWeight(const T5EncoderLayerWeight& other);
    T5EncoderLayerWeight& operator=(const T5EncoderLayerWeight& other);

#ifdef SPARSITY_ENABLED
    void compress_weights(cublasMMWrapper& cublas_wrapper, int hidden_dim);
#endif

    AttentionWeight<T> attention_weights;
    LayerNormWeight<T> attn_layernorm_weights;
    FfnWeight<T> ffn_weights;
    LayerNormWeight<T> ffn_layernorm_weights;
    bool t5_with_bias_;

    void loadModel(std::string dir_path, FtCudaDataType model_file_type);

    void setT5WithBias(bool t5_with_bias_para);

private:
    void setWeightPtr();
    void mallocWeights();
    void initialize();

    size_t head_num_;
    size_t size_per_head_;
    size_t d_model_;
    size_t inter_size_;
    size_t tensor_para_size_;
    size_t tensor_para_rank_;

    int real_weights_num_;

    bool is_maintain_buffer = false;

    // Assume bias added
    const static int weights_num_ = 16;
    T* weights_ptr[weights_num_];
    size_t weights_size[weights_num_];

    T* sp_weights_ptr[6];
    bool is_maintain_sp_buffer = false;
};

}  // namespace fastertransformer
