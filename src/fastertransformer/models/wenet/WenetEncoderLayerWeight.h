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
struct RelPositionMultiHeadedAttentionWeight {

    DenseWeight<T> query_weight;
    DenseWeight<T> key_weight;
    DenseWeight<T> value_weight;
    DenseWeight<T> attention_output_weight;

    DenseWeight<T> pos_weight;
    const T*       pos_bias_u = nullptr;
    const T*       pos_bias_v = nullptr;
};

template<typename T>
struct ConformerConvWeight {
    DenseWeight<T>     pointwise_conv1_weight;
    DenseWeight<T>     depthwise_conv_weight;
    LayerNormWeight<T> norm_weights;
    DenseWeight<T>     pointwise_conv2_weight;
};

template<typename T>
struct WenetEncoderLayerWeight {

    WenetEncoderLayerWeight() = default;
    WenetEncoderLayerWeight(const size_t layer_id,
                            const size_t head_num,
                            const size_t size_per_head,
                            const size_t inter_size,
                            const size_t conv_module_kernel_size,
                            const bool   use_layernorm_in_conv_module = false);

    ~WenetEncoderLayerWeight();
    WenetEncoderLayerWeight(const WenetEncoderLayerWeight& other);
    WenetEncoderLayerWeight& operator=(const WenetEncoderLayerWeight& other);
    /*
    #ifdef SPARSITY_ENABLED
        void compress_weights(cublasMMWrapper& cublas_wrapper, int hidden_dim);
    #endif
    */

    LayerNormWeight<T> norm_ff_macaron_weights;
    FfnWeight<T>       feed_forward_macaron_weights;

    LayerNormWeight<T>                       attn_layernorm_weights;
    RelPositionMultiHeadedAttentionWeight<T> attention_weights;

    LayerNormWeight<T>     norm_conv_weights;
    ConformerConvWeight<T> conv_module_weights;

    LayerNormWeight<T> ffn_layernorm_weights;
    FfnWeight<T>       ffn_weights;

    LayerNormWeight<T> norm_final_weights;

    void loadModel(std::string dir_path, FtCudaDataType model_file_type);

private:
    void setWeightPtr();
    void mallocWeights();
    void initialize();

    size_t layer_id_;
    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t conv_module_kernel_size_;

    bool             is_maintain_buffer = false;
    const static int weights_num_       = 37;
    int              real_weights_num_;
    T*               weights_ptr[weights_num_];
    size_t           weights_size[weights_num_];
    /*
        T* sp_weights_ptr[6];
        bool is_maintain_sp_buffer = false;
    */

    // for model structure
    bool use_layernorm_in_conv_module_ = false;
};

}  // namespace fastertransformer
