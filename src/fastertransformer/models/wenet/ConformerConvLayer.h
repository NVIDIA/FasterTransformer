/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2022.  Authored by Yuqing Ding.
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

#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/kernels/matrix_vector_multiplication.h"
#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/models/wenet/WenetEncoderLayerWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include <vector>

namespace fastertransformer {

template<typename T>
class ConformerConvLayer: public BaseLayer {
private:
    // meta data
    size_t head_num_;
    size_t size_per_head_;
    size_t conv_module_kernel_size_;

    // calculated data
    size_t hidden_units_;

    int int8_mode_ = 0;

    // for model structure
    const bool use_layernorm_in_conv_module_ = false;

    void allocateBuffer();
    void freeBuffer();
    void allocateBuffer(size_t token_num);

protected:
    T* input_remove_padding_ = nullptr;
    T* inter_buf_            = nullptr;
    T* inter2_buf_           = nullptr;
    T* normed_inter_buf_     = nullptr;

public:
    ConformerConvLayer(size_t           max_batch_size,
                       size_t           max_seq_len,
                       size_t           head_num,
                       size_t           size_per_head,
                       size_t           conv_module_kernel_size,
                       cudaStream_t     stream,
                       cublasMMWrapper* cublas_wrapper,
                       IAllocator*      allocator,
                       bool             is_free_buffer_after_forward,
                       bool             sparse                       = false,
                       int              int8_mode                    = 0,
                       bool             use_layernorm_in_conv_module = false);

    ConformerConvLayer(ConformerConvLayer<T> const& conformer_conv_layer);

    virtual ~ConformerConvLayer();

    virtual void forward(std::vector<fastertransformer::Tensor>*       output_tensors,
                         const std::vector<fastertransformer::Tensor>* input_tensors,
                         const ConformerConvWeight<T>*                 conformer_conv_weights);

    void
    forward(TensorMap* output_tensors, TensorMap* input_tensors, const ConformerConvWeight<T>* conformer_conv_weights);
};

}  // namespace fastertransformer
