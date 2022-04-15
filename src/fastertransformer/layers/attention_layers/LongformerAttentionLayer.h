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

#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/layers/DenseWeight.h"

namespace fastertransformer {

template<typename T>
class LongformerAttentionLayer: public BaseLayer {
private:
    size_t head_num_;
    size_t size_per_head_;
    size_t local_attn_window_size_;
    size_t max_global_token_num_;
    size_t max_batch_size_;
    size_t max_seq_len_;
    float attn_scaler_;

    // interal buffers
    void* internal_vars_device_;
    T* attn_buffer_;

    cudaStream_t memcpy_stream_;

    void allocateBuffer() override;
    void freeBuffer() override;

public:
    LongformerAttentionLayer(size_t head_num,
                             size_t size_per_head,
                             size_t local_attn_window_size,
                             size_t max_global_token_num,
                             size_t max_batch_size,
                             size_t max_seq_len,
                             float attn_scaler,
                             cudaStream_t stream,
                             cublasMMWrapper* cublas_wrapper,
                             IAllocator* allocator,
                             bool is_free_buffer_after_forward = false);
    ~LongformerAttentionLayer();
    void forward(std::vector<fastertransformer::Tensor>* output_tensors,
                 const std::vector<fastertransformer::Tensor>* input_tensors);
};

}  // namespace fastertransformer