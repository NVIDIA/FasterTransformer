/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include "src/fastertransformer/layers/attention_layers/BaseAttentionLayer.h"

namespace fastertransformer {

template<typename T>
class GptContextAttentionLayer: public BaseAttentionLayer<T> {
private:
    // buffer handling
    size_t max_batch_size_ = 0;
    size_t max_seq_len_ = 0;

    // metadata
    const size_t head_num_;
    const size_t size_per_head_;
    const size_t hidden_units_;
    const size_t local_head_num_;
    const size_t local_hidden_units_;
    const size_t rotary_embedding_dim_;

    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size, size_t seq_len);
    void freeBuffer() override;
    bool isValidBatchSize(size_t batch_size);
    bool isValidSeqLen(size_t seq_len);

    using BaseAttentionLayer<T>::is_free_buffer_after_forward_;
    using BaseAttentionLayer<T>::is_allocate_buffer_;
    using BaseAttentionLayer<T>::cublas_wrapper_;
    using BaseAttentionLayer<T>::allocator_;

    bool is_qk_buf_float_;

protected:
    using BaseAttentionLayer<T>::stream_;
    using BaseAttentionLayer<T>::sparse_;
    T* qkv_buf_ = nullptr;
    T* q_buf_2_ = nullptr;
    T* k_buf_2_ = nullptr;
    T* v_buf_2_ = nullptr;
    T* qk_buf_ = nullptr;
    float* qk_buf_float_ = nullptr;
    T* qkv_buf_2_ = nullptr;
    T* qkv_buf_3_ = nullptr;

public:
    GptContextAttentionLayer(size_t max_batch_size,
                             size_t max_seq_len,
                             size_t head_num,
                             size_t size_per_head,
                             cudaStream_t stream,
                             cublasMMWrapper* cublas_wrapper,
                             IAllocator* allocator,
                             bool is_free_buffer_after_forward,
                             bool is_qk_buf_float,
                             bool sparse = false);

    GptContextAttentionLayer(size_t max_batch_size,
                             size_t max_seq_len,
                             size_t head_num,
                             size_t size_per_head,
                             size_t local_head_num,
                             cudaStream_t stream,
                             cublasMMWrapper* cublas_wrapper,
                             IAllocator* allocator,
                             bool is_free_buffer_after_forward,
                             bool is_qk_buf_float,
                             bool sparse = false);

    GptContextAttentionLayer(size_t max_batch_size,
                             size_t max_seq_len,
                             size_t head_num,
                             size_t size_per_head,
                             size_t local_head_num,
                             size_t rotary_embedding_dim,
                             cudaStream_t stream,
                             cublasMMWrapper* cublas_wrapper,
                             IAllocator* allocator,
                             bool is_free_buffer_after_forward,
                             bool is_qk_buf_float,
                             bool sparse = false);

    GptContextAttentionLayer(GptContextAttentionLayer<T> const& attention_layer);

    virtual ~GptContextAttentionLayer();

    void forward(std::vector<fastertransformer::Tensor>* output_tensors,
                 const std::vector<fastertransformer::Tensor>* input_tensors,
                 const AttentionWeight<T>* attention_weights) override;
};

}  // namespace fastertransformer
