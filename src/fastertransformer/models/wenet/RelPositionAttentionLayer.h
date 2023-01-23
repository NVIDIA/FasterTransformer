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

#include "src/fastertransformer/layers/attention_layers/BaseAttentionLayer.h"
#include "src/fastertransformer/models/wenet/WenetEncoderLayerWeight.h"

namespace fastertransformer {

template<typename T>
class RelPositionAttentionLayer: public BaseAttentionLayer<T> {
private:
    // metadata
    size_t head_num_;
    size_t size_per_head_;
    size_t hidden_units_;
    size_t d_model_;
    bool   sparse_;
    float  q_scaling_;

    T* last_q_w  = nullptr;
    T* last_k_w  = nullptr;
    T* last_v_w  = nullptr;
    T* last_q_in = nullptr;
    T* last_k_in = nullptr;
    T* last_v_in = nullptr;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t seq_len);

protected:
    using BaseAttentionLayer<T>::stream_;
    using BaseAttentionLayer<T>::is_free_buffer_after_forward_;
    using BaseAttentionLayer<T>::is_allocate_buffer_;
    using BaseAttentionLayer<T>::cublas_wrapper_;
    using BaseAttentionLayer<T>::allocator_;

    T* q_buf_ = nullptr;  // (BS, N, H)
    T* k_buf_ = nullptr;  // (BS, N, H)
    T* v_buf_ = nullptr;

    T* q_buf_2_ = nullptr;  // (B,N,S,H)  // with bias_u
    T* k_buf_2_ = nullptr;  // (B,N,H,S)
    T* v_buf_2_ = nullptr;  // (B,N,S,H)

    T* p_buf_        = nullptr;  // (BS, N, H)
    T* p_buf_2_      = nullptr;  // (B,N,H,S)
    T* q_buf_bias_v_ = nullptr;  // (B,N,S,H)
    T* qp_buf_       = nullptr;  // (B,N,S,S)

    T* qk_buf_ = nullptr;  // (B,N,S,S)

    T* qkv_buf_   = nullptr;  //(B,N,S,H)
    T* qkv_buf_2_ = nullptr;

    T** batch_qkv_kernel_ptr_ = nullptr;
    T** batch_qkv_input_ptr_  = nullptr;
    T** batch_qkv_buf_ptr_    = nullptr;

public:
    RelPositionAttentionLayer(size_t           max_batch_size,
                              size_t           max_seq_len,
                              size_t           head_num,
                              size_t           size_per_head,
                              float            q_scaling,
                              cudaStream_t     stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator*      allocator,
                              bool             is_free_buffer_after_forward,
                              bool             sparse = false);

    RelPositionAttentionLayer(size_t           max_batch_size,
                              size_t           max_seq_len,
                              size_t           head_num,
                              size_t           size_per_head,
                              size_t           d_model,
                              float            q_scaling,
                              cudaStream_t     stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator*      allocator,
                              bool             is_free_buffer_after_forward,
                              bool             sparse = false);

    RelPositionAttentionLayer(RelPositionAttentionLayer<T> const& attention_layer);

    ~RelPositionAttentionLayer();

    void
    forward(TensorMap* output_tensors, TensorMap* input_tensors, const AttentionWeight<T>* attention_weights) override;

    void forward(TensorMap*                                      output_tensors,
                 TensorMap*                                      input_tensors,
                 const RelPositionMultiHeadedAttentionWeight<T>* attention_weights);
};

}  // namespace fastertransformer
