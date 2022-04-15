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

#include "src/fastertransformer/kernels/matrix_vector_multiplication.h"
#include "src/fastertransformer/layers/attention_layers/BaseAttentionLayer.h"

namespace fastertransformer {

template<typename T>
class DecoderSelfAttentionLayer: public BaseAttentionLayer<T> {
private:
    // buffer handling
    size_t max_batch_size_;

    // metadata
    const size_t head_num_;
    const size_t size_per_head_;
    const size_t hidden_units_;
    const size_t local_head_num_;
    const size_t local_hidden_units_;
    const size_t d_model_;
    const float q_scaling_;
    const size_t rotary_embedding_dim_;

    const int int8_mode_ = 0;

    void allocateBuffer() override;
    void freeBuffer() override;
    bool isValidBatchSize(size_t batch_size);
    void allocateBuffer(size_t batch_size);

    using BaseAttentionLayer<T>::is_free_buffer_after_forward_;
    using BaseAttentionLayer<T>::is_allocate_buffer_;
    using BaseAttentionLayer<T>::cublas_wrapper_;
    using BaseAttentionLayer<T>::allocator_;

protected:
    T* qkv_buf_ = nullptr;
    T* context_buf_ = nullptr;
    using BaseAttentionLayer<T>::stream_;
    using BaseAttentionLayer<T>::sparse_;

public:
    DecoderSelfAttentionLayer(size_t max_batch_size,
                              size_t head_num,
                              size_t size_per_head,
                              size_t local_head_num,
                              size_t rotary_embedding_dim,
                              size_t d_model,
                              const float q_scaling,
                              cudaStream_t stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator* allocator,
                              bool is_free_buffer_after_forward,
                              bool sparse = false,
                              int int8_mode = 0);

    DecoderSelfAttentionLayer(size_t max_batch_size,
                              size_t head_num,
                              size_t size_per_head,
                              cudaStream_t stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator* allocator,
                              bool is_free_buffer_after_forward,
                              bool sparse = false,
                              int int8_mode = 0);

    DecoderSelfAttentionLayer(size_t max_batch_size,
                              size_t head_num,
                              size_t size_per_head,
                              const float q_scaling,
                              cudaStream_t stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator* allocator,
                              bool is_free_buffer_after_forward,
                              bool sparse = false,
                              int int8_mode = 0);

    DecoderSelfAttentionLayer(size_t max_batch_size,
                              size_t head_num,
                              size_t size_per_head,
                              size_t local_head_num,
                              cudaStream_t stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator* allocator,
                              bool is_free_buffer_after_forward,
                              bool sparse = false,
                              int int8_mode = 0);

    DecoderSelfAttentionLayer(size_t max_batch_size,
                              size_t head_num,
                              size_t size_per_head,
                              size_t local_head_num,
                              size_t d_model,
                              const float q_scaling,
                              cudaStream_t stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator* allocator,
                              bool is_free_buffer_after_forward,
                              bool sparse = false,
                              int int8_mode = 0);

    DecoderSelfAttentionLayer(size_t max_batch_size,
                              size_t head_num,
                              size_t size_per_head,
                              size_t local_head_num,
                              size_t rotary_embedding_dim,
                              cudaStream_t stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator* allocator,
                              bool is_free_buffer_after_forward,
                              bool sparse = false,
                              int int8_mode = 0);

    DecoderSelfAttentionLayer(DecoderSelfAttentionLayer<T> const& attention_layer);

    ~DecoderSelfAttentionLayer();

    void forward(std::vector<fastertransformer::Tensor>* output_tensors,
                 const std::vector<fastertransformer::Tensor>* input_tensors,
                 const AttentionWeight<T>* attention_weights) override;
};

template<typename T>
void fusedQKV_masked_attention_dispatch(const T* qkv_buf,
                                        const T* qkv_bias,
                                        T* key_cache,
                                        T* value_cache,
                                        T* context_buf,
                                        const bool* finished,
                                        const int* sequence_lengths,
                                        const int max_batch_size,
                                        const int inference_batch_size,
                                        const int head_num,
                                        const int size_per_head,
                                        const int rotary_embedding_dim,
                                        const int max_seq_len,
                                        const int max_input_len,
                                        const int* input_lengths,
                                        const int step,
                                        cudaStream_t stream);

template<typename T>
void fusedQKV_masked_attention_dispatch(const T* qkv_buf,
                                        const T* qkv_bias,
                                        T* key_cache,
                                        T* value_cache,
                                        T* context_buf,
                                        const bool* finished,
                                        const int* sequence_lengths,
                                        const int max_batch_size,
                                        const int inference_batch_size,
                                        const int head_num,
                                        const int size_per_head,
                                        const int max_seq_len,
                                        const int max_input_len,
                                        const int* input_lengths,
                                        const int step,
                                        cudaStream_t stream)
{
    fusedQKV_masked_attention_dispatch(qkv_buf,
                                       qkv_bias,
                                       key_cache,
                                       value_cache,
                                       context_buf,
                                       finished,
                                       sequence_lengths,
                                       max_batch_size,
                                       inference_batch_size,
                                       head_num,
                                       size_per_head,
                                       /*rotary_embedding_dim */ 0,
                                       max_seq_len,
                                       max_input_len,
                                       input_lengths,
                                       step,
                                       stream);
}

}  // namespace fastertransformer
