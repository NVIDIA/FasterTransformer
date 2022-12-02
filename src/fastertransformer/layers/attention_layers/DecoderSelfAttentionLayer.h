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

#include "src/fastertransformer/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
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
    const float  q_scaling_;
    const size_t rotary_embedding_dim_;
    const bool   neox_rotary_style_;

    std::shared_ptr<CutlassFpAIntBGemmRunner<T, uint8_t>> weight_only_int8_fc_runner_;

    void allocateBuffer() override;
    void freeBuffer() override;
    bool isValidBatchSize(size_t batch_size);
    void allocateBuffer(size_t batch_size);

    using BaseAttentionLayer<T>::is_free_buffer_after_forward_;
    using BaseAttentionLayer<T>::is_allocate_buffer_;
    using BaseAttentionLayer<T>::cublas_wrapper_;

protected:
    T*     qkv_buf_              = nullptr;
    T*     context_buf_          = nullptr;
    char*  mixed_gemm_workspace_ = nullptr;
    size_t mixed_gemm_ws_bytes_  = 0;
    using BaseAttentionLayer<T>::stream_;
    using BaseAttentionLayer<T>::sparse_;
    using BaseAttentionLayer<T>::allocator_;

    // int8_mode_ == 0 means we don't use any mechanism related to INT8.
    // int8_mode_ == 1 for weight quantized only gemm for GPT
    // int8_mode_ == 2 for SmoothQuant O3 (per tensor scales)
    const int int8_mode_ = 0;

public:
    DecoderSelfAttentionLayer(size_t           max_batch_size,
                              size_t           head_num,
                              size_t           size_per_head,
                              size_t           local_head_num,
                              size_t           rotary_embedding_dim,
                              bool             neox_rotary_style,
                              size_t           d_model,
                              const float      q_scaling,
                              cudaStream_t     stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator*      allocator,
                              bool             is_free_buffer_after_forward,
                              bool             sparse    = false,
                              int              int8_mode = 0);

    DecoderSelfAttentionLayer(size_t           max_batch_size,
                              size_t           head_num,
                              size_t           size_per_head,
                              cudaStream_t     stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator*      allocator,
                              bool             is_free_buffer_after_forward,
                              bool             sparse    = false,
                              int              int8_mode = 0);

    DecoderSelfAttentionLayer(size_t           max_batch_size,
                              size_t           head_num,
                              size_t           size_per_head,
                              const float      q_scaling,
                              cudaStream_t     stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator*      allocator,
                              bool             is_free_buffer_after_forward,
                              bool             sparse    = false,
                              int              int8_mode = 0);

    DecoderSelfAttentionLayer(size_t           max_batch_size,
                              size_t           head_num,
                              size_t           size_per_head,
                              size_t           local_head_num,
                              cudaStream_t     stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator*      allocator,
                              bool             is_free_buffer_after_forward,
                              bool             sparse    = false,
                              int              int8_mode = 0);

    DecoderSelfAttentionLayer(size_t           max_batch_size,
                              size_t           head_num,
                              size_t           size_per_head,
                              size_t           local_head_num,
                              size_t           d_model,
                              const float      q_scaling,
                              cudaStream_t     stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator*      allocator,
                              bool             is_free_buffer_after_forward,
                              bool             sparse    = false,
                              int              int8_mode = 0);

    DecoderSelfAttentionLayer(size_t           max_batch_size,
                              size_t           head_num,
                              size_t           size_per_head,
                              size_t           local_head_num,
                              size_t           rotary_embedding_dim,
                              bool             neox_rotary_style,
                              cudaStream_t     stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator*      allocator,
                              bool             is_free_buffer_after_forward,
                              bool             sparse    = false,
                              int              int8_mode = 0);

    DecoderSelfAttentionLayer(DecoderSelfAttentionLayer<T> const& attention_layer);

    ~DecoderSelfAttentionLayer();

    void
    forward(TensorMap* output_tensors, TensorMap* input_tensors, const AttentionWeight<T>* attention_weights) override;
};

template<typename T>
void fusedQKV_masked_attention_dispatch(const T*     qkv_buf,
                                        const T*     qkv_bias,
                                        const T*     relative_attention_bias,
                                        T*           key_cache,
                                        T*           value_cache,
                                        const int*   cache_indir,
                                        T*           context_buf,
                                        const bool*  finished,
                                        const int*   sequence_lengths,
                                        const int    max_batch_size,
                                        const int    inference_batch_size,
                                        const int    beam_width,
                                        const int    head_num,
                                        const int    size_per_head,
                                        const int    rotary_embedding_dim,
                                        const bool   neox_rotary_style,
                                        const int    memory_max_len,
                                        const int*   prefix_prompt_lengths,
                                        const int    max_prefix_prompt_length,
                                        const int    max_input_len,
                                        const int*   total_padding_tokens,
                                        const int    step,
                                        const float  q_scaling,
                                        const int    relative_attention_bias_stride,
                                        const T*     linear_bias_slopes,
                                        const bool*  masked_tokens,
                                        const int*   ia3_tasks,
                                        const T*     ia3_key_weights,
                                        const T*     ia3_value_weights,
                                        const float* qkv_scale_out,
                                        const float* attention_out_scale,
                                        const int    int8_mode,
                                        cudaStream_t stream);

}  // namespace fastertransformer
