/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "3rdparty/trt_fused_multihead_attention/qkvToContext.h"
#include "src/fastertransformer/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "src/fastertransformer/kernels/cutlass_kernels/int8_gemm/int8_gemm.h"
#include "src/fastertransformer/layers/attention_layers/BaseAttentionLayer.h"

namespace fastertransformer {

template<typename T>
class GptContextAttentionLayer: public BaseAttentionLayer<T> {
private:
    // buffer handling
    size_t max_batch_size_ = 0;
    size_t max_seq_len_    = 0;

    // metadata
    const size_t head_num_;
    const size_t size_per_head_;
    const size_t hidden_units_;
    const size_t local_head_num_;
    const size_t local_hidden_units_;
    const size_t rotary_embedding_dim_;
    const bool   neox_rotary_style_;

    // fmha runner
    int                        sm_ = getSMVersion();
    std::unique_ptr<MHARunner> dispatcher_fp16;

    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size, size_t seq_len, bool allocate_qk_buf);
    void freeBuffer() override;

    using BaseAttentionLayer<T>::is_free_buffer_after_forward_;
    using BaseAttentionLayer<T>::is_allocate_buffer_;
    using BaseAttentionLayer<T>::cublas_wrapper_;

    bool is_qk_buf_float_;

    std::shared_ptr<CutlassFpAIntBGemmRunner<T, uint8_t>> weight_only_int8_fc_runner_;
    std::shared_ptr<CutlassInt8GemmRunner<T>>             int8_fc_runner_;

protected:
    using BaseAttentionLayer<T>::allocator_;
    using BaseAttentionLayer<T>::stream_;
    using BaseAttentionLayer<T>::sparse_;
    T*     qkv_buf_              = nullptr;
    T*     q_buf_2_              = nullptr;
    T*     k_buf_2_              = nullptr;
    T*     v_buf_2_              = nullptr;
    T*     qk_buf_               = nullptr;
    float* qk_buf_float_         = nullptr;
    T*     qkv_buf_2_            = nullptr;
    T*     qkv_buf_3_            = nullptr;
    char*  mixed_gemm_workspace_ = nullptr;
    size_t mixed_gemm_ws_bytes_  = 0;
    char*  int8_gemm_workspace_  = nullptr;
    size_t int8_gemm_ws_bytes_   = 0;

    // int8_mode_ == 0 means we don't use any mechanism related to INT8.
    // int8_mode_ == 1 for weight quantized only gemm for GPT
    // int8_mode_ == 2 for SmoothQuant O3 (per tensor scales)
    const int int8_mode_ = 0;

public:
    GptContextAttentionLayer(size_t           max_batch_size,
                             size_t           max_seq_len,
                             size_t           head_num,
                             size_t           size_per_head,
                             cudaStream_t     stream,
                             cublasMMWrapper* cublas_wrapper,
                             IAllocator*      allocator,
                             bool             is_free_buffer_after_forward,
                             bool             is_qk_buf_float,
                             bool             sparse    = false,
                             int              int8_mode = 0);

    GptContextAttentionLayer(size_t           max_batch_size,
                             size_t           max_seq_len,
                             size_t           head_num,
                             size_t           size_per_head,
                             size_t           local_head_num,
                             cudaStream_t     stream,
                             cublasMMWrapper* cublas_wrapper,
                             IAllocator*      allocator,
                             bool             is_free_buffer_after_forward,
                             bool             is_qk_buf_float,
                             bool             sparse    = false,
                             int              int8_mode = 0);

    GptContextAttentionLayer(size_t           max_batch_size,
                             size_t           max_seq_len,
                             size_t           head_num,
                             size_t           size_per_head,
                             size_t           local_head_num,
                             size_t           rotary_embedding_dim,
                             bool             neox_rotary_style_,
                             cudaStream_t     stream,
                             cublasMMWrapper* cublas_wrapper,
                             IAllocator*      allocator,
                             bool             is_free_buffer_after_forward,
                             bool             is_qk_buf_float,
                             bool             sparse    = false,
                             int              int8_mode = 0);

    GptContextAttentionLayer(GptContextAttentionLayer<T> const& attention_layer);

    virtual ~GptContextAttentionLayer();

    void
    forward(TensorMap* output_tensors, TensorMap* input_tensors, const AttentionWeight<T>* attention_weights) override;
};

}  // namespace fastertransformer
