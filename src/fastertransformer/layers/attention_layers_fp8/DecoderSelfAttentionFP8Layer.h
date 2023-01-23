/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "src/fastertransformer/layers/attention_layers_fp8/AttentionFP8Weight.h"
#include "src/fastertransformer/utils/cublasFP8MMWrapper.h"
#include "src/fastertransformer/utils/cuda_fp8_utils.h"

namespace fastertransformer {

template<typename T1, typename T2>
class DecoderSelfAttentionFP8Layer: public BaseAttentionLayer<T1> {
private:
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

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size);

    using BaseAttentionLayer<T1>::is_free_buffer_after_forward_;
    using BaseAttentionLayer<T1>::is_allocate_buffer_;
    using BaseAttentionLayer<T1>::cublas_wrapper_;
    using BaseAttentionLayer<T1>::allocator_;

protected:
#ifdef FP8_MHA
    T1* qkv_buf_ = nullptr;
#else
    T2* qkv_buf_ = nullptr;
#endif
    T2* high_precision_context_buf_ = nullptr;
    T1* context_buf_                = nullptr;
    using BaseAttentionLayer<T1>::stream_;
    using BaseAttentionLayer<T1>::sparse_;

    T2* bf16_key_cache   = nullptr;
    T2* bf16_value_cache = nullptr;

public:
    DecoderSelfAttentionFP8Layer(size_t           head_num,
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
                                 bool             sparse = false);

    DecoderSelfAttentionFP8Layer(DecoderSelfAttentionFP8Layer<T1, T2> const& attention_layer);

    ~DecoderSelfAttentionFP8Layer();

    void
    forward(TensorMap* output_tensors, TensorMap* input_tensors, const AttentionWeight<T1>* attention_weights) override;
};

}  // namespace fastertransformer
