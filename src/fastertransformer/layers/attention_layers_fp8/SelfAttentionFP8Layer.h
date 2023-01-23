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

#include "3rdparty/trt_fused_multihead_attention/qkvToContext.h"
#include "src/fastertransformer/layers/attention_layers_fp8/BaseAttentionFP8Layer.h"

namespace fastertransformer {

template<typename T1, typename T2>
class SelfAttentionFP8Layer: public BaseAttentionFP8Layer<T1, T2> {
private:
    // metadata
    size_t head_num_;
    size_t size_per_head_;
    size_t hidden_units_;
    size_t d_model_;
    bool   sparse_;
    float  q_scaling_;
    int    fp8_mode_;
    int    sm_;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t seq_len);

protected:
    using BaseAttentionFP8Layer<T1, T2>::stream_;
    using BaseAttentionFP8Layer<T1, T2>::is_free_buffer_after_forward_;
    using BaseAttentionFP8Layer<T1, T2>::is_allocate_buffer_;
    using BaseAttentionFP8Layer<T1, T2>::cublas_wrapper_;
    using BaseAttentionFP8Layer<T1, T2>::allocator_;

    std::unique_ptr<MHARunner> dispatcher_fp8;

    T1* qkv_buf_       = nullptr;
    T2* qk_buf_bfloat_ = nullptr;

    T1* q_buf_2_ = nullptr;
    T1* k_buf_2_ = nullptr;
    T1* v_buf_2_ = nullptr;

    T1* qk_buf_    = nullptr;
    T1* qkv_buf_2_ = nullptr;
    T1* qkv_buf_3_ = nullptr;
    T2* qkv_buf_4_ = nullptr;

public:
    SelfAttentionFP8Layer(size_t           head_num,
                          size_t           size_per_head,
                          size_t           d_model,
                          float            q_scaling,
                          int              fp8_mode,
                          int              sm,
                          cudaStream_t     stream,
                          cublasMMWrapper* cublas_wrapper,
                          IAllocator*      allocator,
                          bool             is_free_buffer_after_forward,
                          bool             sparse = false);

    SelfAttentionFP8Layer(SelfAttentionFP8Layer<T1, T2> const& attention_layer);

    ~SelfAttentionFP8Layer();

    void forward(TensorMap*                        output_tensors,
                 TensorMap*                        input_tensors,
                 const AttentionFP8Weight<T1, T2>* attention_weights) override;
};

}  // namespace fastertransformer
