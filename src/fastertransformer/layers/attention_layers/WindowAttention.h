/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/kernels/reverse_roll_kernels.h"
#include "src/fastertransformer/kernels/transform_mask_kernels.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"
#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/layers/attention_layers/BaseAttentionLayer.h"
#include "src/fastertransformer/models/swin/SwinWeight.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <memory>

#define TRT_MAX_LEN 384

namespace fastertransformer {
template<typename T>
class WindowAttention: public BaseAttentionLayer<T> {

private:
    int max_batch_ = 1;
    int dim_ = 96;
    int num_head_ = 2;
    int window_size_ = 7;
    int head_dim_ = 48;
    int input_resolution_ = 56;
    int window_len_ = 49;
    int embed_dim_ = 96;
    int window_num_ = 64;
    int size_per_head_;
    bool qkv_bias_ = true;
    float qk_scale_ = 1.0f;
    bool use_trt_ = false;

    void allocateBuffer() override;
    void freeBuffer() override;

    int dispatcher_fp16_num_head_ = -1;
    std::unique_ptr<MHARunner> dispatcher_fp16_;

    using BaseAttentionLayer<T>::stream_;
    using BaseAttentionLayer<T>::is_free_buffer_after_forward_;
    using BaseAttentionLayer<T>::is_allocate_buffer_;
    using BaseAttentionLayer<T>::cublas_wrapper_;
    using BaseAttentionLayer<T>::allocator_;

    T *buf_ = nullptr, *qkv_buf_ = nullptr;
    T *q_buf_ = nullptr, *k_buf_ = nullptr, *v_buf_ = nullptr, *qk_buf_ = nullptr;

    half *trt_attention_mask_ = nullptr, *trt_relative_position_bias_ = nullptr;

public:
    static int trt_getS(const int actual_seqlen);

    static size_t roundByteSize(const size_t size, const int factor);

    static size_t
    getBufSize(const int batch, const int num_head, const int window_num, const int window_len, const int dim);

    WindowAttention(int max_batch,
                    int window_size,
                    cudaStream_t stream,
                    cublasMMWrapper* cublas_wrapper,
                    IAllocator* allocator,
                    bool is_free_buffer_after_forward = false,
                    bool qkv_bias = true,
                    float qk_scale = 1.0f);

    ~WindowAttention();

    void forward(std::vector<fastertransformer::Tensor>* output_tensors,
                 const std::vector<fastertransformer::Tensor>* input_tensors,
                 const AttentionWeight<T>* attention_weights);

};  // class WindowAttention
}  // namespace fastertransformer
