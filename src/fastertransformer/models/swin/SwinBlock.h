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

#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/attention_layers/WindowAttention.h"

namespace fastertransformer {

template<typename T>
class SwinTransformerBlock: public BaseLayer {
private:
    int max_batch_ = 1;
    int window_size_ = 7;
    int window_len_ = 49;
    int window_num_ = 64;
    int embed_dim_ = 96;
    float mlp_ratio_ = 4.0f;
    bool qkv_bias_ = true;
    float qk_scale_ = 1.0f;

    T* buf_ = nullptr;

    T *mlp_buf_ = nullptr, *normed_attn_out_buf_ = nullptr, *attention_output_ = nullptr,
      *normed_shifted_input_ = nullptr;

    WindowAttention<T>* atten_ = nullptr;

public:
    static size_t getBufSize(const int batch,
                             const int input_resolution,
                             const int mlp_dim,
                             const int window_num,
                             const int window_len,
                             const int dim)
    {
        // normed_shifted_partition_input || mlp_buf_
        size_t buf_size = batch * input_resolution * input_resolution * mlp_dim * sizeof(T) +
                          // attention_output_
                          batch * window_num * window_len * dim * sizeof(T) +
                          // skip_buf_
                          batch * input_resolution * input_resolution * dim * sizeof(T);
        return (buf_size + 31) / 32 * 32;
    }

    void allocateBuffer();

    void freeBuffer();

    SwinTransformerBlock(int max_batch,
                         int window_size,
                         float mlp_ratio,
                         cudaStream_t stream,
                         cublasMMWrapper* cublas_wrapper,
                         IAllocator* allocator,
                         bool is_free_buffer_after_forward,
                         bool qkv_bias,
                         float qk_scale = 1.0f);

    ~SwinTransformerBlock();

    void forward(std::vector<Tensor>* output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 SwinTransformerBlockWeight<T>& swin_block_weights);

};  // class SwinTransformerBlock
}  // namespace fastertransformer
