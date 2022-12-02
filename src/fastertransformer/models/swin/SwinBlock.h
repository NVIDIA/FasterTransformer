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
#include "src/fastertransformer/kernels/image_shift_partition_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/attention_layers/WindowAttention.h"

namespace fastertransformer {

template<typename T>
class SwinTransformerBlock: public BaseLayer {
private:
    const int   max_batch_   = 1;
    const int   window_size_ = 7;
    const float mlp_ratio_   = 4.0f;
    const bool  qkv_bias_    = true;
    const float qk_scale_    = 1.0f;
    const float layernorm_eps_;
    const int   version_ = 1;

    T* buf_ = nullptr;

    T *mlp_buf_ = nullptr, *normed_attn_out_buf_ = nullptr, *attention_output_ = nullptr,
      *normed_shifted_input_ = nullptr;

    WindowAttention<T>* atten_ = nullptr;

public:
    void allocateBuffer();

    void freeBuffer();

    void allocateBuffer(int batch, int input_resolution, int dim);

    SwinTransformerBlock(int              max_batch,
                         int              window_size,
                         float            mlp_ratio,
                         float            layernorm_eps_,
                         cudaStream_t     stream,
                         cublasMMWrapper* cublas_wrapper,
                         IAllocator*      allocator,
                         bool             is_free_buffer_after_forward,
                         bool             qkv_bias,
                         float            qk_scale = 1.0f,
                         int              version  = 1);

    ~SwinTransformerBlock();

    void
    forward(TensorMap* output_tensors, TensorMap* input_tensors, SwinTransformerBlockWeight<T>& swin_block_weights);

};  // class SwinTransformerBlock
}  // namespace fastertransformer
