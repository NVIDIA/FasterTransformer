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

#include "src/fastertransformer/kernels/activation_int8_kernels.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/dequantize_kernels.h"
#include "src/fastertransformer/kernels/layernorm_int8_kernels.h"
#include "src/fastertransformer/layers/FfnINT8Weight.h"
#include "src/fastertransformer/layers/attention_layers_int8/WindowAttentionINT8.h"

namespace fastertransformer {
template<typename T>
class SwinTransformerINT8Block: public BaseLayer {
private:
    int int8_mode = 0;
    int max_batch_ = 1;
    int window_size_ = 7;
    int window_len_ = 49;
    int patches_resolution_ = 56;
    int embed_dim_ = 96;
    float mlp_ratio_ = 4.0f;
    bool qkv_bias_ = true;
    float qk_scale_ = 1.0f;
    size_t max_buf_size_ = 0;

    int8_t* buf_ = nullptr;

    int8_t *mlp_buf_ = nullptr, *skip_buf_ = nullptr, *attention_output_ = nullptr, *normed_shifted_input_ = nullptr;
    int8_t *mlp_output_ = nullptr, *input_int8 = nullptr;
    WindowAttentionINT8<T>* atten_ = nullptr;

    void allocateBuffer();

    void freeBuffer();

public:
    SwinTransformerINT8Block(int int8_mode,
                             int max_batch,
                             int window_size,
                             int patches_resolution,
                             int embed_dim,
                             float mlp_ratio,
                             bool qkv_bias,
                             cudaStream_t stream,
                             cublasMMWrapper* cublas_wrapper,
                             IAllocator* allocator,
                             bool is_free_buffer_after_forward,
                             float qk_scale = 1.0f);

    ~SwinTransformerINT8Block();

    void forward(std::vector<Tensor>* output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 SwinTransformerINT8BlockWeight<T>& swin_block_weights);

};  // class SwinTransformerINT8Block
}  // namespace fastertransformer
