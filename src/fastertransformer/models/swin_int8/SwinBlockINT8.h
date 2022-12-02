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
#include "src/fastertransformer/kernels/image_shift_partition_kernels.h"
#include "src/fastertransformer/kernels/layernorm_int8_kernels.h"
#include "src/fastertransformer/layers/FfnINT8Weight.h"
#include "src/fastertransformer/layers/attention_layers_int8/WindowAttentionINT8.h"

namespace fastertransformer {
template<typename T>
class SwinTransformerINT8Block: public BaseLayer {
private:
    int   int8_mode    = 0;
    int   window_size_ = 7;
    float mlp_ratio_   = 4.0f;
    float layernorm_eps_;
    int   version_ = 1;

    int8_t* buf_ = nullptr;

    int8_t *mlp_buf_ = nullptr, *skip_buf_ = nullptr, *attention_output_ = nullptr, *normed_shifted_input_ = nullptr;
    int8_t *mlp_output_ = nullptr, *input_int8 = nullptr;
    WindowAttentionINT8<T>* atten_ = nullptr;

    void allocateBuffer() override;
    void allocateBuffer(int batch, int input_resolution, int dim);

    void freeBuffer();

public:
    SwinTransformerINT8Block(int              int8_mode,
                             int              max_batch,
                             int              window_size,
                             float            mlp_ratio,
                             float            layernorm_eps_,
                             bool             qkv_bias,
                             cudaStream_t     stream,
                             cublasMMWrapper* cublas_wrapper,
                             IAllocator*      allocator,
                             bool             is_free_buffer_after_forward,
                             float            qk_scale = 1.0f,
                             int              version  = 1);

    ~SwinTransformerINT8Block();

    void
    forward(TensorMap* output_tensors, TensorMap* input_tensors, SwinTransformerINT8BlockWeight<T>& swin_block_weights);

};  // class SwinTransformerINT8Block
}  // namespace fastertransformer
