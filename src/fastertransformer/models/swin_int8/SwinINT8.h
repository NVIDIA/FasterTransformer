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
#include "src/fastertransformer/models/swin_int8/SwinBasicLayerINT8.h"
#include "src/fastertransformer/utils/allocator.h"
#include "src/fastertransformer/utils/conv2d.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {
template<typename T>
class SwinTransformerINT8 {
private:
    int int8_mode = 0;
    int max_batch_ = 1;
    int img_size_ = 224;
    int patch_size_ = 4;
    int in_chans_ = 3;
    int embed_dim_ = 96;
    int window_size_ = 7;
    int* depths_;
    int* num_heads_;
    bool ape_ = false;
    bool patch_norm_ = true;
    float mlp_ratio_ = 4.0f;
    bool qkv_bias_ = true;
    int patches_resolution_ = 56;
    int layer_num_ = 4;
    int qk_scale_ = 1.0f;
    size_t max_buf_size_ = 0;
    size_t max_basic_layer_buf_size_ = 0;
    size_t max_block_buf_size_ = 0;
    size_t max_window_attention_buf_size_ = 0;
    IAllocator* allocator_ = nullptr;
    cudnnHandle_t cudnn_handle_;
    cudaStream_t stream_;
    cublasMMWrapper* cublas_wrapper_;
    bool is_free_buffer_after_forward_;
    bool is_allocate_buffer_ = false;

    T* buf_ = nullptr;
    T* x_patch_embed_ = nullptr;
    T* basic_layer_output_ = nullptr;
    // for avgPool_ones
    T* avg_pool_ones_ = nullptr;
    T* buffer_COL32 = nullptr;

    SwinTransformerINT8BasicLayer<T>* basic_layer_ = nullptr;

    static size_t getBufSize(const int batch, const int patches_resolution, const int layer_num, const int embed_dim);

    void allocateBuffer();

    void freeBuffer();

    // input is [B, C_in, H, W]
    // output is [B, H, W, C_out]
    void patchEmbed(T* output,
                    const T* input,
                    const T* kernel,
                    const T* bias,
                    const T* gamma,
                    const T* beta,
                    const int batch,
                    const int img_size,
                    const int patch_size,
                    const int patches_resolution,
                    const int in_chans,
                    const int embed_dim,
                    const bool patch_norm);

public:
    SwinTransformerINT8(int int8_mode,
                        int max_batch,
                        int img_size,
                        int patch_size,
                        int in_chans,
                        int embed_dim,
                        int window_size,
                        int* depths,
                        int* num_heads,
                        bool ape,
                        bool patch_norm,
                        int layer_num,
                        float mlp_ratio,
                        cudnnHandle_t cudnn_handle,
                        cudaStream_t stream,
                        cublasMMWrapper* cublas_wrapper,
                        IAllocator* allocator,
                        bool is_free_buffer_after_forward,
                        bool qkv_bias = true,
                        float qk_scale = 1.0f);

    ~SwinTransformerINT8();

    void forward(std::vector<Tensor>* output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 SwinTransformerINT8Weight<T>& swin_weights);

};  // class SwinTransformerINT8
}  // namespace fastertransformer
