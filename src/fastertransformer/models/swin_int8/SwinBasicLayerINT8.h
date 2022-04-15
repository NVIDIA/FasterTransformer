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

#include "src/fastertransformer/models/swin_int8/SwinBlockINT8.h"

namespace fastertransformer {
template<typename T>
class SwinTransformerINT8BasicLayer: public BaseLayer {
private:
    int int8_mode = 0;
    int max_batch_ = 1;
    int patches_resolution_ = 64;
    int embed_dim_ = 96;
    int window_size_ = 7;
    float mlp_ratio_ = 4.0f;
    bool qkv_bias_ = true;
    float qk_scale_ = 1.0f;
    const size_t max_buf_size_ = 0;

    // T* buf_ = nullptr;
    T* block_output_ = nullptr;
    int8_t* gemm_out_buf_ = nullptr;
    SwinTransformerINT8Block<T>* block_ = nullptr;

    void allocateBuffer();

    void freeBuffer();

    // input is [B, H, W, C]
    // merge_layernorm_buf is [B, H/2, W/2, 4*C]
    // output is [B, H/2, W/2, 2*C]
    void patchMerge(T* output,
                    int8_t* gemm_out_buf,
                    int8_t* merge_layernorm_buf,
                    const T* input,
                    const T* gamma,
                    const T* beta,
                    const int8_t* weight,
                    int batch,
                    const ScaleList* scalePtr,
                    int input_resolution,
                    int dim,
                    int sm);

public:
    // dim & input_resolution will be used to malloc the max buf size
    SwinTransformerINT8BasicLayer(int int8_mode,
                                  int max_batch,
                                  int window_size,
                                  int patches_resolution,
                                  int embed_dim,
                                  float mlp_ratio,
                                  bool qkv_bias,
                                  float qk_scale,
                                  cudaStream_t stream,
                                  cublasMMWrapper* cublas_wrapper,
                                  IAllocator* allocator,
                                  bool is_free_buffer_after_forward);

    ~SwinTransformerINT8BasicLayer();

    void forward(std::vector<Tensor>* output_tensors,
                 std::vector<Tensor>* input_tensors,
                 SwinTransformerINT8BasicLayerWeight<T>& swin_basic_layer_weights);

};  // class SwinTransformerINT8BasicLayer
}  // namespace fastertransformer
