/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fastertransformer {

enum LayerNormType {
    pre_layernorm,
    post_layernorm
};

template<typename T>
struct LayerNormWeight {
    const T* gamma = nullptr;
    const T* beta = nullptr;
};

template<typename T>
void invokeAddBiasResidualLayerNorm(T* out,
                                    const T* input,
                                    const T* bias,
                                    const T* gamma,
                                    const T* beta,
                                    const int m,
                                    const int n,
                                    cudaStream_t stream);

template<typename T>
void invokeGeneralAddBiasResidualPreLayerNorm(T* output,
                                              T* norm_output,
                                              const T* input,
                                              const T* gamma,
                                              const T* beta,
                                              const T* bias,
                                              int m,
                                              int n,
                                              cudaStream_t stream,
                                              int opt_version = 2);

template<typename T>
void invokeGeneralLayerNorm(T* out,
                            const T* input,
                            const T* gamma,
                            const T* beta,
                            const int m,
                            const int n,
                            cudaStream_t stream,
                            int opt_version = 2);

template<typename T>
void invokeGeneralT5LayerNorm(
    T* out, const T* input, const T* gamma, const T* beta, const int m, const int n, cudaStream_t stream);

template<typename T>
void invokeGeneralAddResidualT5PreLayerNorm(
    T* output, T* norm_output, const T* input, const T* gamma, int m, int n, cudaStream_t stream);

template<typename T>
void invokeGeneralAddBiasResidualT5PreLayerNorm(T* output,
                                                T* norm_output,
                                                const T* input,
                                                const T* gamma,
                                                const T* beta,
                                                const T* bias,
                                                int m,
                                                int n,
                                                cudaStream_t stream);

template<typename T>
void invokeLayernormShiftPartition(T* out,
                                   const T* input,
                                   const T* gamma,
                                   const T* beta,
                                   int batch,
                                   int H,
                                   int W,
                                   int n,
                                   int shift_size,
                                   int window_size,
                                   cudaStream_t stream);

template<typename T>
void invokeAddBiasLayernorm(
    T* out, const T* bias, const T* gamma, const T* beta, int m, int n, cudaStream_t stream, int opt_version = 2);

template<typename T>
void invokeMergeLayernorm(
    T* output, const T* input, const T* gamma, const T* beta, int batch, int H, int W, int n, cudaStream_t stream);

}  // namespace fastertransformer
