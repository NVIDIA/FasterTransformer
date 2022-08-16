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
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fastertransformer {

template<typename T>
void invokeAddBiasGelu(T* out, const T* bias, const int m, const int n, cudaStream_t stream);

template<typename T>
void invokeAddBiasGatedGelu(
    T* hidden1, const T* hidden2, const T* bias1, const T* bias2, const int m, const int n, cudaStream_t stream);

template<typename T>
void invokeAddBiasRelu(T* out, const T* bias, const int m, const int n, cudaStream_t stream);

template<typename T>
void invokeAddBiasGatedRelu(
    T* hidden1, const T* hidden2, const T* bias1, const T* bias2, const int m, const int n, cudaStream_t stream);

template<typename F_T, typename B_T>
void invokeAddBias(F_T* out, const B_T* bias, const int m, const int n, cudaStream_t stream);

template<typename T>
void invokeAddBiasGeluV2(T* out, const T* bias, const int m, const int n, cudaStream_t stream);

template<typename T>
void invokeAddBiasGatedSilu(
    T* hidden1, const T* hidden2, const T* bias1, const T* bias2, const int m, const int n, cudaStream_t stream);

template<typename T>
void invokeAddBiasSilu(T* out, const T* bias, const int m, const int n, cudaStream_t stream);

}  // namespace fastertransformer
