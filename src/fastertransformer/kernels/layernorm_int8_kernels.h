/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "int8_utils.cuh"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fastertransformer {

template<typename T>
void invokeAddBiasResidualLayerNormCol32(T* output,
                                         const int32_t* input1,
                                         const T* input2,
                                         const T* bias,
                                         const T* gamma,
                                         const T* beta,
                                         int m,
                                         int n,
                                         cudaStream_t stream,
                                         const float* weight_amax,
                                         const float* input1_amax_ptr);

template<typename T>
void invokeAddBiasResidualLayerNormCol32(int8_t* output,
                                         const int8_t* input1,
                                         const int8_t* input2,
                                         const T* bias,
                                         const T* gamma,
                                         const T* beta,
                                         int m,
                                         int n,
                                         cudaStream_t stream,
                                         const float* input1_deQFactor_ptr,
                                         const float* input2_deQFactor_ptr,
                                         const float* output_scale_ptr);

template<typename T>
void invokeAddBiasResidualLayerNormCol32(T* output,
                                         const int8_t* input1,
                                         const int8_t* input2,
                                         const T* bias,
                                         const T* gamma,
                                         const T* beta,
                                         int m,
                                         int n,
                                         cudaStream_t stream,
                                         const float* input1_deQFactor_ptr,
                                         const float* input2_deQFactor_ptr);

template<typename T>
void invokeAddBiasResidualLayerNormCol32_noRes(int8_t* output,
                                               int8_t* input1,
                                               T* input2,
                                               const T* bias,
                                               const T* gamma,
                                               const T* beta,
                                               int m,
                                               int n,
                                               cudaStream_t stream,
                                               const float* input1_deQFactor_ptr,
                                               const float* output_scale_ptr);

template<typename T>
void invokeAddBiasResidualLayerNormCol32_noRes(int8_t* output,
                                               int32_t* input1,
                                               T* input2,
                                               const T* bias,
                                               const T* gamma,
                                               const T* beta,
                                               int m,
                                               int n,
                                               cudaStream_t stream,
                                               const float* weight_amax,
                                               const float* input1_amax_ptr,
                                               const float* output_scale_ptr);

template<typename T>
void invokeLayernormCol32(int8_t* out,
                          const T* input,
                          const T* gamma,
                          const T* beta,
                          int m,
                          int n,
                          const float* norm_scale_ptr,
                          cudaStream_t stream);

template<typename T>
void invokeLayernormShiftPartitionCol32(int8_t* out,
                                        const T* input,
                                        const T* gamma,
                                        const T* beta,
                                        int batch,
                                        int H,
                                        int W,
                                        int n,
                                        const float* norm_scale_ptr,
                                        int shift_size,
                                        int window_size,
                                        cudaStream_t stream);

template<typename T>
void invokeMergeLayerNormCol32(int8_t* output,
                               const T* input,
                               const T* gamma,
                               const T* beta,
                               int batch,
                               const float* merge_inFactor,
                               int H,
                               int W,
                               int n,
                               cudaStream_t stream);

template<typename T>
void invokeAddBiasResidualLayerNormRow(int8_t* output,
                                       const int8_t* input1,
                                       const int8_t* input2,
                                       const T* bias,
                                       const T* gamma,
                                       const T* beta,
                                       int m,
                                       int n,
                                       cudaStream_t stream,
                                       const float* input1_deQFactor_ptr,
                                       const float* input2_deQFactor_ptr,
                                       const float* output_scale_ptr);

template<typename T>
void invokeAddBiasResidualLayerNormRow(T* output,
                                       const int8_t* input1,
                                       const int8_t* input2,
                                       const T* bias,
                                       const T* gamma,
                                       const T* beta,
                                       int m,
                                       int n,
                                       cudaStream_t stream,
                                       const float* input1_deQFactor_ptr,
                                       const float* input2_deQFactor_ptr);

}  // namespace fastertransformer
