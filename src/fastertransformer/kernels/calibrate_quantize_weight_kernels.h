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

#include "int8_utils.cuh"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fastertransformer {

template<typename T>
void invokeLdnCalibrateWeightPerChannel(float* scale, const T* src, const int k, const int n, cudaStream_t stream);

template<typename T>
void invokeLdkCalibrateQuantizeWeightPerChannel(
    int8_t* dst, float* scale, const T* src, const int n, const int k, cudaStream_t stream);

template<typename T>
void invokeLdnTransposeQuantizeWeightPerChannel(
    int8_t* dst, const float* scale, const T* src, const int k, const int n, cudaStream_t stream);

}  // namespace fastertransformer
