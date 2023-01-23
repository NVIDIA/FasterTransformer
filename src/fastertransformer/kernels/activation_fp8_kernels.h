/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/utils/cuda_fp8_utils.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fastertransformer {

template<typename T1, typename T2>
struct FP8ActivationParam {
    T2*                in;
    T1*                out;
    T2 const*          bias;
    float const* const input_scale;
    float const* const input_scale_2;
    float const* const input_scale_2_min;
    float const* const output_scale;
    const uint32_t     m;
    const uint32_t     n;
    cudaStream_t       stream;
};

template<typename T1, typename T2>
void invokeFP8AddBiasGelu(FP8ActivationParam<T1, T2> param);

template<typename T1, typename T2>
void invokeFP8AddBiasRelu(FP8ActivationParam<T1, T2> param);

// template<typename T1, typename T2>
// void invokeFP8AddBias(FP8ActivationParam<T1, T2> param);

}  // namespace fastertransformer
