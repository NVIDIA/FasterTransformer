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
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fastertransformer {

// format:
// 0: row major
// 1: CUBLASLT_ORDER_COL32_2R_4R4
// 2: CUBLASLT_ORDER_COL4_4R2_8C
template<typename T>
void invokeQuantizeWeight(int8_t* dst,
                          const T* src,
                          const float* amax,
                          const int n,
                          const int k,
                          const int format,
                          cudaStream_t stream,
                          const int scale_is_vector = 1);

}  // namespace fastertransformer
