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
#include "src/fastertransformer/utils/cuda_utils.h"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fastertransformer {

template<typename T>
void invokeShiftPartition(
    T* out, const T* input, int batch, int H, int W, int n, int shift_size, int window_size, cudaStream_t stream);

template<typename T>
void invokeShiftPartitionCol32(int8_t*      out,
                               const T*     input,
                               int          batch,
                               int          H,
                               int          W,
                               int          n,
                               const float* scale_ptr,
                               int          shift_size,
                               int          window_size,
                               cudaStream_t stream);

}  // namespace fastertransformer
