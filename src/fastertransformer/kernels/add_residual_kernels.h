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

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fastertransformer {

template<typename T>
void invokeAddBiasResidual(T* output, const T* input, const T* bias, const int m, const int n, cudaStream_t stream);

template<typename T>
void invokeAddBiasAttentionFfnResidual(
  T* block_output,
  const T* ffn_output,
  const T* attn_output,
  const T* block_input,
  const T* bias,
  const int m,
  const int n,
  cudaStream_t stream
);

}  // namespace fastertransformer
