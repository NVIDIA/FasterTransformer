/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/utils/Tensor.h"

namespace fastertransformer {

template<typename T>
void invokeAddQKBiasTransposeRepeat(T*           q_buf,
                                    T*           k_buf,
                                    T*           Q,
                                    const T*     bias_Q,
                                    T*           K,
                                    const T*     bias_K,
                                    const int    batch_size,
                                    const int    attention_span,
                                    const int    head_num,
                                    const int    size_per_head,
                                    cudaStream_t stream);

template<typename T>
void invokeDisentangledAttention(
    T* result, T* c2c, T* c2p, T* p2c, const int batch_dim, const int seq_dim, const int span, cudaStream_t stream);

}  // namespace fastertransformer
