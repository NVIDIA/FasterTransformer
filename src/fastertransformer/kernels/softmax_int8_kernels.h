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

#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fastertransformer {

template<typename T>
void invokeSoftmaxCOL32(int8_t* output,
                        const int32_t* input,
                        const T* attr_mask,
                        const int batch_size,
                        const int head_num,
                        const int seq_len,
                        const float scalar1a,
                        const float* scalar1b,
                        const float* scalar1c,
                        const float* amax_ptr,
                        cudaStream_t stream);

template<typename T>
void invokeSoftmaxCOL32(int8_t* output,
                        const int8_t* input,
                        const T* attr_mask,
                        const int batch_size,
                        const int head_num,
                        const int seq_len,
                        const float scalar1a,
                        const float* scalar1b,
                        const float* amax_ptr,
                        cudaStream_t stream);

template<typename T>
void invokeSoftmaxWithRelPosBiasCOL32(int8_t* a_buf,
                                      int8_t* qk_buf_int8,
                                      const T* attn_mask,
                                      const T* relative_pos_bias,
                                      const int batch_size,
                                      const int num_head,
                                      const int window_num,
                                      const int window_len,
                                      const float scalar,
                                      const float* deQ_scale_ptr,
                                      const float* out_scale_ptr,
                                      cudaStream_t stream);
}  // namespace fastertransformer
