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
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fastertransformer {

void invokeTransposeCOL32(int8_t* dst,
                          const int8_t* src,
                          const int batch_size,
                          const int seq_len,
                          const int head_num,
                          const int size_per_head,
                          const float* bmm2_deQFactor,
                          const float* out_scale_ptr,
                          cudaStream_t stream);

void invokeTransposeCOL32(int8_t* dst,
                          const int* src,
                          const int batch_size,
                          const int seq_len,
                          const int head_num,
                          const int size_per_head,
                          const float* v_buf_addBias_deQFactor,
                          const float* qk_afterSM_deQFactor,
                          const float* out_scale_ptr,
                          cudaStream_t stream);

void invokeTransposeCOL32RebuildPadding(int8_t* dst,
                                        const int* src,
                                        const int* sequence_id_map,
                                        const int valid_word_num,
                                        const int batch_size,
                                        const int seq_len,
                                        const int head_num,
                                        const int size_per_head,
                                        const float* v_buf_addBias_deQFactor,
                                        const float* qk_afterSM_deQFactor,
                                        const float* out_scale_ptr,
                                        cudaStream_t stream);

void invokeTransposeCOL32RebuildPadding(int8_t* dst,
                                        const int8_t* src,
                                        const int* sequence_id_map,
                                        const int valid_word_num,
                                        const int batch_size,
                                        const int seq_len,
                                        const int head_num,
                                        const int size_per_head,
                                        const float* bmm2_deQFactor,
                                        const float* out_scale_ptr,
                                        cudaStream_t stream);

void invokeTransposeCOL32ToRow(int8_t* dst,
                               const int8_t* src,
                               const int batch_size,
                               const int seq_len,
                               const int head_num,
                               const int size_per_head,
                               const float* bmm2_deQFactor,
                               const float* out_scale_ptr,
                               cudaStream_t stream);

void invokeTransposeCOL32ToRowRebuildPadding(int8_t* dst,
                                             const int8_t* src,
                                             const int* sequence_id_map,
                                             const int valid_word_num,
                                             const int batch_size,
                                             const int seq_len,
                                             const int head_num,
                                             const int size_per_head,
                                             const float* bmm2_deQFactor,
                                             const float* out_scale_ptr,
                                             cudaStream_t stream);

}  // namespace fastertransformer
