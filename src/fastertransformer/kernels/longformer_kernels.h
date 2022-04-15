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

namespace fastertransformer {

template<typename T>
size_t getInitLongformerCubStorage(const int seq_len);

template<typename T>
void invokeLocalAttnMaskShift(T* local_attn_mask, T* out, int batch_size, int seq_len, cudaStream_t stream);

template<typename T>
void invokeInitLongformerIdx(T* global_attn_mask,
                             int* seq_idx,
                             int* global_idx,
                             int* global_token_nums,
                             int seq_len,
                             int batch_size,
                             void* cub_storage,
                             cudaStream_t stream);

template<typename T>
void invokeLongformerMHASoftmax(const T* global_attn_mask,
                                const int* global_idx,
                                const int* global_token_nums,
                                void* input_ptrs,
                                const T* local_attn_mask,
                                float scaler,
                                int seq_len,
                                int head_num,
                                int batch_size,
                                int local_attn_window_size,
                                cudaStream_t stream);

}  // namespace fastertransformer