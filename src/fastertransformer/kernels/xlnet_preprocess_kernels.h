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
#include "src/fastertransformer/utils/cuda_utils.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fastertransformer {

template<typename T>
void blockAttnMask(dim3& grid, dim3& block, int batch_size, int seq_len);

template<typename T>
void genWordEmdK(
    int batch_size, int seq_len, int hidden_dim, T* word_emb_k, T* params_word_emb_k, int* inp_k, cudaStream_t stream);

template<typename T>
void preProcess(int batch_size,
                int seq_len,
                int hidden_dim,
                T* attn_mask,
                float* input_mask,
                T* seg_mat,
                int* seg_id,
                T* attr_k_head_r,
                cudaStream_t stream);
}  // namespace fastertransformer
