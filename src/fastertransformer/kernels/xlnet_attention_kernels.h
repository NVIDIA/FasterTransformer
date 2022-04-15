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

#define FINAL_MASK 0xffffffff
const float epsilon = 0.0f;
template<typename T>
void invokePrepareMatrixes(int batch_size,
                           int seq_len,
                           int hidden_dim,
                           int size_per_head,
                           T* q_buf,
                           T* q_buf_bd,
                           T* q_buf_ef,
                           T* k_buf,
                           T* k_buf_bd,
                           T* k_buf_ef,
                           T* query_buf,
                           T* key_buf,
                           T* k_head_r,
                           T* attr_seg_embed,
                           T* attr_bias_Q_w,
                           T* attr_bias_Q_r,
                           T* attr_bias_Q_s,
                           cudaStream_t stream);

template<typename T>
void invokeTranspose102(
    int batch_size, int seq_len, int head_num, T* qk_buf_ef_trans, T* qk_buf_ef, cudaStream_t stream);

template<typename T>
void invokeTranspose201(
    int batch_size, int seq_len, int head_num, T* qk_buf_ef_seg_trans, T* qk_buf_ef_seg, cudaStream_t stream);

template<typename T>
void invokeRelShiftBd(int batch_size, int head_num, int seq_len, T* qk_buf_bd_shift, T* qk_buf_bd, cudaStream_t stream);

template<typename T>
void invokeCalAttnScore(int batch_size,
                        int head_num,
                        int seq_len,
                        int size_per_head,
                        float q_scaling,
                        T* attn_score,
                        T* qk_buf,
                        T* qk_buf_bd_shift,
                        T* qk_buf_ef_seg_trans,
                        T* attn_mask,
                        T* value_buf_trans,
                        T* value_buf,
                        cudaStream_t stream);

template<typename T>
void invokeTranspose102v2(
    int batch_size, int seq_len, int head_num, int size_per_head, T* attn_vec_trans, T* attn_vec, cudaStream_t stream);

template<typename T>
void invokeAddResidualLayerNorm(int batch_size,
                                int seq_len,
                                int hidden_dim,
                                T* attn_layernorm,
                                T* attn_out,
                                const T* in_tensor,
                                const T* attr_layernorm_gamma,
                                const T* attr_layernorm_beta,
                                cudaStream_t stream);

template<typename T>
void invokeGelu(
    int batch_size, int seq_len, int hidden_dim_ff, T* output_fc1, const T* attr_fc1_bias, cudaStream_t stream);

}  // namespace fastertransformer
