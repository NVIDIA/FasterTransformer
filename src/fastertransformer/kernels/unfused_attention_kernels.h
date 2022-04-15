/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
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
void invokeAddQKVBiasTranspose(T* q_buf,
                               T* k_buf,
                               T* v_buf,
                               T* Q,
                               const T* bias_Q,
                               T* K,
                               const T* bias_K,
                               T* V,
                               const T* bias_V,
                               const int batch_size,
                               const int seq_len,
                               const int head_num,
                               const int size_per_head,
                               cudaStream_t stream);

template<typename T, typename T_IN>
void invokeMaskedSoftMax(T* buffer,
                         const T_IN* buffer_src,
                         const T* attr_mask,
                         const int batch_size,
                         const int seq_len,
                         const int head_num,
                         const T scalar,
                         cudaStream_t stream);

template<typename T>
void invokeTransposeQKV(T* dst,
                        T* src,
                        const int batch_size,
                        const int seq_len,
                        const int head_num,
                        const int size_per_head,
                        cudaStream_t stream);

template<typename T>
void invokeAddQKVBiasRebuildPadding(T* Q,
                                    const T* bias_Q,
                                    T* K,
                                    const T* bias_K,
                                    T* V,
                                    const T* bias_V,
                                    T* q_buf,
                                    T* k_buf,
                                    T* v_buf,
                                    const int batch_size,
                                    const int seq_len,
                                    const int head_num,
                                    const int size_per_head,
                                    const int valid_word_num,
                                    const int* mask_offset,
                                    cudaStream_t stream);

template<typename T>
void invokeTransposeAttentionOutRemovePadding(T* src,
                                              T* dst,
                                              const int valid_word_num,
                                              const int batch_size,
                                              const int seq_len,
                                              const int head_num,
                                              const int size_per_head,
                                              const int* mask_offset,
                                              cudaStream_t stream);

template<typename T>
void invokeAddFusedQKVBiasTranspose(T* q_buf,
                                    T* k_buf,
                                    T* v_buf,
                                    T* QKV,
                                    const T* qkv_bias,
                                    const int batch_size,
                                    const int seq_len,
                                    const int head_num,
                                    const int size_per_head,
                                    cudaStream_t stream)
{
    invokeAddFusedQKVBiasTranspose(
        q_buf, k_buf, v_buf, QKV, qkv_bias, batch_size, seq_len, head_num, size_per_head, 0, stream);
}

template<typename T>
void invokeAddFusedQKVBiasTranspose(T* q_buf,
                                    T* k_buf,
                                    T* v_buf,
                                    T* QKV,
                                    const T* qkv_bias,
                                    const int batch_size,
                                    const int seq_len,
                                    const int head_num,
                                    const int size_per_head,
                                    const int rotary_embedding_dim,
                                    cudaStream_t stream);

template<typename T>
void invokeTranspose4d(T* dst,
                       T* src,
                       const int local_batch_size,
                       const int seq_len,
                       const int size_per_head,
                       const int local_hidden_units,
                       const int local_head_num,
                       const int batch_size,
                       const int ite,
                       cudaStream_t stream);

template<typename T>
void invokeTranspose4dBatchMajor(T* k_dst,
                                 T* v_dst,
                                 const T* k_src,
                                 const T* v_src,
                                 const int local_batch_size,
                                 const int seq_len,
                                 const int max_seq_len,
                                 const int size_per_head,
                                 const int local_head_num,
                                 cudaStream_t stream);

template<typename T>
void invokeAddRelativeAttentionBias(T* qk_buf,
                                    const T* relative_attention_bias,
                                    const int batch_size,
                                    const int head_num,
                                    const int seq_len,
                                    cudaStream_t stream);

template<typename T>
void invokeAddHead3SizeQKVBias(const T* mm_qkv,
                               const T* bias_qkv,
                               T* q_buf_,
                               T* k_buf_,
                               T* v_buf_,
                               const int batch,
                               const int window_num,
                               const int window_len,
                               const int head_num,
                               const int size_per_head,
                               cudaStream_t stream);

template<typename T>
void invokeMaskedSoftMaxWithRelPosBias(T* qk_buf,
                                       const T* attn_mask,
                                       const T* relative_pos_bias,
                                       const int batch_size,
                                       const int num_head,
                                       const int window_num,
                                       const int window_len,
                                       const float qk_scale,
                                       cudaStream_t stream);

}  // namespace fastertransformer
