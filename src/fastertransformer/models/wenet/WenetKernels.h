/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2022.  Authored by Yuqing Ding.
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

#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fastertransformer {

void invokeGetWenetOutLens(int* out, const int* in, const int batch_size, const int max_seq_len, cudaStream_t stream);

// not supported yet
// void invokeGetPaddingOffset(size_t* d_token_num,
//                             int* tmp_mask_offset,
//                             const int* sequence_lengths,
//                             const int batch_size,
//                             const int max_seq_len,
//                             cudaStream_t stream);

// void invokeGetBatchIDStartEnd(
//     int* bid_start_end, const int* sequence_length, const int batch_size, const int seq_len, cudaStream_t stream);

// template<typename T>
// void invokeBiasRebuildPadding(
//     T* dst, const T* src, const T* bias, const int* padding_offset, const int m, const int n, cudaStream_t stream);

template<typename T>
void invokeAddBiasMul(T* out, const T* bias, const T scale, const int m, const int n, cudaStream_t stream);

template<typename T>
void invokeCMVN(T* out, const T* in, const T* mean, const T* istd, const int m, const int n, cudaStream_t stream);

template<typename T>
void invokeSlice(
    T* out, const T* in, const int batch_size, const int seq_len, const int hidden_unit, cudaStream_t stream);

template<typename T>
void invokeTranspose0213(T*           dst,
                         T*           src,
                         const int    batch_size,
                         const int    seq_len,
                         const int    head_num,
                         const int    size_per_head,
                         cudaStream_t stream);

template<typename T>
void invokeEmbedDecoderInput(T*           out,
                             const int*   in,
                             const T*     embed_weights,
                             const T*     encoding_weights,
                             const int    vocab_size,
                             const int    max_len,
                             const int    batch_size,
                             const int    seq_len,
                             const int    hidden_units,
                             cudaStream_t stream);

template<typename T>
void invokeMaskDecoderOutput(float*       score,
                             const T*     decoder_output,
                             const int*   decoder_sequence_length,
                             const int*   decoder_input,
                             const int    batch_size,
                             const int    max_seq_len,
                             const int    vocab_size,
                             cudaStream_t stream);

template<typename T>
void invokeBuildBestIndex(int*         best_index,
                          const float* decoder_score,
                          const T*     ctc_score,
                          const T      ctc_weight,
                          const int    batch_size,
                          const int    beam_size,
                          cudaStream_t stream);

// template<typename T>
// void invokeBiasGlu(T* out, const T* in, const T* bias, const int m, const int n, cudaStream_t stream);

template<typename T>
void invokeMaskBiasGlu(T*           out,
                       const T*     in,
                       const T*     bias,
                       const int    m,
                       const int    n,
                       const T*     attr_mask,
                       const int    seq_len,
                       cudaStream_t stream);

template<typename T>
void invokeMaskBias(T*           out,
                    const T*     in,
                    const T*     bias,
                    const int    m,
                    const int    n,
                    const T*     attr_mask,
                    const int    seq_len,
                    cudaStream_t stream);

template<typename T>
void invokeGeneralScaleAddBiasResidualPreLayerNorm(T*           output,
                                                   T*           norm_output,
                                                   const T*     input,
                                                   const T*     gamma,
                                                   const T*     beta,
                                                   const T*     bias,
                                                   int          m,
                                                   int          n,
                                                   cudaStream_t stream,
                                                   int          opt_version    = 2,
                                                   float        scale_input    = 1.0f,
                                                   float        scale_residual = 1.0f);

template<typename T>
void invokeScaleAddBiasResidual(T*           output,
                                const T*     input,
                                const T*     bias,
                                const int    m,
                                const int    n,
                                cudaStream_t stream,
                                float        scale_input    = 1.0f,
                                float        scale_residual = 1.0f);

template<typename T>
void invokeConformerDepthwiseConvBiasSilu(T*           out,
                                          const T*     in,
                                          const T*     weight,
                                          const T*     bias,
                                          const int    batch_size,
                                          const int    seq_len,
                                          const int    hidden_unit,
                                          const int    kernel_size,
                                          const int    pad_size,
                                          cudaStream_t stream);

template<typename T>
void invokeConformerDepthwiseConvBias(T*           out,
                                      const T*     in,
                                      const T*     weight,
                                      const T*     bias,
                                      const int    batch_size,
                                      const int    seq_len,
                                      const int    hidden_unit,
                                      const int    kernel_size,
                                      const int    pad_size,
                                      cudaStream_t stream);

// template<typename T>
// void invokeVarLenConformerDepthwiseConvBiasSilu(T* out,
//                                                 const T* in,
//                                                 const T* weight,
//                                                 const T* bias,
//                                                 const int* bid_start_end,
//                                                 const T* bias_before_glu,
//                                                 const int m,
//                                                 const int batch_size,
//                                                 const int seq_len,
//                                                 const int hidden_unit,
//                                                 const int kernel_size,
//                                                 const int pad_size,
//                                                 cudaStream_t stream);

template<typename T>
void invokeAddQKVPBiasTranspose(T*           q_buf,
                                T*           k_buf,
                                T*           v_buf,
                                T*           Q,
                                const T*     bias_Q,
                                T*           K,
                                const T*     bias_K,
                                T*           V,
                                const T*     bias_V,
                                T*           p_buf,
                                T*           P,
                                T*           q_buf_bias_v,
                                const T*     pos_bias_u,
                                const T*     pos_bias_v,
                                const int    batch_size,
                                const int    seq_len,
                                const int    head_num,
                                const int    size_per_head,
                                cudaStream_t stream);

template<typename T, typename T_IN>
void invokeAddMaskedSoftMax(T*           buffer,
                            const T_IN*  buffer_src,
                            const T_IN*  qp_buf,
                            const T*     attr_mask,
                            const int    batch_size,
                            const int    seq_len,
                            const int    head_num,
                            const T      scalar,
                            cudaStream_t stream);

template<typename T>
void invokeAddQKVBiasTranspose(T*           q_buf,
                               T*           k_buf,
                               T*           v_buf,
                               T*           Q,
                               const T*     bias_Q,
                               T*           K,
                               const T*     bias_K,
                               T*           V,
                               const T*     bias_V,
                               const int    batch_size,
                               const int    seq_len1,
                               const int    seq_len2,
                               const int    head_num,
                               const int    size_per_head,
                               cudaStream_t stream);

template<typename T>
void invokeRepeatBeamSize(T* out, const T* in, const int m, const int n, const int beam_size, cudaStream_t stream);

template<typename T, bool IS_CROSS>
void invokeBuildDecoderAttentionMask(T*           attention_mask,
                                     const int*   sequence_lengths1,
                                     const int*   sequence_lengths2,
                                     const int    batch_size,
                                     const int    max_seq_len1,
                                     const int    max_seq_len2,
                                     cudaStream_t stream);

template<typename T>
void invokeBiasLogSoftmax(float*       log_probs,
                          const T*     logits,
                          const T*     bias,
                          const int*   lengths,
                          const size_t max_input_length,
                          const size_t batch_size,
                          const size_t vocab_size,
                          const size_t vocab_size_padded,
                          bool         batch_first,
                          cudaStream_t stream);

}  // namespace fastertransformer
