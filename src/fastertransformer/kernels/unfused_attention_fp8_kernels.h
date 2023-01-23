/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/utils/cuda_fp8_utils.h"
#include <cuda_runtime.h>
#include <stdint.h>

namespace fastertransformer {

template<typename T1, typename T2>
struct FP8AddFusedQKVBiasRebuildPaddingParam {
    T1*          q_buf;
    T1*          k_buf;
    T1*          v_buf;
    T1*          QKV_T1;
    T2*          QKV_T2;
    const T2*    qkv_bias;
    const float* input_scale;
    const float* input_scale_2;      // TODO(bhsueh) remove this param
    const float* input_scale_2_min;  // TODO(bhsueh) remove this param
    const float* output_scale;
    const int*   padding_offset;  // TODO(bhsueh) remove this param
    const int*   padding_offset_prefix_sum;

#ifdef FP8_MHA
    T1* v_cache;
#else
    T2* v_cache;
#endif

    const uint32_t token_num;  // TODO(bhsueh) remove this param
    const uint32_t batch_size;
    const uint32_t seq_len;
    const uint32_t seq_len_padded;
    const uint32_t max_seq_len;
    const uint32_t head_num;
    const uint32_t size_per_head;
    const uint32_t rotary_embedding_dim;
    cudaStream_t   stream;
};

template<typename T1, typename T2>
void invokeFP8AddFusedQKVBiasRebuildPadding(FP8AddFusedQKVBiasRebuildPaddingParam<T1, T2> param);

template<typename T1, typename T2>
struct FP8Transpose4dBatchMajorParam {
    T2*            k_dst;
    T2*            v_dst;
    const T1*      k_src;
    const T1*      v_src;
    const float*   scale;
    const uint32_t local_batch_size;
    const uint32_t seq_len;
    const uint32_t max_seq_len;
    const uint32_t size_per_head;
    const uint32_t local_head_num;
    const uint32_t seq_len_padded;
    cudaStream_t   stream;
};

template<typename T1, typename T2>
void invokeFP8Transpose4dBatchMajor(FP8Transpose4dBatchMajorParam<T1, T2> param);

template<typename T, typename T_IN>
struct FP8MaskedSoftMaxParam {
    T*             buffer;
    const T_IN*    buffer_src;
    const T*       attr_mask;
    const int*     padding_offset_prefix_sum;
    const uint32_t batch_size;
    const uint32_t seq_len;
    const uint32_t head_num;
    const float    scalar;
    const float*   input_scale;
    const float*   output_scale;
    cudaStream_t   stream;
};

template<typename T, typename T_IN>
void invokeFP8MaskedSoftMax(FP8MaskedSoftMaxParam<T, T_IN> param);

template<typename T1, typename T2>
struct FP8AddQKVBiasRebuildPaddingParam {
    T1*          q_buf;
    T1*          k_buf;
    T1*          v_buf;
    const T1*    QKV;
    const T2*    qkv_bias;
    const int    batch_size;
    const int    seq_len;
    const int    head_num;
    const int    size_per_head;
    const int    valid_word_num;
    const int*   mask_offset;
    cudaStream_t stream;
};

template<typename T1, typename T2>
void invokeFP8AddQKVBiasRebuildPadding(FP8AddQKVBiasRebuildPaddingParam<T1, T2> param);

template<typename T1, typename T2>
struct FP8TrtAddQKVBiasParam {
    T1*          qkv_tgt;
    const T1*    qkv_src;
    const T2*    qkv_bias;
    const float* input_scale;
    const float* output_scale;
    const size_t valid_word_num;
    const size_t head_num;
    const size_t size_per_head;
    const size_t hidden_unit;
    cudaStream_t stream;
};

template<typename T1, typename T2>
void invokeFP8TrtAddQKVBias(FP8TrtAddQKVBiasParam<T1, T2> param);

template<typename T_IN, typename T_OUT>
struct FP8TransposeAttentionOutRemovePaddingParam {
    T_OUT*       dst;
    const T_IN*  src;
    const float* scale;
    const int    valid_word_num;
    const int    batch_size;
    const int    seq_len;
    const int    head_num;
    const int    size_per_head;
    const int*   padding_offset;
    cudaStream_t stream;
};

template<typename T_IN, typename T_OUT>
void invokeFP8TransposeAttentionOutRemovePadding(FP8TransposeAttentionOutRemovePaddingParam<T_IN, T_OUT> param);

void invokeTmpHanldKCache(__nv_bfloat16* dst_k,
                          __nv_fp8_e4m3* src_k,
                          const float*   scale,
                          int            batch_size,
                          int            seq_len,
                          int            padded_seq_len,
                          int            head_num,
                          int            size_per_head,
                          cudaStream_t   stream);

void invokeTmpHanldVCache(__nv_bfloat16* dst_v,
                          __nv_fp8_e4m3* src_v,
                          const float*   scale,
                          int            batch_size,
                          int            seq_len,
                          int            padded_seq_len,
                          int            head_num,
                          int            size_per_head,
                          cudaStream_t   stream);

}  // namespace fastertransformer
