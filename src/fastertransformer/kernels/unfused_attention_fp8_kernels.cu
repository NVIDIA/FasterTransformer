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

#include "src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"
#include "src/fastertransformer/kernels/unfused_attention_fp8_kernels.h"
#include "src/fastertransformer/utils/cuda_bf16_fallbacks.cuh"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
struct Vec_t {
};
template<>
struct Vec_t<float> {
    using Type = float2;
};
template<>
struct Vec_t<half> {
    using Type = half2;
};

template<>
struct Vec_t<__nv_fp8_e4m3> {
    using Type = __nv_fp8x2_e4m3;
};

template<>
struct Vec_t<__nv_bfloat16> {
    using Type = nv_bfloat162;
};

template<typename Vec1, typename Vec2>
__device__ __inline__ Vec1 convert_vec(Vec2 in_vec)
{
    return Vec1(in_vec);
}

template<>
__device__ __inline__ float2 convert_vec(half2 in_vec)
{
    return float2{(float)in_vec.x, (float)in_vec.y};
}

template<>
__device__ __inline__ float2 convert_vec(__nv_bfloat162 in_vec)
{
    return float2{(float)in_vec.x, (float)in_vec.y};
}

template<>
__device__ __inline__ __nv_bfloat162 convert_vec(half2 in_vec)
{
    __nv_bfloat162 out = cuda_cast<__nv_bfloat162, half2>(in_vec);
    return out;
}

#ifdef ENABLE_FP8
template<>
__device__ __inline__ float2 convert_vec(__nv_fp8x2_e4m3 in_vec)
{
    return (float2)in_vec;
}

template<>
__device__ __inline__ __nv_bfloat162 convert_vec(__nv_fp8x2_e4m3 in_vec)
{
    return fp8x2_e4m3_to_bfloat2(&in_vec);
}
#endif

// This optimization has bug
// #define OPT_TRANSPOSE

#ifdef OPT_TRANSPOSE
template<typename T1, typename T2, int SEQ_GROUP_SIZE, int SIZE_PER_HEAD>
__global__ void FP8AddFusedQKVBiasRebuildPaddingKernel(FP8AddFusedQKVBiasRebuildPaddingParam<T1, T2> param)
{
    using T1_4 = __nv_fp8x4_e4m3;
    using T2_2 = typename TypeConverter<T2>::Type;

    __shared__ T2 src_v[SEQ_GROUP_SIZE * (SIZE_PER_HEAD + 2)];
    T2_2*         src_v_2 = (T2_2*)(src_v);

    const int batch_idx = blockIdx.x;
    const int seq_idx   = blockIdx.y * SEQ_GROUP_SIZE + threadIdx.y;
    const int head_idx  = blockIdx.z;
    const int tidx      = threadIdx.x;
    int       Dh_div_4  = param.size_per_head / 4;
    bool      is_valid_seq_idx =
        param.padding_offset_prefix_sum == nullptr ?
                 seq_idx < param.seq_len :
                 seq_idx < (param.padding_offset_prefix_sum[batch_idx + 1] - param.padding_offset_prefix_sum[batch_idx]);

    if (tidx < Dh_div_4 && is_valid_seq_idx) {

        const int sentence_idx =
            (param.padding_offset_prefix_sum == nullptr ? param.seq_len * batch_idx :
                                                          param.padding_offset_prefix_sum[batch_idx])
            + seq_idx;

        int Dh_div_2 = param.size_per_head / 2;
        int n_div_4  = param.head_num * Dh_div_4;
        int n_div_2  = param.head_num * Dh_div_2;

        const int hidden_idx = head_idx * Dh_div_4 + tidx;

        const int q_idx = sentence_idx * 3 * n_div_4 + hidden_idx;
        const int k_idx = sentence_idx * 3 * n_div_4 + hidden_idx + n_div_4;
        const int v_idx = sentence_idx * 3 * n_div_4 + hidden_idx + 2 * n_div_4;

        T1_4* qkv_ptr = (T1_4*)(param.QKV_T1);
        T2_2  q[2];
        T2_2  k[2];
        T2_2  v[2];

        fp8x4_e4m3_to_bfloat2(&q[0], &q[1], &qkv_ptr[q_idx]);
        fp8x4_e4m3_to_bfloat2(&k[0], &k[1], &qkv_ptr[k_idx]);
        fp8x4_e4m3_to_bfloat2(&v[0], &v[1], &qkv_ptr[v_idx]);
        T2_2 input_scale2 = cuda_cast<T2_2>(param.input_scale == nullptr ? 1.0f : __ldg(param.input_scale));

        if (param.input_scale_2 != nullptr) {
            // q.x = q.x * input_scale * __ldg(param.input_scale_2 + hidden_idx)
            //       * (param.input_scale_2_min == nullptr ? 1.0f : ldg(param.input_scale_2_min));
            // q.y = q.y * input_scale * __ldg(param.input_scale_2 + hidden_idx)
            //       * (param.input_scale_2_min == nullptr ? 1.0f : ldg(param.input_scale_2_min));
            // k.x = k.x * input_scale * __ldg(param.input_scale_2 + hidden_idx + n)
            //       * (param.input_scale_2_min == nullptr ? 1.0f : ldg(param.input_scale_2_min));
            // k.y = k.y * input_scale * __ldg(param.input_scale_2 + hidden_idx + n)
            //       * (param.input_scale_2_min == nullptr ? 1.0f : ldg(param.input_scale_2_min));
            // v.x = v.x * input_scale * __ldg(param.input_scale_2 + hidden_idx + 2 * n)
            //       * (param.input_scale_2_min == nullptr ? 1.0f : ldg(param.input_scale_2_min));
            // v.y = v.y * input_scale * __ldg(param.input_scale_2 + hidden_idx + 2 * n)
            //       * (param.input_scale_2_min == nullptr ? 1.0f : ldg(param.input_scale_2_min));
        }
        else {
            q[0] = hmul2(q[0], input_scale2);
            q[1] = hmul2(q[1], input_scale2);
            k[0] = hmul2(k[0], input_scale2);
            k[1] = hmul2(k[1], input_scale2);
            v[0] = hmul2(v[0], input_scale2);
            v[1] = hmul2(v[1], input_scale2);
        }

        T2_2* bias_ptr = (T2_2*)(param.qkv_bias);

        q[0] = mmha::add(q[0], bias_ptr[head_idx * Dh_div_2 + 2 * tidx]);
        q[1] = mmha::add(q[1], bias_ptr[head_idx * Dh_div_2 + 2 * tidx + 1]);
        k[0] = mmha::add(k[0], bias_ptr[head_idx * Dh_div_2 + 1 * n_div_2 + 2 * tidx]);
        k[1] = mmha::add(k[1], bias_ptr[head_idx * Dh_div_2 + 1 * n_div_2 + 2 * tidx + 1]);
        v[0] = mmha::add(v[0], bias_ptr[head_idx * Dh_div_2 + 2 * n_div_2 + 2 * tidx]);
        v[1] = mmha::add(v[1], bias_ptr[head_idx * Dh_div_2 + 2 * n_div_2 + 2 * tidx + 1]);

        mmha::apply_rotary_embedding(q[0], k[0], 2 * tidx, param.rotary_embedding_dim, seq_idx);
        mmha::apply_rotary_embedding(q[1], k[1], 2 * tidx + 1, param.rotary_embedding_dim, seq_idx);

        T2_2 output_scale2 = cuda_cast<T2_2>(param.output_scale == nullptr ? 1.0f : __ldg(param.output_scale));

        q[0] = hmul2(q[0], output_scale2);
        q[1] = hmul2(q[1], output_scale2);
        k[0] = hmul2(k[0], output_scale2);
        k[1] = hmul2(k[1], output_scale2);
        v[0] = hmul2(v[0], output_scale2);
        v[1] = hmul2(v[1], output_scale2);

        T1_4* q_out_ptr = (T1_4*)(param.q_buf);
        T1_4* k_out_ptr = (T1_4*)(param.k_buf);

        src_v_2[threadIdx.y * (SIZE_PER_HEAD / 2 + 1) + tidx * 2 + 0] = v[0];
        src_v_2[threadIdx.y * (SIZE_PER_HEAD / 2 + 1) + tidx * 2 + 1] = v[1];

        // q_buf, k_buf: [batch, head_num, seq_len_paaded, size_per_head]
        const int dest_idx = n_div_4 * param.seq_len_padded * batch_idx + Dh_div_4 * param.seq_len_padded * head_idx
                             + Dh_div_4 * seq_idx + tidx;

        q_out_ptr[dest_idx] = T1_4(q[0], q[1]);
        k_out_ptr[dest_idx] = T1_4(k[0], k[1]);
    }
    else {
        src_v_2[threadIdx.y * (SIZE_PER_HEAD / 2 + 1) + tidx * 2 + 0] = cuda_cast<T2_2>(0.0f);
        src_v_2[threadIdx.y * (SIZE_PER_HEAD / 2 + 1) + tidx * 2 + 1] = cuda_cast<T2_2>(0.0f);
    }

    __syncthreads();

    const int seq_group_id = seq_idx / SEQ_GROUP_SIZE;
    // v_buf: [batch, head_num, size_per_head, seq_len_paaded]
    T1_4* v_out_ptr = (T1_4*)(param.v_buf);
    for (int new_id = threadIdx.x * blockDim.y + threadIdx.y; new_id < SIZE_PER_HEAD * (SEQ_GROUP_SIZE / 4);
         new_id += blockDim.x * blockDim.y) {
        int new_size_id = new_id / (SEQ_GROUP_SIZE / 4);
        int new_seq_id  = new_id % (SEQ_GROUP_SIZE / 4);

        T2_2 val_1;
        T2_2 val_2;
        val_1.x = src_v[(4 * new_seq_id + 0) * (SIZE_PER_HEAD + 2) + new_size_id];
        val_1.y = src_v[(4 * new_seq_id + 1) * (SIZE_PER_HEAD + 2) + new_size_id];
        val_2.x = src_v[(4 * new_seq_id + 2) * (SIZE_PER_HEAD + 2) + new_size_id];
        val_2.y = src_v[(4 * new_seq_id + 3) * (SIZE_PER_HEAD + 2) + new_size_id];

        v_out_ptr[param.head_num * param.size_per_head * (param.seq_len_padded / 4) * batch_idx
                  + param.size_per_head * (param.seq_len_padded / 4) * head_idx
                  + new_size_id * (param.seq_len_padded / 4) + seq_group_id * (SEQ_GROUP_SIZE / 4) + new_seq_id] =
            T1_4(val_1, val_2);
    }
}

template<typename T1, typename T2>
void invokeFP8AddFusedQKVBiasRebuildPadding(FP8AddFusedQKVBiasRebuildPaddingParam<T1, T2> param)
{
    // To implement rotary embeddings, each thread processes two QKV elems:
    const int seq_group_size = 64;
    dim3      block((param.size_per_head / 4 + 1) / 2 * 2, seq_group_size);
    dim3      grid(param.batch_size, (param.seq_len + seq_group_size - 1) / seq_group_size, param.head_num);

    FT_CHECK(block.x * block.y <= 1024);
    if (param.size_per_head == 64) {
        FP8AddFusedQKVBiasRebuildPaddingKernel<T1, T2, seq_group_size, 64><<<grid, block, 0, param.stream>>>(param);
    }
    else {
        FT_CHECK(false);
    }
}

// template void invokeFP8AddFusedQKVBiasRebuildPadding<__nv_fp8_e4m3, __nv_fp8_e4m3>(
//     FP8AddFusedQKVBiasRebuildPaddingParam<__nv_fp8_e4m3, __nv_fp8_e4m3> param);
// template void invokeFP8AddFusedQKVBiasRebuildPadding<__nv_fp8_e4m3, half>(
//     FP8AddFusedQKVBiasRebuildPaddingParam<__nv_fp8_e4m3, half> param);
template void invokeFP8AddFusedQKVBiasRebuildPadding<__nv_fp8_e4m3, __nv_bfloat16>(
    FP8AddFusedQKVBiasRebuildPaddingParam<__nv_fp8_e4m3, __nv_bfloat16> param);
#else
template<typename T1, typename T2, bool INPUT_T1>
__global__ void FP8AddFusedQKVBiasRebuildPaddingKernel(FP8AddFusedQKVBiasRebuildPaddingParam<T1, T2> param)
{
    using VecFP8_t                   = typename Vec_t<T1>::Type;
    using VecBias_t                  = typename Vec_t<T2>::Type;
    using PACKED_BF16                = __nv_bfloat162_2;
    using PACKED_FP8                 = __nv_fp8x2_x2_e4m3;
    using ARRAY_FP8                  = __nv_fp8_4_e4m3;
    using ARRAY_BF16                 = __nv_bfloat164;
    using VecBF16_t                  = typename Vec_t<__nv_bfloat16>::Type;
    constexpr int ELEMENT_PER_THREAD = 4;
    constexpr int NUM_PACKS          = 2;
    // const int sentence_idx = blockIdx.z;
    // const int padded_row_id = sentence_idx + (param.padding_offset == nullptr ? 0 :
    // param.padding_offset[sentence_idx]);
    const int  batch_idx         = blockIdx.z / param.head_num;
    const int  head_idx          = blockIdx.z % param.head_num;
    const int  seq_idx           = threadIdx.y + blockIdx.y * blockDim.y;
    const int  sentence_idx      = batch_idx * param.seq_len + seq_idx;
    const int  size_per_head_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const bool V_CACHE_STORE     = param.v_cache != nullptr;
    const int  v_dest_idx_0      = param.size_per_head * param.seq_len_padded * param.head_num * batch_idx
                             + param.size_per_head * param.seq_len_padded * head_idx;
    __shared__ T1 sdata[32 * 33 * ELEMENT_PER_THREAD];
#pragma unroll
    for (int i = 0; i < ELEMENT_PER_THREAD; i++) {
        sdata[blockIdx.y * 33 * ELEMENT_PER_THREAD + blockIdx.x * ELEMENT_PER_THREAD + i] = (T1)0.0f;
    }

    if (size_per_head_idx * ELEMENT_PER_THREAD < param.size_per_head && seq_idx < param.seq_len_padded) {
        const int hidden_idx = head_idx * param.size_per_head + size_per_head_idx * ELEMENT_PER_THREAD;
        const int n          = param.head_num * param.size_per_head;

        // src QKV: [token_num, 3, head, hidden]
        const int q_idx = sentence_idx * 3 * n + hidden_idx;
        const int k_idx = sentence_idx * 3 * n + hidden_idx + n;
        const int v_idx = sentence_idx * 3 * n + hidden_idx + 2 * n;

        PACKED_BF16 q, k, v;

        if (INPUT_T1) {
            PACKED_FP8 q_input = *reinterpret_cast<const PACKED_FP8*>(&param.QKV_T1[q_idx]);
            PACKED_FP8 k_input = *reinterpret_cast<const PACKED_FP8*>(&param.QKV_T1[k_idx]);
            PACKED_FP8 v_input = *reinterpret_cast<const PACKED_FP8*>(&param.QKV_T1[v_idx]);
#pragma unroll
            for (int i = 0; i < NUM_PACKS; i++) {
                q.array[i] = convert_vec<VecBF16_t, VecFP8_t>(q_input.array[i]);
                k.array[i] = convert_vec<VecBF16_t, VecFP8_t>(k_input.array[i]);
                v.array[i] = convert_vec<VecBF16_t, VecFP8_t>(v_input.array[i]);
            }
        }
        else {
            q = *reinterpret_cast<const PACKED_BF16*>(&param.QKV_T2[q_idx]);
            k = *reinterpret_cast<const PACKED_BF16*>(&param.QKV_T2[k_idx]);
            v = *reinterpret_cast<const PACKED_BF16*>(&param.QKV_T2[v_idx]);
        }

        __nv_bfloat162 input_scale2 =
            __float2bfloat162_rn(param.input_scale == nullptr ? 1.0f : __ldg(param.input_scale));

        PACKED_BF16 q_bias, k_bias, v_bias;
        q_bias = *reinterpret_cast<const PACKED_BF16*>(&param.qkv_bias[hidden_idx]);
        k_bias = *reinterpret_cast<const PACKED_BF16*>(&param.qkv_bias[hidden_idx + n]);
        v_bias = *reinterpret_cast<const PACKED_BF16*>(&param.qkv_bias[hidden_idx + 2 * n]);
#pragma unroll
        for (int i = 0; i < NUM_PACKS; i++) {
            q.array[i] = mmha::add(hmul2(q.array[i], input_scale2), q_bias.array[i]);
            k.array[i] = mmha::add(hmul2(k.array[i], input_scale2), k_bias.array[i]);
            v.array[i] = mmha::add(hmul2(v.array[i], input_scale2), v_bias.array[i]);
        }

        // mmha::apply_rotary_embedding(q, k, tidx, param.rotary_embedding_dim, seq_idx);
        const int v_cache_idx = param.size_per_head * param.max_seq_len * param.head_num * batch_idx
                                + param.size_per_head * param.max_seq_len * head_idx + seq_idx * param.size_per_head
                                + size_per_head_idx * ELEMENT_PER_THREAD;
#ifndef FP8_MHA
        if (V_CACHE_STORE && seq_idx < param.seq_len) {
            *reinterpret_cast<PACKED_BF16*>(&param.v_cache[v_cache_idx]) = v;
        }
#endif

        __nv_bfloat162 output_scale2 =
            __float2bfloat162_rn(param.output_scale == nullptr ? 1.0f : __ldg(param.output_scale));
#pragma unroll
        for (int i = 0; i < NUM_PACKS; i++) {
            q.array[i] = hmul2(q.array[i], output_scale2);
            k.array[i] = hmul2(k.array[i], output_scale2);
            v.array[i] = hmul2(v.array[i], output_scale2);
#ifdef FP8_MHA
            if (V_CACHE_STORE && seq_idx < param.seq_len) {
                reinterpret_cast<VecFP8_t*>(&param.v_cache[v_cache_idx])[i] =
                    convert_vec<VecFP8_t, VecBF16_t>(v.array[i]);
            }
#endif
        }

        // q_buf, k_buf: [batch, head_num, seq_len_paaded, size_per_head]
        const int dest_idx = param.size_per_head * param.seq_len_padded * param.head_num * batch_idx
                             + param.size_per_head * param.seq_len_padded * head_idx + param.size_per_head * seq_idx
                             + size_per_head_idx * ELEMENT_PER_THREAD;

        PACKED_FP8 q_output, k_output;
#pragma unroll
        for (int i = 0; i < NUM_PACKS; i++) {
            q_output.array[i] = convert_vec<VecFP8_t, VecBF16_t>(q.array[i]);
            k_output.array[i] = convert_vec<VecFP8_t, VecBF16_t>(k.array[i]);
        }
        *reinterpret_cast<PACKED_FP8*>(&param.q_buf[dest_idx]) = q_output;
        *reinterpret_cast<PACKED_FP8*>(&param.k_buf[dest_idx]) = k_output;

        ARRAY_BF16 v_val = *reinterpret_cast<ARRAY_BF16*>(&v);
#pragma unroll
        for (int i = 0; i < ELEMENT_PER_THREAD; i++) {
            sdata[threadIdx.y * 33 * ELEMENT_PER_THREAD + (threadIdx.x * ELEMENT_PER_THREAD + i)] = (T1)v_val.array[i];
        }
    }

    __syncthreads();

    const int trans_size_per_head_idx = threadIdx.y + blockIdx.x * blockDim.x;
    const int trans_seq_idx           = threadIdx.x + blockIdx.y * blockDim.y;
    if (trans_size_per_head_idx * ELEMENT_PER_THREAD < param.size_per_head && trans_seq_idx < param.seq_len_padded) {
#pragma unroll
        for (int i = 0; i < ELEMENT_PER_THREAD; i++) {
            param.v_buf[v_dest_idx_0 + (trans_size_per_head_idx * ELEMENT_PER_THREAD + i) * param.seq_len_padded
                        + trans_seq_idx] =
                sdata[threadIdx.x * 33 * ELEMENT_PER_THREAD + threadIdx.y * ELEMENT_PER_THREAD + i];
        }
    }
}

template<typename T1, typename T2>
void invokeFP8AddFusedQKVBiasRebuildPadding(FP8AddFusedQKVBiasRebuildPaddingParam<T1, T2> param)
{
    // To implement rotary embeddings, each thread processes two QKV elems:
    // dim3 block((param.size_per_head / 2 + 31) / 32 * 32);
    // dim3 grid(param.head_num, param.token_num);
    const int pack_size = 4;
    FT_CHECK(param.size_per_head % pack_size == 0);
    dim3 block(32, 32);
    dim3 grid((param.size_per_head / pack_size + 31) / 32,
              (param.seq_len_padded + 31) / 32,
              param.head_num * param.batch_size);
    // FT_CHECK(block.x * block.y <= 1024);
    // shared memory configuration
    const int carveout = 50;  // prefer shared memory capacity 50% of maximum
    if (param.QKV_T1 == nullptr) {
        assert(param.QKV_T2 != nullptr);
        cudaFuncSetAttribute((const void*)FP8AddFusedQKVBiasRebuildPaddingKernel<T1, T2, false>,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             carveout);
        FP8AddFusedQKVBiasRebuildPaddingKernel<T1, T2, false><<<grid, block, 0, param.stream>>>(param);
    }
    else {
        cudaFuncSetAttribute((const void*)FP8AddFusedQKVBiasRebuildPaddingKernel<T1, T2, true>,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             carveout);
        FP8AddFusedQKVBiasRebuildPaddingKernel<T1, T2, true><<<grid, block, 0, param.stream>>>(param);
    }
}

template void invokeFP8AddFusedQKVBiasRebuildPadding<__nv_fp8_e4m3, __nv_fp8_e4m3>(
    FP8AddFusedQKVBiasRebuildPaddingParam<__nv_fp8_e4m3, __nv_fp8_e4m3> param);
template void invokeFP8AddFusedQKVBiasRebuildPadding<__nv_fp8_e4m3, half>(
    FP8AddFusedQKVBiasRebuildPaddingParam<__nv_fp8_e4m3, half> param);
template void invokeFP8AddFusedQKVBiasRebuildPadding<__nv_fp8_e4m3, __nv_bfloat16>(
    FP8AddFusedQKVBiasRebuildPaddingParam<__nv_fp8_e4m3, __nv_bfloat16> param);

#endif

template<typename T1, typename T2>
__global__ void FP8TrtAddQKVBiasKernel(FP8TrtAddQKVBiasParam<T1, T2> param)
{
    // Add bias ([3, head, size]), and then transpose from
    // [valid_word_num, 3, head, size] -> [valid_word_num, head, 3, size]

    const T1* qkv_src_ptr = param.qkv_src + blockIdx.x * 3 * param.hidden_unit;
    const T2* bias_ptr    = param.qkv_bias;
    T1*       qkv_tgt_ptr = param.qkv_tgt + blockIdx.x * 3 * param.hidden_unit;

    qkv_tgt_ptr[threadIdx.x * 3 * param.size_per_head + blockIdx.y * param.size_per_head + threadIdx.y] =
        (T1)(((float)qkv_src_ptr[blockIdx.y * param.hidden_unit + threadIdx.x * param.size_per_head + threadIdx.y]
                  * __ldg(param.input_scale)
              + (float)bias_ptr[blockIdx.y * param.hidden_unit + threadIdx.x * param.size_per_head + threadIdx.y])
             * __ldg(param.output_scale));
}

template<>
__global__ void FP8TrtAddQKVBiasKernel(FP8TrtAddQKVBiasParam<__nv_fp8_e4m3, __nv_bfloat16> param)
{
    // Add bias ([3, head, size]), and then transpose from
    // [valid_word_num, 3, head, size] -> [valid_word_num, head, 3, size]

    using T1_4 = __nv_fp8x4_e4m3;
    using T2_2 = typename TypeConverter<__nv_bfloat16>::Type;

    const T1_4* qkv_src_ptr = (T1_4*)(param.qkv_src + blockIdx.x * 3 * param.hidden_unit);
    const T2_2* bias_ptr    = (T2_2*)param.qkv_bias;
    T1_4*       qkv_tgt_ptr = (T1_4*)(param.qkv_tgt + blockIdx.x * 3 * param.hidden_unit);

    const int size_div_4   = param.size_per_head / 4;
    const int hidden_div_4 = param.hidden_unit / 4;
    const int src_id       = threadIdx.z * hidden_div_4 + threadIdx.y * size_div_4 + threadIdx.x;

    T2_2 val1, val2;
    fp8x4_e4m3_to_bfloat2(&val1, &val2, &qkv_src_ptr[src_id]);
    T2_2      input_scale_2  = cuda_cast<T2_2, float>(__ldg(param.input_scale));
    T2_2      output_scale_2 = cuda_cast<T2_2, float>(__ldg(param.output_scale));
    const int bias_id_0      = src_id * 2;
    val1                     = hmul2(hadd2(hmul2(val1, input_scale_2), bias_ptr[bias_id_0]), output_scale_2);
    val2                     = hmul2(hadd2(hmul2(val2, input_scale_2), bias_ptr[bias_id_0 + 1]), output_scale_2);

    qkv_tgt_ptr[(threadIdx.y * 3 * size_div_4 + threadIdx.z * size_div_4) + threadIdx.x] = __nv_fp8x4_e4m3(val1, val2);
}

template<typename T1, typename T2>
void invokeFP8TrtAddQKVBias(FP8TrtAddQKVBiasParam<T1, T2> param)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (std::is_same<T1, __nv_fp8_e4m3>::value && std::is_same<T2, __nv_bfloat16>::value) {
        dim3 grid(param.valid_word_num);
        dim3 block(param.size_per_head / 4, param.head_num, 3);
        FP8TrtAddQKVBiasKernel<T1, T2><<<grid, block, 0, param.stream>>>(param);
    }
    else {
        dim3 grid(param.valid_word_num, 3);
        dim3 block(param.head_num, param.size_per_head);

        FP8TrtAddQKVBiasKernel<T1, T2><<<grid, block, 0, param.stream>>>(param);
    }
}

#ifdef ENABLE_FP8
template void
invokeFP8TrtAddQKVBias<__nv_fp8_e4m3, __nv_bfloat16>(FP8TrtAddQKVBiasParam<__nv_fp8_e4m3, __nv_bfloat16> param);
#endif

template<typename T1, typename T2>
__global__ void transpose_4d_batch_major_k_cache(T2*          k_dst,
                                                 const T1*    k_src,
                                                 const float* scale,
                                                 const int    head_num,
                                                 const int    size_per_head,
                                                 const int    seq_len,
                                                 const int    max_seq_len,
                                                 const int    seq_len_padded)
{
    const int     batch_id = blockIdx.y;
    const int     head_id  = blockIdx.z;
    constexpr int X_ELEMS  = 16;

    auto key_src = reinterpret_cast<const uint4*>(k_src + batch_id * head_num * size_per_head * seq_len_padded
                                                  + head_id * size_per_head * seq_len_padded);
    auto key_dst = reinterpret_cast<uint4*>(k_dst + batch_id * head_num * size_per_head * max_seq_len
                                            + head_id * size_per_head * max_seq_len);

    const int out_idx             = blockIdx.x * blockDim.x + threadIdx.x;
    int       size_per_head_div_x = size_per_head / X_ELEMS;
    if (out_idx >= size_per_head_div_x * max_seq_len) {
        return;
    }

    int       idx            = out_idx;
    const int k_seq_len_id   = idx % max_seq_len;
    idx                      = (idx - k_seq_len_id) / max_seq_len;
    const int k_head_size_id = idx % size_per_head_div_x;

    if (k_seq_len_id < seq_len) {
        key_dst[out_idx] = key_src[k_seq_len_id * size_per_head_div_x + k_head_size_id];
    }
}

template<>
__global__ void transpose_4d_batch_major_k_cache(__nv_bfloat16*       k_dst,
                                                 const __nv_fp8_e4m3* k_src,
                                                 const float*         scale,
                                                 const int            head_num,
                                                 const int            size_per_head,
                                                 const int            seq_len,
                                                 const int            max_seq_len,
                                                 const int            seq_len_padded)
{
    const int     batch_id  = blockIdx.y;
    const int     head_id   = blockIdx.z;
    constexpr int X_ELEMS   = 8;
    const float   scale_val = scale[0];
    using fp8_8             = __nv_fp8_8_e4m3;
    using bf16_8            = __nv_bfloat168;

    auto key_src = reinterpret_cast<const fp8_8*>(k_src + batch_id * head_num * size_per_head * seq_len_padded
                                                  + head_id * size_per_head * seq_len_padded);
    auto key_dst = reinterpret_cast<bf16_8*>(k_dst + batch_id * head_num * size_per_head * max_seq_len
                                             + head_id * size_per_head * max_seq_len);

    const int out_idx             = blockIdx.x * blockDim.x + threadIdx.x;
    int       size_per_head_div_x = size_per_head / X_ELEMS;
    if (out_idx >= size_per_head_div_x * max_seq_len) {
        return;
    }

    int       idx            = out_idx;
    const int k_seq_len_id   = idx % max_seq_len;
    idx                      = (idx - k_seq_len_id) / max_seq_len;
    const int k_head_size_id = idx % size_per_head_div_x;

    if (k_seq_len_id < seq_len) {
        fp8_8  src_val = key_src[k_seq_len_id * size_per_head_div_x + k_head_size_id];
        bf16_8 dst_val;
#pragma unroll
        for (int i = 0; i < X_ELEMS; i++) {
            dst_val.array[i] = __float2bfloat16_rn((float)src_val.array[i] * scale_val);
        }
        key_dst[out_idx] = dst_val;
    }
}

template<typename T1, typename T2>
__global__ void transpose_4d_batch_major_v_cache(T2*          v_dst,
                                                 const T1*    v_src,
                                                 const float* scale,
                                                 const int    head_num,
                                                 const int    size_per_head,
                                                 const int    seq_len,
                                                 const int    max_seq_len,
                                                 const int    seq_len_padded)
{
    const int      batch_id    = blockIdx.y;
    const int      head_id     = blockIdx.z;
    constexpr bool BF16_OUTPUT = std::is_same<T2, __nv_bfloat16>::value;
    float          scale_val   = BF16_OUTPUT ? scale[0] : 1.0f;

    // 16 byte loads will handle "x" dimension
    // NOTE: need transpose, so cannot take x dimension
    auto val_src = reinterpret_cast<const T1*>(v_src + batch_id * head_num * size_per_head * seq_len_padded
                                               + head_id * size_per_head * seq_len_padded);
    auto val_dst = reinterpret_cast<T2*>(v_dst + batch_id * head_num * size_per_head * max_seq_len
                                         + head_id * size_per_head * max_seq_len);

    // idx is over output dimension L * size_per_head / x for values
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size_per_head * seq_len)
        return;

    int seq_idx       = idx / size_per_head;
    int head_size_idx = idx % size_per_head;
    val_dst[idx]      = (T2)((float)val_src[head_size_idx * seq_len_padded + seq_idx] * scale_val);
}

// template<typename T1, typename T2>
// __global__ void transpose_4d_batch_major_v_cache(T2* v_dst,
//                                                  const T1* v_src,
//                                                  const float* scale,
//                                                  const int head_num,
//                                                  const int size_per_head,
//                                                  const int seq_len,
//                                                  const int max_seq_len,
//                                                  const int seq_len_padded)
// {
//     const int batch_id = blockIdx.z / head_num;
//     const int head_id = blockIdx.z % head_num;
//     constexpr bool BF16_OUTPUT = std::is_same<T2, __nv_bfloat16>::value;
//     float scale_val = BF16_OUTPUT ? scale[0] : 1.0f;
//     __shared__ T2 s_transpose_buffer[32][33];

//     // 16 byte loads will handle "x" dimension
//     // NOTE: need transpose, so cannot take x dimension
//     auto val_src = reinterpret_cast<const T1*>(v_src + batch_id * head_num * size_per_head * seq_len_padded
//                                                   + head_id * size_per_head * seq_len_padded);
//     auto val_dst = reinterpret_cast<T2*>(v_dst + batch_id * head_num * size_per_head * max_seq_len
//                                             + head_id * size_per_head * max_seq_len);

//     // idx is over output dimension L * size_per_head / x for values
//     for (int y_block_id = 0; y_block_id * blockDim.x < size_per_head; y_block_id ++) {
//         for (int x_block_id = 0; x_block_id * blockDim.x < seq_len; x_block_id ++) {
//             int seq_block_id = threadIdx.x + blockDim.x * x_block_id;
//             int head_size_id = threadIdx.y + blockDim.y * y_block_id;
//             if (seq_block_id < seq_len && head_size_id < size_per_head) {
//                 s_transpose_buffer[threadIdx.y][threadIdx.x] = (T2) ((float) val_src[head_size_id * seq_len_padded +
//                 seq_block_id] * scale_val);
//             }
//             __syncthreads();
//             int transposed_head_size_id = threadIdx.x + blockDim.y * y_block_id;
//             int transposed_seq_block_id = threadIdx.y + blockDim.x * x_block_id;
//             if (transposed_head_size_id < size_per_head && transposed_seq_block_id < seq_len) {
//                 val_dst[transposed_seq_block_id * size_per_head + transposed_head_size_id] =
//                 s_transpose_buffer[threadIdx.x][threadIdx.y];
//             }
//         }
//     }
// }

template<typename T1, typename T2>
void invokeFP8Transpose4dBatchMajor(FP8Transpose4dBatchMajorParam<T1, T2> param)
{
    constexpr int block_sz = 128;
    constexpr int x        = std::is_same<T2, __nv_bfloat16>::value ? 8 : 16;
    int           size     = param.max_seq_len * param.size_per_head / x;
    dim3          grid((size + block_sz - 1) / block_sz, param.local_batch_size, param.local_head_num);
    dim3          grid_v(
        (param.seq_len * param.size_per_head + block_sz - 1) / block_sz, param.local_batch_size, param.local_head_num);

    transpose_4d_batch_major_k_cache<<<grid, block_sz, 0, param.stream>>>(param.k_dst,
                                                                          param.k_src,
                                                                          param.scale,
                                                                          param.local_head_num,
                                                                          param.size_per_head,
                                                                          param.seq_len,
                                                                          param.max_seq_len,
                                                                          param.seq_len_padded);

    // transpose_4d_batch_major_v_cache<<<grid_v, block_sz, 0, param.stream>>>(param.v_dst,
    //                                                                         param.v_src,
    //                                                                         param.scale,
    //                                                                         param.local_head_num,
    //                                                                         param.size_per_head,
    //                                                                         param.seq_len,
    //                                                                         param.max_seq_len,
    //                                                                         param.seq_len_padded);
}

template void invokeFP8Transpose4dBatchMajor<__nv_fp8_e4m3, __nv_fp8_e4m3>(
    FP8Transpose4dBatchMajorParam<__nv_fp8_e4m3, __nv_fp8_e4m3> param);
template void invokeFP8Transpose4dBatchMajor<__nv_fp8_e4m3, __nv_bfloat16>(
    FP8Transpose4dBatchMajorParam<__nv_fp8_e4m3, __nv_bfloat16> param);

template<int ITEMS_PER_THREAD, typename T, typename T_IN>
__global__ void softmax_kernel(T* qk_buf_,
                               const T_IN* __restrict__ qk_buf_src,
                               const T* __restrict__ attr_mask,
                               const int    batch_size,
                               const int    head_num,
                               const int    seq_len,
                               const float  scalar,
                               const float* input_scale,
                               const float* output_scale)
{
    float input_scale_val  = input_scale == nullptr ? 1.0f : __ldg(input_scale);
    float output_scale_val = output_scale == nullptr ? 1.0f : __ldg(output_scale);
    for (int seq_id = blockIdx.x; seq_id < seq_len; seq_id += gridDim.x) {
        float            data[ITEMS_PER_THREAD];
        int              qk_offset;
        __shared__ float s_mean, s_max;
        float            local_max = -1e20f;
        for (int i = 0; blockDim.x * i + threadIdx.x < seq_len; i++) {
            qk_offset =
                ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id) * seq_len + blockDim.x * i + threadIdx.x;
            int mask_offset = (blockIdx.y * seq_len + seq_id) * seq_len + blockDim.x * i + threadIdx.x;

            float qk       = static_cast<float>(qk_buf_src[qk_offset]) * input_scale_val;
            float mask_val = static_cast<float>(attr_mask[mask_offset]);

            mask_val = (1.0f - mask_val) * -10000.0f;

            data[i]   = qk * static_cast<float>(scalar) + mask_val;
            local_max = fmax(local_max, data[i]);
        }

        float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float local_sum = 0;
        for (int i = 0; blockDim.x * i + threadIdx.x < seq_len; i++) {
            data[i] = __expf(data[i] - s_max);
            local_sum += data[i];
        }
        float sum_val = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);
        if (threadIdx.x == 0) {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < seq_len; i++) {
            qk_offset =
                ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id) * seq_len + blockDim.x * i + threadIdx.x;
            qk_buf_[qk_offset] = (T)(data[i] * s_mean * output_scale_val);
        }
    }
}

template<typename T_OUT, typename T_IN, typename T_COMPUTE, int ITEMS_PER_THREAD, int NUM>
__global__ void softmax_kernel_v5_half2(T_OUT*       qk_buf,
                                        const T_IN*  qk_buf_src,
                                        const T_OUT* attr_mask,
                                        const int*   padding_offset_prefix_sum,
                                        const int    batch_size,
                                        const int    head_num,
                                        const int    seq_len,
                                        const float  scalar,
                                        const float* input_scale,
                                        const float* output_scale)
{
    using T2_OUT     = typename Vec_t<T_OUT>::Type;
    using T2_IN      = typename Vec_t<T_IN>::Type;
    using T2_COMPUTE = typename Vec_t<T_COMPUTE>::Type;

    T2_OUT*       qk_buf_out2     = (T2_OUT*)qk_buf;
    const T2_OUT* attr_mask_half2 = (const T2_OUT*)attr_mask;
    T2_IN*        qk_buf_in2      = (T2_IN*)qk_buf_src;

    T2_COMPUTE input_scale_2  = cuda_cast<T2_COMPUTE, float>(input_scale == nullptr ? 1.0f : __ldg(input_scale));
    T2_COMPUTE output_scale_2 = cuda_cast<T2_COMPUTE, float>(output_scale == nullptr ? 1.0f : __ldg(output_scale));

    for (int seq_id = blockIdx.x; seq_id < seq_len; seq_id += gridDim.x * NUM) {
        if ((padding_offset_prefix_sum != nullptr)
            && (seq_id >= padding_offset_prefix_sum[blockIdx.y + 1] - padding_offset_prefix_sum[blockIdx.y])) {

            T2_COMPUTE zero_val;
            zero_val.x       = 0.0f;
            zero_val.y       = 0.0f;
            const int offset = (blockIdx.y * head_num * seq_len + blockIdx.z * seq_len + seq_id) * seq_len / 2;
            for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2); i++) {
                qk_buf_out2[offset + i] = T2_OUT(zero_val);
            }
            continue;
        }

        T2_COMPUTE data[NUM][ITEMS_PER_THREAD];

        int qk_offset[NUM];

        __shared__ float s_sum[NUM], s_max[NUM];
        float            local_max[NUM];
#pragma unroll
        for (int j = 0; j < NUM; j++) {
            local_max[j] = -1e20f;
        }

        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                qk_offset[j] = ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id + j * gridDim.x) * (seq_len / 2)
                               + blockDim.x * i + threadIdx.x;
            }

            T2_COMPUTE mask_val[NUM];
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                const int mask_offset =
                    (blockIdx.y * seq_len + seq_id + j * gridDim.x) * (seq_len / 2) + blockDim.x * i + threadIdx.x;
                mask_val[j] = cuda_cast<T2_COMPUTE>(attr_mask_half2[mask_offset]);
            }

            T2_COMPUTE qk[NUM];
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                qk[j] = hmul2<T2_COMPUTE>(cuda_cast<T2_COMPUTE>(qk_buf_in2[qk_offset[j]]), input_scale_2);
            }

#pragma unroll
            for (int j = 0; j < NUM; j++) {
                mask_val[j] = hmul2<T2_COMPUTE>(hsub2<T2_COMPUTE>(cuda_cast<T2_COMPUTE>(1.0f), mask_val[j]),
                                                cuda_cast<T2_COMPUTE>(-10000.0f));
            }

#pragma unroll
            for (int j = 0; j < NUM; j++) {
                data[j][i]   = hadd2<T2_COMPUTE>(hmul2<T2_COMPUTE>(qk[j], cuda_cast<T2_COMPUTE>(scalar)), mask_val[j]);
                local_max[j] = fmax(local_max[j], fmax((float)data[j][i].x, (float)data[j][i].y));
            }
        }
        if (blockDim.x <= 32) {
            warpReduceMaxV2<float, NUM>(local_max);
        }
        else {
            blockReduceMaxV2<float, NUM>(local_max);
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                s_max[j] = local_max[j];
            }
        }
        __syncthreads();

        float local_sum[NUM];
#pragma unroll
        for (int j = 0; j < NUM; j++) {
            local_sum[j] = {0.f};
        }

        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                data[j][i] = hexp2<T2_COMPUTE>(hsub2<T2_COMPUTE>(data[j][i], cuda_cast<T2_COMPUTE>(s_max[j])));
            }

#pragma unroll
            for (int j = 0; j < NUM; j++) {
                local_sum[j] += (float)(data[j][i].x + data[j][i].y);
            }
        }

        if (blockDim.x <= 32) {
            warpReduceSumV2<float, NUM>(local_sum);
        }
        else {
            blockReduceSumV2<float, NUM>(local_sum);
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                s_sum[j] = __fdividef(1.0f, local_sum[j] + 1e-6f);
            }
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                qk_offset[j] = ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id + j * gridDim.x) * (seq_len / 2)
                               + blockDim.x * i + threadIdx.x;
            }

#pragma unroll
            for (int j = 0; j < NUM; j++) {
                qk_buf_out2[qk_offset[j]] = T2_OUT(
                    hmul2<T2_COMPUTE>(hmul2<T2_COMPUTE>(data[j][i], cuda_cast<T2_COMPUTE>(s_sum[j])), output_scale_2));
            }
        }
    }
}

#define SOFTMAX_KERNEL_HALF2(ITEMS_PER_THREAD)                                                                         \
    block.x /= ITEMS_PER_THREAD;                                                                                       \
    assert(block.x <= 1024);                                                                                           \
    grid.x /= 4;                                                                                                       \
    softmax_kernel_v5_half2<T, T_IN, __nv_bfloat16, ITEMS_PER_THREAD, 4>                                               \
        <<<grid, block, 0, param.stream>>>((T*)param.buffer,                                                           \
                                           (T_IN*)param.buffer_src,                                                    \
                                           (const T*)param.attr_mask,                                                  \
                                           param.padding_offset_prefix_sum,                                            \
                                           param.batch_size,                                                           \
                                           param.head_num,                                                             \
                                           param.seq_len,                                                              \
                                           param.scalar,                                                               \
                                           param.input_scale,                                                          \
                                           param.output_scale);

template<typename T, typename T_IN>
void invokeFP8MaskedSoftMax(FP8MaskedSoftMaxParam<T, T_IN> param)
{
    dim3 grid(param.seq_len, param.batch_size, param.head_num);
    if (param.batch_size * param.head_num > 360) {
        grid.x = ceil(float(param.seq_len) / 32.0f);
    }

    bool is_half2 = sizeof(T) == 1 && sizeof(T_IN) == 2 && param.seq_len % 2 == 0;

    if (is_half2 && grid.x % 4 == 0) {
        dim3 block((param.seq_len / (is_half2 ? 2 : 1) + 31) / 32 * 32);
        if (block.x > 2048 && block.x <= 4096) {
            SOFTMAX_KERNEL_HALF2(4)
        }
        else if (block.x > 1024) {
            SOFTMAX_KERNEL_HALF2(2)
        }
        else if (block.x > 0) {
            SOFTMAX_KERNEL_HALF2(1)
        }
        else {
            FT_CHECK(param.seq_len <= 4096);
        }
    }
    else {
        dim3 block((param.seq_len + 31) / 32 * 32);
        if (block.x > 2048 && block.x <= 4096) {
            softmax_kernel<4, T, T_IN><<<grid, block, 0, param.stream>>>(param.buffer,
                                                                         param.buffer_src,
                                                                         param.attr_mask,
                                                                         param.batch_size,
                                                                         param.head_num,
                                                                         param.seq_len,
                                                                         param.scalar,
                                                                         param.input_scale,
                                                                         param.output_scale);
        }
        else if (block.x > 1024) {
            softmax_kernel<2, T, T_IN><<<grid, block, 0, param.stream>>>(param.buffer,
                                                                         param.buffer_src,
                                                                         param.attr_mask,
                                                                         param.batch_size,
                                                                         param.head_num,
                                                                         param.seq_len,
                                                                         param.scalar,
                                                                         param.input_scale,
                                                                         param.output_scale);
        }
        else if (block.x > 0) {
            softmax_kernel<1, T, T_IN><<<grid, block, 0, param.stream>>>(param.buffer,
                                                                         param.buffer_src,
                                                                         param.attr_mask,
                                                                         param.batch_size,
                                                                         param.head_num,
                                                                         param.seq_len,
                                                                         param.scalar,
                                                                         param.input_scale,
                                                                         param.output_scale);
        }
        else {
            FT_CHECK(param.seq_len <= 4096);
        }
    }
}

template void
invokeFP8MaskedSoftMax<__nv_fp8_e4m3, __nv_fp8_e4m3>(FP8MaskedSoftMaxParam<__nv_fp8_e4m3, __nv_fp8_e4m3> param);

template void invokeFP8MaskedSoftMax<__nv_fp8_e4m3, float>(FP8MaskedSoftMaxParam<__nv_fp8_e4m3, float> param);

template void
invokeFP8MaskedSoftMax<__nv_fp8_e4m3, __nv_bfloat16>(FP8MaskedSoftMaxParam<__nv_fp8_e4m3, __nv_bfloat16> param);

template<typename T>
struct Pack4_type {
    using type = __nv_bfloat164;
};

template<>
struct Pack4_type<__nv_fp8_e4m3> {
    using type = __nv_fp8_4_e4m3;
};

template<typename T_IN, typename T_OUT>
__global__ void FP8TransposeAttentionOutRemovePadding(T_OUT*       dst,
                                                      const T_IN*  src,
                                                      const float* scale_ptr,
                                                      const int    batch_size,
                                                      const int    seq_len,
                                                      const int    head_num,
                                                      const int    size_per_head,
                                                      const int*   padding_offset)
{
    // transpose from [batch_size, head_num, seq_len, size_per_head] to [batch_size, seq_len, head_num, size_per_head]
    using pack4_in                = typename Pack4_type<T_IN>::type;
    using pack4_out               = typename Pack4_type<T_OUT>::type;
    constexpr int   pack_size     = 4;
    const int       padded_row_id = blockIdx.x + (padding_offset == nullptr ? 0 : padding_offset[blockIdx.x]);
    const int       src_batch_id  = padded_row_id / seq_len;
    const int       src_seq_id    = padded_row_id % seq_len;
    const pack4_in* src_packed    = reinterpret_cast<const pack4_in*>(src);
    pack4_out*      dst_packed    = reinterpret_cast<pack4_out*>(dst);

    const float scale = (scale_ptr == nullptr) ? 1.0f : scale_ptr[0];

    for (int idx = threadIdx.x; idx < head_num * size_per_head; idx += blockDim.x) {
        const int head_id   = idx / size_per_head;
        const int hidden_id = idx % size_per_head;
        pack4_out out_val;
#pragma unroll
        for (int i = 0; i < pack_size; i++) {
            // TODO: bfloat162 computation ?
            out_val.array[i] =
                (T_OUT)((float)(src_packed[src_batch_id * head_num * seq_len * size_per_head
                                           + head_id * seq_len * size_per_head + src_seq_id * size_per_head + hidden_id]
                                    .array[i])
                        * scale);
        }
        dst_packed[blockIdx.x * head_num * size_per_head + idx] = out_val;
    }
}

template<typename T_IN, typename T_OUT>
void invokeFP8TransposeAttentionOutRemovePadding(FP8TransposeAttentionOutRemovePaddingParam<T_IN, T_OUT> param)
{
    // NOTE: fp8_4_t optimization
    assert(param.size_per_head % 4 == 0);
    int block_size = param.head_num * param.size_per_head;
    block_size     = std::min(block_size, 512);
    FP8TransposeAttentionOutRemovePadding<<<param.valid_word_num, block_size, 0, param.stream>>>((T_OUT*)param.dst,
                                                                                                 (const T_IN*)param.src,
                                                                                                 param.scale,
                                                                                                 param.batch_size,
                                                                                                 param.seq_len,
                                                                                                 param.head_num,
                                                                                                 param.size_per_head
                                                                                                     / 4,
                                                                                                 param.padding_offset);
    // if (std::is_same<T_IN, __nv_fp8_e4m3>::value && std::is_same<T_OUT, __nv_fp8_e4m3>::value) {
    //     int block_size = param.head_num * (param.size_per_head);
    //     block_size = std::min(block_size, 512);
    //     FP8TransposeAttentionOutRemovePadding<<<param.valid_word_num, block_size, 0, param.stream>>>(
    //         (__nv_fp8_e4m3*)param.dst,
    //         (__nv_fp8_e4m3*)param.src,
    //         (const float*) nullptr,
    //         param.batch_size,
    //         param.seq_len,
    //         param.head_num,
    //         param.size_per_head / 4,
    //         param.padding_offset);
    // }
    // else {
    //     int block_size = param.head_num * param.size_per_head;
    //     block_size = std::min(block_size, 512);
    //     FP8TransposeAttentionOutRemovePadding<<<param.valid_word_num, block_size, 0, param.stream>>>(
    //         (T_OUT*) param.dst,
    //         (const T_IN*) param.src,
    //         param.scale,
    //         param.batch_size,
    //         param.seq_len,
    //         param.head_num,
    //         param.size_per_head,
    //         param.padding_offset);
    // }
}

template void invokeFP8TransposeAttentionOutRemovePadding<__nv_fp8_e4m3, __nv_fp8_e4m3>(
    FP8TransposeAttentionOutRemovePaddingParam<__nv_fp8_e4m3, __nv_fp8_e4m3> param);

template void invokeFP8TransposeAttentionOutRemovePadding<__nv_bfloat16, __nv_fp8_e4m3>(
    FP8TransposeAttentionOutRemovePaddingParam<__nv_bfloat16, __nv_fp8_e4m3> param);

__global__ void tmpHanldKCache(__nv_bfloat16* dst_k,
                               __nv_fp8_e4m3* src_k,
                               const float*   scale,
                               int            batch_size,
                               int            seq_len,
                               int            padded_seq_len,
                               int            head_num,
                               int            size_per_head)
{
    int batch_id = blockIdx.y;
    int head_id  = blockIdx.z;

    __nv_bfloat16* dst_k_ptr =
        dst_k + batch_id * head_num * seq_len * size_per_head + head_id * seq_len * size_per_head;
    __nv_fp8_e4m3* src_k_ptr =
        src_k + batch_id * head_num * padded_seq_len * size_per_head + head_id * padded_seq_len * size_per_head;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * size_per_head) {
        return;
    }

    float scale_val    = scale == nullptr ? 1.0 : __ldg(scale);
    int   out_idx      = idx;
    dst_k_ptr[out_idx] = (__nv_bfloat16)((float)(src_k_ptr[out_idx]) * scale_val);
}

void invokeTmpHanldKCache(__nv_bfloat16* dst_k,
                          __nv_fp8_e4m3* src_k,
                          const float*   scale,
                          int            batch_size,
                          int            seq_len,
                          int            padded_seq_len,
                          int            head_num,
                          int            size_per_head,
                          cudaStream_t   stream)
{
    // from [batch, head_num, seq_len_paaded, size_per_head] to [batch, head_num, seq_len, size_per_head]

    int  block_sz = 128;
    dim3 grid((seq_len * size_per_head + block_sz - 1) / block_sz, batch_size, head_num);

    tmpHanldKCache<<<grid, block_sz, 0, stream>>>(
        dst_k, src_k, scale, batch_size, seq_len, padded_seq_len, head_num, size_per_head);
}

__global__ void tmpHanldVCache(__nv_bfloat16* dst_v,
                               __nv_fp8_e4m3* src_v,
                               const float*   scale,
                               int            batch_size,
                               int            seq_len,
                               int            padded_seq_len,
                               int            head_num,
                               int            size_per_head)
{
    int batch_id = blockIdx.y;
    int head_id  = blockIdx.z;

    __nv_bfloat16* dst_v_ptr =
        dst_v + batch_id * head_num * seq_len * size_per_head + head_id * seq_len * size_per_head;
    __nv_fp8_e4m3* src_v_ptr =
        src_v + batch_id * head_num * padded_seq_len * size_per_head + head_id * padded_seq_len * size_per_head;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * size_per_head) {
        return;
    }

    float scale_val     = scale == nullptr ? 1.0 : __ldg(scale);
    int   out_idx       = idx;
    int   seq_idx       = idx / size_per_head;
    int   head_size_idx = idx % size_per_head;
    dst_v_ptr[out_idx]  = (__nv_bfloat16)((float)(src_v_ptr[head_size_idx * padded_seq_len + seq_idx]) * scale_val);
}

void invokeTmpHanldVCache(__nv_bfloat16* dst_v,
                          __nv_fp8_e4m3* src_v,
                          const float*   scale,
                          int            batch_size,
                          int            seq_len,
                          int            padded_seq_len,
                          int            head_num,
                          int            size_per_head,
                          cudaStream_t   stream)
{
    // from [batch, head_num, size_per_head, seq_len_paaded] to [batch, head_num, seq_len, size_per_head]
    int  block_sz = 128;
    dim3 grid((seq_len * size_per_head + block_sz - 1) / block_sz, batch_size, head_num);

    tmpHanldVCache<<<grid, block_sz, 0, stream>>>(
        dst_v, src_v, scale, batch_size, seq_len, padded_seq_len, head_num, size_per_head);
}

}  // namespace fastertransformer
