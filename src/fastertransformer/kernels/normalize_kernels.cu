/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/kernels/image_shift_partition_kernels.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"

namespace fastertransformer {

/*******************  invokeNormalizeForFMHA  ***********************/

// input should be the qkv for trt fmha kernels with shape of [batch, seqlen, num_head, 3, size_per_head]
// do normlization on size_per_head for q & k [-, -, -, 0, *] && [-, -, -, 1, *] && *(logit_scale/sqrt(size_per_head))
// for q (since fmha will divide sqrt(size_per_head), we multiple sqrt(size_per_head) first)
// grid(2*num_head, seqlen, batch)
// block((size_per_head/2 + 31)/31*32)
__global__ void normalize_for_FMHA_kernel(
    half2* data, const half* logit_scales, int batch, int seqlen, int num_head, int size_per_head, float factor_of_fmha)
{
    const int        batch_seqlen_id = blockIdx.z * seqlen + blockIdx.y;
    const int        head_id         = blockIdx.x / 2;
    const int        qkv_id          = blockIdx.x % 2;
    const int        size_id         = threadIdx.x;
    const size_t     input_idx   = ((batch_seqlen_id * num_head + head_id) * 3 + qkv_id) * size_per_head / 2 + size_id;
    const float      logit_scale = ((qkv_id == 0) ? float(logit_scales[head_id]) * factor_of_fmha : 1.0f);
    const half2      zero        = {half(0.0f), half(0.0f)};
    const bool       flag        = size_id < size_per_head / 2;
    half2            input_val   = (flag) ? data[input_idx] : zero;
    float2           input_val_float2 = __half22float2(input_val);
    __shared__ float norm_factor;
    const float      local_sum     = input_val_float2.x * input_val_float2.x + input_val_float2.y * input_val_float2.y;
    const float      local_sum_all = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum(local_sum);

    if (threadIdx.x == 0) {
        norm_factor = rsqrtf(local_sum_all + 1e-6) * logit_scale;
    }
    __syncthreads();
    input_val_float2.x *= norm_factor;
    input_val_float2.y *= norm_factor;
    input_val = __float22half2_rn(input_val_float2);
    if (flag) {
        data[input_idx] = input_val;
    }
}

// This kernel is designed for size_per_head = 32, typical case in swin
// input should be the qkv for trt fmha kernels with shape of [batch, seqlen, num_head, 3, size_per_head]
// do normlization on size_per_head for q & k [-, -, -, 0, *] && [-, -, -, 1, *] && *(logit_scale/sqrt(size_per_head))
// for q (since fmha will divide sqrt(size_per_head), we multiple sqrt(size_per_head) first)
// grid(batch*seqlen*num_head/(LOOP*HEADS_PER_WARP*WARPS_PER_BLOCK))
// block(32*WARPS_PER_BLOCK) each warp load 64*LOOP*HEADS_PER_WARP elements (LOOP*HEADS_PER_WARP heads of q&k) and do
// normlization
template<int LOOP, int HEADS_PER_WARP, int WARPS_PER_BLOCK>
__global__ void normalize_for_FMHA_headz32_kernel(
    half2* data, const half* logit_scales, int batch, int seqlen, int num_head, int size_per_head, float factor_of_fmha)
{

    __shared__ half2 data_shm[LOOP * WARPS_PER_BLOCK * HEADS_PER_WARP * 32];
    __shared__ float norm_factor[LOOP * WARPS_PER_BLOCK * HEADS_PER_WARP * 2];
    const int        batch_seqlen_head_offset         = blockIdx.x * LOOP * WARPS_PER_BLOCK * HEADS_PER_WARP;
    const int        seqlen_head_offset               = batch_seqlen_head_offset % (seqlen * num_head);
    const int        tid                              = threadIdx.x;
    const int        warp_id                          = tid / 32;
    const int        tid_in_warp                      = tid % 32;
    const int        HEADS_PER_WARP_x_WARPS_PER_BLOCK = HEADS_PER_WARP * WARPS_PER_BLOCK;

// load from gmem to smem
#pragma unroll
    for (int loop_i = 0; loop_i < LOOP; loop_i++) {
        const int head_offset = loop_i * HEADS_PER_WARP_x_WARPS_PER_BLOCK + warp_id * HEADS_PER_WARP;
#pragma unroll
        for (int head_i = 0; head_i < HEADS_PER_WARP; head_i++) {
            // one warp loads one head (32 threads load 32 half2)
            const size_t input_idx =
                ((batch_seqlen_head_offset + head_offset + head_i) * 3) * size_per_head / 2 + tid_in_warp;
            // we need to ensure no out of memory address when launch kernel.
            const half2 input_val = data[input_idx];
            const int   shm_idx   = (head_offset + head_i) * 32 + tid_in_warp;
            data_shm[shm_idx]     = input_val;
        }
    }
    __syncthreads();

    // we use one warp to deal with HEADS_PER_WARP heads at one time,
    // so one thread deals with part of one single head at one time
    float     local_sums[LOOP];
    const int threads_per_head = 32 / HEADS_PER_WARP;
    // each head has 32 half2
    const int half2Size_per_thread = 32 / threads_per_head;
    const int head_in_warp         = tid_in_warp / threads_per_head;
    const int id_offset_in_head    = tid_in_warp % threads_per_head;
    const int size_offset_in_head  = half2Size_per_thread * id_offset_in_head;

    const int head_offset_of_warp = warp_id * HEADS_PER_WARP + head_in_warp;

#pragma unroll
    for (int loop_i = 0; loop_i < LOOP; loop_i++) {
        float     local_sum = 0.0f;
        const int shm_offset =
            (loop_i * HEADS_PER_WARP_x_WARPS_PER_BLOCK + head_offset_of_warp) * 32 + size_offset_in_head;
#pragma unroll
        for (int size_i = 0; size_i < half2Size_per_thread; size_i++) {
            const int    shm_idx = shm_offset + size_i;
            const float2 tmp     = __half22float2(data_shm[shm_idx]);
            local_sum += tmp.x * tmp.x + tmp.y * tmp.y;
        }
        local_sums[loop_i] = local_sum;
    }

    const int  threads_per_head_2 = threads_per_head / 2;
    const bool is_q               = id_offset_in_head < threads_per_head_2;
#pragma unroll
    for (int loop_i = 0; loop_i < LOOP; loop_i++) {
        const int seqlen_head_id = seqlen_head_offset + loop_i * HEADS_PER_WARP_x_WARPS_PER_BLOCK + head_offset_of_warp;
        const int head_id        = seqlen_head_id % num_head;
        float     local_sum      = local_sums[loop_i];
#pragma unroll
        for (int i = 1; i < threads_per_head_2; i <<= 1) {
            local_sum += __shfl_xor_sync(FINAL_MASK, local_sum, i, 32);
        }
        if (id_offset_in_head % threads_per_head_2 == 0) {
            const float logit_scale     = is_q ? float(logit_scales[head_id]) * factor_of_fmha : 1.f;
            const int   norm_factor_idx = 2 * (loop_i * HEADS_PER_WARP_x_WARPS_PER_BLOCK + head_offset_of_warp)
                                        + id_offset_in_head / threads_per_head_2;
            norm_factor[norm_factor_idx] = rsqrtf(local_sum + 1e-6) * logit_scale;
        }
    }
    __syncthreads();

// normalize and store to gmem
#pragma unroll
    for (int loop_i = 0; loop_i < LOOP; loop_i++) {
        const int head_offset = loop_i * HEADS_PER_WARP_x_WARPS_PER_BLOCK + warp_id * HEADS_PER_WARP;
#pragma unroll
        for (int head_i = 0; head_i < HEADS_PER_WARP; head_i++) {
            // we need to ensure no out of memory address when launch kernel, one warp deals with one head
            const size_t output_idx =
                ((batch_seqlen_head_offset + head_offset + head_i) * 3) * size_per_head / 2 + tid_in_warp;
            const int head_idx        = loop_i * HEADS_PER_WARP_x_WARPS_PER_BLOCK + warp_id * HEADS_PER_WARP + head_i;
            const int shm_idx         = (head_idx)*32 + tid_in_warp;
            const int norm_factor_idx = 2 * (head_idx) + tid_in_warp / 16;
            float     norm_factor_    = norm_factor[norm_factor_idx];
            half2     input_val       = data_shm[shm_idx];
            float2    input_val_float = __half22float2(input_val);
            input_val_float.x *= norm_factor_;
            input_val_float.y *= norm_factor_;
            data[output_idx] = __float22half2_rn(input_val_float);
        }
    }
}

#define NORMALIZE_FMHA_HEAD32_MACRO(LOOP, HEADS_PER_WARP, WARPS_PER_BLOCK)                                             \
    dim3 grid(batch* seqlen_num_head / (LOOP * HEADS_PER_WARP * WARPS_PER_BLOCK));                                     \
    dim3 block(32 * WARPS_PER_BLOCK);                                                                                  \
    normalize_for_FMHA_headz32_kernel<LOOP, HEADS_PER_WARP, WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(              \
        (half2*)data, (const half*)logit_scales, batch, seqlen, num_head, size_per_head, sqrt(size_per_head));

// input should be the qkv for trt fmha kernels with shape of [batch, seqlen, num_head, 3, size_per_head]
// do normlization on size_per_head for q & k [-, -, -, 0, *] && [-, -, -, 1, *] && *(logit_scale/sqrt(size_per_head))
// for q (since fmha will divide sqrt(size_per_head), we multiple sqrt(size_per_head) first)
template<typename T>
void invokeNormalizeForFMHA(
    T* data, const T* logit_scales, int batch, int seqlen, int num_head, int size_per_head, cudaStream_t stream)
{
    if (std::is_same<T, half>::value) {
        if (size_per_head == 32) {
            const int seqlen_num_head = seqlen * num_head;
            // LOOP = 2, HEADS_PER_WARP = 4, WARPS_PER_BLOCK = 2
            if (seqlen_num_head % (2 * 4 * 2) == 0) {
                NORMALIZE_FMHA_HEAD32_MACRO(2, 4, 2)
            }
            // LOOP = 1, HEADS_PER_WARP = 4, WARPS_PER_BLOCK = 2
            else if (seqlen_num_head % (1 * 4 * 2) == 0) {
                NORMALIZE_FMHA_HEAD32_MACRO(1, 4, 2)
            }
            // LOOP = 1, HEADS_PER_WARP = 2, WARPS_PER_BLOCK = 2
            else if (seqlen_num_head % (1 * 2 * 2) == 0) {
                NORMALIZE_FMHA_HEAD32_MACRO(1, 2, 2)
            }
            // LOOP = 1, HEADS_PER_WARP = 2, WARPS_PER_BLOCK = 1
            else if (seqlen_num_head % (1 * 2 * 1) == 0) {
                NORMALIZE_FMHA_HEAD32_MACRO(1, 2, 1)
            }
            // LOOP = 1, HEADS_PER_WARP = 1, WARPS_PER_BLOCK = 1
            else {
                NORMALIZE_FMHA_HEAD32_MACRO(1, 1, 1)
            }
        }
        else if (size_per_head % 2 == 0) {
            dim3 grid(2 * num_head, seqlen, batch);
            dim3 block((size_per_head / 2 + 31) / 32 * 32);
            normalize_for_FMHA_kernel<<<grid, block, 0, stream>>>(
                (half2*)data, (const half*)logit_scales, batch, seqlen, num_head, size_per_head, sqrt(size_per_head));
        }
        else {
            printf("[ERROR][invokeNormalizeForFMHA] only supports size_per_head %% 2 == 0!\n");
            exit(-1);
        }
    }
    else {
        printf("[ERROR][invokeNormalizeForFMHA] only supports half I/O!\n");
        exit(-1);
    }
}

#undef NORMALIZE_FMHA_HEAD32_MACRO

template void invokeNormalizeForFMHA<half>(
    half* data, const half* logit_scales, int batch, int seqlen, int num_head, int size_per_head, cudaStream_t stream);

template void invokeNormalizeForFMHA<float>(float*       data,
                                            const float* logit_scales,
                                            int          batch,
                                            int          seqlen,
                                            int          num_head,
                                            int          size_per_head,
                                            cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeNormalizeForFMHA<__nv_bfloat16>(__nv_bfloat16*       data,
                                                    const __nv_bfloat16* logit_scales,
                                                    int                  batch,
                                                    int                  seqlen,
                                                    int                  num_head,
                                                    int                  size_per_head,
                                                    cudaStream_t         stream);

#endif

// This kernel is designed for size_per_head = 32, typical case in swin
// input should be the qkv for trt fmha kernels with shape of [batch, seqlen, num_head, 3, size_per_head]
// do normlization on size_per_head for q & k [-, -, -, 0, *] && [-, -, -, 1, *] && *(logit_scale/sqrt(size_per_head))
// for q (since fmha will divide sqrt(size_per_head), we multiple sqrt(size_per_head) first)
// grid(batch*seqlen*num_head/(LOOP*HEADS_PER_WARP*WARPS_PER_BLOCK))
// block(32*WARPS_PER_BLOCK) each warp load 64*LOOP*HEADS_PER_WARP elements (LOOP*HEADS_PER_WARP heads of q&k) and do
// normlization
template<int LOOP, int HEADS_PER_WARP, int WARPS_PER_BLOCK>
__global__ void normalize_for_FMHA_headz32_INT8_kernel(char2*      data,
                                                       const half* logit_scales,
                                                       int         batch,
                                                       int         seqlen,
                                                       int         num_head,
                                                       int         size_per_head,
                                                       const float query_deQ_scale,
                                                       const float key_deQ_scale,
                                                       const float query_Q_scale,
                                                       const float key_Q_scale)
{

    __shared__ char2 data_shm[LOOP * WARPS_PER_BLOCK * HEADS_PER_WARP * 32];
    __shared__ float norm_factor[LOOP * WARPS_PER_BLOCK * HEADS_PER_WARP * 2];
    const int        batch_seqlen_head_offset         = blockIdx.x * LOOP * WARPS_PER_BLOCK * HEADS_PER_WARP;
    const int        seqlen_head_offset               = batch_seqlen_head_offset % (seqlen * num_head);
    const int        tid                              = threadIdx.x;
    const int        warp_id                          = tid / 32;
    const int        tid_in_warp                      = tid % 32;
    const int        HEADS_PER_WARP_x_WARPS_PER_BLOCK = HEADS_PER_WARP * WARPS_PER_BLOCK;

// load from gmem to smem
#pragma unroll
    for (int loop_i = 0; loop_i < LOOP; loop_i++) {
        const int head_offset = loop_i * HEADS_PER_WARP_x_WARPS_PER_BLOCK + warp_id * HEADS_PER_WARP;
#pragma unroll
        for (int head_i = 0; head_i < HEADS_PER_WARP; head_i++) {
            // one warp loads one head (32 threads load 32 half2)
            const size_t input_idx =
                ((batch_seqlen_head_offset + head_offset + head_i) * 3) * size_per_head / 2 + tid_in_warp;
            // we need to ensure no out of memory address when launch kernel.
            const char2 input_val = data[input_idx];
            const int   shm_idx   = (head_offset + head_i) * 32 + tid_in_warp;
            data_shm[shm_idx]     = input_val;
        }
    }
    __syncthreads();

    // we use one warp to deal with HEADS_PER_WARP heads at one time,
    // so one thread deals with part of one single head at one time
    float     local_sums[LOOP];
    const int threads_per_head = 32 / HEADS_PER_WARP;
    // each head has 32 half2
    const int half2Size_per_thread = 32 / threads_per_head;
    const int head_in_warp         = tid_in_warp / threads_per_head;
    const int id_offset_in_head    = tid_in_warp % threads_per_head;
    const int size_offset_in_head  = half2Size_per_thread * id_offset_in_head;

    const int head_offset_of_warp = warp_id * HEADS_PER_WARP + head_in_warp;

    const int  threads_per_head_2 = threads_per_head / 2;
    const bool is_q               = id_offset_in_head < threads_per_head_2;
    float      deQ_scale          = is_q ? query_deQ_scale : key_deQ_scale;
    float      Q_scale            = is_q ? query_Q_scale : key_Q_scale;

#pragma unroll
    for (int loop_i = 0; loop_i < LOOP; loop_i++) {
        float     local_sum = 0.0f;
        const int shm_offset =
            (loop_i * HEADS_PER_WARP_x_WARPS_PER_BLOCK + head_offset_of_warp) * 32 + size_offset_in_head;
#pragma unroll
        for (int size_i = 0; size_i < half2Size_per_thread; size_i++) {
            const int   shm_idx = shm_offset + size_i;
            const char2 tmp     = data_shm[shm_idx];
            float2      tmpFloat;
            tmpFloat.x = static_cast<float>(tmp.x) * deQ_scale;
            tmpFloat.y = static_cast<float>(tmp.y) * deQ_scale;
            local_sum += tmpFloat.x * tmpFloat.x + tmpFloat.y * tmpFloat.y;
        }
        local_sums[loop_i] = local_sum;
    }

#pragma unroll
    for (int loop_i = 0; loop_i < LOOP; loop_i++) {
        const int seqlen_head_id = seqlen_head_offset + loop_i * HEADS_PER_WARP_x_WARPS_PER_BLOCK + head_offset_of_warp;
        const int head_id        = seqlen_head_id % num_head;
        float     local_sum      = local_sums[loop_i];
#pragma unroll
        for (int i = 1; i < threads_per_head_2; i <<= 1) {
            local_sum += __shfl_xor_sync(FINAL_MASK, local_sum, i, 32);
        }
        if (id_offset_in_head % threads_per_head_2 == 0) {
            const float logit_scale     = is_q ? float(logit_scales[head_id]) : 1.f;
            const int   norm_factor_idx = 2 * (loop_i * HEADS_PER_WARP_x_WARPS_PER_BLOCK + head_offset_of_warp)
                                        + id_offset_in_head / threads_per_head_2;
            norm_factor[norm_factor_idx] = rsqrtf(local_sum + 1e-6) * logit_scale;
        }
    }
    __syncthreads();

// normalize and store to gmem
#pragma unroll
    for (int loop_i = 0; loop_i < LOOP; loop_i++) {
        const int head_offset = loop_i * HEADS_PER_WARP_x_WARPS_PER_BLOCK + warp_id * HEADS_PER_WARP;
#pragma unroll
        for (int head_i = 0; head_i < HEADS_PER_WARP; head_i++) {
            // we need to ensure no out of memory address when launch kernel, one warp deals with one head
            const size_t output_idx =
                ((batch_seqlen_head_offset + head_offset + head_i) * 3) * size_per_head / 2 + tid_in_warp;
            const int head_idx        = loop_i * HEADS_PER_WARP_x_WARPS_PER_BLOCK + warp_id * HEADS_PER_WARP + head_i;
            const int shm_idx         = (head_idx)*32 + tid_in_warp;
            const int norm_factor_idx = 2 * (head_idx) + tid_in_warp / 16;
            deQ_scale                 = (tid_in_warp < 16) ? query_deQ_scale : key_deQ_scale;
            Q_scale                   = (tid_in_warp < 16) ? query_Q_scale : key_Q_scale;
            float  norm_factor_       = norm_factor[norm_factor_idx];
            char2  input_val          = data_shm[shm_idx];
            float2 input_val_float;
            input_val_float.x = static_cast<float>(input_val.x) * deQ_scale * norm_factor_ * Q_scale;
            input_val_float.y = static_cast<float>(input_val.y) * deQ_scale * norm_factor_ * Q_scale;
            input_val.x       = float_to_int8_rn(input_val_float.x);
            input_val.y       = float_to_int8_rn(input_val_float.y);
            data[output_idx]  = input_val;
        }
    }
}

// input should be the qkv for trt fmha kernels with shape of [batch, seqlen, num_head, 3, size_per_head]
// do normlization on size_per_head for q & k [-, -, -, 0, *] && [-, -, -, 1, *] && *(logit_scale/sqrt(size_per_head))
// for q (since fmha will divide sqrt(size_per_head), we multiple sqrt(size_per_head) first)
// grid(2*num_head, seqlen, batch)
// block((size_per_head/2 + 31)/31*32)
__global__ void normalize_for_FMHA_INT8_kernel(char2*      data,
                                               const half* logit_scales,
                                               int         batch,
                                               int         seqlen,
                                               int         num_head,
                                               int         size_per_head,
                                               const float query_deQ_scale,
                                               const float key_deQ_scale,
                                               const float query_Q_scale,
                                               const float key_Q_scale)
{
    const int    batch_seqlen_id  = blockIdx.z * seqlen + blockIdx.y;
    const int    head_id          = blockIdx.x / 2;
    const int    qkv_id           = blockIdx.x % 2;
    const int    size_id          = threadIdx.x;
    const size_t input_idx        = ((batch_seqlen_id * num_head + head_id) * 3 + qkv_id) * size_per_head / 2 + size_id;
    const float  logit_scale      = ((qkv_id == 0) ? float(logit_scales[head_id]) : 1.0f);
    const float  deQ_scale        = (qkv_id == 0) ? query_deQ_scale : key_deQ_scale;
    const float  Q_scale          = (qkv_id == 0) ? query_Q_scale : key_Q_scale;
    float2       input_val_float2 = {0.0f, 0.0f};
    const bool   flag             = size_id < size_per_head / 2;
    if (flag) {
        char2 input_data   = data[input_idx];
        input_val_float2.x = static_cast<float>(input_data.x) * deQ_scale;
        input_val_float2.y = static_cast<float>(input_data.y) * deQ_scale;
    }
    __shared__ float norm_factor;
    const float      local_sum     = input_val_float2.x * input_val_float2.x + input_val_float2.y * input_val_float2.y;
    const float      local_sum_all = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum(local_sum);

    if (threadIdx.x == 0) {
        norm_factor = rsqrtf(local_sum_all + 1e-6) * logit_scale * Q_scale;
    }
    __syncthreads();
    if (flag) {
        char2 output_val;
        output_val.x    = float_to_int8_rn(input_val_float2.x * norm_factor);
        output_val.y    = float_to_int8_rn(input_val_float2.y * norm_factor);
        data[input_idx] = output_val;
    }
}

#define NORMALIZE_FMHA_HEAD32_MACRO(LOOP, HEADS_PER_WARP, WARPS_PER_BLOCK)                                             \
    dim3 grid(batch* seqlen_num_head / (LOOP * HEADS_PER_WARP * WARPS_PER_BLOCK));                                     \
    dim3 block(32 * WARPS_PER_BLOCK);                                                                                  \
    normalize_for_FMHA_headz32_INT8_kernel<LOOP, HEADS_PER_WARP, WARPS_PER_BLOCK>                                      \
        <<<grid, block, 0, stream>>>((char2*)data,                                                                     \
                                     (const half*)logit_scales,                                                        \
                                     batch,                                                                            \
                                     seqlen,                                                                           \
                                     num_head,                                                                         \
                                     size_per_head,                                                                    \
                                     query_deQ_scale,                                                                  \
                                     key_deQ_scale,                                                                    \
                                     query_Q_scale,                                                                    \
                                     key_Q_scale);

// input should be the qkv for trt fmha kernels with shape of [batch, seqlen, num_head, 3, size_per_head]
// do normlization on size_per_head for q & k [-, -, -, 0, *] && [-, -, -, 1, *] && *(logit_scale/sqrt(size_per_head))
// for q (since fmha will divide sqrt(size_per_head), we multiple sqrt(size_per_head) first)
template<typename T>
void invokeNormalizeForFMHA(int8_t*      data,
                            const T*     logit_scales,
                            int          batch,
                            int          seqlen,
                            int          num_head,
                            int          size_per_head,
                            cudaStream_t stream,
                            const float  query_deQ_scale,
                            const float  key_deQ_scale,
                            const float  query_Q_scale,
                            const float  key_Q_scale)
{
    if (std::is_same<T, half>::value) {
        if (size_per_head == 32) {
            const int seqlen_num_head = seqlen * num_head;
            if (seqlen_num_head % (2 * 4 * 2) == 0) {
                NORMALIZE_FMHA_HEAD32_MACRO(2, 4, 2)
            }
            else if (seqlen_num_head % (1 * 4 * 2) == 0) {
                NORMALIZE_FMHA_HEAD32_MACRO(1, 4, 2)
            }
            else if (seqlen_num_head % (1 * 2 * 2) == 0) {
                NORMALIZE_FMHA_HEAD32_MACRO(1, 2, 2)
            }
            else if (seqlen_num_head % (1 * 2 * 1) == 0) {
                NORMALIZE_FMHA_HEAD32_MACRO(1, 2, 1)
            }
            else {
                NORMALIZE_FMHA_HEAD32_MACRO(1, 1, 1)
            }
        }
        else if (size_per_head % 2 == 0) {
            dim3 grid(2 * num_head, seqlen, batch);
            dim3 block((size_per_head / 2 + 31) / 32 * 32);
            normalize_for_FMHA_INT8_kernel<<<grid, block, 0, stream>>>((char2*)data,
                                                                       (const half*)logit_scales,
                                                                       batch,
                                                                       seqlen,
                                                                       num_head,
                                                                       size_per_head,
                                                                       query_deQ_scale,
                                                                       key_deQ_scale,
                                                                       query_Q_scale,
                                                                       key_Q_scale);
        }
        else {
            printf("[ERROR][invokeNormalizeForFMHA(INT8 version)] only supports size_per_head %% 2 == 0!\n");
            exit(-1);
        }
    }
    else {
        printf("[ERROR][invokeNormalizeForFMHA(INT8 version)] only supports half Input!\n");
        exit(-1);
    }
}

#undef NORMALIZE_FMHA_HEAD32_MACRO

template void invokeNormalizeForFMHA<float>(int8_t*      data,
                                            const float* logit_scales,
                                            int          batch,
                                            int          seqlen,
                                            int          num_head,
                                            int          size_per_head,
                                            cudaStream_t stream,
                                            const float  query_deQ_scale,
                                            const float  key_deQ_scale,
                                            const float  query_Q_scale,
                                            const float  key_Q_scale);

template void invokeNormalizeForFMHA<half>(int8_t*      data,
                                           const half*  logit_scales,
                                           int          batch,
                                           int          seqlen,
                                           int          num_head,
                                           int          size_per_head,
                                           cudaStream_t stream,
                                           const float  query_deQ_scale,
                                           const float  key_deQ_scale,
                                           const float  query_Q_scale,
                                           const float  key_Q_scale);

/*******************  invokeNormalize  ***********************/

// input should be [batch, num_head, seqlen, size_per_head]
// do normlization on size_per_head && *(logit_scale)
// grid(seqlen, num_head, batch)
// block((size_per_head + 31)/31*32)
// TODO : the trick of normalize_for_FMHA_headz32_kernel can be used here
template<typename T, typename T2>
__global__ void
normalize_kernel(T* data, const T2* logit_scales, int batch, int seqlen, int num_head, int size_per_head)
{
    const int        batch_id    = blockIdx.z;
    const int        head_id     = blockIdx.y;
    const int        seq_id      = blockIdx.x;
    const int        size_id     = threadIdx.x;
    const int        input_idx   = ((batch_id * num_head + head_id) * seqlen + seq_id) * size_per_head + size_id;
    const float      logit_scale = (logit_scales != NULL) ? static_cast<float>(logit_scales[head_id]) : 1.0f;
    const bool       flag        = size_id < size_per_head;
    float            input_val   = (flag) ? static_cast<float>(data[input_idx]) : 0.0f;
    __shared__ float norm_factor;
    const float      local_sum     = input_val * input_val;
    const float      local_sum_all = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum(local_sum);

    if (threadIdx.x == 0) {
        norm_factor = rsqrt(local_sum_all + 1e-6) * logit_scale;
    }
    __syncthreads();
    input_val *= norm_factor;
    if (flag) {
        data[input_idx] = input_val;
    }
}

// input should be [batch, num_head, seqlen, size_per_head]
// do normlization on size_per_head && *(logit_scale/sqrt(size_per_head))
// grid(seqlen, num_head, batch)
// block((size_per_head/2 + 31)/31*32)
// TODO : the trick of normalize_for_FMHA_headz32_kernel can be used here
template<>
__global__ void
normalize_kernel(half2* data, const half* logit_scales, int batch, int seqlen, int num_head, int size_per_head)
{
    const int        batch_id    = blockIdx.z;
    const int        head_id     = blockIdx.y;
    const int        seq_id      = blockIdx.x;
    const int        size_id     = threadIdx.x;
    const int        input_idx   = ((batch_id * num_head + head_id) * seqlen + seq_id) * size_per_head / 2 + size_id;
    const float      logit_scale = (logit_scales != NULL) ? float(logit_scales[head_id]) : 1.0f;
    const half2      zero        = {half(0.0f), half(0.0f)};
    const bool       flag        = size_id < size_per_head / 2;
    half2            input_val   = (flag) ? data[input_idx] : zero;
    float2           input_val_float2 = __half22float2(input_val);
    __shared__ float norm_factor;
    const float      local_sum     = input_val_float2.x * input_val_float2.x + input_val_float2.y * input_val_float2.y;
    const float      local_sum_all = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum(local_sum);

    if (threadIdx.x == 0) {
        norm_factor = rsqrtf(local_sum_all + 1e-6) * logit_scale;
    }
    __syncthreads();
    input_val_float2.x *= norm_factor;
    input_val_float2.y *= norm_factor;
    input_val = __float22half2_rn(input_val_float2);
    if (flag) {
        data[input_idx] = input_val;
    }
}

extern __shared__ char normalize_kernel_v2_shm[];

// input should be [batch, num_head, seqlen, size_per_head]
// do normlization on size_per_head && *(logit_scale)
// grid(batch*num_head*seqlen/(HEADS_PER_WARP*WARPS_PER_BLOCK))
// block(32*WARPS_PER_BLOCK)
// size_per_head % ELEMENT_PER_LDG == 0
template<typename T_IO, typename T, int HEADS_PER_WARP, int WARPS_PER_BLOCK, int ELEMENT_PER_LDG>
__global__ void
normalize_kernel_v2(T_IO* data, const T* logit_scales, int batch, int seqlen, int num_head, int size_per_head)
{
    const int   tidx                 = threadIdx.x;
    const int   bidx                 = blockIdx.x;
    const int   threads_per_head     = 32 / HEADS_PER_WARP;
    const int   lane_id              = tidx % 32;
    const int   warp_id              = tidx / 32;
    const int   head_offset_of_grid  = bidx * WARPS_PER_BLOCK * HEADS_PER_WARP;
    const int   head_offset_of_block = warp_id * HEADS_PER_WARP;
    const int   head_id_in_warp      = lane_id / threads_per_head;
    const int   head_id              = head_offset_of_grid + head_offset_of_block + head_id_in_warp;
    const int   num_head_id          = (head_id / seqlen) % num_head;
    const int   ldg_per_head         = size_per_head / ELEMENT_PER_LDG;
    const float logit_scale          = (logit_scales != NULL) ? static_cast<float>(logit_scales[num_head_id]) : 1.0f;

    T_IO*            normalize_shm = (T_IO*)normalize_kernel_v2_shm;
    __shared__ float norm_factor[WARPS_PER_BLOCK * HEADS_PER_WARP];  // one factor for one head
    const int        input_offset = head_offset_of_grid * ldg_per_head;
    // load from gmem to smem
#pragma unroll
    for (int i = tidx; i < HEADS_PER_WARP * WARPS_PER_BLOCK * ldg_per_head; i += blockDim.x) {
        const int input_idx = input_offset + i;
        normalize_shm[i]    = data[input_idx];
    }
    __syncthreads();

    // local sum
    float     local_sum           = 0.0f;
    const int elements_per_thread = (size_per_head + threads_per_head - 1) / threads_per_head;
    const int thread_id_in_head   = tidx % threads_per_head;
    const int size_offset_in_head = elements_per_thread * thread_id_in_head;
    const int shm_offset          = (head_offset_of_block + head_id_in_warp) * size_per_head + size_offset_in_head;
    const T*  shm_ptr             = (const T*)normalize_shm;
#pragma unroll
    for (int size_i = 0; size_i < elements_per_thread && (size_i + size_offset_in_head < size_per_head); size_i++) {
        const int   shm_idx = shm_offset + size_i;
        const float tmp     = static_cast<float>(shm_ptr[shm_idx]);
        local_sum += tmp * tmp;
    }

    // reduction to get norm_factor
#pragma unroll
    for (int i = 1; i < threads_per_head; i <<= 1) {
        local_sum += __shfl_xor_sync(FINAL_MASK, local_sum, i, 32);
    }
    if (thread_id_in_head == 0) {
        const int norm_factor_idx    = head_offset_of_block + head_id_in_warp;
        norm_factor[norm_factor_idx] = rsqrtf(local_sum + 1e-6) * logit_scale;
    }
    __syncthreads();

// normalize and sts to gmem
#pragma unroll
    for (int i = tidx; i < HEADS_PER_WARP * WARPS_PER_BLOCK * ldg_per_head; i += blockDim.x) {
        const int   norm_factor_idx = i / ldg_per_head;
        const float norm_factor_val = norm_factor[norm_factor_idx];
        T_IO        val             = normalize_shm[i];
        T*          val_ptr         = (T*)(&val);
#pragma unroll
        for (int ei = 0; ei < ELEMENT_PER_LDG; ei++) {
            val_ptr[ei] = T(static_cast<float>(val_ptr[ei]) * norm_factor_val);
        }
        const int input_idx = input_offset + i;
        data[input_idx]     = val;
    }
}

#define NORMALIZE_MACRO(HEADS_PER_WARP_, T_4, T_2, T)                                                                  \
    dim3      grid(total_head_count / (HEADS_PER_WARP_ * WARPS_PER_BLOCK));                                            \
    dim3      block(32 * WARPS_PER_BLOCK);                                                                             \
    const int shm_size = HEADS_PER_WARP_ * WARPS_PER_BLOCK * size_per_head * sizeof(T);                                \
    if (size_per_head % 4 == 0) {                                                                                      \
        normalize_kernel_v2<T_4, T, HEADS_PER_WARP_, WARPS_PER_BLOCK, 4><<<grid, block, shm_size, stream>>>(           \
            (T_4*)data, (const T*)logit_scales, batch, seqlen, num_head, size_per_head);                               \
    }                                                                                                                  \
    else if (size_per_head % 2 == 0) {                                                                                 \
        normalize_kernel_v2<T_2, T, HEADS_PER_WARP_, WARPS_PER_BLOCK, 2><<<grid, block, shm_size, stream>>>(           \
            (T_2*)data, (const T*)logit_scales, batch, seqlen, num_head, size_per_head);                               \
    }                                                                                                                  \
    else {                                                                                                             \
        normalize_kernel_v2<T, T, HEADS_PER_WARP_, WARPS_PER_BLOCK, 1><<<grid, block, shm_size, stream>>>(             \
            (T*)data, (const T*)logit_scales, batch, seqlen, num_head, size_per_head);                                 \
    }

// input should be [batch, num_head, seqlen, size_per_head]
// do normlization on size_per_head && *(logit_scale/sqrt(size_per_head))
template<typename T>
void invokeNormalize(
    T* data, const T* logit_scales, int batch, int seqlen, int num_head, int size_per_head, cudaStream_t stream)
{
    const int WARPS_PER_BLOCK  = 4;
    const int total_head_count = batch * num_head * seqlen;
    if (std::is_same<T, float>::value) {
        // WARPS_PER_BLOCK = 4, HEADS_PER_WARP = 4
        if (total_head_count % (WARPS_PER_BLOCK * 4) == 0) {
            NORMALIZE_MACRO(4, float4, float2, float);
        }
        // WARPS_PER_BLOCK = 4, HEAD_PER_WARPS = 2
        else if (total_head_count % (WARPS_PER_BLOCK * 2) == 0) {
            NORMALIZE_MACRO(2, float4, float2, float);
        }
        // WARPS_PER_BLOCK = 4, HEAD_PER_WARPS = 1
        else if (total_head_count % WARPS_PER_BLOCK == 0) {
            NORMALIZE_MACRO(1, float4, float2, float);
        }
        else {
            dim3 grid(seqlen, num_head, batch);
            dim3 block((size_per_head + 31) / 32 * 32);
            normalize_kernel<<<grid, block, 0, stream>>>(
                (float*)data, (const float*)logit_scales, batch, seqlen, num_head, size_per_head);
        }
    }
    else if (std::is_same<T, half>::value) {
        // WARPS_PER_BLOCK = 4, HEADS_PER_WARP = 4
        if (total_head_count % (WARPS_PER_BLOCK * 4) == 0) {
            NORMALIZE_MACRO(4, half4, half2, half);
        }
        // WARPS_PER_BLOCK = 4, HEAD_PER_WARPS = 2
        else if (total_head_count % (WARPS_PER_BLOCK * 2) == 0) {
            NORMALIZE_MACRO(2, half4, half2, half);
        }
        // WARPS_PER_BLOCK = 4, HEAD_PER_WARPS = 1
        else if (total_head_count % WARPS_PER_BLOCK == 0) {
            NORMALIZE_MACRO(1, half4, half2, half);
        }
        else {
            if (size_per_head % 2 == 0) {
                dim3 grid(seqlen, num_head, batch);
                dim3 block((size_per_head / 2 + 31) / 32 * 32);
                normalize_kernel<<<grid, block, 0, stream>>>(
                    (half2*)data, (const half*)logit_scales, batch, seqlen, num_head, size_per_head);
            }
            else {
                dim3 grid(seqlen, num_head, batch);
                dim3 block((size_per_head + 31) / 32 * 32);
                normalize_kernel<<<grid, block, 0, stream>>>(
                    (half*)data, (const half*)logit_scales, batch, seqlen, num_head, size_per_head);
            }
        }
    }
#ifdef ENABLE_BF16
    else {
        dim3 grid(seqlen, num_head, batch);
        dim3 block((size_per_head + 31) / 32 * 32);
        normalize_kernel<<<grid, block, 0, stream>>>(data, logit_scales, batch, seqlen, num_head, size_per_head);
    }
#endif
}

#undef NORMALIZE_MACRO

template void invokeNormalize<half>(
    half* data, const half* logit_scales, int batch, int seqlen, int num_head, int size_per_head, cudaStream_t stream);

template void invokeNormalize<float>(float*       data,
                                     const float* logit_scales,
                                     int          batch,
                                     int          seqlen,
                                     int          num_head,
                                     int          size_per_head,
                                     cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeNormalize<__nv_bfloat16>(__nv_bfloat16*       data,
                                             const __nv_bfloat16* logit_scales,
                                             int                  batch,
                                             int                  seqlen,
                                             int                  num_head,
                                             int                  size_per_head,
                                             cudaStream_t         stream);

#endif

/*******************  invokeNormalize  ***********************/

// input should be [batch, num_head, seqlen, size_per_head]
// do normlization on size_per_head && *(logit_scale)
// grid(seqlen, num_head, batch)
// block((size_per_head + 31)/31*32)
template<typename T>
__global__ void normalize_kernel(int8_t*     data,
                                 const T*    logit_scales,
                                 int         batch,
                                 int         seqlen,
                                 int         num_head,
                                 int         size_per_head,
                                 const float deQ_scale,
                                 const float Q_scale)
{
    const int        batch_id    = blockIdx.z;
    const int        head_id     = blockIdx.y;
    const int        seq_id      = blockIdx.x;
    const int        size_id     = threadIdx.x;
    const int        input_idx   = ((batch_id * num_head + head_id) * seqlen + seq_id) * size_per_head + size_id;
    const float      logit_scale = (logit_scales != NULL) ? static_cast<float>(logit_scales[head_id]) : 1.0f;
    const bool       flag        = size_id < size_per_head;
    float            input_val   = (flag) ? static_cast<float>(data[input_idx]) * deQ_scale : 0.0f;
    __shared__ float norm_factor;
    const float      local_sum     = input_val * input_val;
    const float      local_sum_all = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum(local_sum);

    if (threadIdx.x == 0) {
        norm_factor = rsqrt(local_sum_all + 1e-6) * logit_scale;
    }
    __syncthreads();
    input_val *= norm_factor;
    if (flag) {
        data[input_idx] = float_to_int8_rn(input_val * Q_scale);
    }
}

// input should be [batch, num_head, seqlen, size_per_head]
// do normlization on size_per_head && *(logit_scale/sqrt(size_per_head))
// grid(seqlen, num_head, batch)
// block((size_per_head/2 + 31)/31*32)
template<>
__global__ void normalize_kernel(int8_t*     data,
                                 const half* logit_scales,
                                 int         batch,
                                 int         seqlen,
                                 int         num_head,
                                 int         size_per_head,
                                 const float deQ_scale,
                                 const float Q_scale)
{
    const int   batch_id         = blockIdx.z;
    const int   head_id          = blockIdx.y;
    const int   seq_id           = blockIdx.x;
    const int   size_id          = threadIdx.x;
    const int   input_idx        = ((batch_id * num_head + head_id) * seqlen + seq_id) * size_per_head / 2 + size_id;
    const float logit_scale      = (logit_scales != NULL) ? float(logit_scales[head_id]) : 1.0f;
    float2      input_val_float2 = {0.0f, 0.0f};
    const bool  flag             = size_id < size_per_head / 2;
    char2*      dataPtr          = (char2*)data;
    if (flag) {
        char2 dataTmp      = dataPtr[input_idx];
        input_val_float2.x = static_cast<float>(dataTmp.x) * deQ_scale;
        input_val_float2.y = static_cast<float>(dataTmp.y) * deQ_scale;
    }
    __shared__ float norm_factor;
    const float      local_sum     = input_val_float2.x * input_val_float2.x + input_val_float2.y * input_val_float2.y;
    const float      local_sum_all = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum(local_sum);

    if (threadIdx.x == 0) {
        norm_factor = rsqrtf(local_sum_all + 1e-6) * logit_scale;
    }
    __syncthreads();
    input_val_float2.x *= norm_factor;
    input_val_float2.y *= norm_factor;
    if (flag) {
        char2 dataTmp;
        dataTmp.x          = float_to_int8_rn(input_val_float2.x * Q_scale);
        dataTmp.y          = float_to_int8_rn(input_val_float2.y * Q_scale);
        dataPtr[input_idx] = dataTmp;
    }
}

// input should be [batch, num_head, seqlen, size_per_head]
// do normlization on size_per_head && *(logit_scale)
// grid(batch*num_head*seqlen/(HEADS_PER_WARP*WARPS_PER_BLOCK))
// block(32*WARPS_PER_BLOCK)
// size_per_head % ELEMENT_PER_LDG == 0
template<int HEADS_PER_WARP, int WARPS_PER_BLOCK, int ELEMENT_PER_LDG>
__global__ void normalize_kernel_v2(char4*      data,
                                    const half* logit_scales,
                                    int         batch,
                                    int         seqlen,
                                    int         num_head,
                                    int         size_per_head,
                                    const float deQ_scale,
                                    const float Q_scale)
{
    const int   tidx                 = threadIdx.x;
    const int   bidx                 = blockIdx.x;
    const int   threads_per_head     = 32 / HEADS_PER_WARP;
    const int   lane_id              = tidx % 32;
    const int   warp_id              = tidx / 32;
    const int   head_offset_of_grid  = bidx * WARPS_PER_BLOCK * HEADS_PER_WARP;
    const int   head_offset_of_block = warp_id * HEADS_PER_WARP;
    const int   head_id_in_warp      = lane_id / threads_per_head;
    const int   head_id              = head_offset_of_grid + head_offset_of_block + head_id_in_warp;
    const int   num_head_id          = (head_id / seqlen) % num_head;
    const int   ldg_per_head         = size_per_head / ELEMENT_PER_LDG;
    const float logit_scale          = (logit_scales != NULL) ? static_cast<float>(logit_scales[num_head_id]) : 1.0f;

    char4*           normalize_shm = (char4*)normalize_kernel_v2_shm;
    __shared__ float norm_factor[WARPS_PER_BLOCK * HEADS_PER_WARP];  // one factor for one head
    const int        input_offset = head_offset_of_grid * ldg_per_head;
    // load from gmem to smem
#pragma unroll
    for (int i = tidx; i < HEADS_PER_WARP * WARPS_PER_BLOCK * ldg_per_head; i += blockDim.x) {
        const int input_idx = input_offset + i;
        normalize_shm[i]    = data[input_idx];
    }
    __syncthreads();

    // local sum
    float         local_sum           = 0.0f;
    const int     elements_per_thread = (size_per_head + threads_per_head - 1) / threads_per_head;
    const int     thread_id_in_head   = tidx % threads_per_head;
    const int     size_offset_in_head = elements_per_thread * thread_id_in_head;
    const int     shm_offset          = (head_offset_of_block + head_id_in_warp) * size_per_head + size_offset_in_head;
    const int8_t* shm_ptr             = (const int8_t*)normalize_shm;
#pragma unroll
    for (int size_i = 0; size_i < elements_per_thread && (size_i + size_offset_in_head < size_per_head); size_i++) {
        const int   shm_idx = shm_offset + size_i;
        const float tmp     = static_cast<float>(shm_ptr[shm_idx]) * deQ_scale;
        local_sum += tmp * tmp;
    }

    // reduction to get norm_factor
#pragma unroll
    for (int i = 1; i < threads_per_head; i <<= 1) {
        local_sum += __shfl_xor_sync(FINAL_MASK, local_sum, i, 32);
    }
    if (thread_id_in_head == 0) {
        const int norm_factor_idx    = head_offset_of_block + head_id_in_warp;
        norm_factor[norm_factor_idx] = rsqrtf(local_sum + 1e-6) * logit_scale;
    }
    __syncthreads();

// normalize and sts to gmem
#pragma unroll
    for (int i = tidx; i < HEADS_PER_WARP * WARPS_PER_BLOCK * ldg_per_head; i += blockDim.x) {
        const int   norm_factor_idx = i / ldg_per_head;
        const float norm_factor_val = norm_factor[norm_factor_idx];
        char4       val             = normalize_shm[i];
        int8_t*     val_ptr         = (int8_t*)(&val);
#pragma unroll
        for (int ei = 0; ei < ELEMENT_PER_LDG; ei++) {
            val_ptr[ei] = float_to_int8_rn(static_cast<float>(val_ptr[ei]) * deQ_scale * norm_factor_val * Q_scale);
        }
        const int input_idx = input_offset + i;
        data[input_idx]     = val;
    }
}

#define NORMALIZE_MACRO                                                                                                \
    const int shm_size = WARPS_PER_BLOCK * HEADS_PER_WARP * size_per_head;                                             \
    dim3      grid(total_head_count / (WARPS_PER_BLOCK * HEADS_PER_WARP));                                             \
    dim3      block(32 * WARPS_PER_BLOCK);                                                                             \
    normalize_kernel_v2<HEADS_PER_WARP, WARPS_PER_BLOCK, 4><<<grid, block, shm_size, stream>>>(                        \
        (char4*)data, (const half*)logit_scales, batch, seqlen, num_head, size_per_head, deQ_scale, Q_scale);

// input should be [batch, num_head, seqlen, size_per_head]
// do normlization on size_per_head && *(logit_scale/sqrt(size_per_head))
template<typename T>
void invokeNormalize(int8_t*      data,
                     const T*     logit_scales,
                     int          batch,
                     int          seqlen,
                     int          num_head,
                     int          size_per_head,
                     cudaStream_t stream,
                     const float  deQ_scale,
                     const float  Q_scale)
{
    if (std::is_same<T, half>::value) {
        if (size_per_head % 4 == 0) {
            const int HEADS_PER_WARP   = 4;
            const int total_head_count = seqlen * num_head * batch;
            if (total_head_count % (HEADS_PER_WARP * 4) == 0) {
                const int WARPS_PER_BLOCK = 4;
                NORMALIZE_MACRO
            }
            else if (total_head_count % (HEADS_PER_WARP * 2) == 0) {
                const int WARPS_PER_BLOCK = 2;
                NORMALIZE_MACRO
            }
            else if (total_head_count % (HEADS_PER_WARP * 1) == 0) {
                const int WARPS_PER_BLOCK = 1;
                NORMALIZE_MACRO
            }
            else {
                dim3 grid(seqlen, num_head, batch);
                dim3 block((size_per_head / 2 + 31) / 32 * 32);
                normalize_kernel<<<grid, block, 0, stream>>>((int8_t*)data,
                                                             (const half*)logit_scales,
                                                             batch,
                                                             seqlen,
                                                             num_head,
                                                             size_per_head,
                                                             deQ_scale,
                                                             Q_scale);
            }
        }
        else if (size_per_head % 2 == 0) {
            dim3 grid(seqlen, num_head, batch);
            dim3 block((size_per_head / 2 + 31) / 32 * 32);
            normalize_kernel<<<grid, block, 0, stream>>>(
                (int8_t*)data, (const half*)logit_scales, batch, seqlen, num_head, size_per_head, deQ_scale, Q_scale);
        }
        else {
            printf("[ERROR][invokeNormalize(INT8 version)] only supports size_per_head %% 2 == 0!\n");
            exit(-1);
        }
    }
    else {
        printf("[ERROR][invokeNormalize(INT8 version)] only supports [T=half] !\n");
        exit(-1);
    }
}

#undef NORMALIZE_MACRO

template void invokeNormalize<half>(int8_t*      data,
                                    const half*  logit_scales,
                                    int          batch,
                                    int          seqlen,
                                    int          num_head,
                                    int          size_per_head,
                                    cudaStream_t stream,
                                    const float  deQ_scale,
                                    const float  Q_scale);

template void invokeNormalize<float>(int8_t*      data,
                                     const float* logit_scales,
                                     int          batch,
                                     int          seqlen,
                                     int          num_head,
                                     int          size_per_head,
                                     cudaStream_t stream,
                                     const float  deQ_scale,
                                     const float  Q_scale);

}  // namespace fastertransformer
