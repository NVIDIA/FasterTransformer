/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <cassert>
#include <cmath>
#include <cuda_fp16.h>

#include "bfloat16_fallback_kenrels.cuh"
#include "matrix_vector_multiplication.h"
#include "reduce_kernel_utils.cuh"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"

namespace fastertransformer {

typedef struct half4 {
    half x, y, z, w;
} half4;

#ifdef ENABLE_BF16
typedef struct bf164 {
    __nv_bfloat16 x, y, z, w;
} bf164;
#endif

template<int NUM, typename T>
struct ARRAY {
    T data[NUM];
};

extern __shared__ float cgBlockReduceSumElements_shm[];

// T = float4
// weight is int8 [n, k] row-major
// input is [k]
// scale_list is [n] for per_channel quantization.
// output is [n]
// each thread deals with at least 4 cols (k)
// each block deals with nPerThread rows (n)
// assume n % nPerThread == 0 && k % 4 == 0
// grid(n/nPerThread)
template<int m, int nPerThread>
__global__ void int8WeightPerChannelLdkMultiplication(
    const char4* weight, const float4* input, const float* scale_list, void* output, const int k_4)
{

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int row_idx = bidx * nPerThread;
    const size_t b_offset = row_idx * k_4;

    using array = struct ARRAY<nPerThread, float>;
    const array scale = *((const array*)scale_list + bidx);
    array sum_list[m];
#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            sum_list[m_i].data[i] = 0.0f;
        }
    }

    for (int k_idx = tidx; k_idx < k_4; k_idx += blockDim.x) {
        float4 input_val[m];
#pragma unroll
        for (int m_i = 0; m_i < m; m_i++) {
            input_val[m_i] = input[k_idx + m_i * k_4];
        }
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            const char4 weight_val = weight[b_offset + i * k_4 + k_idx];
#pragma unroll
            for (int m_i = 0; m_i < m; m_i++) {
                sum_list[m_i].data[i] += ((static_cast<float>(weight_val.x) * input_val[m_i].x)
                                          + (static_cast<float>(weight_val.y) * input_val[m_i].y)
                                          + (static_cast<float>(weight_val.z) * input_val[m_i].z)
                                          + (static_cast<float>(weight_val.w) * input_val[m_i].w))
                                         * scale.data[i];
            }
        }
    }
#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
        cgBlockReduceSumElements<nPerThread>(sum_list[m_i].data, cgBlockReduceSumElements_shm);
        __syncthreads();
    }
    if (tidx == 0) {
        for (int m_i = 0; m_i < m; m_i++) {
            *((array*)output + bidx + m_i * gridDim.x) = sum_list[m_i];
        }
    }
}

///////////////////////////////////////////////////////////////////////
// FP16 & FP32 accumulators
// for T = half4
// weight is int8 [n, k] row-major
// input is [m, k]
// scale_list is [n] for per_channel quantization.
// output is [m, n]
// each thread deals with at least m * 4 cols (k)
// each block deals with nPerThread m * rows (n)
// assume n % nPerThread == 0 && k % 4 == 0
// grid(n/nPerThread)
template<int m, int nPerThread>
__global__ void int8WeightPerChannelLdkMultiplication(
    const char4* weight, const half4* input, const float* scale_list, void* output, const int k_4)
{

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int row_idx = bidx * nPerThread;
    const size_t b_offset = row_idx * k_4;

    using array = struct ARRAY<nPerThread, float>;
    array sum_list[m];
#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            sum_list[m_i].data[i] = 0.0f;
        }
    }

    for (int k_idx = tidx; k_idx < k_4; k_idx += blockDim.x) {
        half2 input_val_0[m];
        half2 input_val_1[m];
#pragma unroll
        for (int m_i = 0; m_i < m; m_i++) {
            const half4 input_val = input[k_idx + m_i * k_4];
            input_val_0[m_i] = {input_val.x, input_val.y};
            input_val_1[m_i] = {input_val.z, input_val.w};
        }
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            const char4 weight_val = weight[b_offset + i * k_4 + k_idx];
            const half2 weight_val_0 = {static_cast<half>(weight_val.x), static_cast<half>(weight_val.y)};
            const half2 weight_val_1 = {static_cast<half>(weight_val.z), static_cast<half>(weight_val.w)};
#pragma unroll
            for (int m_i = 0; m_i < m; m_i++) {
                const half2 weight_val_2 =
                    __hadd2(__hmul2(input_val_0[m_i], weight_val_0), __hmul2(input_val_1[m_i], weight_val_1));
                sum_list[m_i].data[i] += static_cast<float>(weight_val_2.x + weight_val_2.y);
            }
        }
    }
#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
        cgBlockReduceSumElements<nPerThread>(sum_list[m_i].data, cgBlockReduceSumElements_shm);
        __syncthreads();
    }
    if (tidx == 0) {
        using array_half = struct ARRAY<nPerThread, half>;
        const array scale = *((const array*)scale_list + bidx);
#pragma unroll
        for (int m_i = 0; m_i < m; m_i++) {
            array_half sum_list_half;
#pragma unroll
            for (int i = 0; i < nPerThread; i++) {
                sum_list_half.data[i] = __float2half_rn(sum_list[m_i].data[i] * scale.data[i]);
            }
            *((array_half*)output + bidx + m_i * gridDim.x) = sum_list_half;
        }
    }
}

#ifdef ENABLE_BF16
template<int m, int nPerThread>
__global__ void int8WeightPerChannelLdkMultiplication(
    const char4* weight, const bf164* input, const float* scale_list, void* output, const int k_4)
{

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int row_idx = bidx * nPerThread;
    const size_t b_offset = row_idx * k_4;

    using array = struct ARRAY<nPerThread, float>;
    array sum_list[m];
#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            sum_list[m_i].data[i] = 0.0f;
        }
    }

    for (int k_idx = tidx; k_idx < k_4; k_idx += blockDim.x) {
        __nv_bfloat162 input_val_0[m];
        __nv_bfloat162 input_val_1[m];
#pragma unroll
        for (int m_i = 0; m_i < m; m_i++) {
            const bf164 input_val = input[k_idx + m_i * k_4];
            input_val_0[m_i] = {input_val.x, input_val.y};
            input_val_1[m_i] = {input_val.z, input_val.w};
        }
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            const char4 weight_val = weight[b_offset + i * k_4 + k_idx];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            const __nv_bfloat162 weight_val_0 = {static_cast<__nv_bfloat16>(weight_val.x),
                                                 static_cast<__nv_bfloat16>(weight_val.y)};
            const __nv_bfloat162 weight_val_1 = {static_cast<__nv_bfloat16>(weight_val.z),
                                                 static_cast<__nv_bfloat16>(weight_val.w)};
#else
            const __nv_bfloat162 weight_val_0 = {__float2bfloat16(static_cast<float>(weight_val.x)),
                                                 __float2bfloat16(static_cast<float>(weight_val.y))};
            const __nv_bfloat162 weight_val_1 = {__float2bfloat16(static_cast<float>(weight_val.z)),
                                                 __float2bfloat16(static_cast<float>(weight_val.w))};
#endif
#pragma unroll
            for (int m_i = 0; m_i < m; m_i++) {
                const __nv_bfloat162 weight_val_2 =
                    bf16hadd2(bf16hmul2(input_val_0[m_i], weight_val_0), bf16hmul2(input_val_1[m_i], weight_val_1));
                sum_list[m_i].data[i] += static_cast<float>(weight_val_2.x + weight_val_2.y);
            }
        }
    }
#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
        cgBlockReduceSumElements<nPerThread>(sum_list[m_i].data, cgBlockReduceSumElements_shm);
        __syncthreads();
    }
    if (tidx == 0) {
        using array_half = struct ARRAY<nPerThread, __nv_bfloat16>;
        const array scale = *((const array*)scale_list + bidx);
#pragma unroll
        for (int m_i = 0; m_i < m; m_i++) {
            array_half sum_list_half;
#pragma unroll
            for (int i = 0; i < nPerThread; i++) {
                sum_list_half.data[i] = __float2bfloat16_rn(sum_list[m_i].data[i] * scale.data[i]);
            }
            *((array_half*)output + bidx + m_i * gridDim.x) = sum_list_half;
        }
    }
}
#endif
///////////////////////////////////////////////////////////////////////

#define RUN(M, TYPE)                                                                                                   \
    int8WeightPerChannelLdkMultiplication<M, nPerThread><<<grid, block, shm_size, stream>>>(                           \
        (const char4*)weight, (const TYPE*)input, scale_list, (void*)output, k / 4);

template<typename T>
void int8WeightPerChannelLdkMultiplicationLauncher(const int8_t* weight,
                                                   const T* input,
                                                   const float* scale_list,
                                                   T* output,
                                                   const int m,
                                                   const int n,
                                                   const int k,
                                                   cudaStream_t stream)
{
    const int nPerThread = 2;
    if ((n % nPerThread != 0) || (k % 4 != 0)) {
        printf("[ERROR][int8WeightPerChannelLdkMultiplicationLauncher] (%d % %d != 0) || (%d % 4 != 0).\n",
               n,
               nPerThread,
               k);
        exit(-1);
    }

    dim3 grid(n / nPerThread);
    dim3 block;
    // block size tuned with gpt-3 parameter
    if (k > 10000) {
        block.x = 256;
    }
    else if (k > 2000) {
        block.x = 128;
    }
    else {
        block.x = 64;
    }
    while (block.x * 4 > k) {
        block.x /= 2;
    }
    block.x = (block.x + 31) / 32 * 32;
    const size_t shm_size = block.x * nPerThread * sizeof(float);
    if (m == 1) {
        if (std::is_same<T, half>::value) {
            RUN(1, half4)
        }
#ifdef ENABLE_BF16
        else if (std::is_same<T, __nv_bfloat16>::value) {
            RUN(1, bf164);
        }
#endif
        else {
            RUN(1, float4)
        }
    }
    else if (m == 2) {
        if (std::is_same<T, half>::value) {
            RUN(2, half4)
        }
#ifdef ENABLE_BF16
        else if (std::is_same<T, __nv_bfloat16>::value) {
            RUN(2, bf164);
        }
#endif
        else {
            RUN(2, float4)
        }
    }
    else {
        printf("[ERROR][int8WeightPerChannelLdkMultiplicationLauncher] not support m == %d.\n", m);
        exit(-1);
    }
}

template void int8WeightPerChannelLdkMultiplicationLauncher(const int8_t* matrix,
                                                            const float* vector,
                                                            const float* scale_list,
                                                            float* output,
                                                            const int m,
                                                            const int n,
                                                            const int k,
                                                            cudaStream_t stream);

template void int8WeightPerChannelLdkMultiplicationLauncher(const int8_t* matrix,
                                                            const half* vector,
                                                            const float* scale_list,
                                                            half* output,
                                                            const int m,
                                                            const int n,
                                                            const int k,
                                                            cudaStream_t stream);

#ifdef ENABLE_BF16
template void int8WeightPerChannelLdkMultiplicationLauncher(const int8_t* matrix,
                                                            const __nv_bfloat16* vector,
                                                            const float* scale_list,
                                                            __nv_bfloat16* output,
                                                            const int m,
                                                            const int n,
                                                            const int k,
                                                            cudaStream_t stream);
#endif
/////////////////////////////////////////////////////////////////////

}  // namespace fastertransformer
