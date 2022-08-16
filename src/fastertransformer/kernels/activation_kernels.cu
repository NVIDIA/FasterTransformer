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

#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/kernels/bfloat16_fallback_kenrels.cuh"
#include "src/fastertransformer/utils/cuda_utils.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

namespace fastertransformer {

__forceinline__ __device__ float copysignf_pos(float a, float b)
{
    float r;
    r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
    return r;
}

__inline__ __device__ float tanh_opt(float x)
{
#if (__CUDA_ARCH__ >= 750 && CUDART_VERSION >= 11000)
    float r;
    asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
    return r;
#else
    const float exp_val = -1.f * fabs(2 * x);
    return copysignf_pos((1.0f - __expf(exp_val)) / (__expf(exp_val) + 1.0f), x);
#endif
}

template<typename T>
__inline__ __device__ T gelu(T x)
{
    float cdf = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (x + 0.044715f * x * x * x))));
    return x * cdf;
}

template<>
__inline__ __device__ half2 gelu(half2 val)
{
    half2  val_pow3 = __hmul2(val, __hmul2(val, val));
    float2 tmp_pow  = __half22float2(val_pow3);
    float2 tmp      = __half22float2(val);

    tmp.x = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
    tmp.y = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
    return __hmul2(val, __float22half2_rn(tmp));
}

#ifdef ENABLE_BF16
template<>
__inline__ __device__ __nv_bfloat162 gelu(__nv_bfloat162 val)
{
    __nv_bfloat162 val_pow3 = bf16hmul2(val, bf16hmul2(val, val));
    float2         tmp_pow  = bf1622float2(val_pow3);
    float2         tmp      = bf1622float2(val);

    tmp.x = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
    tmp.y = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
    return bf16hmul2(val, __floats2bfloat162_rn(tmp.x, tmp.y));
}
#endif

template<typename T>
__global__ void addBiasGelu(T* out, const T* __restrict bias, int m, int n)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        T val = out[id];
        if (bias != nullptr) {
            T reg_bias = __ldg(&bias[id % n]);
            val        = val + reg_bias;
        }
        out[id] = (T)(gelu(val));
    }
}

template<>
__global__ void addBiasGelu(half* out, const half* __restrict bias, int m, int n)
{
    half2*       out_ptr  = (half2*)out;
    const half2* bias_ptr = (half2*)bias;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        half2 val = out_ptr[id];
        if (bias != nullptr) {
            half2 reg_bias = __ldg(&bias_ptr[id % n]);
            val            = __hadd2(val, reg_bias);
        }
        out_ptr[id] = gelu(val);
    }
}

#ifdef ENABLE_BF16
template<>
__global__ void addBiasGelu(__nv_bfloat16* out, const __nv_bfloat16* __restrict bias, int m, int n)
{
    __nv_bfloat162*       out_ptr  = (__nv_bfloat162*)out;
    const __nv_bfloat162* bias_ptr = (__nv_bfloat162*)bias;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        __nv_bfloat162 val = out_ptr[id];
        if (bias != nullptr) {
            __nv_bfloat162 reg_bias = ldg(&bias_ptr[id % n]);
            val                     = bf16hadd2(val, reg_bias);
        }
        out_ptr[id] = gelu(val);
    }
}
#endif

template<typename T>
void invokeAddBiasGelu(T* out, const T* bias, const int m, const int n, cudaStream_t stream)
{
    const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16 and bf16
    dim3      block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x  = m;
    }
    else {
        block.x = 1024;
        grid.x  = ceil(m * n / 1024.);
    }
    addBiasGelu<T><<<grid, block, 0, stream>>>(out, bias, m, n / data_type_factor);
}

template void invokeAddBiasGelu(float* out, const float* bias, const int m, const int n, cudaStream_t stream);
template void invokeAddBiasGelu(half* out, const half* bias, const int m, const int n, cudaStream_t stream);
#ifdef ENABLE_BF16
template void
invokeAddBiasGelu(__nv_bfloat16* out, const __nv_bfloat16* bias, const int m, const int n, cudaStream_t stream);
#endif

// Invoke GeGlu (gated glue)
template<typename T>
__global__ void
addBiasGatedGelu(T* hidden1, const T* hidden2, const T* __restrict bias1, const T* __restrict bias2, int m, int n)
{
    const bool use_bias = bias1 != nullptr && bias2 != nullptr;
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        T val1 = hidden1[id];
        T val2 = hidden2[id];
        if (use_bias) {
            T reg_bias1 = __ldg(&bias1[id % n]);
            T reg_bias2 = __ldg(&bias2[id % n]);
            hidden1[id] = (T)(gelu(val1 + reg_bias1) * (val2 + reg_bias2));
        }
        else {
            hidden1[id] = (T)(gelu(val1) * val2);
        }
    }
}

template<>
__global__ void addBiasGatedGelu(
    half* hidden1, const half* hidden2, const half* __restrict bias1, const half* __restrict bias2, int m, int n)
{
    half2*       hidden1_ptr = (half2*)hidden1;
    const half2* hidden2_ptr = (half2*)hidden2;
    const half2* bias1_ptr   = (half2*)bias1;
    const half2* bias2_ptr   = (half2*)bias2;
    const bool   use_bias    = bias1 != nullptr && bias2 != nullptr;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        half2 val1 = hidden1_ptr[id];
        half2 val2 = hidden2_ptr[id];
        if (use_bias) {
            half2 reg_bias1 = __ldg(&bias1_ptr[id % n]);
            half2 reg_bias2 = __ldg(&bias2_ptr[id % n]);
            hidden1_ptr[id] = __hmul2(gelu(__hadd2(val1, reg_bias1)), __hadd2(val2, reg_bias2));
        }
        else {
            hidden1_ptr[id] = __hmul2(gelu(val1), val2);
        }
    }
}

#ifdef ENABLE_BF16
template<>
__global__ void addBiasGatedGelu(__nv_bfloat16*       hidden1,
                                 const __nv_bfloat16* hidden2,
                                 const __nv_bfloat16* __restrict bias1,
                                 const __nv_bfloat16* __restrict bias2,
                                 int m,
                                 int n)
{
    __nv_bfloat162*       hidden1_ptr = (__nv_bfloat162*)hidden1;
    const __nv_bfloat162* hidden2_ptr = (__nv_bfloat162*)hidden2;
    const __nv_bfloat162* bias1_ptr   = (__nv_bfloat162*)bias1;
    const __nv_bfloat162* bias2_ptr   = (__nv_bfloat162*)bias2;
    const bool            use_bias    = bias1 != nullptr && bias2 != nullptr;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        __nv_bfloat162 val1 = hidden1_ptr[id];
        __nv_bfloat162 val2 = hidden2_ptr[id];
        if (use_bias) {
            __nv_bfloat162 reg_bias1 = ldg(&bias1_ptr[id % n]);
            __nv_bfloat162 reg_bias2 = ldg(&bias2_ptr[id % n]);
            hidden1_ptr[id]          = bf16hmul2(gelu(bf16hadd2(val1, reg_bias1)), bf16hadd2(val2, reg_bias2));
        }
        else {
            hidden1_ptr[id] = bf16hmul2(gelu(val1), val2);
        }
    }
}
#endif

template<typename T>
void invokeAddBiasGatedGelu(
    T* hidden1, const T* hidden2, const T* bias1, const T* bias2, const int m, const int n, cudaStream_t stream)
{
    const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16 and bf16
    dim3      block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x  = m;
    }
    else {
        block.x = 1024;
        grid.x  = ceil(m * n / 1024.);
    }
    addBiasGatedGelu<T><<<grid, block, 0, stream>>>(hidden1, hidden2, bias1, bias2, m, n / data_type_factor);
}

// GELU(hidden1 + bias1) * (hidden2 + bias2)
template void invokeAddBiasGatedGelu(float*       hidden1,
                                     const float* hidden2,
                                     const float* bias1,
                                     const float* bias2,
                                     const int    m,
                                     const int    n,
                                     cudaStream_t stream);
template void invokeAddBiasGatedGelu(half*        hidden1,
                                     const half*  hidden2,
                                     const half*  bias1,
                                     const half*  bias2,
                                     const int    m,
                                     const int    n,
                                     cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeAddBiasGatedGelu(__nv_bfloat16*       hidden1,
                                     const __nv_bfloat16* hidden2,
                                     const __nv_bfloat16* bias1,
                                     const __nv_bfloat16* bias2,
                                     const int            m,
                                     const int            n,
                                     cudaStream_t         stream);
#endif

template<typename T>
__global__ void add_bias_relu(T* out, const T* __restrict bias, int m, int n)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        T val = out[id];
        if (bias != nullptr) {
            val = val + ldg(&bias[id % n]);
        }
        out[id] = val > (T)0.0f ? val : (T)0.0f;
    }
}

template<>
__global__ void add_bias_relu(half* out, const half* __restrict bias, int m, int n)
{
    half2*       out_ptr  = (half2*)out;
    const half2* bias_ptr = (half2*)bias;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        half2 val = out_ptr[id];
        if (bias != nullptr) {
            val = val + __ldg(&bias_ptr[id % n]);
        }
        val.x       = val.x > (half)0.0f ? val.x : (half)0.0f;
        val.y       = val.y > (half)0.0f ? val.y : (half)0.0f;
        out_ptr[id] = val;
    }
}

#ifdef ENABLE_BF16
template<>
__global__ void add_bias_relu(__nv_bfloat16* out, const __nv_bfloat16* __restrict bias, int m, int n)
{
    __nv_bfloat162*       out_ptr  = (__nv_bfloat162*)out;
    const __nv_bfloat162* bias_ptr = (__nv_bfloat162*)bias;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        __nv_bfloat162 val = out_ptr[id];
        if (bias != nullptr) {
            val = bf16hadd2(val, ldg(&bias_ptr[id % n]));
        }
        val.x       = val.x > (__nv_bfloat16)0.0f ? val.x : (__nv_bfloat16)0.0f;
        val.y       = val.y > (__nv_bfloat16)0.0f ? val.y : (__nv_bfloat16)0.0f;
        out_ptr[id] = val;
    }
}
#endif

template<typename T>
void invokeAddBiasRelu(T* out, const T* bias, const int m, const int n, cudaStream_t stream)
{
    const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16 and bf16
    dim3      block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x  = m;
    }
    else {
        block.x = 1024;
        grid.x  = ceil(m * n / 1024.);
    }
    add_bias_relu<T><<<grid, block, 0, stream>>>(out, bias, m, n / data_type_factor);
}

template void invokeAddBiasRelu(float* out, const float* bias, const int m, const int n, cudaStream_t stream);
template void invokeAddBiasRelu(half* out, const half* bias, const int m, const int n, cudaStream_t stream);
#ifdef ENABLE_BF16
template void
invokeAddBiasRelu(__nv_bfloat16* out, const __nv_bfloat16* bias, const int m, const int n, cudaStream_t stream);
#endif

// Invoke GeGlu (gated glue)
template<typename T>
__global__ void
addBiasGatedRelu(T* hidden1, const T* hidden2, const T* __restrict bias1, const T* __restrict bias2, int m, int n)
{
    const bool use_bias = bias1 != nullptr && bias2 != nullptr;
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        T val1 = hidden1[id];
        T val2 = hidden2[id];
        if (use_bias) {
            T reg_bias1 = __ldg(&bias1[id % n]);
            T reg_bias2 = __ldg(&bias2[id % n]);
            val1 += reg_bias1;
            val2 += reg_bias2;
        }
        hidden1[id] = val1 > (T)0.0f ? val1 * val2 : (T)0.0f;
    }
}

template<>
__global__ void addBiasGatedRelu(
    half* hidden1, const half* hidden2, const half* __restrict bias1, const half* __restrict bias2, int m, int n)
{
    half2*       hidden1_ptr = (half2*)hidden1;
    const half2* hidden2_ptr = (half2*)hidden2;
    const half2* bias1_ptr   = (half2*)bias1;
    const half2* bias2_ptr   = (half2*)bias2;
    const bool   use_bias    = bias1 != nullptr && bias2 != nullptr;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        half2 val1 = hidden1_ptr[id];
        half2 val2 = hidden2_ptr[id];
        if (use_bias) {
            half2 reg_bias1 = __ldg(&bias1_ptr[id % n]);
            half2 reg_bias2 = __ldg(&bias2_ptr[id % n]);
            val1            = __hadd2(val1, reg_bias1);
            val2            = __hadd2(val2, reg_bias2);
        }
        val1.x          = val1.x > (half)0.0f ? val1.x * val2.x : (half)0.0f;
        val1.y          = val1.y > (half)0.0f ? val1.y * val2.y : (half)0.0f;
        hidden1_ptr[id] = val1;
    }
}

#ifdef ENABLE_BF16
template<>
__global__ void addBiasGatedRelu(__nv_bfloat16*       hidden1,
                                 const __nv_bfloat16* hidden2,
                                 const __nv_bfloat16* __restrict bias1,
                                 const __nv_bfloat16* __restrict bias2,
                                 int m,
                                 int n)
{
    __nv_bfloat162*       hidden1_ptr = (__nv_bfloat162*)hidden1;
    const __nv_bfloat162* hidden2_ptr = (__nv_bfloat162*)hidden2;
    const __nv_bfloat162* bias1_ptr   = (__nv_bfloat162*)bias1;
    const __nv_bfloat162* bias2_ptr   = (__nv_bfloat162*)bias2;
    const bool            use_bias    = bias1 != nullptr && bias2 != nullptr;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        __nv_bfloat162 val1 = hidden1_ptr[id];
        __nv_bfloat162 val2 = hidden2_ptr[id];
        if (use_bias) {
            __nv_bfloat162 reg_bias1 = ldg(&bias1_ptr[id % n]);
            __nv_bfloat162 reg_bias2 = ldg(&bias2_ptr[id % n]);
            val1                     = bf16hadd2(val1, reg_bias1);
            val2                     = bf16hadd2(val2, reg_bias2);
        }
        val1.x          = val1.x > (__nv_bfloat16)0.0f ? bf16hadd(val1.x, val2.x) : (__nv_bfloat16)0.0f;
        val1.y          = val1.y > (__nv_bfloat16)0.0f ? bf16hadd(val1.y, val2.y) : (__nv_bfloat16)0.0f;
        hidden1_ptr[id] = val1;
    }
}
#endif

template<typename T>
void invokeAddBiasGatedRelu(
    T* hidden1, const T* hidden2, const T* bias1, const T* bias2, const int m, const int n, cudaStream_t stream)
{
    const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16 and bf16
    dim3      block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x  = m;
    }
    else {
        block.x = 1024;
        grid.x  = ceil(m * n / 1024.);
    }
    addBiasGatedRelu<T><<<grid, block, 0, stream>>>(hidden1, hidden2, bias1, bias2, m, n / data_type_factor);
}

// GELU(hidden1 + bias1) * (hidden2 + bias2)
template void invokeAddBiasGatedRelu(float*       hidden1,
                                     const float* hidden2,
                                     const float* bias1,
                                     const float* bias2,
                                     const int    m,
                                     const int    n,
                                     cudaStream_t stream);
template void invokeAddBiasGatedRelu(half*        hidden1,
                                     const half*  hidden2,
                                     const half*  bias1,
                                     const half*  bias2,
                                     const int    m,
                                     const int    n,
                                     cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeAddBiasGatedRelu(__nv_bfloat16*       hidden1,
                                     const __nv_bfloat16* hidden2,
                                     const __nv_bfloat16* bias1,
                                     const __nv_bfloat16* bias2,
                                     const int            m,
                                     const int            n,
                                     cudaStream_t         stream);
#endif

template<typename H_T, typename B_T>
__global__ void add_bias(H_T* out, const B_T* __restrict bias, int m, int n)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        out[id] = out[id] + (H_T)ldg(&bias[id % n]);
    }
}

template<>
__global__ void add_bias(half* out, const half* __restrict bias, int m, int n)
{
    half2*       out_ptr  = (half2*)out;
    const half2* bias_ptr = (half2*)bias;
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        out_ptr[id] = out_ptr[id] + __ldg(&bias_ptr[id % n]);
    }
}

#ifdef ENABLE_BF16
template<>
__global__ void add_bias(__nv_bfloat16* out, const __nv_bfloat16* __restrict bias, int m, int n)
{
    __nv_bfloat162*       out_ptr  = (__nv_bfloat162*)out;
    const __nv_bfloat162* bias_ptr = (__nv_bfloat162*)bias;
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        out_ptr[id] = bf16hadd2(out_ptr[id], ldg(&bias_ptr[id % n]));
    }
}
#endif

template<typename H_T, typename B_T>
void invokeAddBias(H_T* out, const B_T* bias, const int m, const int n, cudaStream_t stream)
{
    const int data_type_factor = 4 / sizeof(H_T);  // 1 for fp32, 2 for fp16 and bf16
    dim3      block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x  = m;
    }
    else {
        block.x = 1024;
        grid.x  = ceil(m * n / 1024.);
    }
    add_bias<<<grid, block, 0, stream>>>(out, bias, m, n / data_type_factor);
}

template void invokeAddBias(float* out, const float* bias, const int m, const int n, cudaStream_t stream);
template void invokeAddBias(half* out, const half* bias, const int m, const int n, cudaStream_t stream);
template void invokeAddBias(float* out, const half* bias, const int m, const int n, cudaStream_t stream);
#ifdef ENABLE_BF16
template void
invokeAddBias(__nv_bfloat16* out, const __nv_bfloat16* bias, const int m, const int n, cudaStream_t stream);
template void invokeAddBias(float* out, const __nv_bfloat16* bias, const int m, const int n, cudaStream_t stream);
#endif

template<typename T2, int N>
__global__ void addBiasGeluV2(T2* out, const T2* __restrict bias, const int size)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < size; id += blockDim.x * gridDim.x) {
        T2 val = out[id];
        if (bias != nullptr) {
            T2 reg_bias = ldg(&bias[id % N]);
            val         = hadd2(val, reg_bias);
        }
        out[id] = gelu(val);
    }
}

template<typename T2, int N, int ELEMENT_PER_ROUND>
__global__ void addBiasGeluV3(T2* out, const T2* __restrict bias, const int size)
{
    T2 buffer[ELEMENT_PER_ROUND];
    T2 tmp_bias[ELEMENT_PER_ROUND];
    for (int id = blockIdx.x * blockDim.x * ELEMENT_PER_ROUND + threadIdx.x * ELEMENT_PER_ROUND; id < size;
         id += blockDim.x * gridDim.x * ELEMENT_PER_ROUND) {
#pragma unroll
        for (int i = 0; i < ELEMENT_PER_ROUND; i++) {
            buffer[i] = out[id + i];
            if (bias != nullptr) {
                tmp_bias[i] = ldg(&bias[(id + i) % N]);
            }
        }
#pragma unroll
        for (int i = 0; i < ELEMENT_PER_ROUND; i++) {
            if (bias != nullptr) {
                buffer[i] = hadd2(buffer[i], tmp_bias[i]);
            }
            out[id + i] = gelu(buffer[i]);
        }
    }
}

#define ADD_BIAS_GELU(HALF_N, ELEMENT_PER_ROUND)                                                                       \
    case HALF_N:                                                                                                       \
        if (ELEMENT_PER_ROUND > 1) {                                                                                   \
            grid.x = grid.x / ELEMENT_PER_ROUND;                                                                       \
            addBiasGeluV3<T2, HALF_N, ELEMENT_PER_ROUND>                                                               \
                <<<grid, block, 0, stream>>>((T2*)out, (const T2*)bias, m * half_n);                                   \
        }                                                                                                              \
        else {                                                                                                         \
            addBiasGeluV2<T2, HALF_N><<<grid, block, 0, stream>>>((T2*)out, (const T2*)bias, m * half_n);              \
        }                                                                                                              \
        break;

template<typename T>
void invokeAddBiasGeluV2(T* out, const T* bias, const int m, const int n, cudaStream_t stream)
{
    if (n % 2 == 0 && sizeof(T) == 2) {
        const int half_n = n / 2;
        dim3      block, grid;
        block.x  = std::min(half_n, 512);
        grid.x   = (m * half_n + (block.x - 1)) / block.x;
        using T2 = typename TypeConverter<T>::Type;

        if (grid.x >= 512) {
            switch (half_n) {
                ADD_BIAS_GELU(256, 1)
                ADD_BIAS_GELU(512, 1)
                ADD_BIAS_GELU(1024, 1)
                ADD_BIAS_GELU(1536, 1)
                ADD_BIAS_GELU(2048, 1)
                ADD_BIAS_GELU(4096, 2)
                ADD_BIAS_GELU(8192, 2)
                ADD_BIAS_GELU(16384, 2)
                ADD_BIAS_GELU(24576, 2)
                ADD_BIAS_GELU(40960, 4)
                default:
                    invokeAddBiasGelu(out, bias, m, n, stream);
                    break;
            }
        }
        else {
            switch (half_n) {
                ADD_BIAS_GELU(256, 1)
                ADD_BIAS_GELU(512, 1)
                ADD_BIAS_GELU(1024, 1)
                ADD_BIAS_GELU(1536, 1)
                ADD_BIAS_GELU(2048, 1)
                ADD_BIAS_GELU(4096, 1)
                ADD_BIAS_GELU(8192, 2)
                ADD_BIAS_GELU(16384, 2)
                ADD_BIAS_GELU(24576, 2)
                ADD_BIAS_GELU(40960, 2)
                default:
                    invokeAddBiasGelu(out, bias, m, n, stream);
                    break;
            }
        }
    }
    else {
        invokeAddBiasGelu(out, bias, m, n, stream);
    }
}

#undef ADD_BIAS_GELU

template void invokeAddBiasGeluV2(float* out, const float* bias, const int m, const int n, cudaStream_t stream);
template void invokeAddBiasGeluV2(half* out, const half* bias, const int m, const int n, cudaStream_t stream);
#ifdef ENABLE_BF16
template void
invokeAddBiasGeluV2(__nv_bfloat16* out, const __nv_bfloat16* bias, const int m, const int n, cudaStream_t stream);
#endif  // ENABLE_BF16

template<typename T>
__inline__ __device__ T silu(T x)
{
    return (T)((float)x / (1.0f + __expf((float)-x)));
}

template<typename T>
__global__ void add_bias_silu(T* out, const T* __restrict bias, int m, int n)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        T val = out[id];
        if (bias != nullptr) {
            val = val + ldg(&bias[id % n]);
        }
        out[id] = silu(val);
    }
}

template<>
__global__ void add_bias_silu(half* out, const half* __restrict bias, int m, int n)
{
    half2*       out_ptr  = (half2*)out;
    const half2* bias_ptr = (half2*)bias;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        half2 val = out_ptr[id];
        if (bias != nullptr) {
            val = val + __ldg(&bias_ptr[id % n]);
        }
        val.x       = silu(val.x);
        val.y       = silu(val.y);
        out_ptr[id] = val;
    }
}

#ifdef ENABLE_BF16
template<>
__global__ void add_bias_silu(__nv_bfloat16* out, const __nv_bfloat16* __restrict bias, int m, int n)
{
    __nv_bfloat162*       out_ptr  = (__nv_bfloat162*)out;
    const __nv_bfloat162* bias_ptr = (__nv_bfloat162*)bias;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        __nv_bfloat162 val = out_ptr[id];
        if (bias != nullptr) {
            val = bf16hadd2(val, ldg(&bias_ptr[id % n]));
        }
        val.x       = silu(val.x);
        val.y       = silu(val.y);
        out_ptr[id] = val;
    }
}
#endif

template<typename T>
void invokeAddBiasSilu(T* out, const T* bias, const int m, const int n, cudaStream_t stream)
{
    const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16 and bf16
    dim3      block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x  = m;
    }
    else {
        block.x = 1024;
        grid.x  = ceil(m * n / 1024.);
    }
    add_bias_silu<T><<<grid, block, 0, stream>>>(out, bias, m, n / data_type_factor);
}

template void invokeAddBiasSilu(float* out, const float* bias, const int m, const int n, cudaStream_t stream);
template void invokeAddBiasSilu(half* out, const half* bias, const int m, const int n, cudaStream_t stream);
#ifdef ENABLE_BF16
template void
invokeAddBiasSilu(__nv_bfloat16* out, const __nv_bfloat16* bias, const int m, const int n, cudaStream_t stream);
#endif

// Invoke GeGlu (gated glue)
template<typename T>
__global__ void
addBiasGatedSilu(T* hidden1, const T* hidden2, const T* __restrict bias1, const T* __restrict bias2, int m, int n)
{
    const bool use_bias = bias1 != nullptr && bias2 != nullptr;
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        T val1 = hidden1[id];
        T val2 = hidden2[id];
        if (use_bias) {
            T reg_bias1 = __ldg(&bias1[id % n]);
            T reg_bias2 = __ldg(&bias2[id % n]);
            val1 += reg_bias1;
            val2 += reg_bias2;
        }
        hidden1[id] = silu(val1) * val2;
    }
}

template<>
__global__ void addBiasGatedSilu(
    half* hidden1, const half* hidden2, const half* __restrict bias1, const half* __restrict bias2, int m, int n)
{
    half2*       hidden1_ptr = (half2*)hidden1;
    const half2* hidden2_ptr = (half2*)hidden2;
    const half2* bias1_ptr   = (half2*)bias1;
    const half2* bias2_ptr   = (half2*)bias2;
    const bool   use_bias    = bias1 != nullptr && bias2 != nullptr;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        half2 val1 = hidden1_ptr[id];
        half2 val2 = hidden2_ptr[id];
        if (use_bias) {
            half2 reg_bias1 = __ldg(&bias1_ptr[id % n]);
            half2 reg_bias2 = __ldg(&bias2_ptr[id % n]);
            val1            = __hadd2(val1, reg_bias1);
            val2            = __hadd2(val2, reg_bias2);
        }
        val1.x          = silu(val1.x) * val2.x;
        val1.y          = silu(val1.y) * val2.y;
        hidden1_ptr[id] = val1;
    }
}

#ifdef ENABLE_BF16
template<>
__global__ void addBiasGatedSilu(__nv_bfloat16*       hidden1,
                                 const __nv_bfloat16* hidden2,
                                 const __nv_bfloat16* __restrict bias1,
                                 const __nv_bfloat16* __restrict bias2,
                                 int m,
                                 int n)
{
    __nv_bfloat162*       hidden1_ptr = (__nv_bfloat162*)hidden1;
    const __nv_bfloat162* hidden2_ptr = (__nv_bfloat162*)hidden2;
    const __nv_bfloat162* bias1_ptr   = (__nv_bfloat162*)bias1;
    const __nv_bfloat162* bias2_ptr   = (__nv_bfloat162*)bias2;
    const bool            use_bias    = bias1 != nullptr && bias2 != nullptr;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        __nv_bfloat162 val1 = hidden1_ptr[id];
        __nv_bfloat162 val2 = hidden2_ptr[id];
        if (use_bias) {
            __nv_bfloat162 reg_bias1 = ldg(&bias1_ptr[id % n]);
            __nv_bfloat162 reg_bias2 = ldg(&bias2_ptr[id % n]);
            val1                     = bf16hadd2(val1, reg_bias1);
            val2                     = bf16hadd2(val2, reg_bias2);
        }
        val1.x          = (__nv_bfloat16)(silu((float)val1.x) * (float)val2.x);
        val1.y          = (__nv_bfloat16)(silu((float)val1.y) * (float)val2.y);
        hidden1_ptr[id] = val1;
    }
}
#endif

template<typename T>
void invokeAddBiasGatedSilu(
    T* hidden1, const T* hidden2, const T* bias1, const T* bias2, const int m, const int n, cudaStream_t stream)
{
    const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16 and bf16
    dim3      block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x  = m;
    }
    else {
        block.x = 1024;
        grid.x  = ceil(m * n / 1024.);
    }
    addBiasGatedSilu<T><<<grid, block, 0, stream>>>(hidden1, hidden2, bias1, bias2, m, n / data_type_factor);
}

template void invokeAddBiasGatedSilu(float*       hidden1,
                                     const float* hidden2,
                                     const float* bias1,
                                     const float* bias2,
                                     const int    m,
                                     const int    n,
                                     cudaStream_t stream);
template void invokeAddBiasGatedSilu(half*        hidden1,
                                     const half*  hidden2,
                                     const half*  bias1,
                                     const half*  bias2,
                                     const int    m,
                                     const int    n,
                                     cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeAddBiasGatedSilu(__nv_bfloat16*       hidden1,
                                     const __nv_bfloat16* hidden2,
                                     const __nv_bfloat16* bias1,
                                     const __nv_bfloat16* bias2,
                                     const int            m,
                                     const int            n,
                                     cudaStream_t         stream);
#endif

}  // namespace fastertransformer
