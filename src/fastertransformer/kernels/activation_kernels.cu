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

#include "src/fastertransformer/kernels/activation_kernels.h"
namespace fastertransformer {

template<typename T>
__inline__ __device__ T gelu(T x)
{
    float cdf = 0.5f * (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
    return x * cdf;
}

template<>
__inline__ __device__ half2 gelu(half2 val)
{
    half2 val_pow3 = __hmul2(val, __hmul2(val, val));
    float2 tmp_pow = __half22float2(val_pow3);
    float2 tmp = __half22float2(val);

    tmp.x = 0.5f * (1.0f + tanhf((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
    tmp.y = 0.5f * (1.0f + tanhf((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
    return __hmul2(val, __float22half2_rn(tmp));
}

template<typename T>
__global__ void addBiasGelu(T* out, const T* __restrict bias, int m, int n)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        T reg_bias = __ldg(&bias[id % n]);
        T val = out[id] + reg_bias;
        out[id] = (T)(gelu(val));
    }
}

template<>
__global__ void addBiasGelu(half* out, const half* __restrict bias, int m, int n)
{
    half2* out_ptr = (half2*)out;
    const half2* bias_ptr = (half2*)bias;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        half2 reg_bias = __ldg(&bias_ptr[id % n]);
        half2 val = out_ptr[id] + reg_bias;
        out_ptr[id] = gelu(val);
    }
}

template<typename T>
void invokeAddBiasGelu(T* out, const T* bias, const int m, const int n, cudaStream_t stream)
{
    const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16
    dim3 block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x = m;
    }
    else {
        block.x = 1024;
        grid.x = ceil(m * n / 1024.);
    }
    addBiasGelu<T><<<grid, block, 0, stream>>>(out, bias, m, n / data_type_factor);
}

template void invokeAddBiasGelu(float* out, const float* bias, const int m, const int n, cudaStream_t stream);
template void invokeAddBiasGelu(half* out, const half* bias, const int m, const int n, cudaStream_t stream);

template<typename T>
__global__ void add_bias_relu(T* out, const T* __restrict bias, int m, int n)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        T reg_bias = __ldg(&bias[id % n]);
        T val = out[id] + reg_bias;
        out[id] = (T)(val > 0.0f ? val : 0.0f);
    }
}

template<>
__global__ void add_bias_relu(half* out, const half* __restrict bias, int m, int n)
{
    half2* out_ptr = (half2*)out;
    const half2* bias_ptr = (half2*)bias;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        half2 reg_bias = __ldg(&bias_ptr[id % n]);
        half2 val = out_ptr[id] + reg_bias;
        val.x = val.x > (half)0.0f ? val.x : (half)0.0f;
        val.y = val.y > (half)0.0f ? val.y : (half)0.0f;
        out_ptr[id] = val;
    }
}

template<typename T>
void invokeAddBiasRelu(T* out, const T* bias, const int m, const int n, cudaStream_t stream)
{
    const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16
    dim3 block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x = m;
    }
    else {
        block.x = 1024;
        grid.x = ceil(m * n / 1024.);
    }
    add_bias_relu<T><<<grid, block, 0, stream>>>(out, bias, m, n / data_type_factor);
}

template void invokeAddBiasRelu(float* out, const float* bias, const int m, const int n, cudaStream_t stream);
template void invokeAddBiasRelu(half* out, const half* bias, const int m, const int n, cudaStream_t stream);

template<typename H_T, typename B_T>
__global__ void add_bias(H_T* out, const B_T* __restrict bias, int m, int n)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        out[id] = out[id] + (H_T) __ldg(&bias[id % n]);
    }
}

template<typename H_T, typename B_T>
void invokeAddBias(H_T* out, const B_T* bias, const int m, const int n, cudaStream_t stream)
{
    const int data_type_factor = 4 / sizeof(H_T);  // 1 for fp32, 2 for fp16
    dim3 block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x = m;
    }
    else {
        block.x = 1024;
        grid.x = ceil(m * n / 1024.);
    }
    add_bias<<<grid, block, 0, stream>>>(out, bias, m, n / data_type_factor);
}

template void invokeAddBias(float* out, const float* bias, const int m, const int n, cudaStream_t stream);
template void invokeAddBias(half* out, const half* bias, const int m, const int n, cudaStream_t stream);
template void invokeAddBias(float* out, const half* bias, const int m, const int n, cudaStream_t stream);
}  // namespace fastertransformer
