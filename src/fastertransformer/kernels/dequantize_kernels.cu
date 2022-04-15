/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "dequantize_kernels.h"
#include "reduce_kernel_utils.cuh"

namespace fastertransformer {

/***********************invoke deQuantization**************************/
__global__ void dequantized_kernel(float4* dst, const char4* src, const int size_div_4, const float* scale_ptr)
{
    int tid = (blockIdx.x * blockDim.x + threadIdx.x);
    if (tid < size_div_4) {
        const float scale = __ldg(scale_ptr);
        char4 tmp = __ldg(src + tid);
        int outIdx = tid;

        float4 float4Tmp;
        float4Tmp.x = static_cast<float>(tmp.x) * scale;
        float4Tmp.y = static_cast<float>(tmp.y) * scale;
        float4Tmp.z = static_cast<float>(tmp.z) * scale;
        float4Tmp.w = static_cast<float>(tmp.w) * scale;
        dst[outIdx] = float4Tmp;
    }
}

__global__ void dequantized_kernel(half4* dst, const char4* src, const int size_div_4, const float* scale_ptr)
{
    int tid = (blockIdx.x * blockDim.x + threadIdx.x);
    if (tid < size_div_4) {
        const float scale = __ldg(scale_ptr);
        char4 tmp = __ldg(src + tid);
        int outIdx = tid;

        half4 half4Tmp;
        half4Tmp.x = static_cast<half>(static_cast<float>(tmp.x) * scale);
        half4Tmp.y = static_cast<half>(static_cast<float>(tmp.y) * scale);
        half4Tmp.z = static_cast<half>(static_cast<float>(tmp.z) * scale);
        half4Tmp.w = static_cast<half>(static_cast<float>(tmp.w) * scale);
        dst[outIdx] = half4Tmp;
    }
}

template<typename T>
void invokeDequantization(T* dst, const int8_t* src, const int size, const float* scale_ptr, cudaStream_t stream)
{

    if (size % 4 != 0) {
        printf("[ERROR][invokeQuantization] size should be a multiple of 4.\n");
        exit(-1);
    }
    dim3 grid((size + 255) / 256);
    dim3 block(64);
    if (sizeof(T) == sizeof(float)) {
        dequantized_kernel<<<grid, block, 0, stream>>>((float4*)dst, (const char4*)src, size / 4, scale_ptr);
    }
    else if (sizeof(T) == sizeof(half)) {
        dequantized_kernel<<<grid, block, 0, stream>>>((half4*)dst, (const char4*)src, size / 4, scale_ptr);
    }
}

template void
invokeDequantization<float>(float* dst, const int8_t* src, const int size, const float* scale_ptr, cudaStream_t stream);

template void
invokeDequantization<half>(half* dst, const int8_t* src, const int size, const float* scale_ptr, cudaStream_t stream);

/***********************invoke deQuantization**************************/
__global__ void dequantized_kernel_INT32(
    float4* dst, const int4* src, const int size_div_4, const float* input_amax_ptr, const float* weight_amax_ptr)
{
    int tid = (blockIdx.x * blockDim.x + threadIdx.x);
    if (tid < size_div_4) {
        const float scale = (1.0f / __ldg(input_amax_ptr)) * (__ldg(weight_amax_ptr) / 127.0f);
        int4 tmp = src[tid];
        int outIdx = tid;

        float4 float4Tmp;
        float4Tmp.x = static_cast<float>(tmp.x) * scale;
        float4Tmp.y = static_cast<float>(tmp.y) * scale;
        float4Tmp.z = static_cast<float>(tmp.z) * scale;
        float4Tmp.w = static_cast<float>(tmp.w) * scale;
        dst[outIdx] = float4Tmp;
    }
}

__global__ void dequantized_kernel_INT32(
    half4* dst, const int4* src, const int size_div_4, const float* input_amax_ptr, const float* weight_amax_ptr)
{
    int tid = (blockIdx.x * blockDim.x + threadIdx.x);
    if (tid < size_div_4) {
        const float scale = (1.0f / __ldg(input_amax_ptr)) * (__ldg(weight_amax_ptr) / 127.0f);
        int4 tmp = src[tid];
        int outIdx = tid;

        half4 half4Tmp;
        half4Tmp.x = static_cast<half>(static_cast<float>(tmp.x) * scale);
        half4Tmp.y = static_cast<half>(static_cast<float>(tmp.y) * scale);
        half4Tmp.z = static_cast<half>(static_cast<float>(tmp.z) * scale);
        half4Tmp.w = static_cast<half>(static_cast<float>(tmp.w) * scale);
        dst[outIdx] = half4Tmp;
    }
}

template<typename T>
void invokeDequantization_INT32(T* dst,
                                const int32_t* src,
                                const int size,
                                cudaStream_t stream,
                                const float* input_amax_ptr,
                                const float* weight_amax_ptr)
{

    if (size % 4 != 0) {
        printf("[ERROR][invokeQuantization] size should be a multiple of 4.\n");
        exit(-1);
    }
    dim3 grid((size + 255) / 256);
    dim3 block(64);
    if (sizeof(T) == sizeof(float)) {
        dequantized_kernel_INT32<<<grid, block, 0, stream>>>(
            (float4*)dst, (const int4*)src, size / 4, input_amax_ptr, weight_amax_ptr);
    }
    else if (sizeof(T) == sizeof(half)) {
        dequantized_kernel_INT32<<<grid, block, 0, stream>>>(
            (half4*)dst, (const int4*)src, size / 4, input_amax_ptr, weight_amax_ptr);
    }
}

template void invokeDequantization_INT32<float>(float* dst,
                                                const int32_t* src,
                                                const int size,
                                                cudaStream_t stream,
                                                const float* input_amax_ptr,
                                                const float* weight_amax_ptr);

template void invokeDequantization_INT32<half>(half* dst,
                                               const int32_t* src,
                                               const int size,
                                               cudaStream_t stream,
                                               const float* input_amax_ptr,
                                               const float* weight_amax_ptr);

}  // namespace fastertransformer