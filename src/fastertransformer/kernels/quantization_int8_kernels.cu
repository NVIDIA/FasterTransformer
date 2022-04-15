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

#include "src/fastertransformer/kernels/quantization_int8_kernels.h"
namespace fastertransformer {

__global__ void quantized_kernel(char4* dst, const float4* src, const int size_div_4, const float* scale_ptr)
{
    int tid = (blockIdx.x * blockDim.x + threadIdx.x);
    if (tid < size_div_4) {
        const float scale = __ldg(scale_ptr);
        char4 tmp;
        const float4 floatTmp = __ldg(src + tid);
        tmp.x = float_to_int8_rn(floatTmp.x * scale);
        tmp.y = float_to_int8_rn(floatTmp.y * scale);
        tmp.z = float_to_int8_rn(floatTmp.z * scale);
        tmp.w = float_to_int8_rn(floatTmp.w * scale);
        dst[tid] = tmp;
    }
}

__global__ void quantized_kernel(char4* dst, const half2* src, const int size_div_4, const float* scale_ptr)
{
    int tid = (blockIdx.x * blockDim.x + threadIdx.x);
    if (tid < size_div_4) {
        const float scale = __ldg(scale_ptr);
        char4 tmp;
        int src_id = tid << 1;

        const half2 half2Tmp = __ldg(src + src_id);
        tmp.x = float_to_int8_rn(static_cast<float>(half2Tmp.x) * scale);
        tmp.y = float_to_int8_rn(static_cast<float>(half2Tmp.y) * scale);

        const half2 half2Tmp2 = __ldg(src + src_id + 1);
        tmp.z = float_to_int8_rn(static_cast<float>(half2Tmp2.x) * scale);
        tmp.w = float_to_int8_rn(static_cast<float>(half2Tmp2.y) * scale);
        dst[tid] = tmp;
    }
}

template<typename T>
void invokeQuantization(int8_t* dst, const T* src, const int size, const float* scale_ptr, cudaStream_t stream)
{
    if (size % 4 != 0) {
        printf("[ERROR][invokeQuantization] size should be a multiple of 4.\n");
        exit(-1);
    }
    dim3 grid((size + 255) / 256);
    dim3 block(64);
    if (sizeof(T) == sizeof(float)) {
        quantized_kernel<<<grid, block, 0, stream>>>((char4*)dst, (const float4*)src, size / 4, scale_ptr);
    }
    else if (sizeof(T) == sizeof(half)) {
        quantized_kernel<<<grid, block, 0, stream>>>((char4*)dst, (const half2*)src, size / 4, scale_ptr);
    }
}

template void
invokeQuantization<float>(int8_t* dst, const float* src, const int size, const float* scale_ptr, cudaStream_t stream);

template void
invokeQuantization<half>(int8_t* dst, const half* src, const int size, const float* scale_ptr, cudaStream_t stream);

}  // namespace fastertransformer