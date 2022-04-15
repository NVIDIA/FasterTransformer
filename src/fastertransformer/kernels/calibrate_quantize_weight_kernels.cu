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

#include "src/fastertransformer/kernels/calibrate_quantize_weight_kernels.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"

namespace fastertransformer {

// src is [k, n] row-major
// scale is [n]
// grid(n)
// block(k)
// TODO : Improve for memory coalesing
template<typename T>
__global__ void ldn_calibrate_weight_per_channel(float* scale, const T* src, const int k, const int n)
{
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    scale += bidx;
    float amax_val = 0.0f;
    for (int k_i = tidx; k_i < k; k_i += blockDim.x) {
        float val = fabs(static_cast<float>(src[k_i * n + bidx]));
        if (amax_val < val) {
            amax_val = val;
        }
    }
    const float block_amax_val = blockReduceMax(amax_val);
    if (tidx == 0) {
        scale[0] = block_amax_val / 127.0f;
    }
}

template<typename T>
void invokeLdnCalibrateWeightPerChannel(float* scale, const T* src, const int k, const int n, cudaStream_t stream)
{

    dim3 grid(n);
    dim3 block((k + 31) / 32 * 32);
    if (block.x > 1024) {
        block.x = 1024;
    }
    ldn_calibrate_weight_per_channel<<<grid, block, 0, stream>>>(scale, src, k, n);
}

template void
invokeLdnCalibrateWeightPerChannel(float* scale, const float* src, const int k, const int n, cudaStream_t stream);

template void
invokeLdnCalibrateWeightPerChannel(float* scale, const half* src, const int k, const int n, cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeLdnCalibrateWeightPerChannel(
    float* scale, const __nv_bfloat16* src, const int k, const int n, cudaStream_t stream);
#endif

//---------------------------------------------------------------------------------

// src is [n, k] row-major
// dst is [n, k] row-major
// scale is [n]
// grid(n)
// block(k)
template<typename T>
__global__ void ldk_calibrate_quantize_weight_per_channel(int8_t* dst, float* scale, const T* src, const int k)
{
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    scale += bidx;
    src += bidx * k;
    dst += bidx * k;
    T amax_val = 0.0f;
    const T zero = static_cast<T>(0.0f);
    for (int k_i = tidx; k_i < k; k_i += blockDim.x) {
        T val = src[k_i];
        val = val > zero ? val : -val;
        if (amax_val > val) {
            amax_val = val;
        }
    }
    __shared__ float s_amax;
    const float block_amax_val = blockReduceMax(static_cast<float>(amax_val));
    if (tidx == 0) {
        s_amax = block_amax_val;
        scale[0] = block_amax_val / 127.0f;
    }
    __syncthreads();

    for (int k_i = tidx; k_i < k; k_i += blockDim.x) {
        T val = src[k_i];
        dst[k_i] = float_to_int8_rn(127.0f * static_cast<float>(val) / s_amax);
    }
}

template<typename T>
void invokeLdkCalibrateQuantizeWeightPerChannel(
    int8_t* dst, float* scale, const T* src, const int n, const int k, cudaStream_t stream)
{

    dim3 grid(n);
    dim3 block((k + 31) / 32 * 32);
    if (block.x > 1024) {
        block.x = 1024;
    }
    ldk_calibrate_quantize_weight_per_channel<<<grid, block, 0, stream>>>(dst, scale, src, k);
}

template void invokeLdkCalibrateQuantizeWeightPerChannel(
    int8_t* dst, float* scale, const float* src, const int n, const int k, cudaStream_t stream);

template void invokeLdkCalibrateQuantizeWeightPerChannel(
    int8_t* dst, float* scale, const half* src, const int n, const int k, cudaStream_t stream);

//---------------------------------------------------------------

// src is [k, n] row-major
// dst is [n, k] row-major
template<typename T>
__global__ void
ldn_transpose_quantize_weight_per_channel(int8_t* dst, const float* scale, const T* src, const int k, const int n)
{
    __shared__ T shm[32][33];
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    int n_idx = blockIdx.x * 32 + tidx;
    int k_idx = blockIdx.y * 32 + tidy;
    if (n_idx < n && k_idx < k) {
        shm[tidx][tidy] = src[k_idx * n + n_idx];
    }

    __syncthreads();
    n_idx = blockIdx.x * 32 + tidy;
    k_idx = blockIdx.y * 32 + tidx;
    if (n_idx < n && k_idx < k) {
        dst[n_idx * k + k_idx] = float_to_int8_rn(static_cast<float>(shm[tidy][tidx]) / scale[n_idx]);
    }
}

// src is [k, n] row-major
// dst is [n, k] row-major
template<typename T>
void invokeLdnTransposeQuantizeWeightPerChannel(
    int8_t* dst, const float* scale, const T* src, const int k, const int n, cudaStream_t stream)
{
    dim3 grid(n / 32, k / 32);
    dim3 block(32, 32);
    ldn_transpose_quantize_weight_per_channel<<<grid, block, 0, stream>>>(dst, scale, src, k, n);
}

template void invokeLdnTransposeQuantizeWeightPerChannel(
    int8_t* dst, const float* scale, const float* src, const int k, const int n, cudaStream_t stream);

template void invokeLdnTransposeQuantizeWeightPerChannel(
    int8_t* dst, const float* scale, const half* src, const int k, const int n, cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeLdnTransposeQuantizeWeightPerChannel(
    int8_t* dst, const float* scale, const __nv_bfloat16* src, const int k, const int n, cudaStream_t stream);
#endif

}  // namespace fastertransformer
