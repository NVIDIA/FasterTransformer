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

#include "src/fastertransformer/kernels/add_residual_kernels.h"

namespace fastertransformer {

template<typename T>
__global__ void addBiasResidual(T* output, const T* input, const T* bias, const int m, const int n)
{
    const int col_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (col_index < n) {
        T bias_val = (bias == nullptr) ? (T)(0.0f) : bias[col_index];
        output[blockIdx.x * n + col_index] =
            output[blockIdx.x * n + col_index] + input[blockIdx.x * n + col_index] + bias_val;
    }
}

template<typename T>
void invokeAddBiasResidual(T* output, const T* input, const T* bias, const int m, const int n, cudaStream_t stream)
{
    int blocks_per_row = ceil(float(n) / 1024);
    dim3 grid(m, blocks_per_row);
    dim3 block(min(n, 1024));
    addBiasResidual<<<grid, block, 0, stream>>>(output, input, bias, m, n);
}

template<typename T>
__global__ void addBiasAttentionFfnResidual(
    T* block_output,
    const T* ffn_output,
    const T* attn_output,
    const T* block_input,
    const T* bias,
    const int m, const int n)
{
    const int col_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (col_index < n) {
        block_output[blockIdx.x * n + col_index] =
            ffn_output[blockIdx.x * n + col_index] +
            attn_output[blockIdx.x * n + col_index] +
            block_input[blockIdx.x * n + col_index] +
            bias[col_index];
    }
}


template<typename T>
__global__ void addBiasAttentionFfnResidual(
    T* block_output,
    const T* ffn_output,
    const T* attn_output,
    const T* bias,
    const int m, const int n)
{
    const int col_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (col_index < n) {
        block_output[blockIdx.x * n + col_index] +=
            ffn_output[blockIdx.x * n + col_index] +
            attn_output[blockIdx.x * n + col_index] +
            bias[col_index];
    }
}

template<typename T>
void invokeAddBiasAttentionFfnResidual(
    T* block_output,
    const T* ffn_output,
    const T* attn_output,
    const T* block_input,
    const T* bias,
    const int m,
    const int n,
    cudaStream_t stream
)
{
    int blocks_per_row = ceil(float(n) / 1024);
    dim3 grid(m, blocks_per_row);
    dim3 block(min(n, 1024));
    if (block_output == block_input) {
        addBiasAttentionFfnResidual<<<grid, block, 0, stream>>>(
            block_output,
            ffn_output,
            attn_output,
            bias,
            m, n
        );
    } else {
        addBiasAttentionFfnResidual<<<grid, block, 0, stream>>>(
            block_output,
            ffn_output,
            attn_output,
            block_input,
            bias,
            m, n
        );
    }
}

template void invokeAddBiasResidual(
    float* output, const float* input, const float* bias, const int m, const int n, cudaStream_t stream);

template void invokeAddBiasResidual(
    half* output, const half* input, const half* bias, const int m, const int n, cudaStream_t stream);

template void invokeAddBiasAttentionFfnResidual(
    float* block_output, const float* ffn_output, const float* attn_output, const float* input, const float* bias, const int m, const int n, cudaStream_t stream);

template void invokeAddBiasAttentionFfnResidual(
    half* block_output, const half* ffn_output, const half* attn_output, const half* input, const half* bias, const int m, const int n, cudaStream_t stream);

template<typename T>
__global__ void T5addResidual(T* output, const T* input, const int m, const int n)
{
    const int col_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (col_index < n) {
        float out_val = (float)output[blockIdx.x * n + col_index] + (float)input[blockIdx.x * n + col_index];
        output[blockIdx.x * n + col_index] = (T)((std::is_same<T, half>::value && (out_val > 64512 || out_val < -64512)) ?
            (out_val > 0 ? 64512 : -64512) : out_val);
    }
}

template<typename T>
void invokeT5AddResidual(T* output, const T* input, const int m, const int n, cudaStream_t stream)
{
    int blocks_per_row = ceil(float(n) / 1024);
    dim3 grid(m, blocks_per_row);
    dim3 block(min(n, 1024));
    T5addResidual<<<grid, block, 0, stream>>>(output, input, m, n);
}

template void invokeT5AddResidual(float* output, const float* input, const int m, const int n, cudaStream_t stream);
template void invokeT5AddResidual(half* output, const half* input, const int m, const int n, cudaStream_t stream);

}  // namespace fastertransformer
