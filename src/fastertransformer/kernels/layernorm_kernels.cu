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

#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"

namespace fastertransformer {

// TODO(bhsueh) add half2 implementation
template<typename T, int N>
__global__ void
addBiasResidualPostLayerNorm(T* out, const T* input, const T* bias, const T* gamma, const T* beta, int m, int n)
{
    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;
    float local_out_cache[N];

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
        float local_out = (float)(out[blockIdx.x * n + idx] + input[blockIdx.x * n + idx] + __ldg(&bias[idx]));
        mean += local_out;
        // save local_out to local_out_cache to save some recompute
        local_out_cache[i] = local_out;
        idx += blockDim.x;
    }

    mean = blockReduceSum<float>(mean);
    if (threadIdx.x == 0)
        s_mean = mean / n;
    __syncthreads();

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
        float local_out = local_out_cache[i];
        variance += (local_out - s_mean) * (local_out - s_mean);
        idx += blockDim.x;
    }
    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0)
        s_variance = variance / n + 1e-6f;
    __syncthreads();

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
        float local_out = local_out_cache[i];
        out[blockIdx.x * n + idx] =
            (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg(&gamma[idx])) + (float)(__ldg(&beta[idx])));
        idx += blockDim.x;
    }
}

template<int N>
__global__ void addBiasResidualPostLayerNormHalf(
    half* out, const half* input, const half* bias, const half* gamma, const half* beta, int m, int n)
{
    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    half2* out_ptr = (half2*)out;
    const half2* input_ptr = (const half2*)input;
    const half2* bias_ptr = (const half2*)bias;
    const half2* gamma_ptr = (const half2*)gamma;
    const half2* beta_ptr = (const half2*)beta;

    float2 out_fp2_cache[N];

    float local_out = 0.0f;
#pragma unroll N
    for (int idx = threadIdx.x, i = 0; idx < n / 2 && i < N; ++i) {
        int id = blockIdx.x * n / 2 + idx;
        float2 local_out_fp2 = __half22float2(__hadd2(__hadd2(out_ptr[id], input_ptr[id]), __ldg(&bias_ptr[idx])));
        local_out += local_out_fp2.x;
        local_out += local_out_fp2.y;
        // save local_out_fp2 to out_fp2_cache to save some recomputation
        out_fp2_cache[i] = local_out_fp2;
        idx += blockDim.x;
    }

    mean = blockReduceSum<float>(local_out);
    if (threadIdx.x == 0)
        s_mean = mean / n;
    __syncthreads();

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; i < N && idx < n / 2; ++i) {
        float2 local_out_fp2 = out_fp2_cache[i];
        variance += (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
        variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
        idx += blockDim.x;
    }

    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0)
        s_variance = rsqrtf(variance / n + 1e-6f);
    __syncthreads();

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; i < N && idx < n / 2; ++i) {
        int id = blockIdx.x * n / 2 + idx;
        float2 local_out_fp2 = out_fp2_cache[i];
        float2 gamma_val = __half22float2(__ldg(&gamma_ptr[idx]));
        float2 beta_val = __half22float2(__ldg(&beta_ptr[idx]));
        local_out_fp2.x = (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
        local_out_fp2.y = (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
        out_ptr[id] = __float22half2_rn(local_out_fp2);
        idx += blockDim.x;
    }
}

template<typename T>
__global__ void
generalAddBiasResidualPostLayerNorm(T* out, const T* input, const T* bias, const T* gamma, const T* beta, int m, int n)
{
    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float local_out = (float)(out[blockIdx.x * n + idx] + input[blockIdx.x * n + idx] + __ldg(&bias[idx]));
        mean += local_out;
        // save local_out to out to save some recompute
        out[blockIdx.x * n + idx] = local_out;
    }

    mean = blockReduceSum<float>(mean);
    if (threadIdx.x == 0)
        s_mean = mean / n;
    __syncthreads();

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float local_out = out[blockIdx.x * n + idx];
        variance += (local_out - s_mean) * (local_out - s_mean);
    }
    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0)
        s_variance = variance / n + 1e-6f;
    __syncthreads();

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float local_out = out[blockIdx.x * n + idx];
        out[blockIdx.x * n + idx] =
            (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg(&gamma[idx])) + (float)(__ldg(&beta[idx])));
    }
}

template<>
__global__ void generalAddBiasResidualPostLayerNorm(
    half* out, const half* input, const half* bias, const half* gamma, const half* beta, int m, int n)
{
    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    half2* out_ptr = (half2*)out;
    const half2* input_ptr = (const half2*)input;
    const half2* bias_ptr = (const half2*)bias;
    const half2* gamma_ptr = (const half2*)gamma;
    const half2* beta_ptr = (const half2*)beta;

    float local_out = 0.0f;
    for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
        int id = blockIdx.x * n / 2 + idx;
        half2 tmp = __hadd2(__hadd2(out_ptr[id], input_ptr[id]), __ldg(&bias_ptr[idx]));
        float2 local_out_fp2 = __half22float2(tmp);
        local_out += local_out_fp2.x;
        local_out += local_out_fp2.y;
        // save tmp to out_ptr to save some recomputation
        out_ptr[id] = tmp;
    }

    mean = blockReduceSum<float>(local_out);
    if (threadIdx.x == 0)
        s_mean = mean / n;
    __syncthreads();

    for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
        int id = blockIdx.x * n / 2 + idx;
        float2 local_out_fp2 = __half22float2(out_ptr[id]);
        variance += (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
        variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
    }

    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0)
        s_variance = rsqrtf(variance / n + 1e-6f);
    __syncthreads();

    for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
        int id = blockIdx.x * n / 2 + idx;
        float2 local_out_fp2 = __half22float2(out_ptr[id]);
        float2 gamma_val = __half22float2(__ldg(&gamma_ptr[idx]));
        float2 beta_val = __half22float2(__ldg(&beta_ptr[idx]));
        local_out_fp2.x = (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
        local_out_fp2.y = (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
        out_ptr[id] = __float22half2_rn(local_out_fp2);
    }
}

template<typename T>
__global__ void addBiasResidualPostLayerNormV2(T* out,
                                               const T* __restrict input,
                                               const T* __restrict bias,
                                               const T* __restrict gamma,
                                               const T* __restrict beta,
                                               int n)
{
    const int ite = 4;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;
    float local_out[ite];

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        int id = bid * n + col_id;
        local_out[i] = (float)(out[id] + __ldg(&input[id]) + __ldg(&bias[col_id]));
        sum += local_out[i];
    }

    mean = blockReduceSum<float>(sum);
    if (tid == 0)
        s_mean = mean / n;
    __syncthreads();

    float var = 0.0f;
#pragma unroll
    for (int i = 0; i < ite; i++) {
        float diff = local_out[i] - s_mean;
        var += diff * diff;
    }

    variance = blockReduceSum<float>(var);
    if (tid == 0)
        s_variance = rsqrtf(variance / n + 1e-6f);
    __syncthreads();

#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        int id = bid * n + col_id;
        out[id] =
            (T)((local_out[i] - s_mean) * s_variance * (float)__ldg(&gamma[col_id]) + (float)__ldg(&beta[col_id]));
    }
}

template<>
__global__ void addBiasResidualPostLayerNormV2(half* out,
                                               const half* __restrict input,
                                               const half* __restrict bias,
                                               const half* __restrict gamma,
                                               const half* __restrict beta,
                                               int n)
{
    const int ite = 4;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;
    half2 local_out_half2[ite];

    half2* out_ptr = (half2*)out;
    const half2* input_ptr = (const half2*)input;
    const half2* bias_ptr = (const half2*)bias;
    const half2* gamma_ptr = (const half2*)gamma;
    const half2* beta_ptr = (const half2*)beta;

    // float sum = 0.0f;
    half2 sum = __float2half2_rn(0.0f);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        int id = bid * n / 2 + col_id;
        local_out_half2[i] = out_ptr[id] + __ldg(&input_ptr[id]) + __ldg(&bias_ptr[col_id]);
        sum += local_out_half2[i];
    }

    mean = blockReduceSum<float>((float)(sum.x + sum.y));
    if (threadIdx.x == 0)
        s_mean = mean / n;
    __syncthreads();

    float var = 0.0f;
    half2 s_mean_2 = __float2half2_rn(s_mean);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        local_out_half2[i] = local_out_half2[i] - s_mean_2;
        float v1 = (float)local_out_half2[i].x;
        float v2 = (float)local_out_half2[i].y;
        var += v1 * v1 + v2 * v2;
    }

    variance = blockReduceSum<float>(var);
    if (threadIdx.x == 0)
        s_variance = rsqrtf(variance / n + 1e-6f);
    __syncthreads();

    half2 s_var_2 = __float2half2_rn(s_variance);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        int id = bid * n / 2 + col_id;
        out_ptr[id] = local_out_half2[i] * s_var_2 * __ldg(&gamma_ptr[col_id]) + __ldg(&beta_ptr[col_id]);
    }
}

template<typename T>
void invokeAddBiasResidualLayerNorm(
    T* out, const T* input, const T* bias, const T* gamma, const T* beta, int m, int n, cudaStream_t stream)
{
    dim3 grid(m);
    dim3 block(std::min(n, 1024));
    if (n == 768 || n == 1024) {
        addBiasResidualPostLayerNormV2<T><<<grid, n / 4, 0, stream>>>(out, input, bias, gamma, beta, n);
    }
    else {
        block.x = std::min(n, 1024);
        int num_trips = (n + block.x - 1) / block.x;
        if (num_trips == 1) {
            addBiasResidualPostLayerNorm<T, 1><<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, m, n);
        }
        else if (num_trips == 2) {
            addBiasResidualPostLayerNorm<T, 2><<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, m, n);
        }
        else {
            generalAddBiasResidualPostLayerNorm<T><<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, m, n);
        }
    }
}

template<>
void invokeAddBiasResidualLayerNorm(half* out,
                                    const half* input,
                                    const half* bias,
                                    const half* gamma,
                                    const half* beta,
                                    int m,
                                    int n,
                                    cudaStream_t stream)
{
    dim3 grid(m);
    dim3 block(std::min(n, 1024));

    if (m >= 512 && (n == 768 || n == 1024)) {
        addBiasResidualPostLayerNormV2<half><<<grid, n / 8, 0, stream>>>(out, input, bias, gamma, beta, n);
    }
    else {
        block.x = std::min(n, 1024);
        int num_trips = (n + block.x - 1) / block.x;
        if (num_trips == 1) {
            addBiasResidualPostLayerNorm<half, 1><<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, m, n);
        }
        else if (num_trips == 2) {
            addBiasResidualPostLayerNorm<half, 2><<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, m, n);
        }
        else {
            generalAddBiasResidualPostLayerNorm<half><<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, m, n);
        }
    }
}

template void invokeAddBiasResidualLayerNorm(float* out,
                                             const float* input,
                                             const float* bias,
                                             const float* gamma,
                                             const float* beta,
                                             int m,
                                             int n,
                                             cudaStream_t stream);
template void invokeAddBiasResidualLayerNorm(half* out,
                                             const half* input,
                                             const half* bias,
                                             const half* gamma,
                                             const half* beta,
                                             int m,
                                             int n,
                                             cudaStream_t stream);

template<typename T>
__global__ void generalAddBiasResidualLayerNorm(const T* __restrict input,
                                                const T* __restrict gamma,
                                                const T* __restrict beta,
                                                const T* __restrict bias,
                                                T* output,
                                                T* norm_output,
                                                int m,
                                                int n)
{
    int tid = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float local_out = (float)(__ldg(&input[blockIdx.x * n + i]));
        local_out += (float)(output[blockIdx.x * n + i]);
        if (bias != nullptr) {
            local_out += (float)(__ldg(&bias[i]));
        }
        output[blockIdx.x * n + i] = (T)local_out;
        local_sum += local_out;
    }

    mean = blockReduceSum(local_sum);

    if (threadIdx.x == 0)
        s_mean = mean / n;
    __syncthreads();

    float local_var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = (float)(output[blockIdx.x * n + i]) - s_mean;
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0)
        s_variance = rsqrtf(variance / n + 1e-6);
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        float beta_val = (beta == nullptr) ? 0.0f : (float)(__ldg(&beta[i]));
        norm_output[blockIdx.x * n + i] =
            (T)((((float)output[blockIdx.x * n + i] - s_mean) * s_variance) * (float)(__ldg(&gamma[i])) + beta_val);
    }
}

template<typename T>
void invokeGeneralAddBiasResidualPreLayerNorm(T* output,
                                              T* norm_output,
                                              const T* input,
                                              const T* gamma,
                                              const T* beta,
                                              const T* bias,
                                              int m,
                                              int n,
                                              cudaStream_t stream)
{
    dim3 grid(m);
    dim3 block(min(n, 1024));

    /* For general cases, n is equal to hidden_units, e.g., 512/1024.
    Since we have warp shuffle inside the code, block.x % 32 should be 0.
    */

    if (n % 32 != 0)
        block.x = 1024;

    block.x = block.x / (4 / sizeof(T));  // if using half, only need half of block.x

    /* should pay attention to the rsqrt precision*/
    generalAddBiasResidualLayerNorm<T>
        <<<grid, block, 0, stream>>>(input, gamma, beta, bias, output, norm_output, m, n);  // For gpt-3
}

template void invokeGeneralAddBiasResidualPreLayerNorm(float* output,
                                                       float* norm_output,
                                                       const float* input,
                                                       const float* gamma,
                                                       const float* beta,
                                                       const float* bias,
                                                       int m,
                                                       int n,
                                                       cudaStream_t stream);

template void invokeGeneralAddBiasResidualPreLayerNorm(half* output,
                                                       half* norm_output,
                                                       const half* input,
                                                       const half* gamma,
                                                       const half* beta,
                                                       const half* bias,
                                                       int m,
                                                       int n,
                                                       cudaStream_t stream);

template<typename T>
__global__ void generalAddResidualT5LayerNorm(
    const T* __restrict input, const T* __restrict gamma, T* output, T* norm_output, int m, int n)
{
    // layernorm module in the T5 style No bias and no subtraction of mean.
    __shared__ float s_variance;
    float variance = 0.0f;

    float local_var_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        output[blockIdx.x * n + i] =
            clamp_inf_for_half<T>((float)__ldg(&input[blockIdx.x * n + i]) + (float)output[blockIdx.x * n + i]);

        float diff = (float)(output[blockIdx.x * n + i]);
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + 1e-6);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float out_val = (((float)output[blockIdx.x * n + i]) * s_variance) * (float)(__ldg(&gamma[i]));
        norm_output[blockIdx.x * n + i] =
            clamp_inf_for_half<T>((((float)output[blockIdx.x * n + i]) * s_variance) * (float)(__ldg(&gamma[i])));
    }
}

template<typename T>
void invokeGeneralAddResidualT5PreLayerNorm(
    T* output, T* norm_output, const T* input, const T* gamma, int m, int n, cudaStream_t stream)
{
    dim3 grid(m);
    dim3 block(min(n, 1024));

    /* For general cases, n is equal to hidden_units, e.g., 512/1024.
    Since we have warp shuffle inside the code, block.x % 32 should be 0.
    */

    if (n % 32 != 0) {
        block.x = 1024;
    }

    block.x = block.x / (4 / sizeof(T));  // if using half, only need half of block.x

    /* should pay attention to the rsqrt precision*/
    generalAddResidualT5LayerNorm<T><<<grid, block, 0, stream>>>(input, gamma, output, norm_output, m, n);
}

template void invokeGeneralAddResidualT5PreLayerNorm(
    float* output, float* norm_output, const float* input, const float* gamma, int m, int n, cudaStream_t stream);

template void invokeGeneralAddResidualT5PreLayerNorm(
    half* output, half* norm_output, const half* input, const half* gamma, int m, int n, cudaStream_t stream);

template<typename T>
__global__ void generalLayerNorm(
    const T* __restrict input, const T* __restrict gamma, const T* __restrict beta, T* output, int m, int n)
{
    const int tid = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local_sum += (float)(__ldg(&input[blockIdx.x * n + i]));
    }

    mean = blockReduceSum(local_sum);

    if (threadIdx.x == 0)
        s_mean = mean / n;
    __syncthreads();

    float local_var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = (float)(__ldg(&input[blockIdx.x * n + i])) - s_mean;
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0)
        s_variance = rsqrtf(variance / n + 1e-6);
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        float beta_val = (beta == nullptr) ? 0.0f : (float)__ldg(&beta[i]);
        output[blockIdx.x * n + i] =
            (T)((((float)input[blockIdx.x * n + i] - s_mean) * s_variance) * (float)(__ldg(&gamma[i])) + beta_val);
    }
}

template<typename T>
void invokeGeneralLayerNorm(
    T* out, const T* input, const T* gamma, const T* beta, const int m, const int n, cudaStream_t stream)
{
    dim3 grid(m);
    dim3 block(min(n, 1024));

    /* For general cases, n is equal to hidden_units, e.g., 512/1024.
        Since we have warp shuffle inside the code, block.x % 32 should be 0.
    */
    if (n % 32 != 0)
        block.x = 1024;

    block.x = block.x / (4 / sizeof(T));  // if using half, only need half of block.x

    /* should pay attention to the rsqrt precision*/
    generalLayerNorm<T><<<grid, block, 0, stream>>>(input, gamma, beta, out, m, n);  // For gpt-3
}

template void invokeGeneralLayerNorm(float* out,
                                     const float* input,
                                     const float* gamma,
                                     const float* beta,
                                     const int m,
                                     const int n,
                                     cudaStream_t stream);
template void invokeGeneralLayerNorm(
    half* out, const half* input, const half* gamma, const half* beta, const int m, const int n, cudaStream_t stream);

template<typename T>
__global__ void generalT5LayerNorm(const T* __restrict input, const T* __restrict gamma, T* output, int m, int n)
{
    // layernorm module in the T5 style No bias and no subtraction of mean.
    const int tid = threadIdx.x;

    __shared__ float s_variance;
    float variance = 0.0f;

    float local_var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = (float)(__ldg(&input[blockIdx.x * n + i]));
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0)
        s_variance = rsqrtf(variance / n + 1e-6);
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        output[blockIdx.x * n + i] =
            clamp_inf_for_half<T>((((float)input[blockIdx.x * n + i]) * s_variance) * (float)(__ldg(&gamma[i])));
    }
}

template<typename T>
void invokeGeneralT5LayerNorm(T* out, const T* input, const T* gamma, const int m, const int n, cudaStream_t stream)
{
    dim3 grid(m);
    dim3 block(min(n, 1024));

    /* For general cases, n is equal to hidden_units, e.g., 512/1024.
        Since we have warp shuffle inside the code, block.x % 32 should be 0.
    */
    if (n % 32 != 0)
        block.x = 1024;

    block.x = block.x / (4 / sizeof(T));  // if using half, only need half of block.x

    /* should pay attention to the rsqrt precision*/
    generalT5LayerNorm<T><<<grid, block, 0, stream>>>(input, gamma, out, m, n);  // For gpt-3
}

template void invokeGeneralT5LayerNorm(
    float* out, const float* input, const float* gamma, const int m, const int n, cudaStream_t stream);
template void invokeGeneralT5LayerNorm(
    half* out, const half* input, const half* gamma, const int m, const int n, cudaStream_t stream);

}  // namespace fastertransformer