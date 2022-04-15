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

#include "src/fastertransformer/kernels/bfloat16_fallback_kenrels.cuh"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"

namespace fastertransformer {

// * Note that typename T is half2 or bfloat2 type
template<typename T, bool IS_OUTPUT, bool IS_BIAS, bool IS_RESIDUAL, bool IS_BETA, int UNROLL_FACTOR>
__global__ void generalAddBiasResidualLayerNormOpt(T* normed_output,
                                                   T* output,
                                                   const T* __restrict bias,
                                                   const T* __restrict residual,
                                                   const T* __restrict gamma,
                                                   const T* __restrict beta,
                                                   int m,
                                                   int n)
{
    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    T local_sum = float2type2<T>(0.0f);
#pragma unroll
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = blockIdx.x * n + i;
        T val = float2type2<T>(0.0f);

        if (IS_BIAS) {
            val = hadd2(val, ldg(&bias[i]));
        }
        if (IS_RESIDUAL) {
            val = hadd2(val, ldg(&residual[index]));
        }

        if (IS_OUTPUT) {
            val = hadd2(val, output[index]);
        }
        output[index] = val;
        local_sum = hadd2(local_sum, val);
    }

    mean = blockReduceSum((float)(local_sum.x + local_sum.y));

    if (threadIdx.x == 0) {
        s_mean = mean / n / 2;
    }
    __syncthreads();

    float local_var_sum = 0.0f;
#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        T val = output[blockIdx.x * n + i];
        float diff_1 = (float)(val.x) - s_mean;
        float diff_2 = (float)(val.y) - s_mean;
        local_var_sum += (diff_1 * diff_1 + diff_2 * diff_2);
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n / 2 + 1e-6f);
    }
    __syncthreads();

    T mean_2 = float2type2<T>(s_mean);
    T var_2 = float2type2<T>(s_variance);
#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = blockIdx.x * n + i;
        T val = hmul2(hmul2(hsub2(output[index], mean_2), var_2), ldg(&gamma[i]));
        if (IS_BETA) {
            val = hadd2(val, ldg(&beta[i]));
        }
        normed_output[index] = val;
    }
}

// * Note that typename T is half2 or bfloat2 type
template<typename T, bool IS_OUTPUT, bool IS_BIAS, bool IS_RESIDUAL, bool IS_BETA, int UNROLL_FACTOR>
__global__ void generalAddBiasResidualLayerNormOpt2(T* normed_output,
                                                    T* output,
                                                    const T* __restrict bias,
                                                    const T* __restrict residual,
                                                    const T* __restrict gamma,
                                                    const T* __restrict beta,
                                                    int m,
                                                    int n)
{
    __shared__ float s_mean;
    __shared__ float s_variance;
    float x_sum = 0.0f;
    float x2_sum = 0.0f;
    const int b_offset = blockIdx.x * n;
    using T1 = typename TypeConverter<T>::Type;

#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = b_offset + i;
        float val_1 = 0.0f;
        float val_2 = 0.0f;
        T tmp;

        if (IS_BIAS) {
            tmp = ldg(&bias[i]);
            val_1 += static_cast<float>(tmp.x);
            val_2 += static_cast<float>(tmp.y);
        }
        if (IS_RESIDUAL) {
            tmp = ldg(&residual[index]);
            val_1 += static_cast<float>(tmp.x);
            val_2 += static_cast<float>(tmp.y);
        }

        if (IS_OUTPUT) {
            tmp = ldg(&output[index]);
            val_1 += static_cast<float>(tmp.x);
            val_2 += static_cast<float>(tmp.y);
        }
        tmp.x = float2type<T1>(val_1);
        tmp.y = float2type<T1>(val_2);
        output[index] = tmp;
        x_sum += val_1 + val_2;
        x2_sum += val_1 * val_1 + val_2 * val_2;
    }
    float sums[2];
    sums[0] = x_sum;
    sums[1] = x2_sum;
    blockReduceSumV2<float, 2>(sums);

    if (threadIdx.x == 0) {
        s_mean = sums[0] / n / 2;
        s_variance = rsqrtf(sums[1] / n / 2 - s_mean * s_mean + 1e-6f);
    }
    __syncthreads();

    T mean_2 = float2type2<T>(s_mean);
    T var_2 = float2type2<T>(s_variance);

#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = b_offset + i;
        T val = hmul2(hmul2(hsub2(output[index], mean_2), var_2), ldg(&gamma[i]));
        if (IS_BETA) {
            val = hadd2(val, ldg(&beta[i]));
        }
        normed_output[index] = val;
    }
}

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
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
        float local_out = local_out_cache[i];
        variance += (local_out - s_mean) * (local_out - s_mean);
        idx += blockDim.x;
    }
    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0) {
        s_variance = variance / n + 1e-6f;
    }
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
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; i < N && idx < n / 2; ++i) {
        float2 local_out_fp2 = out_fp2_cache[i];
        variance += (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
        variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
        idx += blockDim.x;
    }

    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + 1e-6f);
    }
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
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float local_out = out[blockIdx.x * n + idx];
        variance += (local_out - s_mean) * (local_out - s_mean);
    }
    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0) {
        s_variance = variance / n + 1e-6f;
    }
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
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
        int id = blockIdx.x * n / 2 + idx;
        float2 local_out_fp2 = __half22float2(out_ptr[id]);
        variance += (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
        variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
    }

    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + 1e-6f);
    }
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
    if (tid == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float var = 0.0f;
#pragma unroll
    for (int i = 0; i < ite; i++) {
        float diff = local_out[i] - s_mean;
        var += diff * diff;
    }

    variance = blockReduceSum<float>(var);
    if (tid == 0) {
        s_variance = rsqrtf(variance / n + 1e-6f);
    }
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
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
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
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + 1e-6f);
    }
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
        float local_out = (float)(ldg(&input[blockIdx.x * n + i]));
        local_out += (float)(output[blockIdx.x * n + i]);
        if (bias != nullptr) {
            local_out += (float)(ldg(&bias[i]));
        }
        output[blockIdx.x * n + i] = (T)local_out;
        local_sum += local_out;
    }

    mean = blockReduceSum(local_sum);

    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float local_var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = (float)(output[blockIdx.x * n + i]) - s_mean;
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + 1e-6f);
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        float beta_val = (beta == nullptr) ? 0.0f : (float)(ldg(&beta[i]));
        norm_output[blockIdx.x * n + i] =
            (T)((((float)output[blockIdx.x * n + i] - s_mean) * s_variance) * (float)(ldg(&gamma[i])) + beta_val);
    }
}

#define HALF_LAYERNORM_OPT(UNROLL_FACTOR)                                                                              \
    generalAddBiasResidualLayerNormOpt<T2, true, true, true, true, UNROLL_FACTOR>                                      \
        <<<grid, block, 0, stream>>>((T2*)norm_output,                                                                 \
                                     (T2*)output,                                                                      \
                                     (const T2*)bias,                                                                  \
                                     (const T2*)input,                                                                 \
                                     (const T2*)gamma,                                                                 \
                                     (const T2*)beta,                                                                  \
                                     m,                                                                                \
                                     half_n);

#define HALF_LAYERNORM_OPT2(UNROLL_FACTOR)                                                                             \
    generalAddBiasResidualLayerNormOpt2<T2, true, true, true, true, UNROLL_FACTOR>                                     \
        <<<grid, block, 0, stream>>>((T2*)norm_output,                                                                 \
                                     (T2*)output,                                                                      \
                                     (const T2*)bias,                                                                  \
                                     (const T2*)input,                                                                 \
                                     (const T2*)gamma,                                                                 \
                                     (const T2*)beta,                                                                  \
                                     m,                                                                                \
                                     half_n);

template<typename T>
void invokeGeneralAddBiasResidualPreLayerNorm(T* output,
                                              T* norm_output,
                                              const T* input,
                                              const T* gamma,
                                              const T* beta,
                                              const T* bias,
                                              int m,
                                              int n,
                                              cudaStream_t stream,
                                              int opt_version)
{
    if (opt_version > 0 && sizeof(T) == 2 && n % 2 == 0) {
        dim3 grid(m);
        int half_n = n / 2;
        int half_n_32 = (half_n + 31) / 32 * 32;
        dim3 block(min(half_n_32, 512));
        int rolls_per_thread = half_n / block.x;
        int unroll_factor = 8;
        while (unroll_factor > rolls_per_thread && unroll_factor > 1) {
            unroll_factor /= 2;
        }
        using T2 = typename TypeConverter<T>::Type;
        if (opt_version == 1) {
            if (unroll_factor == 1) {
                HALF_LAYERNORM_OPT(1);
            }
            else if (unroll_factor == 2) {
                HALF_LAYERNORM_OPT(2);
            }
            else if (unroll_factor == 3) {
                HALF_LAYERNORM_OPT(3);
            }
            else if (unroll_factor == 4) {
                HALF_LAYERNORM_OPT(4);
            }
            else if (unroll_factor == 8) {
                HALF_LAYERNORM_OPT(8);
            }
        }
        else {
            if (unroll_factor == 1) {
                HALF_LAYERNORM_OPT2(1);
            }
            else if (unroll_factor == 2) {
                HALF_LAYERNORM_OPT2(2);
            }
            else if (unroll_factor == 3) {
                HALF_LAYERNORM_OPT2(3);
            }
            else if (unroll_factor == 4) {
                HALF_LAYERNORM_OPT2(4);
            }
            else if (unroll_factor == 8) {
                HALF_LAYERNORM_OPT2(8);
            }
        }
    }
    else {

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
        generalAddBiasResidualLayerNorm<T>
            <<<grid, block, 0, stream>>>(input, gamma, beta, bias, output, norm_output, m, n);  // For gpt-3
    }
}

#undef HALF_LAYERNORM_OPT
#undef HALF_LAYERNORM_OPT2

template void invokeGeneralAddBiasResidualPreLayerNorm(float* output,
                                                       float* norm_output,
                                                       const float* input,
                                                       const float* gamma,
                                                       const float* beta,
                                                       const float* bias,
                                                       int m,
                                                       int n,
                                                       cudaStream_t stream,
                                                       int opt_version);

template void invokeGeneralAddBiasResidualPreLayerNorm(half* output,
                                                       half* norm_output,
                                                       const half* input,
                                                       const half* gamma,
                                                       const half* beta,
                                                       const half* bias,
                                                       int m,
                                                       int n,
                                                       cudaStream_t stream,
                                                       int opt_version);

#ifdef ENABLE_BF16
template void invokeGeneralAddBiasResidualPreLayerNorm(__nv_bfloat16* output,
                                                       __nv_bfloat16* norm_output,
                                                       const __nv_bfloat16* input,
                                                       const __nv_bfloat16* gamma,
                                                       const __nv_bfloat16* beta,
                                                       const __nv_bfloat16* bias,
                                                       int m,
                                                       int n,
                                                       cudaStream_t stream,
                                                       int opt_version);
#endif

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
        s_variance = rsqrtf(variance / n + 1e-6f);
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
void invokeGeneralAddBiasResidualT5PreLayerNorm(T* output,
                                                T* norm_output,
                                                const T* input,
                                                const T* gamma,
                                                const T* beta,
                                                const T* bias,
                                                int m,
                                                int n,
                                                cudaStream_t stream)
{
    if (beta != nullptr && bias != nullptr) {
        invokeGeneralAddBiasResidualPreLayerNorm(output, norm_output, input, gamma, beta, bias, m, n, stream);
    }
    else {
        invokeGeneralAddResidualT5PreLayerNorm(output, norm_output, input, gamma, m, n, stream);
    }
    return;
}

template void invokeGeneralAddBiasResidualT5PreLayerNorm(float* output,
                                                         float* norm_output,
                                                         const float* input,
                                                         const float* gamma,
                                                         const float* beta,
                                                         const float* bias,
                                                         int m,
                                                         int n,
                                                         cudaStream_t stream);

template void invokeGeneralAddBiasResidualT5PreLayerNorm(half* output,
                                                         half* norm_output,
                                                         const half* input,
                                                         const half* gamma,
                                                         const half* beta,
                                                         const half* bias,
                                                         int m,
                                                         int n,
                                                         cudaStream_t stream);

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
        local_sum += (float)(ldg(&input[blockIdx.x * n + i]));
    }

    mean = blockReduceSum(local_sum);

    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float local_var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = (float)(ldg(&input[blockIdx.x * n + i])) - s_mean;
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + 1e-6f);
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        float beta_val = (beta == nullptr) ? 0.0f : (float)ldg(&beta[i]);
        output[blockIdx.x * n + i] =
            (T)((((float)input[blockIdx.x * n + i] - s_mean) * s_variance) * (float)(ldg(&gamma[i])) + beta_val);
    }
}

#define HALF_LAYERNORM_OPT(UNROLL_FACTOR)                                                                              \
    generalAddBiasResidualLayerNormOpt<T2, false, false, true, true, UNROLL_FACTOR><<<grid, block, 0, stream>>>(       \
        (T2*)out, (T2*)out, nullptr, (const T2*)input, (const T2*)gamma, (const T2*)beta, m, half_n);

#define HALF_LAYERNORM_OPT2(UNROLL_FACTOR)                                                                             \
    generalAddBiasResidualLayerNormOpt2<T2, false, false, true, true, UNROLL_FACTOR><<<grid, block, 0, stream>>>(      \
        (T2*)out, (T2*)out, nullptr, (const T2*)input, (const T2*)gamma, (const T2*)beta, m, half_n);

template<typename T>
void invokeGeneralLayerNorm(T* out,
                            const T* input,
                            const T* gamma,
                            const T* beta,
                            const int m,
                            const int n,
                            cudaStream_t stream,
                            int opt_version)
{
    dim3 grid(m);
    if (n % 2 == 0 && std::is_same<T, half>::value && opt_version > 0) {
        int half_n = n / 2;
        int half_n_32 = (half_n + 31) / 32 * 32;
        dim3 block(min(half_n_32, 512));
        int rolls_per_thread = half_n / block.x;
        int unroll_factor = 8;
        while (unroll_factor > rolls_per_thread && unroll_factor > 1) {
            unroll_factor /= 2;
        }
        using T2 = typename TypeConverter<T>::Type;
        if (opt_version == 1) {
            if (unroll_factor == 1) {
                HALF_LAYERNORM_OPT(1);
            }
            else if (unroll_factor == 2) {
                HALF_LAYERNORM_OPT(2);
            }
            else if (unroll_factor == 3) {
                HALF_LAYERNORM_OPT(3);
            }
            else if (unroll_factor == 4) {
                HALF_LAYERNORM_OPT(4);
            }
            else if (unroll_factor == 8) {
                HALF_LAYERNORM_OPT(8);
            }
        }
        else {
            if (unroll_factor == 1) {
                HALF_LAYERNORM_OPT2(1);
            }
            else if (unroll_factor == 2) {
                HALF_LAYERNORM_OPT2(2);
            }
            else if (unroll_factor == 3) {
                HALF_LAYERNORM_OPT2(3);
            }
            else if (unroll_factor == 4) {
                HALF_LAYERNORM_OPT2(4);
            }
            else if (unroll_factor == 8) {
                HALF_LAYERNORM_OPT2(8);
            }
        }
    }
    else {
        dim3 block(min(n, 1024));

        /* For general cases, n is equal to hidden_units, e.g., 512/1024.
            Since we have warp shuffle inside the code, block.x % 32 should be 0.
        */
        if (n % 32 != 0) {
            block.x = 1024;
        }

        /* should pay attention to the rsqrt precision*/
        generalLayerNorm<T><<<grid, block, 0, stream>>>(input, gamma, beta, out, m, n);  // For gpt-3
    }
}

#undef HALF_LAYERNORM_OPT
#undef HALF_LAYERNORM_OPT2

template void invokeGeneralLayerNorm(float* out,
                                     const float* input,
                                     const float* gamma,
                                     const float* beta,
                                     const int m,
                                     const int n,
                                     cudaStream_t stream,
                                     int opt_version);
template void invokeGeneralLayerNorm(half* out,
                                     const half* input,
                                     const half* gamma,
                                     const half* beta,
                                     const int m,
                                     const int n,
                                     cudaStream_t stream,
                                     int opt_version);
#ifdef ENABLE_BF16
template void invokeGeneralLayerNorm(__nv_bfloat16* out,
                                     const __nv_bfloat16* input,
                                     const __nv_bfloat16* gamma,
                                     const __nv_bfloat16* beta,
                                     const int m,
                                     const int n,
                                     cudaStream_t stream,
                                     int opt_version);
#endif

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

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + 1e-6f);
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        output[blockIdx.x * n + i] =
            clamp_inf_for_half<T>((((float)input[blockIdx.x * n + i]) * s_variance) * (float)(__ldg(&gamma[i])));
    }
}

template<typename T>
void invokeGeneralT5LayerNorm(
    T* out, const T* input, const T* gamma, const T* beta, const int m, const int n, cudaStream_t stream)
{
    if (beta != nullptr) {
        invokeGeneralLayerNorm(out, input, gamma, beta, m, n, stream);
        return;
    }

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
    generalT5LayerNorm<T><<<grid, block, 0, stream>>>(input, gamma, out, m, n);  // For gpt-3
}

template void invokeGeneralT5LayerNorm(float* out,
                                       const float* input,
                                       const float* gamma,
                                       const float* beta,
                                       const int m,
                                       const int n,
                                       cudaStream_t stream);
template void invokeGeneralT5LayerNorm(
    half* out, const half* input, const half* gamma, const half* beta, const int m, const int n, cudaStream_t stream);

/*******************  invokeLayernormShiftPartition  ***********************/

template<typename T>
__global__ void layernorm_shift_partition(T* out,
                                          const T* input,
                                          const T* gamma,
                                          const T* beta,
                                          int batch,
                                          int H,
                                          int W,
                                          int n,
                                          int shift_size,
                                          int window_size)
{
    int tid = threadIdx.x;
    const int batch_offset = blockIdx.z * gridDim.y * gridDim.x;
    const int bid = batch_offset + blockIdx.y * gridDim.x + blockIdx.x;
    const int shifted_H_idx = (shift_size != 0) ? ((blockIdx.y - shift_size + gridDim.y) % gridDim.y) : blockIdx.y;
    const int shifted_W_idx = (shift_size != 0) ? ((blockIdx.x - shift_size + gridDim.x) % gridDim.x) : blockIdx.x;
    const int window_H_idx = shifted_H_idx / window_size;
    const int window_W_idx = shifted_W_idx / window_size;
    const int stride_of_window_H = W / window_size;
    const int window_idx = window_H_idx * stride_of_window_H + window_W_idx;
    const int idx_in_window = (shifted_H_idx % window_size) * window_size + (shifted_W_idx % window_size);
    const int output_bid = batch_offset + window_idx * window_size * window_size + idx_in_window;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    float local_out = (tid < n) ? (float)(__ldg(input + bid * n + tid)) : 0.0f;

    mean = blockReduceSum<float>(local_out);
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float diff = (tid < n) ? (local_out - s_mean) : 0.0f;
    variance = blockReduceSum<float>(diff * diff);
    if (threadIdx.x == 0) {
        s_variance = variance / n + 1e-6f;
    }
    __syncthreads();

    if (tid < n) {
        out[output_bid * n + tid] =
            (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg(&gamma[tid])) + (float)(__ldg(&beta[tid])));
    }
}

template<>
__global__ void layernorm_shift_partition(half2* out_ptr,
                                          const half2* input_ptr,
                                          const half2* gamma_ptr,
                                          const half2* beta_ptr,
                                          int batch,
                                          int H,
                                          int W,
                                          int n,
                                          int shift_size,
                                          int window_size)
{
    const int batch_offset = blockIdx.z * gridDim.y * gridDim.x;
    const int bid = batch_offset + blockIdx.y * gridDim.x + blockIdx.x;
    const int shifted_H_idx = (shift_size != 0) ? ((blockIdx.y - shift_size + gridDim.y) % gridDim.y) : blockIdx.y;
    const int shifted_W_idx = (shift_size != 0) ? ((blockIdx.x - shift_size + gridDim.x) % gridDim.x) : blockIdx.x;
    const int window_H_idx = shifted_H_idx / window_size;
    const int window_W_idx = shifted_W_idx / window_size;
    const int stride_of_window_H = W / window_size;
    const int window_idx = window_H_idx * stride_of_window_H + window_W_idx;
    const int idx_in_window = (shifted_H_idx % window_size) * window_size + (shifted_W_idx % window_size);
    const int output_bid = batch_offset + window_idx * window_size * window_size + idx_in_window;
    int tid = threadIdx.x;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;
    float2 local_out_fp2;

    float local_out = 0.0f;
    int id = bid * n + tid;
    if (tid < n) {
        local_out_fp2 = __half22float2(__ldg(input_ptr + id));
        local_out += local_out_fp2.x;
        local_out += local_out_fp2.y;
    }

    mean = blockReduceSum<float>(local_out);
    if (threadIdx.x == 0) {
        s_mean = mean / (n * 2);
    }
    __syncthreads();

    if (tid < n) {
        variance = (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
        variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
    }
    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / (n * 2) + 1e-6f);
    }
    __syncthreads();

    if (tid < n) {
        float2 gamma_val = __half22float2(__ldg(&gamma_ptr[tid]));
        float2 beta_val = __half22float2(__ldg(&beta_ptr[tid]));
        local_out_fp2.x = (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
        local_out_fp2.y = (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
        out_ptr[output_bid * n + tid] = __float22half2_rn(local_out_fp2);
    }
}

template<typename T>
__global__ void layernorm_shift_partition_v2(T* out,
                                             const T* __restrict input,
                                             const T* __restrict gamma,
                                             const T* __restrict beta,
                                             int batch,
                                             int H,
                                             int W,
                                             int n,
                                             int shift_size,
                                             int window_size)
{
    const int ite = 4;
    const int tid = threadIdx.x;
    const int batch_offset = blockIdx.z * gridDim.y * gridDim.x;
    const int bid = batch_offset + blockIdx.y * gridDim.x + blockIdx.x;
    const int shifted_H_idx = (shift_size != 0) ? ((blockIdx.y - shift_size + gridDim.y) % gridDim.y) : blockIdx.y;
    const int shifted_W_idx = (shift_size != 0) ? ((blockIdx.x - shift_size + gridDim.x) % gridDim.x) : blockIdx.x;
    const int window_H_idx = shifted_H_idx / window_size;
    const int window_W_idx = shifted_W_idx / window_size;
    const int stride_of_window_H = W / window_size;
    const int window_idx = window_H_idx * stride_of_window_H + window_W_idx;
    const int idx_in_window = (shifted_H_idx % window_size) * window_size + (shifted_W_idx % window_size);
    const int output_bid = batch_offset + window_idx * window_size * window_size + idx_in_window;
    const int offset = bid * n;
    const int output_offset = output_bid * n;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;
    float local_out[ite];

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        if (col_id < n) {
            local_out[i] = (float)(__ldg(input + offset + col_id));
            sum += local_out[i];
        }
    }

    mean = blockReduceSum<float>(sum);
    if (tid == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float var = 0.0f;
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        if (col_id < n) {
            float diff = local_out[i] - s_mean;
            local_out[i] = diff;
            var += diff * diff;
        }
    }

    variance = blockReduceSum<float>(var);
    if (tid == 0) {
        s_variance = rsqrtf(variance / n + 1e-6f);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        if (col_id < n) {
            out[output_offset + col_id] =
                (T)(local_out[i] * s_variance * (float)__ldg(&gamma[col_id]) + (float)__ldg(&beta[col_id]));
        }
    }
}

template<>
__global__ void layernorm_shift_partition_v2(half2* out_ptr,
                                             const half2* __restrict input_ptr,
                                             const half2* __restrict gamma_ptr,
                                             const half2* __restrict beta_ptr,
                                             int batch,
                                             int H,
                                             int W,
                                             int n,
                                             int shift_size,
                                             int window_size)
{
    const int ite = 4;
    const int tid = threadIdx.x;
    const int batch_offset = blockIdx.z * gridDim.y * gridDim.x;
    const int bid = batch_offset + blockIdx.y * gridDim.x + blockIdx.x;
    const int shifted_H_idx = (shift_size != 0) ? ((blockIdx.y - shift_size + gridDim.y) % gridDim.y) : blockIdx.y;
    const int shifted_W_idx = (shift_size != 0) ? ((blockIdx.x - shift_size + gridDim.x) % gridDim.x) : blockIdx.x;
    const int window_H_idx = shifted_H_idx / window_size;
    const int window_W_idx = shifted_W_idx / window_size;
    const int stride_of_window_H = W / window_size;
    const int window_idx = window_H_idx * stride_of_window_H + window_W_idx;
    const int idx_in_window = (shifted_H_idx % window_size) * window_size + (shifted_W_idx % window_size);
    const int output_bid = batch_offset + window_idx * window_size * window_size + idx_in_window;
    const int offset = bid * n;
    const int output_offset = output_bid * n;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;
    half2 local_out_half2[ite];
    const half2 zero = {static_cast<half>(0.0f), static_cast<half>(0.0f)};

    // float sum = 0.0f;
    half2 sum = __float2half2_rn(0.0f);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        if (col_id < n) {
            local_out_half2[i] = __ldg(input_ptr + offset + col_id);
            sum += local_out_half2[i];
        }
    }

    mean = blockReduceSum<float>((float)(sum.x + sum.y));
    if (threadIdx.x == 0) {
        s_mean = mean / (n * 2);
    }
    __syncthreads();

    float var = 0.0f;
    half2 s_mean_2 = __float2half2_rn(s_mean);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        if (col_id < n) {
            local_out_half2[i] = local_out_half2[i] - s_mean_2;
            float v1 = (float)local_out_half2[i].x;
            float v2 = (float)local_out_half2[i].y;
            var += v1 * v1 + v2 * v2;
        }
    }

    variance = blockReduceSum<float>(var);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / (n * 2) + 1e-6f);
    }
    __syncthreads();

    half2 s_var_2 = __float2half2_rn(s_variance);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        if (col_id < n) {
            out_ptr[output_offset + col_id] =
                local_out_half2[i] * s_var_2 * __ldg(&gamma_ptr[col_id]) + __ldg(&beta_ptr[col_id]);
        }
    }
}

template<typename T>
void invokeLayernormShiftPartition(T* out,
                                   const T* input,
                                   const T* gamma,
                                   const T* beta,
                                   int batch,
                                   int H,
                                   int W,
                                   int n,
                                   int shift_size,
                                   int window_size,
                                   cudaStream_t stream)
{
    dim3 grid(W, H, batch);
    int blockSize = (n + 31) / 32 * 32;
    if (blockSize >= 768) {
        blockSize = ((blockSize / 4) + 31) / 32 * 32;
        layernorm_shift_partition_v2<T>
            <<<grid, blockSize, 0, stream>>>(out, input, gamma, beta, batch, H, W, n, shift_size, window_size);
    }
    else {
        layernorm_shift_partition<T>
            <<<grid, blockSize, 0, stream>>>(out, input, gamma, beta, batch, H, W, n, shift_size, window_size);
    }
}

template<>
void invokeLayernormShiftPartition(half* out,
                                   const half* input,
                                   const half* gamma,
                                   const half* beta,
                                   int batch,
                                   int H,
                                   int W,
                                   int n,
                                   int shift_size,
                                   int window_size,
                                   cudaStream_t stream)
{
    dim3 grid(W, H, batch);
    int blockSize = n / 2;
    blockSize = (blockSize + 31) / 32 * 32;

    if ((batch * H * W >= 512 && blockSize >= 768) || blockSize > 1024) {
        blockSize = ((blockSize / 4) + 31) / 32 * 32;
        layernorm_shift_partition_v2<<<grid, blockSize, 0, stream>>>((half2*)out,
                                                                     (const half2*)input,
                                                                     (const half2*)gamma,
                                                                     (const half2*)beta,
                                                                     batch,
                                                                     H,
                                                                     W,
                                                                     n / 2,
                                                                     shift_size,
                                                                     window_size);
    }
    else {
        layernorm_shift_partition<<<grid, blockSize, 0, stream>>>((half2*)out,
                                                                  (const half2*)input,
                                                                  (const half2*)gamma,
                                                                  (const half2*)beta,
                                                                  batch,
                                                                  H,
                                                                  W,
                                                                  n / 2,
                                                                  shift_size,
                                                                  window_size);
    }
}

template void invokeLayernormShiftPartition<float>(float* out,
                                                   const float* input,
                                                   const float* gamma,
                                                   const float* beta,
                                                   int batch,
                                                   int H,
                                                   int W,
                                                   int n,
                                                   int shift_size,
                                                   int window_size,
                                                   cudaStream_t stream);

template void invokeLayernormShiftPartition<half>(half* out,
                                                  const half* input,
                                                  const half* gamma,
                                                  const half* beta,
                                                  int batch,
                                                  int H,
                                                  int W,
                                                  int n,
                                                  int shift_size,
                                                  int window_size,
                                                  cudaStream_t stream);

/*******************  invokeAddBiasLayernorm  ***********************/

template<typename T>
__global__ void add_bias_layernorm(T* out, const T* bias, const T* gamma, const T* beta, int n)
{
    int tid = threadIdx.x;
    const int bid = blockIdx.x;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    float local_out = (tid < n) ? (float)(out[bid * n + tid] + ldg(&bias[tid])) : 0.0f;

    mean = blockReduceSum<float>(local_out);
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float diff = (tid < n) ? (local_out - s_mean) : 0.0f;
    variance = blockReduceSum<float>(diff * diff);
    if (threadIdx.x == 0) {
        s_variance = variance / n + 1e-6f;
    }
    __syncthreads();

    if (tid < n) {
        out[bid * n + tid] =
            (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(ldg(&gamma[tid])) + (float)(ldg(&beta[tid])));
    }
}

template<typename T>
__global__ void
add_bias_layernorm_v2(T* out, const T* __restrict bias, const T* __restrict gamma, const T* __restrict beta, int n)
{
    const int ite = 4;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int offset = bid * n;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;
    float local_out[ite];

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        local_out[i] = (col_id < n) ? (float)(out[offset + col_id] + ldg(&bias[col_id])) : 0.0f;
        sum += local_out[i];
    }

    mean = blockReduceSum<float>(sum);
    if (tid == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float var = 0.0f;
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        float diff = (col_id < n) ? (local_out[i] - s_mean) : 0.0f;
        var += diff * diff;
    }

    variance = blockReduceSum<float>(var);
    if (tid == 0) {
        s_variance = rsqrtf(variance / n + 1e-6f);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        if (col_id < n) {
            out[offset + col_id] =
                (T)((local_out[i] - s_mean) * s_variance * (float)ldg(&gamma[col_id]) + (float)ldg(&beta[col_id]));
        }
    }
}

#define HALF_LAYERNORM_OPT(UNROLL_FACTOR)                                                                              \
    generalAddBiasResidualLayerNormOpt<T2, false, true, true, true, UNROLL_FACTOR><<<grid, block, 0, stream>>>(        \
        (T2*)out, (T2*)out, (const T2*)bias, (const T2*)out, (const T2*)gamma, (const T2*)beta, m, half_n);

#define HALF_LAYERNORM_OPT2(UNROLL_FACTOR)                                                                             \
    generalAddBiasResidualLayerNormOpt2<T2, false, true, true, true, UNROLL_FACTOR><<<grid, block, 0, stream>>>(       \
        (T2*)out, (T2*)out, (const T2*)bias, (const T2*)out, (const T2*)gamma, (const T2*)beta, m, half_n);

template<typename T>
void invokeAddBiasLayernorm(
    T* out, const T* bias, const T* gamma, const T* beta, int m, int n, cudaStream_t stream, int opt_version)
{
    dim3 grid(m);
    if (n % 2 == 0 && std::is_same<T, half>::value && opt_version > 0) {
        int half_n = n / 2;
        int half_n_32 = (half_n + 31) / 32 * 32;
        dim3 block(min(half_n_32, 512));
        int rolls_per_thread = half_n / block.x;
        int unroll_factor = 8;
        while (unroll_factor > rolls_per_thread && unroll_factor > 1) {
            unroll_factor /= 2;
        }
        using T2 = typename TypeConverter<T>::Type;
        if (opt_version == 1) {
            if (unroll_factor == 1) {
                HALF_LAYERNORM_OPT(1);
            }
            else if (unroll_factor == 2) {
                HALF_LAYERNORM_OPT(2);
            }
            else if (unroll_factor == 3) {
                HALF_LAYERNORM_OPT(3);
            }
            else if (unroll_factor == 4) {
                HALF_LAYERNORM_OPT(4);
            }
            else if (unroll_factor == 8) {
                HALF_LAYERNORM_OPT(8);
            }
        }
        else {
            if (unroll_factor == 1) {
                HALF_LAYERNORM_OPT2(1);
            }
            else if (unroll_factor == 2) {
                HALF_LAYERNORM_OPT2(2);
            }
            else if (unroll_factor == 3) {
                HALF_LAYERNORM_OPT2(3);
            }
            else if (unroll_factor == 4) {
                HALF_LAYERNORM_OPT2(4);
            }
            else if (unroll_factor == 8) {
                HALF_LAYERNORM_OPT2(8);
            }
        }
    }
    else {
        int blockSize = (n + 31) / 32 * 32;
        if (blockSize >= 768) {
            blockSize = ((blockSize / 4) + 31) / 32 * 32;
            add_bias_layernorm_v2<T><<<grid, blockSize, 0, stream>>>(out, bias, gamma, beta, n);
        }
        else {
            add_bias_layernorm<T><<<grid, blockSize, 0, stream>>>(out, bias, gamma, beta, n);
        }
    }
}

#undef HALF_LAYERNORM_OPT
#undef HALF_LAYERNORM_OPT2

template void invokeAddBiasLayernorm<float>(float* out,
                                            const float* bias,
                                            const float* gamma,
                                            const float* beta,
                                            int m,
                                            int n,
                                            cudaStream_t stream,
                                            int opt_version);

template void invokeAddBiasLayernorm<half>(half* out,
                                           const half* bias,
                                           const half* gamma,
                                           const half* beta,
                                           int m,
                                           int n,
                                           cudaStream_t stream,
                                           int opt_version);
#ifdef ENABLE_BF16
template void invokeAddBiasLayernorm<__nv_bfloat16>(__nv_bfloat16* out,
                                                    const __nv_bfloat16* bias,
                                                    const __nv_bfloat16* gamma,
                                                    const __nv_bfloat16* beta,
                                                    int m,
                                                    int n,
                                                    cudaStream_t stream,
                                                    int opt_version);
#endif

/*******************  invokeMergeLayernorm  ***********************/

// input is [batch, 2*H, 2*W, n/4]
// output is [batch, H, W, n]
// grid (W, H, batch)
// block (n)
template<typename T>
__global__ void merge_layernorm_v2(T* out,
                                   const T* __restrict input,
                                   const T* __restrict gamma,
                                   const T* __restrict beta,
                                   int batch,
                                   int H,
                                   int W,
                                   int n)
{
    const int ite = 4;
    const int tid = threadIdx.x;
    const int W_idx = blockIdx.x;
    const int H_idx = blockIdx.y;
    const size_t batch_offset = blockIdx.z * H * W * n;
    const int input_H_stride = W * n / 2;
    const int output_H_stride = W * n;
    const int n_4 = n >> 2;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;
    float local_out[ite];

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        if (col_id < n) {
            int part_id = col_id / n_4;
            int offset_in_W = part_id / 2;
            int offset_in_H = part_id % 2;
            size_t input_id = batch_offset + (2 * H_idx + offset_in_H) * input_H_stride
                              + (2 * W_idx + offset_in_W) * n_4 + (col_id % n_4);
            local_out[i] = (float)(__ldg(input + input_id));
            sum += local_out[i];
        }
    }

    mean = blockReduceSum<float>(sum);
    if (tid == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float var = 0.0f;
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        if (col_id < n) {
            local_out[i] = local_out[i] - s_mean;
            var += local_out[i] * local_out[i];
        }
    }

    variance = blockReduceSum<float>(var);
    if (tid == 0) {
        s_variance = rsqrtf(variance / n + 1e-6f);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        if (col_id < n) {
            size_t output_idx = batch_offset + H_idx * output_H_stride + W_idx * n + col_id;
            out[output_idx] =
                (T)(local_out[i] * s_variance * (float)__ldg(&gamma[col_id]) + (float)__ldg(&beta[col_id]));
        }
    }
}

// TODO : accelerate with half2
template<typename T>
void invokeMergeLayernorm(
    T* output, const T* input, const T* gamma, const T* beta, int batch, int H, int W, int n, cudaStream_t stream)
{
    if ((W % 2 != 0) || (H % 2 != 0)) {
        printf("[ERROR][invokeMergeLayernorm] H(W) should be a multiple of 2.\n");
        return;
    }
    dim3 grid(W / 2, H / 2, batch);
    int blockSize = 4 * n;
    blockSize = (blockSize + 31) / 32 * 32;
    // TODO
    // if (blockSize >= 768)
    {
        blockSize = ((blockSize / 4) + 31) / 32 * 32;
        merge_layernorm_v2<T><<<grid, blockSize, 0, stream>>>(output, input, gamma, beta, batch, H / 2, W / 2, n * 4);
    }
    /*
    else
      merge_layernorm<T><<<grid, blockSize, 0, stream>>>(output, input, gamma, beta, batch, H/2, W/2, n*4);
    */
}

template void invokeMergeLayernorm<float>(float* output,
                                          const float* input,
                                          const float* gamma,
                                          const float* beta,
                                          int batch,
                                          int H,
                                          int W,
                                          int n,
                                          cudaStream_t stream);

template void invokeMergeLayernorm<half>(half* output,
                                         const half* input,
                                         const half* gamma,
                                         const half* beta,
                                         int batch,
                                         int H,
                                         int W,
                                         int n,
                                         cudaStream_t stream);

}  // namespace fastertransformer