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

#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"
#include "src/fastertransformer/utils/cuda_type_utils.cuh"

namespace fastertransformer {

// * Note that typename T is half2 or bfloat2 type
template<typename T, bool IS_OUTPUT, bool IS_BIAS, int RESIDUAL_NUM, bool IS_BETA, int UNROLL_FACTOR>
__global__ void generalAddBiasResidualLayerNormOpt(T* normed_output,
                                                   T* output,
                                                   const T* __restrict input,
                                                   const T* __restrict bias,
                                                   const T* __restrict residual1,
                                                   const T* __restrict residual2,
                                                   const T* __restrict gamma,
                                                   const T* __restrict beta,
                                                   const float  layernorm_eps,
                                                   int          m,
                                                   int          n,
                                                   const float* scale_inter,
                                                   const float* scale_out,
                                                   const float* scale,
                                                   float*       dynamic_scale,
                                                   const int    int8_mode)
{
    extern __shared__ __align__(sizeof(float)) char _shmem[];  // Align on largest type
    T*                                              shmem = reinterpret_cast<T*>(_shmem);

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    using Int8_Packed_T  = typename packed_as<int8_t, num_elems<T>::value>::type;
    using Int32_Packed_T = typename packed_as<int32_t, num_elems<T>::value>::type;
    using Float_Packed_T = typename packed_as<float, num_elems<T>::value>::type;
    using Scalar_T       = typename packed_as<T, 1>::type;

    const bool scale_input     = int8_mode == 2 && scale_inter != nullptr;
    const bool dynamic_scaling = dynamic_scale != nullptr;

    T local_sum = cuda_cast<T>(0.0f);

    const Float_Packed_T scale_from_int = cuda_cast<Float_Packed_T>(scale_input ? (*scale_inter) * (*scale_out) : 0.0f);
    const Float_Packed_T scale_to_int   = cuda_cast<Float_Packed_T>(int8_mode == 2 ? *scale : 0.0f);

#pragma unroll
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = blockIdx.x * n + i;
        T         val   = cuda_cast<T>(0.0f);

        if (IS_BIAS) {
            val = hadd2(val, ldg(&bias[i]));
        }
        if (RESIDUAL_NUM == 1) {
            val = hadd2(val, ldg(&residual1[index]));
        }
        else if (RESIDUAL_NUM == 2) {
            val = hadd2(hadd2(val, ldg(&residual1[index])), ldg(&residual2[index]));
        }

        if (IS_OUTPUT) {
            T in_val;
            if (scale_input) {
                in_val = cuda_cast<T>(cuda_cast<Float_Packed_T>(reinterpret_cast<const Int32_Packed_T*>(input)[index])
                                      * scale_from_int);
            }
            else {
                in_val = input[index];
            }
            val = hadd2(val, in_val);
        }
        shmem[i]      = val;
        output[index] = val;
        local_sum     = hadd2(local_sum, val);
    }

    mean = blockReduceSum((float)(local_sum.x + local_sum.y));

    if (threadIdx.x == 0) {
        s_mean = mean / n / 2;
    }
    __syncthreads();

    float local_var_sum = 0.0f;
#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        T     val    = input[blockIdx.x * n + i];
        float diff_1 = (float)(val.x) - s_mean;
        float diff_2 = (float)(val.y) - s_mean;
        local_var_sum += (diff_1 * diff_1 + diff_2 * diff_2);
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n / 2 + layernorm_eps);
    }
    __syncthreads();

    T mean_2 = cuda_cast<T>(s_mean);
    T var_2  = cuda_cast<T>(s_variance);

    Scalar_T abs_max = 1e-6f;

#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = blockIdx.x * n + i;
        T         val   = hmul2(hsub2(shmem[i], mean_2), var_2, ldg(&gamma[i]));
        if (IS_BETA) {
            val = hadd2(val, ldg(&beta[i]));
        }

        if (dynamic_scaling) {
            abs_max  = cuda_max(cuda_max<Scalar_T>(cuda_abs(val)), abs_max);
            shmem[i] = val;
        }
        else if (int8_mode == 2) {
            reinterpret_cast<Int8_Packed_T*>(normed_output)[index] =
                cuda_cast<Int8_Packed_T>(cuda_cast<Float_Packed_T>(val) * scale_to_int);
        }
        else {
            normed_output[index] = val;
        }
    }

    if (dynamic_scaling) {
        float       abs_max_f               = blockAllReduceMax(cuda_cast<float>(abs_max));
        const float dynamic_per_token_scale = 127. / abs_max_f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const int index                                        = blockIdx.x * n + i;
            reinterpret_cast<Int8_Packed_T*>(normed_output)[index] = cuda_cast<Int8_Packed_T>(
                cuda_cast<Float_Packed_T>(shmem[i]) * cuda_cast<Float_Packed_T>(dynamic_per_token_scale));
        }
        if (threadIdx.x == 0) {
            dynamic_scale[blockIdx.x] = (*scale * abs_max_f) / 127.f;
        }
    }
}

// * Note that typename T is half2 or bfloat2 type
template<typename T, bool IS_OUTPUT, bool IS_BIAS, int RESIDUAL_NUM, bool IS_BETA, int UNROLL_FACTOR>
__global__ void generalAddBiasResidualLayerNormOpt2(T* normed_output,
                                                    T* output,
                                                    const T* __restrict input,
                                                    const T* __restrict bias,
                                                    const T* __restrict residual1,
                                                    const T* __restrict residual2,
                                                    const T* __restrict gamma,
                                                    const T* __restrict beta,
                                                    const float  layernorm_eps,
                                                    int          m,
                                                    int          n,
                                                    const float* scale_inter,
                                                    const float* scale_out,
                                                    const float* scale,
                                                    float*       dynamic_scale,
                                                    const int    int8_mode)
{
    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T*                                              shmem = reinterpret_cast<T*>(_shmem);

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            x_sum    = 0.0f;
    float            x2_sum   = 0.0f;
    const int        b_offset = blockIdx.x * n;

    using T1             = typename TypeConverter<T>::Type;
    using Int8_Packed_T  = typename packed_as<int8_t, num_elems<T>::value>::type;
    using Int32_Packed_T = typename packed_as<int32_t, num_elems<T>::value>::type;
    using Float_Packed_T = typename packed_as<float, num_elems<T>::value>::type;
    using Scalar_T       = typename packed_as<T, 1>::type;

    const bool           scale_input  = int8_mode == 2 && scale_inter != nullptr;
    const Float_Packed_T scale_vec_in = cuda_cast<Float_Packed_T>(scale_input ? (*scale_inter) * (*scale_out) : 0.0f);
    const Float_Packed_T scale_vec    = cuda_cast<Float_Packed_T>(int8_mode == 2 ? *scale : 0.0f);
    const bool           dynamic_scaling = dynamic_scale != nullptr;

#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = b_offset + i;
        float     val_1 = 0.0f;
        float     val_2 = 0.0f;
        T         tmp;

        if (IS_BIAS) {
            tmp = ldg(&bias[i]);
            val_1 += static_cast<float>(tmp.x);
            val_2 += static_cast<float>(tmp.y);
        }
        if (RESIDUAL_NUM == 1) {
            tmp = ldg(&residual1[index]);
            val_1 += static_cast<float>(tmp.x);
            val_2 += static_cast<float>(tmp.y);
        }
        else if (RESIDUAL_NUM == 2) {
            tmp    = ldg(&residual1[index]);
            T tmp2 = ldg(&residual2[index]);
            val_1 += (static_cast<float>(tmp.x) + static_cast<float>(tmp2.x));
            val_2 += (static_cast<float>(tmp.y) + static_cast<float>(tmp2.y));
        }

        if (IS_OUTPUT) {
            if (scale_input) {
                tmp = cuda_cast<T>(cuda_cast<Float_Packed_T>(reinterpret_cast<const Int32_Packed_T*>(input)[index])
                                   * scale_vec_in);
            }
            else {
                tmp = ldg(&input[index]);
            }
            val_1 += static_cast<float>(tmp.x);
            val_2 += static_cast<float>(tmp.y);
        }
        tmp.x         = cuda_cast<T1>(val_1);
        tmp.y         = cuda_cast<T1>(val_2);
        shmem[i]      = tmp;
        output[index] = tmp;
        x_sum += val_1 + val_2;
        x2_sum += val_1 * val_1 + val_2 * val_2;
    }
    float sums[2];
    sums[0] = x_sum;
    sums[1] = x2_sum;
    blockReduceSumV2<float, 2>(sums);

    if (threadIdx.x == 0) {
        s_mean     = sums[0] / n / 2;
        s_variance = rsqrtf(sums[1] / n / 2 - s_mean * s_mean + layernorm_eps);
    }
    __syncthreads();

    T mean_2 = cuda_cast<T>(s_mean);
    T var_2  = cuda_cast<T>(s_variance);

    Scalar_T abs_max = 1e-6f;

#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = blockIdx.x * n + i;
        T         val   = hmul2(hsub2(shmem[i], mean_2), var_2, ldg(&gamma[i]));
        if (IS_BETA) {
            val = hadd2(val, ldg(&beta[i]));
        }

        if (dynamic_scaling) {
            abs_max  = cuda_max(cuda_max<Scalar_T>(cuda_abs(val)), abs_max);
            shmem[i] = val;
        }
        else if (int8_mode == 2) {
            reinterpret_cast<Int8_Packed_T*>(normed_output)[index] =
                cuda_cast<Int8_Packed_T>(cuda_cast<Float_Packed_T>(val) * scale_vec);
        }
        else {
            normed_output[index] = val;
        }
    }

    if (dynamic_scaling) {
        float       abs_max_f               = blockAllReduceMax(cuda_cast<float>(abs_max));
        const float dynamic_per_token_scale = 127. / abs_max_f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const int index                                        = blockIdx.x * n + i;
            reinterpret_cast<Int8_Packed_T*>(normed_output)[index] = cuda_cast<Int8_Packed_T>(
                cuda_cast<Float_Packed_T>(shmem[i]) * cuda_cast<Float_Packed_T>(dynamic_per_token_scale));
        }
        if (threadIdx.x == 0) {
            dynamic_scale[blockIdx.x] = (*scale * abs_max_f) / 127.f;
        }
    }
}

// TODO(bhsueh) add half2 implementation
template<typename T, int N>
__global__ void addBiasResidualPostLayerNorm(
    T* out, const T* input, const T* bias, const T* gamma, const T* beta, const float layernorm_eps, int m, int n)
{
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;
    float            local_out_cache[N];

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
        float local_out = (float)(add(out[blockIdx.x * n + idx], input[blockIdx.x * n + idx], ldg(&bias[idx])));
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
        s_variance = variance / n + layernorm_eps;
    }
    __syncthreads();

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
        float local_out = local_out_cache[i];
        out[blockIdx.x * n + idx] =
            (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(ldg(&gamma[idx])) + (float)(ldg(&beta[idx])));
        idx += blockDim.x;
    }
}

template<int N>
__global__ void addBiasResidualPostLayerNormHalf(half*       out,
                                                 const half* input,
                                                 const half* bias,
                                                 const half* gamma,
                                                 const half* beta,
                                                 const float layernorm_eps,
                                                 int         m,
                                                 int         n)
{
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    half2*       out_ptr   = (half2*)out;
    const half2* input_ptr = (const half2*)input;
    const half2* bias_ptr  = (const half2*)bias;
    const half2* gamma_ptr = (const half2*)gamma;
    const half2* beta_ptr  = (const half2*)beta;

    float2 out_fp2_cache[N];

    float local_out = 0.0f;
#pragma unroll N
    for (int idx = threadIdx.x, i = 0; idx < n / 2 && i < N; ++i) {
        int    id            = blockIdx.x * n / 2 + idx;
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
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; i < N && idx < n / 2; ++i) {
        int    id            = blockIdx.x * n / 2 + idx;
        float2 local_out_fp2 = out_fp2_cache[i];
        float2 gamma_val     = __half22float2(__ldg(&gamma_ptr[idx]));
        float2 beta_val      = __half22float2(__ldg(&beta_ptr[idx]));
        local_out_fp2.x      = (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
        local_out_fp2.y      = (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
        out_ptr[id]          = __float22half2_rn(local_out_fp2);
        idx += blockDim.x;
    }
}

// Optimization for fp16 and fp16 (bf162 and half2)
template<typename T>
__global__ void generalAddBiasResidualPostLayerNorm(
    T* out, const T* input, const T* bias, const T* gamma, const T* beta, const float layernorm_eps, int m, int n)
{
    using T2 = typename TypeConverter<T>::Type;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    T2*       out_ptr   = (T2*)out;
    const T2* input_ptr = (const T2*)input;
    const T2* bias_ptr  = (const T2*)bias;
    const T2* gamma_ptr = (const T2*)gamma;
    const T2* beta_ptr  = (const T2*)beta;

    float local_out = 0.0f;
    for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
        int    id            = blockIdx.x * n / 2 + idx;
        T2     tmp           = hadd2(hadd2(out_ptr[id], input_ptr[id]), ldg(&bias_ptr[idx]));
        float2 local_out_fp2 = cuda_cast<float2>(tmp);
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
        int    id            = blockIdx.x * n / 2 + idx;
        float2 local_out_fp2 = cuda_cast<float2>(out_ptr[id]);
        variance += (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
        variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
    }

    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
        int    id            = blockIdx.x * n / 2 + idx;
        float2 local_out_fp2 = cuda_cast<float2>(out_ptr[id]);
        float2 gamma_val     = cuda_cast<float2>(ldg(&gamma_ptr[idx]));
        float2 beta_val      = cuda_cast<float2>(ldg(&beta_ptr[idx]));
        local_out_fp2.x      = (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
        local_out_fp2.y      = (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
        out_ptr[id]          = cuda_cast<T2>(local_out_fp2);
    }
}

template<>
__global__ void generalAddBiasResidualPostLayerNorm(float*       out,
                                                    const float* input,
                                                    const float* bias,
                                                    const float* gamma,
                                                    const float* beta,
                                                    const float  layernorm_eps,
                                                    int          m,
                                                    int          n)
{
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

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
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float local_out = out[blockIdx.x * n + idx];
        out[blockIdx.x * n + idx] =
            (float)(((local_out - s_mean) * s_variance) * (float)(__ldg(&gamma[idx])) + (float)(__ldg(&beta[idx])));
    }
}

// applied to half and b16
template<typename T>
__global__ void addBiasResidualPostLayerNormV2(T* out,
                                               const T* __restrict input,
                                               const T* __restrict bias,
                                               const T* __restrict gamma,
                                               const T* __restrict beta,
                                               const float layernorm_eps,
                                               int         n)
{
    using T2             = typename TypeConverter<T>::Type;
    const int        ite = 4;
    const int        tid = threadIdx.x;
    const int        bid = blockIdx.x;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;
    T2               local_out_half2[ite];

    T2*       out_ptr   = (T2*)out;
    const T2* input_ptr = (const T2*)input;
    const T2* bias_ptr  = (const T2*)bias;
    const T2* gamma_ptr = (const T2*)gamma;
    const T2* beta_ptr  = (const T2*)beta;

    // float sum = 0.0f;
    T2 sum = cuda_cast<T2>(0.0f);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id         = i * blockDim.x + tid;
        int id             = bid * n / 2 + col_id;
        local_out_half2[i] = add(out_ptr[id], ldg(&input_ptr[id]), ldg(&bias_ptr[col_id]));
        sum                = add(sum, local_out_half2[i]);
    }

    mean = blockReduceSum<float>((float)(sum.x + sum.y));
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float var      = 0.0f;
    T2    s_mean_2 = cuda_cast<T2>(s_mean);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        local_out_half2[i] = hsub2(local_out_half2[i], s_mean_2);
        float v1           = (float)local_out_half2[i].x;
        float v2           = (float)local_out_half2[i].y;
        var += v1 * v1 + v2 * v2;
    }

    variance = blockReduceSum<float>(var);
    if (tid == 0) {
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

    T2 s_var_2 = cuda_cast<T2>(s_variance);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id  = i * blockDim.x + tid;
        int id      = bid * n / 2 + col_id;
        out_ptr[id] = fma(local_out_half2[i], s_var_2, ldg(&gamma_ptr[col_id]), ldg(&beta_ptr[col_id]));
    }
}

template<>
__global__ void addBiasResidualPostLayerNormV2(float* out,
                                               const float* __restrict input,
                                               const float* __restrict bias,
                                               const float* __restrict gamma,
                                               const float* __restrict beta,
                                               const float layernorm_eps,
                                               int         n)
{
    const int ite = 4;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;
    float            local_out[ite];

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id   = i * blockDim.x + tid;
        int id       = bid * n + col_id;
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
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        int id     = bid * n + col_id;
        out[id] =
            (float)((local_out[i] - s_mean) * s_variance * (float)__ldg(&gamma[col_id]) + (float)__ldg(&beta[col_id]));
    }
}

// bf16 and half data type
template<typename T>
void invokeAddBiasResidualLayerNorm(T*           out,
                                    const T*     input,
                                    const T*     bias,
                                    const T*     gamma,
                                    const T*     beta,
                                    const float  layernorm_eps,
                                    int          m,
                                    int          n,
                                    cudaStream_t stream)
{
    dim3 grid(m);
    dim3 block(std::min(n, 1024));

    if (m >= 512 && (n == 768 || n == 1024)) {
        addBiasResidualPostLayerNormV2<T><<<grid, n / 8, 0, stream>>>(out, input, bias, gamma, beta, layernorm_eps, n);
    }
    else {
        block.x       = std::min(n, 1024);
        int num_trips = (n + block.x - 1) / block.x;
        if (num_trips == 1) {
            addBiasResidualPostLayerNorm<T, 1>
                <<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, layernorm_eps, m, n);
        }
        else if (num_trips == 2) {
            addBiasResidualPostLayerNorm<T, 2>
                <<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, layernorm_eps, m, n);
        }
        else {
            generalAddBiasResidualPostLayerNorm<T>
                <<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, layernorm_eps, m, n);
        }
    }
}

template<>
void invokeAddBiasResidualLayerNorm(float*       out,
                                    const float* input,
                                    const float* bias,
                                    const float* gamma,
                                    const float* beta,
                                    const float  layernorm_eps,
                                    int          m,
                                    int          n,
                                    cudaStream_t stream)
{
    dim3 grid(m);
    dim3 block(std::min(n, 1024));
    if (n == 768 || n == 1024) {
        addBiasResidualPostLayerNormV2<float>
            <<<grid, n / 4, 0, stream>>>(out, input, bias, gamma, beta, layernorm_eps, n);
    }
    else {
        block.x       = std::min(n, 1024);
        int num_trips = (n + block.x - 1) / block.x;
        if (num_trips == 1) {
            addBiasResidualPostLayerNorm<float, 1>
                <<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, layernorm_eps, m, n);
        }
        else if (num_trips == 2) {
            addBiasResidualPostLayerNorm<float, 2>
                <<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, layernorm_eps, m, n);
        }
        else {
            generalAddBiasResidualPostLayerNorm<float>
                <<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, layernorm_eps, m, n);
        }
    }
}

template void invokeAddBiasResidualLayerNorm(float*       out,
                                             const float* input,
                                             const float* bias,
                                             const float* gamma,
                                             const float* beta,
                                             const float  layernorm_eps,
                                             int          m,
                                             int          n,
                                             cudaStream_t stream);
template void invokeAddBiasResidualLayerNorm(half*        out,
                                             const half*  input,
                                             const half*  bias,
                                             const half*  gamma,
                                             const half*  beta,
                                             const float  layernorm_eps,
                                             int          m,
                                             int          n,
                                             cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeAddBiasResidualLayerNorm(__nv_bfloat16*       out,
                                             const __nv_bfloat16* input,
                                             const __nv_bfloat16* bias,
                                             const __nv_bfloat16* gamma,
                                             const __nv_bfloat16* beta,
                                             const float          layernorm_eps,
                                             int                  m,
                                             int                  n,
                                             cudaStream_t         stream);
#endif

template<typename T, int RESIDUAL_NUM>
__global__ void generalAddBiasResidualLayerNorm(const T* __restrict input,
                                                const T* __restrict residual1,
                                                const T* __restrict residual2,
                                                const T* __restrict gamma,
                                                const T* __restrict beta,
                                                const T* __restrict bias,
                                                T*           output,
                                                T*           norm_output,
                                                const float  layernorm_eps,
                                                int          m,
                                                int          n,
                                                const float* scale_inter,
                                                const float* scale_out,
                                                const float* scale,
                                                float*       dynamic_scale,
                                                const int    int8_mode)
{
    int tid = threadIdx.x;

    // NOTE: float shmem may exceed the shared memory limit
    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T*                                              shmem = reinterpret_cast<T*>(_shmem);

    using Int8_Packed_T  = typename packed_as<int8_t, num_elems<T>::value>::type;
    using Int32_Packed_T = typename packed_as<int32_t, num_elems<T>::value>::type;
    using Float_Packed_T = typename packed_as<float, num_elems<T>::value>::type;
    using Scalar_T       = typename packed_as<T, 1>::type;

    const bool dynamic_scaling = dynamic_scale != nullptr;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    const bool  is_input_i32  = int8_mode == 2 && scale_inter != nullptr && scale_out != nullptr;
    const float scale_out_val = is_input_i32 ? (*scale_inter) * (*scale_out) : 0.0f;
    float       local_sum     = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float local_out = 0.0f;
        if (RESIDUAL_NUM == 1) {
            local_out = (float)(ldg(&residual1[blockIdx.x * n + i]));
        }
        else if (RESIDUAL_NUM == 2) {
            local_out = (float)(ldg(&residual1[blockIdx.x * n + i])) + float(ldg(&residual2[blockIdx.x * n + i]));
        }
        if (is_input_i32) {
            local_out += cuda_cast<float>(reinterpret_cast<const int32_t*>(input)[blockIdx.x * n + i]) * scale_out_val;
        }
        else {
            local_out += (float)(input[blockIdx.x * n + i]);
        }

        if (bias != nullptr) {
            local_out += (float)(ldg(&bias[i]));
        }
        shmem[i]                   = (T)local_out;
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
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

    Scalar_T abs_max = 1e-6f;

    const float scale_val = int8_mode == 2 ? *scale : 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float       beta_val = (beta == nullptr) ? 0.0f : (float)(ldg(&beta[i]));
        const float val      = ((((float)shmem[i] - s_mean) * s_variance) * (float)(ldg(&gamma[i])) + beta_val);

        if (dynamic_scaling) {
            abs_max  = cuda_max(cuda_max<Scalar_T, float>(cuda_abs(val)), abs_max);
            shmem[i] = (T)val;
        }
        else if (int8_mode == 2) {
            reinterpret_cast<int8_t*>(norm_output)[blockIdx.x * n + i] = cuda_cast<int8_t>(val * scale_val);
        }
        else {
            norm_output[blockIdx.x * n + i] = (T)val;
        }
    }

    if (dynamic_scaling) {
        float       abs_max_f               = blockAllReduceMax(cuda_cast<float>(abs_max));
        const float dynamic_per_token_scale = 127. / abs_max_f;
        for (int i = tid; i < n; i += blockDim.x) {
            const int index                                      = blockIdx.x * n + i;
            reinterpret_cast<Int8_Packed_T*>(norm_output)[index] = cuda_cast<Int8_Packed_T>(
                cuda_cast<Float_Packed_T>(shmem[i]) * cuda_cast<Float_Packed_T>(dynamic_per_token_scale));
        }
        if (threadIdx.x == 0) {
            dynamic_scale[blockIdx.x] = (*scale * abs_max_f) / 127.f;
        }
    }
}

template<typename T, bool IS_OUTPUT, bool IS_BIAS, int UNROLL_FACTOR, int RESIDUAL_NUM>
void dispatch_generalAddBiasResidualLayerNormOpt_opt_version(T*           norm_output,
                                                             T*           output,
                                                             const T*     input,
                                                             const T*     bias,
                                                             const T*     residual1,
                                                             const T*     residual2,
                                                             const T*     gamma,
                                                             const T*     beta,
                                                             float        layernorm_eps,
                                                             int          m,
                                                             int          half_n,
                                                             const float* scale_inter,
                                                             const float* scale_out,
                                                             const float* scale,
                                                             float*       dynamic_scale,
                                                             int          int8_mode,
                                                             dim3         grid,
                                                             dim3         block,
                                                             cudaStream_t stream,
                                                             int          opt_version)
{
    size_t maxbytes = half_n * sizeof(T);
    if (opt_version == 1) {
        if (maxbytes >= (48 << 10)) {
            check_cuda_error(cudaFuncSetAttribute(
                generalAddBiasResidualLayerNormOpt<T, IS_OUTPUT, IS_BIAS, RESIDUAL_NUM, true, UNROLL_FACTOR>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                maxbytes));
        }
        generalAddBiasResidualLayerNormOpt<T, IS_OUTPUT, IS_BIAS, RESIDUAL_NUM, true, UNROLL_FACTOR>
            <<<grid, block, maxbytes, stream>>>(norm_output,
                                                output,
                                                input,
                                                bias,
                                                residual1,
                                                residual2,
                                                gamma,
                                                beta,
                                                layernorm_eps,
                                                m,
                                                half_n,
                                                scale_inter,
                                                scale_out,
                                                scale,
                                                dynamic_scale,
                                                int8_mode);
    }
    else if (opt_version == 2) {
        if (maxbytes >= (48 << 10)) {
            check_cuda_error(cudaFuncSetAttribute(
                generalAddBiasResidualLayerNormOpt2<T, IS_OUTPUT, IS_BIAS, RESIDUAL_NUM, true, UNROLL_FACTOR>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                maxbytes));
        }
        generalAddBiasResidualLayerNormOpt2<T, IS_OUTPUT, IS_BIAS, RESIDUAL_NUM, true, UNROLL_FACTOR>
            <<<grid, block, maxbytes, stream>>>(norm_output,
                                                output,
                                                input,
                                                bias,
                                                residual1,
                                                residual2,
                                                gamma,
                                                beta,
                                                layernorm_eps,
                                                m,
                                                half_n,
                                                scale_inter,
                                                scale_out,
                                                scale,
                                                dynamic_scale,
                                                int8_mode);
    }
    else {
        FT_CHECK_WITH_INFO(false, "opt_num must be 1 or 2");
    }
}

template<typename T, bool IS_BIAS, int UNROLL_FACTOR, int RESIDUAL_NUM>
void dispatch_generalAddBiasResidualLayerNormOpt_is_output(T*           norm_output,
                                                           T*           output,
                                                           const T*     input,
                                                           const T*     bias,
                                                           const T*     residual1,
                                                           const T*     residual2,
                                                           const T*     gamma,
                                                           const T*     beta,
                                                           float        layernorm_eps,
                                                           int          m,
                                                           int          half_n,
                                                           const float* scale_inter,
                                                           const float* scale_out,
                                                           const float* scale,
                                                           float*       dynamic_scale,
                                                           int          int8_mode,
                                                           dim3         grid,
                                                           dim3         block,
                                                           cudaStream_t stream,
                                                           int          opt_version,
                                                           bool         is_output)
{
    if (is_output) {
        dispatch_generalAddBiasResidualLayerNormOpt_opt_version<T, true, IS_BIAS, UNROLL_FACTOR, RESIDUAL_NUM>(
            norm_output,
            output,
            input,
            bias,
            residual1,
            residual2,
            gamma,
            beta,
            layernorm_eps,
            m,
            half_n,
            scale_inter,
            scale_out,
            scale,
            dynamic_scale,
            int8_mode,
            grid,
            block,
            stream,
            opt_version);
    }
    else {
        dispatch_generalAddBiasResidualLayerNormOpt_opt_version<T, false, IS_BIAS, UNROLL_FACTOR, RESIDUAL_NUM>(
            norm_output,
            output,
            input,
            bias,
            residual1,
            residual2,
            gamma,
            beta,
            layernorm_eps,
            m,
            half_n,
            scale_inter,
            scale_out,
            scale,
            dynamic_scale,
            int8_mode,
            grid,
            block,
            stream,
            opt_version);
    }
}

template<typename T, int UNROLL_FACTOR, int RESIDUAL_NUM>
void dispatch_generalAddBiasResidualLayerNormOpt_bias(T*           norm_output,
                                                      T*           output,
                                                      const T*     input,
                                                      const T*     bias,
                                                      const T*     residual1,
                                                      const T*     residual2,
                                                      const T*     gamma,
                                                      const T*     beta,
                                                      float        layernorm_eps,
                                                      int          m,
                                                      int          half_n,
                                                      const float* scale_inter,
                                                      const float* scale_out,
                                                      const float* scale,
                                                      float*       dynamic_scale,
                                                      int          int8_mode,
                                                      dim3         grid,
                                                      dim3         block,
                                                      cudaStream_t stream,
                                                      int          opt_version,
                                                      bool         is_output)
{
    if (bias != nullptr) {
        dispatch_generalAddBiasResidualLayerNormOpt_is_output<T, true, UNROLL_FACTOR, RESIDUAL_NUM>(norm_output,
                                                                                                    output,
                                                                                                    input,
                                                                                                    bias,
                                                                                                    residual1,
                                                                                                    residual2,
                                                                                                    gamma,
                                                                                                    beta,
                                                                                                    layernorm_eps,
                                                                                                    m,
                                                                                                    half_n,
                                                                                                    scale_inter,
                                                                                                    scale_out,
                                                                                                    scale,
                                                                                                    dynamic_scale,
                                                                                                    int8_mode,
                                                                                                    grid,
                                                                                                    block,
                                                                                                    stream,
                                                                                                    opt_version,
                                                                                                    is_output);
    }
    else {
        dispatch_generalAddBiasResidualLayerNormOpt_is_output<T, false, UNROLL_FACTOR, RESIDUAL_NUM>(norm_output,
                                                                                                     output,
                                                                                                     input,
                                                                                                     bias,
                                                                                                     residual1,
                                                                                                     residual2,
                                                                                                     gamma,
                                                                                                     beta,
                                                                                                     layernorm_eps,
                                                                                                     m,
                                                                                                     half_n,
                                                                                                     scale_inter,
                                                                                                     scale_out,
                                                                                                     scale,
                                                                                                     dynamic_scale,
                                                                                                     int8_mode,
                                                                                                     grid,
                                                                                                     block,
                                                                                                     stream,
                                                                                                     opt_version,
                                                                                                     is_output);
    }
}

template<typename T, int UNROLL_FACTOR>
void dispatch_generalAddBiasResidualLayerNormOpt_residual_num(T*           norm_output,
                                                              T*           output,
                                                              const T*     input,
                                                              const T*     bias,
                                                              const T*     residual1,
                                                              const T*     residual2,
                                                              const T*     gamma,
                                                              const T*     beta,
                                                              float        layernorm_eps,
                                                              int          m,
                                                              int          half_n,
                                                              const float* scale_inter,
                                                              const float* scale_out,
                                                              const float* scale,
                                                              float*       dynamic_scale,
                                                              int          int8_mode,
                                                              dim3         grid,
                                                              dim3         block,
                                                              cudaStream_t stream,
                                                              int          opt_version,
                                                              bool         is_output,
                                                              int          residual_num)
{
    if (residual_num == 1) {
        dispatch_generalAddBiasResidualLayerNormOpt_bias<T, UNROLL_FACTOR, 1>(norm_output,
                                                                              output,
                                                                              input,
                                                                              bias,
                                                                              residual1,
                                                                              residual2,
                                                                              gamma,
                                                                              beta,
                                                                              layernorm_eps,
                                                                              m,
                                                                              half_n,
                                                                              scale_inter,
                                                                              scale_out,
                                                                              scale,
                                                                              dynamic_scale,
                                                                              int8_mode,
                                                                              grid,
                                                                              block,
                                                                              stream,
                                                                              opt_version,
                                                                              is_output);
    }
    else if (residual_num == 2) {
        dispatch_generalAddBiasResidualLayerNormOpt_bias<T, UNROLL_FACTOR, 2>(norm_output,
                                                                              output,
                                                                              input,
                                                                              bias,
                                                                              residual1,
                                                                              residual2,
                                                                              gamma,
                                                                              beta,
                                                                              layernorm_eps,
                                                                              m,
                                                                              half_n,
                                                                              scale_inter,
                                                                              scale_out,
                                                                              scale,
                                                                              dynamic_scale,
                                                                              int8_mode,
                                                                              grid,
                                                                              block,
                                                                              stream,
                                                                              opt_version,
                                                                              is_output);
    }
    else {
        FT_CHECK_WITH_INFO(false, "residual_num must be 1 or 2");
    }
}

template<typename T>
void dispatch_generalAddBiasResidualLayerNormOpt_unroll_factor(T*           norm_output,
                                                               T*           output,
                                                               const T*     input,
                                                               const T*     bias,
                                                               const T*     residual1,
                                                               const T*     residual2,
                                                               const T*     gamma,
                                                               const T*     beta,
                                                               float        layernorm_eps,
                                                               int          m,
                                                               int          half_n,
                                                               const float* scale_inter,
                                                               const float* scale_out,
                                                               const float* scale,
                                                               float*       dynamic_scale,
                                                               int          int8_mode,
                                                               dim3         grid,
                                                               dim3         block,
                                                               cudaStream_t stream,
                                                               int          opt_version,
                                                               bool         is_output,
                                                               int          residual_num,
                                                               int          unroll_factor)
{
    switch (unroll_factor) {
        case 1:
            dispatch_generalAddBiasResidualLayerNormOpt_residual_num<T, 1>(norm_output,
                                                                           output,
                                                                           input,
                                                                           bias,
                                                                           residual1,
                                                                           residual2,
                                                                           gamma,
                                                                           beta,
                                                                           layernorm_eps,
                                                                           m,
                                                                           half_n,
                                                                           scale_inter,
                                                                           scale_out,
                                                                           scale,
                                                                           dynamic_scale,
                                                                           int8_mode,
                                                                           grid,
                                                                           block,
                                                                           stream,
                                                                           opt_version,
                                                                           is_output,
                                                                           residual_num);
            break;
        case 2:
            dispatch_generalAddBiasResidualLayerNormOpt_residual_num<T, 2>(norm_output,
                                                                           output,
                                                                           input,
                                                                           bias,
                                                                           residual1,
                                                                           residual2,
                                                                           gamma,
                                                                           beta,
                                                                           layernorm_eps,
                                                                           m,
                                                                           half_n,
                                                                           scale_inter,
                                                                           scale_out,
                                                                           scale,
                                                                           dynamic_scale,
                                                                           int8_mode,
                                                                           grid,
                                                                           block,
                                                                           stream,
                                                                           opt_version,
                                                                           is_output,
                                                                           residual_num);
            break;
        case 4:
            dispatch_generalAddBiasResidualLayerNormOpt_residual_num<T, 4>(norm_output,
                                                                           output,
                                                                           input,
                                                                           bias,
                                                                           residual1,
                                                                           residual2,
                                                                           gamma,
                                                                           beta,
                                                                           layernorm_eps,
                                                                           m,
                                                                           half_n,
                                                                           scale_inter,
                                                                           scale_out,
                                                                           scale,
                                                                           dynamic_scale,
                                                                           int8_mode,
                                                                           grid,
                                                                           block,
                                                                           stream,
                                                                           opt_version,
                                                                           is_output,
                                                                           residual_num);
            break;
        case 8:
            dispatch_generalAddBiasResidualLayerNormOpt_residual_num<T, 8>(norm_output,
                                                                           output,
                                                                           input,
                                                                           bias,
                                                                           residual1,
                                                                           residual2,
                                                                           gamma,
                                                                           beta,
                                                                           layernorm_eps,
                                                                           m,
                                                                           half_n,
                                                                           scale_inter,
                                                                           scale_out,
                                                                           scale,
                                                                           dynamic_scale,
                                                                           int8_mode,
                                                                           grid,
                                                                           block,
                                                                           stream,
                                                                           opt_version,
                                                                           is_output,
                                                                           residual_num);
            break;
        default:
            FT_CHECK_WITH_INFO(false, "unroll_factor must be 1, 2, 4 or 8");
    }
}

/* output      <- output + bias + residual_1 + residual_2
 * output_norm <- LN(output) */
template<typename T>
void invokeGeneralAddBiasResidualPreLayerNorm(T*           output,
                                              T*           norm_output,
                                              const T*     input,
                                              const T*     residual1,
                                              const T*     residual2,
                                              const T*     gamma,
                                              const T*     beta,
                                              const T*     bias,
                                              const float  layernorm_eps,
                                              int          m,
                                              int          n,
                                              const float* scale_inter,
                                              const float* scale_out,
                                              float*       scale,
                                              float*       dynamic_scale,
                                              const int    int8_mode,
                                              cudaStream_t stream,
                                              int          opt_version)
{
    const int  residual_num  = residual2 == nullptr ? 1 : 2;
    const bool dynamic_quant = dynamic_scale != nullptr;

    if (opt_version > 0 && sizeof(T) == 2 && n % 2 == 0) {
        dim3 grid(m);
        int  half_n    = n / 2;
        int  half_n_32 = (half_n + 31) / 32 * 32;
        dim3 block(min(half_n_32, 512));
        int  rolls_per_thread = half_n / block.x;
        int  unroll_factor    = 8;
        while (unroll_factor > rolls_per_thread && unroll_factor > 1) {
            unroll_factor /= 2;
        }

        using T2 = typename TypeConverter<T>::Type;

        /* we launch (and instantiate) the kernel by specializing for unroll_factor -> residual_num -> is_bias ->
         * opt_version */
        dispatch_generalAddBiasResidualLayerNormOpt_unroll_factor((T2*)norm_output,
                                                                  (T2*)output,
                                                                  (const T2*)input,
                                                                  (const T2*)bias,
                                                                  (const T2*)residual1,
                                                                  (const T2*)residual2,
                                                                  (const T2*)gamma,
                                                                  (const T2*)beta,
                                                                  layernorm_eps,
                                                                  m,
                                                                  half_n,
                                                                  scale_inter,
                                                                  scale_out,
                                                                  scale,
                                                                  dynamic_scale,
                                                                  int8_mode,
                                                                  grid,
                                                                  block,
                                                                  stream,
                                                                  opt_version,
                                                                  true,  // is_output
                                                                  residual_num,
                                                                  unroll_factor);
    }
    else {

        dim3 grid(m);
        dim3 block(min(n, 1024));

        /* For general cases, n is equal to hidden_units, e.g., 512/1024.
        Since we have warp shuffle inside the code, block.x % 32 should be 0.
        */
        block.x = (block.x + 31) / 32 * 32;

        size_t maxbytes = n * sizeof(T);
        if (residual_num == 1) {
            if (maxbytes >= (48 << 10)) {
                check_cuda_error(cudaFuncSetAttribute(
                    generalAddBiasResidualLayerNorm<T, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
            }
            generalAddBiasResidualLayerNorm<T, 1><<<grid, block, maxbytes, stream>>>(input,
                                                                                     residual1,
                                                                                     residual2,
                                                                                     gamma,
                                                                                     beta,
                                                                                     bias,
                                                                                     output,
                                                                                     norm_output,
                                                                                     layernorm_eps,
                                                                                     m,
                                                                                     n,
                                                                                     scale_inter,
                                                                                     scale_out,
                                                                                     scale,
                                                                                     dynamic_scale,
                                                                                     int8_mode);
        }
        else if (residual_num == 2) {
            if (maxbytes >= (48 << 10)) {
                check_cuda_error(cudaFuncSetAttribute(
                    generalAddBiasResidualLayerNorm<T, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
            }
            generalAddBiasResidualLayerNorm<T, 2><<<grid, block, maxbytes, stream>>>(input,
                                                                                     residual1,
                                                                                     residual2,
                                                                                     gamma,
                                                                                     beta,
                                                                                     bias,
                                                                                     output,
                                                                                     norm_output,
                                                                                     layernorm_eps,
                                                                                     m,
                                                                                     n,
                                                                                     scale_inter,
                                                                                     scale_out,
                                                                                     scale,
                                                                                     dynamic_scale,
                                                                                     int8_mode);
        }
    }
}

#define INSTANTIATE_INVOKE_GENERAL_ADD_BIAS_RESIDUAL_PRE_LAYER_NORM(T)                                                 \
    template void invokeGeneralAddBiasResidualPreLayerNorm(T*           output,                                        \
                                                           T*           norm_output,                                   \
                                                           const T*     input,                                         \
                                                           const T*     residual1,                                     \
                                                           const T*     residual2,                                     \
                                                           const T*     gamma,                                         \
                                                           const T*     beta,                                          \
                                                           const T*     bias,                                          \
                                                           const float  layernorm_eps,                                 \
                                                           int          m,                                             \
                                                           int          n,                                             \
                                                           const float* scale_inter,                                   \
                                                           const float* scale_out,                                     \
                                                           float*       scale,                                         \
                                                           float*       dynamic_scale,                                 \
                                                           const int    int8_mode,                                     \
                                                           cudaStream_t stream,                                        \
                                                           int          opt_version)
INSTANTIATE_INVOKE_GENERAL_ADD_BIAS_RESIDUAL_PRE_LAYER_NORM(float);
INSTANTIATE_INVOKE_GENERAL_ADD_BIAS_RESIDUAL_PRE_LAYER_NORM(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_GENERAL_ADD_BIAS_RESIDUAL_PRE_LAYER_NORM(__nv_bfloat16);
#endif
#undef INSTANTIATE_INVOKE_GENERAL_ADD_BIAS_RESIDUAL_PRE_LAYER_NORM

template<typename T>
__global__ void generalAddResidualT5LayerNorm(const T* __restrict input,
                                              const T* __restrict gamma,
                                              T*          output,
                                              T*          norm_output,
                                              const float layernorm_eps,
                                              int         m,
                                              int         n)
{
    // layernorm module in the T5 style No bias and no subtraction of mean.
    __shared__ float s_variance;
    float            variance = 0.0f;

    float local_var_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        output[blockIdx.x * n + i] =
            clamp_inf_for_half<T>((float)ldg(&input[blockIdx.x * n + i]) + (float)output[blockIdx.x * n + i]);

        float diff = (float)(output[blockIdx.x * n + i]);
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / (float)n + layernorm_eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        norm_output[blockIdx.x * n + i] =
            clamp_inf_for_half<T>((((float)output[blockIdx.x * n + i]) * s_variance) * (float)(ldg(&gamma[i])));
    }
}

template<typename T>
void invokeGeneralAddResidualT5PreLayerNorm(T*           output,
                                            T*           norm_output,
                                            const T*     input,
                                            const T*     gamma,
                                            const float  layernorm_eps,
                                            int          m,
                                            int          n,
                                            cudaStream_t stream)
{
    dim3 grid(m);
    dim3 block(min(n, 1024));

    /* For general cases, n is equal to hidden_units, e.g., 512/1024.
    Since we have warp shuffle inside the code, block.x % 32 should be 0.
    */

    if (n % 32 != 0) {
        block.x = 1024;
    }

    // TODO(bhsueh) add 16bitx2 implementation
    /* should pay attention to the rsqrt precision*/
    generalAddResidualT5LayerNorm<T>
        <<<grid, block, 0, stream>>>(input, gamma, output, norm_output, layernorm_eps, m, n);
}

template void invokeGeneralAddResidualT5PreLayerNorm(float*       output,
                                                     float*       norm_output,
                                                     const float* input,
                                                     const float* gamma,
                                                     const float  layernorm_eps,
                                                     int          m,
                                                     int          n,
                                                     cudaStream_t stream);

template void invokeGeneralAddResidualT5PreLayerNorm(half*        output,
                                                     half*        norm_output,
                                                     const half*  input,
                                                     const half*  gamma,
                                                     const float  layernorm_eps,
                                                     int          m,
                                                     int          n,
                                                     cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeGeneralAddResidualT5PreLayerNorm(__nv_bfloat16*       output,
                                                     __nv_bfloat16*       norm_output,
                                                     const __nv_bfloat16* input,
                                                     const __nv_bfloat16* gamma,
                                                     const float          layernorm_eps,
                                                     int                  m,
                                                     int                  n,
                                                     cudaStream_t         stream);
#endif

template<typename T>
void invokeGeneralAddBiasResidualT5PreLayerNorm(T*           output,
                                                T*           norm_output,
                                                const T*     input,
                                                const T*     gamma,
                                                const T*     beta,
                                                const T*     bias,
                                                const float  layernorm_eps,
                                                int          m,
                                                int          n,
                                                cudaStream_t stream)
{
    if (beta != nullptr) {
        invokeGeneralAddBiasResidualPreLayerNorm(output,
                                                 norm_output,
                                                 output,
                                                 input,
                                                 (const T*)nullptr,
                                                 gamma,
                                                 beta,
                                                 bias,
                                                 layernorm_eps,
                                                 m,
                                                 n,
                                                 (float*)nullptr,
                                                 (float*)nullptr,
                                                 (float*)nullptr,
                                                 (float*)nullptr,
                                                 0,
                                                 stream);
    }
    else {
        FT_CHECK_WITH_INFO(bias == nullptr, "bias should be nullptr when beta is nullptr");
        invokeGeneralAddResidualT5PreLayerNorm(output, norm_output, input, gamma, layernorm_eps, m, n, stream);
    }
    return;
}

template void invokeGeneralAddBiasResidualT5PreLayerNorm(float*       output,
                                                         float*       norm_output,
                                                         const float* input,
                                                         const float* gamma,
                                                         const float* beta,
                                                         const float* bias,
                                                         const float  layernorm_eps,
                                                         int          m,
                                                         int          n,
                                                         cudaStream_t stream);

template void invokeGeneralAddBiasResidualT5PreLayerNorm(half*        output,
                                                         half*        norm_output,
                                                         const half*  input,
                                                         const half*  gamma,
                                                         const half*  beta,
                                                         const half*  bias,
                                                         const float  layernorm_eps,
                                                         int          m,
                                                         int          n,
                                                         cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeGeneralAddBiasResidualT5PreLayerNorm(__nv_bfloat16*       output,
                                                         __nv_bfloat16*       norm_output,
                                                         const __nv_bfloat16* input,
                                                         const __nv_bfloat16* gamma,
                                                         const __nv_bfloat16* beta,
                                                         const __nv_bfloat16* bias,
                                                         const float          layernorm_eps,
                                                         int                  m,
                                                         int                  n,
                                                         cudaStream_t         stream);
#endif

template<typename T, bool DYNAMIC_SCALING = false>
__global__ void generalLayerNorm(const T* __restrict input,
                                 const T* __restrict gamma,
                                 const T* __restrict beta,
                                 T*          normed_output,
                                 const float layernorm_eps,
                                 int         m,
                                 int         n,
                                 float*      scale,
                                 float*      dynamic_scale,
                                 const int   int8_mode)
{
    const int tid = threadIdx.x;

    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T*                                              shmem = reinterpret_cast<T*>(_shmem);

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    using Int8_Packed_T  = typename packed_as<int8_t, num_elems<T>::value>::type;
    using Int32_Packed_T = typename packed_as<int32_t, num_elems<T>::value>::type;
    using Float_Packed_T = typename packed_as<float, num_elems<T>::value>::type;
    using Scalar_T       = typename packed_as<T, 1>::type;

    const Float_Packed_T scale_to_int = cuda_cast<Float_Packed_T>(int8_mode == 2 ? *scale : 0.0f);

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
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

    Scalar_T abs_max = 1e-6f;

    for (int i = tid; i < n; i += blockDim.x) {
        const int index    = blockIdx.x * n + i;
        float     beta_val = (beta == nullptr) ? 0.0f : (float)ldg(&beta[i]);
        T         val      = (T)((((float)input[index] - s_mean) * s_variance) * (float)(ldg(&gamma[i])) + beta_val);

        if (DYNAMIC_SCALING) {
            abs_max  = cuda_max(cuda_max<Scalar_T, T>(cuda_abs(val)), abs_max);
            shmem[i] = val;
        }
        else if (int8_mode == 2) {
            reinterpret_cast<Int8_Packed_T*>(normed_output)[index] =
                cuda_cast<Int8_Packed_T>(cuda_cast<Float_Packed_T>(val) * scale_to_int);
        }
        else {
            normed_output[index] = val;
        }
    }

    if (DYNAMIC_SCALING) {
        float          abs_max_f               = blockAllReduceMax(cuda_cast<float>(abs_max));
        const Scalar_T dynamic_per_token_scale = 127. / abs_max_f;
        for (int i = tid; i < n; i += blockDim.x) {
            const int index                                        = blockIdx.x * n + i;
            reinterpret_cast<Int8_Packed_T*>(normed_output)[index] = cuda_cast<Int8_Packed_T>(
                cuda_cast<Float_Packed_T>(shmem[i]) * cuda_cast<Float_Packed_T>(dynamic_per_token_scale));
        }
        if (threadIdx.x == 0) {
            dynamic_scale[blockIdx.x] = (*scale * abs_max_f) / 127.f;
        }
    }
}

template<typename T>
void invokeGeneralLayerNorm(T*           out,
                            const T*     input,
                            const T*     gamma,
                            const T*     beta,
                            const float  layernorm_eps,
                            const int    m,
                            const int    n,
                            float*       scale,
                            float*       dynamic_scale,
                            const int    int8_mode,
                            cudaStream_t stream,
                            int          opt_version)
{
    dim3       grid(m);
    const bool dynamic_quant = dynamic_scale != nullptr;
#ifdef ENABLE_BF16
    if (n % 2 == 0 && (std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value)
#else
    if (n % 2 == 0 && (std::is_same<T, half>::value)
#endif
        && opt_version > 0) {
        int  half_n    = n / 2;
        int  half_n_32 = (half_n + 31) / 32 * 32;
        dim3 block(min(half_n_32, 512));
        int  rolls_per_thread = half_n / block.x;
        int  unroll_factor    = 8;
        while (unroll_factor > rolls_per_thread && unroll_factor > 1) {
            unroll_factor /= 2;
        }
        using T2 = typename TypeConverter<T>::Type;

        /* we launch (and instantiate) the kernel by specializing for unroll_factor -> residual_num -> is_bias ->
         * opt_version */
        dispatch_generalAddBiasResidualLayerNormOpt_unroll_factor((T2*)out,
                                                                  (T2*)out,
                                                                  (const T2*)out,
                                                                  (const T2*)nullptr,
                                                                  (const T2*)input,
                                                                  (const T2*)nullptr,
                                                                  (const T2*)gamma,
                                                                  (const T2*)beta,
                                                                  layernorm_eps,
                                                                  m,
                                                                  half_n,
                                                                  nullptr,
                                                                  nullptr,
                                                                  scale,
                                                                  dynamic_scale,
                                                                  int8_mode,
                                                                  grid,
                                                                  block,
                                                                  stream,
                                                                  opt_version,
                                                                  false,  // is_output
                                                                  1,      // residual_num
                                                                  unroll_factor);
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
        if (dynamic_quant) {
            size_t maxbytes = n * sizeof(T);
            if (maxbytes >= (48 << 10)) {
                check_cuda_error(cudaFuncSetAttribute(
                    generalLayerNorm<T, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
            }
            generalLayerNorm<T, true><<<grid, block, maxbytes, stream>>>(
                input, gamma, beta, out, layernorm_eps, m, n, scale, dynamic_scale, int8_mode);  // For gpt-3
        }
        else {
            generalLayerNorm<T, false><<<grid, block, 0, stream>>>(
                input, gamma, beta, out, layernorm_eps, m, n, scale, dynamic_scale, int8_mode);  // For gpt-3
        }
    }
}

template void invokeGeneralLayerNorm(float*       out,
                                     const float* input,
                                     const float* gamma,
                                     const float* beta,
                                     const float  layernorm_eps,
                                     const int    m,
                                     const int    n,
                                     float*       scale,
                                     float*       dynamic_scale,
                                     const int    int8_mode,
                                     cudaStream_t stream,
                                     int          opt_version);
template void invokeGeneralLayerNorm(half*        out,
                                     const half*  input,
                                     const half*  gamma,
                                     const half*  beta,
                                     const float  layernorm_eps,
                                     const int    m,
                                     const int    n,
                                     float*       scale,
                                     float*       dynamic_scale,
                                     const int    int8_mode,
                                     cudaStream_t stream,
                                     int          opt_version);
#ifdef ENABLE_BF16
template void invokeGeneralLayerNorm(__nv_bfloat16*       out,
                                     const __nv_bfloat16* input,
                                     const __nv_bfloat16* gamma,
                                     const __nv_bfloat16* beta,
                                     const float          layernorm_eps,
                                     const int            m,
                                     const int            n,
                                     float*               scale,
                                     float*               dynamic_scale,
                                     const int            int8_mode,
                                     cudaStream_t         stream,
                                     int                  opt_version);
#endif

template<typename T>
__global__ void generalT5LayerNorm(
    const T* __restrict input, const T* __restrict gamma, T* output, const float layernorm_eps, int m, int n)
{
    // layernorm module in the T5 style No bias and no subtraction of mean.
    const int tid = threadIdx.x;

    __shared__ float s_variance;
    float            variance = 0.0f;

    float local_var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = (float)(ldg(&input[blockIdx.x * n + i]));
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / (float)n + layernorm_eps);
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        output[blockIdx.x * n + i] =
            clamp_inf_for_half<T>((((float)input[blockIdx.x * n + i]) * s_variance) * (float)(ldg(&gamma[i])));
    }
}

template<typename T>
void invokeGeneralT5LayerNorm(T*           out,
                              const T*     input,
                              const T*     gamma,
                              const T*     beta,
                              const float  layernorm_eps,
                              const int    m,
                              const int    n,
                              cudaStream_t stream)
{
    if (beta != nullptr) {
        invokeGeneralLayerNorm(out, input, gamma, beta, layernorm_eps, m, n, (float*)nullptr, 0, stream);
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
    generalT5LayerNorm<T><<<grid, block, 0, stream>>>(input, gamma, out, layernorm_eps, m, n);  // For gpt-3
}

template void invokeGeneralT5LayerNorm(float*       out,
                                       const float* input,
                                       const float* gamma,
                                       const float* beta,
                                       const float  layernorm_eps,
                                       const int    m,
                                       const int    n,
                                       cudaStream_t stream);
template void invokeGeneralT5LayerNorm(half*        out,
                                       const half*  input,
                                       const half*  gamma,
                                       const half*  beta,
                                       const float  layernorm_eps,
                                       const int    m,
                                       const int    n,
                                       cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeGeneralT5LayerNorm(__nv_bfloat16*       out,
                                       const __nv_bfloat16* input,
                                       const __nv_bfloat16* gamma,
                                       const __nv_bfloat16* beta,
                                       const float          layernorm_eps,
                                       const int            m,
                                       const int            n,
                                       cudaStream_t         stream);
#endif

/*******************  invokeLayernormShiftPartition  ***********************/

// applied to half2 and bfloat162
template<typename T2>
__global__ void layernorm_shift_partition(T2*         out_ptr,
                                          const T2*   input_ptr,
                                          const T2*   gamma_ptr,
                                          const T2*   beta_ptr,
                                          const float layernorm_eps,
                                          int         batch,
                                          int         H,
                                          int         W,
                                          int         n,
                                          int         shift_size,
                                          int         window_size)
{
    const int batch_offset       = blockIdx.z * gridDim.y * gridDim.x;
    const int bid                = batch_offset + blockIdx.y * gridDim.x + blockIdx.x;
    const int shifted_H_idx      = (shift_size != 0) ? ((blockIdx.y - shift_size + gridDim.y) % gridDim.y) : blockIdx.y;
    const int shifted_W_idx      = (shift_size != 0) ? ((blockIdx.x - shift_size + gridDim.x) % gridDim.x) : blockIdx.x;
    const int window_H_idx       = shifted_H_idx / window_size;
    const int window_W_idx       = shifted_W_idx / window_size;
    const int stride_of_window_H = W / window_size;
    const int window_idx         = window_H_idx * stride_of_window_H + window_W_idx;
    const int idx_in_window      = (shifted_H_idx % window_size) * window_size + (shifted_W_idx % window_size);
    const int output_bid         = batch_offset + window_idx * window_size * window_size + idx_in_window;
    int       tid                = threadIdx.x;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;
    float2           local_out_fp2;

    float local_out = 0.0f;
    int   id        = bid * n + tid;
    if (tid < n) {
        local_out_fp2 = cuda_cast<float2>(ldg(input_ptr + id));
        local_out += local_out_fp2.x;
        local_out += local_out_fp2.y;
    }

    mean = blockReduceSum<float>(local_out);
    if (threadIdx.x == 0)
        s_mean = mean / (n * 2);
    __syncthreads();

    if (tid < n) {
        variance = (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
        variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
    }
    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / (n * 2) + layernorm_eps);
    }
    __syncthreads();

    if (tid < n) {
        float2 gamma_val              = cuda_cast<float2>(ldg(&gamma_ptr[tid]));
        float2 beta_val               = cuda_cast<float2>(ldg(&beta_ptr[tid]));
        local_out_fp2.x               = (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
        local_out_fp2.y               = (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
        out_ptr[output_bid * n + tid] = cuda_cast<T2>(local_out_fp2);
    }
}

// applied to float
template<>
__global__ void layernorm_shift_partition<float>(float*       out,
                                                 const float* input,
                                                 const float* gamma,
                                                 const float* beta,
                                                 const float  layernorm_eps,
                                                 int          batch,
                                                 int          H,
                                                 int          W,
                                                 int          n,
                                                 int          shift_size,
                                                 int          window_size)
{
    int       tid                = threadIdx.x;
    const int batch_offset       = blockIdx.z * gridDim.y * gridDim.x;
    const int bid                = batch_offset + blockIdx.y * gridDim.x + blockIdx.x;
    const int shifted_H_idx      = (shift_size != 0) ? ((blockIdx.y - shift_size + gridDim.y) % gridDim.y) : blockIdx.y;
    const int shifted_W_idx      = (shift_size != 0) ? ((blockIdx.x - shift_size + gridDim.x) % gridDim.x) : blockIdx.x;
    const int window_H_idx       = shifted_H_idx / window_size;
    const int window_W_idx       = shifted_W_idx / window_size;
    const int stride_of_window_H = W / window_size;
    const int window_idx         = window_H_idx * stride_of_window_H + window_W_idx;
    const int idx_in_window      = (shifted_H_idx % window_size) * window_size + (shifted_W_idx % window_size);
    const int output_bid         = batch_offset + window_idx * window_size * window_size + idx_in_window;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    float local_out = (tid < n) ? (float)(__ldg(input + bid * n + tid)) : 0.0f;

    mean = blockReduceSum<float>(local_out);
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float diff = (tid < n) ? (local_out - s_mean) : 0.0f;
    variance   = blockReduceSum<float>(diff * diff);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

    if (tid < n) {
        out[output_bid * n + tid] =
            (float)(((local_out - s_mean) * s_variance) * (float)(__ldg(&gamma[tid])) + (float)(__ldg(&beta[tid])));
    }
}

// Applied to half2 and bfloat162
template<typename T2>
__global__ void layernorm_shift_partition_v2(T2* out_ptr,
                                             const T2* __restrict input_ptr,
                                             const T2* __restrict gamma_ptr,
                                             const T2* __restrict beta_ptr,
                                             const float layernorm_eps,
                                             int         batch,
                                             int         H,
                                             int         W,
                                             int         n,
                                             int         shift_size,
                                             int         window_size)
{
    using T1                     = typename TypeConverter<T2>::Type;  // half2 to half, bfloat162 to bfloat
    const int ite                = 4;
    const int tid                = threadIdx.x;
    const int batch_offset       = blockIdx.z * gridDim.y * gridDim.x;
    const int bid                = batch_offset + blockIdx.y * gridDim.x + blockIdx.x;
    const int shifted_H_idx      = (shift_size != 0) ? ((blockIdx.y - shift_size + gridDim.y) % gridDim.y) : blockIdx.y;
    const int shifted_W_idx      = (shift_size != 0) ? ((blockIdx.x - shift_size + gridDim.x) % gridDim.x) : blockIdx.x;
    const int window_H_idx       = shifted_H_idx / window_size;
    const int window_W_idx       = shifted_W_idx / window_size;
    const int stride_of_window_H = W / window_size;
    const int window_idx         = window_H_idx * stride_of_window_H + window_W_idx;
    const int idx_in_window      = (shifted_H_idx % window_size) * window_size + (shifted_W_idx % window_size);
    const int output_bid         = batch_offset + window_idx * window_size * window_size + idx_in_window;
    const int offset             = bid * n;
    const int output_offset      = output_bid * n;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;
    T2               local_out_half2[ite];
    const T2         zero = {static_cast<T1>(0.0f), static_cast<T1>(0.0f)};

    // float sum = 0.0f;
    T2 sum = cuda_cast<T2>(0.0f);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        if (col_id < n) {
            local_out_half2[i] = ldg(input_ptr + offset + col_id);
            sum                = add(sum, local_out_half2[i]);
        }
    }

    mean = blockReduceSum<float>((float)(sum.x + sum.y));
    if (threadIdx.x == 0) {
        s_mean = mean / (n * 2);
    }
    __syncthreads();

    float var      = 0.0f;
    T2    s_mean_2 = cuda_cast<T2>(s_mean);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        if (col_id < n) {
            local_out_half2[i] = hsub2(local_out_half2[i], s_mean_2);
            float v1           = (float)local_out_half2[i].x;
            float v2           = (float)local_out_half2[i].y;
            var += v1 * v1 + v2 * v2;
        }
    }

    variance = blockReduceSum<float>(var);
    if (tid == 0) {
        s_variance = rsqrtf(variance / (n * 2) + layernorm_eps);
    }
    __syncthreads();

    T2 s_var_2 = cuda_cast<T2>(s_variance);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        if (col_id < n) {
            out_ptr[output_offset + col_id] =
                fma(local_out_half2[i], s_var_2, ldg(&gamma_ptr[col_id]), ldg(&beta_ptr[col_id]));
        }
    }
}

template<>
__global__ void layernorm_shift_partition_v2<float>(float* out,
                                                    const float* __restrict input,
                                                    const float* __restrict gamma,
                                                    const float* __restrict beta,
                                                    const float layernorm_eps,
                                                    int         batch,
                                                    int         H,
                                                    int         W,
                                                    int         n,
                                                    int         shift_size,
                                                    int         window_size)
{
    const int ite                = 4;
    const int tid                = threadIdx.x;
    const int batch_offset       = blockIdx.z * gridDim.y * gridDim.x;
    const int bid                = batch_offset + blockIdx.y * gridDim.x + blockIdx.x;
    const int shifted_H_idx      = (shift_size != 0) ? ((blockIdx.y - shift_size + gridDim.y) % gridDim.y) : blockIdx.y;
    const int shifted_W_idx      = (shift_size != 0) ? ((blockIdx.x - shift_size + gridDim.x) % gridDim.x) : blockIdx.x;
    const int window_H_idx       = shifted_H_idx / window_size;
    const int window_W_idx       = shifted_W_idx / window_size;
    const int stride_of_window_H = W / window_size;
    const int window_idx         = window_H_idx * stride_of_window_H + window_W_idx;
    const int idx_in_window      = (shifted_H_idx % window_size) * window_size + (shifted_W_idx % window_size);
    const int output_bid         = batch_offset + window_idx * window_size * window_size + idx_in_window;
    const int offset             = bid * n;
    const int output_offset      = output_bid * n;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;
    float            local_out[ite];

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
            float diff   = local_out[i] - s_mean;
            local_out[i] = diff;
            var += diff * diff;
        }
    }

    variance = blockReduceSum<float>(var);
    if (tid == 0) {
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        if (col_id < n) {
            out[output_offset + col_id] =
                (float)(local_out[i] * s_variance * (float)__ldg(&gamma[col_id]) + (float)__ldg(&beta[col_id]));
        }
    }
}

// Applied to half or Bfloat16
template<typename T>
void invokeLayernormShiftPartition(T*           out,
                                   const T*     input,
                                   const T*     gamma,
                                   const T*     beta,
                                   const float  layernorm_eps,
                                   int          batch,
                                   int          H,
                                   int          W,
                                   int          n,
                                   int          shift_size,
                                   int          window_size,
                                   cudaStream_t stream)
{
    dim3 grid(W, H, batch);
    int  blockSize = n / 2;
    blockSize      = (blockSize + 31) / 32 * 32;

    using T2 = typename TypeConverter<T>::Type;  // bf162 or half2

    if ((batch * H * W >= 512 && blockSize >= 768) || blockSize > 1024) {
        blockSize = ((blockSize / 4) + 31) / 32 * 32;
        layernorm_shift_partition_v2<T2><<<grid, blockSize, 0, stream>>>((T2*)out,
                                                                         (const T2*)input,
                                                                         (const T2*)gamma,
                                                                         (const T2*)beta,
                                                                         layernorm_eps,
                                                                         batch,
                                                                         H,
                                                                         W,
                                                                         n / 2,
                                                                         shift_size,
                                                                         window_size);
    }
    else {
        layernorm_shift_partition<T2><<<grid, blockSize, 0, stream>>>((T2*)out,
                                                                      (const T2*)input,
                                                                      (const T2*)gamma,
                                                                      (const T2*)beta,
                                                                      layernorm_eps,
                                                                      batch,
                                                                      H,
                                                                      W,
                                                                      n / 2,
                                                                      shift_size,
                                                                      window_size);
    }
}

template<>
void invokeLayernormShiftPartition<float>(float*       out,
                                          const float* input,
                                          const float* gamma,
                                          const float* beta,
                                          const float  layernorm_eps,
                                          int          batch,
                                          int          H,
                                          int          W,
                                          int          n,
                                          int          shift_size,
                                          int          window_size,
                                          cudaStream_t stream)
{
    dim3 grid(W, H, batch);
    int  blockSize = (n + 31) / 32 * 32;
    if (blockSize >= 768) {
        blockSize = ((blockSize / 4) + 31) / 32 * 32;
        layernorm_shift_partition_v2<float><<<grid, blockSize, 0, stream>>>(
            out, input, gamma, beta, layernorm_eps, batch, H, W, n, shift_size, window_size);
    }
    else {
        layernorm_shift_partition<float><<<grid, blockSize, 0, stream>>>(
            out, input, gamma, beta, layernorm_eps, batch, H, W, n, shift_size, window_size);
    }
}

template void invokeLayernormShiftPartition<float>(float*       out,
                                                   const float* input,
                                                   const float* gamma,
                                                   const float* beta,
                                                   const float  layernorm_eps,
                                                   int          batch,
                                                   int          H,
                                                   int          W,
                                                   int          n,
                                                   int          shift_size,
                                                   int          window_size,
                                                   cudaStream_t stream);

template void invokeLayernormShiftPartition<half>(half*        out,
                                                  const half*  input,
                                                  const half*  gamma,
                                                  const half*  beta,
                                                  const float  layernorm_eps,
                                                  int          batch,
                                                  int          H,
                                                  int          W,
                                                  int          n,
                                                  int          shift_size,
                                                  int          window_size,
                                                  cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeLayernormShiftPartition<__nv_bfloat16>(__nv_bfloat16*       out,
                                                           const __nv_bfloat16* input,
                                                           const __nv_bfloat16* gamma,
                                                           const __nv_bfloat16* beta,
                                                           const float          layernorm_eps,
                                                           int                  batch,
                                                           int                  H,
                                                           int                  W,
                                                           int                  n,
                                                           int                  shift_size,
                                                           int                  window_size,
                                                           cudaStream_t         stream);
#endif

/*******************  invokeAddBiasLayernorm  ***********************/

template<typename T>
__global__ void add_bias_layernorm(T* out, const T* bias, const T* gamma, const T* beta, float layernorm_eps, int n)
{
    int              tid = threadIdx.x;
    const int        bid = blockIdx.x;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    float local_out = (tid < n) ? (float)(out[bid * n + tid] + ldg(&bias[tid])) : 0.0f;

    mean = blockReduceSum<float>(local_out);
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float diff = (tid < n) ? (local_out - s_mean) : 0.0f;
    variance   = blockReduceSum<float>(diff * diff);
    if (threadIdx.x == 0) {
        s_variance = variance / n + layernorm_eps;
    }
    __syncthreads();

    if (tid < n) {
        out[bid * n + tid] =
            (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(ldg(&gamma[tid])) + (float)(ldg(&beta[tid])));
    }
}

template<typename T>
__global__ void add_bias_layernorm_v2(
    T* out, const T* __restrict bias, const T* __restrict gamma, const T* __restrict beta, float layernorm_eps, int n)
{
    const int ite    = 4;
    const int tid    = threadIdx.x;
    const int bid    = blockIdx.x;
    const int offset = bid * n;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;
    float            local_out[ite];

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id   = i * blockDim.x + tid;
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
        int   col_id = i * blockDim.x + tid;
        float diff   = (col_id < n) ? (local_out[i] - s_mean) : 0.0f;
        var += diff * diff;
    }

    variance = blockReduceSum<float>(var);
    if (tid == 0) {
        s_variance = rsqrtf(variance / n + layernorm_eps);
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

template<typename T>
void invokeAddBiasLayernorm(T*           out,
                            const T*     bias,
                            const T*     gamma,
                            const T*     beta,
                            float        layernorm_eps,
                            int          m,
                            int          n,
                            cudaStream_t stream,
                            int          opt_version)
{
    dim3 grid(m);
    if (n % 2 == 0 && std::is_same<T, half>::value && opt_version > 0) {
        int  half_n    = n / 2;
        int  half_n_32 = (half_n + 31) / 32 * 32;
        dim3 block(min(half_n_32, 512));
        int  rolls_per_thread = half_n / block.x;
        int  unroll_factor    = 8;
        while (unroll_factor > rolls_per_thread && unroll_factor > 1) {
            unroll_factor /= 2;
        }
        using T2 = typename TypeConverter<T>::Type;

        /* we launch (and instantiate) the kernel by specializing for unroll_factor -> residual_num -> is_bias ->
         * opt_version */
        dispatch_generalAddBiasResidualLayerNormOpt_unroll_factor((T2*)out,
                                                                  (T2*)out,
                                                                  (const T2*)out,
                                                                  (const T2*)bias,
                                                                  (const T2*)out,
                                                                  (const T2*)nullptr,
                                                                  (const T2*)gamma,
                                                                  (const T2*)beta,
                                                                  layernorm_eps,
                                                                  m,
                                                                  half_n,
                                                                  nullptr,
                                                                  nullptr,
                                                                  nullptr,
                                                                  nullptr,
                                                                  0,
                                                                  grid,
                                                                  block,
                                                                  stream,
                                                                  opt_version,
                                                                  false,  // is_output
                                                                  1,      // residual_num
                                                                  unroll_factor);
    }
    else {
        int blockSize = (n + 31) / 32 * 32;
        if (blockSize >= 768) {
            blockSize = ((blockSize / 4) + 31) / 32 * 32;
            add_bias_layernorm_v2<T><<<grid, blockSize, 0, stream>>>(out, bias, gamma, beta, layernorm_eps, n);
        }
        else {
            add_bias_layernorm<T><<<grid, blockSize, 0, stream>>>(out, bias, gamma, beta, layernorm_eps, n);
        }
    }
}

template void invokeAddBiasLayernorm<float>(float*       out,
                                            const float* bias,
                                            const float* gamma,
                                            const float* beta,
                                            const float  layernorm_eps,
                                            int          m,
                                            int          n,
                                            cudaStream_t stream,
                                            int          opt_version);

template void invokeAddBiasLayernorm<half>(half*        out,
                                           const half*  bias,
                                           const half*  gamma,
                                           const half*  beta,
                                           const float  layernorm_eps,
                                           int          m,
                                           int          n,
                                           cudaStream_t stream,
                                           int          opt_version);
#ifdef ENABLE_BF16
template void invokeAddBiasLayernorm<__nv_bfloat16>(__nv_bfloat16*       out,
                                                    const __nv_bfloat16* bias,
                                                    const __nv_bfloat16* gamma,
                                                    const __nv_bfloat16* beta,
                                                    const float          layernorm_eps,
                                                    int                  m,
                                                    int                  n,
                                                    cudaStream_t         stream,
                                                    int                  opt_version);
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
                                   const float layernorm_eps,
                                   int         batch,
                                   int         H,
                                   int         W,
                                   int         n)
{
    const int    ite             = 4;
    const int    tid             = threadIdx.x;
    const int    W_idx           = blockIdx.x;
    const int    H_idx           = blockIdx.y;
    const size_t batch_offset    = blockIdx.z * H * W * n;
    const int    input_H_stride  = W * n / 2;
    const int    output_H_stride = W * n;
    const int    n_4             = n >> 2;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;
    float            local_out[ite];

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        if (col_id < n) {
            int    part_id     = col_id / n_4;
            int    offset_in_W = part_id / 2;
            int    offset_in_H = part_id % 2;
            size_t input_id    = batch_offset + (2 * H_idx + offset_in_H) * input_H_stride
                              + (2 * W_idx + offset_in_W) * n_4 + (col_id % n_4);
            local_out[i] = (float)(ldg(input + input_id));
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
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        if (col_id < n) {
            size_t output_idx = batch_offset + H_idx * output_H_stride + W_idx * n + col_id;
            out[output_idx]   = (T)(local_out[i] * s_variance * (float)ldg(&gamma[col_id]) + (float)ldg(&beta[col_id]));
        }
    }
}

// TODO : accelerate with half2
template<typename T>
void invokeMergeLayernorm(T*           output,
                          const T*     input,
                          const T*     gamma,
                          const T*     beta,
                          float        layernorm_eps,
                          int          batch,
                          int          H,
                          int          W,
                          int          n,
                          cudaStream_t stream)
{
    if ((W % 2 != 0) || (H % 2 != 0)) {
        printf("[ERROR][invokeMergeLayernorm] H(W) should be a multiple of 2.\n");
        return;
    }
    dim3 grid(W / 2, H / 2, batch);
    int  blockSize = 4 * n;
    blockSize      = (blockSize + 31) / 32 * 32;
    // TODO
    // if (blockSize >= 768)
    {
        blockSize = ((blockSize / 4) + 31) / 32 * 32;
        merge_layernorm_v2<T>
            <<<grid, blockSize, 0, stream>>>(output, input, gamma, beta, layernorm_eps, batch, H / 2, W / 2, n * 4);
    }
    /*
    else
      merge_layernorm<T><<<grid, blockSize, 0, stream>>>(output, input, gamma, beta, batch, H/2, W/2, n*4);
    */
}

template void invokeMergeLayernorm<float>(float*       output,
                                          const float* input,
                                          const float* gamma,
                                          const float* beta,
                                          float        layernorm_eps,
                                          int          batch,
                                          int          H,
                                          int          W,
                                          int          n,
                                          cudaStream_t stream);

template void invokeMergeLayernorm<half>(half*        output,
                                         const half*  input,
                                         const half*  gamma,
                                         const half*  beta,
                                         float        layernorm_eps,
                                         int          batch,
                                         int          H,
                                         int          W,
                                         int          n,
                                         cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeMergeLayernorm<__nv_bfloat16>(__nv_bfloat16*       output,
                                                  const __nv_bfloat16* input,
                                                  const __nv_bfloat16* gamma,
                                                  const __nv_bfloat16* beta,
                                                  const float          layernorm_eps,
                                                  int                  batch,
                                                  int                  H,
                                                  int                  W,
                                                  int                  n,
                                                  cudaStream_t         stream);
#endif

/*******************  invokeAddBiasLayernormAddRes  ***********************/
template<typename T, int T_per_thread>
__global__ void add_bias_layernorm_add_res(
    T* out, const T* input, const T* bias, const T* gamma, const T* beta, const float layernorm_eps, int n)
{
    int              tid  = threadIdx.x;
    const int        bid  = blockIdx.x;
    const int        bdim = blockDim.x;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    T   local_val[T_per_thread];
    int b_offset = bid * n;
    input        = input + b_offset;
    out          = out + b_offset;

    float local_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < T_per_thread; i++) {
        int index = i * bdim + tid;
        if (index < n) {
            T out_val    = out[index];
            T bias_val   = bias[index];
            T tmp        = out_val + bias_val;
            local_val[i] = tmp;
            local_sum += float(tmp);
        }
    }

    mean = blockReduceSum<float>(local_sum);
    if (tid == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    local_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < T_per_thread; i++) {
        int index = i * bdim + tid;
        if (index < n) {
            float tmp = float(local_val[i]) - s_mean;
            local_sum += tmp * tmp;
        }
    }

    variance = blockReduceSum<float>(local_sum);
    if (tid == 0) {
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < T_per_thread; i++) {
        int index = i * bdim + tid;
        if (index < n) {
            T gamma_val  = gamma[index];
            T beta_val   = beta[index];
            T input_val  = input[index];
            T output_val = T((float(local_val[i]) - s_mean) * s_variance * (float)(gamma_val) + (float)(beta_val)
                             + (float)(input_val));
            out[index]   = output_val;
        }
    }
}

template<int T_per_thread>
__global__ void add_bias_layernorm_add_res_e2(half2*       out,
                                              const half2* input,
                                              const half2* bias,
                                              const half2* gamma,
                                              const half2* beta,
                                              const float  layernorm_eps,
                                              int          n)
{
    int              tid  = threadIdx.x;
    const int        bid  = blockIdx.x;
    const int        bdim = blockDim.x;
    __shared__ float s_mean;
    __shared__ float s_variance;
    const int        n_2 = n >> 1;

    half2 local_val[T_per_thread];
    int   b_offset = bid * n_2;
    input          = input + b_offset;
    out            = out + b_offset;

    float local_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < T_per_thread; i++) {
        int index = i * bdim + tid;
        if (index < n_2) {
            half2 out_val     = out[index];
            half2 bias_val    = bias[index];
            half2 tmp         = __hadd2(out_val, bias_val);
            local_val[i]      = tmp;
            float2 tmp_float2 = __half22float2(tmp);
            local_sum += tmp_float2.x + tmp_float2.y;
        }
    }

    local_sum = blockReduceSum<float>(local_sum);
    if (tid == 0) {
        s_mean = local_sum / n;
    }
    __syncthreads();

    local_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < T_per_thread; i++) {
        int index = i * bdim + tid;
        if (index < n_2) {
            float2 tmp_float2 = __half22float2(local_val[i]);
            tmp_float2.x      = tmp_float2.x - s_mean;
            tmp_float2.y      = tmp_float2.y - s_mean;
            local_sum += tmp_float2.x * tmp_float2.x + tmp_float2.y * tmp_float2.y;
        }
    }

    local_sum = blockReduceSum<float>(local_sum);
    if (tid == 0) {
        s_variance = rsqrtf(local_sum / n + layernorm_eps);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < T_per_thread; i++) {
        int index = i * bdim + tid;
        if (index < n_2) {
            float2 gamma_val = __half22float2(gamma[index]);
            float2 beta_val  = __half22float2(beta[index]);
            float2 input_val = __half22float2(input[index]);
            float2 output_val;
            float2 tmp_float2 = __half22float2(local_val[i]);
            output_val.x      = (tmp_float2.x - s_mean) * s_variance * gamma_val.x + beta_val.x + input_val.x;
            output_val.y      = (tmp_float2.y - s_mean) * s_variance * gamma_val.y + beta_val.y + input_val.y;
            out[index]        = __float22half2_rn(output_val);
        }
    }
}

template<int T_per_thread>
__global__ void add_bias_layernorm_add_res_e2(float2*       out,
                                              const float2* input,
                                              const float2* bias,
                                              const float2* gamma,
                                              const float2* beta,
                                              const float   layernorm_eps,
                                              int           n)
{
    int              tid  = threadIdx.x;
    const int        bid  = blockIdx.x;
    const int        bdim = blockDim.x;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;
    const int        n_2      = n >> 1;

    float2 local_val[T_per_thread];
    int    b_offset = bid * n_2;
    input           = input + b_offset;
    out             = out + b_offset;

    float local_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < T_per_thread; i++) {
        int index = i * bdim + tid;
        if (index < n_2) {
            float2 out_val  = out[index];
            float2 bias_val = bias[index];
            float2 tmp;
            tmp.x        = out_val.x + bias_val.x;
            tmp.y        = out_val.y + bias_val.y;
            local_val[i] = tmp;
            local_sum += tmp.x + tmp.y;
        }
    }

    mean = blockReduceSum<float>(local_sum);
    if (tid == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    local_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < T_per_thread; i++) {
        int index = i * bdim + tid;
        if (index < n_2) {
            float2 tmp = local_val[i];
            tmp.x      = tmp.x - s_mean;
            tmp.y      = tmp.y - s_mean;
            local_sum += tmp.x * tmp.x + tmp.y * tmp.y;
        }
    }

    variance = blockReduceSum<float>(local_sum);
    if (tid == 0) {
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < T_per_thread; i++) {
        int index = i * bdim + tid;
        if (index < n_2) {
            float2 gamma_val = gamma[index];
            float2 beta_val  = beta[index];
            float2 input_val = input[index];
            float2 output_val;
            float2 tmp   = local_val[i];
            output_val.x = (tmp.x - s_mean) * s_variance * gamma_val.x + beta_val.x + input_val.x;
            output_val.y = (tmp.y - s_mean) * s_variance * gamma_val.y + beta_val.y + input_val.y;
            out[index]   = output_val;
        }
    }
}

#define MACRO_ADD_BIAS_LN_ADD_RES(T_per_TVec)                                                                          \
    blockSize = (blockSize + TVec_per_thread - 1) / TVec_per_thread;                                                   \
    if (T_per_TVec == 2) {                                                                                             \
        if (std::is_same<T, half>::value) {                                                                            \
            add_bias_layernorm_add_res_e2<TVec_per_thread><<<m, blockSize, 0, stream>>>((half2*)out,                   \
                                                                                        (const half2*)input,           \
                                                                                        (const half2*)bias,            \
                                                                                        (const half2*)gamma,           \
                                                                                        (const half2*)beta,            \
                                                                                        layernorm_eps,                 \
                                                                                        n);                            \
        }                                                                                                              \
        else if (std::is_same<T, float>::value) {                                                                      \
            add_bias_layernorm_add_res_e2<TVec_per_thread><<<m, blockSize, 0, stream>>>((float2*)out,                  \
                                                                                        (const float2*)input,          \
                                                                                        (const float2*)bias,           \
                                                                                        (const float2*)gamma,          \
                                                                                        (const float2*)beta,           \
                                                                                        layernorm_eps,                 \
                                                                                        n);                            \
        }                                                                                                              \
        else {                                                                                                         \
            FT_CHECK_WITH_INFO(false, "[ERROR][invokeAddBiasLayernormAddRes] unsupported dataType.\n");                \
        }                                                                                                              \
    }                                                                                                                  \
    else {                                                                                                             \
        if (std::is_same<T, half>::value) {                                                                            \
            add_bias_layernorm_add_res<half, TVec_per_thread><<<m, blockSize, 0, stream>>>((half*)out,                 \
                                                                                           (const half*)input,         \
                                                                                           (const half*)bias,          \
                                                                                           (const half*)gamma,         \
                                                                                           (const half*)beta,          \
                                                                                           layernorm_eps,              \
                                                                                           n);                         \
        }                                                                                                              \
        else if (std::is_same<T, float>::value) {                                                                      \
            add_bias_layernorm_add_res<float, TVec_per_thread><<<m, blockSize, 0, stream>>>((float*)out,               \
                                                                                            (const float*)input,       \
                                                                                            (const float*)bias,        \
                                                                                            (const float*)gamma,       \
                                                                                            (const float*)beta,        \
                                                                                            layernorm_eps,             \
                                                                                            n);                        \
        }                                                                                                              \
        else {                                                                                                         \
            FT_CHECK_WITH_INFO(false, "[ERROR][invokeAddBiasLayernormAddRes] unsupported dataType.\n");                \
        }                                                                                                              \
    }

template<typename T>
void invokeAddBiasLayernormAddRes(T*           out,
                                  const T*     input,
                                  const T*     bias,
                                  const T*     gamma,
                                  const T*     beta,
                                  const float  layernorm_eps,
                                  int          m,
                                  int          n,
                                  cudaStream_t stream)
{
    if (n % 2 == 0) {
        const int T_per_TVec = 2;
        int       blockSize  = (n / 2 + 31) / 32 * 32;
        if (blockSize <= 1024) {
            const int TVec_per_thread = 1;
            MACRO_ADD_BIAS_LN_ADD_RES(T_per_TVec);
        }
        else if (blockSize <= 2048) {
            const int TVec_per_thread = 2;
            MACRO_ADD_BIAS_LN_ADD_RES(T_per_TVec);
        }
        else if (blockSize <= 4096) {
            const int TVec_per_thread = 4;
            MACRO_ADD_BIAS_LN_ADD_RES(T_per_TVec);
        }
        else if (blockSize <= 8192) {
            const int TVec_per_thread = 8;
            MACRO_ADD_BIAS_LN_ADD_RES(T_per_TVec);
        }
        else if (blockSize <= 16384) {
            const int TVec_per_thread = 16;
            MACRO_ADD_BIAS_LN_ADD_RES(T_per_TVec);
        }
        else {
            FT_CHECK_WITH_INFO(false, "[ERROR][invokeAddBiasLayernormAddRes] unsupported n.\n");
        }
    }
    else {
        const int T_per_TVec = 1;
        int       blockSize  = (n + 31) / 32 * 32;
        if (blockSize <= 1024) {
            const int TVec_per_thread = 1;
            MACRO_ADD_BIAS_LN_ADD_RES(T_per_TVec);
        }
        else if (blockSize <= 2048) {
            const int TVec_per_thread = 2;
            MACRO_ADD_BIAS_LN_ADD_RES(T_per_TVec);
        }
        else if (blockSize <= 4096) {
            const int TVec_per_thread = 4;
            MACRO_ADD_BIAS_LN_ADD_RES(T_per_TVec);
        }
        else if (blockSize <= 8192) {
            const int TVec_per_thread = 8;
            MACRO_ADD_BIAS_LN_ADD_RES(T_per_TVec);
        }
        else if (blockSize <= 16384) {
            const int TVec_per_thread = 16;
            MACRO_ADD_BIAS_LN_ADD_RES(T_per_TVec);
        }
        else if (blockSize <= 32768) {
            const int TVec_per_thread = 32;
            MACRO_ADD_BIAS_LN_ADD_RES(T_per_TVec);
        }
        else {
            FT_CHECK_WITH_INFO(false, "[ERROR][invokeAddBiasLayernormAddRes] unsupported n.\n");
        }
    }
}

template void invokeAddBiasLayernormAddRes(float*       out,
                                           const float* input,
                                           const float* bias,
                                           const float* gamma,
                                           const float* beta,
                                           const float  layernorm_eps,
                                           int          m,
                                           int          n,
                                           cudaStream_t stream);

template void invokeAddBiasLayernormAddRes(half*        out,
                                           const half*  input,
                                           const half*  bias,
                                           const half*  gamma,
                                           const half*  beta,
                                           const float  layernorm_eps,
                                           int          m,
                                           int          n,
                                           cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeAddBiasLayernormAddRes(__nv_bfloat16*       out,
                                           const __nv_bfloat16* input,
                                           const __nv_bfloat16* bias,
                                           const __nv_bfloat16* gamma,
                                           const __nv_bfloat16* beta,
                                           const float          layernorm_eps,
                                           int                  m,
                                           int                  n,
                                           cudaStream_t         stream);
#endif

}  // namespace fastertransformer
