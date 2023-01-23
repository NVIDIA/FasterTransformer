/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/kernels/layernorm_fp8_kernels.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include <cuda_fp16.h>

namespace fastertransformer {

template<typename T, int QUANTIZE_MODE>
__global__ void
quatizeVectorE4M3(__nv_fp8_e4m3* output, float const* input_qua_amax_ptr, T const* input, uint32_t size, uint32_t n)
{
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += blockDim.x * gridDim.x) {
        if (QUANTIZE_MODE == 0) {
            output[i] = __nv_fp8_e4m3((float)(input[i]) / __ldg(input_qua_amax_ptr + (i % n)));
        }
        else {
            output[i] = __nv_fp8_e4m3((float)(input[i]) / __ldg(input_qua_amax_ptr));
        }
    }
}

template<typename T, int QUANTIZE_MODE>
void invokeQuatizeVectorE4M3(__nv_fp8_e4m3* output,
                             float const*   input_qua_amax_ptr,
                             T const*       input,
                             uint32_t       size,
                             uint32_t       n,
                             cudaStream_t   stream)
{
    dim3 grid(1);
    dim3 block(256);
    quatizeVectorE4M3<T, QUANTIZE_MODE><<<grid, block, 0, stream>>>(output, input_qua_amax_ptr, input, size, n);
}

template void invokeQuatizeVectorE4M3<float, 0>(__nv_fp8_e4m3* output,
                                                float const*   input_qua_amax_ptr,
                                                float const*   input,
                                                uint32_t       size,
                                                uint32_t       n,
                                                cudaStream_t   stream);
template void invokeQuatizeVectorE4M3<half, 0>(__nv_fp8_e4m3* output,
                                               float const*   input_qua_amax_ptr,
                                               half const*    input,
                                               uint32_t       size,
                                               uint32_t       n,
                                               cudaStream_t   stream);
template void invokeQuatizeVectorE4M3<__nv_bfloat16, 0>(__nv_fp8_e4m3*       output,
                                                        float const*         input_qua_amax_ptr,
                                                        __nv_bfloat16 const* input,
                                                        uint32_t             size,
                                                        uint32_t             n,
                                                        cudaStream_t         stream);

template void invokeQuatizeVectorE4M3<float, 1>(__nv_fp8_e4m3* output,
                                                float const*   input_qua_amax_ptr,
                                                float const*   input,
                                                uint32_t       size,
                                                uint32_t       n,
                                                cudaStream_t   stream);
template void invokeQuatizeVectorE4M3<half, 1>(__nv_fp8_e4m3* output,
                                               float const*   input_qua_amax_ptr,
                                               half const*    input,
                                               uint32_t       size,
                                               uint32_t       n,
                                               cudaStream_t   stream);
template void invokeQuatizeVectorE4M3<__nv_bfloat16, 1>(__nv_fp8_e4m3*       output,
                                                        float const*         input_qua_amax_ptr,
                                                        __nv_bfloat16 const* input,
                                                        uint32_t             size,
                                                        uint32_t             n,
                                                        cudaStream_t         stream);

template<typename T, int QUANTIZE_MODE>
__global__ void
dequatizeVectorE4M3(T* output, float const* qua_amax_ptr, __nv_fp8_e4m3 const* input, uint32_t size, uint32_t n)
{
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += blockDim.x * gridDim.x) {
        if (QUANTIZE_MODE == 0) {
            output[i] = float(input[i]) * __ldg(qua_amax_ptr + (i % n));
        }
        else {
            output[i] = float(input[i]) * __ldg(qua_amax_ptr);
        }
    }
}

template<typename T, int QUANTIZE_MODE>
void invokeDequatizeVectorE4M3(
    T* output, float const* qua_amax_ptr, __nv_fp8_e4m3 const* input, uint32_t size, uint32_t n, cudaStream_t stream)
{
    dim3 grid(1);
    dim3 block(256);
    dequatizeVectorE4M3<T, QUANTIZE_MODE><<<grid, block, 0, stream>>>(output, qua_amax_ptr, input, size, n);
}

template void invokeDequatizeVectorE4M3<float, 0>(float*               output,
                                                  float const*         qua_amax_ptr,
                                                  __nv_fp8_e4m3 const* input,
                                                  uint32_t             size,
                                                  uint32_t             n,
                                                  cudaStream_t         stream);
template void invokeDequatizeVectorE4M3<half, 0>(half*                output,
                                                 float const*         qua_amax_ptr,
                                                 __nv_fp8_e4m3 const* input,
                                                 uint32_t             size,
                                                 uint32_t             n,
                                                 cudaStream_t         stream);
template void invokeDequatizeVectorE4M3<__nv_bfloat16, 0>(__nv_bfloat16*       output,
                                                          float const*         qua_amax_ptr,
                                                          __nv_fp8_e4m3 const* input,
                                                          uint32_t             size,
                                                          uint32_t             n,
                                                          cudaStream_t         stream);

template void invokeDequatizeVectorE4M3<float, 1>(float*               output,
                                                  float const*         qua_amax_ptr,
                                                  __nv_fp8_e4m3 const* input,
                                                  uint32_t             size,
                                                  uint32_t             n,
                                                  cudaStream_t         stream);
template void invokeDequatizeVectorE4M3<half, 1>(half*                output,
                                                 float const*         qua_amax_ptr,
                                                 __nv_fp8_e4m3 const* input,
                                                 uint32_t             size,
                                                 uint32_t             n,
                                                 cudaStream_t         stream);
template void invokeDequatizeVectorE4M3<__nv_bfloat16, 1>(__nv_bfloat16*       output,
                                                          float const*         qua_amax_ptr,
                                                          __nv_fp8_e4m3 const* input,
                                                          uint32_t             size,
                                                          uint32_t             n,
                                                          cudaStream_t         stream);

// IDEA: bfloat162 computation ?
template<typename T1, typename T2, int QUANTIZE_MODE, int PACKED_SIZE>
__global__ void LayerNorm(FP8LayerNormParam<T1, T2> param)
{
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    float local_sum     = 0.0f;
    using PackedInType  = typename PackType<T2, PACKED_SIZE>::type;
    using PackedOutType = typename PackType<T1, PACKED_SIZE>::type;
    float local_outs[PACKED_SIZE];

    float input_scalar  = 1.0f;
    float output_scalar = 1.0f;
    if (QUANTIZE_MODE == 0) {
        // per channel
        input_scalar = __ldg(param.input_deq_ptr + threadIdx.x);
        // output_scalar = __ldg(param.output_qua_ptr + threadIdx.x);
        output_scalar = __ldg(param.output_qua_ptr);  // must per tensor because it is quantize of input tensor of GEMM
    }
    else if (QUANTIZE_MODE == 1) {
        // For per tensor quantization, assume x = input, s = input_scalar, x' = x * s
        // Then Norm(x') = E[x'] / sqrt(V[x']).
        // Because E[x'] = E[sx] = sE[x], V[X'] = V[sx] = s^2 * V[x]
        // E[x'] / sqrt(V[x']) = (sE[x]) / sqrt(s^2 V[x]) = (sE[x]) / (s sqrt(V[x]))
        // = E[x] / sqrt(V[x]) = Norm(x)
        // So, we can skip the input_scalar to prevent the useless computation cost and memory
        // cost. But suggest to add to flag to open/close to prevent issue due to precision.

        // Besides, we can consdier to multiply the output_scalar into gamma and beta
        // when loading the weight to prevent the additional computation coat and memory
        // cost.

        // input_scalar = __ldg(param.input_deq_ptr);
        input_scalar  = 1.0f;  // We can skip the input scalar by above proof
        output_scalar = __ldg(param.output_qua_ptr);
    }

    for (int j = 0; threadIdx.x * PACKED_SIZE + j * PACKED_SIZE * blockDim.x < param.n; j++) {
        const int    offset    = j * PACKED_SIZE * blockDim.x;
        PackedInType packed_in = reinterpret_cast<const PackedInType*>(
            &param.input[blockIdx.x * param.n + offset + threadIdx.x * PACKED_SIZE])[0];
#pragma unroll
        for (int packed_i = 0; packed_i < PACKED_SIZE; packed_i++) {
            local_outs[packed_i] = (float)(packed_in.array[packed_i]) * input_scalar;
            local_sum += local_outs[packed_i];
        }
    }

    mean = blockReduceSum(local_sum);

    if (threadIdx.x == 0)
        s_mean = mean / param.n;
    __syncthreads();

    float local_var_sum = 0.0f;
    for (int j = 0; threadIdx.x * PACKED_SIZE + j * PACKED_SIZE * blockDim.x < param.n; j++) {
#pragma unroll
        for (int packed_i = 0; packed_i < PACKED_SIZE; packed_i += 1) {
            local_var_sum += (local_outs[packed_i] - s_mean) * (local_outs[packed_i] - s_mean);
        }
    }

    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0)
        s_variance = rsqrtf(variance / param.n + 1e-6);
    __syncthreads();

    for (int j = 0; threadIdx.x * PACKED_SIZE + j * PACKED_SIZE * blockDim.x < param.n; j++) {
        const int     offset = j * PACKED_SIZE * blockDim.x;
        PackedOutType packed_out;
        PackedInType  packed_gamma =
            reinterpret_cast<const PackedInType*>(&param.gamma[threadIdx.x * PACKED_SIZE + offset])[0];
        PackedInType packed_beta =
            reinterpret_cast<const PackedInType*>(&param.beta[threadIdx.x * PACKED_SIZE + offset])[0];
#pragma unroll
        for (int packed_i = 0; packed_i < PACKED_SIZE; packed_i += 1) {
            packed_out.array[packed_i] =
                (T1)((((local_outs[packed_i] - s_mean) * s_variance * (float)(packed_gamma.array[packed_i])
                       + (float)packed_beta.array[packed_i]))
                     * output_scalar);
        }
        reinterpret_cast<PackedOutType*>(
            &param.normed_output[blockIdx.x * param.n + offset + threadIdx.x * PACKED_SIZE])[0] = packed_out;
    }
}

template<typename T1, typename T2, int QUANTIZE_MODE>
__global__ void LayerNormE4M3x4(FP8LayerNormParam<T1, T2> param)
{
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    __nv_bfloat162* input1     = (__nv_bfloat162*)(param.input + blockIdx.x * param.n + 4 * threadIdx.x);
    __nv_bfloat162* input2     = (__nv_bfloat162*)(param.input + blockIdx.x * param.n + 4 * threadIdx.x + 2);
    __nv_bfloat162  local_out1 = __nv_bfloat162(*input1);
    __nv_bfloat162  local_out2 = __nv_bfloat162(*input2);
    // float input_scalar;
    // float output_scalar;

    if (QUANTIZE_MODE == 0) {
        // For per channel quantization.
        local_out1.x = local_out1.x * (__nv_bfloat16)__ldg(param.input_deq_ptr + threadIdx.x * 4 + 0);
        local_out1.y = local_out1.y * (__nv_bfloat16)__ldg(param.input_deq_ptr + threadIdx.x * 4 + 1);
        local_out2.x = local_out2.x * (__nv_bfloat16)__ldg(param.input_deq_ptr + threadIdx.x * 4 + 2);
        local_out2.y = local_out2.y * (__nv_bfloat16)__ldg(param.input_deq_ptr + threadIdx.x * 4 + 3);
    }
    else if (QUANTIZE_MODE == 1) {
        // For per tensor quantization, assume x = input, s = input_scalar, x' = x * s
        // Then Norm(x') = E[x'] / sqrt(V[x']).
        // Because E[x'] = E[sx] = sE[x], V[X'] = V[sx] = s^2 * V[x]
        // E[x'] / sqrt(V[x']) = (sE[x]) / sqrt(s^2 V[x]) = (sE[x]) / (s sqrt(V[x]))
        // = E[x] / sqrt(V[x]) = Norm(x)
        // So, we can skip the input_scalar to prevent the useless computation cost and memory
        // cost. But suggest to add to flag to open/close to prevent issue due to precision.

        // Besides, we can consdier to multiply the output_scalar into gamma and beta
        // when loading the weight to prevent the additional computation coat and memory
        // cost.

        // __nv_bfloat16 input_scalar = __ldg(param.input_deq_ptr);
        // __nv_bfloat16 input_scalar = 1.0f; // We can skip the input scalar by above proof
        // local_out1.x = local_out1.x * input_scalar;
        // local_out1.y = local_out1.y * input_scalar;
        // local_out2.x = local_out2.x * input_scalar;
        // local_out2.y = local_out2.y * input_scalar;
    }

    float local_sum = 0.0f;
    local_sum       = (float)(local_out1.x + local_out1.y + local_out2.x + local_out2.y);

    mean = blockReduceSum(local_sum);

    if (threadIdx.x == 0)
        s_mean = mean / param.n;
    __syncthreads();

    float local_var_sum = 0.0f;
    float diff1         = (float)local_out1.x - s_mean;
    float diff2         = (float)local_out1.y - s_mean;
    float diff3         = (float)local_out2.y - s_mean;
    float diff4         = (float)local_out2.y - s_mean;
    local_var_sum += diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4;
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0)
        s_variance = rsqrtf(variance / param.n + 1e-6);
    __syncthreads();

    float output_scalar[4];
    if (QUANTIZE_MODE == 0 && false) {  // must per tensor because it is quantize of input tensor of GEMM
        // For per channel quantization.
        output_scalar[0] = (float)__ldg(param.output_qua_ptr + threadIdx.x * 4 + 0);
        output_scalar[1] = (float)__ldg(param.output_qua_ptr + threadIdx.x * 4 + 1);
        output_scalar[2] = (float)__ldg(param.output_qua_ptr + threadIdx.x * 4 + 2);
        output_scalar[3] = (float)__ldg(param.output_qua_ptr + threadIdx.x * 4 + 3);
    }
    else if (QUANTIZE_MODE == 1 || true) {
        output_scalar[0] = (float)__ldg(param.output_qua_ptr);
        output_scalar[1] = (float)__ldg(param.output_qua_ptr);
        output_scalar[2] = (float)__ldg(param.output_qua_ptr);
        output_scalar[3] = (float)__ldg(param.output_qua_ptr);
    }

    __nv_bfloat162 result1;
    __nv_bfloat162 result2;
    result1.x = (__nv_bfloat16)((((float)local_out1.x - s_mean) * s_variance * (float)param.gamma[threadIdx.x * 4 + 0]
                                 + (float)param.beta[threadIdx.x * 4 + 0])
                                * output_scalar[0]);
    result1.y = (__nv_bfloat16)((((float)local_out1.y - s_mean) * s_variance * (float)param.gamma[threadIdx.x * 4 + 1]
                                 + (float)param.beta[threadIdx.x * 4 + 1])
                                * output_scalar[1]);
    result2.x = (__nv_bfloat16)((((float)local_out2.x - s_mean) * s_variance * (float)param.gamma[threadIdx.x * 4 + 2]
                                 + (float)param.beta[threadIdx.x * 4 + 2])
                                * output_scalar[2]);
    result2.y = (__nv_bfloat16)((((float)local_out2.y - s_mean) * s_variance * (float)param.gamma[threadIdx.x * 4 + 3]
                                 + (float)param.beta[threadIdx.x * 4 + 3])
                                * output_scalar[3]);

    __nv_fp8x4_e4m3  output_val = __nv_fp8x4_e4m3(result1, result2);
    __nv_fp8x4_e4m3* output_ptr = (__nv_fp8x4_e4m3*)(param.normed_output + blockIdx.x * param.n);
    output_ptr[threadIdx.x]     = output_val;
}

#define LN_KERNEL(PACKED_SIZE)                                                                                         \
    dim3 grid(param.m);                                                                                                \
    dim3 block(min(param.n / PACKED_SIZE, 1024));                                                                      \
    LayerNorm<T1, T2, QUANTIZE_MODE, PACKED_SIZE><<<grid, block, 0, param.stream>>>(param);

template<typename T1, typename T2, int QUANTIZE_MODE>
void invokeFP8LayerNorm(FP8LayerNormParam<T1, T2> param)
{
    assert(param.n % 2 == 0);
    if (param.n % 8 == 0) {
        LN_KERNEL(8);
    }
    else if (param.n % 4 == 0) {
        LN_KERNEL(4);
    }
    else if (param.n % 2 == 0) {
        LN_KERNEL(2);
    }
}

// template void invokeFP8LayerNorm<__nv_fp8_e4m3, float, 0>(FP8LayerNormParam<__nv_fp8_e4m3, float> param);
// template void invokeFP8LayerNorm<float, float, 0>(FP8LayerNormParam<float, float> param);
// template void invokeFP8LayerNorm<half, half, 0>(FP8LayerNormParam<half, half> param);
// template void invokeFP8LayerNorm<__nv_fp8_e4m3, half, 0>(FP8LayerNormParam<__nv_fp8_e4m3, half> param);
template void
invokeFP8LayerNorm<__nv_fp8_e4m3, __nv_bfloat16, 0>(FP8LayerNormParam<__nv_fp8_e4m3, __nv_bfloat16> param);

// template void invokeFP8LayerNorm<__nv_fp8_e4m3, float, 1>(FP8LayerNormParam<__nv_fp8_e4m3, float> param);
// template void invokeFP8LayerNorm<float, float, 1>(FP8LayerNormParam<float, float> param);
// template void invokeFP8LayerNorm<half, half, 1>(FP8LayerNormParam<half, half> param);
// template void invokeFP8LayerNorm<__nv_fp8_e4m3, half, 1>(FP8LayerNormParam<__nv_fp8_e4m3, half> param);
// template void
// invokeFP8LayerNorm<__nv_fp8_e4m3, __nv_bfloat16, 1>(FP8LayerNormParam<__nv_fp8_e4m3, __nv_bfloat16> param);

template<typename T1, typename T2>
__global__ void generalFP8IOPostLayerNorm(T1*       normed_output,
                                          const T1* input,
                                          const T2* __restrict gamma,
                                          const T2* __restrict beta,
                                          const float* input_scalar,
                                          const float* output_scalar,
                                          int          m,
                                          int          n)
{
    int tid = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    float local_sum    = 0.0f;
    float local_sum_sq = 0.0f;

    for (int i = tid; i < n; i += blockDim.x) {
        // float local_out = (float)(__ldg(&input[blockIdx.x * n + i]));
        float local_out = (float)(input[blockIdx.x * n + i]);
        local_sum += local_out;
        local_sum_sq += local_out * local_out;
    }
    __syncthreads();  // TODO check where should we put sync

    mean = blockReduceSum(local_sum);
    __syncthreads();  // TODO check where should we put sync
    variance = blockReduceSum(local_sum_sq);
    __syncthreads();  // TODO check where should we put sync

    if (threadIdx.x == 0) {
        s_mean     = mean / n;
        s_variance = rsqrtf((variance / n) - (s_mean * s_mean) + 1e-6);
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        // float beta_val = (beta == nullptr) ? 0.0f : (float)(__ldg(&beta[i]));
        float beta_val = (beta == nullptr) ? 0.0f : (float)(beta[i]);
        normed_output[blockIdx.x * n + i] =
            (T1)(((((float)input[blockIdx.x * n + i] - s_mean) * s_variance) * (float)(gamma[i]) + beta_val)
                 * (float)(__ldg(output_scalar)));
        // (T1)((((float)output[blockIdx.x * n + i] - s_mean) * s_variance) * (float)(__ldg(&gamma[i])) + beta_val);
    }
}

template<typename T1, typename T2, int QUANTIZE_MODE>
void invokeGeneralFP8IOPostLayerNorm(GeneralFP8IOPostLayerNormParam<T1, T2> param)
{
    dim3 grid(param.m);
    dim3 block(min(param.n, 1024));

    /* For general cases, n is equal to hidden_units, e.g., 512/1024.
    Since we have warp shuffle inside the code, block.x % 32 should be 0.
    */

    if (param.n % 32 != 0) {
        block.x = 1024;
    }

    /* should pay attention to the rsqrt precision*/
    generalFP8IOPostLayerNorm<T1, T2><<<grid, block, 0, param.stream>>>(param.normed_output,
                                                                        param.input,
                                                                        param.gamma,
                                                                        param.beta,
                                                                        param.input_deq_ptr,
                                                                        param.output_qua_ptr,
                                                                        param.m,
                                                                        param.n);  // For gpt-3
}

template void invokeGeneralFP8IOPostLayerNorm<__nv_fp8_e4m3, __nv_bfloat16, 0>(
    GeneralFP8IOPostLayerNormParam<__nv_fp8_e4m3, __nv_bfloat16> param);

template void invokeGeneralFP8IOPostLayerNorm<__nv_fp8_e4m3, __nv_bfloat16, 1>(
    GeneralFP8IOPostLayerNormParam<__nv_fp8_e4m3, __nv_bfloat16> param);

template<typename T1, typename T2>
__global__ void generalFP8AddBiasResidualLayerNorm(const T2* __restrict input,
                                                   const T2* __restrict gamma,
                                                   const T2* __restrict beta,
                                                   const T2* __restrict bias,
                                                   T2*          output,
                                                   T1*          norm_output,
                                                   const float* input_scale,
                                                   const float* output_scale,
                                                   int          m,
                                                   int          n)
{
    int tid = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    float local_sum    = 0.0f;
    float local_sum_sq = 0.0f;

    float input_scale_val  = input_scale == nullptr ? 1.0f : __ldg(input_scale);
    float output_scale_val = output_scale == nullptr ? 1.0f : __ldg(output_scale);

    for (int i = tid; i < n; i += blockDim.x) {
        // float local_out = (float)(__ldg(&input[blockIdx.x * n + i]));
        float local_out = (float)(input[blockIdx.x * n + i]);
        local_out += (float)(output[blockIdx.x * n + i]) * input_scale_val;
        if (bias != nullptr) {
            // local_out += (float)(__ldg(&bias[i]));
            local_out += (float)(bias[i]);
        }
        output[blockIdx.x * n + i] = (T2)local_out;
        local_sum += local_out;
        local_sum_sq += local_out * local_out;
    }
    __syncthreads();  // TODO check where should we put sync

    mean = blockReduceSum(local_sum);
    __syncthreads();  // TODO check where should we put sync
    variance = blockReduceSum(local_sum_sq);
    __syncthreads();  // TODO check where should we put sync

    if (threadIdx.x == 0) {
        s_mean     = mean / n;
        s_variance = rsqrtf((variance / n) - (s_mean * s_mean) + 1e-6);
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        // float beta_val = (beta == nullptr) ? 0.0f : (float)(__ldg(&beta[i]));
        float beta_val = (beta == nullptr) ? 0.0f : (float)(beta[i]);
        norm_output[blockIdx.x * n + i] =
            (T1)(((((float)output[blockIdx.x * n + i] - s_mean) * s_variance) * (float)(gamma[i]) + beta_val)
                 * output_scale_val);
    }
}

// TODO: implement T2 = half2
template<>
__global__ void generalFP8AddBiasResidualLayerNorm(const __nv_bfloat162_2_xy* __restrict input,
                                                   const __nv_bfloat162_2_xy* __restrict gamma,
                                                   const __nv_bfloat162_2_xy* __restrict beta,
                                                   const __nv_bfloat162_2_xy* __restrict bias,
                                                   __nv_bfloat162_2_xy* output,
                                                   __nv_fp8x4_e4m3*     norm_output,
                                                   const float*         input_scale,
                                                   const float*         output_scale,
                                                   int                  m,
                                                   int                  n)
{
    using bf16_4 = __nv_bfloat162_2_xy;
    using bf16_2 = __nv_bfloat162;
    using fp8_4  = __nv_fp8x4_e4m3;
    int tid      = threadIdx.x;
    int real_n   = n * 4;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    float local_sum    = 0.0f;
    float local_sum_sq = 0.0f;

    bf16_2 input_scale_val = input_scale == nullptr ? cuda_cast<bf16_2>(1.0f) : cuda_cast<bf16_2>(__ldg(input_scale));
    bf16_2 output_scale_val =
        output_scale == nullptr ? cuda_cast<bf16_2>(1.0f) : cuda_cast<bf16_2>(__ldg(output_scale));

    for (int i = tid; i < n; i += blockDim.x) {
        // float local_out = (float)(__ldg(&input[blockIdx.x * n + i]));
        bf16_4 local_out = input[blockIdx.x * n + i];
        local_out.x      = hadd2(hmul2(output[blockIdx.x * n + i].x, input_scale_val), local_out.x);
        local_out.y      = hadd2(hmul2(output[blockIdx.x * n + i].y, input_scale_val), local_out.y);
        if (bias != nullptr) {
            // local_out += (float)(__ldg(&bias[i]));
            local_out.x = hadd2(local_out.x, bias[i].x);
            local_out.y = hadd2(local_out.y, bias[i].y);
        }
        output[blockIdx.x * n + i] = local_out;
        // NOTE: need float accum here, or low task accuracy (summarization)
        local_sum += (float)(local_out.x.x) + (float)(local_out.x.y) + (float)(local_out.y.x) + (float)(local_out.y.y);
        local_sum_sq += (float)local_out.x.x * (float)local_out.x.x + (float)local_out.y.x * (float)local_out.y.x
                        + (float)local_out.x.y * (float)local_out.x.y + (float)local_out.y.y * (float)local_out.y.y;
    }
    __syncthreads();  // TODO check where should we put sync

    mean = blockReduceSum(local_sum);
    __syncthreads();  // TODO check where should we put sync
    variance = blockReduceSum(local_sum_sq);
    __syncthreads();  // TODO check where should we put sync

    if (threadIdx.x == 0) {
        s_mean     = mean / real_n;
        s_variance = rsqrtf((variance / real_n) - (s_mean * s_mean) + 1e-6);
    }
    __syncthreads();

    bf16_2 s_mean_2 = cuda_cast<bf16_2>(s_mean);
    bf16_2 s_var_2  = cuda_cast<bf16_2>(s_variance);

    for (int i = tid; i < n; i += blockDim.x) {
        bf16_4 norm_output_val;
        norm_output_val.x =
            hmul2(hadd2(hmul2(hmul2(hsub2(output[blockIdx.x * n + i].x, s_mean_2), s_var_2), gamma[i].x), beta[i].x),
                  output_scale_val);
        norm_output_val.y =
            hmul2(hadd2(hmul2(hmul2(hsub2(output[blockIdx.x * n + i].y, s_mean_2), s_var_2), gamma[i].y), beta[i].y),
                  output_scale_val);
        norm_output[blockIdx.x * n + i] = fp8_4(norm_output_val.x, norm_output_val.y);
    }
}

// TODO: implement T2 = half2
template<>
__global__ void generalFP8AddBiasResidualLayerNorm(const __nv_bfloat162* __restrict input,
                                                   const __nv_bfloat162* __restrict gamma,
                                                   const __nv_bfloat162* __restrict beta,
                                                   const __nv_bfloat162* __restrict bias,
                                                   __nv_bfloat162*  output,
                                                   __nv_fp8x2_e4m3* norm_output,
                                                   const float*     input_scale,
                                                   const float*     output_scale,
                                                   int              m,
                                                   int              n)
{
    using bf16_2 = __nv_bfloat162;
    using fp8_2  = __nv_fp8x2_e4m3;
    int tid      = threadIdx.x;
    int real_n   = n * 2;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    float local_sum    = 0.0f;
    float local_sum_sq = 0.0f;

    bf16_2 input_scale_val = input_scale == nullptr ? cuda_cast<bf16_2>(1.0f) : cuda_cast<bf16_2>(__ldg(input_scale));
    bf16_2 output_scale_val =
        output_scale == nullptr ? cuda_cast<bf16_2>(1.0f) : cuda_cast<bf16_2>(__ldg(output_scale));

    for (int i = tid; i < n; i += blockDim.x) {
        // float local_out = (float)(__ldg(&input[blockIdx.x * n + i]));
        bf16_2 local_out = input[blockIdx.x * n + i];
        local_out        = hadd2(hmul2(output[blockIdx.x * n + i], input_scale_val), local_out);
        if (bias != nullptr) {
            // local_out += (float)(__ldg(&bias[i]));
            local_out = hadd2(local_out, bias[i]);
        }
        output[blockIdx.x * n + i] = local_out;
        local_sum += (float)(local_out.x) + (float)(local_out.y);
        local_sum_sq += (float)local_out.x * (float)local_out.x + (float)local_out.y * (float)local_out.y;
    }
    __syncthreads();  // TODO check where should we put sync

    mean = blockReduceSum(local_sum);
    __syncthreads();  // TODO check where should we put sync
    variance = blockReduceSum(local_sum_sq);
    __syncthreads();  // TODO check where should we put sync

    if (threadIdx.x == 0) {
        s_mean     = mean / real_n;
        s_variance = rsqrtf((variance / real_n) - (s_mean * s_mean) + 1e-6);
    }
    __syncthreads();

    bf16_2 s_mean_2 = cuda_cast<bf16_2>(s_mean);
    bf16_2 s_var_2  = cuda_cast<bf16_2>(s_variance);

    for (int i = tid; i < n; i += blockDim.x) {
        norm_output[blockIdx.x * n + i] =
            fp8_2(hmul2(hadd2(hmul2(hmul2(hsub2(output[blockIdx.x * n + i], s_mean_2), s_var_2), gamma[i]), beta[i]),
                        output_scale_val));
    }
}

template<typename T1, typename T2, int QUANTIZE_MODE>
void invokeGeneralFP8AddBiasResidualPreLayerNorm(GeneralFP8AddBiasResidualPreLayerNormParam<T1, T2> param)
{
    dim3 grid(param.m);

    /* For general cases, n is equal to hidden_units, e.g., 512/1024.
    Since we have warp shuffle inside the code, block.x % 32 should be 0.
    */
    // NOTE: only T2 = bfloat supported yet
    if (param.n % 4 == 0) {
        dim3 block(min(param.n / 4, 1024));
        generalFP8AddBiasResidualLayerNorm<<<grid, block, 0, param.stream>>>((const __nv_bfloat162_2_xy*)param.residual,
                                                                             (const __nv_bfloat162_2_xy*)param.gamma,
                                                                             (const __nv_bfloat162_2_xy*)param.beta,
                                                                             (const __nv_bfloat162_2_xy*)param.bias,
                                                                             (__nv_bfloat162_2_xy*)param.output,
                                                                             (__nv_fp8x4_e4m3*)param.normed_output,
                                                                             param.input_deq_ptr,
                                                                             param.output_qua_ptr,
                                                                             param.m,
                                                                             param.n / 4);  // For gpt-3
    }
    else if (param.n % 2 == 0) {
        dim3 block(min(param.n / 2, 1024));
        generalFP8AddBiasResidualLayerNorm<<<grid, block, 0, param.stream>>>((const __nv_bfloat162*)param.residual,
                                                                             (const __nv_bfloat162*)param.gamma,
                                                                             (const __nv_bfloat162*)param.beta,
                                                                             (const __nv_bfloat162*)param.bias,
                                                                             (__nv_bfloat162*)param.output,
                                                                             (__nv_fp8x2_e4m3*)param.normed_output,
                                                                             param.input_deq_ptr,
                                                                             param.output_qua_ptr,
                                                                             param.m,
                                                                             param.n / 2);  // For gpt-3
    }
    else {
        dim3 block(min(param.n, 1024));
        if (param.n % 32 != 0) {
            block.x = 1024;
        }
        // const int vec_n = 4 / sizeof(T2);
        // block.x = block.x / vec_n; // We don't need this setting because we don't have bfloat162 implementation now

        /* should pay attention to the rsqrt precision*/
        generalFP8AddBiasResidualLayerNorm<T1, T2><<<grid, block, 0, param.stream>>>(param.residual,
                                                                                     param.gamma,
                                                                                     param.beta,
                                                                                     param.bias,
                                                                                     param.output,
                                                                                     param.normed_output,
                                                                                     param.input_deq_ptr,
                                                                                     param.output_qua_ptr,
                                                                                     param.m,
                                                                                     param.n);  // For gpt-3
    }
    return;
}

template void invokeGeneralFP8AddBiasResidualPreLayerNorm<__nv_fp8_e4m3, float, 0>(
    GeneralFP8AddBiasResidualPreLayerNormParam<__nv_fp8_e4m3, float> param);
template void invokeGeneralFP8AddBiasResidualPreLayerNorm<float, float, 0>(
    GeneralFP8AddBiasResidualPreLayerNormParam<float, float> param);
template void invokeGeneralFP8AddBiasResidualPreLayerNorm<half, half, 0>(
    GeneralFP8AddBiasResidualPreLayerNormParam<half, half> param);
template void invokeGeneralFP8AddBiasResidualPreLayerNorm<__nv_fp8_e4m3, half, 0>(
    GeneralFP8AddBiasResidualPreLayerNormParam<__nv_fp8_e4m3, half> param);
template void invokeGeneralFP8AddBiasResidualPreLayerNorm<__nv_fp8_e4m3, __nv_bfloat16, 0>(
    GeneralFP8AddBiasResidualPreLayerNormParam<__nv_fp8_e4m3, __nv_bfloat16> param);

template void invokeGeneralFP8AddBiasResidualPreLayerNorm<__nv_fp8_e4m3, float, 1>(
    GeneralFP8AddBiasResidualPreLayerNormParam<__nv_fp8_e4m3, float> param);
template void invokeGeneralFP8AddBiasResidualPreLayerNorm<float, float, 1>(
    GeneralFP8AddBiasResidualPreLayerNormParam<float, float> param);
template void invokeGeneralFP8AddBiasResidualPreLayerNorm<half, half, 1>(
    GeneralFP8AddBiasResidualPreLayerNormParam<half, half> param);
template void invokeGeneralFP8AddBiasResidualPreLayerNorm<__nv_fp8_e4m3, half, 1>(
    GeneralFP8AddBiasResidualPreLayerNormParam<__nv_fp8_e4m3, half> param);
template void invokeGeneralFP8AddBiasResidualPreLayerNorm<__nv_fp8_e4m3, __nv_bfloat16, 1>(
    GeneralFP8AddBiasResidualPreLayerNormParam<__nv_fp8_e4m3, __nv_bfloat16> param);

template<typename T1, typename T2, int QUANTIZE_MODE>
__global__ void generalFP8IOAddBiasResidualPostLayerNormV1(GeneralFP8IOAddBiasResidualPostLayerNormParam<T1, T2> param)
{
    int tid = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    // float mean = 0.0f;
    // float variance = 0.0f;

    float local_sum    = 0.0f;
    float local_sum_sq = 0.0f;

    float input_scalar;
    float output_scalar;
    if (QUANTIZE_MODE == 0) {
        // per channel
        input_scalar  = __ldg(param.input_scale + threadIdx.x);
        output_scalar = __ldg(param.output_scale);  // must per tensor because it is quantize of input tensor of GEMM
    }
    else if (QUANTIZE_MODE == 1) {
        input_scalar  = __ldg(param.input_scale);
        output_scalar = __ldg(param.output_scale);
    }
    else if (QUANTIZE_MODE == QUANTIZE_MODE::PER_CHANNEL_WEIGHT_PER_TENSOR_ACT) {
        input_scalar = __ldg(param.input_scale) * __ldg(param.input_scale_2 + threadIdx.x)
                       * (param.input_scale_2_min == nullptr ? 1.0f : ldg(param.input_scale_2_min));
        output_scalar = __ldg(param.output_scale);
    }

    for (int i = tid; i < param.n; i += blockDim.x) {
        // float local_out = (float)(__ldg(&param.input[blockIdx.x * param.n + i])) * input_scalar;
        float local_out = (float)(param.input[blockIdx.x * param.n + i]) * input_scalar;
        local_out       = local_out + (float)(param.residual[blockIdx.x * param.n + i]) * __ldg(param.residual_scale);
        if (param.bias != nullptr) {
            // local_out += (float)(__ldg(&bias[i]));
            local_out += (float)(param.bias[i]);
        }

        param.normed_output[blockIdx.x * param.n + i] = (T1)local_out;  // TODO This conversion has bug
        local_sum += local_out;
        local_sum_sq += local_out * local_out;
    }
    __syncthreads();  // TODO check where should we put sync

    float sums[2];
    sums[0] = local_sum;
    sums[1] = local_sum_sq;
    blockReduceSumV2<float, 2>(sums);

    if (threadIdx.x == 0) {
        s_mean     = sums[0] / param.n;
        s_variance = rsqrtf(sums[1] / param.n - s_mean * s_mean + 1e-12f);
    }
    __syncthreads();

    for (int i = tid; i < param.n; i += blockDim.x) {
        // float beta_val = (beta == nullptr) ? 0.0f : (float)(__ldg(&beta[i]));
        float beta_val = (param.beta == nullptr) ? 0.0f : (float)(param.beta[i]);
        param.normed_output[blockIdx.x * param.n + i] =
            (T1)(((((float)param.normed_output[blockIdx.x * param.n + i] - s_mean) * s_variance)
                      * (float)(param.gamma[i])
                  + beta_val)
                 * output_scalar);
    }
}

template<typename T1, typename T2>
__global__ void generalFP8IOAddBiasResidualPostLayerNormV2(GeneralFP8IOAddBiasResidualPostLayerNormParam<T1, T2> param)
{
    using T1_4 = __nv_fp8x4_e4m3;
    using T2_2 = typename TypeConverter<T2>::Type;
    __shared__ float s_mean;
    __shared__ float s_variance;

    float local_sum    = 0.0f;
    float local_sum_sq = 0.0f;

    T2_2 input_scalar    = cuda_cast<T2_2>(__ldg(param.input_scale));
    T2_2 output_scalar   = cuda_cast<T2_2>(__ldg(param.output_scale));
    T2_2 residual_scalar = cuda_cast<T2_2>(__ldg(param.residual_scale));

    const int n = param.n / 4;

    T1_4* input_ptr         = (T1_4*)(param.input);
    T1_4* residual_ptr      = (T1_4*)(param.residual);
    T1_4* normed_output_ptr = (T1_4*)(param.normed_output);
    T2_2* bias_ptr          = (T2_2*)(param.bias);
    T2_2* gamma_ptr         = (T2_2*)(param.gamma);
    T2_2* beta_ptr          = (T2_2*)(param.beta);

    T2_2 local_outs[2];

    T2_2      val_0, val_1;
    const int index_0 = 2 * threadIdx.x;
    const int index_1 = index_0 + 1;

    fp8x4_e4m3_to_bfloat2(&val_0, &val_1, &input_ptr[blockIdx.x * n + threadIdx.x]);
    val_0         = hmul2(val_0, input_scalar);
    val_1         = hmul2(val_1, input_scalar);
    local_outs[0] = val_0;
    local_outs[1] = val_1;

    fp8x4_e4m3_to_bfloat2(&val_0, &val_1, &residual_ptr[blockIdx.x * n + threadIdx.x]);
    val_0 = hmul2(val_0, residual_scalar);
    val_1 = hmul2(val_1, residual_scalar);
    local_outs[0] += val_0;
    local_outs[1] += val_1;

    if (bias_ptr != nullptr) {
        local_outs[0] = hadd2(local_outs[0], bias_ptr[index_0]);
        local_outs[1] = hadd2(local_outs[1], bias_ptr[index_1]);
    }

    local_sum += (float)(local_outs[0].x + local_outs[0].y + local_outs[1].x + local_outs[1].y);
    local_sum_sq += (float)local_outs[0].x * (float)local_outs[0].x + (float)local_outs[0].y * (float)local_outs[0].y
                    + (float)local_outs[1].x * (float)local_outs[1].x + (float)local_outs[1].y * (float)local_outs[1].y;

    __syncthreads();  // TODO check where should we put sync

    float sums[2];
    sums[0] = local_sum;
    sums[1] = local_sum_sq;
    blockReduceSumV2<float, 2>(sums);

    if (threadIdx.x == 0) {
        s_mean     = sums[0] / param.n;
        s_variance = rsqrtf(sums[1] / param.n - s_mean * s_mean + 1e-12f);
    }
    __syncthreads();

    T2_2 s_mean_2 = cuda_cast<T2_2>(s_mean);
    T2_2 s_var_2  = cuda_cast<T2_2>(s_variance);

    // {[(x - mean) * var * gamma] + beta} * output_scale
    local_outs[0] =
        hmul2(hadd2(hmul2(hmul2(hsub2(local_outs[0], s_mean_2), s_var_2), gamma_ptr[index_0]), beta_ptr[index_0]),
              output_scalar);
    local_outs[1] =
        hmul2(hadd2(hmul2(hmul2(hsub2(local_outs[1], s_mean_2), s_var_2), gamma_ptr[index_1]), beta_ptr[index_1]),
              output_scalar);

    normed_output_ptr[blockIdx.x * n + threadIdx.x] = T1_4(local_outs[0], local_outs[1]);
}

template<typename T1, typename T2, int ELEMENT_PER_THREAD, int WARP_NUM>
__global__ void generalFP8IOAddBiasResidualPostLayerNormV3(GeneralFP8IOAddBiasResidualPostLayerNormParam<T1, T2> param)
{
    // Each warp handle one row. So, we can save the cost of sync of block.
    // But when param.m is small, the launched blocks are too small the performance is worse
    // than V2.
    using T1_4 = __nv_fp8x4_e4m3;
    using T2_2 = typename TypeConverter<T2>::Type;

    float local_sum    = 0.0f;
    float local_sum_sq = 0.0f;

    T2_2 input_scalar    = cuda_cast<T2_2>(__ldg(param.input_scale));
    T2_2 output_scalar   = cuda_cast<T2_2>(__ldg(param.output_scale));
    T2_2 residual_scalar = cuda_cast<T2_2>(__ldg(param.residual_scale));

    const int n = param.n / 4;

    T1_4* input_ptr         = (T1_4*)(param.input);
    T1_4* residual_ptr      = (T1_4*)(param.residual);
    T1_4* normed_output_ptr = (T1_4*)(param.normed_output);
    T2_2* bias_ptr          = (T2_2*)(param.bias);
    T2_2* gamma_ptr         = (T2_2*)(param.gamma);
    T2_2* beta_ptr          = (T2_2*)(param.beta);

    T2_2 local_outs[ELEMENT_PER_THREAD][2];

    const int row_id = blockIdx.x * blockDim.y + threadIdx.y;
    if (row_id > param.m) {
        return;
    }

    T2_2 val_0, val_1;

#pragma unroll
    for (int i = 0; i < ELEMENT_PER_THREAD; i++) {

        fp8x4_e4m3_to_bfloat2(&val_0, &val_1, &input_ptr[row_id * n + i * blockDim.x + threadIdx.x]);
        val_0            = hmul2(val_0, input_scalar);
        val_1            = hmul2(val_1, input_scalar);
        local_outs[i][0] = val_0;
        local_outs[i][1] = val_1;

        fp8x4_e4m3_to_bfloat2(&val_0, &val_1, &residual_ptr[row_id * n + i * blockDim.x + threadIdx.x]);
        val_0 = hmul2(val_0, residual_scalar);
        val_1 = hmul2(val_1, residual_scalar);
        local_outs[i][0] += val_0;
        local_outs[i][1] += val_1;

        if (bias_ptr != nullptr) {
            local_outs[i][0] = hadd2(local_outs[i][0], __ldg(bias_ptr + 2 * (i * blockDim.x + threadIdx.x) + 0));
            local_outs[i][1] = hadd2(local_outs[i][1], __ldg(bias_ptr + 2 * (i * blockDim.x + threadIdx.x) + 1));
        }

        val_0 = hadd2(local_outs[i][0], local_outs[i][1]);
        local_sum += (float)(val_0.x + val_0.y);
        val_0 = hmul2(local_outs[i][0], local_outs[i][0]);
        val_1 = hmul2(local_outs[i][1], local_outs[i][1]);
        val_1 = hadd2(val_1, val_0);
        local_sum_sq += (float)(val_1.x + val_1.y);
    }

    float sums[2];
    sums[0] = local_sum;
    sums[1] = local_sum_sq;
    warpReduceSumV2<float, 2>(sums);

    sums[0] = sums[0] / (float)(param.n);
    sums[1] = rsqrtf(sums[1] / (float)(param.n) - sums[0] * sums[0] + 1e-12f);

    T2_2 s_mean_2 = cuda_cast<T2_2>(sums[0]);
    T2_2 s_var_2  = cuda_cast<T2_2>(sums[1]);
#pragma unroll
    for (int i = 0; i < ELEMENT_PER_THREAD; i++) {

        // {[(x - mean) * var * gamma] + beta} * output_scale
        local_outs[i][0] = hmul2(hadd2(hmul2(hmul2(hsub2(local_outs[i][0], s_mean_2), s_var_2),
                                             ldg(gamma_ptr + 2 * (i * blockDim.x + threadIdx.x) + 0)),
                                       ldg(beta_ptr + 2 * (i * blockDim.x + threadIdx.x) + 0)),
                                 output_scalar);
        local_outs[i][1] = hmul2(hadd2(hmul2(hmul2(hsub2(local_outs[i][1], s_mean_2), s_var_2),
                                             ldg(gamma_ptr + 2 * (i * blockDim.x + threadIdx.x) + 1)),
                                       ldg(beta_ptr + 2 * (i * blockDim.x + threadIdx.x) + 1)),
                                 output_scalar);
        normed_output_ptr[row_id * n + i * blockDim.x + threadIdx.x] = T1_4(local_outs[i][0], local_outs[i][1]);
    }
}

template<typename T1, typename T2, int QUANTIZE_MODE>
void invokeGeneralFP8IOAddBiasResidualPostLayerNorm(GeneralFP8IOAddBiasResidualPostLayerNormParam<T1, T2> param)
{
    dim3 grid(param.m);
    dim3 block(min(param.n, 1024));
    FT_CHECK(param.n <= 1024);
    if (param.n % 32 != 0) {
        block.x = 1024;
    }

    if (param.n % 4 == 0) {
        // TODO (bhsueh) check the condition here
        if (param.m > 1024) {
            block.x            = 32;
            const int WARP_NUM = 8;
            block.y            = WARP_NUM;
            grid.x             = (grid.x + (WARP_NUM - 1)) / WARP_NUM;
            if (param.n == 1024) {
                generalFP8IOAddBiasResidualPostLayerNormV3<T1, T2, 8, WARP_NUM>
                    <<<grid, block, 0, param.stream>>>(param);
            }
            else if (param.n == 768) {
                generalFP8IOAddBiasResidualPostLayerNormV3<T1, T2, 6, WARP_NUM>
                    <<<grid, block, 0, param.stream>>>(param);
            }
        }
        else {
            block.x /= 4;
            generalFP8IOAddBiasResidualPostLayerNormV2<T1, T2><<<grid, block, 0, param.stream>>>(param);
        }
    }
    else {
        generalFP8IOAddBiasResidualPostLayerNormV1<T1, T2, QUANTIZE_MODE><<<grid, block, 0, param.stream>>>(param);
    }
}

template void invokeGeneralFP8IOAddBiasResidualPostLayerNorm<__nv_fp8_e4m3, __nv_bfloat16, PER_CHANNEL>(
    GeneralFP8IOAddBiasResidualPostLayerNormParam<__nv_fp8_e4m3, __nv_bfloat16> param);

template void invokeGeneralFP8IOAddBiasResidualPostLayerNorm<__nv_fp8_e4m3, __nv_bfloat16, PER_TENSOR>(
    GeneralFP8IOAddBiasResidualPostLayerNormParam<__nv_fp8_e4m3, __nv_bfloat16> param);

template void
invokeGeneralFP8IOAddBiasResidualPostLayerNorm<__nv_fp8_e4m3, __nv_bfloat16, PER_CHANNEL_WEIGHT_PER_TENSOR_ACT>(
    GeneralFP8IOAddBiasResidualPostLayerNormParam<__nv_fp8_e4m3, __nv_bfloat16> param);

template<typename T1, typename T2, int ELEMENT_PER_THREAD>
__global__ void removePaddingEmbLookupLayerNormFP8Out(RemovePaddingEmbLookupLayerNormFP8OutParam<T1, T2> param)
{
    float local_outs[ELEMENT_PER_THREAD];

    __shared__ float s_mean;
    __shared__ float s_variance;

    float local_sum    = 0.0f;
    float local_sum_sq = 0.0f;

    float output_scalar = __ldg(param.output_scale);
    for (int i = 0; i < ELEMENT_PER_THREAD; i++) {
        int index      = blockDim.x * i + threadIdx.x;
        int padded_row = blockIdx.x + (param.padding_offset == nullptr ? 0 : param.padding_offset[blockIdx.x]);
        int position_id =
            param.position_ids == nullptr ? padded_row % param.max_seq_len : param.position_ids[padded_row];
        int   token_type_id = param.token_type_ids == nullptr ? 0 : param.token_type_ids[padded_row];
        int   input_id      = param.input_ids[padded_row];
        float local_out     = (input_id == 0 ? 0.0f : (float)param.word_embeddings[input_id * param.n + index])
                          + (float)param.position_embeddings[position_id * param.n + index]
                          + (float)param.token_type_embeddings[token_type_id * param.n + index];

        local_outs[i] = local_out;
        local_sum += local_out;
        local_sum_sq += local_out * local_out;
    }
    __syncthreads();  // TODO check where should we put sync

    float sums[2] = {0.0f};
    sums[0]       = local_sum;
    sums[1]       = local_sum_sq;
    blockReduceSumV2<float, 2>(sums);

    if (threadIdx.x == 0) {
        s_mean     = sums[0] / ((float)param.n);
        s_variance = rsqrtf(sums[1] / ((float)param.n) - s_mean * s_mean + 1e-12f);
    }
    __syncthreads();

    for (int i = 0; i < ELEMENT_PER_THREAD; i++) {
        int   index    = blockDim.x * i + threadIdx.x;
        float beta_val = (param.beta == nullptr) ? 0.0f : (float)(param.beta[index]);
        param.normed_output[blockIdx.x * param.n + index] =
            (T1)((((local_outs[i] - s_mean) * s_variance) * (float)(param.gamma[index]) + beta_val) * output_scalar);
    }
}

template<typename T1, typename T2>
__global__ void removePaddingEmbLookupLayerNormFP8OutV2(RemovePaddingEmbLookupLayerNormFP8OutParam<T1, T2> param)
{
    using T1_4 = __nv_fp8x4_e4m3;
    using T2_2 = typename TypeConverter<T2>::Type;

    T2_2 local_outs[2];

    __shared__ float s_mean;
    __shared__ float s_variance;

    float local_sum    = 0.0f;
    float local_sum_sq = 0.0f;

    T2_2 output_scalar = cuda_cast<T2_2>(__ldg(param.output_scale));

    T2_2* word_emb_ptr      = (T2_2*)param.word_embeddings;
    T2_2* pos_emb_ptr       = (T2_2*)param.position_embeddings;
    T2_2* type_emb_ptr      = (T2_2*)param.token_type_embeddings;
    T1_4* normed_output_ptr = (T1_4*)param.normed_output;
    T2_2* gamma_ptr         = (T2_2*)(param.gamma);
    T2_2* beta_ptr          = (T2_2*)(param.beta);

    int n_div_2 = param.n / 2;
    int n_div_4 = param.n / 4;

    const int index_0 = 2 * threadIdx.x;
    const int index_1 = index_0 + 1;

    int padded_row    = blockIdx.x + (param.padding_offset == nullptr ? 0 : param.padding_offset[blockIdx.x]);
    int position_id   = param.position_ids == nullptr ? padded_row % param.max_seq_len : param.position_ids[padded_row];
    int token_type_id = param.token_type_ids == nullptr ? 0 : param.token_type_ids[padded_row];
    int input_id      = param.input_ids[padded_row];

    local_outs[0] = (input_id == 0 ? cuda_cast<T2_2>(0.0f) : word_emb_ptr[input_id * n_div_2 + index_0])
                    + pos_emb_ptr[position_id * n_div_2 + index_0] + type_emb_ptr[token_type_id * n_div_2 + index_0];
    local_outs[1] = (input_id == 0 ? cuda_cast<T2_2>(0.0f) : word_emb_ptr[input_id * n_div_2 + index_1])
                    + pos_emb_ptr[position_id * n_div_2 + index_1] + type_emb_ptr[token_type_id * n_div_2 + index_1];

    local_sum += (float)(local_outs[0].x + local_outs[0].y + local_outs[1].x + local_outs[1].y);
    local_sum_sq += (float)local_outs[0].x * (float)local_outs[0].x + (float)local_outs[0].y * (float)local_outs[0].y
                    + (float)local_outs[1].x * (float)local_outs[1].x + (float)local_outs[1].y * (float)local_outs[1].y;
    __syncthreads();  // TODO check where should we put sync

    float sums[2] = {0.0f};
    sums[0]       = local_sum;
    sums[1]       = local_sum_sq;
    blockReduceSumV2<float, 2>(sums);

    if (threadIdx.x == 0) {
        s_mean     = sums[0] / ((float)param.n);
        s_variance = rsqrtf(sums[1] / ((float)param.n) - s_mean * s_mean + 1e-12f);
    }
    __syncthreads();

    T2_2 s_mean_2 = cuda_cast<T2_2>(s_mean);
    T2_2 s_var_2  = cuda_cast<T2_2>(s_variance);

    // {[(x - mean) * var * gamma] + beta} * output_scale
    local_outs[0] =
        hmul2(hadd2(hmul2(hmul2(hsub2(local_outs[0], s_mean_2), s_var_2), gamma_ptr[index_0]), beta_ptr[index_0]),
              output_scalar);
    local_outs[1] =
        hmul2(hadd2(hmul2(hmul2(hsub2(local_outs[1], s_mean_2), s_var_2), gamma_ptr[index_1]), beta_ptr[index_1]),
              output_scalar);

    normed_output_ptr[blockIdx.x * n_div_4 + threadIdx.x] = T1_4(local_outs[0], local_outs[1]);
}

template<typename T1, typename T2>
void invokeRemovePaddingEmbLookupLayerNormFP8Out(RemovePaddingEmbLookupLayerNormFP8OutParam<T1, T2> param)
{
    dim3 grid(param.m);
    dim3 block(min(param.n, 1024));
    FT_CHECK(param.n <= 1024);
    /* For general cases, n is equal to hidden_units, e.g., 512/1024.
    Since we have warp shuffle inside the code, block.x % 32 should be 0.
    */

    if (param.n % 32 != 0) {
        block.x = 1024;
    }

    if (param.n <= 1024) {
        if (block.x % 4 == 0) {
            block.x /= 4;
            removePaddingEmbLookupLayerNormFP8OutV2<T1, T2><<<grid, block, 0, param.stream>>>(param);
        }
        else {
            removePaddingEmbLookupLayerNormFP8Out<T1, T2, 1><<<grid, block, 0, param.stream>>>(param);
        }
    }
}

template void invokeRemovePaddingEmbLookupLayerNormFP8Out<__nv_fp8_e4m3, float>(
    RemovePaddingEmbLookupLayerNormFP8OutParam<__nv_fp8_e4m3, float> param);
template void invokeRemovePaddingEmbLookupLayerNormFP8Out<__nv_fp8_e4m3, __nv_bfloat16>(
    RemovePaddingEmbLookupLayerNormFP8OutParam<__nv_fp8_e4m3, __nv_bfloat16> param);

}  // namespace fastertransformer
