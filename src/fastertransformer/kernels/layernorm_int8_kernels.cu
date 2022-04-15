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

#include "src/fastertransformer/kernels/layernorm_int8_kernels.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"

namespace fastertransformer {

// input1/input2/output matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n)
//(grid, block) must be (m, n)
// for per_channel_quantization for weight
__global__ void add_bias_input_layernorm_COL32_int32I_DataTypeO(float* output,
                                                                const int32_t* input1,
                                                                const float* input2,
                                                                const float* bias,
                                                                const float* gamma,
                                                                const float* beta,
                                                                int m,
                                                                int n,
                                                                const float* weight_amax,
                                                                const float* input1_amax_ptr)
{
    const float input1_amax = __ldg(input1_amax_ptr);
    int col_start = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    float local_out;
    int outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31));

    float tmp = static_cast<float>(__ldg(input1 + outIdx)) * __ldg(weight_amax + col_start) * input1_amax
                * 0.000062f;  //(1/127/127);
    float inputTmp = __ldg(input2 + outIdx);

    local_out = tmp + inputTmp + __ldg(bias + col_start);

    mean = blockReduceSum<float>(local_out);
    if (threadIdx.x == 0) {
        s_mean = __fdividef(mean, n);
    }
    __syncthreads();

    local_out = local_out - s_mean;

    variance = blockReduceSum<float>(local_out * local_out);
    if (threadIdx.x == 0) {
        s_variance = __fdividef(variance, n) + 1e-6f;
        s_variance = rsqrtf(s_variance);
    }
    __syncthreads();

    local_out = (local_out * s_variance) * __ldg(gamma + col_start) + __ldg(beta + col_start);

    output[outIdx] = local_out;
}

__global__ void add_bias_input_layernorm_COL32_int32I_DataTypeO(half2* output,
                                                                const int2* input1,
                                                                const half2* input2,
                                                                const half2* bias,
                                                                const half2* gamma,
                                                                const half2* beta,
                                                                int m,
                                                                int n,
                                                                const float2* weight_amax,
                                                                const float* input1_amax_ptr)
{
    int col_start = threadIdx.x << 1;

    const float input1_amax = __ldg(input1_amax_ptr);

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    float2 local_out;
    int outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 1;

    const int2 input1Tmp = __ldg(input1 + outIdx);
    const float2 weightTmp = __ldg(weight_amax + threadIdx.x);

    float2 addTmp2;
    addTmp2.x = static_cast<float>(input1Tmp.x) * weightTmp.x * input1_amax * 0.000062f;  //(1/127/127);
    addTmp2.y = static_cast<float>(input1Tmp.y) * weightTmp.y * input1_amax * 0.000062f;  //(1/127/127);

    const half2 inputTmp = __ldg(input2 + outIdx);
    const half2 biasTmp = __ldg(bias + threadIdx.x);

    local_out = __half22float2(__hadd2(inputTmp, biasTmp));
    local_out.x = local_out.x + addTmp2.x;
    local_out.y = local_out.y + addTmp2.y;

    mean = blockReduceSum<float>(local_out.x + local_out.y);
    if (threadIdx.x == 0) {
        s_mean = __fdividef(mean, n);
    }
    __syncthreads();

    local_out.x = local_out.x - s_mean;
    local_out.y = local_out.y - s_mean;

    variance = blockReduceSum<float>(local_out.x * local_out.x + local_out.y * local_out.y);
    if (threadIdx.x == 0) {
        s_variance = __fdividef(variance, n) + 1e-6f;
        s_variance = rsqrtf(s_variance);
    }
    __syncthreads();

    float2 outputTmp;
    const half2 gammaTmp = __ldg(gamma + threadIdx.x);
    const half2 betaTmp = __ldg(beta + threadIdx.x);

    outputTmp.x = (local_out.x * s_variance) * static_cast<float>(gammaTmp.x) + static_cast<float>(betaTmp.x);
    outputTmp.y = (local_out.y * s_variance) * static_cast<float>(gammaTmp.y) + static_cast<float>(betaTmp.y);

    output[outIdx] = __float22half2_rn(outputTmp);
}

template<typename T>
void invokeAddBiasResidualLayerNormCol32(T* output,
                                         const int32_t* input1,
                                         const T* input2,
                                         const T* bias,
                                         const T* gamma,
                                         const T* beta,
                                         int m,
                                         int n,
                                         cudaStream_t stream,
                                         const float* weight_amax,
                                         const float* input1_amax_ptr)
{

    dim3 grid(m);
    dim3 block(n);
    if (sizeof(T) == sizeof(half)) {
        block.x /= 2;
        assert(block.x <= 1024);
        add_bias_input_layernorm_COL32_int32I_DataTypeO<<<grid, block, 0, stream>>>((half2*)output,
                                                                                    (const int2*)input1,
                                                                                    (const half2*)input2,
                                                                                    (const half2*)bias,
                                                                                    (const half2*)gamma,
                                                                                    (const half2*)beta,
                                                                                    m,
                                                                                    n,
                                                                                    (const float2*)weight_amax,
                                                                                    input1_amax_ptr);
    }
    else {
        assert(block.x <= 1024);
        add_bias_input_layernorm_COL32_int32I_DataTypeO<<<grid, block, 0, stream>>>((float*)output,
                                                                                    input1,
                                                                                    (const float*)input2,
                                                                                    (const float*)bias,
                                                                                    (const float*)gamma,
                                                                                    (const float*)beta,
                                                                                    m,
                                                                                    n,
                                                                                    weight_amax,
                                                                                    input1_amax_ptr);
    }
}

template void invokeAddBiasResidualLayerNormCol32(float* output,
                                                  const int32_t* input1,
                                                  const float* input2,
                                                  const float* bias,
                                                  const float* gamma,
                                                  const float* beta,
                                                  int m,
                                                  int n,
                                                  cudaStream_t stream,
                                                  const float* weight_amax,
                                                  const float* input1_amax_ptr);
template void invokeAddBiasResidualLayerNormCol32(half* output,
                                                  const int32_t* input1,
                                                  const half* input2,
                                                  const half* bias,
                                                  const half* gamma,
                                                  const half* beta,
                                                  int m,
                                                  int n,
                                                  cudaStream_t stream,
                                                  const float* weight_amax,
                                                  const float* input1_amax_ptr);

// input1/input2/out matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n)
//(grid, block) must be (m, n/4)
// using char4
template<typename T>
__global__ void add_bias_input_layernorm_COL32_int8IO(int8_t* output,
                                                      const int8_t* input1,
                                                      const int8_t* input2,
                                                      const T* bias,
                                                      const T* gamma,
                                                      const T* beta,
                                                      int m,
                                                      int n,
                                                      const float* input1_deQFactor_ptr,
                                                      const float* input2_deQFactor_ptr,
                                                      const float* output_scale_ptr)
{
    const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
    const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
    const float output_scale = __ldg(output_scale_ptr);
    int col_start = threadIdx.x << 2;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    float local_out[4];
    int outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 2;
    char4* outTmpPtr = (char4*)output;
    char4* input1TmpPtr = (char4*)input1;
    char4* input2TmpPtr = (char4*)input2;
    char4 input1Tmp = __ldg(input1TmpPtr + outIdx);
    char4 input2Tmp = __ldg(input2TmpPtr + outIdx);

    int col_start_tmp = col_start;
    local_out[0] = static_cast<float>(input2Tmp.x) * input2_deQFactor
                   + static_cast<float>(input1Tmp.x) * input1_deQFactor
                   + static_cast<float>(__ldg(bias + col_start_tmp));
    col_start_tmp = col_start_tmp + 1;
    local_out[1] = static_cast<float>(input2Tmp.y) * input2_deQFactor
                   + static_cast<float>(input1Tmp.y) * input1_deQFactor
                   + static_cast<float>(__ldg(bias + col_start_tmp));
    col_start_tmp = col_start_tmp + 1;
    local_out[2] = static_cast<float>(input2Tmp.z) * input2_deQFactor
                   + static_cast<float>(input1Tmp.z) * input1_deQFactor
                   + static_cast<float>(__ldg(bias + col_start_tmp));
    col_start_tmp = col_start_tmp + 1;
    local_out[3] = static_cast<float>(input2Tmp.w) * input2_deQFactor
                   + static_cast<float>(input1Tmp.w) * input1_deQFactor
                   + static_cast<float>(__ldg(bias + col_start_tmp));

    mean = blockReduceSum<float>(local_out[0] + local_out[1] + local_out[2] + local_out[3]);
    if (threadIdx.x == 0) {
        s_mean = mean * __fdividef(1.0f, n);
    }
    __syncthreads();

    local_out[0] = local_out[0] - s_mean;
    local_out[1] = local_out[1] - s_mean;
    local_out[2] = local_out[2] - s_mean;
    local_out[3] = local_out[3] - s_mean;
    variance = blockReduceSum<float>(local_out[0] * local_out[0] + local_out[1] * local_out[1]
                                     + local_out[2] * local_out[2] + local_out[3] * local_out[3]);
    if (threadIdx.x == 0) {
        s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
        s_variance = rsqrtf(s_variance);
    }
    __syncthreads();

    local_out[0] = (local_out[0] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                   + static_cast<float>(__ldg(beta + col_start));
    input2Tmp.x = float_to_int8_rn(local_out[0] * output_scale);

    col_start = col_start + 1;
    local_out[1] = (local_out[1] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                   + static_cast<float>(__ldg(beta + col_start));
    input2Tmp.y = float_to_int8_rn(local_out[1] * output_scale);

    col_start = col_start + 1;
    local_out[2] = (local_out[2] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                   + static_cast<float>(__ldg(beta + col_start));
    input2Tmp.z = float_to_int8_rn(local_out[2] * output_scale);

    col_start = col_start + 1;
    local_out[3] = (local_out[3] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                   + static_cast<float>(__ldg(beta + col_start));
    input2Tmp.w = float_to_int8_rn(local_out[3] * output_scale);

    outTmpPtr[outIdx] = input2Tmp;
}

template<>
__global__ void add_bias_input_layernorm_COL32_int8IO(int8_t* output,
                                                      const int8_t* input1,
                                                      const int8_t* input2,
                                                      const half2* bias,
                                                      const half2* gamma,
                                                      const half2* beta,
                                                      int m,
                                                      int n,
                                                      const float* input1_deQFactor_ptr,
                                                      const float* input2_deQFactor_ptr,
                                                      const float* output_scale_ptr)
{
    const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
    const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
    const float output_scale = __ldg(output_scale_ptr);
    int col_start = threadIdx.x << 2;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    float local_out[4];
    int outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 2;
    char4* outTmpPtr = (char4*)output;
    char4* input1TmpPtr = (char4*)input1;
    char4* input2TmpPtr = (char4*)input2;
    char4 input1Tmp = __ldg(input1TmpPtr + outIdx);
    char4 input2Tmp = __ldg(input2TmpPtr + outIdx);

    int col_start_tmp = col_start;
    half2 biasTmp = __ldg(bias + (col_start_tmp >> 1));
    local_out[0] = static_cast<float>(input2Tmp.x) * input2_deQFactor
                   + static_cast<float>(input1Tmp.x) * input1_deQFactor + static_cast<float>(biasTmp.x);
    col_start_tmp = col_start_tmp + 1;
    local_out[1] = static_cast<float>(input2Tmp.y) * input2_deQFactor
                   + static_cast<float>(input1Tmp.y) * input1_deQFactor + static_cast<float>(biasTmp.y);

    col_start_tmp = col_start_tmp + 1;
    biasTmp = __ldg(bias + (col_start_tmp >> 1));
    local_out[2] = static_cast<float>(input2Tmp.z) * input2_deQFactor
                   + static_cast<float>(input1Tmp.z) * input1_deQFactor + static_cast<float>(biasTmp.x);
    col_start_tmp = col_start_tmp + 1;
    local_out[3] = static_cast<float>(input2Tmp.w) * input2_deQFactor
                   + static_cast<float>(input1Tmp.w) * input1_deQFactor + static_cast<float>(biasTmp.y);

    mean = blockReduceSum<float>(local_out[0] + local_out[1] + local_out[2] + local_out[3]);
    if (threadIdx.x == 0) {
        s_mean = mean * __fdividef(1.0f, n);
    }
    __syncthreads();

    local_out[0] = local_out[0] - s_mean;
    local_out[1] = local_out[1] - s_mean;
    local_out[2] = local_out[2] - s_mean;
    local_out[3] = local_out[3] - s_mean;
    variance = blockReduceSum<float>(local_out[0] * local_out[0] + local_out[1] * local_out[1]
                                     + local_out[2] * local_out[2] + local_out[3] * local_out[3]);
    if (threadIdx.x == 0) {
        s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
        s_variance = rsqrtf(s_variance);
    }
    __syncthreads();

    col_start_tmp = col_start >> 1;
    biasTmp = __ldg(gamma + col_start_tmp);
    half2 betaTmp = __ldg(beta + col_start_tmp);

    local_out[0] = (local_out[0] * s_variance) * static_cast<float>(biasTmp.x) + static_cast<float>(betaTmp.x);
    input2Tmp.x = float_to_int8_rn(local_out[0] * output_scale);

    col_start = col_start + 1;
    local_out[1] = (local_out[1] * s_variance) * static_cast<float>(biasTmp.y) + static_cast<float>(betaTmp.y);
    input2Tmp.y = float_to_int8_rn(local_out[1] * output_scale);

    col_start = col_start + 1;
    col_start_tmp = col_start >> 1;
    biasTmp = __ldg(gamma + col_start_tmp);
    betaTmp = __ldg(beta + col_start_tmp);
    local_out[2] = (local_out[2] * s_variance) * static_cast<float>(biasTmp.x) + static_cast<float>(betaTmp.x);
    input2Tmp.z = float_to_int8_rn(local_out[2] * output_scale);

    col_start = col_start + 1;
    local_out[3] = (local_out[3] * s_variance) * static_cast<float>(biasTmp.y) + static_cast<float>(betaTmp.y);
    input2Tmp.w = float_to_int8_rn(local_out[3] * output_scale);

    outTmpPtr[outIdx] = input2Tmp;
}

template<typename T>
void invokeAddBiasResidualLayerNormCol32(int8_t* output,
                                         const int8_t* input1,
                                         const int8_t* input2,
                                         const T* bias,
                                         const T* gamma,
                                         const T* beta,
                                         int m,
                                         int n,
                                         cudaStream_t stream,
                                         const float* input1_deQFactor_ptr,
                                         const float* input2_deQFactor_ptr,
                                         const float* output_scale_ptr)
{
    dim3 grid(m);
    dim3 block(n / 4);
    assert(n <= 1024);
    if (sizeof(T) == sizeof(half)) {
        add_bias_input_layernorm_COL32_int8IO<<<grid, block, 0, stream>>>(output,
                                                                          input1,
                                                                          input2,
                                                                          (const half2*)bias,
                                                                          (const half2*)gamma,
                                                                          (const half2*)beta,
                                                                          m,
                                                                          n,
                                                                          input1_deQFactor_ptr,
                                                                          input2_deQFactor_ptr,
                                                                          output_scale_ptr);
    }
    else {
        add_bias_input_layernorm_COL32_int8IO<T><<<grid, block, 0, stream>>>(output,
                                                                             input1,
                                                                             input2,
                                                                             bias,
                                                                             gamma,
                                                                             beta,
                                                                             m,
                                                                             n,
                                                                             input1_deQFactor_ptr,
                                                                             input2_deQFactor_ptr,
                                                                             output_scale_ptr);
    }
}

template void invokeAddBiasResidualLayerNormCol32(int8_t* output,
                                                  const int8_t* input1,
                                                  const int8_t* input2,
                                                  const float* bias,
                                                  const float* gamma,
                                                  const float* beta,
                                                  int m,
                                                  int n,
                                                  cudaStream_t stream,
                                                  const float* input1_deQFactor_ptr,
                                                  const float* input2_deQFactor_ptr,
                                                  const float* output_scale_ptr);

template void invokeAddBiasResidualLayerNormCol32(int8_t* output,
                                                  const int8_t* input1,
                                                  const int8_t* input2,
                                                  const half* bias,
                                                  const half* gamma,
                                                  const half* beta,
                                                  int m,
                                                  int n,
                                                  cudaStream_t stream,
                                                  const float* input1_deQFactor_ptr,
                                                  const float* input2_deQFactor_ptr,
                                                  const float* output_scale_ptr);

// input1/input2/out matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n)
//(grid, block) must be (m, n)
template<typename T>
__global__ void add_bias_input_layernorm_COL32_int8I_DataTypeO(T* output,
                                                               const int8_t* input1,
                                                               const int8_t* input2,
                                                               const T* bias,
                                                               const T* gamma,
                                                               const T* beta,
                                                               int m,
                                                               int n,
                                                               const float* input1_deQFactor_ptr,
                                                               const float* input2_deQFactor_ptr)
{
    const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
    const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
    int col_start = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    float local_out;
    int idx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31));

    local_out = static_cast<float>(__ldg(input2 + idx)) * input2_deQFactor
                + static_cast<float>(__ldg(input1 + idx)) * input1_deQFactor
                + static_cast<float>(__ldg(bias + col_start));

    mean = blockReduceSum<float>(local_out);
    if (threadIdx.x == 0) {
        s_mean = mean * __fdividef(1.0f, n);
    }
    __syncthreads();

    local_out = local_out - s_mean;

    variance = blockReduceSum<float>(local_out * local_out);

    if (threadIdx.x == 0) {
        s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
        s_variance = rsqrtf(s_variance);
    }
    __syncthreads();

    local_out = (local_out * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                + static_cast<float>(__ldg(beta + col_start));

    output[idx] = local_out;
}

// input1/input2/out matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n)
//(grid, block) must be (m, n/2)
template<>
__global__ void add_bias_input_layernorm_COL32_int8I_DataTypeO(half2* output,
                                                               const int8_t* input1,
                                                               const int8_t* input2,
                                                               const half2* bias,
                                                               const half2* gamma,
                                                               const half2* beta,
                                                               int m,
                                                               int n,
                                                               const float* input1_deQFactor_ptr,
                                                               const float* input2_deQFactor_ptr)
{
    const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
    const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
    int col_start = threadIdx.x << 1;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    float2 local_out;
    int idx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 1;

    const char2* input1_ptr2 = (const char2*)input1;
    const char2* input2_ptr2 = (const char2*)input2;
    char2 input_tmp1 = __ldg(input1_ptr2 + idx);
    char2 input_tmp2 = __ldg(input2_ptr2 + idx);

    half2 bias_tmp = __ldg(bias + threadIdx.x);

    local_out.x = static_cast<float>(input_tmp1.x) * input1_deQFactor
                  + static_cast<float>(input_tmp2.x) * input2_deQFactor + static_cast<float>(bias_tmp.x);

    local_out.y = static_cast<float>(input_tmp1.y) * input1_deQFactor
                  + static_cast<float>(input_tmp2.y) * input2_deQFactor + static_cast<float>(bias_tmp.y);

    mean = blockReduceSum<float>(local_out.x + local_out.y);
    if (threadIdx.x == 0) {
        s_mean = mean * __fdividef(1.0f, n);
    }
    __syncthreads();

    local_out.x = local_out.x - s_mean;

    local_out.y = local_out.y - s_mean;

    variance = blockReduceSum<float>(local_out.x * local_out.x + local_out.y * local_out.y);

    if (threadIdx.x == 0) {
        s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
        s_variance = rsqrtf(s_variance);
    }
    __syncthreads();

    half2 gamma_tmp = __ldg(gamma + threadIdx.x);
    half2 beta_tmp = __ldg(beta + threadIdx.x);

    local_out.x = (local_out.x * s_variance) * static_cast<float>(gamma_tmp.x) + static_cast<float>(beta_tmp.x);
    local_out.y = (local_out.y * s_variance) * static_cast<float>(gamma_tmp.y) + static_cast<float>(beta_tmp.y);

    bias_tmp.x = half(local_out.x);
    bias_tmp.y = half(local_out.y);

    output[idx] = bias_tmp;
}

template<typename T>
void invokeAddBiasResidualLayerNormCol32(T* output,
                                         const int8_t* input1,
                                         const int8_t* input2,
                                         const T* bias,
                                         const T* gamma,
                                         const T* beta,
                                         int m,
                                         int n,
                                         cudaStream_t stream,
                                         const float* input1_deQFactor_ptr,
                                         const float* input2_deQFactor_ptr)
{
    dim3 grid(m);
    dim3 block(n);
    if (sizeof(T) == sizeof(half)) {
        assert(n / 2 <= 1024 && n % 2 == 0);
        block.x = n / 2;
        add_bias_input_layernorm_COL32_int8I_DataTypeO<<<grid, block, 0, stream>>>((half2*)output,
                                                                                   input1,
                                                                                   input2,
                                                                                   (const half2*)bias,
                                                                                   (const half2*)gamma,
                                                                                   (const half2*)beta,
                                                                                   m,
                                                                                   n,
                                                                                   input1_deQFactor_ptr,
                                                                                   input2_deQFactor_ptr);
    }
    else {
        assert(n <= 1024);
        add_bias_input_layernorm_COL32_int8I_DataTypeO<T><<<grid, block, 0, stream>>>(
            output, input1, input2, bias, gamma, beta, m, n, input1_deQFactor_ptr, input2_deQFactor_ptr);
    }
}

template void invokeAddBiasResidualLayerNormCol32<float>(float* output,
                                                         const int8_t* input1,
                                                         const int8_t* input2,
                                                         const float* bias,
                                                         const float* gamma,
                                                         const float* beta,
                                                         int m,
                                                         int n,
                                                         cudaStream_t stream,
                                                         const float* input1_deQFactor_ptr,
                                                         const float* input2_deQFactor_ptr);

template void invokeAddBiasResidualLayerNormCol32<half>(half* output,
                                                        const int8_t* input1,
                                                        const int8_t* input2,
                                                        const half* bias,
                                                        const half* gamma,
                                                        const half* beta,
                                                        int m,
                                                        int n,
                                                        cudaStream_t stream,
                                                        const float* input1_deQFactor_ptr,
                                                        const float* input2_deQFactor_ptr);

// input1/input2/out matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n)
//(grid, block) must be (m, n/4)
// using char4
template<typename T>
__global__ void add_bias_input_layernorm_COL32_int8IO_noRes(int8_t* output,
                                                            int8_t* input1,
                                                            T* input2,
                                                            const T* bias,
                                                            const T* gamma,
                                                            const T* beta,
                                                            int m,
                                                            int n,
                                                            const float* input1_deQFactor_ptr,
                                                            const float* output_scale_ptr)
{
    const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
    const float output_scale = __ldg(output_scale_ptr);
    int col_start = threadIdx.x << 2;
    bool qual = (col_start < n);

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    float local_out[4];
    int outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 2;
    char4* input1TmpPtr = (char4*)input1;
    char4 input1Tmp;
    if (qual) {
        input1Tmp = __ldg(input1TmpPtr + outIdx);
        int col_start_tmp = col_start;
        local_out[0] = static_cast<float>(input1Tmp.x) * input1_deQFactor
                       + static_cast<float>(input2[(outIdx << 2) + 0]) + static_cast<float>(bias[col_start_tmp]);
        input2[(outIdx << 2) + 0] = local_out[0];

        col_start_tmp = col_start_tmp + 1;
        local_out[1] = static_cast<float>(input1Tmp.y) * input1_deQFactor
                       + static_cast<float>(input2[(outIdx << 2) + 1]) + static_cast<float>(bias[col_start_tmp]);
        input2[(outIdx << 2) + 1] = local_out[1];

        col_start_tmp = col_start_tmp + 1;
        local_out[2] = static_cast<float>(input1Tmp.z) * input1_deQFactor
                       + static_cast<float>(input2[(outIdx << 2) + 2]) + static_cast<float>(bias[col_start_tmp]);
        input2[(outIdx << 2) + 2] = local_out[2];

        col_start_tmp = col_start_tmp + 1;
        local_out[3] = static_cast<float>(input1Tmp.w) * input1_deQFactor
                       + static_cast<float>(input2[(outIdx << 2) + 3]) + static_cast<float>(bias[col_start_tmp]);
        input2[(outIdx << 2) + 3] = local_out[3];
    }

    mean = blockReduceSum<float>(qual ? local_out[0] + local_out[1] + local_out[2] + local_out[3] : 0.0f);
    if (threadIdx.x == 0) {
        s_mean = mean * __fdividef(1.0f, n);
    }
    __syncthreads();

    if (qual) {
        local_out[0] = local_out[0] - s_mean;
        local_out[1] = local_out[1] - s_mean;
        local_out[2] = local_out[2] - s_mean;
        local_out[3] = local_out[3] - s_mean;
    }
    variance = blockReduceSum<float>(qual ? local_out[0] * local_out[0] + local_out[1] * local_out[1]
                                                + local_out[2] * local_out[2] + local_out[3] * local_out[3] :
                                            0.0f);
    if (threadIdx.x == 0) {
        s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
        s_variance = rsqrtf(s_variance);
    }
    __syncthreads();

    char4 outputTmp;
    char4* outputTmpPtr = (char4*)output;
    if (qual) {
        local_out[0] = (local_out[0] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                       + static_cast<float>(__ldg(beta + col_start));
        outputTmp.x = float_to_int8_rn(local_out[0] * output_scale);

        col_start = col_start + 1;
        local_out[1] = (local_out[1] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                       + static_cast<float>(__ldg(beta + col_start));
        outputTmp.y = float_to_int8_rn(local_out[1] * output_scale);

        col_start = col_start + 1;
        local_out[2] = (local_out[2] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                       + static_cast<float>(__ldg(beta + col_start));
        outputTmp.z = float_to_int8_rn(local_out[2] * output_scale);

        col_start = col_start + 1;
        local_out[3] = (local_out[3] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                       + static_cast<float>(__ldg(beta + col_start));
        outputTmp.w = float_to_int8_rn(local_out[3] * output_scale);

        outputTmpPtr[outIdx] = outputTmp;
    }
}

template<>
__global__ void add_bias_input_layernorm_COL32_int8IO_noRes(int8_t* output,
                                                            int8_t* input1,
                                                            half2* input2,
                                                            const half2* bias,
                                                            const half2* gamma,
                                                            const half2* beta,
                                                            int m,
                                                            int n,
                                                            const float* input1_deQFactor_ptr,
                                                            const float* output_scale_ptr)
{
    const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
    const float output_scale = __ldg(output_scale_ptr);
    int col_start = threadIdx.x << 1;
    bool qual = (col_start < n);

    __shared__ float s_mean;
    __shared__ float s_variance;
    float sums[2] = {0.0f, 0.0f};
    // float mean = 0.0f;
    // float variance = 0.0f;

    float local_out[2];
    int outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 1;
    char2* input1TmpPtr = (char2*)input1;
    char2 input1Tmp;
    if (qual) {
        input1Tmp = input1TmpPtr[outIdx];
        half2 biasTmp = bias[threadIdx.x];
        half2 input2Tmp = input2[outIdx];
        local_out[0] = static_cast<float>(input1Tmp.x) * input1_deQFactor + static_cast<float>(input2Tmp.x)
                       + static_cast<float>(biasTmp.x);
        local_out[1] = static_cast<float>(input1Tmp.y) * input1_deQFactor + static_cast<float>(input2Tmp.y)
                       + static_cast<float>(biasTmp.y);

        input2Tmp.x = local_out[0];
        input2Tmp.y = local_out[1];
        input2[outIdx] = input2Tmp;
        for (int i = 0; i < 2; i++) {
            sums[0] += local_out[i];
            sums[1] += local_out[i] * local_out[i];
        }
    }

    blockReduceSumV2<float, 2>(sums);

    if (threadIdx.x == 0) {
        s_mean = sums[0] * __fdividef(1.0f, n);
        s_variance = rsqrtf(sums[1] * __fdividef(1.0f, n) - s_mean * s_mean + 1e-6);
    }
    __syncthreads();

    char2 outputTmp;
    char2* outputTmpPtr = (char2*)output;
    if (qual) {
        half2 gammaTmp = gamma[threadIdx.x];
        half2 betaTmp = beta[threadIdx.x];
        local_out[0] =
            (local_out[0] - s_mean) * s_variance * static_cast<float>(gammaTmp.x) + static_cast<float>(betaTmp.x);
        outputTmp.x = float_to_int8_rn(local_out[0] * output_scale);

        local_out[1] =
            (local_out[1] - s_mean) * s_variance * static_cast<float>(gammaTmp.y) + static_cast<float>(betaTmp.y);
        outputTmp.y = float_to_int8_rn(local_out[1] * output_scale);

        outputTmpPtr[outIdx] = outputTmp;
    }
}

template<typename T>
void invokeAddBiasResidualLayerNormCol32_noRes(int8_t* output,
                                               int8_t* input1,
                                               T* input2,
                                               const T* bias,
                                               const T* gamma,
                                               const T* beta,
                                               int m,
                                               int n,
                                               cudaStream_t stream,
                                               const float* input1_deQFactor_ptr,
                                               const float* output_scale_ptr)
{
    dim3 grid(m);
    int blockSize = (n / 4 + 31) / 32 * 32;
    dim3 block(blockSize);
    assert(blockSize <= 1024);
    if (sizeof(T) == sizeof(half)) {
        blockSize = (n / 2 + 31) / 32 * 32;
        assert(blockSize <= 1024);
        add_bias_input_layernorm_COL32_int8IO_noRes<<<grid, blockSize, 0, stream>>>(output,
                                                                                    input1,
                                                                                    (half2*)input2,
                                                                                    (const half2*)bias,
                                                                                    (const half2*)gamma,
                                                                                    (const half2*)beta,
                                                                                    m,
                                                                                    n,
                                                                                    input1_deQFactor_ptr,
                                                                                    output_scale_ptr);
    }
    else {
        add_bias_input_layernorm_COL32_int8IO_noRes<T><<<grid, block, 0, stream>>>(
            output, input1, input2, bias, gamma, beta, m, n, input1_deQFactor_ptr, output_scale_ptr);
    }
}

template void invokeAddBiasResidualLayerNormCol32_noRes(int8_t* output,
                                                        int8_t* input1,
                                                        float* input2,
                                                        const float* bias,
                                                        const float* gamma,
                                                        const float* beta,
                                                        int m,
                                                        int n,
                                                        cudaStream_t stream,
                                                        const float* input1_deQFactor_ptr,
                                                        const float* output_scale_ptr);

template void invokeAddBiasResidualLayerNormCol32_noRes(int8_t* output,
                                                        int8_t* input1,
                                                        half* input2,
                                                        const half* bias,
                                                        const half* gamma,
                                                        const half* beta,
                                                        int m,
                                                        int n,
                                                        cudaStream_t stream,
                                                        const float* input1_deQFactor_ptr,
                                                        const float* output_scale_ptr);

// input1/input2/out matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n)
//(grid, block) must be (m, n/4)
// using char4
template<typename T>
__global__ void add_bias_input_layernorm_COL32_int8IO_noRes(int8_t* output,
                                                            int32_t* input1,
                                                            T* input2,
                                                            const T* bias,
                                                            const T* gamma,
                                                            const T* beta,
                                                            int m,
                                                            int n,
                                                            const float* weight_amax,
                                                            const float* input1_amax_ptr,
                                                            const float* output_scale_ptr)
{
    const float input1_amax = __ldg(input1_amax_ptr);
    const float output_scale = __ldg(output_scale_ptr);
    int col_start = threadIdx.x << 2;
    bool qual = (col_start < n);

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    float local_out[4];
    int outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 2;
    int4* input1TmpPtr = (int4*)input1;
    int4 input1Tmp;
    if (qual) {
        input1Tmp = __ldg(input1TmpPtr + outIdx);
        int col_start_tmp = col_start;
        // NOTE: 0.000062f = 1 / 127/.0f / 127.0f
        local_out[0] = static_cast<float>(input1Tmp.x) * input1_amax * weight_amax[col_start_tmp] * 0.000062f
                       + static_cast<float>(input2[(outIdx << 2) + 0]) + static_cast<float>(bias[col_start_tmp]);
        input2[(outIdx << 2) + 0] = local_out[0];

        col_start_tmp = col_start_tmp + 1;
        local_out[1] = static_cast<float>(input1Tmp.y) * input1_amax * weight_amax[col_start_tmp] * 0.000062f
                       + static_cast<float>(input2[(outIdx << 2) + 1]) + static_cast<float>(bias[col_start_tmp]);
        input2[(outIdx << 2) + 1] = local_out[1];

        col_start_tmp = col_start_tmp + 1;
        local_out[2] = static_cast<float>(input1Tmp.z) * input1_amax * weight_amax[col_start_tmp] * 0.000062f
                       + static_cast<float>(input2[(outIdx << 2) + 2]) + static_cast<float>(bias[col_start_tmp]);
        input2[(outIdx << 2) + 2] = local_out[2];

        col_start_tmp = col_start_tmp + 1;
        local_out[3] = static_cast<float>(input1Tmp.w) * input1_amax * weight_amax[col_start_tmp] * 0.000062f
                       + static_cast<float>(input2[(outIdx << 2) + 3]) + static_cast<float>(bias[col_start_tmp]);
        input2[(outIdx << 2) + 3] = local_out[3];
    }

    mean = blockReduceSum<float>(qual ? local_out[0] + local_out[1] + local_out[2] + local_out[3] : 0.0f);
    if (threadIdx.x == 0) {
        s_mean = mean * __fdividef(1.0f, n);
    }
    __syncthreads();

    if (qual) {
        local_out[0] = local_out[0] - s_mean;
        local_out[1] = local_out[1] - s_mean;
        local_out[2] = local_out[2] - s_mean;
        local_out[3] = local_out[3] - s_mean;
    }
    variance = blockReduceSum<float>(qual ? local_out[0] * local_out[0] + local_out[1] * local_out[1]
                                                + local_out[2] * local_out[2] + local_out[3] * local_out[3] :
                                            0.0f);
    if (threadIdx.x == 0) {
        s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
        s_variance = rsqrtf(s_variance);
    }
    __syncthreads();

    char4* outputTmpPtr = (char4*)output;
    char4 outputTmp;
    if (qual) {
        local_out[0] = (local_out[0] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                       + static_cast<float>(__ldg(beta + col_start));
        outputTmp.x = float_to_int8_rn(local_out[0] * output_scale);

        col_start = col_start + 1;
        local_out[1] = (local_out[1] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                       + static_cast<float>(__ldg(beta + col_start));
        outputTmp.y = float_to_int8_rn(local_out[1] * output_scale);

        col_start = col_start + 1;
        local_out[2] = (local_out[2] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                       + static_cast<float>(__ldg(beta + col_start));
        outputTmp.z = float_to_int8_rn(local_out[2] * output_scale);

        col_start = col_start + 1;
        local_out[3] = (local_out[3] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                       + static_cast<float>(__ldg(beta + col_start));
        outputTmp.w = float_to_int8_rn(local_out[3] * output_scale);

        outputTmpPtr[outIdx] = outputTmp;
    }
}

template<>
__global__ void add_bias_input_layernorm_COL32_int8IO_noRes(int8_t* output,
                                                            int32_t* input1,
                                                            half2* input2,
                                                            const half2* bias,
                                                            const half2* gamma,
                                                            const half2* beta,
                                                            int m,
                                                            int n,
                                                            const float* weight_amax,
                                                            const float* input1_amax_ptr,
                                                            const float* output_scale_ptr)
{
    const float2* weight_scale_ptr = (const float2*)weight_amax;
    const float input1_amax = __ldg(input1_amax_ptr);
    const float output_scale = __ldg(output_scale_ptr);
    int col_start = threadIdx.x << 1;
    bool qual = (col_start < n);

    __shared__ float s_mean;
    __shared__ float s_variance;
    float sums[2] = {0.0f, 0.0f};
    // float mean = 0.0f;
    // float variance = 0.0f;

    float local_out[2];
    int outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 1;
    int2* input1TmpPtr = (int2*)input1;
    int2 input1Tmp;
    if (qual) {
        const float2 weight_scale = __ldg(weight_scale_ptr + threadIdx.x);
        input1Tmp = input1TmpPtr[outIdx];
        half2 biasTmp = bias[threadIdx.x];
        half2 input2Tmp = input2[outIdx];
        // NOTE: 0.000062f = 1 / 127/.0f / 127.0f
        local_out[0] =
            static_cast<float>(input1Tmp.x) * input1_amax * weight_scale.x * 0.000062f + static_cast<float>(biasTmp.x);
        local_out[1] =
            static_cast<float>(input1Tmp.y) * input1_amax * weight_scale.y * 0.000062f + static_cast<float>(biasTmp.y);

        local_out[0] += static_cast<float>(input2Tmp.x);
        local_out[1] += static_cast<float>(input2Tmp.y);

        input2Tmp.x = local_out[0];
        input2Tmp.y = local_out[1];
        input2[outIdx] = input2Tmp;
        for (int i = 0; i < 2; i++) {
            sums[0] += local_out[i];
            sums[1] += local_out[i] * local_out[i];
        }
    }

    blockReduceSumV2<float, 2>(sums);

    if (threadIdx.x == 0) {
        s_mean = sums[0] * __fdividef(1.0f, n);
        s_variance = rsqrtf(sums[1] * __fdividef(1.0f, n) - s_mean * s_mean + 1e-6);
    }
    __syncthreads();

    char2* outputTmpPtr = (char2*)output;
    char2 outputTmp;
    if (qual) {
        half2 gammaTmp = gamma[threadIdx.x];
        half2 betaTmp = beta[threadIdx.x];
        local_out[0] =
            (local_out[0] - s_mean) * s_variance * static_cast<float>(gammaTmp.x) + static_cast<float>(betaTmp.x);
        outputTmp.x = float_to_int8_rn(local_out[0] * output_scale);

        local_out[1] =
            (local_out[1] - s_mean) * s_variance * static_cast<float>(gammaTmp.y) + static_cast<float>(betaTmp.y);
        outputTmp.y = float_to_int8_rn(local_out[1] * output_scale);

        outputTmpPtr[outIdx] = outputTmp;
    }
}

template<typename T>
void invokeAddBiasResidualLayerNormCol32_noRes(int8_t* output,
                                               int32_t* input1,
                                               T* input2,
                                               const T* bias,
                                               const T* gamma,
                                               const T* beta,
                                               int m,
                                               int n,
                                               cudaStream_t stream,
                                               const float* weight_amax,
                                               const float* input1_amax_ptr,
                                               const float* output_scale_ptr)
{
    dim3 grid(m);
    int blockSize = (n / 4 + 31) / 32 * 32;
    dim3 block(blockSize);
    assert(blockSize <= 1024);
    if (sizeof(T) == sizeof(half)) {
        blockSize = (n / 2 + 31) / 32 * 32;
        assert(blockSize <= 1024);
        add_bias_input_layernorm_COL32_int8IO_noRes<<<grid, blockSize, 0, stream>>>(output,
                                                                                    input1,
                                                                                    (half2*)input2,
                                                                                    (const half2*)bias,
                                                                                    (const half2*)gamma,
                                                                                    (const half2*)beta,
                                                                                    m,
                                                                                    n,
                                                                                    weight_amax,
                                                                                    input1_amax_ptr,
                                                                                    output_scale_ptr);
    }
    else {
        add_bias_input_layernorm_COL32_int8IO_noRes<T><<<grid, block, 0, stream>>>(
            output, input1, input2, bias, gamma, beta, m, n, weight_amax, input1_amax_ptr, output_scale_ptr);
    }
}

template void invokeAddBiasResidualLayerNormCol32_noRes(int8_t* output,
                                                        int32_t* input1,
                                                        float* input2,
                                                        const float* bias,
                                                        const float* gamma,
                                                        const float* beta,
                                                        int m,
                                                        int n,
                                                        cudaStream_t stream,
                                                        const float* weight_amax,
                                                        const float* input1_amax_ptr,
                                                        const float* output_scale_ptr);

template void invokeAddBiasResidualLayerNormCol32_noRes(int8_t* output,
                                                        int32_t* input1,
                                                        half* input2,
                                                        const half* bias,
                                                        const half* gamma,
                                                        const half* beta,
                                                        int m,
                                                        int n,
                                                        cudaStream_t stream,
                                                        const float* weight_amax,
                                                        const float* input1_amax_ptr,
                                                        const float* output_scale_ptr);

/*******************  invokeLayernormCol32  ***********************/

// input1/input2/out matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n)
//(grid, block) must be (m, n/4)
// using char4
template<typename T>
__global__ void layernorm_COL32_DataTypeI_int8O(
    int8_t* out, const T* input, const T* gamma, const T* beta, int m, int n, const float* output_scale_ptr)
{
    const float output_scale = __ldg(output_scale_ptr);
    int col_start = threadIdx.x << 2;
    bool qual = (col_start < n);

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    float local_out[4];
    int outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 2;

    if (qual) {
        local_out[0] = static_cast<float>(input[(outIdx << 2) + 0]);
        local_out[1] = static_cast<float>(input[(outIdx << 2) + 1]);
        local_out[2] = static_cast<float>(input[(outIdx << 2) + 2]);
        local_out[3] = static_cast<float>(input[(outIdx << 2) + 3]);
    }

    mean = blockReduceSum<float>(qual ? local_out[0] + local_out[1] + local_out[2] + local_out[3] : 0.0f);
    if (threadIdx.x == 0) {
        s_mean = mean * __fdividef(1.0f, n);
    }
    __syncthreads();

    if (qual) {
        local_out[0] = local_out[0] - s_mean;
        local_out[1] = local_out[1] - s_mean;
        local_out[2] = local_out[2] - s_mean;
        local_out[3] = local_out[3] - s_mean;
    }
    variance = blockReduceSum<float>(qual ? local_out[0] * local_out[0] + local_out[1] * local_out[1]
                                                + local_out[2] * local_out[2] + local_out[3] * local_out[3] :
                                            0.0f);
    if (threadIdx.x == 0) {
        s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
        s_variance = rsqrtf(s_variance);
    }
    __syncthreads();

    char4 outTmp;
    char4* outPtr = (char4*)out;
    if (qual) {
        local_out[0] = (local_out[0] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                       + static_cast<float>(__ldg(beta + col_start));
        outTmp.x = float_to_int8_rn(local_out[0] * output_scale);

        col_start = col_start + 1;
        local_out[1] = (local_out[1] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                       + static_cast<float>(__ldg(beta + col_start));
        outTmp.y = float_to_int8_rn(local_out[1] * output_scale);

        col_start = col_start + 1;
        local_out[2] = (local_out[2] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                       + static_cast<float>(__ldg(beta + col_start));
        outTmp.z = float_to_int8_rn(local_out[2] * output_scale);

        col_start = col_start + 1;
        local_out[3] = (local_out[3] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                       + static_cast<float>(__ldg(beta + col_start));
        outTmp.w = float_to_int8_rn(local_out[3] * output_scale);

        outPtr[outIdx] = outTmp;
    }
}

template<>
__global__ void layernorm_COL32_DataTypeI_int8O(
    int8_t* out, const half2* input, const half2* gamma, const half2* beta, int m, int n, const float* output_scale_ptr)
{
    const float output_scale = __ldg(output_scale_ptr);
    int col_start = threadIdx.x << 1;
    bool qual = (col_start < n);

    __shared__ float s_mean;
    __shared__ float s_variance;
    float sums[2] = {0.0f, 0.0f};
    // float mean = 0.0f;
    // float variance = 0.0f;

    float local_out[2];
    int outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 1;

    if (qual) {
        half2 inputTmp = input[outIdx];
        local_out[0] = static_cast<float>(inputTmp.x);
        local_out[1] = static_cast<float>(inputTmp.y);

        for (int i = 0; i < 2; i++) {
            sums[0] += local_out[i];
            sums[1] += local_out[i] * local_out[i];
        }
    }

    blockReduceSumV2<float, 2>(sums);

    if (threadIdx.x == 0) {
        s_mean = sums[0] * __fdividef(1.0f, n);
        s_variance = rsqrtf(sums[1] * __fdividef(1.0f, n) - s_mean * s_mean + 1e-6);
    }
    __syncthreads();

    char2 outTmp;
    char2* outPtr = (char2*)out;
    if (qual) {
        half2 gammaTmp = gamma[threadIdx.x];
        half2 betaTmp = beta[threadIdx.x];
        local_out[0] =
            (local_out[0] - s_mean) * s_variance * static_cast<float>(gammaTmp.x) + static_cast<float>(betaTmp.x);
        outTmp.x = float_to_int8_rn(local_out[0] * output_scale);

        local_out[1] =
            (local_out[1] - s_mean) * s_variance * static_cast<float>(gammaTmp.y) + static_cast<float>(betaTmp.y);
        outTmp.y = float_to_int8_rn(local_out[1] * output_scale);

        outPtr[outIdx] = outTmp;
    }
}

template<typename T>
void invokeLayernormCol32(int8_t* out,
                          const T* input,
                          const T* gamma,
                          const T* beta,
                          int m,
                          int n,
                          const float* output_scale_ptr,
                          cudaStream_t stream)
{
    dim3 grid(m);
    int blockSize = (n / 4 + 31) / 32 * 32;
    dim3 block(blockSize);
    assert(blockSize <= 1024);
    if (sizeof(T) == sizeof(half)) {
        blockSize = (n / 2 + 31) / 32 * 32;
        assert(blockSize <= 1024);
        layernorm_COL32_DataTypeI_int8O<<<grid, blockSize, 0, stream>>>(
            out, (const half2*)input, (const half2*)gamma, (const half2*)beta, m, n, output_scale_ptr);
    }
    else {
        layernorm_COL32_DataTypeI_int8O<T><<<grid, block, 0, stream>>>(out, input, gamma, beta, m, n, output_scale_ptr);
    }
}

template void invokeLayernormCol32(int8_t* out,
                                   const float* input,
                                   const float* gamma,
                                   const float* beta,
                                   int m,
                                   int n,
                                   const float* output_scale_ptr,
                                   cudaStream_t stream);

template void invokeLayernormCol32(int8_t* out,
                                   const half* input,
                                   const half* gamma,
                                   const half* beta,
                                   int m,
                                   int n,
                                   const float* output_scale_ptr,
                                   cudaStream_t stream);

/*******************  invokeLayernormShiftPartitionCol32  ***********************/

template<typename T>
__global__ void layernorm_shift_partition_COL32_noRes(int8_t* out,
                                                      const T* input,
                                                      const T* gamma,
                                                      const T* beta,
                                                      int batch,
                                                      int H,
                                                      int W,
                                                      int n,
                                                      const float* norm_scale_ptr,
                                                      int shift_size,
                                                      int window_size)
{
    float norm_scale = __ldg(norm_scale_ptr);
    int tid = threadIdx.x;
    const int batch_offset = blockIdx.z * gridDim.y * gridDim.x;
    const int m = gridDim.z * gridDim.y * gridDim.x;
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

    const int offset_col32_in = (tid & 0xffffffe0) * m + (bid << 5) + (tid & 31);
    float local_out = (tid < n) ? (float)(__ldg(input + offset_col32_in)) : 0.0f;

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
        const int offset_col32_out = (tid & 0xffffffe0) * m + (output_bid << 5) + (tid & 31);
        local_out =
            ((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg(&gamma[tid])) + (float)(__ldg(&beta[tid]));
        out[offset_col32_out] = float_to_int8_rn(norm_scale * local_out);
    }
}

template<>
__global__ void layernorm_shift_partition_COL32_noRes(int8_t* out_ptr,
                                                      const half4* input_ptr,
                                                      const half4* gamma_ptr,
                                                      const half4* beta_ptr,
                                                      int batch,
                                                      int H,
                                                      int W,
                                                      int n,
                                                      const float* norm_scale_ptr,
                                                      int shift_size,
                                                      int window_size)
{
    float norm_scale = __ldg(norm_scale_ptr);

    const int batch_offset = blockIdx.z * gridDim.y * gridDim.x;
    const int m = gridDim.z * gridDim.y * gridDim.x;
    const int bid = batch_offset + blockIdx.y * gridDim.x + blockIdx.x;

    const int shifted_H_idx = (shift_size != 0) ? ((blockIdx.y - shift_size + gridDim.y) % gridDim.y) : blockIdx.y;
    const int shifted_W_idx = (shift_size != 0) ? ((blockIdx.x - shift_size + gridDim.x) % gridDim.x) : blockIdx.x;
    const int window_H_idx = shifted_H_idx / window_size;
    const int window_W_idx = shifted_W_idx / window_size;
    const int stride_of_window_H = W / window_size;
    const int window_idx = window_H_idx * stride_of_window_H + window_W_idx;
    const int idx_in_window = (shifted_H_idx % window_size) * window_size + (shifted_W_idx % window_size);
    const int output_bid = batch_offset + window_idx * window_size * window_size + idx_in_window;

    int tid = threadIdx.x << 2;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float sums[2] = {0.0f, 0.0f};
    half4 inputTmp;
    float inputBuf[4];

    char4* output_ptr = (char4*)out_ptr;
    char4 int8_buf;

    const int offset_col32 = (tid & 0xffffffe0) * m + (bid << 5) + (tid & 31);
    if (tid < n) {
        inputTmp = input_ptr[offset_col32 >> 2];
        inputBuf[0] = static_cast<float>(inputTmp.x);
        inputBuf[1] = static_cast<float>(inputTmp.y);
        inputBuf[2] = static_cast<float>(inputTmp.z);
        inputBuf[3] = static_cast<float>(inputTmp.w);
        for (int i = 0; i < 4; i++) {
            sums[0] += inputBuf[i];
            sums[1] += inputBuf[i] * inputBuf[i];
        }
    }

    blockReduceSumV2<float, 2>(sums);

    if (threadIdx.x == 0) {
        s_mean = sums[0] / n;
        s_variance = rsqrtf(sums[1] / n - s_mean * s_mean + 1e-6);
    }
    __syncthreads();

    if (tid < n) {
        half4 gamma_val = gamma_ptr[tid >> 2];
        half4 beta_val = beta_ptr[tid >> 2];
        inputBuf[0] =
            (inputBuf[0] - s_mean) * s_variance * static_cast<float>(gamma_val.x) + static_cast<float>(beta_val.x);
        inputBuf[1] =
            (inputBuf[1] - s_mean) * s_variance * static_cast<float>(gamma_val.y) + static_cast<float>(beta_val.y);
        inputBuf[2] =
            (inputBuf[2] - s_mean) * s_variance * static_cast<float>(gamma_val.z) + static_cast<float>(beta_val.z);
        inputBuf[3] =
            (inputBuf[3] - s_mean) * s_variance * static_cast<float>(gamma_val.w) + static_cast<float>(beta_val.w);

        const int offset_col32_out = (tid & 0xffffffe0) * m + (output_bid << 5) + (tid & 31);
        // const int offset_colMajor_out = output_bid * n + tid;
        // const int offset_out = index_CUBLASLT_ORDER_COL32_2R_4R4(tid, output_bid, m << 5);
        int8_buf.x = float_to_int8_rn(norm_scale * inputBuf[0]);
        int8_buf.y = float_to_int8_rn(norm_scale * inputBuf[1]);
        int8_buf.z = float_to_int8_rn(norm_scale * inputBuf[2]);
        int8_buf.w = float_to_int8_rn(norm_scale * inputBuf[3]);
        output_ptr[offset_col32_out >> 2] = int8_buf;
    }
}

template<typename T>
__global__ void layernorm_shift_partition_v2_COL32_noRes(int8_t* out,
                                                         const T* __restrict input,
                                                         const T* __restrict gamma,
                                                         const T* __restrict beta,
                                                         int batch,
                                                         int H,
                                                         int W,
                                                         int n,
                                                         const float* norm_scale_ptr,
                                                         int shift_size,
                                                         int window_size)
{
    float norm_scale = __ldg(norm_scale_ptr);
    const int ite = 4;
    const int tid = threadIdx.x;
    const int batch_offset = blockIdx.z * gridDim.y * gridDim.x;
    const int m = gridDim.z * gridDim.y * gridDim.x;
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
    float local_out[ite];

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        if (col_id < n) {
            const int offset_col32_in = (col_id & 0xffffffe0) * m + (bid << 5) + (col_id & 31);
            local_out[i] = (float)(__ldg(input + offset_col32_in));
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
            const int offset_col32_out = (col_id & 0xffffffe0) * m + (output_bid << 5) + (col_id & 31);
            out[offset_col32_out] = float_to_int8_rn(
                norm_scale * (local_out[i] * s_variance * (float)__ldg(&gamma[col_id]) + (float)__ldg(&beta[col_id])));
        }
    }
}

template<>
__global__ void layernorm_shift_partition_v2_COL32_noRes(int8_t* out_ptr,
                                                         const half2* __restrict input_ptr,
                                                         const half2* __restrict gamma_ptr,
                                                         const half2* __restrict beta_ptr,
                                                         int batch,
                                                         int H,
                                                         int W,
                                                         int n,
                                                         const float* norm_scale_ptr,
                                                         int shift_size,
                                                         int window_size)
{
    float norm_scale = __ldg(norm_scale_ptr);
    const int ite = 4;
    const int tid = threadIdx.x;
    const int batch_offset = blockIdx.z * gridDim.y * gridDim.x;
    const int m = gridDim.z * gridDim.y * gridDim.x;
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
    half2 local_out_half2[ite];
    const half2 zero = {static_cast<half>(0.0f), static_cast<half>(0.0f)};

    char2* output_ptr = (char2*)out_ptr;
    char2 int8_buf;

    // float sum = 0.0f;
    half2 sum = __float2half2_rn(0.0f);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = (i * blockDim.x + tid) << 1;
        if (col_id < n) {
            const int offset_col32 = (col_id & 0xffffffe0) * m + (bid << 5) + (col_id & 31);
            local_out_half2[i] = __ldg(input_ptr + (offset_col32 >> 1));

            sum += local_out_half2[i];
        }
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
        int col_id = (i * blockDim.x + tid) << 1;
        if (col_id < n) {
            local_out_half2[i] = local_out_half2[i] - s_mean_2;
            float v1 = (float)local_out_half2[i].x;
            float v2 = (float)local_out_half2[i].y;
            var += v1 * v1 + v2 * v2;
        }
    }

    variance = blockReduceSum<float>(var);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + 1e-6f);
    }
    __syncthreads();

    half2 s_var_2 = __float2half2_rn(s_variance);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = (i * blockDim.x + tid) << 1;
        if (col_id < n) {
            const int offset_col32_out = (col_id & 0xffffffe0) * m + (output_bid << 5) + (col_id & 31);
            half2 outVal =
                local_out_half2[i] * s_var_2 * __ldg(&gamma_ptr[col_id >> 1]) + __ldg(&beta_ptr[col_id >> 1]);
            int8_buf.x = float_to_int8_rn(norm_scale * static_cast<float>(outVal.x));
            int8_buf.y = float_to_int8_rn(norm_scale * static_cast<float>(outVal.y));
            output_ptr[offset_col32_out >> 1] = int8_buf;
        }
    }
}

template<typename T>
void invokeLayernormShiftPartitionCol32(int8_t* out,
                                        const T* input,
                                        const T* gamma,
                                        const T* beta,
                                        int batch,
                                        int H,
                                        int W,
                                        int n,
                                        const float* norm_scale_ptr,
                                        int shift_size,
                                        int window_size,
                                        cudaStream_t stream)
{
    dim3 grid(W, H, batch);
    int blockSize = (n + 31) / 32 * 32;
    if (blockSize >= 768) {
        blockSize = ((blockSize / 4) + 31) / 32 * 32;
        layernorm_shift_partition_v2_COL32_noRes<T><<<grid, blockSize, 0, stream>>>(
            out, input, gamma, beta, batch, H, W, n, norm_scale_ptr, shift_size, window_size);
    }
    else {
        layernorm_shift_partition_COL32_noRes<T><<<grid, blockSize, 0, stream>>>(
            out, input, gamma, beta, batch, H, W, n, norm_scale_ptr, shift_size, window_size);
    }
}

template<>
void invokeLayernormShiftPartitionCol32(int8_t* out,
                                        const half* input,
                                        const half* gamma,
                                        const half* beta,
                                        int batch,
                                        int H,
                                        int W,
                                        int n,
                                        const float* norm_scale_ptr,
                                        int shift_size,
                                        int window_size,
                                        cudaStream_t stream)
{
    dim3 grid(W, H, batch);
    int blockSize = n / 2;
    blockSize = (blockSize + 31) / 32 * 32;

    if ((batch * H * W >= 512 && blockSize >= 768) || blockSize > 1024) {
        blockSize = ((blockSize / 4) + 31) / 32 * 32;
        layernorm_shift_partition_v2_COL32_noRes<<<grid, blockSize, 0, stream>>>(out,
                                                                                 (const half2*)input,
                                                                                 (const half2*)gamma,
                                                                                 (const half2*)beta,
                                                                                 batch,
                                                                                 H,
                                                                                 W,
                                                                                 n,
                                                                                 norm_scale_ptr,
                                                                                 shift_size,
                                                                                 window_size);
    }
    else {
        blockSize = (n / 4 + 32) / 32 * 32;
        layernorm_shift_partition_COL32_noRes<<<grid, blockSize, 0, stream>>>(out,
                                                                              (const half4*)input,
                                                                              (const half4*)gamma,
                                                                              (const half4*)beta,
                                                                              batch,
                                                                              H,
                                                                              W,
                                                                              n,
                                                                              norm_scale_ptr,
                                                                              shift_size,
                                                                              window_size);
    }
}

template void invokeLayernormShiftPartitionCol32(int8_t* out,
                                                 const float* input,
                                                 const float* gamma,
                                                 const float* beta,
                                                 int batch,
                                                 int H,
                                                 int W,
                                                 int n,
                                                 const float* norm_scale_ptr,
                                                 int shift_size,
                                                 int window_size,
                                                 cudaStream_t stream);

template void invokeLayernormShiftPartitionCol32(int8_t* out,
                                                 const half* input,
                                                 const half* gamma,
                                                 const half* beta,
                                                 int batch,
                                                 int H,
                                                 int W,
                                                 int n,
                                                 const float* norm_scale_ptr,
                                                 int shift_size,
                                                 int window_size,
                                                 cudaStream_t stream);

/*******************  invokeMergeLayerNormCol32  ***********************/

// input is [batch, 2*H, 2*W, n/4]
// output is [batch, H, W, n]
// grid (W, H, batch)
// block (n)
template<typename T>
__global__ void merge_layernorm_v2(int8_t* out,
                                   const T* __restrict input,
                                   const T* __restrict gamma,
                                   const T* __restrict beta,
                                   int batch,
                                   const float* merge_inFactor,
                                   int H,
                                   int W,
                                   int n)
{
    const int ite = 4;
    const int tid = threadIdx.x;
    const int W_idx = blockIdx.x;
    const int H_idx = blockIdx.y;
    const float out_scale = __ldg(merge_inFactor);
    // const size_t batch_offset = blockIdx.z * H * W * n;
    // const int input_H_stride = W*n/2;
    // const int output_H_stride = W*n;
    const int n_4 = n >> 2;
    const int m = batch * 4 * H * W;

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
            // size_t input_id = batch_offset + (2*H_idx + offset_in_H)*input_H_stride + (2*W_idx + offset_in_W)*n_4 +
            // (col_id % n_4);

            int col_input = col_id % n_4;
            int row_input = blockIdx.z * H * W * 4 + (2 * H_idx + offset_in_H) * W * 2 + (2 * W_idx + offset_in_W);
            int input_idx_col32 = ((col_input >> 5) << 5) * m + (row_input << 5) + (col_input & 31);
            local_out[i] = (float)(__ldg(input + input_idx_col32));
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
            // size_t output_idx = batch_offset + (H_idx*W + W_idx)*n + col_id;

            int col_output = col_id;
            int row_output = blockIdx.z * H * W + H_idx * W + W_idx;
            int output_idx_col32 = ((col_output >> 5) << 5) * (m >> 2) + (row_output << 5) + (col_output & 31);
            out[output_idx_col32] = float_to_int8_rn(
                out_scale * (local_out[i] * s_variance * (float)__ldg(&gamma[col_id]) + (float)__ldg(&beta[col_id])));
        }
    }
}

// TODO : accelerate with half2
template<typename T>
void invokeMergeLayerNormCol32(int8_t* output,
                               const T* input,
                               const T* gamma,
                               const T* beta,
                               int batch,
                               const float* merge_inFactor,
                               int H,
                               int W,
                               int n,
                               cudaStream_t stream)
{
    if ((W % 2 != 0) || (H % 2 != 0)) {
        printf("[ERROR][invokeMergeLayerNormCol32] H(W) should be a multiple of 2.\n");
        return;
    }
    dim3 grid(W / 2, H / 2, batch);
    int blockSize = 4 * n;
    blockSize = (blockSize + 31) / 32 * 32;
    // TODO
    // if (blockSize >= 768)
    {
        blockSize = ((blockSize / 4) + 31) / 32 * 32;
        merge_layernorm_v2<T>
            <<<grid, blockSize, 0, stream>>>(output, input, gamma, beta, batch, merge_inFactor, H / 2, W / 2, n * 4);
    }
    /*
    else
      merge_layernorm<T><<<grid, blockSize, 0, stream>>>(output, input, gamma, beta, batch, H/2, W/2, n*4);
    */
}

template void invokeMergeLayerNormCol32<float>(int8_t* output,
                                               const float* input,
                                               const float* gamma,
                                               const float* beta,
                                               int batch,
                                               const float* merge_inFactor,
                                               int H,
                                               int W,
                                               int n,
                                               cudaStream_t stream);

template void invokeMergeLayerNormCol32<half>(int8_t* output,
                                              const half* input,
                                              const half* gamma,
                                              const half* beta,
                                              int batch,
                                              const float* merge_inFactor,
                                              int H,
                                              int W,
                                              int n,
                                              cudaStream_t stream);

// input1/input2/out matrix with layout of row major (m*n)
//(grid, block) must be (m, n/4)
// using char4
template<typename T>
__global__ void add_bias_input_layernorm_ROW_int8IO(int8_t* output,
                                                    const int8_t* input1,
                                                    const int8_t* input2,
                                                    const T* bias,
                                                    const T* gamma,
                                                    const T* beta,
                                                    int m,
                                                    int n,
                                                    const float* input1_deQFactor_ptr,
                                                    const float* input2_deQFactor_ptr,
                                                    const float* output_scale_ptr)
{
    const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
    const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
    const float output_scale = __ldg(output_scale_ptr);
    int col_start = threadIdx.x << 2;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    float local_out[4];
    int outIdx = (blockIdx.x * n + col_start) >> 2;
    char4* outTmpPtr = (char4*)output;
    char4* input1TmpPtr = (char4*)input1;
    char4* input2TmpPtr = (char4*)input2;
    char4 input1Tmp = __ldg(input1TmpPtr + outIdx);
    char4 input2Tmp = __ldg(input2TmpPtr + outIdx);

    int col_start_tmp = col_start;
    local_out[0] = static_cast<float>(input2Tmp.x) * input2_deQFactor
                   + static_cast<float>(input1Tmp.x) * input1_deQFactor
                   + static_cast<float>(__ldg(bias + col_start_tmp));
    col_start_tmp = col_start_tmp + 1;
    local_out[1] = static_cast<float>(input2Tmp.y) * input2_deQFactor
                   + static_cast<float>(input1Tmp.y) * input1_deQFactor
                   + static_cast<float>(__ldg(bias + col_start_tmp));
    col_start_tmp = col_start_tmp + 1;
    local_out[2] = static_cast<float>(input2Tmp.z) * input2_deQFactor
                   + static_cast<float>(input1Tmp.z) * input1_deQFactor
                   + static_cast<float>(__ldg(bias + col_start_tmp));
    col_start_tmp = col_start_tmp + 1;
    local_out[3] = static_cast<float>(input2Tmp.w) * input2_deQFactor
                   + static_cast<float>(input1Tmp.w) * input1_deQFactor
                   + static_cast<float>(__ldg(bias + col_start_tmp));

    mean = blockReduceSum<float>(local_out[0] + local_out[1] + local_out[2] + local_out[3]);
    if (threadIdx.x == 0) {
        s_mean = mean * __fdividef(1.0f, n);
    }
    __syncthreads();

    local_out[0] = local_out[0] - s_mean;
    local_out[1] = local_out[1] - s_mean;
    local_out[2] = local_out[2] - s_mean;
    local_out[3] = local_out[3] - s_mean;
    variance = blockReduceSum<float>(local_out[0] * local_out[0] + local_out[1] * local_out[1]
                                     + local_out[2] * local_out[2] + local_out[3] * local_out[3]);
    if (threadIdx.x == 0) {
        s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
        s_variance = rsqrtf(s_variance);
    }
    __syncthreads();

    local_out[0] = (local_out[0] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                   + static_cast<float>(__ldg(beta + col_start));
    input2Tmp.x = float_to_int8_rn(local_out[0] * output_scale);

    col_start = col_start + 1;
    local_out[1] = (local_out[1] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                   + static_cast<float>(__ldg(beta + col_start));
    input2Tmp.y = float_to_int8_rn(local_out[1] * output_scale);

    col_start = col_start + 1;
    local_out[2] = (local_out[2] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                   + static_cast<float>(__ldg(beta + col_start));
    input2Tmp.z = float_to_int8_rn(local_out[2] * output_scale);

    col_start = col_start + 1;
    local_out[3] = (local_out[3] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                   + static_cast<float>(__ldg(beta + col_start));
    input2Tmp.w = float_to_int8_rn(local_out[3] * output_scale);

    outTmpPtr[outIdx] = input2Tmp;
}

template<>
__global__ void add_bias_input_layernorm_ROW_int8IO(int8_t* output,
                                                    const int8_t* input1,
                                                    const int8_t* input2,
                                                    const half2* bias,
                                                    const half2* gamma,
                                                    const half2* beta,
                                                    int m,
                                                    int n,
                                                    const float* input1_deQFactor_ptr,
                                                    const float* input2_deQFactor_ptr,
                                                    const float* output_scale_ptr)
{
    const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
    const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
    const float output_scale = __ldg(output_scale_ptr);
    int col_start = threadIdx.x << 2;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    float local_out[4];
    int outIdx = (blockIdx.x * n + col_start) >> 2;
    char4* outTmpPtr = (char4*)output;
    char4* input1TmpPtr = (char4*)input1;
    char4* input2TmpPtr = (char4*)input2;
    char4 input1Tmp = __ldg(input1TmpPtr + outIdx);
    char4 input2Tmp = __ldg(input2TmpPtr + outIdx);

    int col_start_tmp = col_start;
    half2 biasTmp = __ldg(bias + (col_start_tmp >> 1));
    local_out[0] = static_cast<float>(input2Tmp.x) * input2_deQFactor
                   + static_cast<float>(input1Tmp.x) * input1_deQFactor + static_cast<float>(biasTmp.x);
    col_start_tmp = col_start_tmp + 1;
    local_out[1] = static_cast<float>(input2Tmp.y) * input2_deQFactor
                   + static_cast<float>(input1Tmp.y) * input1_deQFactor + static_cast<float>(biasTmp.y);

    col_start_tmp = col_start_tmp + 1;
    biasTmp = __ldg(bias + (col_start_tmp >> 1));
    local_out[2] = static_cast<float>(input2Tmp.z) * input2_deQFactor
                   + static_cast<float>(input1Tmp.z) * input1_deQFactor + static_cast<float>(biasTmp.x);
    col_start_tmp = col_start_tmp + 1;
    local_out[3] = static_cast<float>(input2Tmp.w) * input2_deQFactor
                   + static_cast<float>(input1Tmp.w) * input1_deQFactor + static_cast<float>(biasTmp.y);

    mean = blockReduceSum<float>(local_out[0] + local_out[1] + local_out[2] + local_out[3]);
    if (threadIdx.x == 0) {
        s_mean = mean * __fdividef(1.0f, n);
    }
    __syncthreads();

    local_out[0] = local_out[0] - s_mean;
    local_out[1] = local_out[1] - s_mean;
    local_out[2] = local_out[2] - s_mean;
    local_out[3] = local_out[3] - s_mean;
    variance = blockReduceSum<float>(local_out[0] * local_out[0] + local_out[1] * local_out[1]
                                     + local_out[2] * local_out[2] + local_out[3] * local_out[3]);
    if (threadIdx.x == 0) {
        s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
        s_variance = rsqrtf(s_variance);
    }
    __syncthreads();

    col_start_tmp = col_start >> 1;
    biasTmp = __ldg(gamma + col_start_tmp);
    half2 betaTmp = __ldg(beta + col_start_tmp);

    local_out[0] = (local_out[0] * s_variance) * static_cast<float>(biasTmp.x) + static_cast<float>(betaTmp.x);
    input2Tmp.x = float_to_int8_rn(local_out[0] * output_scale);

    col_start = col_start + 1;
    local_out[1] = (local_out[1] * s_variance) * static_cast<float>(biasTmp.y) + static_cast<float>(betaTmp.y);
    input2Tmp.y = float_to_int8_rn(local_out[1] * output_scale);

    col_start = col_start + 1;
    col_start_tmp = col_start >> 1;
    biasTmp = __ldg(gamma + col_start_tmp);
    betaTmp = __ldg(beta + col_start_tmp);
    local_out[2] = (local_out[2] * s_variance) * static_cast<float>(biasTmp.x) + static_cast<float>(betaTmp.x);
    input2Tmp.z = float_to_int8_rn(local_out[2] * output_scale);

    col_start = col_start + 1;
    local_out[3] = (local_out[3] * s_variance) * static_cast<float>(biasTmp.y) + static_cast<float>(betaTmp.y);
    input2Tmp.w = float_to_int8_rn(local_out[3] * output_scale);

    outTmpPtr[outIdx] = input2Tmp;
}

template<typename T>
void invokeAddBiasResidualLayerNormRow(int8_t* output,
                                       const int8_t* input1,
                                       const int8_t* input2,
                                       const T* bias,
                                       const T* gamma,
                                       const T* beta,
                                       int m,
                                       int n,
                                       cudaStream_t stream,
                                       const float* input1_deQFactor_ptr,
                                       const float* input2_deQFactor_ptr,
                                       const float* output_scale_ptr)
{
    dim3 grid(m);
    dim3 block(n / 4);
    assert(n <= 1024);
    if (sizeof(T) == sizeof(half)) {
        add_bias_input_layernorm_ROW_int8IO<<<grid, block, 0, stream>>>(output,
                                                                        input1,
                                                                        input2,
                                                                        (const half2*)bias,
                                                                        (const half2*)gamma,
                                                                        (const half2*)beta,
                                                                        m,
                                                                        n,
                                                                        input1_deQFactor_ptr,
                                                                        input2_deQFactor_ptr,
                                                                        output_scale_ptr);
    }
    else {
        add_bias_input_layernorm_ROW_int8IO<T><<<grid, block, 0, stream>>>(output,
                                                                           input1,
                                                                           input2,
                                                                           bias,
                                                                           gamma,
                                                                           beta,
                                                                           m,
                                                                           n,
                                                                           input1_deQFactor_ptr,
                                                                           input2_deQFactor_ptr,
                                                                           output_scale_ptr);
    }
}

template void invokeAddBiasResidualLayerNormRow(int8_t* output,
                                                const int8_t* input1,
                                                const int8_t* input2,
                                                const float* bias,
                                                const float* gamma,
                                                const float* beta,
                                                int m,
                                                int n,
                                                cudaStream_t stream,
                                                const float* input1_deQFactor_ptr,
                                                const float* input2_deQFactor_ptr,
                                                const float* output_scale_ptr);

template void invokeAddBiasResidualLayerNormRow(int8_t* output,
                                                const int8_t* input1,
                                                const int8_t* input2,
                                                const half* bias,
                                                const half* gamma,
                                                const half* beta,
                                                int m,
                                                int n,
                                                cudaStream_t stream,
                                                const float* input1_deQFactor_ptr,
                                                const float* input2_deQFactor_ptr,
                                                const float* output_scale_ptr);

// input1/input2/out matrix with layout of row major (m*n)
//(grid, block) must be (m, n)
template<typename T>
__global__ void add_bias_input_layernorm_ROW_int8I_DataTypeO(T* output,
                                                             const int8_t* input1,
                                                             const int8_t* input2,
                                                             const T* bias,
                                                             const T* gamma,
                                                             const T* beta,
                                                             int m,
                                                             int n,
                                                             const float* input1_deQFactor_ptr,
                                                             const float* input2_deQFactor_ptr)
{
    const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
    const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
    int col_start = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    float local_out;
    int idx = blockIdx.x * n + col_start;

    local_out = static_cast<float>(__ldg(input2 + idx)) * input2_deQFactor
                + static_cast<float>(__ldg(input1 + idx)) * input1_deQFactor
                + static_cast<float>(__ldg(bias + col_start));

    mean = blockReduceSum<float>(local_out);
    if (threadIdx.x == 0) {
        s_mean = mean * __fdividef(1.0f, n);
    }
    __syncthreads();

    local_out = local_out - s_mean;

    variance = blockReduceSum<float>(local_out * local_out);

    if (threadIdx.x == 0) {
        s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
        s_variance = rsqrtf(s_variance);
    }
    __syncthreads();

    local_out = (local_out * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                + static_cast<float>(__ldg(beta + col_start));

    output[idx] = local_out;
}

// input1/input2/out matrix with layout of row major (m*n)
//(grid, block) must be (m, n/2)
template<>
__global__ void add_bias_input_layernorm_ROW_int8I_DataTypeO(half2* output,
                                                             const int8_t* input1,
                                                             const int8_t* input2,
                                                             const half2* bias,
                                                             const half2* gamma,
                                                             const half2* beta,
                                                             int m,
                                                             int n,
                                                             const float* input1_deQFactor_ptr,
                                                             const float* input2_deQFactor_ptr)
{
    const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
    const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
    int col_start = threadIdx.x << 1;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    float2 local_out;
    int idx = (blockIdx.x * n + col_start) >> 1;

    const char2* input1_ptr2 = (const char2*)input1;
    const char2* input2_ptr2 = (const char2*)input2;
    char2 input_tmp1 = __ldg(input1_ptr2 + idx);
    char2 input_tmp2 = __ldg(input2_ptr2 + idx);

    half2 bias_tmp = __ldg(bias + threadIdx.x);

    local_out.x = static_cast<float>(input_tmp1.x) * input1_deQFactor
                  + static_cast<float>(input_tmp2.x) * input2_deQFactor + static_cast<float>(bias_tmp.x);

    local_out.y = static_cast<float>(input_tmp1.y) * input1_deQFactor
                  + static_cast<float>(input_tmp2.y) * input2_deQFactor + static_cast<float>(bias_tmp.y);

    mean = blockReduceSum<float>(local_out.x + local_out.y);
    if (threadIdx.x == 0) {
        s_mean = mean * __fdividef(1.0f, n);
    }
    __syncthreads();

    local_out.x = local_out.x - s_mean;

    local_out.y = local_out.y - s_mean;

    variance = blockReduceSum<float>(local_out.x * local_out.x + local_out.y * local_out.y);

    if (threadIdx.x == 0) {
        s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
        s_variance = rsqrtf(s_variance);
    }
    __syncthreads();

    half2 gamma_tmp = __ldg(gamma + threadIdx.x);
    half2 beta_tmp = __ldg(beta + threadIdx.x);

    local_out.x = (local_out.x * s_variance) * static_cast<float>(gamma_tmp.x) + static_cast<float>(beta_tmp.x);
    local_out.y = (local_out.y * s_variance) * static_cast<float>(gamma_tmp.y) + static_cast<float>(beta_tmp.y);

    bias_tmp.x = half(local_out.x);
    bias_tmp.y = half(local_out.y);

    output[idx] = bias_tmp;
}

template<typename T>
void invokeAddBiasResidualLayerNormRow(T* output,
                                       const int8_t* input1,
                                       const int8_t* input2,
                                       const T* bias,
                                       const T* gamma,
                                       const T* beta,
                                       int m,
                                       int n,
                                       cudaStream_t stream,
                                       const float* input1_deQFactor_ptr,
                                       const float* input2_deQFactor_ptr)
{
    dim3 grid(m);
    dim3 block(n);
    if (sizeof(T) == sizeof(half)) {
        assert(n / 2 <= 1024 && n % 2 == 0);
        block.x = n / 2;
        add_bias_input_layernorm_ROW_int8I_DataTypeO<<<grid, block, 0, stream>>>((half2*)output,
                                                                                 input1,
                                                                                 input2,
                                                                                 (const half2*)bias,
                                                                                 (const half2*)gamma,
                                                                                 (const half2*)beta,
                                                                                 m,
                                                                                 n,
                                                                                 input1_deQFactor_ptr,
                                                                                 input2_deQFactor_ptr);
    }
    else {
        assert(n <= 1024);
        add_bias_input_layernorm_ROW_int8I_DataTypeO<T><<<grid, block, 0, stream>>>(
            output, input1, input2, bias, gamma, beta, m, n, input1_deQFactor_ptr, input2_deQFactor_ptr);
    }
}

template void invokeAddBiasResidualLayerNormRow<float>(float* output,
                                                       const int8_t* input1,
                                                       const int8_t* input2,
                                                       const float* bias,
                                                       const float* gamma,
                                                       const float* beta,
                                                       int m,
                                                       int n,
                                                       cudaStream_t stream,
                                                       const float* input1_deQFactor_ptr,
                                                       const float* input2_deQFactor_ptr);

template void invokeAddBiasResidualLayerNormRow<half>(half* output,
                                                      const int8_t* input1,
                                                      const int8_t* input2,
                                                      const half* bias,
                                                      const half* gamma,
                                                      const half* beta,
                                                      int m,
                                                      int n,
                                                      cudaStream_t stream,
                                                      const float* input1_deQFactor_ptr,
                                                      const float* input2_deQFactor_ptr);

}  // namespace fastertransformer