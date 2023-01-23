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

#include "src/fastertransformer/kernels/layernorm_int8_kernels.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"

namespace fastertransformer {

// input1/input2/output matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n)
//(grid, block) must be (m, n)
// for per_channel_quantization for weight
__global__ void add_bias_input_layernorm_COL32_int32I_DataTypeO(float*         output,
                                                                const int32_t* input1,
                                                                const float*   input2,
                                                                const float*   bias,
                                                                const float*   gamma,
                                                                const float*   beta,
                                                                int            m,
                                                                int            n,
                                                                const float*   weight_amax,
                                                                const float*   input1_amax_ptr)
{
    const float input1_amax = __ldg(input1_amax_ptr);
    int         col_start   = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    float local_out;
    int   outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31));

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

__global__ void add_bias_input_layernorm_COL32_int32I_DataTypeO(half2*        output,
                                                                const int2*   input1,
                                                                const half2*  input2,
                                                                const half2*  bias,
                                                                const half2*  gamma,
                                                                const half2*  beta,
                                                                int           m,
                                                                int           n,
                                                                const float2* weight_amax,
                                                                const float*  input1_amax_ptr)
{
    int col_start = threadIdx.x << 1;

    const float input1_amax = __ldg(input1_amax_ptr);

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    float2 local_out;
    int    outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 1;

    const int2   input1Tmp = __ldg(input1 + outIdx);
    const float2 weightTmp = __ldg(weight_amax + threadIdx.x);

    float2 addTmp2;
    addTmp2.x = static_cast<float>(input1Tmp.x) * weightTmp.x * input1_amax * 0.000062f;  //(1/127/127);
    addTmp2.y = static_cast<float>(input1Tmp.y) * weightTmp.y * input1_amax * 0.000062f;  //(1/127/127);

    const half2 inputTmp = __ldg(input2 + outIdx);
    const half2 biasTmp  = __ldg(bias + threadIdx.x);

    local_out   = __half22float2(__hadd2(inputTmp, biasTmp));
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

    float2      outputTmp;
    const half2 gammaTmp = __ldg(gamma + threadIdx.x);
    const half2 betaTmp  = __ldg(beta + threadIdx.x);

    outputTmp.x = (local_out.x * s_variance) * static_cast<float>(gammaTmp.x) + static_cast<float>(betaTmp.x);
    outputTmp.y = (local_out.y * s_variance) * static_cast<float>(gammaTmp.y) + static_cast<float>(betaTmp.y);

    output[outIdx] = __float22half2_rn(outputTmp);
}

template<typename T>
void invokeAddBiasResidualLayerNormCol32(T*             output,
                                         const int32_t* input1,
                                         const T*       input2,
                                         const T*       bias,
                                         const T*       gamma,
                                         const T*       beta,
                                         int            m,
                                         int            n,
                                         cudaStream_t   stream,
                                         const float*   weight_amax,
                                         const float*   input1_amax_ptr)
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

template void invokeAddBiasResidualLayerNormCol32(float*         output,
                                                  const int32_t* input1,
                                                  const float*   input2,
                                                  const float*   bias,
                                                  const float*   gamma,
                                                  const float*   beta,
                                                  int            m,
                                                  int            n,
                                                  cudaStream_t   stream,
                                                  const float*   weight_amax,
                                                  const float*   input1_amax_ptr);
template void invokeAddBiasResidualLayerNormCol32(half*          output,
                                                  const int32_t* input1,
                                                  const half*    input2,
                                                  const half*    bias,
                                                  const half*    gamma,
                                                  const half*    beta,
                                                  int            m,
                                                  int            n,
                                                  cudaStream_t   stream,
                                                  const float*   weight_amax,
                                                  const float*   input1_amax_ptr);

// input1/input2/out matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n)
//(grid, block) must be (m, n/4)
// using char4
template<typename T>
__global__ void add_bias_input_layernorm_COL32_int8IO(int8_t*       output,
                                                      const int8_t* input1,
                                                      const int8_t* input2,
                                                      const T*      bias,
                                                      const T*      gamma,
                                                      const T*      beta,
                                                      int           m,
                                                      int           n,
                                                      const float*  input1_deQFactor_ptr,
                                                      const float*  input2_deQFactor_ptr,
                                                      const float*  output_scale_ptr)
{
    const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
    const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
    const float output_scale     = __ldg(output_scale_ptr);
    int         col_start        = threadIdx.x << 2;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    float  local_out[4];
    int    outIdx       = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 2;
    char4* outTmpPtr    = (char4*)output;
    char4* input1TmpPtr = (char4*)input1;
    char4* input2TmpPtr = (char4*)input2;
    char4  input1Tmp    = __ldg(input1TmpPtr + outIdx);
    char4  input2Tmp    = __ldg(input2TmpPtr + outIdx);

    int col_start_tmp = col_start;
    local_out[0]      = static_cast<float>(input2Tmp.x) * input2_deQFactor
                   + static_cast<float>(input1Tmp.x) * input1_deQFactor
                   + static_cast<float>(__ldg(bias + col_start_tmp));
    col_start_tmp = col_start_tmp + 1;
    local_out[1]  = static_cast<float>(input2Tmp.y) * input2_deQFactor
                   + static_cast<float>(input1Tmp.y) * input1_deQFactor
                   + static_cast<float>(__ldg(bias + col_start_tmp));
    col_start_tmp = col_start_tmp + 1;
    local_out[2]  = static_cast<float>(input2Tmp.z) * input2_deQFactor
                   + static_cast<float>(input1Tmp.z) * input1_deQFactor
                   + static_cast<float>(__ldg(bias + col_start_tmp));
    col_start_tmp = col_start_tmp + 1;
    local_out[3]  = static_cast<float>(input2Tmp.w) * input2_deQFactor
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
    variance     = blockReduceSum<float>(local_out[0] * local_out[0] + local_out[1] * local_out[1]
                                     + local_out[2] * local_out[2] + local_out[3] * local_out[3]);
    if (threadIdx.x == 0) {
        s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
        s_variance = rsqrtf(s_variance);
    }
    __syncthreads();

    local_out[0] = (local_out[0] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                   + static_cast<float>(__ldg(beta + col_start));
    input2Tmp.x = float_to_int8_rn(local_out[0] * output_scale);

    col_start    = col_start + 1;
    local_out[1] = (local_out[1] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                   + static_cast<float>(__ldg(beta + col_start));
    input2Tmp.y = float_to_int8_rn(local_out[1] * output_scale);

    col_start    = col_start + 1;
    local_out[2] = (local_out[2] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                   + static_cast<float>(__ldg(beta + col_start));
    input2Tmp.z = float_to_int8_rn(local_out[2] * output_scale);

    col_start    = col_start + 1;
    local_out[3] = (local_out[3] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                   + static_cast<float>(__ldg(beta + col_start));
    input2Tmp.w = float_to_int8_rn(local_out[3] * output_scale);

    outTmpPtr[outIdx] = input2Tmp;
}

template<>
__global__ void add_bias_input_layernorm_COL32_int8IO(int8_t*       output,
                                                      const int8_t* input1,
                                                      const int8_t* input2,
                                                      const half2*  bias,
                                                      const half2*  gamma,
                                                      const half2*  beta,
                                                      int           m,
                                                      int           n,
                                                      const float*  input1_deQFactor_ptr,
                                                      const float*  input2_deQFactor_ptr,
                                                      const float*  output_scale_ptr)
{
    const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
    const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
    const float output_scale     = __ldg(output_scale_ptr);
    int         col_start        = threadIdx.x << 2;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    float  local_out[4];
    int    outIdx       = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 2;
    char4* outTmpPtr    = (char4*)output;
    char4* input1TmpPtr = (char4*)input1;
    char4* input2TmpPtr = (char4*)input2;
    char4  input1Tmp    = __ldg(input1TmpPtr + outIdx);
    char4  input2Tmp    = __ldg(input2TmpPtr + outIdx);

    int   col_start_tmp = col_start;
    half2 biasTmp       = __ldg(bias + (col_start_tmp >> 1));
    local_out[0]        = static_cast<float>(input2Tmp.x) * input2_deQFactor
                   + static_cast<float>(input1Tmp.x) * input1_deQFactor + static_cast<float>(biasTmp.x);
    col_start_tmp = col_start_tmp + 1;
    local_out[1]  = static_cast<float>(input2Tmp.y) * input2_deQFactor
                   + static_cast<float>(input1Tmp.y) * input1_deQFactor + static_cast<float>(biasTmp.y);

    col_start_tmp = col_start_tmp + 1;
    biasTmp       = __ldg(bias + (col_start_tmp >> 1));
    local_out[2]  = static_cast<float>(input2Tmp.z) * input2_deQFactor
                   + static_cast<float>(input1Tmp.z) * input1_deQFactor + static_cast<float>(biasTmp.x);
    col_start_tmp = col_start_tmp + 1;
    local_out[3]  = static_cast<float>(input2Tmp.w) * input2_deQFactor
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
    variance     = blockReduceSum<float>(local_out[0] * local_out[0] + local_out[1] * local_out[1]
                                     + local_out[2] * local_out[2] + local_out[3] * local_out[3]);
    if (threadIdx.x == 0) {
        s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
        s_variance = rsqrtf(s_variance);
    }
    __syncthreads();

    col_start_tmp = col_start >> 1;
    biasTmp       = __ldg(gamma + col_start_tmp);
    half2 betaTmp = __ldg(beta + col_start_tmp);

    local_out[0] = (local_out[0] * s_variance) * static_cast<float>(biasTmp.x) + static_cast<float>(betaTmp.x);
    input2Tmp.x  = float_to_int8_rn(local_out[0] * output_scale);

    col_start    = col_start + 1;
    local_out[1] = (local_out[1] * s_variance) * static_cast<float>(biasTmp.y) + static_cast<float>(betaTmp.y);
    input2Tmp.y  = float_to_int8_rn(local_out[1] * output_scale);

    col_start     = col_start + 1;
    col_start_tmp = col_start >> 1;
    biasTmp       = __ldg(gamma + col_start_tmp);
    betaTmp       = __ldg(beta + col_start_tmp);
    local_out[2]  = (local_out[2] * s_variance) * static_cast<float>(biasTmp.x) + static_cast<float>(betaTmp.x);
    input2Tmp.z   = float_to_int8_rn(local_out[2] * output_scale);

    col_start    = col_start + 1;
    local_out[3] = (local_out[3] * s_variance) * static_cast<float>(biasTmp.y) + static_cast<float>(betaTmp.y);
    input2Tmp.w  = float_to_int8_rn(local_out[3] * output_scale);

    outTmpPtr[outIdx] = input2Tmp;
}

template<typename T>
void invokeAddBiasResidualLayerNormCol32(int8_t*       output,
                                         const int8_t* input1,
                                         const int8_t* input2,
                                         const T*      bias,
                                         const T*      gamma,
                                         const T*      beta,
                                         int           m,
                                         int           n,
                                         cudaStream_t  stream,
                                         const float*  input1_deQFactor_ptr,
                                         const float*  input2_deQFactor_ptr,
                                         const float*  output_scale_ptr)
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

template void invokeAddBiasResidualLayerNormCol32(int8_t*       output,
                                                  const int8_t* input1,
                                                  const int8_t* input2,
                                                  const float*  bias,
                                                  const float*  gamma,
                                                  const float*  beta,
                                                  int           m,
                                                  int           n,
                                                  cudaStream_t  stream,
                                                  const float*  input1_deQFactor_ptr,
                                                  const float*  input2_deQFactor_ptr,
                                                  const float*  output_scale_ptr);

template void invokeAddBiasResidualLayerNormCol32(int8_t*       output,
                                                  const int8_t* input1,
                                                  const int8_t* input2,
                                                  const half*   bias,
                                                  const half*   gamma,
                                                  const half*   beta,
                                                  int           m,
                                                  int           n,
                                                  cudaStream_t  stream,
                                                  const float*  input1_deQFactor_ptr,
                                                  const float*  input2_deQFactor_ptr,
                                                  const float*  output_scale_ptr);

// input1/input2/out matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n)
//(grid, block) must be (m, n)
template<typename T>
__global__ void add_bias_input_layernorm_COL32_int8I_DataTypeO(T*            output,
                                                               const int8_t* input1,
                                                               const int8_t* input2,
                                                               const T*      bias,
                                                               const T*      gamma,
                                                               const T*      beta,
                                                               int           m,
                                                               int           n,
                                                               const float*  input1_deQFactor_ptr,
                                                               const float*  input2_deQFactor_ptr)
{
    const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
    const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
    int         col_start        = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    float local_out;
    int   idx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31));

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
__global__ void add_bias_input_layernorm_COL32_int8I_DataTypeO(half2*        output,
                                                               const int8_t* input1,
                                                               const int8_t* input2,
                                                               const half2*  bias,
                                                               const half2*  gamma,
                                                               const half2*  beta,
                                                               int           m,
                                                               int           n,
                                                               const float*  input1_deQFactor_ptr,
                                                               const float*  input2_deQFactor_ptr)
{
    const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
    const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
    int         col_start        = threadIdx.x << 1;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    float2 local_out;
    int    idx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 1;

    const char2* input1_ptr2 = (const char2*)input1;
    const char2* input2_ptr2 = (const char2*)input2;
    char2        input_tmp1  = __ldg(input1_ptr2 + idx);
    char2        input_tmp2  = __ldg(input2_ptr2 + idx);

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
    half2 beta_tmp  = __ldg(beta + threadIdx.x);

    local_out.x = (local_out.x * s_variance) * static_cast<float>(gamma_tmp.x) + static_cast<float>(beta_tmp.x);
    local_out.y = (local_out.y * s_variance) * static_cast<float>(gamma_tmp.y) + static_cast<float>(beta_tmp.y);

    bias_tmp.x = half(local_out.x);
    bias_tmp.y = half(local_out.y);

    output[idx] = bias_tmp;
}

template<typename T>
void invokeAddBiasResidualLayerNormCol32(T*            output,
                                         const int8_t* input1,
                                         const int8_t* input2,
                                         const T*      bias,
                                         const T*      gamma,
                                         const T*      beta,
                                         int           m,
                                         int           n,
                                         cudaStream_t  stream,
                                         const float*  input1_deQFactor_ptr,
                                         const float*  input2_deQFactor_ptr)
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

template void invokeAddBiasResidualLayerNormCol32<float>(float*        output,
                                                         const int8_t* input1,
                                                         const int8_t* input2,
                                                         const float*  bias,
                                                         const float*  gamma,
                                                         const float*  beta,
                                                         int           m,
                                                         int           n,
                                                         cudaStream_t  stream,
                                                         const float*  input1_deQFactor_ptr,
                                                         const float*  input2_deQFactor_ptr);

template void invokeAddBiasResidualLayerNormCol32<half>(half*         output,
                                                        const int8_t* input1,
                                                        const int8_t* input2,
                                                        const half*   bias,
                                                        const half*   gamma,
                                                        const half*   beta,
                                                        int           m,
                                                        int           n,
                                                        cudaStream_t  stream,
                                                        const float*  input1_deQFactor_ptr,
                                                        const float*  input2_deQFactor_ptr);

/*******************  invokeAddBiasLayernormAddRes  ***********************/
template<typename T, int T_per_thread>
__global__ void add_bias_layernorm_add_res(int8_t*       out_int8,
                                           const int8_t* out,
                                           T*            residual_input,
                                           const T*      bias,
                                           const T*      gamma,
                                           const T*      beta,
                                           const float   layernorm_eps,
                                           int           m,
                                           int           n,
                                           const float*  input_deQFactor_ptr,
                                           const float*  out_QFactor_ptr)
{
    const float      input_deQFactor = __ldg(input_deQFactor_ptr);
    const float      out_QFactor     = out_QFactor_ptr == nullptr ? -1.0f : __ldg(out_QFactor_ptr);
    int              tid             = threadIdx.x;
    const int        bdim            = blockDim.x;
    __shared__ float s_mean;
    __shared__ float s_variance;

    float local_val[T_per_thread];

    // float sums[2] = {0.0f, 0.0f};
    float local_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < T_per_thread; i++) {
        int col_start = i * bdim + tid;
        if (col_start < n) {
            T bias_val = bias[col_start];

            int inIdx    = (col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31);
            T   out_val  = static_cast<T>(out[inIdx]) * static_cast<T>(input_deQFactor);
            T   tmp      = out_val + bias_val;
            local_val[i] = tmp;
            local_sum += local_val[i];
        }
    }

    blockReduceSum<float>(local_sum);

    if (threadIdx.x == 0) {
        s_mean = local_sum * __fdividef(1.0f, n);
    }
    __syncthreads();

    local_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < T_per_thread; i++) {
        local_sum += (local_val[i] - s_mean) * (local_val[i] - s_mean);
    }

    blockReduceSum<float>(local_sum);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(local_sum * __fdividef(1.0f, n) + layernorm_eps);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < T_per_thread; i++) {
        int col_start = i * bdim + tid;
        if (col_start < n) {
            T gamma_val = gamma[col_start];
            T beta_val  = beta[col_start];

            int   outIdx  = (col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31);
            T     res_val = residual_input[outIdx];
            float output_val =
                (local_val[i] - s_mean) * s_variance * (float)(gamma_val) + (float)(beta_val) + (float)(res_val);

            if (out_QFactor_ptr != nullptr) {
                residual_input[outIdx] = (T)output_val;
                out_int8[outIdx]       = float_to_int8_rn(output_val * out_QFactor);
            }
            else {
                T* out_T_ptr      = reinterpret_cast<T*>(out_int8);
                out_T_ptr[outIdx] = (T)output_val;
            }
        }
    }
}

template<int T_per_thread>
__global__ void add_bias_layernorm_add_res_e2(char2*       out_int8,
                                              const char2* out,
                                              half2*       residual_input,
                                              const half2* bias,
                                              const half2* gamma,
                                              const half2* beta,
                                              const float  layernorm_eps,
                                              int          m,
                                              int          n,
                                              const float* input_deQFactor_ptr,
                                              const float* out_QFactor_ptr)
{
    const float      input_deQFactor = __ldg(input_deQFactor_ptr);
    const float      out_QFactor     = out_QFactor_ptr == nullptr ? -1.0f : __ldg(out_QFactor_ptr);
    int              tid             = threadIdx.x;
    const int        bdim            = blockDim.x;
    __shared__ float s_mean;
    __shared__ float s_variance;
    const int        n_2 = n >> 1;

    half2 local_val[T_per_thread];

    float local_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < T_per_thread; i++) {
        // int index = i * bdim + tid;
        int col_start = i * bdim + tid;
        if (col_start < n_2) {
            half2 bias_val = bias[col_start];

            int   col_id = col_start << 1;
            int   inIdx  = ((col_id & 0xffffffe0) * m + (blockIdx.x << 5) + (col_id & 31)) >> 1;
            char2 outTmp = out[inIdx];
            half2 out_val;
            out_val.x = static_cast<half>(outTmp.x) * static_cast<half>(input_deQFactor);
            out_val.y = static_cast<half>(outTmp.y) * static_cast<half>(input_deQFactor);

            half2 tmp         = __hadd2(out_val, bias_val);
            local_val[i]      = tmp;
            float2 tmp_float2 = __half22float2(tmp);
            local_sum += tmp_float2.x + tmp_float2.y;
        }
    }

    local_sum = blockReduceSum<float>(local_sum);

    if (threadIdx.x == 0) {
        s_mean = local_sum * __fdividef(1.0f, n);
    }
    __syncthreads();

    local_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < T_per_thread; i++) {
        int col_start = i * bdim + tid;
        if (col_start < n_2) {
            float2 tmp_float2 = __half22float2(local_val[i]);
            local_sum +=
                (tmp_float2.x - s_mean) * (tmp_float2.x - s_mean) + (tmp_float2.y - s_mean) * (tmp_float2.y - s_mean);
        }
    }

    local_sum = blockReduceSum<float>(local_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(local_sum * __fdividef(1.0f, n) + layernorm_eps);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < T_per_thread; i++) {
        int col_start = i * bdim + tid;
        if (col_start < n_2) {
            float2 gamma_val = __half22float2(gamma[col_start]);
            float2 beta_val  = __half22float2(beta[col_start]);

            int    col_id  = col_start << 1;
            int    outIdx  = ((col_id & 0xffffffe0) * m + (blockIdx.x << 5) + (col_id & 31)) >> 1;
            float2 res_val = __half22float2(residual_input[outIdx]);
            float2 output_val;
            float2 tmp_float2 = __half22float2(local_val[i]);
            output_val.x      = (tmp_float2.x - s_mean) * s_variance * gamma_val.x + beta_val.x + res_val.x;
            output_val.y      = (tmp_float2.y - s_mean) * s_variance * gamma_val.y + beta_val.y + res_val.y;
            if (out_QFactor_ptr != nullptr) {
                residual_input[outIdx] = __float22half2_rn(output_val);
                char2 outTmp;
                outTmp.x         = float_to_int8_rn(output_val.x * out_QFactor);
                outTmp.y         = float_to_int8_rn(output_val.y * out_QFactor);
                out_int8[outIdx] = outTmp;
            }
            else {
                half2* out_ptr  = reinterpret_cast<half2*>(out_int8);
                out_ptr[outIdx] = __float22half2_rn(output_val);
            }
        }
    }
}

#define MACRO_ADD_BIAS_LN_ADD_RES(T_per_TVec)                                                                          \
    blockSize = (blockSize + TVec_per_thread - 1) / TVec_per_thread;                                                   \
    if (T_per_TVec == 2) {                                                                                             \
        if (std::is_same<T, half>::value) {                                                                            \
            add_bias_layernorm_add_res_e2<TVec_per_thread><<<m, blockSize, 0, stream>>>((char2*)out_int8,              \
                                                                                        (const char2*)out,             \
                                                                                        (half2*)residual_input,        \
                                                                                        (const half2*)bias,            \
                                                                                        (const half2*)gamma,           \
                                                                                        (const half2*)beta,            \
                                                                                        layernorm_eps,                 \
                                                                                        m,                             \
                                                                                        n,                             \
                                                                                        input_deQFactor_ptr,           \
                                                                                        out_QFactor_ptr);              \
        }                                                                                                              \
        else {                                                                                                         \
            FT_CHECK_WITH_INFO(false, "[invokeAddBiasLayernormAddRes] unsupported dataType.");                         \
        }                                                                                                              \
    }                                                                                                                  \
    else {                                                                                                             \
        if (std::is_same<T, half>::value) {                                                                            \
            add_bias_layernorm_add_res<half, TVec_per_thread><<<m, blockSize, 0, stream>>>((int8_t*)out_int8,          \
                                                                                           (const int8_t*)out,         \
                                                                                           (half*)residual_input,      \
                                                                                           (const half*)bias,          \
                                                                                           (const half*)gamma,         \
                                                                                           (const half*)beta,          \
                                                                                           layernorm_eps,              \
                                                                                           m,                          \
                                                                                           n,                          \
                                                                                           input_deQFactor_ptr,        \
                                                                                           out_QFactor_ptr);           \
        }                                                                                                              \
        else {                                                                                                         \
            FT_CHECK_WITH_INFO(false, "[invokeAddBiasLayernormAddRes] unsupported dataType.");                         \
        }                                                                                                              \
    }

template<typename T>
void invokeAddBiasLayernormAddResCol32(int8_t*      out_int8,
                                       int8_t*      out,
                                       T*           residual_input,
                                       const T*     bias,
                                       const T*     gamma,
                                       const T*     beta,
                                       const float  layernorm_eps,
                                       int          m,
                                       int          n,
                                       cudaStream_t stream,
                                       const float* input_deQFactor_ptr,
                                       const float* out_QFactor_ptr)
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
            FT_CHECK_WITH_INFO(false, "[invokeAddBiasLayernormAddRes] unsupported dataType.");
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
            FT_CHECK_WITH_INFO(false, "[invokeAddBiasLayernormAddRes] unsupported dataType.");
        }
    }
}

template void invokeAddBiasLayernormAddResCol32(int8_t*      out_int8,
                                                int8_t*      out,
                                                float*       residual_input,
                                                const float* bias,
                                                const float* gamma,
                                                const float* beta,
                                                const float  layernorm_eps,
                                                int          m,
                                                int          n,
                                                cudaStream_t stream,
                                                const float* input_deQFactor_ptr,
                                                const float* out_QFactor_ptr);

template void invokeAddBiasLayernormAddResCol32(int8_t*      out_int8,
                                                int8_t*      out,
                                                half*        residual_input,
                                                const half*  bias,
                                                const half*  gamma,
                                                const half*  beta,
                                                const float  layernorm_eps,
                                                int          m,
                                                int          n,
                                                cudaStream_t stream,
                                                const float* input_deQFactor_ptr,
                                                const float* out_QFactor_ptr);

/*******************  invokeGeneralAddBiasResidualPreLayerNormCol32  ***********************/

// each warp process 1 row, and increase the num of warps per block to increase occupancy
template<int WPB, int VPT, bool WRITE_RESIDUAL, bool IS_BIAS, bool IS_INPUT>
__global__ void add_bias_input_layernorm_COL32_int8IO_warpReduce(int8_t*       output,
                                                                 int8_t*       input1,
                                                                 __half*       residual,
                                                                 const __half* bias,
                                                                 const __half* gamma,
                                                                 const __half* beta,
                                                                 int           m,
                                                                 const int     n,
                                                                 const float   dqScaleIn,
                                                                 const float   qScale)
{
    const int lane_id   = threadIdx.x % 32;
    const int warp_id   = threadIdx.x / 32;
    const int col_start = lane_id * VPT;
    const int row_id    = blockIdx.x * WPB + warp_id;

    const int idx = (col_start & 0xffffffe0) * m + (row_id << 5) + (col_start & 31);

    if (row_id >= m || col_start >= n)
        return;

    int8_t in_local[VPT];

    __half in_local_dq[VPT];
    __half bias_local[VPT];
    __half gamma_local[VPT];
    copy<sizeof(__half) * VPT>(&residual[idx], in_local_dq);
    if (IS_INPUT) {
        copy<sizeof(int8_t) * VPT>(&input1[idx], in_local);
    }
    if (IS_BIAS) {
        copy<sizeof(__half) * VPT>(&bias[col_start], bias_local);
    }

    float local_sum = 0.0f;
#pragma unroll
    for (int it = 0; it < VPT; it++) {
        // DQ input
        if (IS_INPUT) {
            in_local_dq[it] += dqScaleIn * in_local[it];
        }
        if (IS_BIAS) {
            in_local_dq[it] += bias_local[it];
        }
        local_sum += (float)in_local_dq[it];
    }
    // load parameters
    copy<sizeof(__half) * VPT>(&beta[col_start], bias_local);
    copy<sizeof(__half) * VPT>(&gamma[col_start], gamma_local);

    __shared__ float mu[WPB];      // mean
    __shared__ float rsigma[WPB];  // 1 / std.dev.

    local_sum = warpReduceSum<float>(local_sum);

    if (lane_id == 0) {
        mu[warp_id] = local_sum * __fdividef(1.0f, n);
    }
    __syncthreads();

    local_sum = 0.0f;
#pragma unroll
    for (int it = 0; it < VPT; it++) {
        local_sum += ((float)in_local_dq[it] - mu[warp_id]) * ((float)in_local_dq[it] - mu[warp_id]);
    }
    local_sum = warpReduceSum<float>(local_sum);
    if (lane_id == 0) {
        rsigma[warp_id] = rsqrtf(local_sum * __fdividef(1.0f, n) + 1e-5);
    }
    __syncthreads();
    static_assert(VPT % 4 == 0, "");
    uint32_t out_local[VPT / 4];
#pragma unroll
    for (int it = 0; it < VPT / 4; it++) {
        const float tmp0 =
            (float)gamma_local[it * 4 + 0] * ((float)in_local_dq[it * 4 + 0] - mu[warp_id]) * rsigma[warp_id]
            + (float)bias_local[it * 4 + 0];
        const float tmp1 =
            (float)gamma_local[it * 4 + 1] * ((float)in_local_dq[it * 4 + 1] - mu[warp_id]) * rsigma[warp_id]
            + (float)bias_local[it * 4 + 1];
        const float tmp2 =
            (float)gamma_local[it * 4 + 2] * ((float)in_local_dq[it * 4 + 2] - mu[warp_id]) * rsigma[warp_id]
            + (float)bias_local[it * 4 + 2];
        const float tmp3 =
            (float)gamma_local[it * 4 + 3] * ((float)in_local_dq[it * 4 + 3] - mu[warp_id]) * rsigma[warp_id]
            + (float)bias_local[it * 4 + 3];
        out_local[it] = float4_to_char4(tmp0 * qScale, tmp1 * qScale, tmp2 * qScale, tmp3 * qScale);
    }

    copy<sizeof(int8_t) * VPT>(out_local, &output[idx]);
    if (WRITE_RESIDUAL)
        copy<sizeof(__half) * VPT>(in_local_dq, &residual[idx]);
}

template<int VPT, bool WRITE_RESIDUAL, bool IS_BIAS, bool IS_INPUT>
__global__ void add_bias_input_layernorm_COL32_int8IO(int8_t*       output,
                                                      int8_t*       input1,
                                                      __half*       residual,
                                                      const __half* bias,
                                                      const __half* gamma,
                                                      const __half* beta,
                                                      int           m,
                                                      const int     n,
                                                      const float   dqScaleIn,
                                                      const float   qScale)
{
    // compute idx based on COL32
    // [m,n]-COL32 (blockIdx.x, threadIdx.x * VPT)
    const int col_start = threadIdx.x * VPT;
    const int idx       = (col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31);

    int8_t in_local[VPT];

    __half in_local_dq[VPT];
    __half bias_local[VPT];
    __half gamma_local[VPT];
    copy<sizeof(__half) * VPT>(&residual[idx], in_local_dq);
    if (IS_INPUT) {
        copy<sizeof(int8_t) * VPT>(&input1[idx], in_local);
    }
    if (IS_BIAS) {
        copy<sizeof(__half) * VPT>(&bias[col_start], bias_local);
    }

    float local_sum = 0.0f;
#pragma unroll
    for (int it = 0; it < VPT; it++) {
        // DQ input
        if (IS_INPUT) {
            in_local_dq[it] += dqScaleIn * in_local[it];
        }
        if (IS_BIAS) {
            in_local_dq[it] += bias_local[it];
        }
        local_sum += (float)in_local_dq[it];
    }
    // load parameters
    copy<sizeof(__half) * VPT>(&beta[col_start], bias_local);
    copy<sizeof(__half) * VPT>(&gamma[col_start], gamma_local);

    __shared__ float mu;      // mean
    __shared__ float rsigma;  // 1 / std.dev.

    local_sum = blockReduceSum<float>(local_sum);

    if (threadIdx.x == 0) {
        mu = local_sum * __fdividef(1.0f, n);
    }
    __syncthreads();

    local_sum = 0.0f;
#pragma unroll
    for (int it = 0; it < VPT; it++) {
        local_sum += ((float)in_local_dq[it] - mu) * ((float)in_local_dq[it] - mu);
    }
    local_sum = blockReduceSum<float>(local_sum);
    if (threadIdx.x == 0) {
        rsigma = rsqrtf(local_sum * __fdividef(1.0f, n) + 1e-5);
    }
    __syncthreads();

    static_assert(VPT % 4 == 0, "");
    uint32_t out_local[VPT / 4];
#pragma unroll
    for (int it = 0; it < VPT / 4; it++) {
        const float tmp0 = (float)gamma_local[it * 4 + 0] * ((float)in_local_dq[it * 4 + 0] - mu) * rsigma
                           + (float)bias_local[it * 4 + 0];
        const float tmp1 = (float)gamma_local[it * 4 + 1] * ((float)in_local_dq[it * 4 + 1] - mu) * rsigma
                           + (float)bias_local[it * 4 + 1];
        const float tmp2 = (float)gamma_local[it * 4 + 2] * ((float)in_local_dq[it * 4 + 2] - mu) * rsigma
                           + (float)bias_local[it * 4 + 2];
        const float tmp3 = (float)gamma_local[it * 4 + 3] * ((float)in_local_dq[it * 4 + 3] - mu) * rsigma
                           + (float)bias_local[it * 4 + 3];
        out_local[it] = float4_to_char4(tmp0 * qScale, tmp1 * qScale, tmp2 * qScale, tmp3 * qScale);
    }

    copy<sizeof(int8_t) * VPT>(out_local, &output[idx]);
    if (WRITE_RESIDUAL)
        copy<sizeof(__half) * VPT>(in_local_dq, &residual[idx]);
}

void invokeAddBiasResidualPreLayerNormCol32(int8_t*       output,
                                            int8_t*       input1,
                                            __half*       input2,
                                            const __half* bias,
                                            const __half* gamma,
                                            const __half* beta,
                                            int           m,
                                            const int     n,
                                            cudaStream_t  stream,
                                            const float   dqScaleIn,
                                            const float   qScale)
{
    dim3          grid(m);
    constexpr int VPT = 16 / sizeof(__half);
    if (n <= VPT * 32 && n % VPT == 0) {
        constexpr int TPB  = 1024 / VPT;
        constexpr int WPB  = (TPB + 31) / 32;
        int           grid = m / WPB;
        add_bias_input_layernorm_COL32_int8IO_warpReduce<WPB, VPT, true, true, true>
            <<<grid, TPB, 0, stream>>>(output, input1, input2, bias, gamma, beta, m, n, dqScaleIn, qScale);
    }
    else if (n / VPT <= 1024 && n % VPT == 0) {
        const int blockSize = n / VPT;
        add_bias_input_layernorm_COL32_int8IO<VPT, true, true, true>
            <<<grid, blockSize, 0, stream>>>(output, input1, input2, bias, gamma, beta, m, n, dqScaleIn, qScale);
    }
    else {
        printf("[FT ERROR]Unsupported dimension n=%d for layernorm, because n / 8 > 1024 || n %% 8 != 0. \n", n);
        exit(0);
    }
}

// input1/input2/out matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n)
//(grid, block) must be (m, n/4)
// using char4
template<typename T>
__global__ void add_bias_input_layernorm_COL32_int8IO_noRes(int8_t*      output,
                                                            int32_t*     input1,
                                                            T*           input2,
                                                            const T*     bias,
                                                            const T*     gamma,
                                                            const T*     beta,
                                                            int          m,
                                                            int          n,
                                                            const float* weight_amax,
                                                            const float* input1_amax_ptr,
                                                            const float* output_scale_ptr)
{
    const float input1_amax  = __ldg(input1_amax_ptr);
    const float output_scale = __ldg(output_scale_ptr);
    int         col_start    = threadIdx.x << 2;
    bool        qual         = (col_start < n);

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    float local_out[4];
    int   outIdx       = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 2;
    int4* input1TmpPtr = (int4*)input1;
    int4  input1Tmp;
    if (qual) {
        input1Tmp         = __ldg(input1TmpPtr + outIdx);
        int col_start_tmp = col_start;
        // NOTE: 0.000062f = 1 / 127/.0f / 127.0f
        local_out[0] = static_cast<float>(input1Tmp.x) * input1_amax * weight_amax[col_start_tmp] * 0.000062f
                       + static_cast<float>(input2[(outIdx << 2) + 0]) + static_cast<float>(bias[col_start_tmp]);
        input2[(outIdx << 2) + 0] = local_out[0];

        col_start_tmp = col_start_tmp + 1;
        local_out[1]  = static_cast<float>(input1Tmp.y) * input1_amax * weight_amax[col_start_tmp] * 0.000062f
                       + static_cast<float>(input2[(outIdx << 2) + 1]) + static_cast<float>(bias[col_start_tmp]);
        input2[(outIdx << 2) + 1] = local_out[1];

        col_start_tmp = col_start_tmp + 1;
        local_out[2]  = static_cast<float>(input1Tmp.z) * input1_amax * weight_amax[col_start_tmp] * 0.000062f
                       + static_cast<float>(input2[(outIdx << 2) + 2]) + static_cast<float>(bias[col_start_tmp]);
        input2[(outIdx << 2) + 2] = local_out[2];

        col_start_tmp = col_start_tmp + 1;
        local_out[3]  = static_cast<float>(input1Tmp.w) * input1_amax * weight_amax[col_start_tmp] * 0.000062f
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
    char4  outputTmp;
    if (qual) {
        local_out[0] = (local_out[0] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                       + static_cast<float>(__ldg(beta + col_start));
        outputTmp.x = float_to_int8_rn(local_out[0] * output_scale);

        col_start    = col_start + 1;
        local_out[1] = (local_out[1] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                       + static_cast<float>(__ldg(beta + col_start));
        outputTmp.y = float_to_int8_rn(local_out[1] * output_scale);

        col_start    = col_start + 1;
        local_out[2] = (local_out[2] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                       + static_cast<float>(__ldg(beta + col_start));
        outputTmp.z = float_to_int8_rn(local_out[2] * output_scale);

        col_start    = col_start + 1;
        local_out[3] = (local_out[3] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                       + static_cast<float>(__ldg(beta + col_start));
        outputTmp.w = float_to_int8_rn(local_out[3] * output_scale);

        outputTmpPtr[outIdx] = outputTmp;
    }
}

template<>
__global__ void add_bias_input_layernorm_COL32_int8IO_noRes(int8_t*      output,
                                                            int32_t*     input1,
                                                            half2*       input2,
                                                            const half2* bias,
                                                            const half2* gamma,
                                                            const half2* beta,
                                                            int          m,
                                                            int          n,
                                                            const float* weight_amax,
                                                            const float* input1_amax_ptr,
                                                            const float* output_scale_ptr)
{
    const float2* weight_scale_ptr = (const float2*)weight_amax;
    const float   input1_amax      = __ldg(input1_amax_ptr);
    const float   output_scale     = __ldg(output_scale_ptr);
    int           col_start        = threadIdx.x << 1;
    bool          qual             = (col_start < n);

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            sums[2] = {0.0f, 0.0f};
    // float mean = 0.0f;
    // float variance = 0.0f;

    float local_out[2];
    int   outIdx       = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 1;
    int2* input1TmpPtr = (int2*)input1;
    int2  input1Tmp;
    if (qual) {
        const float2 weight_scale = __ldg(weight_scale_ptr + threadIdx.x);
        input1Tmp                 = input1TmpPtr[outIdx];
        half2 biasTmp             = bias[threadIdx.x];
        half2 input2Tmp           = input2[outIdx];
        // NOTE: 0.000062f = 1 / 127/.0f / 127.0f
        local_out[0] =
            static_cast<float>(input1Tmp.x) * input1_amax * weight_scale.x * 0.000062f + static_cast<float>(biasTmp.x);
        local_out[1] =
            static_cast<float>(input1Tmp.y) * input1_amax * weight_scale.y * 0.000062f + static_cast<float>(biasTmp.y);

        local_out[0] += static_cast<float>(input2Tmp.x);
        local_out[1] += static_cast<float>(input2Tmp.y);

        input2Tmp.x    = local_out[0];
        input2Tmp.y    = local_out[1];
        input2[outIdx] = input2Tmp;
        for (int i = 0; i < 2; i++) {
            sums[0] += local_out[i];
            sums[1] += local_out[i] * local_out[i];
        }
    }

    blockReduceSumV2<float, 2>(sums);

    if (threadIdx.x == 0) {
        s_mean     = sums[0] * __fdividef(1.0f, n);
        s_variance = rsqrtf(sums[1] * __fdividef(1.0f, n) - s_mean * s_mean + 1e-6);
    }
    __syncthreads();

    char2* outputTmpPtr = (char2*)output;
    char2  outputTmp;
    if (qual) {
        half2 gammaTmp = gamma[threadIdx.x];
        half2 betaTmp  = beta[threadIdx.x];
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
void invokeAddBiasResidualPreLayerNormCol32(int8_t*      output,
                                            int32_t*     input1,
                                            T*           input2,
                                            const T*     bias,
                                            const T*     gamma,
                                            const T*     beta,
                                            int          m,
                                            int          n,
                                            cudaStream_t stream,
                                            const float* weight_amax,
                                            const float* input1_amax_ptr,
                                            const float* output_scale_ptr)
{
    dim3 grid(m);
    int  blockSize = (n / 4 + 31) / 32 * 32;
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

template void invokeAddBiasResidualPreLayerNormCol32(int8_t*      output,
                                                     int32_t*     input1,
                                                     float*       input2,
                                                     const float* bias,
                                                     const float* gamma,
                                                     const float* beta,
                                                     int          m,
                                                     int          n,
                                                     cudaStream_t stream,
                                                     const float* weight_amax,
                                                     const float* input1_amax_ptr,
                                                     const float* output_scale_ptr);

template void invokeAddBiasResidualPreLayerNormCol32(int8_t*      output,
                                                     int32_t*     input1,
                                                     half*        input2,
                                                     const half*  bias,
                                                     const half*  gamma,
                                                     const half*  beta,
                                                     int          m,
                                                     int          n,
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
    int         col_start    = threadIdx.x << 2;
    bool        qual         = (col_start < n);

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    float local_out[4];
    int   outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 2;

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

    char4  outTmp;
    char4* outPtr = (char4*)out;
    if (qual) {
        local_out[0] = (local_out[0] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                       + static_cast<float>(__ldg(beta + col_start));
        outTmp.x = float_to_int8_rn(local_out[0] * output_scale);

        col_start    = col_start + 1;
        local_out[1] = (local_out[1] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                       + static_cast<float>(__ldg(beta + col_start));
        outTmp.y = float_to_int8_rn(local_out[1] * output_scale);

        col_start    = col_start + 1;
        local_out[2] = (local_out[2] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                       + static_cast<float>(__ldg(beta + col_start));
        outTmp.z = float_to_int8_rn(local_out[2] * output_scale);

        col_start    = col_start + 1;
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
    int         col_start    = threadIdx.x << 1;
    bool        qual         = (col_start < n);

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            sums[2] = {0.0f, 0.0f};
    // float mean = 0.0f;
    // float variance = 0.0f;

    float local_out[2];
    int   outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 1;

    if (qual) {
        half2 inputTmp = input[outIdx];
        local_out[0]   = static_cast<float>(inputTmp.x);
        local_out[1]   = static_cast<float>(inputTmp.y);

        for (int i = 0; i < 2; i++) {
            sums[0] += local_out[i];
            sums[1] += local_out[i] * local_out[i];
        }
    }

    blockReduceSumV2<float, 2>(sums);

    if (threadIdx.x == 0) {
        s_mean     = sums[0] * __fdividef(1.0f, n);
        s_variance = rsqrtf(sums[1] * __fdividef(1.0f, n) - s_mean * s_mean + 1e-6);
    }
    __syncthreads();

    char2  outTmp;
    char2* outPtr = (char2*)out;
    if (qual) {
        half2 gammaTmp = gamma[threadIdx.x];
        half2 betaTmp  = beta[threadIdx.x];
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
void invokeLayernormCol32(int8_t*      out,
                          const T*     input,
                          const T*     gamma,
                          const T*     beta,
                          int          m,
                          int          n,
                          const float* output_scale_ptr,
                          cudaStream_t stream)
{
    dim3 grid(m);
    int  blockSize = (n / 4 + 31) / 32 * 32;
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

template void invokeLayernormCol32(int8_t*      out,
                                   const float* input,
                                   const float* gamma,
                                   const float* beta,
                                   int          m,
                                   int          n,
                                   const float* output_scale_ptr,
                                   cudaStream_t stream);

template void invokeLayernormCol32(int8_t*      out,
                                   const half*  input,
                                   const half*  gamma,
                                   const half*  beta,
                                   int          m,
                                   int          n,
                                   const float* output_scale_ptr,
                                   cudaStream_t stream);

/*******************  invokeLayernormCol32  ***********************/

// input1/input2/out matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n)
//(grid, block) must be (m, n/4)
// using char4
template<typename T>
__global__ void layernorm_COL32_INT8I_DataTypeO(
    T* out, const int8_t* input, const T* gamma, const T* beta, int m, int n, const float* input_deQ_ptr)
{
    const float input_deQ = __ldg(input_deQ_ptr);
    int         col_start = threadIdx.x << 2;
    bool        qual      = (col_start < n);

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    float local_out[4];
    int   outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 2;

    char4* inputPtr = (char4*)input;
    if (qual) {
        char4 inputTmp = inputPtr[outIdx];
        local_out[0]   = static_cast<float>(inputTmp.x) * input_deQ;
        local_out[1]   = static_cast<float>(inputTmp.y) * input_deQ;
        local_out[2]   = static_cast<float>(inputTmp.z) * input_deQ;
        local_out[3]   = static_cast<float>(inputTmp.w) * input_deQ;
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

    float4  outTmp;
    float4* outPtr = (float4*)out;
    if (qual) {
        local_out[0] = (local_out[0] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                       + static_cast<float>(__ldg(beta + col_start));
        outTmp.x = local_out[0];

        col_start    = col_start + 1;
        local_out[1] = (local_out[1] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                       + static_cast<float>(__ldg(beta + col_start));
        outTmp.y = local_out[1];

        col_start    = col_start + 1;
        local_out[2] = (local_out[2] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                       + static_cast<float>(__ldg(beta + col_start));
        outTmp.z = local_out[2];

        col_start    = col_start + 1;
        local_out[3] = (local_out[3] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                       + static_cast<float>(__ldg(beta + col_start));
        outTmp.w = local_out[3];

        outPtr[outIdx] = outTmp;
    }
}

template<>
__global__ void layernorm_COL32_INT8I_DataTypeO(
    half2* out, const int8_t* input, const half2* gamma, const half2* beta, int m, int n, const float* input_deQ_ptr)
{
    const float input_deQ = __ldg(input_deQ_ptr);

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            sums[2] = {0.0f, 0.0f};

#pragma unroll
    for (int i = threadIdx.x; i < n / 2; i += blockDim.x) {
        int    col_start = i << 1;
        float  local_out[2];
        int    outIdx   = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 1;
        char2* inputPtr = (char2*)input;
        char2  inputTmp = inputPtr[outIdx];
        local_out[0]    = static_cast<float>(inputTmp.x) * input_deQ;
        local_out[1]    = static_cast<float>(inputTmp.y) * input_deQ;

        for (int i = 0; i < 2; i++) {
            sums[0] += local_out[i];
            sums[1] += local_out[i] * local_out[i];
        }
    }

    blockReduceSumV2<float, 2>(sums);

    if (threadIdx.x == 0) {
        s_mean     = sums[0] * __fdividef(1.0f, n);
        s_variance = rsqrtf(sums[1] * __fdividef(1.0f, n) - s_mean * s_mean + 1e-6);
    }
    __syncthreads();

    float2 outTmp;

#pragma unroll
    for (int i = threadIdx.x; i < n / 2; i += blockDim.x) {
        int    col_start = i << 1;
        float  local_out[2];
        int    outIdx   = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 1;
        char2* inputPtr = (char2*)input;
        char2  inputTmp = inputPtr[outIdx];
        local_out[0]    = static_cast<float>(inputTmp.x) * input_deQ;
        local_out[1]    = static_cast<float>(inputTmp.y) * input_deQ;

        half2 gammaTmp = gamma[i];
        half2 betaTmp  = beta[i];
        outTmp.x =
            (local_out[0] - s_mean) * s_variance * static_cast<float>(gammaTmp.x) + static_cast<float>(betaTmp.x);
        outTmp.y =
            (local_out[1] - s_mean) * s_variance * static_cast<float>(gammaTmp.y) + static_cast<float>(betaTmp.y);

        out[outIdx] = __float22half2_rn(outTmp);
    }
}

template<typename T>
void invokeLayernormCol32(T*            out,
                          const int8_t* input,
                          const T*      gamma,
                          const T*      beta,
                          int           m,
                          int           n,
                          const float*  input_deQ_ptr,
                          cudaStream_t  stream)
{
    dim3 grid(m);
    if (sizeof(T) == sizeof(half)) {
        int blockSize = (n / 2 + 31) / 32 * 32;
        blockSize     = (blockSize <= 1024) ? blockSize : 1024;
        layernorm_COL32_INT8I_DataTypeO<<<grid, blockSize, 0, stream>>>(
            (half2*)out, input, (const half2*)gamma, (const half2*)beta, m, n, input_deQ_ptr);
    }
    else {
        int blockSize = (n / 4 + 31) / 32 * 32;
        blockSize     = (blockSize <= 1024) ? blockSize : 1024;
        layernorm_COL32_INT8I_DataTypeO<T>
            <<<grid, blockSize, 0, stream>>>(out, input, gamma, beta, m, n, input_deQ_ptr);
    }
}

template void invokeLayernormCol32(float*        out,
                                   const int8_t* input,
                                   const float*  gamma,
                                   const float*  beta,
                                   int           m,
                                   int           n,
                                   const float*  input_deQ_ptr,
                                   cudaStream_t  stream);

template void invokeLayernormCol32(half*         out,
                                   const int8_t* input,
                                   const half*   gamma,
                                   const half*   beta,
                                   int           m,
                                   int           n,
                                   const float*  input_deQ_ptr,
                                   cudaStream_t  stream);

/*******************  invokeLayernormShiftPartitionCol32  ***********************/

// each warp process 1 row, and increase the num of warps per block to increase occupancy
template<int WPB, int VPT>
__global__ void layernorm_shift_partition_COL32_warpReduce(int8_t*       out_ptr,
                                                           const __half* input_ptr,
                                                           const __half* gamma_ptr,
                                                           const __half* beta_ptr,
                                                           int           batch,
                                                           int           H,
                                                           int           W,
                                                           int           n,
                                                           const float   qScale,
                                                           int           shift_size,
                                                           int           window_size)
{
    const int lane_id   = threadIdx.x % 32;
    const int warp_id   = threadIdx.x / 32;
    const int col_start = lane_id * VPT;
    const int row_id    = blockIdx.x * WPB + warp_id;
    if (row_id >= batch * H * W || col_start >= n)
        return;

    const int blockIdx_z = row_id / (H * W);
    const int blockIdx_y = (row_id / W) % H;
    const int blockIdx_x = row_id % W;

    const int batch_offset = blockIdx_z * H * W;
    const int m            = batch * H * W;

    const int shifted_H_idx = (shift_size != 0) ? ((blockIdx_y - shift_size + H) % H) : blockIdx_y;
    const int shifted_W_idx = (shift_size != 0) ? ((blockIdx_x - shift_size + W) % W) : blockIdx_x;
    const int window_H_idx  = shifted_H_idx / window_size;
    const int window_W_idx  = shifted_W_idx / window_size;
    const int window_idx    = window_H_idx * (W / window_size) + window_W_idx;
    const int idx_in_window = (shifted_H_idx % window_size) * window_size + (shifted_W_idx % window_size);
    const int output_bid    = batch_offset + window_idx * window_size * window_size + idx_in_window;

    const int idx    = (col_start & 0xffffffe0) * m + (row_id << 5) + (col_start & 31);
    const int outIdx = (col_start & 0xffffffe0) * m + (output_bid << 5) + (col_start & 31);

    float  in_local_dq[VPT];
    __half in_local[VPT];
    __half beta_local[VPT];
    __half gamma_local[VPT];
    copy<sizeof(__half) * VPT>(&input_ptr[idx], in_local);
    copy<sizeof(__half) * VPT>(&gamma_ptr[col_start], gamma_local);
    copy<sizeof(__half) * VPT>(&beta_ptr[col_start], beta_local);

    float local_sum = 0.0f;
#pragma unroll
    for (int it = 0; it < VPT; it++) {
        in_local_dq[it] = in_local[it];
        local_sum += in_local_dq[it];
    }

    __shared__ float mu[WPB];
    __shared__ float rsigma[WPB];

    local_sum = warpReduceSum<float>(local_sum);

    if (lane_id == 0) {
        mu[warp_id] = local_sum * __fdividef(1.0f, n);
    }
    __syncthreads();

    local_sum = 0.0f;
#pragma unroll
    for (int it = 0; it < VPT; it++) {
        local_sum += (in_local_dq[it] - mu[warp_id]) * (in_local_dq[it] - mu[warp_id]);
    }
    local_sum = warpReduceSum<float>(local_sum);
    if (lane_id == 0) {
        rsigma[warp_id] = rsqrtf(local_sum * __fdividef(1.0f, n) + 1e-5);
    }
    __syncthreads();

    static_assert(VPT % 4 == 0, "");
    uint32_t out_local[VPT / 4];
#pragma unroll
    for (int it = 0; it < VPT; it++) {
        const float tmp0 =
            (in_local_dq[it * 4 + 0] - mu[warp_id]) * rsigma[warp_id] * static_cast<float>(gamma_local[it * 4 + 0])
            + static_cast<float>(beta_local[it * 4 + 0]);
        const float tmp1 =
            (in_local_dq[it * 4 + 1] - mu[warp_id]) * rsigma[warp_id] * static_cast<float>(gamma_local[it * 4 + 1])
            + static_cast<float>(beta_local[it * 4 + 1]);
        const float tmp2 =
            (in_local_dq[it * 4 + 2] - mu[warp_id]) * rsigma[warp_id] * static_cast<float>(gamma_local[it * 4 + 2])
            + static_cast<float>(beta_local[it * 4 + 2]);
        const float tmp3 =
            (in_local_dq[it * 4 + 3] - mu[warp_id]) * rsigma[warp_id] * static_cast<float>(gamma_local[it * 4 + 3])
            + static_cast<float>(beta_local[it * 4 + 3]);
        out_local[it] = float4_to_char4(tmp0 * qScale, tmp1 * qScale, tmp2 * qScale, tmp3 * qScale);
    }

    copy<sizeof(int8_t) * VPT>(out_local, &out_ptr[outIdx]);
}

template<int VPT>
__global__ void layernorm_shift_partition_COL32(int8_t*       out_ptr,
                                                const __half* input_ptr,
                                                const __half* gamma_ptr,
                                                const __half* beta_ptr,
                                                int           batch,
                                                int           H,
                                                int           W,
                                                int           n,
                                                const float   qScale,
                                                int           shift_size,
                                                int           window_size)
{
    const int batch_offset = blockIdx.z * gridDim.y * gridDim.x;
    const int m            = gridDim.z * gridDim.y * gridDim.x;
    const int bid          = batch_offset + blockIdx.y * gridDim.x + blockIdx.x;

    const int shifted_H_idx      = (shift_size != 0) ? ((blockIdx.y - shift_size + gridDim.y) % gridDim.y) : blockIdx.y;
    const int shifted_W_idx      = (shift_size != 0) ? ((blockIdx.x - shift_size + gridDim.x) % gridDim.x) : blockIdx.x;
    const int window_H_idx       = shifted_H_idx / window_size;
    const int window_W_idx       = shifted_W_idx / window_size;
    const int stride_of_window_H = W / window_size;
    const int window_idx         = window_H_idx * stride_of_window_H + window_W_idx;
    const int idx_in_window      = (shifted_H_idx % window_size) * window_size + (shifted_W_idx % window_size);
    const int output_bid         = batch_offset + window_idx * window_size * window_size + idx_in_window;

    int tid = threadIdx.x * VPT;
    if (tid >= n)
        return;
    __shared__ float mu;
    __shared__ float rsigma;
    __half           inputTmp[VPT];
    __half           beta_local[VPT];
    __half           gamma_local[VPT];
    float            in_local_dq[VPT];

    const int offset_col32 = (tid & 0xffffffe0) * m + (bid << 5) + (tid & 31);
    copy<sizeof(__half) * VPT>(&input_ptr[offset_col32], inputTmp);
    copy<sizeof(__half) * VPT>(&gamma_ptr[tid], gamma_local);
    copy<sizeof(__half) * VPT>(&beta_ptr[tid], beta_local);

    float local_sum = 0.0f;
#pragma unroll
    for (int it = 0; it < VPT; it++) {
        in_local_dq[it] = inputTmp[it];
        local_sum += in_local_dq[it];
    }

    local_sum = blockReduceSum<float>(local_sum);

    if (threadIdx.x == 0) {
        mu = local_sum * __fdividef(1.0f, n);
    }
    __syncthreads();

    local_sum = 0.0f;
#pragma unroll
    for (int it = 0; it < VPT; it++) {
        local_sum += (in_local_dq[it] - mu) * (in_local_dq[it] - mu);
    }
    local_sum = blockReduceSum<float>(local_sum);
    if (threadIdx.x == 0) {
        rsigma = rsqrtf(local_sum * __fdividef(1.0f, n) + 1e-5);
    }
    __syncthreads();

    static_assert(VPT % 4 == 0, "");
    uint32_t out_local[VPT / 4];
    if (tid < n) {
#pragma unroll
        for (int it = 0; it < VPT / 4; it++) {
            const float tmp0 = (in_local_dq[it * 4 + 0] - mu) * rsigma * static_cast<float>(gamma_local[it * 4 + 0])
                               + static_cast<float>(beta_local[it * 4 + 0]);
            const float tmp1 = (in_local_dq[it * 4 + 1] - mu) * rsigma * static_cast<float>(gamma_local[it * 4 + 1])
                               + static_cast<float>(beta_local[it * 4 + 1]);
            const float tmp2 = (in_local_dq[it * 4 + 2] - mu) * rsigma * static_cast<float>(gamma_local[it * 4 + 2])
                               + static_cast<float>(beta_local[it * 4 + 2]);
            const float tmp3 = (in_local_dq[it * 4 + 3] - mu) * rsigma * static_cast<float>(gamma_local[it * 4 + 3])
                               + static_cast<float>(beta_local[it * 4 + 3]);
            out_local[it] = float4_to_char4(tmp0 * qScale, tmp1 * qScale, tmp2 * qScale, tmp3 * qScale);
        }
        const int outIdx = (tid & 0xffffffe0) * m + (output_bid << 5) + (tid & 31);
        copy<sizeof(int8_t) * VPT>(out_local, &out_ptr[outIdx]);
    }
}

void invokeLayernormShiftPartitionCol32(int8_t*      out,
                                        const half*  input,
                                        const half*  gamma,
                                        const half*  beta,
                                        int          batch,
                                        int          H,
                                        int          W,
                                        int          n,
                                        const float  norm_scale,
                                        int          shift_size,
                                        int          window_size,
                                        cudaStream_t stream)
{
    dim3 grid(W, H, batch);

    constexpr int VPT       = 16 / sizeof(half);
    int           blockSize = (n / VPT + 31) / 32 * 32;
    if (blockSize == 32 && n % VPT == 0) {
        constexpr int TPB  = 1024 / VPT;
        constexpr int WPB  = (TPB + 31) / 32;
        int           grid = (batch * H * W + WPB - 1) / WPB;
        layernorm_shift_partition_COL32_warpReduce<WPB, VPT><<<grid, TPB, 0, stream>>>(out,
                                                                                       (const __half*)input,
                                                                                       (const __half*)gamma,
                                                                                       (const __half*)beta,
                                                                                       batch,
                                                                                       H,
                                                                                       W,
                                                                                       n,
                                                                                       norm_scale,
                                                                                       shift_size,
                                                                                       window_size);
    }
    else if (n / VPT <= 1024 && n % VPT == 0) {
        layernorm_shift_partition_COL32<VPT><<<grid, blockSize, 0, stream>>>(out,
                                                                             (const __half*)input,
                                                                             (const __half*)gamma,
                                                                             (const __half*)beta,
                                                                             batch,
                                                                             H,
                                                                             W,
                                                                             n,
                                                                             norm_scale,
                                                                             shift_size,
                                                                             window_size);
    }
    else {
        printf("[FT ERROR]Unsupported dimension n=%d for layernorm, because n / 8 > 1024 || n %% 8 != 0. \n", n);
        exit(0);
    }
}

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
                                   int          batch,
                                   const float* merge_inFactor,
                                   int          H,
                                   int          W,
                                   int          n)
{
    const int   ite       = 4;
    const int   tid       = threadIdx.x;
    const int   W_idx     = blockIdx.x;
    const int   H_idx     = blockIdx.y;
    const float out_scale = __ldg(merge_inFactor);
    // const size_t batch_offset = blockIdx.z * H * W * n;
    // const int input_H_stride = W*n/2;
    // const int output_H_stride = W*n;
    const int n_4 = n >> 2;
    const int m   = batch * 4 * H * W;

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
            int part_id     = col_id / n_4;
            int offset_in_W = part_id / 2;
            int offset_in_H = part_id % 2;
            // size_t input_id = batch_offset + (2*H_idx + offset_in_H)*input_H_stride + (2*W_idx + offset_in_W)*n_4 +
            // (col_id % n_4);

            int col_input = col_id % n_4;
            int row_input = blockIdx.z * H * W * 4 + (2 * H_idx + offset_in_H) * W * 2 + (2 * W_idx + offset_in_W);
            int input_idx_col32 = ((col_input >> 5) << 5) * m + (row_input << 5) + (col_input & 31);
            local_out[i]        = (float)(__ldg(input + input_idx_col32));
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

            int col_output        = col_id;
            int row_output        = blockIdx.z * H * W + H_idx * W + W_idx;
            int output_idx_col32  = ((col_output >> 5) << 5) * (m >> 2) + (row_output << 5) + (col_output & 31);
            out[output_idx_col32] = float_to_int8_rn(
                out_scale * (local_out[i] * s_variance * (float)__ldg(&gamma[col_id]) + (float)__ldg(&beta[col_id])));
        }
    }
}

// TODO : accelerate with half2
template<typename T>
void invokeMergeLayerNormCol32(int8_t*      output,
                               const T*     input,
                               const T*     gamma,
                               const T*     beta,
                               int          batch,
                               const float* merge_inFactor,
                               int          H,
                               int          W,
                               int          n,
                               cudaStream_t stream)
{
    if ((W % 2 != 0) || (H % 2 != 0)) {
        printf("[ERROR][invokeMergeLayerNormCol32] H(W) should be a multiple of 2.\n");
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
            <<<grid, blockSize, 0, stream>>>(output, input, gamma, beta, batch, merge_inFactor, H / 2, W / 2, n * 4);
    }
    /*
    else
      merge_layernorm<T><<<grid, blockSize, 0, stream>>>(output, input, gamma, beta, batch, H/2, W/2, n*4);
    */
}

template void invokeMergeLayerNormCol32<float>(int8_t*      output,
                                               const float* input,
                                               const float* gamma,
                                               const float* beta,
                                               int          batch,
                                               const float* merge_inFactor,
                                               int          H,
                                               int          W,
                                               int          n,
                                               cudaStream_t stream);

template void invokeMergeLayerNormCol32<half>(int8_t*      output,
                                              const half*  input,
                                              const half*  gamma,
                                              const half*  beta,
                                              int          batch,
                                              const float* merge_inFactor,
                                              int          H,
                                              int          W,
                                              int          n,
                                              cudaStream_t stream);

// input1/input2/out matrix with layout of row major (m*n)
//(grid, block) must be (m, n/4)
// using char4
template<typename T>
__global__ void add_bias_input_layernorm_ROW_int8IO(int8_t*       output,
                                                    const int8_t* input1,
                                                    const int8_t* input2,
                                                    const T*      bias,
                                                    const T*      gamma,
                                                    const T*      beta,
                                                    int           m,
                                                    int           n,
                                                    const float*  input1_deQFactor_ptr,
                                                    const float*  input2_deQFactor_ptr,
                                                    const float*  output_scale_ptr)
{
    const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
    const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
    const float output_scale     = __ldg(output_scale_ptr);
    int         col_start        = threadIdx.x << 2;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    float  local_out[4];
    int    outIdx       = (blockIdx.x * n + col_start) >> 2;
    char4* outTmpPtr    = (char4*)output;
    char4* input1TmpPtr = (char4*)input1;
    char4* input2TmpPtr = (char4*)input2;
    char4  input1Tmp    = __ldg(input1TmpPtr + outIdx);
    char4  input2Tmp    = __ldg(input2TmpPtr + outIdx);

    int col_start_tmp = col_start;
    local_out[0]      = static_cast<float>(input2Tmp.x) * input2_deQFactor
                   + static_cast<float>(input1Tmp.x) * input1_deQFactor
                   + static_cast<float>(__ldg(bias + col_start_tmp));
    col_start_tmp = col_start_tmp + 1;
    local_out[1]  = static_cast<float>(input2Tmp.y) * input2_deQFactor
                   + static_cast<float>(input1Tmp.y) * input1_deQFactor
                   + static_cast<float>(__ldg(bias + col_start_tmp));
    col_start_tmp = col_start_tmp + 1;
    local_out[2]  = static_cast<float>(input2Tmp.z) * input2_deQFactor
                   + static_cast<float>(input1Tmp.z) * input1_deQFactor
                   + static_cast<float>(__ldg(bias + col_start_tmp));
    col_start_tmp = col_start_tmp + 1;
    local_out[3]  = static_cast<float>(input2Tmp.w) * input2_deQFactor
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
    variance     = blockReduceSum<float>(local_out[0] * local_out[0] + local_out[1] * local_out[1]
                                     + local_out[2] * local_out[2] + local_out[3] * local_out[3]);
    if (threadIdx.x == 0) {
        s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
        s_variance = rsqrtf(s_variance);
    }
    __syncthreads();

    local_out[0] = (local_out[0] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                   + static_cast<float>(__ldg(beta + col_start));
    input2Tmp.x = float_to_int8_rn(local_out[0] * output_scale);

    col_start    = col_start + 1;
    local_out[1] = (local_out[1] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                   + static_cast<float>(__ldg(beta + col_start));
    input2Tmp.y = float_to_int8_rn(local_out[1] * output_scale);

    col_start    = col_start + 1;
    local_out[2] = (local_out[2] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                   + static_cast<float>(__ldg(beta + col_start));
    input2Tmp.z = float_to_int8_rn(local_out[2] * output_scale);

    col_start    = col_start + 1;
    local_out[3] = (local_out[3] * s_variance) * static_cast<float>(__ldg(gamma + col_start))
                   + static_cast<float>(__ldg(beta + col_start));
    input2Tmp.w = float_to_int8_rn(local_out[3] * output_scale);

    outTmpPtr[outIdx] = input2Tmp;
}

template<>
__global__ void add_bias_input_layernorm_ROW_int8IO(int8_t*       output,
                                                    const int8_t* input1,
                                                    const int8_t* input2,
                                                    const half2*  bias,
                                                    const half2*  gamma,
                                                    const half2*  beta,
                                                    int           m,
                                                    int           n,
                                                    const float*  input1_deQFactor_ptr,
                                                    const float*  input2_deQFactor_ptr,
                                                    const float*  output_scale_ptr)
{
    const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
    const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
    const float output_scale     = __ldg(output_scale_ptr);
    int         col_start        = threadIdx.x << 2;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    float  local_out[4];
    int    outIdx       = (blockIdx.x * n + col_start) >> 2;
    char4* outTmpPtr    = (char4*)output;
    char4* input1TmpPtr = (char4*)input1;
    char4* input2TmpPtr = (char4*)input2;
    char4  input1Tmp    = __ldg(input1TmpPtr + outIdx);
    char4  input2Tmp    = __ldg(input2TmpPtr + outIdx);

    int   col_start_tmp = col_start;
    half2 biasTmp       = __ldg(bias + (col_start_tmp >> 1));
    local_out[0]        = static_cast<float>(input2Tmp.x) * input2_deQFactor
                   + static_cast<float>(input1Tmp.x) * input1_deQFactor + static_cast<float>(biasTmp.x);
    col_start_tmp = col_start_tmp + 1;
    local_out[1]  = static_cast<float>(input2Tmp.y) * input2_deQFactor
                   + static_cast<float>(input1Tmp.y) * input1_deQFactor + static_cast<float>(biasTmp.y);

    col_start_tmp = col_start_tmp + 1;
    biasTmp       = __ldg(bias + (col_start_tmp >> 1));
    local_out[2]  = static_cast<float>(input2Tmp.z) * input2_deQFactor
                   + static_cast<float>(input1Tmp.z) * input1_deQFactor + static_cast<float>(biasTmp.x);
    col_start_tmp = col_start_tmp + 1;
    local_out[3]  = static_cast<float>(input2Tmp.w) * input2_deQFactor
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
    variance     = blockReduceSum<float>(local_out[0] * local_out[0] + local_out[1] * local_out[1]
                                     + local_out[2] * local_out[2] + local_out[3] * local_out[3]);
    if (threadIdx.x == 0) {
        s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
        s_variance = rsqrtf(s_variance);
    }
    __syncthreads();

    col_start_tmp = col_start >> 1;
    biasTmp       = __ldg(gamma + col_start_tmp);
    half2 betaTmp = __ldg(beta + col_start_tmp);

    local_out[0] = (local_out[0] * s_variance) * static_cast<float>(biasTmp.x) + static_cast<float>(betaTmp.x);
    input2Tmp.x  = float_to_int8_rn(local_out[0] * output_scale);

    col_start    = col_start + 1;
    local_out[1] = (local_out[1] * s_variance) * static_cast<float>(biasTmp.y) + static_cast<float>(betaTmp.y);
    input2Tmp.y  = float_to_int8_rn(local_out[1] * output_scale);

    col_start     = col_start + 1;
    col_start_tmp = col_start >> 1;
    biasTmp       = __ldg(gamma + col_start_tmp);
    betaTmp       = __ldg(beta + col_start_tmp);
    local_out[2]  = (local_out[2] * s_variance) * static_cast<float>(biasTmp.x) + static_cast<float>(betaTmp.x);
    input2Tmp.z   = float_to_int8_rn(local_out[2] * output_scale);

    col_start    = col_start + 1;
    local_out[3] = (local_out[3] * s_variance) * static_cast<float>(biasTmp.y) + static_cast<float>(betaTmp.y);
    input2Tmp.w  = float_to_int8_rn(local_out[3] * output_scale);

    outTmpPtr[outIdx] = input2Tmp;
}

template<typename T>
void invokeAddBiasResidualLayerNormRow(int8_t*       output,
                                       const int8_t* input1,
                                       const int8_t* input2,
                                       const T*      bias,
                                       const T*      gamma,
                                       const T*      beta,
                                       int           m,
                                       int           n,
                                       cudaStream_t  stream,
                                       const float*  input1_deQFactor_ptr,
                                       const float*  input2_deQFactor_ptr,
                                       const float*  output_scale_ptr)
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

template void invokeAddBiasResidualLayerNormRow(int8_t*       output,
                                                const int8_t* input1,
                                                const int8_t* input2,
                                                const float*  bias,
                                                const float*  gamma,
                                                const float*  beta,
                                                int           m,
                                                int           n,
                                                cudaStream_t  stream,
                                                const float*  input1_deQFactor_ptr,
                                                const float*  input2_deQFactor_ptr,
                                                const float*  output_scale_ptr);

template void invokeAddBiasResidualLayerNormRow(int8_t*       output,
                                                const int8_t* input1,
                                                const int8_t* input2,
                                                const half*   bias,
                                                const half*   gamma,
                                                const half*   beta,
                                                int           m,
                                                int           n,
                                                cudaStream_t  stream,
                                                const float*  input1_deQFactor_ptr,
                                                const float*  input2_deQFactor_ptr,
                                                const float*  output_scale_ptr);

// input1/input2/out matrix with layout of row major (m*n)
//(grid, block) must be (m, n)
template<typename T>
__global__ void add_bias_input_layernorm_ROW_int8I_DataTypeO(T*            output,
                                                             const int8_t* input1,
                                                             const int8_t* input2,
                                                             const T*      bias,
                                                             const T*      gamma,
                                                             const T*      beta,
                                                             int           m,
                                                             int           n,
                                                             const float*  input1_deQFactor_ptr,
                                                             const float*  input2_deQFactor_ptr)
{
    const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
    const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
    int         col_start        = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    float local_out;
    int   idx = blockIdx.x * n + col_start;

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
__global__ void add_bias_input_layernorm_ROW_int8I_DataTypeO(half2*        output,
                                                             const int8_t* input1,
                                                             const int8_t* input2,
                                                             const half2*  bias,
                                                             const half2*  gamma,
                                                             const half2*  beta,
                                                             int           m,
                                                             int           n,
                                                             const float*  input1_deQFactor_ptr,
                                                             const float*  input2_deQFactor_ptr)
{
    const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
    const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
    int         col_start        = threadIdx.x << 1;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    float2 local_out;
    int    idx = (blockIdx.x * n + col_start) >> 1;

    const char2* input1_ptr2 = (const char2*)input1;
    const char2* input2_ptr2 = (const char2*)input2;
    char2        input_tmp1  = __ldg(input1_ptr2 + idx);
    char2        input_tmp2  = __ldg(input2_ptr2 + idx);

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
    half2 beta_tmp  = __ldg(beta + threadIdx.x);

    local_out.x = (local_out.x * s_variance) * static_cast<float>(gamma_tmp.x) + static_cast<float>(beta_tmp.x);
    local_out.y = (local_out.y * s_variance) * static_cast<float>(gamma_tmp.y) + static_cast<float>(beta_tmp.y);

    bias_tmp.x = half(local_out.x);
    bias_tmp.y = half(local_out.y);

    output[idx] = bias_tmp;
}

template<typename T>
void invokeAddBiasResidualLayerNormRow(T*            output,
                                       const int8_t* input1,
                                       const int8_t* input2,
                                       const T*      bias,
                                       const T*      gamma,
                                       const T*      beta,
                                       int           m,
                                       int           n,
                                       cudaStream_t  stream,
                                       const float*  input1_deQFactor_ptr,
                                       const float*  input2_deQFactor_ptr)
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

template void invokeAddBiasResidualLayerNormRow<float>(float*        output,
                                                       const int8_t* input1,
                                                       const int8_t* input2,
                                                       const float*  bias,
                                                       const float*  gamma,
                                                       const float*  beta,
                                                       int           m,
                                                       int           n,
                                                       cudaStream_t  stream,
                                                       const float*  input1_deQFactor_ptr,
                                                       const float*  input2_deQFactor_ptr);

template void invokeAddBiasResidualLayerNormRow<half>(half*         output,
                                                      const int8_t* input1,
                                                      const int8_t* input2,
                                                      const half*   bias,
                                                      const half*   gamma,
                                                      const half*   beta,
                                                      int           m,
                                                      int           n,
                                                      cudaStream_t  stream,
                                                      const float*  input1_deQFactor_ptr,
                                                      const float*  input2_deQFactor_ptr);

}  // namespace fastertransformer