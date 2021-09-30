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

//input1/input2/output matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n)
//(grid, block) must be (m, n)
//for per_channel_quantization for weight
__global__
void add_bias_input_layernorm_COL32_int32I_DataTypeO(float* output, const int32_t* input1, const float* input2, const float* bias, const float* gamma, 
                                                     const float* beta, int m, int n, const float* weight_amax, const float *input1_amax_ptr)
{
  const float input1_amax = __ldg(input1_amax_ptr);
  int col_start = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out;
  int outIdx = ((col_start & 0xffffffe0)*m+(blockIdx.x << 5) + (col_start&31));

  float tmp = static_cast<float>(__ldg(input1 + outIdx)) * __ldg(weight_amax + col_start) * input1_amax * 0.000062f; //(1/127/127);
  float inputTmp = __ldg(input2 + outIdx);

  local_out = tmp + inputTmp + __ldg(bias + col_start);

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = __fdividef(mean, n);
  __syncthreads();

  local_out = local_out - s_mean;

  variance = blockReduceSum<float>(local_out * local_out);
  if(threadIdx.x == 0){
    s_variance = __fdividef(variance, n) + 1e-6f;
    s_variance = rsqrtf(s_variance);
  }
  __syncthreads();

  local_out = (local_out * s_variance) * __ldg(gamma + col_start) + __ldg(beta + col_start);

  output[outIdx] = local_out;
}

__global__
void add_bias_input_layernorm_COL32_int32I_DataTypeO(half2* output, const int2* input1, const half2* input2, const half2* bias, const half2* gamma, 
                                                     const half2* beta, int m, int n, const float2* weight_amax, const float *input1_amax_ptr)
{
  int col_start = threadIdx.x << 1;

  const float input1_amax = __ldg(input1_amax_ptr);

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float2 local_out;
  int outIdx = ((col_start & 0xffffffe0)*m+(blockIdx.x << 5) + (col_start&31)) >> 1;

  const int2 input1Tmp = __ldg(input1 + outIdx);
  const float2 weightTmp = __ldg(weight_amax + threadIdx.x);

  float2 addTmp2;
  addTmp2.x = static_cast<float>(input1Tmp.x) * weightTmp.x * input1_amax * 0.000062f; //(1/127/127);
  addTmp2.y = static_cast<float>(input1Tmp.y) * weightTmp.y * input1_amax * 0.000062f; //(1/127/127);
  
  const half2 inputTmp = __ldg(input2 + outIdx);
  const half2 biasTmp = __ldg(bias + threadIdx.x);

  local_out = __half22float2(__hadd2(inputTmp, biasTmp));
  local_out.x = local_out.x + addTmp2.x;
  local_out.y = local_out.y + addTmp2.y;

  mean = blockReduceSum<float>(local_out.x + local_out.y);
  if(threadIdx.x == 0)
    s_mean = __fdividef(mean, n);
  __syncthreads();

  local_out.x = local_out.x - s_mean;
  local_out.y = local_out.y - s_mean;

  variance = blockReduceSum<float>(local_out.x*local_out.x + local_out.y*local_out.y);
  if(threadIdx.x == 0){
    s_variance = __fdividef(variance, n) + 1e-6f;
    s_variance = rsqrtf(s_variance);
  }
  __syncthreads();

  float2 outputTmp;
  const half2 gammaTmp = __ldg(gamma + threadIdx.x);
  const half2 betaTmp = __ldg(beta + threadIdx.x);

  outputTmp.x = (local_out.x * s_variance) * static_cast<float>(gammaTmp.x) + static_cast<float>(betaTmp.x);
  outputTmp.y = (local_out.y * s_variance) * static_cast<float>(gammaTmp.y) + static_cast<float>(betaTmp.y);

  output[outIdx] =  __float22half2_rn(outputTmp);
}


template <typename T>
void invokeAddBiasResidualLayerNormCol32(T* output, const int32_t* input1, const T* input2, const T* bias, const T* gamma, 
                                         const T* beta, int m, int n, cudaStream_t stream, const float* weight_amax, 
                                         const float* input1_amax_ptr){

  dim3 grid(m);
  dim3 block(n);
  if (sizeof(T) == sizeof(half)){
    block.x /= 2;
    assert(block.x <= 1024);
    add_bias_input_layernorm_COL32_int32I_DataTypeO<<<grid, block, 0, stream>>>((half2 *)output, (const int2*)input1, (const half2 *)input2, (const half2 *)bias, (const half2 *)gamma, 
                                                                                (const half2 *)beta, m, n, (const float2*)weight_amax, input1_amax_ptr);
  }
  else{
    assert(block.x <= 1024);
    add_bias_input_layernorm_COL32_int32I_DataTypeO<<<grid, block, 0, stream>>>((float *)output, input1, (const float*)input2, (const float*)bias, (const float*)gamma, 
                                                                                (const float*)beta, m, n, weight_amax, input1_amax_ptr);
  }
}

template
void invokeAddBiasResidualLayerNormCol32(float* output, const int32_t* input1, const float* input2, const float* bias, const float* gamma, 
                                         const float* beta, int m, int n, cudaStream_t stream, const float* weight_amax, 
                                         const float* input1_amax_ptr);
template
void invokeAddBiasResidualLayerNormCol32(half* output, const int32_t* input1, const half* input2, const half* bias, const half* gamma, 
                                         const half* beta, int m, int n, cudaStream_t stream, const float* weight_amax, 
                                         const float* input1_amax_ptr);


//input1/input2/out matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n)
//(grid, block) must be (m, n/4)
//using char4
template <typename T>
__global__
void add_bias_input_layernorm_COL32_int8IO(int8_t* output, const int8_t* input1, const int8_t* input2, const T* bias, const T* gamma, 
                                           const T* beta, int m, int n, 
                                           const float *input1_deQFactor_ptr, 
                                           const float *input2_deQFactor_ptr, const float *output_scale_ptr)
{
  const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
  const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
  const float output_scale = __ldg(output_scale_ptr);
  int col_start = threadIdx.x << 2;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out[4];
  int outIdx = ((col_start & 0xffffffe0)*m+(blockIdx.x << 5) + (col_start&31)) >> 2;
  char4 *outTmpPtr = (char4*)output;
  char4 *input1TmpPtr = (char4*)input1;
  char4 *input2TmpPtr = (char4*)input2;
  char4 input1Tmp = __ldg(input1TmpPtr+outIdx);
  char4 input2Tmp = __ldg(input2TmpPtr+outIdx);
  
  
  int col_start_tmp = col_start;
  local_out[0] = static_cast<float>(input2Tmp.x)*input2_deQFactor + static_cast<float>(input1Tmp.x)*input1_deQFactor + static_cast<float>(__ldg(bias+col_start_tmp));
  col_start_tmp = col_start_tmp + 1;
  local_out[1] = static_cast<float>(input2Tmp.y)*input2_deQFactor + static_cast<float>(input1Tmp.y)*input1_deQFactor + static_cast<float>(__ldg(bias+col_start_tmp));
  col_start_tmp = col_start_tmp + 1;
  local_out[2] = static_cast<float>(input2Tmp.z)*input2_deQFactor + static_cast<float>(input1Tmp.z)*input1_deQFactor + static_cast<float>(__ldg(bias+col_start_tmp));
  col_start_tmp = col_start_tmp + 1;
  local_out[3] = static_cast<float>(input2Tmp.w)*input2_deQFactor + static_cast<float>(input1Tmp.w)*input1_deQFactor + static_cast<float>(__ldg(bias+col_start_tmp));


  mean = blockReduceSum<float>(local_out[0] + local_out[1] + local_out[2] + local_out[3]);
  if(threadIdx.x == 0)
    s_mean = mean * __fdividef(1.0f, n);
  __syncthreads();
  
  local_out[0] = local_out[0] - s_mean;
  local_out[1] = local_out[1] - s_mean;
  local_out[2] = local_out[2] - s_mean;
  local_out[3] = local_out[3] - s_mean;
  variance = blockReduceSum<float>(local_out[0] * local_out[0] +
                                   local_out[1] * local_out[1] +
                                   local_out[2] * local_out[2] +
                                   local_out[3] * local_out[3]
                                  );
  if(threadIdx.x == 0){
    s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
    s_variance = rsqrtf(s_variance);
  }
  __syncthreads();
  
  local_out[0] = (local_out[0] * s_variance) * static_cast<float>(__ldg(gamma+col_start)) + static_cast<float>(__ldg(beta+col_start));
  input2Tmp.x = float_to_int8_rn(local_out[0] * output_scale);

  col_start = col_start+1;
  local_out[1] = (local_out[1] * s_variance) * static_cast<float>(__ldg(gamma+col_start)) + static_cast<float>(__ldg(beta+col_start));
  input2Tmp.y = float_to_int8_rn(local_out[1] * output_scale);
  
  col_start = col_start+1;
  local_out[2] = (local_out[2] * s_variance) * static_cast<float>(__ldg(gamma+col_start)) + static_cast<float>(__ldg(beta+col_start));
  input2Tmp.z = float_to_int8_rn(local_out[2] * output_scale);
  
  col_start = col_start+1;
  local_out[3] = (local_out[3] * s_variance) * static_cast<float>(__ldg(gamma+col_start)) + static_cast<float>(__ldg(beta+col_start));
  input2Tmp.w = float_to_int8_rn(local_out[3] * output_scale);
  
  outTmpPtr[outIdx] = input2Tmp;
}

template <>
__global__
void add_bias_input_layernorm_COL32_int8IO(int8_t* output, const int8_t* input1, const int8_t* input2, const half2* bias, const half2* gamma, 
                                           const half2* beta, int m, int n, const float *input1_deQFactor_ptr, 
                                           const float *input2_deQFactor_ptr, const float *output_scale_ptr)
{
  const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
  const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
  const float output_scale = __ldg(output_scale_ptr);
  int col_start = threadIdx.x << 2;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out[4];
  int outIdx = ((col_start & 0xffffffe0)*m+(blockIdx.x << 5) + (col_start&31)) >> 2;
  char4 *outTmpPtr = (char4*)output;
  char4 *input1TmpPtr = (char4*)input1;
  char4 *input2TmpPtr = (char4*)input2;
  char4 input1Tmp = __ldg(input1TmpPtr + outIdx);
  char4 input2Tmp = __ldg(input2TmpPtr + outIdx);
  
  int col_start_tmp = col_start;
  half2 biasTmp = __ldg(bias + (col_start_tmp >> 1));  
  local_out[0] = static_cast<float>(input2Tmp.x)*input2_deQFactor + static_cast<float>(input1Tmp.x)*input1_deQFactor + static_cast<float>(biasTmp.x);
  col_start_tmp = col_start_tmp + 1;
  local_out[1] = static_cast<float>(input2Tmp.y)*input2_deQFactor + static_cast<float>(input1Tmp.y)*input1_deQFactor + static_cast<float>(biasTmp.y);
  
  col_start_tmp = col_start_tmp + 1;
  biasTmp = __ldg(bias + (col_start_tmp >> 1));
  local_out[2] = static_cast<float>(input2Tmp.z)*input2_deQFactor + static_cast<float>(input1Tmp.z)*input1_deQFactor + static_cast<float>(biasTmp.x);
  col_start_tmp = col_start_tmp + 1;
  local_out[3] = static_cast<float>(input2Tmp.w)*input2_deQFactor + static_cast<float>(input1Tmp.w)*input1_deQFactor + static_cast<float>(biasTmp.y);


  mean = blockReduceSum<float>(local_out[0] + local_out[1] + local_out[2] + local_out[3]);
  if(threadIdx.x == 0)
    s_mean = mean * __fdividef(1.0f, n);
  __syncthreads();
  
  local_out[0] = local_out[0] - s_mean;
  local_out[1] = local_out[1] - s_mean;
  local_out[2] = local_out[2] - s_mean;
  local_out[3] = local_out[3] - s_mean;
  variance = blockReduceSum<float>(local_out[0] * local_out[0] +
                                   local_out[1] * local_out[1] +
                                   local_out[2] * local_out[2] +
                                   local_out[3] * local_out[3]
                                  );
  if(threadIdx.x == 0){
    s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
    s_variance = rsqrtf(s_variance);
  }
  __syncthreads();
  
  col_start_tmp = col_start >> 1;
  biasTmp = __ldg(gamma+col_start_tmp);
  half2 betaTmp = __ldg(beta+col_start_tmp); 
  
  local_out[0] = (local_out[0] * s_variance) * static_cast<float>(biasTmp.x) + static_cast<float>(betaTmp.x);
  input2Tmp.x = float_to_int8_rn(local_out[0] * output_scale);

  col_start = col_start+1;
  local_out[1] = (local_out[1] * s_variance) * static_cast<float>(biasTmp.y) + static_cast<float>(betaTmp.y);
  input2Tmp.y = float_to_int8_rn(local_out[1] * output_scale);
  
  col_start = col_start+1;
  col_start_tmp = col_start >> 1;
  biasTmp = __ldg(gamma+col_start_tmp);
  betaTmp = __ldg(beta+col_start_tmp);
  local_out[2] = (local_out[2] * s_variance) * static_cast<float>(biasTmp.x) + static_cast<float>(betaTmp.x);
  input2Tmp.z = float_to_int8_rn(local_out[2] * output_scale);
  
  col_start = col_start+1;
  local_out[3] = (local_out[3] * s_variance) * static_cast<float>(biasTmp.y) + static_cast<float>(betaTmp.y);
  input2Tmp.w = float_to_int8_rn(local_out[3] * output_scale);
  
  outTmpPtr[outIdx] = input2Tmp;
}

template<typename T>
void invokeAddBiasResidualLayerNormCol32(int8_t* output, const int8_t* input1, const int8_t* input2, const T* bias, const T* gamma, 
                                         const T* beta, int m, int n, cudaStream_t stream, 
                                         const float *input1_deQFactor_ptr, const float *input2_deQFactor_ptr, const float *output_scale_ptr)
{
  dim3 grid(m);
  dim3 block(n/4);
  assert(n <= 1024);
  if (sizeof(T) == sizeof(half)){
    add_bias_input_layernorm_COL32_int8IO<<<grid, block, 0, stream>>>(output, input1, input2, (const half2*)bias, (const half2*)gamma, 
                                                                      (const half2*)beta, m, n, input1_deQFactor_ptr, 
                                                                      input2_deQFactor_ptr, output_scale_ptr);
  }
  else{
    add_bias_input_layernorm_COL32_int8IO<T><<<grid, block, 0, stream>>>(output, input1, input2, bias, gamma, beta, 
                                                                         m, n, input1_deQFactor_ptr, 
                                                                         input2_deQFactor_ptr, output_scale_ptr);
  }
}

template
void invokeAddBiasResidualLayerNormCol32(int8_t* output, const int8_t* input1, const int8_t* input2, const float* bias, const float* gamma, 
                                         const float* beta, int m, int n, cudaStream_t stream, 
                                         const float *input1_deQFactor_ptr, const float *input2_deQFactor_ptr, const float *output_scale_ptr);
                                         
template
void invokeAddBiasResidualLayerNormCol32(int8_t* output, const int8_t* input1, const int8_t* input2, const half* bias, const half* gamma, 
                                         const half* beta, int m, int n, cudaStream_t stream, 
                                         const float *input1_deQFactor_ptr, const float *input2_deQFactor_ptr, const float *output_scale_ptr);
                                         
//input1/input2/out matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n)
//(grid, block) must be (m, n)
template <typename T>
__global__
void add_bias_input_layernorm_COL32_int8I_DataTypeO(T* output, const int8_t* input1, const int8_t* input2, const T* bias, const T* gamma, 
                                                    const T* beta, int m, int n, 
                                                    const float *input1_deQFactor_ptr, 
                                                    const float *input2_deQFactor_ptr)
{
  const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
  const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
  int col_start = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out;
  int idx = ((col_start & 0xffffffe0)*m+(blockIdx.x << 5) + (col_start&31));
  
  local_out = static_cast<float>(__ldg(input2+idx))*input2_deQFactor + static_cast<float>(__ldg(input1+idx))*input1_deQFactor + static_cast<float>(__ldg(bias+col_start));

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean * __fdividef(1.0f, n);
  __syncthreads();
  
  local_out = local_out - s_mean;

  variance = blockReduceSum<float>(local_out * local_out);
  
  if(threadIdx.x == 0){
    s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
    s_variance = rsqrtf(s_variance);
  }
  __syncthreads();
  
  local_out = (local_out * s_variance) * static_cast<float>(__ldg(gamma+col_start)) + static_cast<float>(__ldg(beta+col_start));
  
  output[idx] = local_out;
}

//input1/input2/out matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n)
//(grid, block) must be (m, n/2)
template <>
__global__
void add_bias_input_layernorm_COL32_int8I_DataTypeO(half2* output, const int8_t* input1, const int8_t* input2, const half2* bias, const half2* gamma, 
                                                    const half2* beta, int m, int n, 
                                                    const float *input1_deQFactor_ptr, 
                                                    const float *input2_deQFactor_ptr)
{
  const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
  const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
  int col_start = threadIdx.x << 1;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float2 local_out;
  int idx = ((col_start & 0xffffffe0)*m+(blockIdx.x << 5) + (col_start&31)) >> 1;
  
  const char2 * input1_ptr2 = (const char2*)input1;
  const char2 * input2_ptr2 = (const char2*)input2;
  char2 input_tmp1 = __ldg(input1_ptr2 + idx);
  char2 input_tmp2 = __ldg(input2_ptr2 + idx);

  half2 bias_tmp = __ldg(bias+threadIdx.x);
  
  local_out.x = static_cast<float>(input_tmp1.x)*input1_deQFactor + static_cast<float>(input_tmp2.x)*input2_deQFactor + static_cast<float>(bias_tmp.x);
  
  local_out.y = static_cast<float>(input_tmp1.y)*input1_deQFactor + static_cast<float>(input_tmp2.y)*input2_deQFactor + static_cast<float>(bias_tmp.y);

  mean = blockReduceSum<float>(local_out.x + local_out.y);
  if(threadIdx.x == 0)
    s_mean = mean * __fdividef(1.0f, n);
  __syncthreads();
  
  local_out.x = local_out.x - s_mean;
  
  local_out.y = local_out.y - s_mean;

  variance = blockReduceSum<float>(local_out.x * local_out.x + local_out.y * local_out.y);
  
  if(threadIdx.x == 0){
    s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
    s_variance = rsqrtf(s_variance);
  }
  __syncthreads();
  
  half2 gamma_tmp = __ldg(gamma+threadIdx.x);
  half2 beta_tmp = __ldg(beta+threadIdx.x);
  
  local_out.x = (local_out.x * s_variance) * static_cast<float>(gamma_tmp.x) + static_cast<float>(beta_tmp.x);
  local_out.y = (local_out.y * s_variance) * static_cast<float>(gamma_tmp.y) + static_cast<float>(beta_tmp.y);
  
  bias_tmp.x = half(local_out.x);
  bias_tmp.y = half(local_out.y);
  
  output[idx] = bias_tmp;
}

template<typename T>
void invokeAddBiasResidualLayerNormCol32(T *output, const int8_t *input1,
                                         const int8_t *input2, const T *bias,
                                         const T *gamma, const T *beta,
                                         int m, int n, cudaStream_t stream, 
                                         const float *input1_deQFactor_ptr, 
                                         const float *input2_deQFactor_ptr)
{
  dim3 grid(m);
  dim3 block(n);
  if (sizeof(T) == sizeof(half)){
    assert(n/2 <= 1024 && n%2 == 0);
    block.x = n/2;
    add_bias_input_layernorm_COL32_int8I_DataTypeO<<<grid, block, 0, stream>>>((half2*)output, input1, input2, (const half2*)bias, (const half2*)gamma, 
                                                                               (const half2*)beta, m, n, input1_deQFactor_ptr, 
                                                                               input2_deQFactor_ptr);
  }
  else{
    assert(n <= 1024);
    add_bias_input_layernorm_COL32_int8I_DataTypeO<T><<<grid, block, 0, stream>>>(output, input1, input2, bias, gamma, beta, 
                                                                                  m, n, input1_deQFactor_ptr,
                                                                                  input2_deQFactor_ptr);
  }
}

template void invokeAddBiasResidualLayerNormCol32<float>(float* output, const int8_t* input1, const int8_t* input2, const float* bias, const float* gamma, const float* beta, int m, int n, cudaStream_t stream, const float *input1_deQFactor_ptr, const float *input2_deQFactor_ptr);

template void invokeAddBiasResidualLayerNormCol32<half>(half* output, const int8_t* input1, const int8_t* input2, const half* bias, const half* gamma, const half* beta, int m, int n, cudaStream_t stream, const float *input1_deQFactor_ptr, const float *input2_deQFactor_ptr);


//input1/input2/out matrix with layout of row major (m*n)
//(grid, block) must be (m, n/4)
//using char4
template <typename T>
__global__
void add_bias_input_layernorm_ROW_int8IO(int8_t* output, const int8_t* input1, const int8_t* input2, const T* bias,
                                         const T* gamma, const T* beta, int m, int n, 
                                         const float *input1_deQFactor_ptr, 
                                         const float *input2_deQFactor_ptr, const float *output_scale_ptr)
{
  const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
  const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
  const float output_scale = __ldg(output_scale_ptr);
  int col_start = threadIdx.x << 2;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out[4];
  int outIdx = (blockIdx.x * n + col_start) >> 2;
  char4 *outTmpPtr = (char4*)output;
  char4 *input1TmpPtr = (char4*)input1;
  char4 *input2TmpPtr = (char4*)input2;
  char4 input1Tmp = __ldg(input1TmpPtr+outIdx);
  char4 input2Tmp = __ldg(input2TmpPtr+outIdx);
  
  
  int col_start_tmp = col_start;
  local_out[0] = static_cast<float>(input2Tmp.x)*input2_deQFactor + static_cast<float>(input1Tmp.x)*input1_deQFactor + static_cast<float>(__ldg(bias+col_start_tmp));
  col_start_tmp = col_start_tmp + 1;
  local_out[1] = static_cast<float>(input2Tmp.y)*input2_deQFactor + static_cast<float>(input1Tmp.y)*input1_deQFactor + static_cast<float>(__ldg(bias+col_start_tmp));
  col_start_tmp = col_start_tmp + 1;
  local_out[2] = static_cast<float>(input2Tmp.z)*input2_deQFactor + static_cast<float>(input1Tmp.z)*input1_deQFactor + static_cast<float>(__ldg(bias+col_start_tmp));
  col_start_tmp = col_start_tmp + 1;
  local_out[3] = static_cast<float>(input2Tmp.w)*input2_deQFactor + static_cast<float>(input1Tmp.w)*input1_deQFactor + static_cast<float>(__ldg(bias+col_start_tmp));


  mean = blockReduceSum<float>(local_out[0] + local_out[1] + local_out[2] + local_out[3]);
  if(threadIdx.x == 0)
    s_mean = mean * __fdividef(1.0f, n);
  __syncthreads();
  
  local_out[0] = local_out[0] - s_mean;
  local_out[1] = local_out[1] - s_mean;
  local_out[2] = local_out[2] - s_mean;
  local_out[3] = local_out[3] - s_mean;
  variance = blockReduceSum<float>(local_out[0] * local_out[0] +
                                   local_out[1] * local_out[1] +
                                   local_out[2] * local_out[2] +
                                   local_out[3] * local_out[3]
                                  );
  if(threadIdx.x == 0){
    s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
    s_variance = rsqrtf(s_variance);
  }
  __syncthreads();
  
  local_out[0] = (local_out[0] * s_variance) * static_cast<float>(__ldg(gamma+col_start)) + static_cast<float>(__ldg(beta+col_start));
  input2Tmp.x = float_to_int8_rn(local_out[0] * output_scale);

  col_start = col_start+1;
  local_out[1] = (local_out[1] * s_variance) * static_cast<float>(__ldg(gamma+col_start)) + static_cast<float>(__ldg(beta+col_start));
  input2Tmp.y = float_to_int8_rn(local_out[1] * output_scale);
  
  col_start = col_start+1;
  local_out[2] = (local_out[2] * s_variance) * static_cast<float>(__ldg(gamma+col_start)) + static_cast<float>(__ldg(beta+col_start));
  input2Tmp.z = float_to_int8_rn(local_out[2] * output_scale);
  
  col_start = col_start+1;
  local_out[3] = (local_out[3] * s_variance) * static_cast<float>(__ldg(gamma+col_start)) + static_cast<float>(__ldg(beta+col_start));
  input2Tmp.w = float_to_int8_rn(local_out[3] * output_scale);
  
  outTmpPtr[outIdx] = input2Tmp;
}

template <>
__global__
void add_bias_input_layernorm_ROW_int8IO(int8_t* output, const int8_t* input1, const int8_t* input2, const half2* bias,
                                         const half2* gamma, const half2* beta, int m, int n, const float *input1_deQFactor_ptr, 
                                         const float *input2_deQFactor_ptr, const float *output_scale_ptr)
{
  const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
  const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
  const float output_scale = __ldg(output_scale_ptr);
  int col_start = threadIdx.x << 2;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out[4];
  int outIdx = (blockIdx.x * n + col_start) >> 2;
  char4 *outTmpPtr = (char4*)output;
  char4 *input1TmpPtr = (char4*)input1;
  char4 *input2TmpPtr = (char4*)input2;
  char4 input1Tmp = __ldg(input1TmpPtr + outIdx);
  char4 input2Tmp = __ldg(input2TmpPtr + outIdx);
  
  int col_start_tmp = col_start;
  half2 biasTmp = __ldg(bias + (col_start_tmp >> 1));  
  local_out[0] = static_cast<float>(input2Tmp.x)*input2_deQFactor + static_cast<float>(input1Tmp.x)*input1_deQFactor + static_cast<float>(biasTmp.x);
  col_start_tmp = col_start_tmp + 1;
  local_out[1] = static_cast<float>(input2Tmp.y)*input2_deQFactor + static_cast<float>(input1Tmp.y)*input1_deQFactor + static_cast<float>(biasTmp.y);
  
  col_start_tmp = col_start_tmp + 1;
  biasTmp = __ldg(bias + (col_start_tmp >> 1));
  local_out[2] = static_cast<float>(input2Tmp.z)*input2_deQFactor + static_cast<float>(input1Tmp.z)*input1_deQFactor + static_cast<float>(biasTmp.x);
  col_start_tmp = col_start_tmp + 1;
  local_out[3] = static_cast<float>(input2Tmp.w)*input2_deQFactor + static_cast<float>(input1Tmp.w)*input1_deQFactor + static_cast<float>(biasTmp.y);


  mean = blockReduceSum<float>(local_out[0] + local_out[1] + local_out[2] + local_out[3]);
  if(threadIdx.x == 0)
    s_mean = mean * __fdividef(1.0f, n);
  __syncthreads();
  
  local_out[0] = local_out[0] - s_mean;
  local_out[1] = local_out[1] - s_mean;
  local_out[2] = local_out[2] - s_mean;
  local_out[3] = local_out[3] - s_mean;
  variance = blockReduceSum<float>(local_out[0] * local_out[0] +
                                   local_out[1] * local_out[1] +
                                   local_out[2] * local_out[2] +
                                   local_out[3] * local_out[3]
                                  );
  if(threadIdx.x == 0){
    s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
    s_variance = rsqrtf(s_variance);
  }
  __syncthreads();
  
  col_start_tmp = col_start >> 1;
  biasTmp = __ldg(gamma+col_start_tmp);
  half2 betaTmp = __ldg(beta+col_start_tmp); 
  
  local_out[0] = (local_out[0] * s_variance) * static_cast<float>(biasTmp.x) + static_cast<float>(betaTmp.x);
  input2Tmp.x = float_to_int8_rn(local_out[0] * output_scale);

  col_start = col_start+1;
  local_out[1] = (local_out[1] * s_variance) * static_cast<float>(biasTmp.y) + static_cast<float>(betaTmp.y);
  input2Tmp.y = float_to_int8_rn(local_out[1] * output_scale);
  
  col_start = col_start+1;
  col_start_tmp = col_start >> 1;
  biasTmp = __ldg(gamma+col_start_tmp);
  betaTmp = __ldg(beta+col_start_tmp);
  local_out[2] = (local_out[2] * s_variance) * static_cast<float>(biasTmp.x) + static_cast<float>(betaTmp.x);
  input2Tmp.z = float_to_int8_rn(local_out[2] * output_scale);
  
  col_start = col_start+1;
  local_out[3] = (local_out[3] * s_variance) * static_cast<float>(biasTmp.y) + static_cast<float>(betaTmp.y);
  input2Tmp.w = float_to_int8_rn(local_out[3] * output_scale);
  
  outTmpPtr[outIdx] = input2Tmp;
}

template<typename T>
void invokeAddBiasResidualLayerNormRow(int8_t* output, const int8_t* input1, const int8_t* input2, const T* bias, const T* gamma, 
                                       const T* beta, int m, int n, cudaStream_t stream, 
                                       const float *input1_deQFactor_ptr, const float *input2_deQFactor_ptr, const float *output_scale_ptr)
{
  dim3 grid(m);
  dim3 block(n/4);
  assert(n <= 1024);
  if (sizeof(T) == sizeof(half)){
    add_bias_input_layernorm_ROW_int8IO<<<grid, block, 0, stream>>>(output, input1, input2, (const half2*)bias, (const half2*)gamma, 
                                                                    (const half2*)beta, m, n, input1_deQFactor_ptr, 
                                                                    input2_deQFactor_ptr, output_scale_ptr);
  }
  else{
    add_bias_input_layernorm_ROW_int8IO<T><<<grid, block, 0, stream>>>(output, input1, input2, bias, gamma, beta, 
                                                                       m, n, input1_deQFactor_ptr, 
                                                                       input2_deQFactor_ptr, output_scale_ptr);
  }
}

template
void invokeAddBiasResidualLayerNormRow(int8_t* output, const int8_t* input1, const int8_t* input2, const float* bias, const float* gamma, 
                                       const float* beta, int m, int n, cudaStream_t stream, 
                                       const float *input1_deQFactor_ptr, const float *input2_deQFactor_ptr, const float *output_scale_ptr);
                                         
template
void invokeAddBiasResidualLayerNormRow(int8_t* output, const int8_t* input1, const int8_t* input2, const half* bias, const half* gamma, 
                                       const half* beta, int m, int n, cudaStream_t stream, 
                                       const float *input1_deQFactor_ptr, const float *input2_deQFactor_ptr, const float *output_scale_ptr);
                                         

//input1/input2/out matrix with layout of row major (m*n)
//(grid, block) must be (m, n)
template <typename T>
__global__
void add_bias_input_layernorm_ROW_int8I_DataTypeO(T* output, const int8_t* input1, const int8_t* input2, const T* bias, const T* gamma, 
                                                  const T* beta, int m, int n, 
                                                  const float *input1_deQFactor_ptr, 
                                                  const float *input2_deQFactor_ptr)
{
  const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
  const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
  int col_start = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out;
  int idx = blockIdx.x * n + col_start;
  
  local_out = static_cast<float>(__ldg(input2+idx))*input2_deQFactor + static_cast<float>(__ldg(input1+idx))*input1_deQFactor + static_cast<float>(__ldg(bias+col_start));

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean * __fdividef(1.0f, n);
  __syncthreads();
  
  local_out = local_out - s_mean;

  variance = blockReduceSum<float>(local_out * local_out);
  
  if(threadIdx.x == 0){
    s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
    s_variance = rsqrtf(s_variance);
  }
  __syncthreads();
  
  local_out = (local_out * s_variance) * static_cast<float>(__ldg(gamma+col_start)) + static_cast<float>(__ldg(beta+col_start));
  
  output[idx] = local_out;
}

//input1/input2/out matrix with layout of row major (m*n)
//(grid, block) must be (m, n/2)
template <>
__global__
void add_bias_input_layernorm_ROW_int8I_DataTypeO(half2* output, const int8_t* input1, const int8_t* input2, const half2* bias, const half2* gamma, 
                                                  const half2* beta, int m, int n, 
                                                  const float *input1_deQFactor_ptr, 
                                                  const float *input2_deQFactor_ptr)
{
  const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
  const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
  int col_start = threadIdx.x << 1;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float2 local_out;
  int idx = (blockIdx.x * n + col_start) >> 1;
  
  const char2 * input1_ptr2 = (const char2*)input1;
  const char2 * input2_ptr2 = (const char2*)input2;
  char2 input_tmp1 = __ldg(input1_ptr2 + idx);
  char2 input_tmp2 = __ldg(input2_ptr2 + idx);

  half2 bias_tmp = __ldg(bias+threadIdx.x);
  
  local_out.x = static_cast<float>(input_tmp1.x)*input1_deQFactor + static_cast<float>(input_tmp2.x)*input2_deQFactor + static_cast<float>(bias_tmp.x);
  
  local_out.y = static_cast<float>(input_tmp1.y)*input1_deQFactor + static_cast<float>(input_tmp2.y)*input2_deQFactor + static_cast<float>(bias_tmp.y);

  mean = blockReduceSum<float>(local_out.x + local_out.y);
  if(threadIdx.x == 0)
    s_mean = mean * __fdividef(1.0f, n);
  __syncthreads();
  
  local_out.x = local_out.x - s_mean;
  
  local_out.y = local_out.y - s_mean;

  variance = blockReduceSum<float>(local_out.x * local_out.x + local_out.y * local_out.y);
  
  if(threadIdx.x == 0){
    s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
    s_variance = rsqrtf(s_variance);
  }
  __syncthreads();
  
  half2 gamma_tmp = __ldg(gamma+threadIdx.x);
  half2 beta_tmp = __ldg(beta+threadIdx.x);
  
  local_out.x = (local_out.x * s_variance) * static_cast<float>(gamma_tmp.x) + static_cast<float>(beta_tmp.x);
  local_out.y = (local_out.y * s_variance) * static_cast<float>(gamma_tmp.y) + static_cast<float>(beta_tmp.y);
  
  bias_tmp.x = half(local_out.x);
  bias_tmp.y = half(local_out.y);
  
  output[idx] = bias_tmp;
}

template<typename T>
void invokeAddBiasResidualLayerNormRow(T *output, const int8_t *input1,
                                       const int8_t *input2, const T *bias,
                                       const T *gamma, const T *beta,
                                       int m, int n, cudaStream_t stream, 
                                       const float *input1_deQFactor_ptr, 
                                       const float *input2_deQFactor_ptr)
{
  dim3 grid(m);
  dim3 block(n);
  if (sizeof(T) == sizeof(half)){
    assert(n/2 <= 1024 && n%2 == 0);
    block.x = n/2;
    add_bias_input_layernorm_ROW_int8I_DataTypeO<<<grid, block, 0, stream>>>((half2*)output, input1, input2, (const half2*)bias, (const half2*)gamma, 
                                                                             (const half2*)beta, m, n, input1_deQFactor_ptr, 
                                                                             input2_deQFactor_ptr);
  }
  else{
    assert(n <= 1024);
    add_bias_input_layernorm_ROW_int8I_DataTypeO<T><<<grid, block, 0, stream>>>(output, input1, input2, bias, gamma, beta, 
                                                                                m, n, input1_deQFactor_ptr,
                                                                                input2_deQFactor_ptr);
  }
}

template void invokeAddBiasResidualLayerNormRow<float>(float* output, const int8_t* input1, const int8_t* input2, const float* bias, const float* gamma, const float* beta, int m, int n, cudaStream_t stream, const float *input1_deQFactor_ptr, const float *input2_deQFactor_ptr);

template void invokeAddBiasResidualLayerNormRow<half>(half* output, const int8_t* input1, const int8_t* input2, const half* bias, const half* gamma, const half* beta, int m, int n, cudaStream_t stream, const float *input1_deQFactor_ptr, const float *input2_deQFactor_ptr);

}  // namespace fastertransformer