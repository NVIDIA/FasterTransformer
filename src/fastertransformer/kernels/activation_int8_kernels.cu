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

#include "src/fastertransformer/kernels/activation_int8_kernels.h"
namespace fastertransformer {

__inline__ __device__ half fast_tanh(half x)
{
#if defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 11) && (__CUDA_ARCH__ >= 750)
    uint16_t raw = reinterpret_cast<uint16_t const&>(x);
    asm volatile("tanh.approx.f16 %0, %1;" : "=h"(raw) : "h"(raw));
    half ret = reinterpret_cast<half const&>(raw);
    return ret;
#else
    return half(tanhf(float(x)));
#endif
}

template<typename T>
__inline__ __device__ T gelu(T x)
{
    float cdf = 0.5f * (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
    return x * cdf;
}

template<>
__inline__ __device__ half gelu(half x)
{
    half val = half(0.7978845608028654) * (x + half(0.044715) * x * x * x);
    half fast_val = fast_tanh(val);
    half cdf = half(0.5) * (half(1.0) + fast_val);
    return x * cdf;
}

template<>
__inline__ __device__ half2 gelu(half2 val)
{
    half2 val_pow3 = __hmul2(val, __hmul2(val, val));
    float2 tmp_pow = __half22float2(val_pow3);
    float2 tmp = __half22float2(val);

    tmp.x = 0.5f * (1.0f + tanhf((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
    tmp.y = 0.5f * (1.0f + tanhf((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
    return __hmul2(val, __float22half2_rn(tmp));
}

// add bias to matrix of m * n, CUBLASLT_ORDER_COL32
// grid, thread = (m), (n/4)
// using char4 as output
// for per-channel-quantization weight
__global__ void add_bias_gelu_COL32_int32I_int8O(int8_t* out,
                                                 const int32_t* input,
                                                 const float* bias,
                                                 const int m,
                                                 const int n,
                                                 const float* weight_amax,
                                                 const float* input_deQFactor_div127_ptr,
                                                 const float* out_scale_ptr)
{

    const float input_deQFactor_div127 = __ldg(input_deQFactor_div127_ptr);
    const float out_scale = __ldg(out_scale_ptr);

    int col_start = threadIdx.x << 2;
    char4* outTmpPtr = (char4*)out;
    char4 tmp;
    int outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 2;
    float val;

    const int4 input4 = __ldg(((const int4*)input) + outIdx);
    const float4 weight4 = __ldg(((const float4*)weight_amax) + threadIdx.x);
    const float4 bias4 = __ldg(((const float4*)bias) + threadIdx.x);

    val = static_cast<float>(input4.x) * weight4.x * input_deQFactor_div127 + bias4.x;
    val = gelu(val);
    tmp.x = float_to_int8_rn(val * out_scale);

    val = static_cast<float>(input4.y) * weight4.y * input_deQFactor_div127 + bias4.y;
    val = gelu(val);
    tmp.y = float_to_int8_rn(val * out_scale);

    col_start = col_start + 1;
    val = static_cast<float>(input4.z) * weight4.z * input_deQFactor_div127 + bias4.z;
    val = gelu(val);
    tmp.z = float_to_int8_rn(val * out_scale);

    col_start = col_start + 1;
    val = static_cast<float>(input4.w) * weight4.w * input_deQFactor_div127 + bias4.w;
    val = gelu(val);
    tmp.w = float_to_int8_rn(val * out_scale);

    outTmpPtr[outIdx] = tmp;
}

__global__ void add_bias_gelu_COL32_int32I_int8O(char4* out,
                                                 const int4* input,
                                                 const half2* bias,
                                                 const int m,
                                                 const int n,
                                                 const float4* weight_amax,
                                                 const float* input_deQFactor_div127_ptr,
                                                 const float* out_scale_ptr)
{
    const float input_deQFactor_div127 = __ldg(input_deQFactor_div127_ptr);
    const float out_scale = __ldg(out_scale_ptr);
    int col_start = threadIdx.x << 2;
    int threadIdx2 = threadIdx.x << 1;
    char4 tmp;
    int outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 2;
    float val;

    const int4 input4 = __ldg(input + outIdx);
    const float4 weight4 = __ldg(weight_amax + threadIdx.x);
    const half2 biasTmp = __ldg(bias + threadIdx2);
    const half2 biasTmp2 = __ldg(bias + threadIdx2 + 1);

    val = static_cast<float>(input4.x) * weight4.x * input_deQFactor_div127 + static_cast<float>(biasTmp.x);
    val = gelu(val);
    tmp.x = float_to_int8_rn(out_scale * val);

    val = static_cast<float>(input4.y) * weight4.y * input_deQFactor_div127 + static_cast<float>(biasTmp.y);
    val = gelu(val);
    tmp.y = float_to_int8_rn(out_scale * val);

    val = static_cast<float>(input4.z) * weight4.z * input_deQFactor_div127 + static_cast<float>(biasTmp2.x);
    val = gelu(val);
    tmp.z = float_to_int8_rn(out_scale * val);

    val = static_cast<float>(input4.w) * weight4.w * input_deQFactor_div127 + static_cast<float>(biasTmp2.y);
    val = gelu(val);
    tmp.w = float_to_int8_rn(out_scale * val);

    out[outIdx] = tmp;
}

template<typename T>
void invokeAddBiasGeluCol32(int8_t* out,
                            const int32_t* in,
                            const T* bias,
                            const int m,
                            const int n,
                            cudaStream_t stream,
                            const float* weight_amax,
                            const float* input_deQFactor_div127_ptr,
                            const float* out_scale_ptr)
{
    dim3 grid(m);
    dim3 block(n / 4);
    assert(block.x <= 1024);
    if (sizeof(T) == sizeof(half)) {
        add_bias_gelu_COL32_int32I_int8O<<<grid, block, 0, stream>>>((char4*)out,
                                                                     (const int4*)in,
                                                                     (const half2*)bias,
                                                                     m,
                                                                     n,
                                                                     (const float4*)weight_amax,
                                                                     input_deQFactor_div127_ptr,
                                                                     out_scale_ptr);
    }
    else {
        add_bias_gelu_COL32_int32I_int8O<<<grid, block, 0, stream>>>(
            out, in, (const float*)bias, m, n, weight_amax, input_deQFactor_div127_ptr, out_scale_ptr);
    }
}

template void invokeAddBiasGeluCol32(int8_t* out,
                                     const int32_t* in,
                                     const float* bias,
                                     const int m,
                                     const int n,
                                     cudaStream_t stream,
                                     const float* weight_amax,
                                     const float* input_deQFactor_div127_ptr,
                                     const float* out_scale_ptr);
template void invokeAddBiasGeluCol32(int8_t* out,
                                     const int32_t* in,
                                     const half* bias,
                                     const int m,
                                     const int n,
                                     cudaStream_t stream,
                                     const float* weight_amax,
                                     const float* input_deQFactor_div127_ptr,
                                     const float* out_scale_ptr);

// add bias to matrix of m * n, CUBLASLT_ORDER_COL32
// grid, thread = (m), (n/4)
// using char4
// for per-tensor-quantization weight
template<typename T>
__global__ void add_bias_gelu_COL32_int8IO(int8_t* out,
                                           const int8_t* input,
                                           const T* bias,
                                           const int m,
                                           const int n,
                                           const float* input_deQFactor_ptr,
                                           const float* out_scale_ptr)
{

    const float input_deQFactor = __ldg(input_deQFactor_ptr);
    const float out_scale = __ldg(out_scale_ptr);

    // int col_start = threadIdx.x << 2;
    char4* outTmpPtr = (char4*)out;
    char4* inputTmpPtr = (char4*)input;
    char4 tmp;
    for (int col_start = threadIdx.x << 2; col_start < n; col_start += (blockDim.x << 2)) {
        int outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 2;
        float val;
        tmp = __ldg(inputTmpPtr + outIdx);
        val = static_cast<float>(tmp.x) * input_deQFactor + static_cast<float>(__ldg(bias + col_start));
        val = gelu(val);
        tmp.x = float_to_int8_rn(val * out_scale);

        col_start = col_start + 1;
        val = static_cast<float>(tmp.y) * input_deQFactor + static_cast<float>(__ldg(bias + col_start));
        val = gelu(val);
        tmp.y = float_to_int8_rn(val * out_scale);

        col_start = col_start + 1;
        val = static_cast<float>(tmp.z) * input_deQFactor + static_cast<float>(__ldg(bias + col_start));
        val = gelu(val);
        tmp.z = float_to_int8_rn(val * out_scale);

        col_start = col_start + 1;
        val = static_cast<float>(tmp.w) * input_deQFactor + static_cast<float>(__ldg(bias + col_start));
        val = gelu(val);
        tmp.w = float_to_int8_rn(val * out_scale);

        outTmpPtr[outIdx] = tmp;
    }
}

template<typename T>
void invokeAddBiasGeluCol32(int8_t* out,
                            const int8_t* in,
                            const T* bias,
                            const int m,
                            const int n,
                            cudaStream_t stream,
                            const float* input_deQFactor_ptr,
                            const float* out_scale_ptr)
{
    dim3 grid;
    dim3 block;
    if (n / 4 <= 1024) {
        block.x = n / 4;
        grid.x = m;
    }
    else {
        block.x = 1024;
        grid.x = m;
    }

    add_bias_gelu_COL32_int8IO<<<grid, block, 0, stream>>>(out, in, bias, m, n, input_deQFactor_ptr, out_scale_ptr);
}

template void invokeAddBiasGeluCol32(int8_t* out,
                                     const int8_t* in,
                                     const float* bias,
                                     const int m,
                                     const int n,
                                     cudaStream_t stream,
                                     const float* input_deQFactor_ptr,
                                     const float* out_scale_ptr);

template void invokeAddBiasGeluCol32(int8_t* out,
                                     const int8_t* in,
                                     const half* bias,
                                     const int m,
                                     const int n,
                                     cudaStream_t stream,
                                     const float* input_deQFactor_ptr,
                                     const float* out_scale_ptr);

/*******************  invokeAddBiasGeluCol32_v2  ***********************/

// add bias to matrix of m * n, CUBLASLT_ORDER_COL32
// grid, thread = (m), (n/4)
// using char4
// for per-tensor-quantization weight
template<typename T>
__global__ void add_bias_gelu_COL32_int8IO(
    int8_t* out, const T* bias, const int m, const int n, const float* input_deQFactor_ptr, const float* out_scale_ptr)
{

    const float input_deQFactor = __ldg(input_deQFactor_ptr);
    const float out_scale = __ldg(out_scale_ptr);

    for (int col_start = threadIdx.x << 2; col_start < n; col_start += (blockDim.x << 2)) {
        char4* outTmpPtr = (char4*)out;
        char4 tmp;
        int outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 2;
        float val;
        tmp = __ldg(outTmpPtr + outIdx);
        val = static_cast<float>(tmp.x) * input_deQFactor + static_cast<float>(__ldg(bias + col_start));
        val = gelu(val);
        tmp.x = float_to_int8_rn(val * out_scale);

        val = static_cast<float>(tmp.y) * input_deQFactor + static_cast<float>(__ldg(bias + col_start + 1));
        val = gelu(val);
        tmp.y = float_to_int8_rn(val * out_scale);

        val = static_cast<float>(tmp.z) * input_deQFactor + static_cast<float>(__ldg(bias + col_start + 2));
        val = gelu(val);
        tmp.z = float_to_int8_rn(val * out_scale);

        val = static_cast<float>(tmp.w) * input_deQFactor + static_cast<float>(__ldg(bias + col_start + 3));
        val = gelu(val);
        tmp.w = float_to_int8_rn(val * out_scale);

        outTmpPtr[outIdx] = tmp;
    }
}

// add bias to matrix of m * n, CUBLASLT_ORDER_COL32
// grid, thread = (m), (n/4)
// using char4
// for per-tensor-quantization weight
template<>
__global__ void add_bias_gelu_COL32_int8IO(int8_t* out,
                                           const half* bias,
                                           const int m,
                                           const int n,
                                           const float* input_deQFactor_ptr,
                                           const float* out_scale_ptr)
{

    const float input_deQFactor = __ldg(input_deQFactor_ptr);
    const float out_scale = __ldg(out_scale_ptr);

    for (int col_start = threadIdx.x << 2; col_start < n; col_start += (blockDim.x << 2)) {
        char4* outTmpPtr = (char4*)out;
        char4* inputTmpPtr = (char4*)out;
        char4 tmp;
        int outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 2;
        half val;
        half input_deQFactor_half = half(input_deQFactor);
        half out_scale_half = half(out_scale);
        tmp = __ldg(inputTmpPtr + outIdx);

        val = static_cast<half>(tmp.x) * input_deQFactor_half + (__ldg(bias + col_start));
        val = gelu(val);
        tmp.x = float_to_int8_rn(float(val * out_scale_half));

        col_start = col_start + 1;
        val = static_cast<half>(tmp.y) * input_deQFactor_half + (__ldg(bias + col_start));
        val = gelu(val);
        tmp.y = float_to_int8_rn(float(val * out_scale_half));

        col_start = col_start + 1;
        val = static_cast<half>(tmp.z) * input_deQFactor_half + (__ldg(bias + col_start));
        val = gelu(val);
        tmp.z = float_to_int8_rn(float(val * out_scale_half));

        col_start = col_start + 1;
        val = static_cast<half>(tmp.w) * input_deQFactor_half + (__ldg(bias + col_start));
        val = gelu(val);
        tmp.w = float_to_int8_rn(float(val * out_scale_half));

        outTmpPtr[outIdx] = tmp;
    }
}

template<typename T>
void invokeAddBiasGeluCol32_v2(int8_t* out,
                               const T* bias,
                               const int m,
                               const int n,
                               cudaStream_t stream,
                               const float* input_deQFactor_ptr,
                               const float* out_scale_ptr)
{
    dim3 grid(m);
    dim3 block(n / 4);
    if (block.x > 1024) {
        block.x = 1024;
    }

    if (sizeof(T) == sizeof(half)) {
        add_bias_gelu_COL32_int8IO<<<grid, block, 0, stream>>>(out, bias, m, n, input_deQFactor_ptr, out_scale_ptr);
    }
    else {
        add_bias_gelu_COL32_int8IO<<<grid, block, 0, stream>>>(out, bias, m, n, input_deQFactor_ptr, out_scale_ptr);
    }
}

template void invokeAddBiasGeluCol32_v2(int8_t* out,
                                        const float* bias,
                                        const int m,
                                        const int n,
                                        cudaStream_t stream,
                                        const float* input_deQFactor_ptr,
                                        const float* out_scale_ptr);

template void invokeAddBiasGeluCol32_v2(int8_t* out,
                                        const half* bias,
                                        const int m,
                                        const int n,
                                        cudaStream_t stream,
                                        const float* input_deQFactor_ptr,
                                        const float* out_scale_ptr);

// add bias to matrix of m * n, row major
// grid, thread = (m), (n/4)
// using char4
// for per-tensor-quantization weight
template<typename T>
__global__ void add_bias_gelu_ROW_int8IO(int8_t* out,
                                         const int8_t* input,
                                         const T* bias,
                                         const int m,
                                         const int n,
                                         const float* input_deQFactor_ptr,
                                         const float* out_scale_ptr)
{

    const float input_deQFactor = __ldg(input_deQFactor_ptr);
    const float out_scale = __ldg(out_scale_ptr);

    int col_start = threadIdx.x << 2;
    char4* outTmpPtr = (char4*)out;
    char4* inputTmpPtr = (char4*)input;
    char4 tmp;
    int outIdx = (blockIdx.x * n + col_start) >> 2;
    float val;
    tmp = __ldg(inputTmpPtr + outIdx);
    val = static_cast<float>(tmp.x) * input_deQFactor + static_cast<float>(__ldg(bias + col_start));
    val = gelu(val);
    tmp.x = float_to_int8_rn(val * out_scale);

    col_start = col_start + 1;
    val = static_cast<float>(tmp.y) * input_deQFactor + static_cast<float>(__ldg(bias + col_start));
    val = gelu(val);
    tmp.y = float_to_int8_rn(val * out_scale);

    col_start = col_start + 1;
    val = static_cast<float>(tmp.z) * input_deQFactor + static_cast<float>(__ldg(bias + col_start));
    val = gelu(val);
    tmp.z = float_to_int8_rn(val * out_scale);

    col_start = col_start + 1;
    val = static_cast<float>(tmp.w) * input_deQFactor + static_cast<float>(__ldg(bias + col_start));
    val = gelu(val);
    tmp.w = float_to_int8_rn(val * out_scale);

    outTmpPtr[outIdx] = tmp;
}

template<typename T>
void invokeAddBiasGeluRow(int8_t* out,
                          const int8_t* in,
                          const T* bias,
                          const int m,
                          const int n,
                          cudaStream_t stream,
                          const float* input_deQFactor_ptr,
                          const float* out_scale_ptr)
{
    dim3 grid(m);
    dim3 block(n / 4);
    assert(block.x <= 1024);

    add_bias_gelu_ROW_int8IO<<<grid, block, 0, stream>>>(out, in, bias, m, n, input_deQFactor_ptr, out_scale_ptr);
}

template void invokeAddBiasGeluRow(int8_t* out,
                                   const int8_t* in,
                                   const float* bias,
                                   const int m,
                                   const int n,
                                   cudaStream_t stream,
                                   const float* input_deQFactor_ptr,
                                   const float* out_scale_ptr);

template void invokeAddBiasGeluRow(int8_t* out,
                                   const int8_t* in,
                                   const half* bias,
                                   const int m,
                                   const int n,
                                   cudaStream_t stream,
                                   const float* input_deQFactor_ptr,
                                   const float* out_scale_ptr);

}  // namespace fastertransformer