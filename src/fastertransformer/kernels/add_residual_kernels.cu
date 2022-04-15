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
__global__ void addBiasAttentionFfnResidual(T* block_output,
                                            const T* ffn_output,
                                            const T* attn_output,
                                            const T* block_input,
                                            const T* bias,
                                            const int m,
                                            const int n)
{
    const int col_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (col_index < n) {
        block_output[blockIdx.x * n + col_index] = ffn_output[blockIdx.x * n + col_index]
                                                   + attn_output[blockIdx.x * n + col_index]
                                                   + block_input[blockIdx.x * n + col_index] + bias[col_index];
    }
}

template<typename T>
__global__ void addBiasAttentionFfnResidual(
    T* block_output, const T* ffn_output, const T* attn_output, const T* bias, const int m, const int n)
{
    const int col_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (col_index < n) {
        block_output[blockIdx.x * n + col_index] +=
            ffn_output[blockIdx.x * n + col_index] + attn_output[blockIdx.x * n + col_index] + bias[col_index];
    }
}

template<typename T>
void invokeAddBiasAttentionFfnResidual(T* block_output,
                                       const T* ffn_output,
                                       const T* attn_output,
                                       const T* block_input,
                                       const T* bias,
                                       const int m,
                                       const int n,
                                       cudaStream_t stream)
{
    int blocks_per_row = ceil(float(n) / 1024);
    dim3 grid(m, blocks_per_row);
    dim3 block(min(n, 1024));
    if (block_output == block_input) {
        addBiasAttentionFfnResidual<<<grid, block, 0, stream>>>(block_output, ffn_output, attn_output, bias, m, n);
    }
    else {
        addBiasAttentionFfnResidual<<<grid, block, 0, stream>>>(
            block_output, ffn_output, attn_output, block_input, bias, m, n);
    }
}

template void invokeAddBiasResidual(
    float* output, const float* input, const float* bias, const int m, const int n, cudaStream_t stream);

template void
invokeAddBiasResidual(half* output, const half* input, const half* bias, const int m, const int n, cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeAddBiasResidual(__nv_bfloat16* output,
                                    const __nv_bfloat16* input,
                                    const __nv_bfloat16* bias,
                                    const int m,
                                    const int n,
                                    cudaStream_t stream);
#endif

template void invokeAddBiasAttentionFfnResidual(float* block_output,
                                                const float* ffn_output,
                                                const float* attn_output,
                                                const float* input,
                                                const float* bias,
                                                const int m,
                                                const int n,
                                                cudaStream_t stream);

template void invokeAddBiasAttentionFfnResidual(half* block_output,
                                                const half* ffn_output,
                                                const half* attn_output,
                                                const half* input,
                                                const half* bias,
                                                const int m,
                                                const int n,
                                                cudaStream_t stream);

template<typename T>
__global__ void T5addResidual(T* output, const T* input, const int m, const int n)
{
    const int col_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (col_index < n) {
        float out_val = (float)output[blockIdx.x * n + col_index] + (float)input[blockIdx.x * n + col_index];
        output[blockIdx.x * n + col_index] =
            (T)((std::is_same<T, half>::value && (out_val > 64512 || out_val < -64512)) ?
                    (out_val > 0 ? 64512 : -64512) :
                    out_val);
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

template<typename T>
void invokeT5AddBiasResidual(T* output, const T* input, const T* bias, const int m, const int n, cudaStream_t stream)
{
    if (bias != nullptr) {
        invokeAddBiasResidual(output, input, bias, m, n, stream);
    }
    else {
        invokeT5AddResidual(output, input, m, n, stream);
    }
    return;
}

template void invokeT5AddBiasResidual(
    float* output, const float* input, const float* bias, const int m, const int n, cudaStream_t stream);
template void invokeT5AddBiasResidual(
    half* output, const half* input, const half* bias, const int m, const int n, cudaStream_t stream);
/*******************  invokeAddBiasResidualCol32  ***********************/
// input1/input2/out matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n)
//(grid, block) must be (m, n/4)
// using char4
template<typename T>
__global__ void add_bias_input_COL32_int8I_DataTypeO(
    T* output, const int8_t* input1, const T* input2, const T* bias, int m, int n, const float* input1_deQFactor_ptr)
{
    const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
    int col_start = threadIdx.x << 2;

    float local_out[4];
    int outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 2;
    char4* input1TmpPtr = (char4*)input1;
    char4 input1Tmp = __ldg(input1TmpPtr + outIdx);

    int col_start_tmp = col_start;
    local_out[0] = static_cast<float>(input2[(outIdx << 2) + 0]) + static_cast<float>(input1Tmp.x) * input1_deQFactor
                   + static_cast<float>(__ldg(bias + col_start_tmp));
    col_start_tmp = col_start_tmp + 1;
    local_out[1] = static_cast<float>(input2[(outIdx << 2) + 1]) + static_cast<float>(input1Tmp.y) * input1_deQFactor
                   + static_cast<float>(__ldg(bias + col_start_tmp));
    col_start_tmp = col_start_tmp + 1;
    local_out[2] = static_cast<float>(input2[(outIdx << 2) + 2]) + static_cast<float>(input1Tmp.z) * input1_deQFactor
                   + static_cast<float>(__ldg(bias + col_start_tmp));
    col_start_tmp = col_start_tmp + 1;
    local_out[3] = static_cast<float>(input2[(outIdx << 2) + 3]) + static_cast<float>(input1Tmp.w) * input1_deQFactor
                   + static_cast<float>(__ldg(bias + col_start_tmp));

    for (int i = 0; i < 4; i++) {
        output[(outIdx << 2) + i] = static_cast<T>(local_out[i]);
    }
}

template<>
__global__ void add_bias_input_COL32_int8I_DataTypeO(half4* output,
                                                     const int8_t* input1,
                                                     const half4* input2,
                                                     const half4* bias,
                                                     int m,
                                                     int n,
                                                     const float* input1_deQFactor_ptr)
{
    const float input1_deQFactor = __ldg(input1_deQFactor_ptr);
    int col_start = (blockIdx.x << 5) + (threadIdx.x << 2);
    int row_start = (blockIdx.y << 5) + (threadIdx.y);

    if (col_start < n && row_start < m) {
        half4 local_out;
        int outIdx = ((col_start & 0xffffffe0) * m + (row_start << 5) + (col_start & 31)) >> 2;
        char4* input1TmpPtr = (char4*)input1;
        char4 input1Tmp = input1TmpPtr[outIdx];
        half4 input2Tmp = input2[outIdx];
        half4 biasTmp = bias[col_start >> 2];

        local_out.x = static_cast<half>((float)input1Tmp.x * input1_deQFactor + (float)biasTmp.x + (float)input2Tmp.x);
        local_out.y = static_cast<half>((float)input1Tmp.y * input1_deQFactor + (float)biasTmp.y + (float)input2Tmp.y);
        local_out.z = static_cast<half>((float)input1Tmp.z * input1_deQFactor + (float)biasTmp.z + (float)input2Tmp.z);
        local_out.w = static_cast<half>((float)input1Tmp.w * input1_deQFactor + (float)biasTmp.w + (float)input2Tmp.w);
        output[outIdx] = local_out;
    }
}

template<typename T>
void invokeAddBiasResidualCol32(T* output,
                                const int8_t* input1,
                                const T* input2,
                                const T* bias,
                                int m,
                                int n,
                                cudaStream_t stream,
                                const float* input1_deQFactor_ptr)
{
    dim3 grid((n + 31) / 32, (m + 31) / 32);
    dim3 block(8, 32);
    assert(block.x <= 1024);
    if (sizeof(T) == 2) {
        add_bias_input_COL32_int8I_DataTypeO<<<grid, block, 0, stream>>>(
            (half4*)output, input1, (const half4*)input2, (const half4*)bias, m, n, input1_deQFactor_ptr);
    }
    else {
        add_bias_input_COL32_int8I_DataTypeO<T>
            <<<grid, block, 0, stream>>>(output, input1, input2, bias, m, n, input1_deQFactor_ptr);
    }
}

template void invokeAddBiasResidualCol32(float* output,
                                         const int8_t* input1,
                                         const float* input2,
                                         const float* bias,
                                         int m,
                                         int n,
                                         cudaStream_t stream,
                                         const float* input1_deQFactor_ptr);

template void invokeAddBiasResidualCol32(half* output,
                                         const int8_t* input1,
                                         const half* input2,
                                         const half* bias,
                                         int m,
                                         int n,
                                         cudaStream_t stream,
                                         const float* input1_deQFactor_ptr);

/*******************  invokeAddBiasResidualCol32  ***********************/
// input1/input2/out matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n)
//(grid, block) must be (m, n/4)
// using char4
template<typename T>
__global__ void add_bias_input_COL32_int32I_DataTypeO(T* output,
                                                      const int32_t* input1,
                                                      const T* input2,
                                                      const T* bias,
                                                      int m,
                                                      int n,
                                                      const float* weight_amax,
                                                      const float* input1_amax_ptr,
                                                      const int scale_is_vector)
{
    int col_start = threadIdx.x << 2;
    const float4* weight_scale_ptr = (const float4*)weight_amax;
    const float4 weight_scale = __ldg(weight_scale_ptr + threadIdx.x * scale_is_vector);
    const float input1_deQ = __ldg(input1_amax_ptr) / 127.0f;

    float local_out[4];
    int outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 2;
    int4* input1TmpPtr = (int4*)input1;
    int4 input1Tmp = input1TmpPtr[outIdx];

    int col_start_tmp = col_start;
    local_out[0] = static_cast<float>(input2[(outIdx << 2) + 0])
                   + static_cast<float>(input1Tmp.x) * input1_deQ * weight_scale.x / 127.0f
                   + static_cast<float>(__ldg(bias + col_start_tmp));
    col_start_tmp = col_start_tmp + 1;
    local_out[1] = static_cast<float>(input2[(outIdx << 2) + 1])
                   + static_cast<float>(input1Tmp.y) * input1_deQ * weight_scale.y / 127.0f
                   + static_cast<float>(__ldg(bias + col_start_tmp));
    col_start_tmp = col_start_tmp + 1;
    local_out[2] = static_cast<float>(input2[(outIdx << 2) + 2])
                   + static_cast<float>(input1Tmp.z) * input1_deQ * weight_scale.z / 127.0f
                   + static_cast<float>(__ldg(bias + col_start_tmp));
    col_start_tmp = col_start_tmp + 1;
    local_out[3] = static_cast<float>(input2[(outIdx << 2) + 3])
                   + static_cast<float>(input1Tmp.w) * input1_deQ * weight_scale.w / 127.0f
                   + static_cast<float>(__ldg(bias + col_start_tmp));

    for (int i = 0; i < 4; i++) {
        output[(outIdx << 2) + i] = static_cast<T>(local_out[i]);
    }
}

template<>
__global__ void add_bias_input_COL32_int32I_DataTypeO(half4* output,
                                                      const int32_t* input1,
                                                      const half4* input2,
                                                      const half4* bias,
                                                      int m,
                                                      int n,
                                                      const float* weight_amax,
                                                      const float* input1_amax_ptr,
                                                      const int scale_is_vector)
{
    int col_start = threadIdx.x << 2;
    const float4* weight_scale_ptr = (const float4*)weight_amax;
    const float4 weight_scale = __ldg(weight_scale_ptr + threadIdx.x * scale_is_vector);
    const float input1_deQ = __ldg(input1_amax_ptr) / 127.0f;

    float local_out[4];
    int outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 2;
    int4* input1TmpPtr = (int4*)input1;
    int4 input1Tmp = input1TmpPtr[outIdx];
    half4 input2Tmp = input2[outIdx];
    half4 biasTmp = bias[threadIdx.x];

    local_out[0] = static_cast<float>(input2Tmp.x)
                   + static_cast<float>(input1Tmp.x) * input1_deQ * weight_scale.x / 127.0f
                   + static_cast<float>(biasTmp.x);
    local_out[1] = static_cast<float>(input2Tmp.y)
                   + static_cast<float>(input1Tmp.y) * input1_deQ * weight_scale.y / 127.0f
                   + static_cast<float>(biasTmp.y);
    local_out[2] = static_cast<float>(input2Tmp.z)
                   + static_cast<float>(input1Tmp.z) * input1_deQ * weight_scale.z / 127.0f
                   + static_cast<float>(biasTmp.z);
    local_out[3] = static_cast<float>(input2Tmp.w)
                   + static_cast<float>(input1Tmp.w) * input1_deQ * weight_scale.w / 127.0f
                   + static_cast<float>(biasTmp.w);

    half4 outTmp;
    outTmp.x = static_cast<half>(local_out[0]);
    outTmp.y = static_cast<half>(local_out[1]);
    outTmp.z = static_cast<half>(local_out[2]);
    outTmp.w = static_cast<half>(local_out[3]);

    output[outIdx] = outTmp;
}

template<typename T>
void invokeAddBiasResidualCol32(T* output,
                                const int32_t* input1,
                                const T* input2,
                                const T* bias,
                                int m,
                                int n,
                                cudaStream_t stream,
                                const float* weight_amax,
                                const float* input1_amax_ptr,
                                const int scale_is_vector)
{
    dim3 grid(m);
    dim3 block(n / 4);
    assert(block.x <= 1024);
    if (sizeof(T) == 2) {
        add_bias_input_COL32_int32I_DataTypeO<<<grid, block, 0, stream>>>((half4*)output,
                                                                          input1,
                                                                          (const half4*)input2,
                                                                          (const half4*)bias,
                                                                          m,
                                                                          n,
                                                                          weight_amax,
                                                                          input1_amax_ptr,
                                                                          scale_is_vector);
    }
    else {
        add_bias_input_COL32_int32I_DataTypeO<T><<<grid, block, 0, stream>>>(
            output, input1, input2, bias, m, n, weight_amax, input1_amax_ptr, scale_is_vector);
    }
}

template void invokeAddBiasResidualCol32(float* output,
                                         const int* input1,
                                         const float* input2,
                                         const float* bias,
                                         int m,
                                         int n,
                                         cudaStream_t stream,
                                         const float* weight_amax,
                                         const float* input1_amax_ptr,
                                         const int scale_is_vector);

template void invokeAddBiasResidualCol32(half* output,
                                         const int* input1,
                                         const half* input2,
                                         const half* bias,
                                         int m,
                                         int n,
                                         cudaStream_t stream,
                                         const float* weight_amax,
                                         const float* input1_amax_ptr,
                                         const int scale_is_vector);

}  // namespace fastertransformer
