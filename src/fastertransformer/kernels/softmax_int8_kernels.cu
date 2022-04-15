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

#include "int8_utils.cuh"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"
#include "src/fastertransformer/kernels/softmax_int8_kernels.h"

namespace fastertransformer {

// input are a series of sub-matrixes of m = seq_len, n = seq_len, CUBLASLT_ORDER_COL32
// grid = (seq_len, batch_size, head_num)
// block.x = max(32, (seq_len/4 + 31)/32*32)
// for int32_t I; int8 O;
template<typename T>
__global__ void softmax_COL32(int8_t* output,
                              const int32_t* input,
                              const T* attr_mask,
                              const int batch_size,
                              const int head_num,
                              const int seq_len,
                              const float scalar1a,
                              const float* scalar1b,
                              const float* scalar1c,
                              const float* amax_ptr,
                              const int head_num_x_seq_len,
                              const int seq_len_x_seq_len)
{
    const float amax = __ldg(amax_ptr);
    const float scalar1 = scalar1a * __ldg(scalar1b) * __ldg(scalar1c);
    int mask_id;
    int threadIdx4 = threadIdx.x << 2;

    char4* buf4Ptr = (char4*)output;

    bool qual = threadIdx4 < seq_len;
    for (int seq_id = blockIdx.x; seq_id < seq_len; seq_id += gridDim.x) {
        char4 tmp4 = {0, 0, 0, 0};
        int inIdx = (blockIdx.y * head_num + blockIdx.z) * (seq_len_x_seq_len) + (threadIdx4 & 0xffffffe0) * seq_len
                    + (seq_id << 5) + (threadIdx4 & 31);

        // set softmax of padding word to 0
        float mask_in_seq = static_cast<float>(__ldg(attr_mask + (blockIdx.y * seq_len_x_seq_len + seq_id)));
        if (mask_in_seq < 0.1f) {
            if (qual) {
                buf4Ptr[inIdx >> 2] = tmp4;
            }
            continue;
        }

        float4 floatTmp4 = {0.0f, 0.0f, 0.0f, 0.0f};

        if (qual) {
            floatTmp4.x = static_cast<float>(__ldg(input + inIdx)) * scalar1;
            floatTmp4.y = static_cast<float>(__ldg(input + inIdx + 1)) * scalar1;
            floatTmp4.z = static_cast<float>(__ldg(input + inIdx + 2)) * scalar1;
            floatTmp4.w = static_cast<float>(__ldg(input + inIdx + 3)) * scalar1;
        }

        float mask_val, max_val;
        max_val = -1e20f;

        __shared__ float s_max, s_sum;

        if (qual) {
            mask_id = threadIdx4 + blockIdx.y * seq_len_x_seq_len + seq_id * seq_len;
            // for x
            mask_val = (1.0f - static_cast<float>(__ldg(attr_mask + mask_id))) * -10000.0f;
            floatTmp4.x = floatTmp4.x + mask_val;
            max_val = fmaxf(max_val, floatTmp4.x);

            // for y
            mask_val = (1.0f - static_cast<float>(__ldg(attr_mask + mask_id + 1))) * -10000.0f;
            floatTmp4.y = floatTmp4.y + mask_val;
            max_val = fmaxf(max_val, floatTmp4.y);

            // for z
            mask_val = (1.0f - static_cast<float>(__ldg(attr_mask + mask_id + 2))) * -10000.0f;
            floatTmp4.z = floatTmp4.z + mask_val;
            max_val = fmaxf(max_val, floatTmp4.z);

            // for w
            mask_val = (1.0f - static_cast<float>(__ldg(attr_mask + mask_id + 3))) * -10000.0f;
            floatTmp4.w = floatTmp4.w + mask_val;
            max_val = fmaxf(max_val, floatTmp4.w);
        }

        max_val = blockDim.x <= 32 ? warpReduceMax(max_val) : blockReduceMax<float>(max_val);

        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float sum_val = 0.0f;

        if (qual) {
            floatTmp4.x = __expf(floatTmp4.x - s_max);
            sum_val += floatTmp4.x;
            floatTmp4.y = __expf(floatTmp4.y - s_max);
            sum_val += floatTmp4.y;
            floatTmp4.z = __expf(floatTmp4.z - s_max);
            sum_val += floatTmp4.z;
            floatTmp4.w = __expf(floatTmp4.w - s_max);
            sum_val += floatTmp4.w;
        }

        sum_val = blockDim.x <= 32 ? warpReduceSum(sum_val) : blockReduceSum<float>(sum_val);

        if (threadIdx.x == 0) {
            s_sum = __fdividef(127.0f, (sum_val + 1e-6f));
            s_sum = __fdividef(s_sum, amax);
        }
        __syncthreads();

        if (qual) {

            tmp4.x = float_to_int8_rn(floatTmp4.x * s_sum);
            tmp4.y = float_to_int8_rn(floatTmp4.y * s_sum);
            tmp4.z = float_to_int8_rn(floatTmp4.z * s_sum);
            tmp4.w = float_to_int8_rn(floatTmp4.w * s_sum);

            buf4Ptr[inIdx >> 2] = tmp4;
        }
    }
}

// input are a series of sub-matrixes of m = seq_len, n = seq_len_padded, CUBLASLT_ORDER_COL32
// seq_len_padded = (seq_len+31)/32*32
// grid = (seq_len, batch_size, head_num)
// block.x = max(32, (seq_len_padded/4 + 31)/32*32)
// for int8_t IO;
template<typename T>
__global__ void softmax_COL32_varlen(int8_t* output,
                                     const int8_t* input,
                                     const T* attr_mask,
                                     const int batch_size,
                                     const int head_num,
                                     const int seq_len,
                                     const int seq_len_padded,
                                     const float scalar1a,
                                     const float* scalar1b,
                                     const float* amax_ptr,
                                     const int seq_len_x_seq_len,
                                     const int seq_len_x_seq_len_padded)
{
    const float amax = __ldg(amax_ptr);
    const float scalar1 = scalar1a * __ldg(scalar1b);
    int mask_id;
    int threadIdx4 = threadIdx.x << 2;

    char4* buf4Ptr = (char4*)output;
    const char4* inBuf4Ptr = (const char4*)input;

    const bool qual = threadIdx4 < seq_len;
    const bool qual_padded = threadIdx4 < seq_len_padded;
    for (int seq_id = blockIdx.x; seq_id < seq_len; seq_id += gridDim.x) {

        char4 tmp4 = {0, 0, 0, 0};
        int inIdx = ((blockIdx.y * head_num + blockIdx.z) * (seq_len_x_seq_len_padded)
                     + (threadIdx4 & 0xffffffe0) * seq_len + (seq_id << 5) + (threadIdx4 & 31))
                    >> 2;

        // set softmax of padding word in rows to 0
        const float mask_in_seq = static_cast<float>(__ldg(attr_mask + (blockIdx.y * seq_len_x_seq_len + seq_id)));
        if (mask_in_seq < 0.1f) {
            if (qual_padded) {
                buf4Ptr[inIdx] = tmp4;
            }
            continue;
        }

        // set softmax of padding word in cols to 0
        float4 floatTmp4 = {0.0f, 0.0f, 0.0f, 0.0f};
        if (qual) {
            tmp4 = __ldg(inBuf4Ptr + inIdx);
            floatTmp4.x = static_cast<float>(tmp4.x) * scalar1;
            floatTmp4.y = static_cast<float>(tmp4.y) * scalar1;
            floatTmp4.z = static_cast<float>(tmp4.z) * scalar1;
            floatTmp4.w = static_cast<float>(tmp4.w) * scalar1;
        }

        float mask_val, max_val;
        max_val = -1e20f;

        __shared__ float s_max, s_sum;

        if (qual) {
            mask_id = threadIdx4 + blockIdx.y * seq_len_x_seq_len + seq_id * seq_len;
            // for x
            mask_val = (1.0f - static_cast<float>(__ldg(attr_mask + mask_id))) * -10000.0f;
            floatTmp4.x = floatTmp4.x + mask_val;
            max_val = fmaxf(max_val, floatTmp4.x);

            // for y
            mask_val = (1.0f - static_cast<float>(__ldg(attr_mask + mask_id + 1))) * -10000.0f;
            floatTmp4.y = floatTmp4.y + mask_val;
            max_val = fmaxf(max_val, floatTmp4.y);

            // for z
            mask_val = (1.0f - static_cast<float>(__ldg(attr_mask + mask_id + 2))) * -10000.0f;
            floatTmp4.z = floatTmp4.z + mask_val;
            max_val = fmaxf(max_val, floatTmp4.z);

            // for w
            mask_val = (1.0f - static_cast<float>(__ldg(attr_mask + mask_id + 3))) * -10000.0f;
            floatTmp4.w = floatTmp4.w + mask_val;
            max_val = fmaxf(max_val, floatTmp4.w);
        }

        max_val = blockDim.x <= 32 ? warpReduceMax(max_val) : blockReduceMax<float>(max_val);

        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float sum_val = 0.0f;

        if (qual) {
            floatTmp4.x = __expf(floatTmp4.x - s_max);
            sum_val += floatTmp4.x;
            floatTmp4.y = __expf(floatTmp4.y - s_max);
            sum_val += floatTmp4.y;
            floatTmp4.z = __expf(floatTmp4.z - s_max);
            sum_val += floatTmp4.z;
            floatTmp4.w = __expf(floatTmp4.w - s_max);
            sum_val += floatTmp4.w;
        }

        sum_val = blockDim.x <= 32 ? warpReduceSum(sum_val) : blockReduceSum<float>(sum_val);

        if (threadIdx.x == 0) {
            s_sum = __fdividef(127.0f, (sum_val + 1e-6f));
            s_sum = __fdividef(s_sum, amax);
        }
        __syncthreads();

        if (qual_padded) {

            tmp4.x = qual ? float_to_int8_rn(floatTmp4.x * s_sum) : static_cast<int8_t>(0);
            tmp4.y = qual ? float_to_int8_rn(floatTmp4.y * s_sum) : static_cast<int8_t>(0);
            tmp4.z = qual ? float_to_int8_rn(floatTmp4.z * s_sum) : static_cast<int8_t>(0);
            tmp4.w = qual ? float_to_int8_rn(floatTmp4.w * s_sum) : static_cast<int8_t>(0);

            buf4Ptr[inIdx] = tmp4;
        }
    }
}

// input are a series of sub-matrixes of m = seq_len, n = seq_len_padded, CUBLASLT_ORDER_COL32
// seq_len_padded = (seq_len+31)/32*32
// grid = (seq_len, batch_size, head_num)
// block.x = max(32, (seq_len_padded + 31)/32*32)
// for int8_t IO, I/O with int8_t element;
template<typename T>
__global__ void softmax_COL32_perElement_varlen(int8_t* output,
                                                const int8_t* input,
                                                const T* attr_mask,
                                                const int batch_size,
                                                const int head_num,
                                                const int seq_len,
                                                const int seq_len_padded,
                                                const float scalar1a,
                                                const float* scalar1b,
                                                const float* amax_ptr,
                                                const int seq_len_x_seq_len,
                                                const int seq_len_x_seq_len_padded)
{
    const float amax = __ldg(amax_ptr);
    const float scalar1 = scalar1a * __ldg(scalar1b);
    int mask_id;
    const int tidx = threadIdx.x;

    const bool qual = tidx < seq_len;
    const bool qual_padded = tidx < seq_len_padded;
    for (int seq_id = blockIdx.x; seq_id < seq_len; seq_id += gridDim.x) {

        int8_t tmp = 0;
        int inIdx = ((blockIdx.y * head_num + blockIdx.z) * (seq_len_x_seq_len_padded) + (tidx & 0xffffffe0) * seq_len
                     + (seq_id << 5) + (tidx & 31));

        // set softmax of padding word in rows to 0
        const float mask_in_seq = static_cast<float>(__ldg(attr_mask + (blockIdx.y * seq_len_x_seq_len + seq_id)));
        if (mask_in_seq < 0.1f) {
            if (qual_padded) {
                output[inIdx] = tmp;
            }
            continue;
        }

        // set softmax of padding word in cols to 0
        float floatTmp = qual ? (static_cast<float>(__ldg(input + inIdx)) * scalar1) : 0.0f;

        float mask_val, max_val;
        max_val = -1e20f;

        __shared__ float s_max, s_sum;

        if (qual) {
            mask_id = tidx + blockIdx.y * seq_len_x_seq_len + seq_id * seq_len;
            mask_val = (1.0f - static_cast<float>(__ldg(attr_mask + mask_id))) * -10000.0f;
            floatTmp = floatTmp + mask_val;
        }

        max_val = blockDim.x <= 32 ? warpReduceMax(floatTmp) : blockReduceMax<float>(floatTmp);

        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float sum_val = 0.0f;

        floatTmp = qual ? __expf(floatTmp - s_max) : floatTmp;

        sum_val = blockDim.x <= 32 ? warpReduceSum(floatTmp) : blockReduceSum<float>(floatTmp);

        if (threadIdx.x == 0) {
            s_sum = __fdividef(127.0f, (sum_val + 1e-6f));
            s_sum = __fdividef(s_sum, amax);
        }
        __syncthreads();

        if (qual_padded) {
            tmp = qual ? float_to_int8_rn(floatTmp * s_sum) : static_cast<int8_t>(0);
            output[inIdx] = tmp;
        }
    }
}

// input are a series of sub-matrixes of m = seq_len, n = seq_len, CUBLASLT_ORDER_COL32
// grid = (seq_len, batch_size, head_num)
// block.x = (seq_len + 31)/32
// for int32_t I; int8 O;
// for seq_len <= 32
template<typename T>
__global__ void softmax_COL32_LE32(int8_t* output,
                                   const int32_t* input,
                                   const T* attr_mask,
                                   const int batch_size,
                                   const int head_num,
                                   const int seq_len,
                                   const float scalar1a,
                                   const float* scalar1b,
                                   const float* scalar1c,
                                   const float* amax_ptr,
                                   const int head_num_x_seq_len,
                                   const int seq_len_x_seq_len)
{
    const float amax = __ldg(amax_ptr);
    const float scalar1 = scalar1a * __ldg(scalar1b) * __ldg(scalar1c);
    int mask_id;
    int threadIdxx = threadIdx.x;
    bool qual = threadIdxx < seq_len;
    for (int seq_id = blockIdx.x; seq_id < seq_len; seq_id += gridDim.x) {
        int inIdx = (blockIdx.y * head_num + blockIdx.z) * (seq_len_x_seq_len) + (threadIdxx & 0xffffffe0) * seq_len
                    + (seq_id << 5) + (threadIdxx & 31);

        // set softmax of padding word to 0
        float mask_in_seq = static_cast<float>(__ldg(attr_mask + (blockIdx.y * seq_len_x_seq_len + seq_id)));
        if (mask_in_seq < 0.1f) {
            if (qual) {
                output[inIdx] = 0;
            }
            continue;
        }

        float floatTmp = qual ? static_cast<float>(__ldg(input + inIdx)) * scalar1 : 0.0f;

        float mask_val, max_val;

        __shared__ float s_max, s_sum;

        mask_id = qual ? threadIdxx + blockIdx.y * seq_len_x_seq_len + seq_id * seq_len : 0;
        mask_val = qual ? (1.0f - static_cast<float>(__ldg(attr_mask + mask_id))) * -10000.0f : 0.0f;
        floatTmp = qual ? floatTmp + mask_val : 0.0f;
        max_val = qual ? floatTmp : -1e20f;

        max_val = blockDim.x <= 32 ? warpReduceMax(max_val) : blockReduceMax<float>(max_val);

        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        floatTmp = qual ? __expf(floatTmp - s_max) : 0.0f;

        float sum_val = blockDim.x <= 32 ? warpReduceSum(floatTmp) : blockReduceSum<float>(floatTmp);

        if (threadIdx.x == 0) {
            s_sum = __fdividef(127.0f, (sum_val + 1e-6f));
            s_sum = __fdividef(s_sum, amax);
        }
        __syncthreads();

        if (qual) {
            output[inIdx] = float_to_int8_rn(floatTmp * s_sum);
        }
    }
}

// input are a series of sub-matrixes of m = seq_len, n = seq_len_padded, CUBLASLT_ORDER_COL32
// seq_len_padded = (seq_len+31)/32*32
// attr_mask is [batch_size, seq_len, seq_len]
// grid = (seq_len, batch_size, head_num)
// block.x = seq_len_padded
// for int8_t IO;
// for seq_len_padded == 32
template<typename T>
__global__ void softmax_COL32_LE32_varlen(int8_t* output,
                                          const int8_t* input,
                                          const T* attr_mask,
                                          const int batch_size,
                                          const int head_num,
                                          const int seq_len,
                                          const int seq_len_padded,
                                          const float scalar1a,
                                          const float* scalar1b,
                                          const float* amax_ptr,
                                          const int seq_len_x_seq_len,
                                          const int seq_len_x_seq_len_padded)
{
    const float amax = __ldg(amax_ptr);
    const float scalar1 = scalar1a * __ldg(scalar1b);
    int mask_id;
    int threadIdxx = threadIdx.x;
    const bool qual = threadIdxx < seq_len;
    const bool qual_padded = threadIdxx < seq_len_padded;
    for (int seq_id = blockIdx.x; seq_id < seq_len; seq_id += gridDim.x) {
        int inIdx = (blockIdx.y * head_num + blockIdx.z) * (seq_len_x_seq_len_padded)
                    + (threadIdxx & 0xffffffe0) * seq_len + (seq_id << 5) + (threadIdxx & 31);

        // set softmax of padding word in rows to 0
        float mask_in_seq = static_cast<float>(__ldg(attr_mask + (blockIdx.y * seq_len_x_seq_len + seq_id)));
        if (mask_in_seq < 0.1f) {
            if (qual_padded) {
                output[inIdx] = 0;
            }
            continue;
        }

        float mask_val, max_val;
        __shared__ float s_max, s_sum;

        // set softmax of padding word in cols to 0
        float floatTmp = qual ? static_cast<float>(__ldg(input + inIdx)) * scalar1 : 0.0f;
        mask_id = qual ? threadIdxx + blockIdx.y * seq_len_x_seq_len + seq_id * seq_len : 0;
        mask_val = qual ? (1.0f - static_cast<float>(__ldg(attr_mask + mask_id))) * -10000.0f : 0.0f;
        floatTmp = qual ? floatTmp + mask_val : 0.0f;
        max_val = qual ? floatTmp : -1e20f;

        max_val = warpReduceMax(max_val);

        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        floatTmp = qual ? __expf(floatTmp - s_max) : 0.0f;

        float sum_val = blockDim.x <= 32 ? warpReduceSum(floatTmp) : blockReduceSum<float>(floatTmp);

        if (threadIdx.x == 0) {
            s_sum = __fdividef(127.0f, (sum_val + 1e-6f));
            s_sum = __fdividef(s_sum, amax);
        }
        __syncthreads();

        if (qual_padded) {
            output[inIdx] = qual ? float_to_int8_rn(floatTmp * s_sum) : static_cast<int8_t>(0);
        }
    }
}

// input are a series of sub-matrixes of m = seq_len, n = seq_len, CUBLASLT_ORDER_COL32
// grid = (seq_len, batch_size, head_num)
// block.x = max(32, (seq_len/2 + 31)/32*32)
// for int32_t I; int8 O;
// for seq_len in (32, 64]
template<typename T>
__global__ void softmax_COL32_LE64(int8_t* output,
                                   const int32_t* input,
                                   const T* attr_mask,
                                   const int batch_size,
                                   const int head_num,
                                   const int seq_len,
                                   const float scalar1a,
                                   const float* scalar1b,
                                   const float* scalar1c,
                                   const float* amax_ptr,
                                   const int head_num_x_seq_len,
                                   const int seq_len_x_seq_len)
{
    const float amax = __ldg(amax_ptr);
    const float scalar1 = scalar1a * __ldg(scalar1b) * __ldg(scalar1c);
    int mask_id;
    int threadIdx2 = threadIdx.x << 1;

    char2* buf2Ptr = (char2*)output;

    bool qual = threadIdx2 < seq_len;
    for (int seq_id = blockIdx.x; seq_id < seq_len; seq_id += gridDim.x) {
        char2 tmp2 = {0, 0};
        int inIdx = (blockIdx.y * head_num + blockIdx.z) * (seq_len_x_seq_len) + (threadIdx2 & 0xffffffe0) * seq_len
                    + (seq_id << 5) + (threadIdx2 & 31);

        // set softmax of padding word to 0
        float mask_in_seq = static_cast<float>(__ldg(attr_mask + (blockIdx.y * seq_len_x_seq_len + seq_id)));
        if (mask_in_seq < 0.1f) {
            if (qual) {
                buf2Ptr[inIdx >> 1] = tmp2;
            }
            continue;
        }

        float2 floatTmp2 = {0.0f, 0.0f};
        if (qual) {
            floatTmp2.x = static_cast<float>(__ldg(input + inIdx)) * scalar1;
            floatTmp2.y = static_cast<float>(__ldg(input + inIdx + 1)) * scalar1;
        }

        float mask_val, max_val;
        max_val = -1e20f;

        __shared__ float s_max, s_sum;

        if (qual) {
            mask_id = threadIdx2 + blockIdx.y * seq_len_x_seq_len + seq_id * seq_len;
            // for x
            mask_val = (1.0f - static_cast<float>(__ldg(attr_mask + mask_id))) * -10000.0f;
            floatTmp2.x = floatTmp2.x + mask_val;

            // for y
            mask_val = (1.0f - static_cast<float>(__ldg(attr_mask + mask_id + 1))) * -10000.0f;
            floatTmp2.y = floatTmp2.y + mask_val;

            max_val = fmaxf(floatTmp2.x, floatTmp2.y);
        }

        max_val = blockDim.x <= 32 ? warpReduceMax(max_val) : blockReduceMax<float>(max_val);

        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float sum_val = 0.0f;

        if (qual) {
            floatTmp2.x = __expf(floatTmp2.x - s_max);
            sum_val += floatTmp2.x;
            floatTmp2.y = __expf(floatTmp2.y - s_max);
            sum_val += floatTmp2.y;
        }

        sum_val = blockDim.x <= 32 ? warpReduceSum(sum_val) : blockReduceSum<float>(sum_val);

        if (threadIdx.x == 0) {
            s_sum = __fdividef(127.0f, (sum_val + 1e-6f));
            s_sum = __fdividef(s_sum, amax);
        }
        __syncthreads();

        if (qual) {
            tmp2.x = float_to_int8_rn(floatTmp2.x * s_sum);
            tmp2.y = float_to_int8_rn(floatTmp2.y * s_sum);
            buf2Ptr[inIdx >> 1] = tmp2;
        }
    }
}

// input are a series of sub-matrixes of m = seq_len, n = seq_len_padded, CUBLASLT_ORDER_COL32
// seq_len_padded = (seq_len+31)/32*32
// grid = (seq_len, batch_size, head_num)
// block.x = 32
// for int8_t IO
// for seq_len in (32, 64]
template<typename T>
__global__ void softmax_COL32_LE64_varlen(int8_t* output,
                                          const int8_t* input,
                                          const T* attr_mask,
                                          const int batch_size,
                                          const int head_num,
                                          const int seq_len,
                                          const int seq_len_padded,
                                          const float scalar1a,
                                          const float* scalar1b,
                                          const float* amax_ptr,
                                          const int seq_len_x_seq_len,
                                          const int seq_len_x_seq_len_padded)
{
    const float amax = __ldg(amax_ptr);
    const float scalar1 = scalar1a * __ldg(scalar1b);
    int mask_id;
    int threadIdx2 = threadIdx.x << 1;

    char2* buf2Ptr = (char2*)output;
    const char2* inBuf2Ptr = (const char2*)input;

    const bool qual = threadIdx2 < seq_len;
    const bool qual_padded = threadIdx2 < seq_len_padded;
    for (int seq_id = blockIdx.x; seq_id < seq_len; seq_id += gridDim.x) {
        char2 tmp2 = {0, 0};
        int inIdx = ((blockIdx.y * head_num + blockIdx.z) * (seq_len_x_seq_len_padded)
                     + (threadIdx2 & 0xffffffe0) * seq_len + (seq_id << 5) + (threadIdx2 & 31))
                    >> 1;

        // set softmax of padding word in rows to 0
        float mask_in_seq = static_cast<float>(__ldg(attr_mask + (blockIdx.y * seq_len_x_seq_len + seq_id)));
        if (mask_in_seq < 0.1f) {
            if (qual_padded) {
                buf2Ptr[inIdx] = tmp2;
            }
            continue;
        }

        // set softmax of padding word in cols to 0
        float2 floatTmp2 = {0.0f, 0.0f};
        if (qual) {
            tmp2 = __ldg(inBuf2Ptr + inIdx);
            floatTmp2.x = static_cast<float>(tmp2.x) * scalar1;
            floatTmp2.y = static_cast<float>(tmp2.y) * scalar1;
        }

        float mask_val, max_val;
        max_val = -1e20f;

        __shared__ float s_max, s_sum;

        if (qual) {
            mask_id = threadIdx2 + blockIdx.y * seq_len_x_seq_len + seq_id * seq_len;
            // for x
            mask_val = (1.0f - static_cast<float>(__ldg(attr_mask + mask_id))) * -10000.0f;
            floatTmp2.x = floatTmp2.x + mask_val;

            // for y
            mask_val = (1.0f - static_cast<float>(__ldg(attr_mask + mask_id + 1))) * -10000.0f;
            floatTmp2.y = floatTmp2.y + mask_val;

            max_val = fmaxf(floatTmp2.x, floatTmp2.y);
        }

        max_val = warpReduceMax(max_val);

        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float sum_val = 0.0f;

        if (qual) {
            floatTmp2.x = __expf(floatTmp2.x - s_max);
            sum_val += floatTmp2.x;
            floatTmp2.y = __expf(floatTmp2.y - s_max);
            sum_val += floatTmp2.y;
        }

        sum_val = warpReduceSum(sum_val);

        if (threadIdx.x == 0) {
            s_sum = __fdividef(127.0f, (sum_val + 1e-6f));
            s_sum = __fdividef(s_sum, amax);
        }
        __syncthreads();

        if (qual_padded) {
            tmp2.x = qual ? float_to_int8_rn(floatTmp2.x * s_sum) : static_cast<int8_t>(0);
            tmp2.y = qual ? float_to_int8_rn(floatTmp2.y * s_sum) : static_cast<int8_t>(0);
            buf2Ptr[inIdx] = tmp2;
        }
    }
}

template<typename T>
void invokeSoftmaxCOL32(int8_t* output,
                        const int32_t* input,
                        const T* attr_mask,
                        const int batch_size,
                        const int head_num,
                        const int seq_len,
                        const float scalar1a,
                        const float* scalar1b,
                        const float* scalar1c,
                        const float* amax_ptr,
                        cudaStream_t stream)
{
    dim3 grid, block;
    grid.x = seq_len;
    grid.y = batch_size;
    grid.z = head_num;

    if (seq_len <= 32) {
        if (batch_size * head_num > 960) {
            grid.x = ceil(float(seq_len) / 32.0f);
        }
        block.x = (seq_len + 31) / 32 * 32;
        softmax_COL32_LE32<<<grid, block, 0, stream>>>(output,
                                                       input,
                                                       attr_mask,
                                                       batch_size,
                                                       head_num,
                                                       seq_len,
                                                       scalar1a,
                                                       scalar1b,
                                                       scalar1c,
                                                       amax_ptr,
                                                       seq_len * head_num,
                                                       seq_len * seq_len);
    }
    else if (seq_len <= 64) {
        assert(seq_len % 2 == 0);
        block.x = (seq_len / 2 + 31) / 32 * 32;
        if (batch_size * head_num > 960) {
            grid.x = ceil(float(seq_len) / 32.0f);
        }
        softmax_COL32_LE64<<<grid, block, 0, stream>>>(output,
                                                       input,
                                                       attr_mask,
                                                       batch_size,
                                                       head_num,
                                                       seq_len,
                                                       scalar1a,
                                                       scalar1b,
                                                       scalar1c,
                                                       amax_ptr,
                                                       seq_len * head_num,
                                                       seq_len * seq_len);
    }
    else {
        assert(seq_len % 4 == 0);
        block.x = (seq_len / 4 + 31) / 32 * 32;
        softmax_COL32<<<grid, block, 0, stream>>>(output,
                                                  input,
                                                  attr_mask,
                                                  batch_size,
                                                  head_num,
                                                  seq_len,
                                                  scalar1a,
                                                  scalar1b,
                                                  scalar1c,
                                                  amax_ptr,
                                                  seq_len * head_num,
                                                  seq_len * seq_len);
    }
}

template void invokeSoftmaxCOL32(int8_t* output,
                                 const int32_t* input,
                                 const float* attr_mask,
                                 const int batch_size,
                                 const int head_num,
                                 const int seq_len,
                                 const float scalar1a,
                                 const float* scalar1b,
                                 const float* scalar1c,
                                 const float* amax_ptr,
                                 cudaStream_t stream);

template void invokeSoftmaxCOL32(int8_t* output,
                                 const int32_t* input,
                                 const half* attr_mask,
                                 const int batch_size,
                                 const int head_num,
                                 const int seq_len,
                                 const float scalar1a,
                                 const float* scalar1b,
                                 const float* scalar1c,
                                 const float* amax_ptr,
                                 cudaStream_t stream);

template<typename T>
void invokeSoftmaxCOL32(int8_t* output,
                        const int8_t* input,
                        const T* attr_mask,
                        const int batch_size,
                        const int head_num,
                        const int seq_len,
                        const float scalar1a,
                        const float* scalar1b,
                        const float* amax_ptr,
                        cudaStream_t stream)
{
    dim3 grid, block;
    grid.x = seq_len;
    grid.y = batch_size;
    grid.z = head_num;
    const int seq_len_padded = (seq_len + 31) / 32 * 32;

    if (seq_len <= 32) {
        if (batch_size * head_num > 960) {
            grid.x = ceil(float(seq_len) / 32.0f);
        }
        block.x = seq_len_padded;
        softmax_COL32_LE32_varlen<<<grid, block, 0, stream>>>(output,
                                                              input,
                                                              attr_mask,
                                                              batch_size,
                                                              head_num,
                                                              seq_len,
                                                              seq_len_padded,
                                                              scalar1a,
                                                              scalar1b,
                                                              amax_ptr,
                                                              seq_len * seq_len,
                                                              seq_len * seq_len_padded);
    }
    else if (seq_len <= 64 && (seq_len % 2 == 0)) {
        block.x = 32;
        if (batch_size * head_num > 960) {
            grid.x = ceil(float(seq_len) / 32.0f);
        }
        softmax_COL32_LE64_varlen<<<grid, block, 0, stream>>>(output,
                                                              input,
                                                              attr_mask,
                                                              batch_size,
                                                              head_num,
                                                              seq_len,
                                                              seq_len_padded,
                                                              scalar1a,
                                                              scalar1b,
                                                              amax_ptr,
                                                              seq_len * seq_len,
                                                              seq_len * seq_len_padded);
    }
    else if (seq_len > 64 && (seq_len % 4 == 0)) {
        block.x = (seq_len_padded / 4 + 31) / 32 * 32;
        softmax_COL32_varlen<<<grid, block, 0, stream>>>(output,
                                                         input,
                                                         attr_mask,
                                                         batch_size,
                                                         head_num,
                                                         seq_len,
                                                         seq_len_padded,
                                                         scalar1a,
                                                         scalar1b,
                                                         amax_ptr,
                                                         seq_len * seq_len,
                                                         seq_len * seq_len_padded);
    }
    else {
        block.x = (seq_len_padded + 31) / 32 * 32;
        softmax_COL32_perElement_varlen<<<grid, block, 0, stream>>>(output,
                                                                    input,
                                                                    attr_mask,
                                                                    batch_size,
                                                                    head_num,
                                                                    seq_len,
                                                                    seq_len_padded,
                                                                    scalar1a,
                                                                    scalar1b,
                                                                    amax_ptr,
                                                                    seq_len * seq_len,
                                                                    seq_len * seq_len_padded);
    }
}

template void invokeSoftmaxCOL32(int8_t* output,
                                 const int8_t* input,
                                 const float* attr_mask,
                                 const int batch_size,
                                 const int head_num,
                                 const int seq_len,
                                 const float scalar1a,
                                 const float* scalar1b,
                                 const float* amax_ptr,
                                 cudaStream_t stream);

template void invokeSoftmaxCOL32(int8_t* output,
                                 const int8_t* input,
                                 const half* attr_mask,
                                 const int batch_size,
                                 const int head_num,
                                 const int seq_len,
                                 const float scalar1a,
                                 const float* scalar1b,
                                 const float* amax_ptr,
                                 cudaStream_t stream);

/*******************  invokeSoftmaxCOL32  ***********************/

// grid = (window_len/word_per_thread, window_num*num_head, batch_size)
// block.x = max(32, (window_len + 31)/32*32)
// qk_buf is [batch, window_num, num_head, window_len, window_len]
// attn_mask is [window_num, window_len, window_len] + row-major
// relative_pos_bias is [num_head, window_len, window_len] + row-majot
template<typename T>
__global__ void softmax_INT8IO_kernel_COL32(int8_t* a_buf,
                                            int8_t* qk_buf_int8,
                                            const T* attn_mask,
                                            const T* relative_pos_bias,
                                            const int batch_size,
                                            const int num_head,
                                            const int window_num,
                                            const int window_len,
                                            const int window_len_x_window_len,
                                            const float scalar,
                                            const float* deQ_scale_ptr,
                                            const float* out_scale_ptr)
{

    bool qual = threadIdx.x < window_len;
    const int padded_winlen = (window_len + 31) / 32 * 32;
    for (int window_id = blockIdx.x; window_id < window_len; window_id += gridDim.x) {
        float tmp = -1e20f;
        __shared__ float s_mean, s_max;
        int qk_offset = (blockIdx.z * gridDim.y + blockIdx.y) * window_len * padded_winlen
                        + ((threadIdx.x >> 5) << 5) * window_len + (window_id << 5) + (threadIdx.x & 31);
        ;
        if (qual) {
            const int offset_in_window = window_id * window_len + threadIdx.x;

            const int relative_pos_bias_offset = (blockIdx.y % num_head) * window_len_x_window_len + offset_in_window;
            float mask_val =
                (attn_mask == nullptr) ?
                    0.0f :
                    static_cast<float>(
                        __ldg(attn_mask + ((blockIdx.y / num_head) * window_len_x_window_len + offset_in_window)));
            tmp = scalar * static_cast<float>(qk_buf_int8[qk_offset]) * __ldg(deQ_scale_ptr) + mask_val
                  + static_cast<float>(__ldg(relative_pos_bias + relative_pos_bias_offset));
        }

        float max_val = blockReduceMax<float>(tmp);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float qk_tmp = qual ? __expf(tmp - s_max) : 0.0f;
        float sum_val = blockReduceSum<float>(qk_tmp);
        if (threadIdx.x == 0) {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();
        a_buf[qk_offset] = qual ? float_to_int8_rn(qk_tmp * s_mean * __ldg(out_scale_ptr)) : 0;
    }
}

template<typename T>
void invokeSoftmaxWithRelPosBiasCOL32(int8_t* a_buf,
                                      int8_t* qk_buf_int8,
                                      const T* attn_mask,
                                      const T* relative_pos_bias,
                                      const int batch_size,
                                      const int num_head,
                                      const int window_num,
                                      const int window_len,
                                      const float scalar,
                                      const float* deQ_scale_ptr,
                                      const float* out_scale_ptr,
                                      cudaStream_t stream)
{
    dim3 grid(window_len, window_num * num_head, batch_size);
    dim3 block((window_len + 31) / 32 * 32);
    softmax_INT8IO_kernel_COL32<<<grid, block, 0, stream>>>(a_buf,
                                                            qk_buf_int8,
                                                            attn_mask,
                                                            relative_pos_bias,
                                                            batch_size,
                                                            num_head,
                                                            window_num,
                                                            window_len,
                                                            window_len * window_len,
                                                            scalar,
                                                            deQ_scale_ptr,
                                                            out_scale_ptr);
}

template void invokeSoftmaxWithRelPosBiasCOL32(int8_t* a_buf,
                                               int8_t* qk_buf_int8,
                                               const float* attn_mask,
                                               const float* relative_pos_bias,
                                               const int batch_size,
                                               const int num_head,
                                               const int window_num,
                                               const int window_len,
                                               const float scalar,
                                               const float* deQ_scale_ptr,
                                               const float* output_scale_ptr,
                                               cudaStream_t stream);

template void invokeSoftmaxWithRelPosBiasCOL32(int8_t* a_buf,
                                               int8_t* qk_buf_int8,
                                               const half* attn_mask,
                                               const half* relative_pos_bias,
                                               const int batch_size,
                                               const int num_head,
                                               const int window_num,
                                               const int window_len,
                                               const float scalar,
                                               const float* deQ_scale_ptr,
                                               const float* output_scale_ptr,
                                               cudaStream_t stream);

}  // namespace fastertransformer
