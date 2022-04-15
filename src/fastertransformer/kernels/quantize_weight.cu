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

#include "src/fastertransformer/kernels/quantize_weight.h"
namespace fastertransformer {

__device__ __host__ int index_CUBLASLT_ORDER_COL4_4R2_8C(int col_id, int row_id, int m_32)
{
    int new_col = col_id >> 5;
    int new_row =  // CUBLASLT_ORDER_COL4_4R2_8C
                   ////row_id/8 is the number of tile of (8 rows 32 columns) -- column-major
                   ////row_id%2 is even row, otherwise odd row
                   ////col_id%COL32_/8 is the number tile of (8 rows 8 columns)
        (((((row_id >> 3) << 3) + ((row_id & 1) << 2) + ((col_id & 31) >> 3)) << 5) +
         ////col_id%8 >= 4 is the right half of (8 rows 8 columns) tile
         ////(row_id%8/2) is (the row id of alternating 4 rows) - 1
         (((((col_id & 7) >= 4) ? 4 : 0) + ((row_id & 7) >> 1)) << 2) +
         ////col_id%4 is the id of 4 cols
         (col_id & 3));
    return new_col * m_32 + new_row;
}

__device__ __host__ int index_CUBLASLT_ORDER_COL32_2R_4R4(int col_id, int row_id, int m_32)
{
    int new_col = col_id >> 5;
    int row_in_tile = row_id & 31;
    int col_in_tile = col_id & 31;
    int new_row =  // CUBLASLT_ORDER_COL32_2R_4R4
        (((row_id >> 5) << 10) +
         //(((row%8)/2*4+row/8)*2+row%2)*32+col
         (((((((row_in_tile & 7) >> 1) << 2) + (row_in_tile >> 3)) << 1) + (row_in_tile & 1)) << 5) + col_in_tile);
    return new_col * m_32 + new_row;
}

__global__ void quantize_weight_kernel(int8_t* dst,
                                       const float* src,
                                       const float* amax,
                                       const int n,
                                       const int k,
                                       const int format,
                                       const int scale_is_vector)
{
    int tid = (blockIdx.x * blockDim.x + threadIdx.x);
    int col_idx = tid / n;
    int row_idx = tid - col_idx * n;
    int new_idx;
    if (format == 0) {
        new_idx = row_idx * k + col_idx;
    }
    else if (format == 1) {
        new_idx = index_CUBLASLT_ORDER_COL32_2R_4R4(col_idx, row_idx, 32 * n);
    }
    else if (format == 2) {
        new_idx = index_CUBLASLT_ORDER_COL4_4R2_8C(col_idx, row_idx, 32 * n);
    }
    if (tid < n * k) {
        int8_t v = float_to_int8_rn(src[tid] * 127.0 / amax[row_idx * scale_is_vector]);
        dst[new_idx] = v;
    }
}

__global__ void quantize_weight_kernel(int8_t* dst,
                                       const half* src,
                                       const float* amax,
                                       const int n,
                                       const int k,
                                       const int format,
                                       const int scale_is_vector)
{
    int tid = (blockIdx.x * blockDim.x + threadIdx.x);
    int col_idx = tid / n;
    int row_idx = tid - col_idx * n;
    int new_idx;
    if (format == 0) {
        new_idx = row_idx * k + col_idx;
    }
    else if (format == 1) {
        new_idx = index_CUBLASLT_ORDER_COL32_2R_4R4(col_idx, row_idx, 32 * n);
    }
    else if (format == 2) {
        new_idx = index_CUBLASLT_ORDER_COL4_4R2_8C(col_idx, row_idx, 32 * n);
    }
    if (tid < n * k) {
        int8_t v = float_to_int8_rn(__half2float(src[tid]) * 127.0 / amax[row_idx * scale_is_vector]);
        dst[new_idx] = v;
    }
}

template<typename T>
void invokeQuantizeWeight(int8_t* dst,
                          const T* src,
                          const float* amax,
                          const int n,
                          const int k,
                          const int format,
                          cudaStream_t stream,
                          const int scale_is_vector)
{
    if (format != 0 && format != 1 & format != 2) {
        printf("[ERROR][invokeQuantizeWeight] format must be one of 0, 1, 2. current value: %d\n", format);
        exit(-1);
    }
    if (scale_is_vector != 0 && scale_is_vector != 1) {
        printf("[ERROR][invokeQuantizeWeight] scale_is_vector must be either 0 or 1. current value: %d\n",
               scale_is_vector);
        exit(-1);
    }
    dim3 grid((n * k + 255) / 256);
    dim3 block(256);
    if (sizeof(T) == sizeof(float)) {
        quantize_weight_kernel<<<grid, block, 0, stream>>>(dst, (const float*)src, amax, n, k, format, scale_is_vector);
    }
    else if (sizeof(T) == sizeof(half)) {
        quantize_weight_kernel<<<grid, block, 0, stream>>>(dst, (const half*)src, amax, n, k, format, scale_is_vector);
    }
}

template void invokeQuantizeWeight<float>(int8_t* dst,
                                          const float* src,
                                          const float* amax,
                                          const int n,
                                          const int k,
                                          const int format,
                                          cudaStream_t stream,
                                          const int scale_is_vector);

template void invokeQuantizeWeight<half>(int8_t* dst,
                                         const half* src,
                                         const float* amax,
                                         const int n,
                                         const int k,
                                         const int format,
                                         cudaStream_t stream,
                                         const int scale_is_vector);

}  // namespace fastertransformer