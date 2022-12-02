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

#include "src/fastertransformer/kernels/image_shift_partition_kernels.h"
#include "src/fastertransformer/utils/cuda_type_utils.cuh"

namespace fastertransformer {

/*******************  invokeShiftPartition  ***********************/

// applied to half2 and bfloat162
template<typename T2>
__global__ void
shift_partition(T2* out_ptr, const T2* input_ptr, int batch, int H, int W, int n, int shift_size, int window_size)
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
    float2    local_out_fp2;

    float local_out = 0.0f;
    int   id        = bid * n + tid;
    if (tid < n) {
        out_ptr[output_bid * n + tid] = input_ptr[id];
    }
}

// applied to float
template<>
__global__ void
shift_partition<float>(float* out, const float* input, int batch, int H, int W, int n, int shift_size, int window_size)
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
    if (tid < n) {
        out[output_bid * n + tid] = (float)(__ldg(input + bid * n + tid));
    }
}

// Applied to half2 and bfloat162
template<typename T2>
__global__ void shift_partition_v2(
    T2* out_ptr, const T2* __restrict input_ptr, int batch, int H, int W, int n, int shift_size, int window_size)
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
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        if (col_id < n) {
            out_ptr[output_offset + col_id] = ldg(input_ptr + offset + col_id);
        }
    }
}

template<>
__global__ void shift_partition_v2<float>(
    float* out, const float* __restrict input, int batch, int H, int W, int n, int shift_size, int window_size)
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

#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        if (col_id < n) {
            out[output_offset + col_id] = (float)(__ldg(input + offset + col_id));
        }
    }
}

// Applied to half or Bfloat16
template<typename T>
void invokeShiftPartition(
    T* out, const T* input, int batch, int H, int W, int n, int shift_size, int window_size, cudaStream_t stream)
{
    dim3 grid(W, H, batch);
    int  blockSize = n / 2;
    blockSize      = (blockSize + 31) / 32 * 32;

    using T2 = typename TypeConverter<T>::Type;  // bf162 or half2

    if ((batch * H * W >= 512 && blockSize >= 768) || blockSize > 1024) {
        blockSize = ((blockSize / 4) + 31) / 32 * 32;
        shift_partition_v2<T2>
            <<<grid, blockSize, 0, stream>>>((T2*)out, (const T2*)input, batch, H, W, n / 2, shift_size, window_size);
    }
    else {
        shift_partition<T2>
            <<<grid, blockSize, 0, stream>>>((T2*)out, (const T2*)input, batch, H, W, n / 2, shift_size, window_size);
    }
}

template<>
void invokeShiftPartition<float>(float*       out,
                                 const float* input,
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
        shift_partition_v2<float><<<grid, blockSize, 0, stream>>>(out, input, batch, H, W, n, shift_size, window_size);
    }
    else {
        shift_partition<float><<<grid, blockSize, 0, stream>>>(out, input, batch, H, W, n, shift_size, window_size);
    }
}

template void invokeShiftPartition<float>(float*       out,
                                          const float* input,
                                          int          batch,
                                          int          H,
                                          int          W,
                                          int          n,
                                          int          shift_size,
                                          int          window_size,
                                          cudaStream_t stream);

template void invokeShiftPartition<half>(
    half* out, const half* input, int batch, int H, int W, int n, int shift_size, int window_size, cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeShiftPartition<__nv_bfloat16>(__nv_bfloat16*       out,
                                                  const __nv_bfloat16* input,
                                                  int                  batch,
                                                  int                  H,
                                                  int                  W,
                                                  int                  n,
                                                  int                  shift_size,
                                                  int                  window_size,
                                                  cudaStream_t         stream);
#endif

/*******************  invokeShiftPartitionCol32  ***********************/

template<typename T>
__global__ void shift_partition_COL32(int8_t*      out,
                                      const T*     input,
                                      int          batch,
                                      int          H,
                                      int          W,
                                      int          n,
                                      const float* norm_scale_ptr,
                                      int          shift_size,
                                      int          window_size)
{
    float     norm_scale         = __ldg(norm_scale_ptr);
    int       tid                = threadIdx.x;
    const int batch_offset       = blockIdx.z * gridDim.y * gridDim.x;
    const int m                  = gridDim.z * gridDim.y * gridDim.x;
    const int bid                = batch_offset + blockIdx.y * gridDim.x + blockIdx.x;
    const int shifted_H_idx      = (shift_size != 0) ? ((blockIdx.y - shift_size + gridDim.y) % gridDim.y) : blockIdx.y;
    const int shifted_W_idx      = (shift_size != 0) ? ((blockIdx.x - shift_size + gridDim.x) % gridDim.x) : blockIdx.x;
    const int window_H_idx       = shifted_H_idx / window_size;
    const int window_W_idx       = shifted_W_idx / window_size;
    const int stride_of_window_H = W / window_size;
    const int window_idx         = window_H_idx * stride_of_window_H + window_W_idx;
    const int idx_in_window      = (shifted_H_idx % window_size) * window_size + (shifted_W_idx % window_size);
    const int output_bid         = batch_offset + window_idx * window_size * window_size + idx_in_window;

    if (tid < n) {
        const int offset_col32_in  = (tid & 0xffffffe0) * m + (bid << 5) + (tid & 31);
        float     local_out        = (float)(__ldg(input + offset_col32_in));
        const int offset_col32_out = (tid & 0xffffffe0) * m + (output_bid << 5) + (tid & 31);
        out[offset_col32_out]      = float_to_int8_rn(norm_scale * local_out);
    }
}

template<>
__global__ void shift_partition_COL32(int8_t*      out_ptr,
                                      const half4* input_ptr,
                                      int          batch,
                                      int          H,
                                      int          W,
                                      int          n,
                                      const float* norm_scale_ptr,
                                      int          shift_size,
                                      int          window_size)
{
    float norm_scale = __ldg(norm_scale_ptr);

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

    int    tid        = threadIdx.x << 2;
    char4* output_ptr = (char4*)out_ptr;
    char4  int8_buf;

    if (tid < n) {
        const int offset_col32     = (tid & 0xffffffe0) * m + (bid << 5) + (tid & 31);
        half4     inputTmp         = input_ptr[offset_col32 >> 2];
        const int offset_col32_out = (tid & 0xffffffe0) * m + (output_bid << 5) + (tid & 31);
        // const int offset_colMajor_out = output_bid * n + tid;
        // const int offset_out = index_CUBLASLT_ORDER_COL32_2R_4R4(tid, output_bid, m << 5);
        int8_buf.x                        = float_to_int8_rn(norm_scale * static_cast<float>(inputTmp.x));
        int8_buf.y                        = float_to_int8_rn(norm_scale * static_cast<float>(inputTmp.y));
        int8_buf.z                        = float_to_int8_rn(norm_scale * static_cast<float>(inputTmp.z));
        int8_buf.w                        = float_to_int8_rn(norm_scale * static_cast<float>(inputTmp.w));
        output_ptr[offset_col32_out >> 2] = int8_buf;
    }
}

template<typename T>
__global__ void shift_partition_v2_COL32(int8_t* out,
                                         const T* __restrict input,
                                         int          batch,
                                         int          H,
                                         int          W,
                                         int          n,
                                         const float* norm_scale_ptr,
                                         int          shift_size,
                                         int          window_size)
{
    float     norm_scale         = __ldg(norm_scale_ptr);
    const int ite                = 4;
    const int tid                = threadIdx.x;
    const int batch_offset       = blockIdx.z * gridDim.y * gridDim.x;
    const int m                  = gridDim.z * gridDim.y * gridDim.x;
    const int bid                = batch_offset + blockIdx.y * gridDim.x + blockIdx.x;
    const int shifted_H_idx      = (shift_size != 0) ? ((blockIdx.y - shift_size + gridDim.y) % gridDim.y) : blockIdx.y;
    const int shifted_W_idx      = (shift_size != 0) ? ((blockIdx.x - shift_size + gridDim.x) % gridDim.x) : blockIdx.x;
    const int window_H_idx       = shifted_H_idx / window_size;
    const int window_W_idx       = shifted_W_idx / window_size;
    const int stride_of_window_H = W / window_size;
    const int window_idx         = window_H_idx * stride_of_window_H + window_W_idx;
    const int idx_in_window      = (shifted_H_idx % window_size) * window_size + (shifted_W_idx % window_size);
    const int output_bid         = batch_offset + window_idx * window_size * window_size + idx_in_window;

#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        if (col_id < n) {
            const int offset_col32_in  = (col_id & 0xffffffe0) * m + (bid << 5) + (col_id & 31);
            float     local_out        = (float)(__ldg(input + offset_col32_in));
            const int offset_col32_out = (col_id & 0xffffffe0) * m + (output_bid << 5) + (col_id & 31);
            out[offset_col32_out]      = float_to_int8_rn(norm_scale * local_out);
        }
    }
}

template<>
__global__ void shift_partition_v2_COL32(int8_t* out_ptr,
                                         const half2* __restrict input_ptr,
                                         int          batch,
                                         int          H,
                                         int          W,
                                         int          n,
                                         const float* norm_scale_ptr,
                                         int          shift_size,
                                         int          window_size)
{
    float     norm_scale   = __ldg(norm_scale_ptr);
    const int ite          = 4;
    const int tid          = threadIdx.x;
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

    char2* output_ptr = (char2*)out_ptr;
    char2  int8_buf;

#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = (i * blockDim.x + tid) << 1;
        if (col_id < n) {
            const int offset_col32            = (col_id & 0xffffffe0) * m + (bid << 5) + (col_id & 31);
            half2     outVal                  = __ldg(input_ptr + (offset_col32 >> 1));
            const int offset_col32_out        = (col_id & 0xffffffe0) * m + (output_bid << 5) + (col_id & 31);
            int8_buf.x                        = float_to_int8_rn(norm_scale * static_cast<float>(outVal.x));
            int8_buf.y                        = float_to_int8_rn(norm_scale * static_cast<float>(outVal.y));
            output_ptr[offset_col32_out >> 1] = int8_buf;
        }
    }
}

template<typename T>
void invokeShiftPartitionCol32(int8_t*      out,
                               const T*     input,
                               int          batch,
                               int          H,
                               int          W,
                               int          n,
                               const float* norm_scale_ptr,
                               int          shift_size,
                               int          window_size,
                               cudaStream_t stream)
{
    dim3 grid(W, H, batch);
    int  blockSize = (n + 31) / 32 * 32;
    if (blockSize >= 768) {
        blockSize = ((blockSize / 4) + 31) / 32 * 32;
        shift_partition_v2_COL32<T>
            <<<grid, blockSize, 0, stream>>>(out, input, batch, H, W, n, norm_scale_ptr, shift_size, window_size);
    }
    else {
        shift_partition_COL32<T>
            <<<grid, blockSize, 0, stream>>>(out, input, batch, H, W, n, norm_scale_ptr, shift_size, window_size);
    }
}

template<>
void invokeShiftPartitionCol32(int8_t*      out,
                               const half*  input,
                               int          batch,
                               int          H,
                               int          W,
                               int          n,
                               const float* scale_ptr,
                               int          shift_size,
                               int          window_size,
                               cudaStream_t stream)
{
    dim3 grid(W, H, batch);
    int  blockSize = n / 2;
    blockSize      = (blockSize + 31) / 32 * 32;

    if ((batch * H * W >= 512 && blockSize >= 768) || blockSize > 1024) {
        blockSize = ((blockSize / 4) + 31) / 32 * 32;
        shift_partition_v2_COL32<<<grid, blockSize, 0, stream>>>(
            out, (const half2*)input, batch, H, W, n, scale_ptr, shift_size, window_size);
    }
    else {
        blockSize = (n / 4 + 32) / 32 * 32;
        shift_partition_COL32<<<grid, blockSize, 0, stream>>>(
            out, (const half4*)input, batch, H, W, n, scale_ptr, shift_size, window_size);
    }
}

template void invokeShiftPartitionCol32(int8_t*      out,
                                        const float* input,
                                        int          batch,
                                        int          H,
                                        int          W,
                                        int          n,
                                        const float* scale_ptr,
                                        int          shift_size,
                                        int          window_size,
                                        cudaStream_t stream);

template void invokeShiftPartitionCol32(int8_t*      out,
                                        const half*  input,
                                        int          batch,
                                        int          H,
                                        int          W,
                                        int          n,
                                        const float* scale_ptr,
                                        int          shift_size,
                                        int          window_size,
                                        cudaStream_t stream);

}  // namespace fastertransformer
