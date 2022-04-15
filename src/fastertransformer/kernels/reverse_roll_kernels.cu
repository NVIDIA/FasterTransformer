/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "reduce_kernel_utils.cuh"
#include "reverse_roll_kernels.h"

namespace fastertransformer {

/*******************  invokeReverseRollCol32  ***********************/
// src is [batch*window_num, window_len, dim]
// dst is [batch, H, W, dim] + rolled
// grid(W, H, batch)
// block(min(1024, dim))

__global__ void reverse_roll_col32(int8_t* dst,
                                   const int8_t* src,
                                   const int batch,
                                   const int window_num,
                                   const int window_len,
                                   const int window_size,
                                   const int H,
                                   const int W,
                                   const int shift_size,
                                   const int dim)
{
    const int batch_idx = blockIdx.z;
    const int HW_idx = (blockIdx.y << 5) + threadIdx.y;
    if (HW_idx < H * W) {
        const int H_idx = HW_idx / W;
        const int W_idx = HW_idx % W;
        const int H_idx_shifted = (H_idx + shift_size) % H;
        const int W_idx_shifted = (W_idx + shift_size) % W;

        const int window_idx = H_idx / window_size * (W / window_size) + W_idx / window_size;
        const int idx_in_window = (H_idx % window_size) * window_size + (W_idx % window_size);
        const int input_offset = (batch_idx * window_num + window_idx) * window_len + idx_in_window;
        const int output_offset = (batch_idx * H + H_idx_shifted) * W + W_idx_shifted;
        const int m = H * W * batch;
        char4* inPtr = (char4*)src;
        char4* outPtr = (char4*)dst;
        const int col_start = (blockIdx.x << 5) + (threadIdx.x << 2);
        const int offset_col32_in = (col_start & 0xffffffe0) * m + (input_offset << 5) + (col_start & 31);
        const int offset_col32_out = (col_start & 0xffffffe0) * m + (output_offset << 5) + (col_start & 31);
        outPtr[offset_col32_out >> 2] = inPtr[offset_col32_in >> 2];
    }
}

void invokeReverseRollCol32(int8_t* dst,
                            const int8_t* src,
                            int batch,
                            int window_num,
                            int window_len,
                            int window_size,
                            int H,
                            int W,
                            int dim,
                            int shift_size,
                            cudaStream_t stream)
{
    dim3 grid((dim + 31) / 32, (H * W + 31) / 32, batch);
    dim3 block(8, 32);
    reverse_roll_col32<<<grid, block, 0, stream>>>(
        dst, src, batch, window_num, window_len, window_size, H, W, shift_size, dim);
}

/*******************  invokeReverseRoll  ***********************/
// src is [batch*window_num, window_len, dim]
// dst is [batch, H, W, dim] + rolled
// grid(W, H, batch)
// block(min(1024, dim))

template<typename T>
__global__ void reverse_roll(T* dst,
                             const T* src,
                             const int batch,
                             const int window_num,
                             const int window_len,
                             const int window_size,
                             const int H,
                             const int W,
                             const int shift_size,
                             const int dim)
{
    const int batch_idx = blockIdx.z;
    const int H_idx_shifted = (blockIdx.y + shift_size) % H;
    const int W_idx_shifted = (blockIdx.x + shift_size) % W;
    const int H_idx = blockIdx.y;
    const int W_idx = blockIdx.x;
    const int window_idx = H_idx / window_size * (W / window_size) + W_idx / window_size;
    const int idx_in_window = (H_idx % window_size) * window_size + (W_idx % window_size);
    const int input_offset = (batch_idx * window_num + window_idx) * window_len + idx_in_window;
    const int output_offset = (batch_idx * H + H_idx_shifted) * W + W_idx_shifted;
    for (int tid = threadIdx.x; tid < dim; tid += blockDim.x) {
        dst[output_offset * dim + tid] = src[input_offset * dim + tid];
    }
}

// src is [batch*window_num, window_len, dim]
// dst is [batch, H, W, dim] + rolled
// grid(W, H, batch)
// block(min(1024, dim))
template<typename T>
void invokeReverseRoll(T* dst,
                       const T* src,
                       int batch,
                       int window_num,
                       int window_len,
                       int window_size,
                       int H,
                       int W,
                       int dim,
                       int shift_size,
                       cudaStream_t stream)
{
    dim3 grid(W, H, batch);
    int blockSize = dim;
    if (std::is_same<T, half>::value && (dim % 2 == 0)) {
        blockSize = dim / 2;
        if (blockSize > 1024) {
            blockSize = 1024;
        }
        reverse_roll<<<grid, blockSize, 0, stream>>>(
            (half2*)dst, (const half2*)src, batch, window_num, window_len, window_size, H, W, shift_size, dim / 2);
    }
    else {
        if (blockSize > 1024) {
            blockSize = 1024;
        }
        reverse_roll<<<grid, blockSize, 0, stream>>>(
            dst, src, batch, window_num, window_len, window_size, H, W, shift_size, dim);
    }
}

template void invokeReverseRoll(float* dst,
                                const float* src,
                                int batch,
                                int window_num,
                                int window_len,
                                int window_size,
                                int H,
                                int W,
                                int dim,
                                int shift_size,
                                cudaStream_t stream);

template void invokeReverseRoll(half* dst,
                                const half* src,
                                int batch,
                                int window_num,
                                int window_len,
                                int window_size,
                                int H,
                                int W,
                                int dim,
                                int shift_size,
                                cudaStream_t stream);

}  // namespace fastertransformer