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

#include <cassert>
#include <cmath>
#include <cuda_fp16.h>

#include "add_bias_transpose_kernels.h"

namespace fastertransformer {

template<typename T>
__global__ void addBiasTransposeToMultiHead(const T* matrices,
                                            const T* biases,
                                            T* output,
                                            const int batch_size,
                                            const int head_num,
                                            const int size_per_head,
                                            const int seq_len,
                                            const int matrices_num,
                                            const int x_repeat_per_block,
                                            const int y_repeat_per_block)
{
    for (int j = 0; j < y_repeat_per_block; j++) {
        int y_offset = blockIdx.y * blockDim.y * y_repeat_per_block + j * blockDim.y + threadIdx.y;
        int bias_id = -1;
        T bias_element;
        for (int i = 0; i < x_repeat_per_block; i++) {
            int x_offset = blockIdx.x * blockDim.x * x_repeat_per_block + i * blockDim.x + threadIdx.x;
            int bias_id_new = x_offset / (batch_size * seq_len);
            if (bias_id_new != bias_id) {
                bias_element = biases[bias_id_new * head_num * size_per_head + y_offset];
                bias_id = bias_id_new;
            }
            if (x_offset < batch_size * seq_len * matrices_num && y_offset < head_num * size_per_head) {
                int matrix_id = x_offset / (batch_size * seq_len);
                int batch_id = (x_offset % (batch_size * seq_len)) / seq_len;
                int seq_id = x_offset % seq_len;
                int head_id = y_offset / size_per_head;
                int head_y_offset = y_offset % size_per_head;

                int output_offset = matrix_id * batch_size * head_num * seq_len * size_per_head
                                    + batch_id * head_num * seq_len * size_per_head + head_id * seq_len * size_per_head
                                    + seq_id * size_per_head + head_y_offset;

                output[output_offset] = matrices[x_offset * head_num * size_per_head + y_offset] + bias_element;
            }
        }
    }
}

template<typename T>
void invokeAddBiasTransposeToMultiHead(const T* matrices,
                                       const T* biases,
                                       T* output,
                                       const int batch_size,
                                       const int head_num,
                                       const int size_per_head,
                                       const int seq_len,
                                       const int matrices_num,
                                       const cudaStream_t stream)
{
    /*
        Matrices are q k v, optionally global k, global k and global q for longformer(so matrices_num matrices in
       total), and each is a (batch_size, seq_len, head_num * size_per_head). tensor and the store order should also
       meets the shape. The bias should be a (matrices_num, head_num * size_per_head). This kernel will split and
       transpose single head (matrices_num, batch_size, seq_len, head_num * size_per_head) into (matrices_num,
       batch_size, head_num, seq_len, size_per_head) and add bias accordingly
    */

    // x direction is along the (matrices_num, batch_size, seq_len) direction
    // y direction is along the (head_num * size_per_head) direction

    const int x_repeat_per_block = 8;
    const int y_repeat_per_block = 1;
    const int block_dim_x = 1;
    const int block_dim_y = 32;

    const int x_total_len = matrices_num * batch_size * seq_len;
    const int y_total_len = head_num * size_per_head;

    dim3 grid((int)std::ceil((float)x_total_len / (x_repeat_per_block * block_dim_x)),
              (int)std::ceil((float)y_total_len / (y_repeat_per_block * block_dim_y)));
    dim3 block(block_dim_x, block_dim_y);

    addBiasTransposeToMultiHead<T><<<grid, block, 0, stream>>>(matrices,
                                                               biases,
                                                               output,
                                                               batch_size,
                                                               head_num,
                                                               size_per_head,
                                                               seq_len,
                                                               matrices_num,
                                                               x_repeat_per_block,
                                                               y_repeat_per_block);
}

template void invokeAddBiasTransposeToMultiHead(const float* matrices,
                                                const float* biases,
                                                float* output,
                                                const int batch_size,
                                                const int head_num,
                                                const int size_per_head,
                                                const int seq_len,
                                                const int matrices_num,
                                                const cudaStream_t stream);

template void invokeAddBiasTransposeToMultiHead(const half* matrices,
                                                const half* biases,
                                                half* output,
                                                const int batch_size,
                                                const int head_num,
                                                const int size_per_head,
                                                const int seq_len,
                                                const int matrices_num,
                                                const cudaStream_t stream);

__inline__ __device__ int target_index(int id1, int id2, int id3, int id4, int dim_1, int dim_2, int dim_3, int dim_4)
{
    return id1 * (dim_2 * dim_3 * dim_4) + id3 * (dim_2 * dim_4) + id2 * dim_4 + id4;
}

template<typename T>
__global__ void transposeMultiHeadToSingleKernel(
    T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    int batch_id = blockIdx.x / (head_num * seq_len);
    int seq_id = blockIdx.x % seq_len;
    int head_id = (blockIdx.x % (head_num * seq_len)) / seq_len;
    dst[batch_id * (head_num * seq_len * size_per_head) + seq_id * head_num * size_per_head + head_id * size_per_head
        + threadIdx.x] = src[blockIdx.x * size_per_head + threadIdx.x];
}

template<>
__global__ void transposeMultiHeadToSingleKernel<half>(
    half* src, half* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int batch_id = tid / (head_num * seq_len * size_per_head);
    int head_id = (tid % (head_num * seq_len * size_per_head)) / (seq_len * size_per_head);
    int seq_id = (tid % (seq_len * size_per_head)) / size_per_head;
    int id = tid % size_per_head;

    int target_id = target_index(batch_id, head_id, seq_id, id, batch_size, head_num, seq_len, size_per_head);
    half2* src_ptr = (half2*)src;
    half2* dst_ptr = (half2*)dst;

    dst_ptr[target_id] = src_ptr[tid];
}

template<typename T>
void invokeTransposeMultiHeadToSingle(T* dst,
                                      T* src,
                                      const int batch_size,
                                      const int seq_len,
                                      const int head_num,
                                      const int size_per_head,
                                      cudaStream_t stream)
{
    dim3 grid, block;
    if (sizeof(T) == 2) {
        const int seq_per_block = 4;
        grid.x = batch_size * head_num * seq_len / seq_per_block;
        block.x = seq_per_block * size_per_head / 2;

        assert(grid.x * seq_per_block == batch_size * head_num * seq_len);

        transposeMultiHeadToSingleKernel<T>
            <<<grid, block, 0, stream>>>(src, dst, batch_size, seq_len, head_num, size_per_head / 2);
    }
    else {
        const int seq_per_block = 1;
        grid.x = batch_size * head_num * seq_len / seq_per_block;
        block.x = seq_per_block * size_per_head;
        transposeMultiHeadToSingleKernel<T>
            <<<grid, block, 0, stream>>>(src, dst, batch_size, seq_len, head_num, size_per_head);
    }
}

template void invokeTransposeMultiHeadToSingle(float* dst,
                                               float* src,
                                               const int batch_size,
                                               const int seq_len,
                                               const int head_num,
                                               const int size_per_head,
                                               cudaStream_t stream);

template void invokeTransposeMultiHeadToSingle(half* dst,
                                               half* src,
                                               const int batch_size,
                                               const int seq_len,
                                               const int head_num,
                                               const int size_per_head,
                                               cudaStream_t stream);

}  // namespace fastertransformer