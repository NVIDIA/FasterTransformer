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
#include <cmath>

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

#include "longformer_kernels.h"
#include "src/fastertransformer/utils/cuda_utils.h"

#include <stdio.h>

namespace fastertransformer {

__global__ void initSeqIdxKernel(int* seq_idx, int seq_len)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = offset; i < seq_len; i += blockDim.x) {
        seq_idx[i] = i;
    }
}

template<typename T>
size_t getInitLongformerCubStorage(const int seq_len)
{
    size_t tmp_storage_bytes = 0;
    int *seq_idx = NULL, *global_idx = NULL, *global_token_nums = NULL;
    void* global_attn_mask = NULL;

    check_cuda_error(cub::DevicePartition::Flagged(
        NULL, tmp_storage_bytes, seq_idx, (T*)global_attn_mask, global_idx, global_token_nums, seq_len));

    return tmp_storage_bytes;
}

template<typename T>
__global__ void localAttnMaskShiftKernel(T* local_attn_mask, T* out, int thread_block_repeat, int total_len)
{
    int i = blockIdx.x * blockDim.x * thread_block_repeat + threadIdx.x;
    int end = i + thread_block_repeat * blockDim.x;
    for (; i < end && i < total_len; i += blockDim.x) {
        out[i] = local_attn_mask[i] * (T)10000 - (T)10000;
    }
}

template<typename T>
void invokeLocalAttnMaskShift(T* local_attn_mask, T* out, int batch_size, int seq_len, cudaStream_t stream)
{
    const int thread_block_repeat = 4;
    const int block_dim = 128;
    int block_num = std::ceil(batch_size * seq_len / (float)block_dim / (float)thread_block_repeat);
    localAttnMaskShiftKernel<<<block_num, block_dim, 0, stream>>>(
        local_attn_mask, out, thread_block_repeat, batch_size * seq_len);
}

template void
invokeLocalAttnMaskShift(float* local_attn_mask, float* out, int batch_size, int seq_len, cudaStream_t stream);

template void
invokeLocalAttnMaskShift(half* local_attn_mask, half* out, int batch_size, int seq_len, cudaStream_t stream);

template<typename T>
void invokeInitLongformerIdx(T* global_attn_mask,
                             int* seq_idx,
                             int* global_idx,
                             int* global_token_nums,
                             int seq_len,
                             int batch_size,
                             void* cub_storage,
                             cudaStream_t stream)
{
    const int threads = 1024;
    int blocks = std::ceil(seq_len / float(threads));
    initSeqIdxKernel<<<blocks, threads, 0, stream>>>(seq_idx, seq_len);
    sync_check_cuda_error();

    size_t storages_bytes = getInitLongformerCubStorage<T>(seq_len);
    for (int i = 0; i < batch_size; ++i) {
        check_cuda_error(cub::DevicePartition::Flagged(cub_storage,
                                                       storages_bytes,
                                                       seq_idx,
                                                       global_attn_mask + i * seq_len,
                                                       global_idx + i * seq_len,
                                                       global_token_nums + i,
                                                       seq_len,
                                                       stream));
    }
}

template void invokeInitLongformerIdx(float* global_attn_mask,
                                      int* seq_idx,
                                      int* global_idx,
                                      int* global_token_nums,
                                      int seq_len,
                                      int batch_size,
                                      void* cub_storage,
                                      cudaStream_t stream);

template void invokeInitLongformerIdx(half* global_attn_mask,
                                      int* seq_idx,
                                      int* global_idx,
                                      int* global_token_nums,
                                      int seq_len,
                                      int batch_size,
                                      void* cub_storage,
                                      cudaStream_t stream);

// Apply softmax to local and global attention. Rewrite the result to the same buffer in-place
template<typename T, int blockSize>
__launch_bounds__(blockSize) __global__ void longformerMHASoftmaxKernel(const T* global_attn,
                                                                        const int* global_idx,
                                                                        const int* global_token_nums,
                                                                        void* input_ptrs,
                                                                        const T* attn_mask,
                                                                        float scaler,
                                                                        int seq_len,
                                                                        int head_num,
                                                                        int attn_window_size)
{
    typedef cub::BlockReduce<float, blockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage breduce_temp;

    size_t* p_inputs = (size_t*)(input_ptrs);
    // use input buffer as output buffer
    size_t* p_outputs = (size_t*)(input_ptrs);
    size_t* input_sizes = (size_t*)(input_ptrs) + 5;
    size_t* input_strides = (size_t*)(input_ptrs) + 10;

    int tid = threadIdx.x;
    const int batch_idx = blockIdx.x / (seq_len * head_num);
    const int row_idx = blockIdx.x % seq_len;
    const int head_idx = (blockIdx.x / seq_len) % head_num;

    // adjust the pointers for the batch
    const T* mask_blk = attn_mask + seq_len * batch_idx;
    const int global_num = global_token_nums[batch_idx];
    const int* global_idx_blk = global_idx + seq_len * batch_idx;

    T* inputs[5];
    T* outputs[5];
    for (int i = 0; i < 5; ++i) {
        inputs[i] = (T*)p_inputs[i] + batch_idx * head_num * input_sizes[i];
        outputs[i] = (T*)p_outputs[i] + batch_idx * head_num * input_sizes[i];
    }

    int col_start = 0;
    int col_end = seq_len;

    // is it local attention token
    int is_local_row = global_attn[row_idx + seq_len * batch_idx] == (T)0.f;

    // if local token
    if (is_local_row) {
        col_start = row_idx - attn_window_size;
        col_end = row_idx + attn_window_size + 1;
    }

    if (col_start < 0) {
        col_start = 0;
    }
    if (col_end > seq_len) {
        col_end = seq_len;
    }

    // if mask is set then set everything to zero to match Python implementation
    if (mask_blk[row_idx] != (T)0.f) {
        if (is_local_row) {
            T* output_blk = nullptr;
            T* output_glb = nullptr;
            int local_offset = row_idx % attn_window_size;
            int local_start = 0;
            int local_end = 3 * attn_window_size;
            if (row_idx < attn_window_size) {
                local_start = 0;
                local_end = 2 * attn_window_size;
                output_blk = outputs[0] + row_idx * input_strides[0] + head_idx * input_sizes[0];
            }
            else if (row_idx < seq_len - attn_window_size) {
                output_blk = outputs[1] + (row_idx - attn_window_size) * input_strides[1] + head_idx * input_sizes[1];
            }
            else {
                local_start = 0;
                local_end = 2 * attn_window_size;
                output_blk = outputs[2] + local_offset * input_strides[2] + head_idx * input_sizes[2];
            }

            for (int i = local_start + tid; i < local_end; i += blockSize) {
                output_blk[i] = 0;
            }

            if ((row_idx - 2 * attn_window_size) >= 0) {
                output_glb = outputs[3] + (row_idx - attn_window_size) * input_strides[3] + head_idx * input_sizes[3];
            }

            if (output_glb != nullptr) {
                for (int i = tid; i < global_num; i += blockSize) {
                    output_glb[i] = 0;
                }
            }
        }
        else {
            T* output_blk = outputs[4];
            for (int i = tid; i < seq_len; i += blockSize) {
                output_blk[i] = 0;
            }
        }
        return;
    }

    __shared__ float sum_shared;
    float sum_input = 0.;
    // calculate max input
    float max_input = -FLT_MAX;
    __shared__ float max_shared;

    if (is_local_row) {
        const T* input_blk = nullptr;
        T* output_blk = nullptr;
        T* output_glb = nullptr;
        int local_offset = row_idx % attn_window_size;
        int local_start = local_offset;
        int local_end = local_start + 2 * attn_window_size + 1;
        int zero_start = 0;
        int zero_end = 3 * attn_window_size;
        if (row_idx < attn_window_size) {
            local_start = 0;
            local_end = local_offset + attn_window_size + 1;
            zero_end = 2 * attn_window_size;

            input_blk = inputs[0] + row_idx * input_strides[0] + head_idx * input_sizes[0];
            output_blk = outputs[0] + row_idx * input_strides[0] + head_idx * input_sizes[0];
        }
        else if (row_idx < seq_len - attn_window_size) {
            input_blk = inputs[1] + (row_idx - attn_window_size) * input_strides[1] + head_idx * input_sizes[1];
            output_blk = outputs[1] + (row_idx - attn_window_size) * input_strides[1] + head_idx * input_sizes[1];
        }
        else {
            local_start = local_offset;
            local_end = 2 * attn_window_size;
            zero_end = 2 * attn_window_size;

            input_blk = inputs[2] + local_offset * input_strides[2] + head_idx * input_sizes[2];
            output_blk = outputs[2] + local_offset * input_strides[2] + head_idx * input_sizes[2];
        }

        const T* input_glb = nullptr;
        int local_global = row_idx - attn_window_size;
        if (local_global > global_num) {
            local_global = global_num;
        }
        if (local_global > 0) {
            input_glb = inputs[3] + (row_idx - attn_window_size) * input_strides[3] + head_idx * input_sizes[3];
        }

        if (row_idx < attn_window_size) {
            output_glb = (T*)outputs[0] + row_idx * input_strides[0] + head_idx * input_sizes[0];
        }
        else if (row_idx < 2 * attn_window_size) {
            output_glb = outputs[1] + (row_idx - attn_window_size) * input_strides[1] + head_idx * input_sizes[1];
        }
        else {
            output_glb = outputs[3] + (row_idx - attn_window_size) * input_strides[3] + head_idx * input_sizes[3];
        }

        for (int i = local_start + tid, ii = col_start + tid; i < local_end; i += blockSize, ii += blockSize) {
            float x = (float)input_blk[i];
            x = x * scaler + (float)mask_blk[ii];
            if (max_input < x) {
                max_input = x;
            }
        }

        if (input_glb != nullptr) {
            for (int i = tid; i < local_global; i += blockSize) {
                float x = (float)input_glb[global_idx_blk[i]];
                x = x * scaler + (float)mask_blk[global_idx_blk[i]];
                if (max_input < x) {
                    max_input = x;
                }
            }
        }

        float max_blk = BlockReduce(breduce_temp).Reduce(max_input, cub::Max());
        if (tid == 0) {
            max_shared = max_blk;
        }
        __syncthreads();

        for (int i = local_start + tid, ii = col_start + tid; i < local_end; i += blockSize, ii += blockSize) {
            float x = (float)input_blk[i];
            x = expf(x * scaler + (float)mask_blk[ii] - max_shared);
            sum_input += x;
        }

        if (input_glb != nullptr) {
            for (int i = tid, ii = col_start + tid; i < local_global; i += blockSize, ii += blockSize) {
                float x = (float)input_glb[global_idx_blk[i]];
                x = expf(x * scaler + (float)mask_blk[ii] - max_shared);
                sum_input += x;
            }
        }

        float sum_blk = BlockReduce(breduce_temp).Reduce(sum_input, cub::Sum());
        if (tid == 0) {
            sum_shared = sum_blk;
        }
        __syncthreads();
        float recip_sum = 1.f / sum_shared;

        for (int i = tid + zero_start; i < local_start; i += blockSize) {
            output_blk[i] = (T)(0.);
        }

        for (int i = tid + local_end; i < zero_end; i += blockSize) {
            output_blk[i] = (T)(0.);
        }

        __syncthreads();

        for (int i = local_start + tid, ii = col_start + tid; i < local_end; i += blockSize, ii += blockSize) {
            float x = (float)input_blk[i];
            x = expf(x * scaler + (float)mask_blk[ii] - max_shared);
            output_blk[i] = (T)(recip_sum * x);
        }

        if (input_glb != nullptr) {
            for (int i = tid; i < local_global; i += blockSize) {
                float x = (float)input_glb[global_idx_blk[i]];
                x = expf(x * scaler + (float)mask_blk[global_idx_blk[i]] - max_shared);
                output_glb[i] = (T)(recip_sum * x);
            }
        }
    }
    else {
        // global tokens
        const T* input_blk = inputs[4] + row_idx * input_strides[4] + head_idx * input_sizes[4];
        T* output_blk = outputs[4] + row_idx * input_strides[4] + head_idx * input_sizes[4];

        for (int i = tid; i < seq_len; i += blockSize) {
            float x = (float)input_blk[i];
            x = x * scaler + (float)mask_blk[i];
            if (max_input < x) {
                max_input = x;
            }
        }

        float max_blk = BlockReduce(breduce_temp).Reduce(max_input, cub::Max());
        if (tid == 0) {
            max_shared = max_blk;
        }
        __syncthreads();

        for (int i = tid; i < seq_len; i += blockSize) {
            float x = (float)input_blk[i];
            x = expf(x * scaler + (float)mask_blk[i] - max_shared);
            sum_input += x;
        }

        float sum_blk = BlockReduce(breduce_temp).Reduce(sum_input, cub::Sum());
        if (tid == 0) {
            sum_shared = sum_blk;
        }
        __syncthreads();
        float recip_sum = 1.f / sum_shared;

        for (int i = tid; i < seq_len; i += blockSize) {
            float x = (float)input_blk[i];
            x = expf(x * scaler + (float)mask_blk[i] - max_shared);
            output_blk[i] = (T)(recip_sum * x);
        }
    }
}

template<typename T>
void invokeLongformerMHASoftmax(const T* global_attn_mask,
                                const int* global_idx,
                                const int* global_token_nums,
                                void* input_ptrs,
                                const T* local_attn_mask,
                                float scaler,
                                int seq_len,
                                int head_num,
                                int batch_size,
                                int local_attn_window_size,
                                cudaStream_t stream)
{
    const int block_size = 64;
    const int grid_size = seq_len * head_num * batch_size;
    longformerMHASoftmaxKernel<T, block_size><<<grid_size, block_size, 0, stream>>>(global_attn_mask,
                                                                                    global_idx,
                                                                                    global_token_nums,
                                                                                    input_ptrs,
                                                                                    local_attn_mask,
                                                                                    scaler,
                                                                                    seq_len,
                                                                                    head_num,
                                                                                    local_attn_window_size);
}

template void invokeLongformerMHASoftmax(const float* global_attn_mask,
                                         const int* global_idx,
                                         const int* global_token_nums,
                                         void* input_ptrs,
                                         const float* local_attn_mask,
                                         float scaler,
                                         int seq_len,
                                         int head_num,
                                         int batch_size,
                                         int local_attn_window_size,
                                         cudaStream_t stream);

template void invokeLongformerMHASoftmax(const half* global_attn_mask,
                                         const int* global_idx,
                                         const int* global_token_nums,
                                         void* input_ptrs,
                                         const half* local_attn_mask,
                                         float scaler,
                                         int seq_len,
                                         int head_num,
                                         int batch_size,
                                         int local_attn_window_size,
                                         cudaStream_t stream);

}  // namespace fastertransformer