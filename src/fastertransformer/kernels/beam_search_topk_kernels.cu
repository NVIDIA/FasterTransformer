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

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

#include "src/fastertransformer/kernels/beam_search_topk_kernels.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"

namespace fastertransformer {
template<typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void beam_topK_kernel(
    const T* log_probs, int* topk_tmp_id_buf, T* topk_tmp_val_buf, const int vocab_size, T diversity_rate)
{
    typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;
    TopK<T, MAX_K> partial;

    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

#pragma unroll
    for (int i = 0; i < MAX_K; ++i) {
        partial.p[i] = -1;
        partial.u[i] = -MAX_T_VAL;
    }

#pragma unroll
    for (int elem_id = thread_id; elem_id < vocab_size; elem_id += THREADBLOCK_SIZE) {
        int index = elem_id + block_id * vocab_size;
        partial.insert(log_probs[index], index);
    }

    TopK<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

    if (thread_id == 0) {
        int index = block_id * MAX_K;

#pragma unroll
        for (int i = 0; i < MAX_K; ++i) {
            topk_tmp_id_buf[index + i] = total.p[i];
            topk_tmp_val_buf[index + i] = total.u[i] + diversity_rate * (T)i;
        }
    }
}

template<typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void batch_topK_kernel(int* topk_tmp_id_buf, T* topk_tmp_val_buf, int* id_buf)
{
    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
    TopK<T, MAX_K> partial;
    if (thread_id == 0) {
        for (int i = 0; i < MAX_K; ++i) {
            partial.p[i] = -1;
            partial.u[i] = -MAX_T_VAL;
        }

        int index = block_id * MAX_K * MAX_K;
        for (int i = 0; i < MAX_K * MAX_K; i++) {
            partial.insert((T)topk_tmp_val_buf[index + i], topk_tmp_id_buf[index + i]);
        }

        index = block_id * MAX_K;
        for (int i = 0; i < MAX_K; i++) {
            id_buf[index + i] = partial.p[i];
        }
    }
}

template<typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void batch_topK_kernel_v2(int* topk_tmp_id_buf, T* topk_tmp_val_buf, int* id_buf)
{
    typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    TopK<T, MAX_K> partial;
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

#pragma unroll
    for (int i = 0; i < MAX_K; ++i) {
        partial.p[i] = -1;
        partial.u[i] = -MAX_T_VAL;
    }

    int ite = MAX_K * MAX_K / THREADBLOCK_SIZE;
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int index = bid * MAX_K * MAX_K + i * THREADBLOCK_SIZE + tid;
        partial.insert((T)topk_tmp_val_buf[index], topk_tmp_id_buf[index]);
    }

    TopK<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

    if (tid == 0) {
#pragma unroll
        for (int i = 0; i < MAX_K; i++) {
            id_buf[bid * MAX_K + i] = total.p[i];
        }
    }
}

template<typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topk_stage_1_opt3(const T* __restrict log_probs,
                                  T* tmp_log_probs,
                                  int* topk_tmp_id_buf,
                                  T* topk_tmp_val_buf,
                                  const bool* finished,
                                  const int k,
                                  const int vocab_size,
                                  const int* end_ids)
{
    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int row_id = bid / BLOCKS_PER_BEAM_;      // row id for log_probs
    const int block_lane = bid % BLOCKS_PER_BEAM_;  // block id for a beam
    const int tmp_log_buf_index = row_id * vocab_size;
    const int tmp_topk_buf_index = row_id * BLOCKS_PER_BEAM_ * k + block_lane * k;
    TopK_2<T> partial;
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    if (finished != nullptr && finished[row_id] == true) {
        if (tid < k) {
            const int index = tmp_topk_buf_index + tid;
            if (block_lane == 0 && tid == 0) {
                const int end_id = end_ids[row_id / k];
                topk_tmp_id_buf[index] = tmp_log_buf_index + end_id;
                topk_tmp_val_buf[index] = log_probs[tmp_log_buf_index + end_id];
            }
            else {
                topk_tmp_id_buf[index] = -1;
                topk_tmp_val_buf[index] = -MAX_T_VAL;
            }
        }
        return;
    }

    for (int elem_id = tid + block_lane * BLOCK_SIZE_; elem_id < vocab_size;
         elem_id += BLOCK_SIZE_ * BLOCKS_PER_BEAM_) {
        int index = elem_id + tmp_log_buf_index;
        tmp_log_probs[index] = log_probs[index];
    }

    for (int ite = 0; ite < k; ite++) {
        partial.init();
#pragma unroll
        for (int elem_id = tid + block_lane * BLOCK_SIZE_; elem_id < vocab_size;
             elem_id += BLOCK_SIZE_ * BLOCKS_PER_BEAM_) {
            int index = elem_id + tmp_log_buf_index;
            partial.insert(tmp_log_probs[index], index);
        }

        TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

        if (tid == 0) {
            const int index = tmp_topk_buf_index + ite;
            topk_tmp_id_buf[index] = total.p;
            topk_tmp_val_buf[index] = total.u;
            tmp_log_probs[total.p] = -MAX_T_VAL;
        }
        __syncthreads();
    }
}

template<typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topk_stage_2_opt3(const int* __restrict topk_tmp_id_buf, T* topk_tmp_val_buf, int* ids, const int k)
{
    const int size = k * k * BLOCKS_PER_BEAM_;
    const int tid = threadIdx.x;
    const int batch_id = blockIdx.x;
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    extern __shared__ char array[];
    T* s_val = topk_tmp_val_buf + batch_id * size;
    int* s_id = (int*)(array);

    TopK_2<T> partial;

    for (int ite = 0; ite < k; ite++) {
        partial.init();
#pragma unroll
        for (int i = tid; i < size; i += BLOCK_SIZE_) {
            partial.insert(s_val[i], i);
        }

        TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

        if (tid == 0) {
            s_id[ite] = total.p;
            s_val[total.p] = -MAX_T_VAL;
        }
        __syncthreads();
    }
    if (tid < k) {
        ids[batch_id * k + tid] = topk_tmp_id_buf[batch_id * size + s_id[tid]];
    }
}

template<typename T, int BLOCK_SIZE, int BLOCKS_PER_BEAM>
__global__ void topk_stage_1_opt2_general(const T* __restrict log_probs,
                                          T* tmp_log_probs,
                                          int* topk_tmp_id_buf,
                                          T* topk_tmp_val_buf,
                                          const int k,
                                          const int vocab_size)
{
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int row_id = bid / BLOCKS_PER_BEAM;      // row id for log_probs
    const int block_lane = bid % BLOCKS_PER_BEAM;  // block id for a beam
    const int tmp_log_buf_index = row_id * vocab_size;
    const int tmp_topk_buf_index = row_id * BLOCKS_PER_BEAM * k + block_lane * k;
    TopK_2<T> partial;

    for (int elem_id = tid + block_lane * BLOCK_SIZE; elem_id < vocab_size; elem_id += BLOCK_SIZE * BLOCKS_PER_BEAM) {
        int index = elem_id + tmp_log_buf_index;
        tmp_log_probs[index] = log_probs[index];
    }

    for (int ite = 0; ite < k; ite++) {
        partial.init();
#pragma unroll
        for (int elem_id = tid + block_lane * BLOCK_SIZE; elem_id < vocab_size;
             elem_id += BLOCK_SIZE * BLOCKS_PER_BEAM) {
            int index = elem_id + tmp_log_buf_index;
            partial.insert(tmp_log_probs[index], index);
        }

        TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

        if (tid == 0) {
            const int index = tmp_topk_buf_index + ite;
            topk_tmp_id_buf[index] = total.p;
            topk_tmp_val_buf[index] = total.u;
            tmp_log_probs[total.p] = -MAX_T_VAL;
        }
        __syncthreads();
    }
}

template<typename T, int BLOCK_SIZE, int BLOCKS_PER_BEAM>
__global__ void
topk_stage_2_opt2_general(const int* __restrict topk_tmp_id_buf, T* topk_tmp_val_buf, int* ids, const int k)
{
    const int size = k * k * BLOCKS_PER_BEAM;
    const int tid = threadIdx.x;
    const int batch_id = blockIdx.x;
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    extern __shared__ char array[];
    T* s_val = topk_tmp_val_buf + batch_id * size;
    int* s_id = (int*)(array);

    TopK_2<T> partial;

    for (int ite = 0; ite < k; ite++) {
        partial.init();
#pragma unroll
        for (int i = tid; i < size; i += BLOCK_SIZE) {
            partial.insert(s_val[i], i);
        }

        TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

        if (tid == 0) {
            s_id[ite] = total.p;
            s_val[total.p] = -MAX_T_VAL;
        }
        __syncthreads();
    }
    if (tid < k) {
        ids[batch_id * k + tid] = topk_tmp_id_buf[batch_id * size + s_id[tid]];
    }
}

#define CASE_K_DIV(K, BLOCK_SIZE_1, BLOCK_SIZE_2)                                                                      \
    case K:                                                                                                            \
        beam_topK_kernel<T, K, BLOCK_SIZE_2><<<batch_size * beam_width, BLOCK_SIZE_2, 0, stream>>>(                    \
            log_probs, topk_tmp_id_buf, topk_tmp_val_buf, vocab_size, diversity_rate);                                 \
        if (K < 10)                                                                                                    \
            batch_topK_kernel<T, K, BLOCK_SIZE_1>                                                                      \
                <<<batch_size, BLOCK_SIZE_1, 0, stream>>>(topk_tmp_id_buf, topk_tmp_val_buf, ids);                     \
        else                                                                                                           \
            batch_topK_kernel_v2<T, K, 32><<<batch_size, 32, 0, stream>>>(topk_tmp_id_buf, topk_tmp_val_buf, ids);     \
        break;

#define CASE_K(K, BLOCK_SIZE_1_, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_)                                                      \
    case K:                                                                                                            \
        topk_stage_1_opt3<float, BLOCK_SIZE_1_, BLOCKS_PER_BEAM_>                                                      \
            <<<batch_size * K * BLOCKS_PER_BEAM_, BLOCK_SIZE_1_, 0, stream>>>(log_probs,                               \
                                                                              temp_log_probs,                          \
                                                                              topk_tmp_id_buf,                         \
                                                                              topk_tmp_val_buf,                        \
                                                                              finished,                                \
                                                                              beam_width,                              \
                                                                              vocab_size,                              \
                                                                              end_ids);                                \
        topk_stage_2_opt3<float, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_>                                                      \
            <<<batch_size, BLOCK_SIZE_2_, K * sizeof(int), stream>>>(                                                  \
                topk_tmp_id_buf, topk_tmp_val_buf, ids, beam_width);                                                   \
        break;

template<typename T>
void invokeTopkBeamSearch(void* workspace,
                          size_t& workspace_size,
                          T* log_probs,
                          int* ids,
                          const bool* finished,
                          const int batch_size,
                          const int beam_width,
                          const int vocab_size_padded_,
                          const T diversity_rate,
                          const int* end_ids,
                          cudaStream_t stream)
{
    const int vocab_size = vocab_size_padded_;

    const int max_block_per_beam = 8;
    int temp_log_probs_buf_size = batch_size * beam_width * vocab_size;                     // type float
    int topk_tmp_ids_buf_size = batch_size * beam_width * beam_width * max_block_per_beam;  // type int
    int topk_tmp_val_buf_size = batch_size * beam_width * beam_width * max_block_per_beam;  // type float

    // prevent memory misalinged address
    temp_log_probs_buf_size = (int)(ceil(temp_log_probs_buf_size / 4.)) * 4;
    topk_tmp_ids_buf_size = (int)(ceil(topk_tmp_ids_buf_size / 4.)) * 4;
    topk_tmp_val_buf_size = (int)(ceil(topk_tmp_val_buf_size / 4.)) * 4;

    if (workspace == nullptr) {
        workspace_size = sizeof(float) * temp_log_probs_buf_size + sizeof(int) * topk_tmp_ids_buf_size
                         + sizeof(float) * topk_tmp_val_buf_size;
        return;
    }
    else {
        T* temp_log_probs = (T*)workspace;
        int* topk_tmp_id_buf = (int*)(temp_log_probs + temp_log_probs_buf_size);
        T* topk_tmp_val_buf = (T*)(topk_tmp_id_buf + topk_tmp_ids_buf_size);
        if (diversity_rate == 0.0f) {
            switch (beam_width) {
                CASE_K(1, 128, 128, 8);
                CASE_K(4, 128, 128, 8);
                CASE_K(10, 128, 128, 8);
                CASE_K(16, 128, 128, 5);
                CASE_K(32, 256, 128, 1);
                CASE_K(64, 256, 256, 1);
                default:
                    topk_stage_1_opt2_general<T, 128, 1><<<batch_size * beam_width * 1, 128, 0, stream>>>(
                        log_probs, temp_log_probs, topk_tmp_id_buf, topk_tmp_val_buf, beam_width, vocab_size);
                    topk_stage_2_opt2_general<T, 128, 1>
                        <<<batch_size,
                           128,
                           beam_width * beam_width * 1 * sizeof(float) + beam_width * sizeof(int),
                           stream>>>(topk_tmp_id_buf, topk_tmp_val_buf, ids, beam_width);
                    break;
            }
        }
        else {
            switch (beam_width) {
                CASE_K_DIV(1, 256, 256);
                CASE_K_DIV(4, 256, 256);
                CASE_K_DIV(16, 256, 64);
                CASE_K_DIV(64, 256, 64);
                default:
                    printf("[ERROR] Topk kernel does not support beamwidth = %d \n", beam_width);
                    exit(0);
                    break;
            }
        }
        return;
    }
}

#undef CASE_K
#undef CASE_K_DIV

template void invokeTopkBeamSearch(void* workspace,
                                   size_t& workspace_size,
                                   float* log_probs,
                                   int* ids,
                                   const bool* finished,
                                   const int batch_size,
                                   const int beam_width,
                                   const int vocab_size_padded_,
                                   const float diversity_rate,
                                   const int* end_ids,
                                   cudaStream_t stream);

template<typename T>
__global__ void tileEncoderResults(T* tiled_output,
                                   int* tiled_sequence_length,
                                   const T* output,
                                   const int* sequence_length,
                                   const uint batch_size,
                                   const uint beam_width,
                                   const uint d_model)
{
    if (blockIdx.x == 0) {
        for (uint i = threadIdx.x; i < batch_size * beam_width; i += blockDim.x) {
            tiled_sequence_length[i] = sequence_length[i / beam_width];
        }
    }

    int tgt_offset =
        blockIdx.x * gridDim.y * gridDim.z * d_model + blockIdx.y * gridDim.z * d_model + blockIdx.z * d_model;
    int src_offset = blockIdx.x * gridDim.z * d_model + blockIdx.z * d_model;
    for (uint i = threadIdx.x; i < d_model; i += blockDim.x) {
        tiled_output[i + tgt_offset] = output[i + src_offset];
    }
}

template<typename T>
void invokeTileEncoderResults(T* tiled_output,
                              int* tiled_sequence_length,
                              const T* output,
                              const int* sequence_length,
                              const size_t batch_size,
                              const size_t beam_width,
                              const size_t mem_max_seq_len,
                              const size_t d_model,
                              cudaStream_t stream)
{
    // tiled_output: [batch_size, beam_width, mem_max_seq_len, d_model]
    // tiled_sequence_length: [batch_size, beam_width]

    // output: [batch_size, mem_max_seq_len, d_model]
    // sequence_length [batch_size]

    dim3 grid(batch_size, beam_width, mem_max_seq_len);

    if (d_model % 2 == 0 && std::is_same<T, half>::value) {
        dim3 block(min(512, (int)(d_model / 2)));
        tileEncoderResults<half2><<<grid, block, 0, stream>>>((half2*)tiled_output,
                                                              tiled_sequence_length,
                                                              (const half2*)output,
                                                              sequence_length,
                                                              batch_size,
                                                              beam_width,
                                                              d_model / 2);
    }
    else {
        dim3 block(min(512, (int)d_model));
        tileEncoderResults<T><<<grid, block, 0, stream>>>(
            tiled_output, tiled_sequence_length, output, sequence_length, batch_size, beam_width, d_model);
    }
}

template void invokeTileEncoderResults(float* tiled_output,
                                       int* tiled_sequence_length,
                                       const float* output,
                                       const int* sequence_length,
                                       const size_t batch_size,
                                       const size_t beam_width,
                                       const size_t mem_max_seq_len,
                                       const size_t d_model,
                                       cudaStream_t stream);

template void invokeTileEncoderResults(half* tiled_output,
                                       int* tiled_sequence_length,
                                       const half* output,
                                       const int* sequence_length,
                                       const size_t batch_size,
                                       const size_t beam_width,
                                       const size_t mem_max_seq_len,
                                       const size_t d_model,
                                       cudaStream_t stream);

template void invokeTileEncoderResults(half2* tiled_output,
                                       int* tiled_sequence_length,
                                       const half2* output,
                                       const int* sequence_length,
                                       const size_t batch_size,
                                       const size_t beam_width,
                                       const size_t mem_max_seq_len,
                                       const size_t d_model,
                                       cudaStream_t stream);
}  // namespace fastertransformer