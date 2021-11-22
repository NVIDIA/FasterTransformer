/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include "src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"
#include "src/fastertransformer/utils/cuda_utils.h"

namespace fastertransformer {

__inline__ __device__ int target_index(int id1, int id2, int id3, int id4, int dim_1, int dim_2, int dim_3, int dim_4)
{
    return id1 * (dim_2 * dim_3 * dim_4) + id3 * (dim_2 * dim_4) + id2 * dim_4 + id4;
}

template<typename T>
__global__ void addQKVBiasTranspose(T* q_out,
                                    T* k_out,
                                    T* v_out,
                                    const T* __restrict q_in,
                                    const T* __restrict bias_q,
                                    const T* __restrict k_in,
                                    const T* __restrict bias_k,
                                    const T* __restrict v_in,
                                    const T* __restrict bias_v,
                                    const int batch_size,
                                    const int seq_len,
                                    const int head_num,
                                    const int size_per_head)
{
    const int n = head_num * size_per_head;
    const int batch_id = blockIdx.x;
    const int word_id = blockIdx.y;
    const int row_id = batch_id * seq_len + word_id;
    for (int col_id = threadIdx.x; col_id < n; col_id += blockDim.x) {
        const int head_id = col_id / size_per_head;
        const int size_id = col_id % size_per_head;
        const int target_id = batch_id * (head_num * seq_len * size_per_head) + head_id * seq_len * size_per_head
                              + word_id * size_per_head + size_id;
        const int src_id = row_id * n + col_id;

        q_out[target_id] = __ldg(&q_in[src_id]);
        q_out[target_id] = q_out[target_id] + __ldg(&bias_q[col_id]);

        k_out[target_id] = __ldg(&k_in[src_id]);
        k_out[target_id] = k_out[target_id] + __ldg(&bias_k[col_id]);

        v_out[target_id] = __ldg(&v_in[src_id]);
        v_out[target_id] = v_out[target_id] + __ldg(&bias_v[col_id]);
    }
}

template<typename T>
__global__ void QKVTranspose(T* q_out,
                             T* k_out,
                             T* v_out,
                             const T* __restrict q_in,
                             const T* __restrict k_in,
                             const T* __restrict v_in,
                             const int batch_size,
                             const int seq_len,
                             const int head_num,
                             const int size_per_head)
{
    const int n = head_num * size_per_head;
    const int batch_id = blockIdx.x;
    const int word_id = blockIdx.y;
    const int row_id = batch_id * seq_len + word_id;
    for (int col_id = threadIdx.x; col_id < n; col_id += blockDim.x) {
        const int head_id = col_id / size_per_head;
        const int size_id = col_id % size_per_head;
        const int target_id = batch_id * (head_num * seq_len * size_per_head) + head_id * seq_len * size_per_head
                              + word_id * size_per_head + size_id;
        const int src_id = row_id * n + col_id;

        q_out[target_id] = __ldg(&q_in[src_id]);
        k_out[target_id] = __ldg(&k_in[src_id]);
        v_out[target_id] = __ldg(&v_in[src_id]);
    }
}

template<typename T>
void invokeAddQKVBiasTranspose(T* q_buf,
                               T* k_buf,
                               T* v_buf,
                               T* Q,
                               const T* bias_Q,
                               T* K,
                               const T* bias_K,
                               T* V,
                               const T* bias_V,
                               const int batch_size,
                               const int seq_len,
                               const int head_num,
                               const int size_per_head,
                               cudaStream_t stream)
{
    const int k = head_num * size_per_head;
    dim3 grid(batch_size, seq_len);
    bool is_add_bias = bias_Q != nullptr;
    if (sizeof(T) == 4 || k % 2 != 0) {
        dim3 block(min(k, 512));
        if (is_add_bias) {
            addQKVBiasTranspose<T><<<grid, block, 0, stream>>>(
                q_buf, k_buf, v_buf, Q, bias_Q, K, bias_K, V, bias_V, batch_size, seq_len, head_num, size_per_head);
        }
        else {
            QKVTranspose<T><<<grid, block, 0, stream>>>(
                q_buf, k_buf, v_buf, Q, K, V, batch_size, seq_len, head_num, size_per_head);
        }
        sync_check_cuda_error();
    }
    else {
        dim3 block(min(k / 2, 512));
        if (is_add_bias) {
            addQKVBiasTranspose<half2><<<grid, block, 0, stream>>>((half2*)q_buf,
                                                                   (half2*)k_buf,
                                                                   (half2*)v_buf,
                                                                   (const half2*)Q,
                                                                   (const half2*)bias_Q,
                                                                   (const half2*)K,
                                                                   (const half2*)bias_K,
                                                                   (const half2*)V,
                                                                   (const half2*)bias_V,
                                                                   batch_size,
                                                                   seq_len,
                                                                   head_num,
                                                                   size_per_head / 2);
        }
        else {
            QKVTranspose<half2><<<grid, block, 0, stream>>>((half2*)q_buf,
                                                            (half2*)k_buf,
                                                            (half2*)v_buf,
                                                            (const half2*)Q,
                                                            (const half2*)K,
                                                            (const half2*)V,
                                                            batch_size,
                                                            seq_len,
                                                            head_num,
                                                            size_per_head / 2);
        }
        sync_check_cuda_error();
    }
}

template void invokeAddQKVBiasTranspose(float* q_buf,
                                        float* k_buf,
                                        float* v_buf,
                                        float* Q,
                                        const float* bias_Q,
                                        float* K,
                                        const float* bias_K,
                                        float* V,
                                        const float* bias_V,
                                        const int batch_size,
                                        const int seq_len,
                                        const int head_num,
                                        const int size_per_head,
                                        cudaStream_t stream);

template void invokeAddQKVBiasTranspose(half* q_buf,
                                        half* k_buf,
                                        half* v_buf,
                                        half* Q,
                                        const half* bias_Q,
                                        half* K,
                                        const half* bias_K,
                                        half* V,
                                        const half* bias_V,
                                        const int batch_size,
                                        const int seq_len,
                                        const int head_num,
                                        const int size_per_head,
                                        cudaStream_t stream);

// TODO(bhsueh) Rename the softmax_kernel_v4 to softmax_kernel
template<int ITEMS_PER_THREAD, typename T, typename T_IN>
__global__ void softmax_kernel_v4(T* qk_buf_,
                                  const T_IN* qk_buf_src,
                                  const T* attr_mask,
                                  const int batch_size,
                                  const int head_num,
                                  const int seq_len,
                                  const T scalar)
{
    for (int seq_id = blockIdx.x; seq_id < seq_len; seq_id += gridDim.x) {
        float data[ITEMS_PER_THREAD];
        int qk_offset;
        __shared__ float s_mean, s_max;
        float local_max = -1e20f;
        for (int i = 0; blockDim.x * i + threadIdx.x < seq_len; i++) {
            qk_offset =
                ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id) * seq_len + blockDim.x * i + threadIdx.x;
            int mask_offset = (blockIdx.y * seq_len + seq_id) * seq_len + blockDim.x * i + threadIdx.x;

            float qk = static_cast<float>(qk_buf_src[qk_offset]);
            float mask_val = static_cast<float>(__ldg(&attr_mask[mask_offset]));

            mask_val = (1.0f - mask_val) * -10000.0f;

            data[i] = qk * static_cast<float>(scalar) + mask_val;
            local_max = fmax(local_max, data[i]);
        }

        float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float local_sum = 0;
        for (int i = 0; blockDim.x * i + threadIdx.x < seq_len; i++) {
            data[i] = __expf(data[i] - s_max);
            local_sum += data[i];
        }
        float sum_val = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);
        if (threadIdx.x == 0) {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < seq_len; i++) {
            qk_offset =
                ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id) * seq_len + blockDim.x * i + threadIdx.x;
            qk_buf_[qk_offset] = (T)(data[i] * s_mean);
        }
    }
}

template<int ITEMS_PER_THREAD>
__global__ void softmax_kernel_v4_half2(half* qk_buf_,
                                        const half* attr_mask,
                                        const int batch_size,
                                        const int head_num,
                                        const int seq_len,
                                        const half scalar)
{
    half2* qk_buf_half2 = (half2*)qk_buf_;
    const half2* attr_mask_half2 = (const half2*)attr_mask;

    for (int seq_id = blockIdx.x; seq_id < seq_len; seq_id += gridDim.x) {
        half2 data[ITEMS_PER_THREAD];
        int qk_offset;
        __shared__ float s_mean, s_max;
        float local_max = -1e20f;
        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
            qk_offset = ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id) * (seq_len / 2) + blockDim.x * i
                        + threadIdx.x;
            int mask_offset = (blockIdx.y * seq_len + seq_id) * (seq_len / 2) + blockDim.x * i + threadIdx.x;

            half2 qk = qk_buf_half2[qk_offset];
            half2 mask_val = __ldg(&attr_mask_half2[mask_offset]);
            mask_val = __hmul2(__hsub2(__float2half2_rn(1.0f), mask_val), __float2half2_rn(-10000.0f));

            data[i] = __hadd2(__hmul2(qk, __half2half2(scalar)), mask_val);

            local_max = fmax(local_max, fmax((float)data[i].x, (float)data[i].y));
        }

        float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float local_sum = 0;
        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
            data[i] = h2exp(__hsub2(data[i], __float2half2_rn(s_max)));
            local_sum += (float)(data[i].x + data[i].y);
        }

        float sum_val = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);

        if (threadIdx.x == 0) {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
            qk_offset = ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id) * (seq_len / 2) + blockDim.x * i
                        + threadIdx.x;
            qk_buf_half2[qk_offset] = __hmul2(data[i], __float2half2_rn(s_mean));
        }
    }
}

template<int ITEMS_PER_THREAD, int NUM>
__global__ void softmax_kernel_v5_half2(half* qk_buf_,
                                        const half* attr_mask,
                                        const int batch_size,
                                        const int head_num,
                                        const int seq_len,
                                        const half scalar)
{
    half2* qk_buf_half2 = (half2*)qk_buf_;
    const half2* attr_mask_half2 = (const half2*)attr_mask;

    for (int seq_id = blockIdx.x; seq_id < seq_len; seq_id += gridDim.x * NUM) {
        half2 data[NUM][ITEMS_PER_THREAD];

        int qk_offset[NUM];

        __shared__ float s_sum[NUM], s_max[NUM];
        float local_max[NUM];
#pragma unroll
        for (int j = 0; j < NUM; j++) {
            local_max[j] = -1e20f;
        }

        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
            int mask_offset[NUM];
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                qk_offset[j] = ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id + j * gridDim.x) * (seq_len / 2)
                               + blockDim.x * i + threadIdx.x;
                mask_offset[j] =
                    (blockIdx.y * seq_len + seq_id + j * gridDim.x) * (seq_len / 2) + blockDim.x * i + threadIdx.x;
            }

            half2 mask_val[NUM];
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                mask_val[j] = __ldg(&attr_mask_half2[mask_offset[j]]);
            }

            half2 qk[NUM];
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                qk[j] = qk_buf_half2[qk_offset[j]];
            }

#pragma unroll
            for (int j = 0; j < NUM; j++) {
                mask_val[j] = __hmul2(__hsub2(__float2half2_rn(1.0f), mask_val[j]), __float2half2_rn(-10000.0f));
            }

#pragma unroll
            for (int j = 0; j < NUM; j++) {
                data[j][i] = __hadd2(__hmul2(qk[j], __half2half2(scalar)), mask_val[j]);
                local_max[j] = fmax(local_max[j], fmax((float)data[j][i].x, (float)data[j][i].y));
            }
        }

        if (blockDim.x <= 32) {
            warpReduceMaxV2<float, NUM>(local_max);
        }
        else {
            blockReduceMaxV2<float, NUM>(local_max);
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                s_max[j] = local_max[j];
            }
        }
        __syncthreads();

        float local_sum[NUM];
#pragma unroll
        for (int j = 0; j < NUM; j++) {
            local_sum[j] = {0.f};
        }

        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                data[j][i] = h2exp(__hsub2(data[j][i], __float2half2_rn(s_max[j])));
            }

#pragma unroll
            for (int j = 0; j < NUM; j++) {
                local_sum[j] += (float)(data[j][i].x + data[j][i].y);
            }
        }

        if (blockDim.x <= 32) {
            warpReduceSumV2<float, NUM>(local_sum);
        }
        else {
            blockReduceSumV2<float, NUM>(local_sum);
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                s_sum[j] = __fdividef(1.0f, local_sum[j] + 1e-6f);
            }
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                qk_offset[j] = ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id + j * gridDim.x) * (seq_len / 2)
                               + blockDim.x * i + threadIdx.x;
            }

#pragma unroll
            for (int j = 0; j < NUM; j++) {
                qk_buf_half2[qk_offset[j]] = __hmul2(data[j][i], __float2half2_rn(s_sum[j]));
            }
        }
    }
}

#define SOFTMAX_KERNEL(ITEMS_PER_THREAD)                                                                               \
    block.x /= ITEMS_PER_THREAD;                                                                                       \
    assert(block.x <= 1024);                                                                                           \
    if (is_half2) {                                                                                                    \
        if (grid.x % 4 == 0) {                                                                                         \
            grid.x /= 4;                                                                                               \
            softmax_kernel_v5_half2<ITEMS_PER_THREAD, 4><<<grid, block, 0, stream>>>(                                  \
                (half*)buffer, (const half*)attr_mask, batch_size, head_num, seq_len, (const half)scalar);             \
        }                                                                                                              \
        else {                                                                                                         \
            softmax_kernel_v4_half2<ITEMS_PER_THREAD><<<grid, block, 0, stream>>>(                                     \
                (half*)buffer, (const half*)attr_mask, batch_size, head_num, seq_len, (const half)scalar);             \
        }                                                                                                              \
    }                                                                                                                  \
    else {                                                                                                             \
        softmax_kernel_v4<ITEMS_PER_THREAD, T, T_IN>                                                                   \
            <<<grid, block, 0, stream>>>(buffer, buffer_src, attr_mask, batch_size, head_num, seq_len, scalar);        \
    }

template<typename T, typename T_IN>
void invokeMaskedSoftMax(T* buffer,
                         const T_IN* buffer_src,
                         const T* attr_mask,
                         const int batch_size,
                         const int seq_len,
                         const int head_num,
                         const T scalar,
                         cudaStream_t stream)
{

    dim3 grid(seq_len, batch_size, head_num);
    if (batch_size * head_num > 360) {
        grid.x = ceil(float(seq_len) / 32.0f);
    }

    bool is_half2 = sizeof(T) == 2 && sizeof(T_IN) == 2 && seq_len % 2 == 0;
    dim3 block((seq_len / (is_half2 ? 2 : 1) + 31) / 32 * 32);

    if (block.x > 3072 && block.x <= 4096) {
        SOFTMAX_KERNEL(4)
    }
    if (block.x > 2048) {
        SOFTMAX_KERNEL(3)
    }
    else if (block.x > 1024) {
        SOFTMAX_KERNEL(2)
    }
    else if (block.x > 0) {
        SOFTMAX_KERNEL(1)
    }
    else {
        FT_CHECK(seq_len <= 4096);
    }
}

template void invokeMaskedSoftMax(float* buffer,
                                  const float* buffer_src,
                                  const float* attr_mask,
                                  const int batch_size,
                                  const int seq_len,
                                  const int head_num,
                                  const float scalar,
                                  cudaStream_t stream);

template void invokeMaskedSoftMax(half* buffer,
                                  const float* buffer_src,
                                  const half* attr_mask,
                                  const int batch_size,
                                  const int seq_len,
                                  const int head_num,
                                  const half scalar,
                                  cudaStream_t stream);

template void invokeMaskedSoftMax(half* buffer,
                                  const half* buffer_src,
                                  const half* attr_mask,
                                  const int batch_size,
                                  const int seq_len,
                                  const int head_num,
                                  const half scalar,
                                  cudaStream_t stream);

template<typename T>
__global__ void
transpose(T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    int batch_id = blockIdx.x / (head_num * seq_len);
    int seq_id = blockIdx.x % seq_len;
    int head_id = (blockIdx.x % (head_num * seq_len)) / seq_len;
    dst[batch_id * (head_num * seq_len * size_per_head) + seq_id * head_num * size_per_head + head_id * size_per_head
        + threadIdx.x] = src[blockIdx.x * size_per_head + threadIdx.x];
}

template<>
__global__ void
transpose(half* src, half* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
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
void invokeTransposeQKV(T* dst,
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

        transpose<T><<<grid, block, 0, stream>>>(src, dst, batch_size, seq_len, head_num, size_per_head / 2);
    }
    else {
        const int seq_per_block = 1;
        grid.x = batch_size * head_num * seq_len / seq_per_block;
        block.x = seq_per_block * size_per_head;
        transpose<T><<<grid, block, 0, stream>>>(src, dst, batch_size, seq_len, head_num, size_per_head);
    }
}

template void invokeTransposeQKV(float* src,
                                 float* dst,
                                 const int batch_size,
                                 const int seq_len,
                                 const int head_num,
                                 const int size_per_head,
                                 cudaStream_t stream);

template void invokeTransposeQKV(half* src,
                                 half* dst,
                                 const int batch_size,
                                 const int seq_len,
                                 const int head_num,
                                 const int size_per_head,
                                 cudaStream_t stream);

template<typename T>
__global__ void add_QKV_bias_rebuild_padding(const T* Q,
                                             const T* bias_Q,
                                             const T* K,
                                             const T* bias_K,
                                             const T* V,
                                             const T* bias_V,
                                             T* q_buf_,
                                             T* k_buf_,
                                             T* v_buf_,
                                             const int batch_size,
                                             const int seq_len,
                                             const int head_num,
                                             const int size_per_head,
                                             const int* mask_offset)
{
    const int bid = blockIdx.x;

    const int tgt_batch_id = (bid + mask_offset[bid]) / seq_len;
    const int tgt_seq_id = (bid + mask_offset[bid]) % seq_len;
    const int n = head_num * size_per_head;
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        const int tgt_head_id = idx / size_per_head;
        const int tgt_hidden_id = idx % size_per_head;

        const int src_id = bid * n + idx;
        const int tgt_id = tgt_batch_id * head_num * seq_len * size_per_head + tgt_head_id * seq_len * size_per_head
                           + tgt_seq_id * size_per_head + tgt_hidden_id;

        q_buf_[tgt_id] = __ldg(&Q[src_id]) + __ldg(&bias_Q[idx]);
        k_buf_[tgt_id] = __ldg(&K[src_id]) + __ldg(&bias_K[idx]);
        v_buf_[tgt_id] = __ldg(&V[src_id]) + __ldg(&bias_V[idx]);
    }
}

template<typename T>
__global__ void rebuild_padding(const T* Q,
                                const T* K,
                                const T* V,
                                T* q_buf_,
                                T* k_buf_,
                                T* v_buf_,
                                const int batch_size,
                                const int seq_len,
                                const int head_num,
                                const int size_per_head,
                                const int* mask_offset)
{
    const int bid = blockIdx.x;

    const int tgt_batch_id = (bid + mask_offset[bid]) / seq_len;
    const int tgt_seq_id = (bid + mask_offset[bid]) % seq_len;
    const int n = head_num * size_per_head;
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        const int tgt_head_id = idx / size_per_head;
        const int tgt_hidden_id = idx % size_per_head;

        const int src_id = bid * n + idx;
        const int tgt_id = tgt_batch_id * head_num * seq_len * size_per_head + tgt_head_id * seq_len * size_per_head
                           + tgt_seq_id * size_per_head + tgt_hidden_id;

        q_buf_[tgt_id] = __ldg(&Q[src_id]);
        k_buf_[tgt_id] = __ldg(&K[src_id]);
        v_buf_[tgt_id] = __ldg(&V[src_id]);
    }
}

template<typename T>
void invokeAddQKVBiasRebuildPadding(T* Q,
                                    const T* bias_Q,
                                    T* K,
                                    const T* bias_K,
                                    T* V,
                                    const T* bias_V,
                                    T* q_buf,
                                    T* k_buf,
                                    T* v_buf,
                                    const int batch_size,
                                    const int seq_len,
                                    const int head_num,
                                    const int size_per_head,
                                    const int valid_word_num,
                                    const int* mask_offset,
                                    cudaStream_t stream)
{
    bool is_half2 = std::is_same<T, half>::value && (size_per_head % 2 == 0);
    int block_size = head_num * size_per_head;
    if (is_half2) {
        while (block_size > 512) {
            if (block_size % 2 == 0) {
                block_size /= 2;
            }
            else {
                is_half2 = false;
                block_size = std::min(block_size, 512);
                break;
            }
        }
    }
    else {
        block_size = std::min(block_size, 512);
    }

    if (bias_Q == nullptr && bias_K == nullptr && bias_V == nullptr) {
        if (is_half2) {
            rebuild_padding<<<valid_word_num, block_size, 0, stream>>>((half2*)Q,
                                                                       (half2*)K,
                                                                       (half2*)V,
                                                                       (half2*)q_buf,
                                                                       (half2*)k_buf,
                                                                       (half2*)v_buf,
                                                                       batch_size,
                                                                       seq_len,
                                                                       head_num,
                                                                       size_per_head / 2,
                                                                       mask_offset);
        }
        else {
            rebuild_padding<<<valid_word_num, block_size, 0, stream>>>(
                Q, K, V, q_buf, k_buf, v_buf, batch_size, seq_len, head_num, size_per_head, mask_offset);
        }
    }
    else if (bias_Q != nullptr && bias_K != nullptr && bias_V != nullptr) {
        if (is_half2) {
            add_QKV_bias_rebuild_padding<<<valid_word_num, block_size, 0, stream>>>((half2*)Q,
                                                                                    (const half2*)bias_Q,
                                                                                    (half2*)K,
                                                                                    (const half2*)bias_K,
                                                                                    (half2*)V,
                                                                                    (const half2*)bias_V,
                                                                                    (half2*)q_buf,
                                                                                    (half2*)k_buf,
                                                                                    (half2*)v_buf,
                                                                                    batch_size,
                                                                                    seq_len,
                                                                                    head_num,
                                                                                    size_per_head / 2,
                                                                                    mask_offset);
        }
        else {
            add_QKV_bias_rebuild_padding<<<valid_word_num, block_size, 0, stream>>>(Q,
                                                                                    bias_Q,
                                                                                    K,
                                                                                    bias_K,
                                                                                    V,
                                                                                    bias_V,
                                                                                    q_buf,
                                                                                    k_buf,
                                                                                    v_buf,
                                                                                    batch_size,
                                                                                    seq_len,
                                                                                    head_num,
                                                                                    size_per_head,
                                                                                    mask_offset);
        }
    }
    else {
        FT_CHECK(false);
    }
}

template void invokeAddQKVBiasRebuildPadding(float* Q,
                                             const float* bias_Q,
                                             float* K,
                                             const float* bias_K,
                                             float* V,
                                             const float* bias_V,
                                             float* q_buf,
                                             float* k_buf,
                                             float* v_buf,
                                             const int batch_size,
                                             const int seq_len,
                                             const int head_num,
                                             const int size_per_head,
                                             const int valid_word_num,
                                             const int* mask_offset,
                                             cudaStream_t stream);

template void invokeAddQKVBiasRebuildPadding(half* Q,
                                             const half* bias_Q,
                                             half* K,
                                             const half* bias_K,
                                             half* V,
                                             const half* bias_V,
                                             half* q_buf,
                                             half* k_buf,
                                             half* v_buf,
                                             const int batch_size,
                                             const int seq_len,
                                             const int head_num,
                                             const int size_per_head,
                                             const int valid_word_num,
                                             const int* mask_offset,
                                             cudaStream_t stream);

template<typename T>
__global__ void transpose_rebuild_padding(const T* src,
                                          T* dst,
                                          const int batch_size,
                                          const int seq_len,
                                          const int head_num,
                                          const int size_per_head,
                                          const int* mask_offset)
{
    // TODO: optimize this kernel?
    // do remove_sequence_length_padding
    const int bid = blockIdx.x;  // batch * seq_len or valid_word_num

    const int src_batch_id = (bid + mask_offset[bid]) / seq_len;
    const int src_seq_id = (bid + mask_offset[bid]) % seq_len;

    const int dst_seq_id = bid;

    for (int idx = threadIdx.x; idx < head_num * size_per_head; idx += blockDim.x) {
        const int head_id = idx / size_per_head;
        const int hidden_id = idx % size_per_head;
        dst[dst_seq_id * head_num * size_per_head + idx] =
            __ldg(&src[src_batch_id * head_num * seq_len * size_per_head + head_id * seq_len * size_per_head
                       + src_seq_id * size_per_head + hidden_id]);
    }
}

template<typename T>
void invokeTransposeAttentionOutRemovePadding(T* src,
                                              T* dst,
                                              const int valid_word_num,
                                              const int batch_size,
                                              const int seq_len,
                                              const int head_num,
                                              const int size_per_head,
                                              const int* mask_offset,
                                              cudaStream_t stream)
{
    bool is_half2 = std::is_same<T, half>::value && (size_per_head % 2 == 0);
    int block_size = head_num * size_per_head;
    if (is_half2) {
        while (block_size > 512) {
            if (block_size % 2 == 0) {
                block_size /= 2;
            }
            else {
                is_half2 = false;
                block_size = std::min(block_size, 1024);
                break;
            }
        }
    }
    else {
        block_size = std::min(block_size, 1024);
    }

    if (is_half2) {
        transpose_rebuild_padding<half2><<<valid_word_num, block_size, 0, stream>>>(
            (half2*)src, (half2*)dst, batch_size, seq_len, head_num, size_per_head / 2, mask_offset);
    }
    else {
        transpose_rebuild_padding<<<valid_word_num, block_size, 0, stream>>>(
            src, dst, batch_size, seq_len, head_num, size_per_head, mask_offset);
    }
}

template void invokeTransposeAttentionOutRemovePadding(float* src,
                                                       float* dst,
                                                       const int valid_word_num,
                                                       const int batch_size,
                                                       const int seq_len,
                                                       const int head_num,
                                                       const int size_per_head,
                                                       const int* mask_offset,
                                                       cudaStream_t stream);

template void invokeTransposeAttentionOutRemovePadding(half* src,
                                                       half* dst,
                                                       const int valid_word_num,
                                                       const int batch_size,
                                                       const int seq_len,
                                                       const int head_num,
                                                       const int size_per_head,
                                                       const int* mask_offset,
                                                       cudaStream_t stream);

template<typename T>
__global__ void add_fusedQKV_bias_transpose_kernel(T* q_buf,
                                                   T* k_buf,
                                                   T* v_buf,
                                                   const T* __restrict QKV,
                                                   const T* __restrict qkv_bias,
                                                   const int batch_size,
                                                   const int seq_len,
                                                   const int head_num,
                                                   const int size_per_head)
{
    // QKV: [m, 3, n]
    // qkv_bias: [3, n]
    // q_buf, k_buf, v_buf: [batch, head_num, seq_len, size_per_head]

    T* qkv_ptr[3] = {q_buf, k_buf, v_buf};
    const int n = head_num * size_per_head;
    for (int index = blockDim.x * blockIdx.x + threadIdx.x; index < batch_size * seq_len * 3 * n;
         index += gridDim.x * blockDim.x) {
        int bias_id = index % (3 * n);
        T val = __ldg(&QKV[index]) + __ldg(&qkv_bias[bias_id]);

        int tmp_index = index;
        const int target_batch_id = tmp_index / (seq_len * 3 * n);
        tmp_index -= target_batch_id * seq_len * 3 * n;
        const int seq_id = tmp_index / (3 * n);
        tmp_index -= seq_id * 3 * n;
        const int qkv_id = tmp_index / n;
        tmp_index -= qkv_id * n;
        const int head_id = tmp_index / size_per_head;
        const int size_id = tmp_index - head_id * size_per_head;

        qkv_ptr[qkv_id][target_batch_id * head_num * seq_len * size_per_head + head_id * seq_len * size_per_head
                        + seq_id * size_per_head + size_id] = val;
    }
}

template<typename T>
struct Vec_t {
};
template<>
struct Vec_t<float> {
    using Type = float2;
};
template<>
struct Vec_t<half> {
    using Type = uint32_t;
};

template<typename T>
__global__ void add_fusedQKV_bias_transpose_kernel(T* q_buf,
                                                   T* k_buf,
                                                   T* v_buf,
                                                   const T* __restrict QKV,
                                                   const T* __restrict qkv_bias,
                                                   const int batch_size,
                                                   const int seq_len,
                                                   const int head_num,
                                                   const int size_per_head,
                                                   const int rotary_embedding_dim)
{
    using Vec_t = typename Vec_t<T>::Type;
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.x;
    const int tidx = threadIdx.x;
    if (tidx * 2 >= size_per_head)
        return;

    const int batch_time_idx = seq_len * batch_idx + seq_idx;
    const int hidden_idx = head_idx * size_per_head + tidx * 2;
    const int n = head_num * size_per_head;

    // src QKV: [batch, time, 3, head, hidden]
    const int q_idx = batch_time_idx * 3 * n + hidden_idx;
    const int k_idx = batch_time_idx * 3 * n + hidden_idx + n;
    const int v_idx = batch_time_idx * 3 * n + hidden_idx + 2 * n;

    Vec_t q = *reinterpret_cast<const Vec_t*>(&QKV[q_idx]);
    Vec_t k = *reinterpret_cast<const Vec_t*>(&QKV[k_idx]);
    Vec_t v = *reinterpret_cast<const Vec_t*>(&QKV[v_idx]);

    // qkv_bias: [3, head, hidden]
    Vec_t q_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
    Vec_t k_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n]);
    Vec_t v_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + 2 * n]);

    q = mmha::add(q, q_bias);
    k = mmha::add(k, k_bias);
    v = mmha::add(v, v_bias);

    mmha::apply_rotary_embedding(q, k, tidx, rotary_embedding_dim, seq_idx);

    // q_buf, k_buf, v_buf: [batch, head_num, seq_len, size_per_head]
    const int dest_idx = size_per_head * seq_len * head_num * batch_idx + size_per_head * seq_len * head_idx
                         + size_per_head * seq_idx + tidx * 2;

    *reinterpret_cast<Vec_t*>(&q_buf[dest_idx]) = q;
    *reinterpret_cast<Vec_t*>(&k_buf[dest_idx]) = k;
    *reinterpret_cast<Vec_t*>(&v_buf[dest_idx]) = v;
}

template<typename T>
void invokeAddFusedQKVBiasTranspose(T* q_buf,
                                    T* k_buf,
                                    T* v_buf,
                                    T* QKV,
                                    const T* qkv_bias,
                                    const int batch_size,
                                    const int seq_len,
                                    const int head_num,
                                    const int size_per_head,
                                    const int rotary_embedding_dim,
                                    cudaStream_t stream)
{
    if (rotary_embedding_dim == 0) {
        const int m = batch_size * seq_len;
        const int n = head_num * size_per_head;
        dim3 block(384);
        dim3 grid((int)(ceil(1.0 * m * n / 384)));
        add_fusedQKV_bias_transpose_kernel<<<grid, block, 0, stream>>>(
            q_buf, k_buf, v_buf, QKV, qkv_bias, batch_size, seq_len, head_num, size_per_head);
    }
    else {
        // To implement rotary embeddings, each thread processes two QKV elems:
        dim3 block((size_per_head / 2 + 31) / 32 * 32);
        dim3 grid(seq_len, head_num, batch_size);
        add_fusedQKV_bias_transpose_kernel<<<grid, block, 0, stream>>>(
            q_buf, k_buf, v_buf, QKV, qkv_bias, batch_size, seq_len, head_num, size_per_head, rotary_embedding_dim);
    }
}

template void invokeAddFusedQKVBiasTranspose(float* q_buf,
                                             float* k_buf,
                                             float* v_buf,
                                             float* QKV,
                                             const float* qkv_bias,
                                             const int batch_size,
                                             const int seq_len,
                                             const int head_num,
                                             const int size_per_head,
                                             const int rotary_embedding_dim,
                                             cudaStream_t stream);

template void invokeAddFusedQKVBiasTranspose(half* q_buf,
                                             half* k_buf,
                                             half* v_buf,
                                             half* QKV,
                                             const half* qkv_bias,
                                             const int batch_size,
                                             const int seq_len,
                                             const int head_num,
                                             const int size_per_head,
                                             const int rotary_embedding_dim,
                                             cudaStream_t stream);

template<typename T>
__global__ void transpose_4d(T* dst,
                             T* src,
                             const int dim0,
                             const int dim1,
                             const int dim2,
                             const int dim3,
                             const int dim0_leading_dim,
                             const int ite)
{
    // transpose from [dim0, dim1, dim2, dim3] to [dim2, X, dim1, dim3]
    // where the dimension of X is dim0_leading_dim, and offset is ite * dim0
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dim0 * dim1 * dim2 * dim3; i += blockDim.x * gridDim.x) {
        int index = i;
        const int d3 = index % dim3;
        index = (index - d3) / dim3;
        const int d2 = index % dim2;
        index = (index - d2) / dim2;
        const int d1 = index % dim1;
        index = (index - d1) / dim1;
        const int d0 = index % dim0;
        index = (index - d0) / dim0;
        dst[d2 * dim0_leading_dim * dim1 * dim3 + (d0 + dim0 * ite) * dim1 * dim3 + d1 * dim3 + d3] = src[i];
    }
}

template<>
__global__ void transpose_4d(half* dst,
                             half* src,
                             const int dim0,
                             const int dim1,
                             const int dim2,
                             const int dim3,
                             const int dim0_leading_dim,
                             const int ite)
{
    half2* dst_ptr = (half2*)dst;
    half2* src_ptr = (half2*)src;
    const int half_dim3 = dim3 / 2;
    // transpose from [dim0, dim1, dim2, half_dim3] to [dim2, dim0, dim1, half_dim3]
    // where the dimension of X is dim0_leading_dim, and offset is ite * dim0
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dim0 * dim1 * dim2 * half_dim3;
         i += blockDim.x * gridDim.x) {
        int index = i;
        const int d3 = index % half_dim3;
        index = (index - d3) / half_dim3;
        const int d2 = index % dim2;
        index = (index - d2) / dim2;
        const int d1 = index % dim1;
        index = (index - d1) / dim1;
        const int d0 = index % dim0;
        index = (index - d0) / dim0;
        dst_ptr[d2 * dim0_leading_dim * dim1 * half_dim3 + (d0 + dim0 * ite) * dim1 * half_dim3 + d1 * half_dim3 + d3] =
            src_ptr[i];
    }
}

template<typename T>
void invokeTranspose4d(T* dst,
                       T* src,
                       const int local_batch_size,
                       const int seq_len,
                       const int size_per_head,
                       const int local_hidden_units,
                       const int local_head_num,
                       const int batch_size,
                       const int ite,
                       cudaStream_t stream)
{
    transpose_4d<<<local_batch_size * seq_len * local_hidden_units / 512, 512 / (4 / (sizeof(T))), 0, stream>>>(
        dst, src, local_batch_size, local_head_num, seq_len, size_per_head, batch_size, ite);
}

template void invokeTranspose4d(float* dst,
                                float* src,
                                const int local_batch_size,
                                const int seq_len,
                                const int size_per_head,
                                const int local_hidden_units,
                                const int local_head_num,
                                const int batch_size,
                                const int ite,
                                cudaStream_t stream);

template void invokeTranspose4d(half* dst,
                                half* src,
                                const int local_batch_size,
                                const int seq_len,
                                const int size_per_head,
                                const int local_hidden_units,
                                const int local_head_num,
                                const int batch_size,
                                const int ite,
                                cudaStream_t stream);

template<typename T>
__global__ void transpose_4d_batch_major_k_cache(
    T* k_dst, const T* k_src, const int head_num, const int size_per_head, const int seq_len, const int max_seq_len)
{
    const int batch_id = blockIdx.y;
    const int head_id = blockIdx.z;
    constexpr int X_ELEMS = (sizeof(T) == 4) ? 4 : 8;

    auto key_src = reinterpret_cast<const uint4*>(k_src + batch_id * head_num * size_per_head * seq_len
                                                  + head_id * size_per_head * seq_len);
    auto key_dst = reinterpret_cast<uint4*>(k_dst + batch_id * head_num * size_per_head * max_seq_len
                                            + head_id * size_per_head * max_seq_len);

    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size_per_head_div_x = size_per_head / X_ELEMS;
    if (out_idx >= size_per_head_div_x * max_seq_len) {
        return;
    }

    int idx = out_idx;
    const int k_seq_len_id = idx % max_seq_len;
    idx = (idx - k_seq_len_id) / max_seq_len;
    const int k_head_size_id = idx % size_per_head_div_x;

    if (k_seq_len_id < seq_len) {
        key_dst[out_idx] = key_src[k_seq_len_id * size_per_head_div_x + k_head_size_id];
    }
}

template<typename T>
__global__ void transpose_4d_batch_major_v_cache(
    T* v_dst, const T* v_src, const int head_num, const int size_per_head, const int seq_len, const int max_seq_len)
{
    const int batch_id = blockIdx.y;
    const int head_id = blockIdx.z;

    // 16 byte loads will handle "x" dimension
    auto val_src = reinterpret_cast<const uint4*>(v_src + batch_id * head_num * size_per_head * seq_len
                                                  + head_id * size_per_head * seq_len);
    auto val_dst = reinterpret_cast<uint4*>(v_dst + batch_id * head_num * size_per_head * max_seq_len
                                            + head_id * size_per_head * max_seq_len);

    // idx is over output dimension L * size_per_head / x for values
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr int X_ELEMS = (sizeof(T) == 4) ? 4 : 8;
    const int size_per_head_div_x = size_per_head / X_ELEMS;

    if (idx >= size_per_head_div_x * seq_len)
        return;

    val_dst[idx] = val_src[idx];
}

template<typename T>
void invokeTranspose4dBatchMajor(T* k_dst,
                                 T* v_dst,
                                 const T* k_src,
                                 const T* v_src,
                                 const int local_batch_size,
                                 const int seq_len,
                                 const int max_seq_len,
                                 const int size_per_head,
                                 const int local_head_num,
                                 cudaStream_t stream)
{
    constexpr int block_sz = 128;
    constexpr int x = (sizeof(T) == 4) ? 4 : 8;
    int size = max_seq_len * size_per_head / x;
    dim3 grid((size + block_sz - 1) / block_sz, local_batch_size, local_head_num);
    dim3 grid_v((seq_len * size_per_head / x + block_sz - 1) / block_sz, local_batch_size, local_head_num);

    transpose_4d_batch_major_k_cache<<<grid, block_sz, 0, stream>>>(
        k_dst, k_src, local_head_num, size_per_head, seq_len, max_seq_len);

    transpose_4d_batch_major_v_cache<<<grid_v, block_sz, 0, stream>>>(
        v_dst, v_src, local_head_num, size_per_head, seq_len, max_seq_len);
}

template void invokeTranspose4dBatchMajor(float* k_dst,
                                          float* v_dst,
                                          const float* k_src,
                                          const float* v_src,
                                          const int local_batch_size,
                                          const int seq_len,
                                          const int max_seq_len,
                                          const int size_per_head,
                                          const int local_head_num,
                                          cudaStream_t stream);

template void invokeTranspose4dBatchMajor(half* k_dst,
                                          half* v_dst,
                                          const half* k_src,
                                          const half* v_src,
                                          const int local_batch_size,
                                          const int seq_len,
                                          const int max_seq_len,
                                          const int size_per_head,
                                          const int local_head_num,
                                          cudaStream_t stream);

template<typename T>
__global__ void addRelativeAttentionBias(
    T* qk_buf, const T* relative_attention_bias, const int batch_size, const int head_num, const int seq_len)
{
    for (int i = threadIdx.x; i < batch_size * seq_len; i += blockDim.x) {
        int batch_id = i / seq_len;
        int seq_id = i % seq_len;

        const int bias_index = blockIdx.x * seq_len + seq_id;
        const int qk_index = batch_id * head_num * seq_len * seq_len + bias_index;
        qk_buf[qk_index] = qk_buf[qk_index] + relative_attention_bias[bias_index];
    }
}

template<typename T>
__global__ void addRelativeAttentionBias(
    half2* qk_buf, const half2* relative_attention_bias, const int batch_size, const int head_num, const int seq_len)
{
    const int half2_seq_len = seq_len / 2;
    for (int i = threadIdx.x; i < batch_size * half2_seq_len; i += blockDim.x) {
        int batch_id = i / half2_seq_len;
        int seq_id = i % half2_seq_len;

        const int bias_index = blockIdx.x * half2_seq_len + seq_id;
        const int qk_index = batch_id * gridDim.x * half2_seq_len + bias_index;
        qk_buf[qk_index] = qk_buf[qk_index] + relative_attention_bias[bias_index];
    }
}

template<typename T>
void invokeAddRelativeAttentionBias(T* qk_buf,
                                    const T* relative_attention_bias,
                                    const int batch_size,
                                    const int head_num,
                                    const int seq_len,
                                    cudaStream_t stream)
{
    // qk_buf: [batch_size, head_num, seq_len, seq_len]
    // relative_attention_bias: [1, head_num, seq_len, seq_len]
    dim3 grid(head_num * seq_len);
    dim3 block(512);
    if (std::is_same<T, half>::value && seq_len % 2 == 0) {
        addRelativeAttentionBias<half2><<<grid, block, 0, stream>>>(
            (half2*)qk_buf, (const half2*)relative_attention_bias, batch_size, head_num, seq_len);
    }
    else {
        addRelativeAttentionBias<<<grid, block, 0, stream>>>(
            qk_buf, relative_attention_bias, batch_size, head_num, seq_len);
    }
}

template void invokeAddRelativeAttentionBias(float* qk_buf,
                                             const float* relative_attention_bias,
                                             const int batch_size,
                                             const int head_num,
                                             const int seq_len,
                                             cudaStream_t stream);

template void invokeAddRelativeAttentionBias(half* qk_buf,
                                             const half* relative_attention_bias,
                                             const int batch_size,
                                             const int head_num,
                                             const int seq_len,
                                             cudaStream_t stream);

}  // namespace fastertransformer
