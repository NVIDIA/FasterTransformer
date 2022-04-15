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

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

#include "src/fastertransformer/kernels/decoder_masked_multihead_attention.h"
#include "src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"
#include "src/fastertransformer/layers/attention_layers/DecoderCrossAttentionLayer.h"

namespace fastertransformer {

const int WARP_SIZE = 32;
const bool ATTENION_OPT = true;
const int ATTENTION_BLOCK_SIZE = 256;

///////////////////////////////////////////////////////////////////////////////////////////////////

template<int HALF_ELEMENTS_PER_WARP_LOAD>
using Copy_half_t = typename std::conditional<
    HALF_ELEMENTS_PER_WARP_LOAD == 32,
    half,
    typename std::conditional<HALF_ELEMENTS_PER_WARP_LOAD == 64,
                              int,
                              typename std::conditional<HALF_ELEMENTS_PER_WARP_LOAD == 128, int2, int4>::type>::type>::
    type;

template<typename T, int ELEMENTS_PER_WARP_LOAD>
using Copy_t = Copy_half_t<sizeof(T) / sizeof(half) * ELEMENTS_PER_WARP_LOAD>;

///////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void cross_attention_kernel(T* query_buf,
                                       const T* Q_bias,
                                       T* key_cache,
                                       const T* K_bias,
                                       T* value_cache,
                                       const T* V_bias,
                                       const int* length_per_sample,
                                       T* context_buf,
                                       const bool* finished,
                                       int batch_size,
                                       int head_num,
                                       int size_per_head,
                                       int step,
                                       const int seq_len,
                                       const T scalar)
{
    if (finished != nullptr && finished[blockIdx.x / head_num] == true) {
        return;
    }
    int tid = threadIdx.x;
    int bid = blockIdx.x / head_num;
    int head_id = blockIdx.x % head_num;

    extern __shared__ __align__(sizeof(T)) unsigned s_buf[];
    T* sq = reinterpret_cast<T*>(s_buf);
    T* logits = reinterpret_cast<T*>(&sq[size_per_head]);

    int length = __ldg(&length_per_sample[bid]);

    int qkv_id = bid * head_num * size_per_head + head_id * size_per_head + tid;
    int qkv_bias_id = head_id * size_per_head + tid;

    if (tid < size_per_head) {
        sq[tid] = query_buf[qkv_id] + Q_bias[qkv_bias_id];
    }
    __syncthreads();

    for (int ite = 0; ite < length; ++ite) {
        int key_id = bid * (seq_len * head_num * size_per_head) + ite * (head_num * size_per_head)
                     + head_id * size_per_head + tid;

        T key = tid < size_per_head ? key_cache[key_id] : (T)(0.0f);

        // For the first step, we should add bias to key memory cache.
        // The KV memory cache only need to be updated at the first step.
        if (step == 1 && tid < size_per_head) {
            key += K_bias[head_id * size_per_head + tid];
            key_cache[key_id] = key;
        }

        T val = (tid < size_per_head) ? key * sq[tid] * scalar : (T)(0.0f);
        T qk = blockReduceSum(val);
        if (threadIdx.x == 0) {
            logits[ite] = qk;
        }
        __syncthreads();  // try to remove
    }
    __syncthreads();

    __shared__ float s_max_val, s_sum;

    float local_i = tid < length ? (float)logits[tid] : -1e20f;
    float max_val = blockReduceMax(local_i);
    if (tid == 0) {
        s_max_val = max_val;
    }
    __syncthreads();

    local_i -= s_max_val;
    float local_o = tid < length ? __expf(local_i) : 0.0f;
    float val = blockReduceSum(local_o);

    if (tid == 0) {
        s_sum = val + 1e-6;
    }
    __syncthreads();
    if (tid < length) {
        logits[tid] = local_o / s_sum;
    }
    __syncthreads();

    if (tid < size_per_head) {
        T sum = (T)0.0f;
        for (int ite = 0; ite < length; ++ite) {
            int value_id = bid * seq_len * head_num * size_per_head + ite * head_num * size_per_head
                           + head_id * size_per_head + tid;

            T value = value_cache[value_id];

            // for the first step, we should add bias to key memory cache
            if (step == 1) {
                value += V_bias[head_id * size_per_head + tid];
                value_cache[value_id] = value;
            }
            sum += value * logits[ite];
        }
        context_buf[bid * head_num * size_per_head + head_id * size_per_head + tid] = sum;
    }
}

template<typename T, int size_per_head, int block_sz>
__global__ void cross_attention_kernel_opt(T* __restrict query_buf,
                                           const T* __restrict Q_bias,
                                           T* __restrict key_cache,
                                           const T* __restrict K_bias,
                                           T* __restrict value_cache,
                                           const T* __restrict V_bias,
                                           const int* length_per_sample,
                                           T* __restrict context_buf,
                                           const bool* finished,
                                           int batch_size,
                                           int head_num,
                                           const int step,
                                           const int seq_len,
                                           const float scalar)
{
    if (finished != nullptr && finished[blockIdx.x / head_num] == true) {
        return;
    }
    typedef Copy_t<T, size_per_head> copy_t;
    const int elems_per_thread = size_per_head / WARP_SIZE;
    union Access_t {
        copy_t v;
        T x[elems_per_thread];  // supported size 1,2,4
    };
    typedef struct Float_n_t {
        float x[elems_per_thread];  // supported size 1,2,4
    } float_n_t;

    __shared__ float_n_t sq[block_sz];
    extern __shared__ float logits[];  // use to store the logits from [0~step]

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warp_num = block_sz / WARP_SIZE;

    typedef cub::BlockReduce<float, block_sz> MaxValBlockReduce;
    typedef cub::BlockReduce<float, block_sz> BlockReduce;
    __shared__ typename MaxValBlockReduce::TempStorage max_val_block_temp_storage;
    __shared__ typename BlockReduce::TempStorage block_temp_storage;

    __shared__ typename cub::WarpReduce<float>::TempStorage temp_storage[warp_num];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x / head_num;
    const int head_id = blockIdx.x % head_num;

    int length = __ldg(&length_per_sample[bid]);

    const int lane_id = tid % WARP_SIZE;

    int qkv_id = bid * head_num * size_per_head + head_id * size_per_head;
    int qkv_bias_id = head_id * size_per_head;

    int key_value_id = bid * (seq_len * head_num * size_per_head) + +head_id * size_per_head;

    query_buf = &query_buf[qkv_id];
    K_bias = &K_bias[qkv_bias_id];
    key_cache = &key_cache[key_value_id];
    Q_bias = &Q_bias[qkv_bias_id];
    V_bias = &V_bias[qkv_bias_id];
    value_cache = &value_cache[key_value_id];
    context_buf = &context_buf[qkv_id];

    Access_t bias_r, key_val_r, query_buf_r;

    // each warp will have its own copy of sq
    query_buf_r.v = *((copy_t*)query_buf + lane_id);
    bias_r.v = *((copy_t*)Q_bias + lane_id);
    float qb_r[elems_per_thread];
    for (int i = 0; i < elems_per_thread; ++i) {
        qb_r[i] = (float)query_buf_r.x[i] + (float)bias_r.x[i];
    }

    // offset for each step
    int offset = head_num * size_per_head;

    bias_r.v = *((copy_t*)K_bias + lane_id);
    for (int ite = warp_id; ite < length; ite += warp_num) {
        key_val_r.v = *((copy_t*)&key_cache[ite * offset] + lane_id);

        // For the first step, we should add bias to key memory cache.
        // The KV memory cache only need to be updated at the first step.
        if (step == 1) {
            for (int i = 0; i < elems_per_thread; i++) {
                key_val_r.x[i] = (float)key_val_r.x[i] + (float)bias_r.x[i];
            }
            *((copy_t*)&key_cache[ite * offset] + lane_id) = key_val_r.v;
        }
        float val = 0.f;
        for (int i = 0; i < elems_per_thread; i++) {
            val = val + (float)key_val_r.x[i] * qb_r[i] * scalar;
        }
        float qk = cub::WarpReduce<float>(temp_storage[warp_id]).Sum(val);
        if (lane_id == 0) {
            logits[ite] = qk;
        }
    }
    __syncthreads();

    __shared__ float s_max_val, s_sum;
    float local_i = -1e20f;
    for (int i = tid; i < length; i += blockDim.x) {
        local_i = max(local_i, logits[i]);
    }

    float max_val = MaxValBlockReduce(max_val_block_temp_storage).Reduce(local_i, cub::Max());
    if (tid == 0) {
        s_max_val = max_val;
    }
    __syncthreads();

    float local_o = 0.0f;
    for (int i = tid; i < length; i += blockDim.x) {
        logits[i] = __expf(logits[i] - s_max_val);
        local_o += logits[i];
    }
    float val = BlockReduce(block_temp_storage).Sum(local_o);

    if (tid == 0) {
        s_sum = val + 1e-6;
    }
    __syncthreads();

    float s_sum_inverse = __fdividef(1.0f, s_sum);
    for (int i = tid; i < length; i += blockDim.x) {
        logits[i] = logits[i] * s_sum_inverse;
    }
    __syncthreads();

    // This optimization introduces discrepancy because of different order in FP32 summation
    float sum_r[elems_per_thread] = {0.f};
    bias_r.v = *((copy_t*)V_bias + lane_id);
    for (int ite = warp_id; ite < length; ite += warp_num) {
        key_val_r.v = *((copy_t*)&value_cache[ite * offset] + lane_id);

        // For the first step, we should add bias to key memory cache.
        if (step == 1) {
            for (int i = 0; i < elems_per_thread; i++) {
                key_val_r.x[i] = (float)key_val_r.x[i] + (float)bias_r.x[i];
            }
            *((copy_t*)&value_cache[ite * offset] + lane_id) = key_val_r.v;
        }
        for (int i = 0; i < elems_per_thread; ++i) {
            sum_r[i] += (float)key_val_r.x[i] * logits[ite];
        }
    }
    for (int i = 0; i < elems_per_thread; i++) {
        sq[warp_id * WARP_SIZE + lane_id].x[i] = sum_r[i];
    }
    __syncthreads();
    if (threadIdx.x < WARP_SIZE) {
#pragma unroll
        for (int j = 1; j < warp_num; j++) {
            for (int i = 0; i < elems_per_thread; ++i) {
                sum_r[i] = sum_r[i] + (float)sq[j * WARP_SIZE + threadIdx.x].x[i];
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
        key_val_r.x[i] = sum_r[i];
    }
    if (threadIdx.x < WARP_SIZE) {
        *((copy_t*)context_buf + lane_id) = key_val_r.v;
    }
}

template<typename T>
void cross_attention_dispatch(T* query_buf,
                              const T* Q_bias,
                              T* key_cache,
                              const T* K_bias,
                              T* value_cache,
                              const T* V_bias,
                              const int* length,
                              T* context_buf,
                              const bool* finished,
                              const int max_batch_size,
                              const int inference_batch_size,
                              const int head_num,
                              const int size_per_head,
                              const int step,
                              const int seq_len,
                              const bool batch_major_cache,
                              const float q_scaling,
                              cudaStream_t stream)
{
    if (!batch_major_cache) {
        const int block_sz = ATTENTION_BLOCK_SIZE;
        float scalar = 1.f / (sqrtf(size_per_head * 1.0f) * q_scaling);

        dim3 grid(inference_batch_size * head_num);

        int cond = size_per_head * ((ATTENION_OPT) ? 1 : 0);
        switch (cond) {
            case 32:
                cross_attention_kernel_opt<T, 32, block_sz>
                    <<<grid, block_sz, sizeof(float) * seq_len, stream>>>(query_buf,
                                                                          Q_bias,
                                                                          key_cache,
                                                                          K_bias,
                                                                          value_cache,
                                                                          V_bias,
                                                                          length,
                                                                          context_buf,
                                                                          finished,
                                                                          max_batch_size,
                                                                          head_num,
                                                                          step,
                                                                          seq_len,
                                                                          scalar);
                break;
            case 64:
                cross_attention_kernel_opt<T, 64, block_sz>
                    <<<grid, block_sz, sizeof(float) * seq_len, stream>>>(query_buf,
                                                                          Q_bias,
                                                                          key_cache,
                                                                          K_bias,
                                                                          value_cache,
                                                                          V_bias,
                                                                          length,
                                                                          context_buf,
                                                                          finished,
                                                                          max_batch_size,
                                                                          head_num,
                                                                          step,
                                                                          seq_len,
                                                                          scalar);
                break;
            case 128:
                cross_attention_kernel_opt<T, 128, block_sz>
                    <<<grid, block_sz, sizeof(float) * seq_len, stream>>>(query_buf,
                                                                          Q_bias,
                                                                          key_cache,
                                                                          K_bias,
                                                                          value_cache,
                                                                          V_bias,
                                                                          length,
                                                                          context_buf,
                                                                          finished,
                                                                          max_batch_size,
                                                                          head_num,
                                                                          step,
                                                                          seq_len,
                                                                          scalar);
                break;
            default:
                // default path

                int block_size = 128;

                if (seq_len <= 64) {
                    block_size = 64;
                }
                else if (seq_len <= 128 && seq_len > size_per_head) {
                    block_size = 128;
                }
                else if (seq_len > 128 && seq_len <= 256) {
                    block_size = 256;
                }
                else if (seq_len > 256 && seq_len <= 512) {
                    block_size = 512;
                }
                else {
                    block_size = 1024;
                }

                if (block_size < size_per_head) {
                    block_size = size_per_head;
                }

                assert(block_size <= 1024);
                dim3 block(block_size);

                int shared_size = sizeof(T) * (size_per_head + seq_len);
                cross_attention_kernel<T><<<grid, block, shared_size, stream>>>(query_buf,
                                                                                Q_bias,
                                                                                key_cache,
                                                                                K_bias,
                                                                                value_cache,
                                                                                V_bias,
                                                                                length,
                                                                                context_buf,
                                                                                finished,
                                                                                max_batch_size,
                                                                                head_num,
                                                                                size_per_head,
                                                                                step,
                                                                                seq_len,
                                                                                scalar);
        }
    }
    else {
        assert(step > 0);
        // assert(size_per_head == 32 || size_per_head == 64 || size_per_head == 128);
        using DataType = typename std::conditional<sizeof(T) == 4, float, uint16_t>::type;
        // Prepare the parameters.
        Cross_multihead_attention_params<DataType> params;
        memset(&params, 0, sizeof(params));
        params.q_bias = reinterpret_cast<const DataType*>(Q_bias);
        params.k_bias = reinterpret_cast<const DataType*>(K_bias);
        params.v_bias = reinterpret_cast<const DataType*>(V_bias);

        // Set the output buffer.
        params.out = reinterpret_cast<DataType*>(context_buf);

        // Set the input buffers.
        params.q = reinterpret_cast<const DataType*>(query_buf);
        params.k = nullptr;
        params.v = nullptr;
        params.stride = 0;
        params.finished = const_cast<bool*>(finished);

        params.memory_length_per_sample = const_cast<int*>(length);

        params.k_cache = reinterpret_cast<DataType*>(key_cache);
        params.v_cache = reinterpret_cast<DataType*>(value_cache);
        params.batch_size = inference_batch_size;
        // TODO(bhsueh) We can use batch but not batch * beam_width in k/v cache in cross attention
        // because they are same for all beams.
        params.beam_width = 1;  // We don't care the beam_width in cross attention, set to 1 is enough.
        params.seq_length = seq_len;
        params.timestep = step - 1;
        params.num_heads = head_num;
        params.hidden_size_per_head = size_per_head;
        params.inv_sqrt_dh = 1.F / (sqrtf((float)params.hidden_size_per_head) * q_scaling);

        cross_multihead_attention(params, stream);
    }
}

template void cross_attention_dispatch(float* query_buf,
                                       const float* Q_bias,
                                       float* key_cache,
                                       const float* K_bias,
                                       float* value_cache,
                                       const float* V_bias,
                                       const int* length,
                                       float* context_buf,
                                       const bool* finished,
                                       const int max_batch_size,
                                       const int inference_batch_size,
                                       const int head_num,
                                       const int size_per_head,
                                       const int step,
                                       const int seq_len,
                                       const bool batch_major_cache,
                                       const float q_scaling,
                                       cudaStream_t stream);

template void cross_attention_dispatch(half* query_buf,
                                       const half* Q_bias,
                                       half* key_cache,
                                       const half* K_bias,
                                       half* value_cache,
                                       const half* V_bias,
                                       const int* length,
                                       half* context_buf,
                                       const bool* finished,
                                       const int max_batch_size,
                                       const int inference_batch_size,
                                       const int head_num,
                                       const int size_per_head,
                                       const int step,
                                       const int seq_len,
                                       const bool batch_major_cache,
                                       const float q_scaling,
                                       cudaStream_t stream);

// Currently need to transpose at the first step in Cross attention
template<typename T>
__global__ void transpose_4d_batch_major_mem_k_cache(
    T* k_dst, const T* k_src, const int head_num, const int size_per_head, const int max_seq_len)
{
    // B, L, H, Dh -> B, H, Dh/x, L, x
    const int batch_id = blockIdx.y;
    const int head_id = blockIdx.z;
    constexpr int X_ELEMS = (sizeof(T) == 4) ? 4 : 8;

    auto key_src = reinterpret_cast<const uint4*>(k_src + batch_id * head_num * size_per_head * max_seq_len
                                                  + head_id * size_per_head);
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

    key_dst[out_idx] = key_src[k_seq_len_id * head_num * size_per_head_div_x + k_head_size_id];
}

template<typename T>
__global__ void transpose_4d_batch_major_mem_v_cache(
    T* v_dst, const T* v_src, const int head_num, const int size_per_head, const int max_seq_len)
{
    // B, L, H, Dh -> B, H, L, Dh
    const int batch_id = blockIdx.y;
    const int head_id = blockIdx.z;

    // 16 byte loads will handle "x" dimension
    auto val_src = reinterpret_cast<const uint4*>(v_src + batch_id * head_num * size_per_head * max_seq_len
                                                  + head_id * size_per_head);
    auto val_dst = reinterpret_cast<uint4*>(v_dst + batch_id * head_num * size_per_head * max_seq_len
                                            + head_id * size_per_head * max_seq_len);

    // idx is over output dimension L * size_per_head / x for values
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr int X_ELEMS = (sizeof(T) == 4) ? 4 : 8;
    const int size_per_head_div_x = size_per_head / X_ELEMS;
    if (out_idx >= size_per_head_div_x * max_seq_len) {
        return;
    }

    int idx = out_idx;
    const int v_head_size_id = idx % size_per_head_div_x;
    idx = (idx - v_head_size_id) / size_per_head_div_x;
    const int v_seq_len_id = idx % max_seq_len;

    val_dst[out_idx] = val_src[v_seq_len_id * head_num * size_per_head_div_x + v_head_size_id];
}

template<typename T>
void transpose_4d_batch_major_memory_kernelLauncher(T* dst,
                                                    const T* src,
                                                    const int local_batch_size,
                                                    const int max_seq_len,
                                                    const int size_per_head,
                                                    const int local_head_num,
                                                    const bool k_cache,
                                                    cudaStream_t stream)
{
    constexpr int block_sz = 128;

    constexpr int x = (sizeof(T) == 4) ? 4 : 8;
    int size = max_seq_len * size_per_head / x;
    dim3 grid((size + block_sz - 1) / block_sz, local_batch_size, local_head_num);

    if (k_cache) {
        transpose_4d_batch_major_mem_k_cache<<<grid, block_sz, 0, stream>>>(
            dst, src, local_head_num, size_per_head, max_seq_len);
    }
    else {
        transpose_4d_batch_major_mem_v_cache<<<grid, block_sz, 0, stream>>>(
            dst, src, local_head_num, size_per_head, max_seq_len);
    }
}

template void transpose_4d_batch_major_memory_kernelLauncher(float* dst,
                                                             const float* src,
                                                             const int local_batch_size,
                                                             const int max_seq_len,
                                                             const int size_per_head,
                                                             const int local_head_num,
                                                             const bool k_cache,
                                                             cudaStream_t stream);

template void transpose_4d_batch_major_memory_kernelLauncher(half* dst,
                                                             const half* src,
                                                             const int local_batch_size,
                                                             const int max_seq_len,
                                                             const int size_per_head,
                                                             const int local_head_num,
                                                             const bool k_cache,
                                                             cudaStream_t stream);

template<typename T>
void DecoderCrossAttentionLayer<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        q_buf_ = reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * hidden_units_, false));
        context_buf_ = reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * hidden_units_, false));

        if (is_batch_major_cache_) {
            mem_cache_buf_ = reinterpret_cast<T*>(
                allocator_->malloc(sizeof(T) * max_batch_size_ * max_mem_seq_len_ * hidden_units_, false));
        }
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void DecoderCrossAttentionLayer<T>::allocateBuffer(size_t batch_size, size_t max_mem_seq_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    q_buf_ = reinterpret_cast<T*>(allocator_->reMalloc(q_buf_, sizeof(T) * batch_size * hidden_units_, false));
    context_buf_ =
        reinterpret_cast<T*>(allocator_->reMalloc(context_buf_, sizeof(T) * batch_size * hidden_units_, false));

    if (is_batch_major_cache_) {
        mem_cache_buf_ = reinterpret_cast<T*>(
            allocator_->reMalloc(mem_cache_buf_, sizeof(T) * batch_size * max_mem_seq_len * hidden_units_, false));
    }
    is_allocate_buffer_ = true;
}

template<typename T>
void DecoderCrossAttentionLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        allocator_->free(q_buf_);
        allocator_->free(context_buf_);
        if (is_batch_major_cache_) {
            allocator_->free(mem_cache_buf_);
        }
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool DecoderCrossAttentionLayer<T>::isValidBatchSize(size_t batch_size)
{
    if (batch_size <= max_batch_size_) {
        return true;
    }
    else {
        freeBuffer();
        max_batch_size_ = batch_size * 1.2;
        return true;
    }
}

template<typename T>
bool DecoderCrossAttentionLayer<T>::isValidSeqLen(size_t seq_len)
{
    if (seq_len <= max_mem_seq_len_) {
        return true;
    }
    else {
        freeBuffer();
        max_mem_seq_len_ = seq_len * 1.2;
        return true;
    }
}

template<typename T>
DecoderCrossAttentionLayer<T>::DecoderCrossAttentionLayer(size_t max_batch_size,
                                                          size_t head_num,
                                                          size_t size_per_head,
                                                          size_t d_model,
                                                          const float q_scaling,
                                                          cudaStream_t stream,
                                                          cublasMMWrapper* cublas_wrapper,
                                                          IAllocator* allocator,
                                                          bool is_free_buffer_after_forward):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_size_(max_batch_size),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num_ * size_per_head_),
    d_model_(d_model),
    q_scaling_(q_scaling)
{
    FT_CHECK(size_per_head_ == 32 || size_per_head_ == 64 || size_per_head_ == 96 || size_per_head_ == 128
             || size_per_head_ == 160 || size_per_head_ == 192 || size_per_head_ == 224 || size_per_head_ == 256);
}

template<typename T>
DecoderCrossAttentionLayer<T>::DecoderCrossAttentionLayer(size_t max_batch_size,
                                                          size_t head_num,
                                                          size_t size_per_head,
                                                          cudaStream_t stream,
                                                          cublasMMWrapper* cublas_wrapper,
                                                          IAllocator* allocator,
                                                          bool is_free_buffer_after_forward):
    DecoderCrossAttentionLayer<T>(max_batch_size,
                                  head_num,
                                  size_per_head,
                                  head_num * size_per_head,
                                  1.0f,
                                  stream,
                                  cublas_wrapper,
                                  allocator,
                                  is_free_buffer_after_forward)
{
}

template<typename T>
DecoderCrossAttentionLayer<T>::DecoderCrossAttentionLayer(size_t max_batch_size,
                                                          size_t head_num,
                                                          size_t size_per_head,
                                                          const float q_scaling,
                                                          cudaStream_t stream,
                                                          cublasMMWrapper* cublas_wrapper,
                                                          IAllocator* allocator,
                                                          bool is_free_buffer_after_forward):
    DecoderCrossAttentionLayer<T>(max_batch_size,
                                  head_num,
                                  size_per_head,
                                  head_num * size_per_head,
                                  q_scaling,
                                  stream,
                                  cublas_wrapper,
                                  allocator,
                                  is_free_buffer_after_forward)
{
}

template<typename T>
DecoderCrossAttentionLayer<T>::DecoderCrossAttentionLayer(DecoderCrossAttentionLayer<T> const& attention_layer):
    BaseAttentionLayer<T>(attention_layer.stream_,
                          attention_layer.cublas_wrapper_,
                          attention_layer.allocator_,
                          attention_layer.is_free_buffer_after_forward_),
    max_batch_size_(attention_layer.max_batch_size_),
    head_num_(attention_layer.head_num_),
    size_per_head_(attention_layer.size_per_head_),
    hidden_units_(attention_layer.hidden_units_),
    d_model_(attention_layer.d_model_),
    q_scaling_(attention_layer.q_scaling_)
{
    FT_CHECK(size_per_head_ == 32 || size_per_head_ == 64 || size_per_head_ == 96 || size_per_head_ == 128
             || size_per_head_ == 160 || size_per_head_ == 192 || size_per_head_ == 224 || size_per_head_ == 256);
}

template<typename T>
DecoderCrossAttentionLayer<T>::~DecoderCrossAttentionLayer()
{
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void DecoderCrossAttentionLayer<T>::forward(std::vector<fastertransformer::Tensor>* output_tensors,
                                            const std::vector<fastertransformer::Tensor>* input_tensors,
                                            const AttentionWeight<T>* attention_weights)
{
    // input tensors:
    //      attention_input [batch_size, d_model],
    //      encoder_output [batch_size, mem_max_seq_len, memory_d_model],
    //      encoder_sequence_length [batch_size],
    //      finished [batch_size],
    //      step [1] on cpu

    // output tensors:
    //      decoder_layer_output [batch_size, d_model],
    //      key_mem_cache [batch_size, mem_max_seq_len, hidden_dimension],
    //      value_mem_cache [batch_size, mem_max_seq_len, hidden_dimension]

    FT_CHECK(input_tensors->size() == 5);
    FT_CHECK(output_tensors->size() == 3);
    FT_CHECK(isValidBatchSize(input_tensors->at(0).shape[0]));
    FT_CHECK(isValidSeqLen(input_tensors->at(1).shape[1]));
    allocateBuffer(input_tensors->at(0).shape[0], input_tensors->at(1).shape[1]);

    const T* attention_input = reinterpret_cast<const T*>(input_tensors->at(0).data);
    Tensor encoder_output_tensor = input_tensors->at(1);
    const int* memory_sequence_length = reinterpret_cast<const int*>(input_tensors->at(2).data);
    const bool* finished = reinterpret_cast<const bool*>(input_tensors->at(3).data);
    const int step = *reinterpret_cast<const int*>(input_tensors->at(4).data);

    T* attention_out = (T*)(output_tensors->at(0).data);
    T* key_mem_cache = (T*)(output_tensors->at(1).data);
    T* value_mem_cache = (T*)(output_tensors->at(2).data);

    const int batch_size = input_tensors->at(0).shape[0];
    const int mem_max_seq_len = encoder_output_tensor.shape[1];
    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          hidden_units_,  // n
                          batch_size,
                          d_model_,  // k
                          attention_weights->query_weight.kernel,
                          hidden_units_,  // n
                          attention_input,
                          d_model_,  // k
                          q_buf_,
                          hidden_units_ /* n */);

    if (step == 1) {
        if (is_batch_major_cache_) {
            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  hidden_units_,
                                  batch_size * mem_max_seq_len,
                                  encoder_output_tensor.shape[2],
                                  attention_weights->key_weight.kernel,
                                  hidden_units_,
                                  encoder_output_tensor.data,
                                  encoder_output_tensor.shape[2],
                                  mem_cache_buf_,
                                  hidden_units_);
            transpose_4d_batch_major_memory_kernelLauncher<T>(
                key_mem_cache, mem_cache_buf_, batch_size, mem_max_seq_len, size_per_head_, head_num_, true, stream_);
            sync_check_cuda_error();

            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  hidden_units_,
                                  batch_size * mem_max_seq_len,
                                  encoder_output_tensor.shape[2],
                                  attention_weights->value_weight.kernel,
                                  hidden_units_,
                                  encoder_output_tensor.data,
                                  encoder_output_tensor.shape[2],
                                  mem_cache_buf_,
                                  hidden_units_);
            transpose_4d_batch_major_memory_kernelLauncher<T>(value_mem_cache,
                                                              mem_cache_buf_,
                                                              batch_size,
                                                              mem_max_seq_len,
                                                              size_per_head_,
                                                              head_num_,
                                                              false,
                                                              stream_);
            sync_check_cuda_error();
        }
        else {
            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  hidden_units_,
                                  batch_size * mem_max_seq_len,
                                  encoder_output_tensor.shape[2],
                                  attention_weights->key_weight.kernel,
                                  hidden_units_,
                                  encoder_output_tensor.data,
                                  encoder_output_tensor.shape[2],
                                  key_mem_cache,
                                  hidden_units_);

            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  hidden_units_,
                                  batch_size * mem_max_seq_len,
                                  encoder_output_tensor.shape[2],
                                  attention_weights->value_weight.kernel,
                                  hidden_units_,
                                  encoder_output_tensor.data,
                                  encoder_output_tensor.shape[2],
                                  value_mem_cache,
                                  hidden_units_);
        }
    }
    sync_check_cuda_error();

    cross_attention_dispatch<T>(q_buf_,
                                attention_weights->query_weight.bias,
                                key_mem_cache,
                                attention_weights->key_weight.bias,
                                value_mem_cache,
                                attention_weights->value_weight.bias,
                                memory_sequence_length,
                                context_buf_,
                                finished,
                                batch_size,
                                batch_size,
                                head_num_,
                                size_per_head_,
                                step,
                                mem_max_seq_len,
                                is_batch_major_cache_,
                                q_scaling_,
                                stream_);
    sync_check_cuda_error();
    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          d_model_,  // n
                          batch_size,
                          hidden_units_,  // k
                          attention_weights->attention_output_weight.kernel,
                          d_model_,  // n
                          context_buf_,
                          hidden_units_,  // k
                          attention_out,
                          d_model_ /* n */);
    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template class DecoderCrossAttentionLayer<float>;
template class DecoderCrossAttentionLayer<half>;

}  // namespace fastertransformer