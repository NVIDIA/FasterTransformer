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

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"
#include "src/fastertransformer/kernels/sampling_topk_kernels.h"

namespace fastertransformer {

__global__ void curandInitialize(curandState_t* state, const int size, const unsigned long long random_seed)
{
    if (threadIdx.x + blockIdx.x * blockDim.x < size) {
        curand_init(random_seed, 0, 0, &state[blockIdx.x * blockDim.x + threadIdx.x]);
    }
}

void invokeCurandInitialize(curandState_t* state,
                            const size_t batch_size,
                            const unsigned long long random_seed,
                            cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((int)(ceil(batch_size * 1.0 / 256)));
    curandInitialize<<<grid, block, 0, stream>>>(state, batch_size, random_seed);
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
                const int end_id = end_ids[row_id];
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

template<typename T>
__global__ void addBiasEndMask(T* logits, const T* bias, const int* end_ids, const bool* finished, const int n)
{
    int bid = blockIdx.x;
    bool finish = finished != nullptr ? finished[bid] : false;
    int offset = bid * n;

    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
    for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
        if (finish) {
            logits[offset + tid] = (tid == end_ids[bid]) ? MAX_T_VAL : -MAX_T_VAL;
        }
        else {
            if (bias != nullptr) {
                logits[offset + tid] += bias[tid];
            }
        }
    }
}

template<typename T>
void invokeAddBiasEndMask(
    T* logits, const T* bias, const int* end_ids, const bool* finished, const int m, const int n, cudaStream_t stream)
{
    dim3 grid(m);
    dim3 block(min(n, 1024));
    /*n is the vocab_size, e.g., 30000, 7000.... vocab_size is usually very big. */
    addBiasEndMask<<<grid, block, 0, stream>>>(logits, bias, end_ids, finished, n);
}

template void invokeAddBiasEndMask(float* logits,
                                   const float* bias,
                                   const int* end_ids,
                                   const bool* finished,
                                   const int m,
                                   const int n,
                                   cudaStream_t stream);

template void invokeAddBiasEndMask(half* logits,
                                   const half* bias,
                                   const int* end_ids,
                                   const bool* finished,
                                   const int m,
                                   const int n,
                                   cudaStream_t stream);

template<typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topk_topp_stage_2_opt3_sampling(const int* __restrict topk_tmp_id_buf,
                                                T* topk_tmp_val_buf,
                                                T* topk_tmp2_val_buf,
                                                int* ids,
                                                int* sequence_length,
                                                bool* finished_buf,
                                                float* cum_log_probs,
                                                float* output_log_probs,
                                                const int k,
                                                const T prob_threshold,
                                                curandState_t* curandstate,
                                                const int* end_ids,
                                                const int vocab_size)
{
    const int size = k * BLOCKS_PER_BEAM_;
    const int tid = threadIdx.x;
    const int batch_id = blockIdx.x;
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    typedef cub::BlockReduce<TopK_2<float>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    extern __shared__ char array[];
    __shared__ float rand_num;
    __shared__ float s_sum;
    __shared__ float s_max;
    T* s_val = topk_tmp_val_buf + batch_id * size;
    int* s_id = (int*)(array);
    s_max = 0.0f;
    s_sum = 0.0f;
    TopK_2<float> partial;

    if (finished_buf != nullptr && finished_buf[batch_id] == true) {
        ids[batch_id] = end_ids[batch_id];
        return;
    }

    for (int index = tid; index < size; index += BLOCK_SIZE_) {
        topk_tmp2_val_buf[batch_id * size + index] = topk_tmp_val_buf[batch_id * size + index];
    }
    __syncthreads();
    float* s_val2 = reinterpret_cast<float*>(s_id + k);

    for (int ite = 0; ite < k; ite++) {
        partial.init();
#pragma unroll
        for (int i = tid; i < size; i += BLOCK_SIZE_) {
            partial.insert((float)s_val[i], i);
        }

        TopK_2<float> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<float>);

        if (ite == 0) {
            s_max = total.u;
        }

        if (tid == 0) {
            s_id[ite] = total.p;
            s_val[total.p] = -MAX_T_VAL;

            // when cum_log_probs are computed, topk_tmp_val_buf (logits_buf_) are already pre-processed by
            // softmax_kernel
            if (cum_log_probs == nullptr && output_log_probs == nullptr) {
                total.u = __expf(total.u - s_max);
            }
            s_val2[ite] = total.u;
            s_sum += total.u;
        }
        __syncthreads();
    }
    if (tid == 0) {
        rand_num = (float)curand_uniform(curandstate + blockIdx.x) * (float)prob_threshold * s_sum;
        for (int i = 0; i < k; i++) {
            float exp_logit = s_val2[i];
            rand_num = rand_num - exp_logit;
            if (rand_num <= 0.0f || i == k - 1) {
                ids[batch_id] = topk_tmp_id_buf[batch_id * size + s_id[i]] % vocab_size;
                if (cum_log_probs != nullptr || output_log_probs != nullptr) {
                    float log_prob = logf(exp_logit);
                    if (cum_log_probs != nullptr) {
                        cum_log_probs[batch_id] += log_prob;
                    }
                    if (output_log_probs != nullptr) {
                        // 'output_log_probs' is the probability induced by the top-k sampling.
                        // We normalize the probability 'exp_logit' of the selected token by
                        // the probability 's_sum' of a set of top-k tokens, meaning the log_prob
                        // is the probability of the selected token, conditioned on the event that
                        // it is selected, i.e.,
                        //   log_prob = log P(i | i is in top-k) = log(exp_logit / s_sum).
                        output_log_probs[batch_id] = log_prob - logf(s_sum);
                    }
                }
                break;
            }
        }
        if (sequence_length != nullptr && finished_buf != nullptr) {
            sequence_length[batch_id] =
                finished_buf[batch_id] ? sequence_length[batch_id] : sequence_length[batch_id] + 1;
            finished_buf[batch_id] = ids[batch_id] == end_ids[batch_id] ? 1 : 0;
        }
    }
}

#define CASE_K(K_MIN, K_MAX, BLOCK_SIZE_1_, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_)                                           \
    case K_MIN ... K_MAX:                                                                                              \
        topk_stage_1_opt3<T, BLOCK_SIZE_1_, BLOCKS_PER_BEAM_>                                                          \
            <<<batch_size * BLOCKS_PER_BEAM_, BLOCK_SIZE_1_, 0, stream>>>(log_probs,                                   \
                                                                          temp_log_probs,                              \
                                                                          topk_tmp_id_buf,                             \
                                                                          topk_tmp_val_buf,                            \
                                                                          finished_buf,                                \
                                                                          candidate_num,                               \
                                                                          vocab_size,                                  \
                                                                          end_ids);                                    \
        topk_topp_stage_2_opt3_sampling<T, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_>                                            \
            <<<batch_size, BLOCK_SIZE_2_, K_MAX * sizeof(int) + K_MAX * sizeof(float), stream>>>(topk_tmp_id_buf,      \
                                                                                                 topk_tmp_val_buf,     \
                                                                                                 topk_tmp2_val_buf,    \
                                                                                                 ids,                  \
                                                                                                 sequence_length,      \
                                                                                                 finished_buf,         \
                                                                                                 cum_log_probs,        \
                                                                                                 output_log_probs,     \
                                                                                                 candidate_num,        \
                                                                                                 1.0f,                 \
                                                                                                 curandstate,          \
                                                                                                 end_ids,              \
                                                                                                 vocab_size);          \
        break;

template<typename T>
void invokeTopKSampling(void* workspace,
                        size_t& workspace_size,
                        T* log_probs,
                        int* ids,
                        int* sequence_length,
                        bool* finished_buf,
                        float* cum_log_probs,
                        float* output_log_probs,
                        curandState_t* curandstate,
                        const int top_k,
                        const int vocab_size_padded,
                        const int* end_ids,
                        cudaStream_t stream,
                        const int batch_size)
{
    // Here, we put batch size as an argument because the batch size of initialization
    // and inference may be different due to pipelint parallelism.
    const int candidate_num = top_k;
    const int vocab_size = vocab_size_padded;

    const int max_block_per_beam = 8;
    int temp_log_probs_buf_size = batch_size * vocab_size;                        // type float
    int topk_tmp_ids_buf_size = batch_size * candidate_num * max_block_per_beam;  // type int
    int topk_tmp_val_buf_size = batch_size * candidate_num * max_block_per_beam;  // type float

    // prevent memory misalinged address
    temp_log_probs_buf_size = (int)(ceil(temp_log_probs_buf_size / 4.)) * 4;
    topk_tmp_ids_buf_size = (int)(ceil(topk_tmp_ids_buf_size / 4.)) * 4;
    topk_tmp_val_buf_size = (int)(ceil(topk_tmp_val_buf_size / 4.)) * 4;

    if (workspace == nullptr) {
        workspace_size = sizeof(T) * temp_log_probs_buf_size + sizeof(int) * topk_tmp_ids_buf_size
                         + 2 * sizeof(T) * topk_tmp_val_buf_size;
        return;
    }
    else {
        T* temp_log_probs = (T*)workspace;
        int* topk_tmp_id_buf = (int*)(temp_log_probs + temp_log_probs_buf_size);
        T* topk_tmp_val_buf = (T*)(topk_tmp_id_buf + topk_tmp_ids_buf_size);
        T* topk_tmp2_val_buf = (T*)(topk_tmp_val_buf + topk_tmp_val_buf_size);

        switch (candidate_num) {
            CASE_K(1, 16, 128, 128, 8);
            CASE_K(17, 32, 256, 128, 8);
            CASE_K(33, 64, 256, 256, 8);
            default:
                printf("[ERROR] Topk kernel does not support candidate_num = %d \n", candidate_num);
                exit(0);
                break;
        }
        return;
    }
}

#undef CASE_K

template void invokeTopKSampling(void* workspace,
                                 size_t& workspace_size,
                                 float* log_probs,
                                 int* ids,
                                 int* sequence_length,
                                 bool* finished_buf,
                                 float* cum_log_probs,
                                 float* output_log_probs,
                                 curandState_t* curandstate,
                                 const int top_k,
                                 const int vocab_size_padded,
                                 const int* end_ids,
                                 cudaStream_t stream,
                                 const int batch_size);

template void invokeTopKSampling(void* workspace,
                                 size_t& workspace_size,
                                 half* log_probs,
                                 int* ids,
                                 int* sequence_length,
                                 bool* finished_buf,
                                 float* cum_log_probs,
                                 float* output_log_probs,
                                 curandState_t* curandstate,
                                 const int top_k,
                                 const int vocab_size_padded,
                                 const int* end_ids,
                                 cudaStream_t stream,
                                 const int batch_size);

#define CASE_K(K_MIN, K_MAX, BLOCK_SIZE_1_, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_)                                           \
    case K_MIN ... K_MAX:                                                                                              \
        topk_stage_1_opt3<T, BLOCK_SIZE_1_, BLOCKS_PER_BEAM_>                                                          \
            <<<batch_size * BLOCKS_PER_BEAM_, BLOCK_SIZE_1_, 0, stream>>>(logits,                                      \
                                                                          temp_logits,                                 \
                                                                          topk_tmp_id_buf,                             \
                                                                          topk_tmp_val_buf,                            \
                                                                          finished_buf,                                \
                                                                          candidate_num,                               \
                                                                          vocab_size,                                  \
                                                                          end_ids);                                    \
        topk_topp_stage_2_opt3_sampling<T, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_>                                            \
            <<<batch_size, BLOCK_SIZE_2_, K_MAX * sizeof(int) + K_MAX * sizeof(float), stream>>>(topk_tmp_id_buf,      \
                                                                                                 topk_tmp_val_buf,     \
                                                                                                 topk_tmp2_val_buf,    \
                                                                                                 output_ids,           \
                                                                                                 sequence_length,      \
                                                                                                 finished_buf,         \
                                                                                                 cum_log_probs,        \
                                                                                                 output_log_probs,     \
                                                                                                 candidate_num,        \
                                                                                                 prob_threshold,       \
                                                                                                 curandstate,          \
                                                                                                 end_ids,              \
                                                                                                 vocab_size);          \
        break;

template<typename T>
void invokeTopKTopPSampling(void* workspace,
                            size_t& workspace_size,
                            int* output_ids,
                            const T* logits,
                            int* sequence_length,
                            bool* finished_buf,
                            float* cum_log_probs,
                            float* output_log_probs,
                            curandState_t* curandstate,
                            const int batch_size,
                            const int top_k,
                            const T top_p,
                            const int vocab_size_padded,
                            const int* end_ids,
                            cudaStream_t stream)
{
    // Here, we put batch size as an argument because the batch size of initialization
    // and inference may be different due to pipeline parallelism.
    const int candidate_num = top_k;
    const T prob_threshold = top_p;
    const int vocab_size = vocab_size_padded;

    const int max_block_per_beam = 8;
    int temp_logits_buf_size = batch_size * vocab_size;                           // type T
    int topk_tmp_ids_buf_size = batch_size * candidate_num * max_block_per_beam;  // type int
    int topk_tmp_val_buf_size = batch_size * candidate_num * max_block_per_beam;  // type T

    // prevent memory misalinged address
    temp_logits_buf_size = (int)(ceil(temp_logits_buf_size / 4.)) * 4;
    topk_tmp_ids_buf_size = (int)(ceil(topk_tmp_ids_buf_size / 4.)) * 4;
    topk_tmp_val_buf_size = (int)(ceil(topk_tmp_val_buf_size / 4.)) * 4;

    if (workspace == nullptr) {
        workspace_size = sizeof(T) * temp_logits_buf_size + sizeof(int) * topk_tmp_ids_buf_size
                         + 2 * sizeof(T) * topk_tmp_val_buf_size;
        return;
    }
    else {
        T* temp_logits = (T*)workspace;
        int* topk_tmp_id_buf = (int*)(temp_logits + temp_logits_buf_size);
        T* topk_tmp_val_buf = (T*)(topk_tmp_id_buf + topk_tmp_ids_buf_size);
        T* topk_tmp2_val_buf = (T*)(topk_tmp_val_buf + topk_tmp_val_buf_size);

        switch (candidate_num) {
            CASE_K(1, 16, 128, 128, 8);
            CASE_K(17, 32, 256, 128, 8);
            CASE_K(33, 64, 256, 256, 8);
            default:
                printf("[ERROR] Topk kernel does not support candidate_num = %d \n", candidate_num);
                exit(0);
                break;
        }
        return;
    }
}

template void invokeTopKTopPSampling(void* workspace,
                                     size_t& workspace_size,
                                     int* output_ids,
                                     const float* logits,
                                     int* sequence_length,
                                     bool* finished_buf,
                                     float* cum_log_probs,
                                     float* output_log_probs,
                                     curandState_t* curandstate,
                                     const int batch_size,
                                     const int top_k,
                                     const float top_p,
                                     const int vocab_size_padded,
                                     const int* end_ids,
                                     cudaStream_t stream);

template void invokeTopKTopPSampling(void* workspace,
                                     size_t& workspace_size,
                                     int* output_ids,
                                     const half* logits,
                                     int* sequence_length,
                                     bool* finished_buf,
                                     float* cum_log_probs,
                                     float* output_log_probs,
                                     curandState_t* curandstate,
                                     const int batch_size,
                                     const int top_k,
                                     const half top_p,
                                     const int vocab_size_padded,
                                     const int* end_ids,
                                     cudaStream_t stream);
}  // namespace fastertransformer
