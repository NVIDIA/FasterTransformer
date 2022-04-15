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

#include "bert_preprocess_kernels.h"

namespace fastertransformer {

__global__ void getPaddingOffsetKernel(size_t* valid_word_num,
                                       int* tmp_mask_offset,
                                       const int* sequence_length,
                                       const int batch_size,
                                       const int max_seq_len)
{
    // do cumulated sum
    int total_seq_len = 0;
    int cum_offset = 0;
    int index = 0;
    for (int i = 0; i < batch_size; i++) {
        const int seq_len = sequence_length[i];
        for (int j = 0; j < seq_len; j++) {
            tmp_mask_offset[index] = cum_offset;
            index++;
        }
        cum_offset += max_seq_len - seq_len;
        total_seq_len += seq_len;
    }
    valid_word_num[0] = (size_t)total_seq_len;
}

void invokeGetPaddingOffset(size_t* h_token_num,
                            size_t* d_token_num,
                            int* tmp_mask_offset,
                            const int* sequence_lengths,
                            const int batch_size,
                            const int max_seq_len,
                            cudaStream_t stream)
{
    getPaddingOffsetKernel<<<1, 1, 0, stream>>>(
        d_token_num, tmp_mask_offset, sequence_lengths, batch_size, max_seq_len);
    sync_check_cuda_error();
    check_cuda_error(cudaMemcpyAsync(h_token_num, d_token_num, sizeof(size_t), cudaMemcpyDeviceToHost, stream));
    sync_check_cuda_error();
}

template<typename T>
__global__ void buildEncoderAttentionMaskKernel(T* attention_mask, const int* sequence_lengths, const int max_seq_len)
{
    // sequence_lengths: [batch_size]
    // attention_mask: [batch_size, 1, max_seq_len, max_seq_len]
    attention_mask += blockIdx.x * max_seq_len * max_seq_len;
    const int length = sequence_lengths[blockIdx.x];
    for (int i = threadIdx.x; i < max_seq_len * max_seq_len; i += blockDim.x) {
        // int row_id = i / max_seq_len;
        int col_id = i % max_seq_len;
        // if (row_id < length && col_id < length) {
        // TODO (bhsueh) check this modification is ok or not on other rmodel
        if (col_id < length) {
            attention_mask[i] = (T)(1.0f);
        }
        else {
            attention_mask[i] = (T)(0.0f);
        }
    }
}

template<typename T>
void invokeBuildEncoderAttentionMask(
    T* attention_mask, const int* sequence_lengths, const int batch_size, const int max_seq_len, cudaStream_t stream)
{
    buildEncoderAttentionMaskKernel<<<batch_size, 256, 0, stream>>>(attention_mask, sequence_lengths, max_seq_len);
}

template void invokeBuildEncoderAttentionMask(float* attention_mask,
                                              const int* sequence_lengths,
                                              const int batch_size,
                                              const int max_seq_len,
                                              cudaStream_t stream);
template void invokeBuildEncoderAttentionMask(half* attention_mask,
                                              const int* sequence_lengths,
                                              const int batch_size,
                                              const int max_seq_len,
                                              cudaStream_t stream);

__global__ void getTrtPaddingOffsetKernel(int* trt_mha_padding_offset, const int* sequence_length, const int batch_size)
{
    // use for get tensorrt fused mha padding offset
    // when we remove the padding

    extern __shared__ int tmp_offset[];
    if (threadIdx.x == 0) {
        tmp_offset[0] = 0;
        for (int i = 0; i < batch_size; i++) {
            tmp_offset[i + 1] = tmp_offset[i] + sequence_length[i];
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < batch_size + 1; i += blockDim.x) {
        trt_mha_padding_offset[i] = tmp_offset[i];
    }
}

void invokeGetTrtPaddingOffset(int* trt_mha_padding_offset,
                               const int* sequence_length,
                               const int batch_size,
                               cudaStream_t stream)
{
    getTrtPaddingOffsetKernel<<<1, 256, sizeof(int) * (batch_size + 1), stream>>>(
        trt_mha_padding_offset, sequence_length, batch_size);
}

__global__ void getTrtPaddingOffsetKernel(int* trt_mha_padding_offset,
                                          const int* sequence_length,
                                          const int request_batch_size,
                                          const int request_seq_len)
{
    // use for get tensorrt fused mha padding offset
    // when we keep the padding

    extern __shared__ int tmp_offset[];
    if (threadIdx.x == 0) {
        tmp_offset[0] = 0;
        for (int i = 0; i < request_batch_size; i++) {
            tmp_offset[i * 2 + 1] = tmp_offset[i * 2] + sequence_length[i];
            tmp_offset[i * 2 + 2] = request_seq_len * (i + 1);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 2 * request_batch_size + 1; i += blockDim.x) {
        trt_mha_padding_offset[i] = tmp_offset[i];
    }
}

void invokeGetTrtPaddingOffset(int* trt_mha_padding_offset,
                               const int* sequence_length,
                               const int request_batch_size,
                               const int request_seq_len,
                               cudaStream_t stream)
{
    getTrtPaddingOffsetKernel<<<1, 256, sizeof(int) * (2 * request_batch_size + 1), stream>>>(
        trt_mha_padding_offset, sequence_length, request_batch_size, request_seq_len);
}

template<typename T>
__global__ void rebuild_sequence_length_padding(const T* src, T* dst, const int* padding_offset, const int n)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int dst_seq_id = bid + padding_offset[bid];
    const int src_seq_id = bid;

    for (int i = tid; i < n; i += blockDim.x) {
        dst[dst_seq_id * n + i] = src[src_seq_id * n + i];
    }
}

template<typename T>
void invokeRebuildPadding(
    T* dst, const T* src, const int* padding_offset, const int m, const int n, cudaStream_t stream)
{
    // src: [token_num, hidden_dim]
    // dst: [batch_size*max_seq_len, hidden_dim]
    rebuild_sequence_length_padding<<<m, 256, 0, stream>>>(src, dst, padding_offset, n);
}

template<typename T>
void invokeRebuildPadding(
    T* dst, const T* src, const int* padding_offset, const int token_num, const int hidden_dim, cudaStream_t stream);
template void invokeRebuildPadding(float* dst,
                                   const float* src,
                                   const int* padding_offset,
                                   const int token_num,
                                   const int hidden_dim,
                                   cudaStream_t stream);
template void invokeRebuildPadding(half* dst,
                                   const half* src,
                                   const int* padding_offset,
                                   const int token_num,
                                   const int hidden_dim,
                                   cudaStream_t stream);

template<typename T>
__global__ void remove_padding(T* tgt, const T* src, const int* padding_offset, const int n)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int src_seq_id = bid + padding_offset[bid];
    const int tgt_seq_id = bid;

    for (int i = tid; i < n; i += blockDim.x) {
        tgt[tgt_seq_id * n + i] = src[src_seq_id * n + i];
    }
}

template<typename T>
void invokeRemovePadding(
    T* dst, const T* src, const int* padding_offset, const int token_num, const int hidden_dim, cudaStream_t stream)
{
    remove_padding<<<token_num, 256, 0, stream>>>(dst, src, padding_offset, hidden_dim);
}

template void invokeRemovePadding(float* dst,
                                  const float* src,
                                  const int* padding_offset,
                                  const int token_num,
                                  const int hidden_dim,
                                  cudaStream_t stream);

template void invokeRemovePadding(half* dst,
                                  const half* src,
                                  const int* padding_offset,
                                  const int token_num,
                                  const int hidden_dim,
                                  cudaStream_t stream);

template<typename T>
__global__ void buildRelativeAttentionBias(T* relative_attention_bias,
                                           const T* relative_attention_bias_table,
                                           const int head_num,
                                           const int seq_len,
                                           const int num_bucket,
                                           const bool is_bidirectional,
                                           const int max_distance)
{

    const int head_id = blockIdx.x;
    for (int seq_id = threadIdx.x; seq_id < seq_len * seq_len; seq_id += blockDim.x) {
        int row_id = seq_id / seq_len;
        int col_id = seq_id % seq_len;

        int relative_position = col_id - row_id;

        int relative_buckets = 0;
        int tmp_num_bucket = num_bucket;
        if (is_bidirectional) {
            tmp_num_bucket /= 2;
            if (relative_position > 0) {
                relative_buckets += tmp_num_bucket;
            }
            else {
                relative_position *= -1;
            }
        }
        else {
            relative_position = abs(relative_position);
        }

        int max_exact = tmp_num_bucket / 2;
        bool is_small = relative_position < max_exact;

        int relative_position_if_large =
            max_exact
            + (int)(logf(relative_position * 1.0f / max_exact) / logf((float)max_distance / max_exact)
                    * (tmp_num_bucket - max_exact));

        relative_position_if_large = min(relative_position_if_large, tmp_num_bucket - 1);

        relative_buckets += is_small ? relative_position : relative_position_if_large;

        relative_attention_bias[head_id * seq_len * seq_len + seq_id] =
            relative_attention_bias_table[head_id * num_bucket + relative_buckets];
    }
}

template<typename T>
void invokeBuildRelativeAttentionBias(T* relative_attention_bias,
                                      const T* relative_attention_bias_table,
                                      const int head_num,
                                      const int seq_len,
                                      const int num_bucket,
                                      const bool is_bidirectional,
                                      const int max_distance,
                                      const PositionEmbeddingType position_embedding_type,
                                      cudaStream_t stream)
{
    if (position_embedding_type == PositionEmbeddingType::absolute) {
        return;
    }
    dim3 grid(head_num);
    dim3 block(256);
    buildRelativeAttentionBias<<<grid, block, 0, stream>>>(relative_attention_bias,
                                                           relative_attention_bias_table,
                                                           head_num,
                                                           seq_len,
                                                           num_bucket,
                                                           is_bidirectional,
                                                           max_distance);
}

template void invokeBuildRelativeAttentionBias(float* relative_attention_bias,
                                               const float* relative_attention_bias_table,
                                               const int head_num,
                                               const int seq_len,
                                               const int num_bucket,
                                               const bool is_bidirectional,
                                               const int max_distance,
                                               const PositionEmbeddingType position_embedding_type,
                                               cudaStream_t stream);

template void invokeBuildRelativeAttentionBias(half* relative_attention_bias,
                                               const half* relative_attention_bias_table,
                                               const int head_num,
                                               const int seq_len,
                                               const int num_bucket,
                                               const bool is_bidirectional,
                                               const int max_distance,
                                               const PositionEmbeddingType position_embedding_type,
                                               cudaStream_t stream);

}  // namespace fastertransformer