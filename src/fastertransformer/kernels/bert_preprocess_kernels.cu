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
        int* tmp_mask_offset, const int* sequence_length, 
        const int batch_size, const int max_seq_len) {
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

void invokeGetPaddingOffset(size_t* h_token_num, size_t* d_token_num,
        int* tmp_mask_offset, const int* sequence_lengths, 
        const int batch_size, const int max_seq_len,
        cudaStream_t stream) {
    getPaddingOffsetKernel<<<1, 1, 0, stream>>>(d_token_num, tmp_mask_offset,
        sequence_lengths, batch_size, max_seq_len);
    sync_check_cuda_error();
    check_cuda_error(cudaMemcpyAsync(h_token_num, d_token_num, sizeof(size_t), cudaMemcpyDeviceToHost, stream));
    sync_check_cuda_error();
}

template <typename T>
__global__ void buildEncoderAttentionMaskKernel(T* attention_mask, const int *sequence_lengths,
        const int max_seq_len){
    // sequence_lengths: [batch_size]
    // attention_mask: [batch_size, 1, max_seq_len, max_seq_len]
    attention_mask += blockIdx.x * max_seq_len * max_seq_len;
    const int length = sequence_lengths[blockIdx.x];
    for (int i = threadIdx.x; i < max_seq_len * max_seq_len; i += blockDim.x) {
        int row_id = i / max_seq_len;
        int col_id = i % max_seq_len;
        if (row_id < length && col_id < length)
            attention_mask[i] = (T)(1.0f);
        else
            attention_mask[i] = (T)(0.0f);
    }
}

template <typename T>
void invokeBuildEncoderAttentionMask(T* attention_mask, const int *sequence_lengths,
        const int batch_size, const int max_seq_len, cudaStream_t stream) {
    buildEncoderAttentionMaskKernel<<<batch_size, 256, 0, stream>>>(
        attention_mask, sequence_lengths, max_seq_len);
}

template void invokeBuildEncoderAttentionMask(float* attention_mask, 
    const int *sequence_lengths, const int batch_size,
    const int max_seq_len, cudaStream_t stream);
template void invokeBuildEncoderAttentionMask(half* attention_mask, 
    const int *sequence_lengths, const int batch_size,
    const int max_seq_len, cudaStream_t stream);

__global__ void getTrtPaddingOffsetKernel(int* trt_mha_padding_offset,
        const int* sequence_length, const int batch_size) {
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
        const int* sequence_length, const int batch_size, cudaStream_t stream) {
    getTrtPaddingOffsetKernel<<<1, 256, sizeof(int) * (batch_size + 1), stream>>>(
        trt_mha_padding_offset, sequence_length, batch_size);
}

__global__ void getTrtPaddingOffsetKernel(int* trt_mha_padding_offset,
        const int* sequence_length, const int request_batch_size,
        const int request_seq_len) {
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
        const int* sequence_length, const int request_batch_size,
        const int request_seq_len, cudaStream_t stream) {
    getTrtPaddingOffsetKernel<<<1, 256, sizeof(int) * (2 * request_batch_size + 1), stream>>>(
        trt_mha_padding_offset, sequence_length, request_batch_size, request_seq_len);
}

template<typename T>
__global__ void rebuild_sequence_length_padding(const T* src, T* dst,
                                            const int* padding_offset,
                                            const int n)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int dst_seq_id = bid + padding_offset[bid];
    const int src_seq_id = bid;

    for(int i = tid; i < n; i += blockDim.x) {
        dst[dst_seq_id * n + i] = src[src_seq_id * n + i];
    }
}

template<typename T>
void invokeRebuildPadding(T* dst, const T* src, const int* padding_offset,
    const int m, const int n, cudaStream_t stream)
{
    // src: [token_num, hidden_dim]
    // dst: [batch_size*max_seq_len, hidden_dim]
    rebuild_sequence_length_padding<<<m, 256, 0, stream>>>(src, dst, padding_offset, n);
}

template <typename T>
void invokeRebuildPadding(T* dst, const T* src, const int* padding_offset,
                        const int token_num, const int hidden_dim, cudaStream_t stream);
template void invokeRebuildPadding(float* dst, const float* src, const int* padding_offset,
                                const int token_num, const int hidden_dim, cudaStream_t stream);
template void invokeRebuildPadding(half* dst, const half* src, const int* padding_offset,
                                const int token_num, const int hidden_dim, cudaStream_t stream);

template<typename T>
__global__ void remove_padding(T* tgt, const T* src,
                            const int* padding_offset, const int n)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int src_seq_id = bid + padding_offset[bid];
    const int tgt_seq_id = bid;


    for(int i = tid; i < n; i += blockDim.x)
    {
        tgt[tgt_seq_id * n + i] = src[src_seq_id * n + i];
    }
}

template <typename T>
void invokeRemovePadding(T* dst, const T* src,
                        const int* padding_offset, const int token_num,
                        const int hidden_dim, cudaStream_t stream) {
    remove_padding<<<token_num, 256, 0, stream>>>(dst, src, 
                                                padding_offset, hidden_dim);
}

template void invokeRemovePadding(float* dst, const float* src,
                                const int* padding_offset, const int token_num,
                                const int hidden_dim, cudaStream_t stream);

template void invokeRemovePadding(half* dst, const half* src,
                                    const int* padding_offset, const int token_num,
                                    const int hidden_dim, cudaStream_t stream);
    
}