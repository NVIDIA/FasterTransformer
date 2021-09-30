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

#include "src/fastertransformer/kernels/gpt_kernels.h"

namespace fastertransformer {

template<typename T>
__global__ void start_id_embedding_position_lookups_kernel(T* from_tensor,
                                                           int* output_ids,
                                                           const T* embedding_table,
                                                           const T* pos_table,
                                                           const int* input_ids,
                                                           const int start_step,
                                                           const int length,
                                                           const int max_length,
                                                           const int batch_size,
                                                           const int hidden_units)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * length * hidden_units;
         index += blockDim.x * gridDim.x) {
        // transpose the input_ids [batch, length] (part of [batch, max_length]) to output_ids [length, batch]
        if (index < batch_size * max_length) {
            const int seq_id = index % max_length;
            const int batch_id = index / max_length;
            if (seq_id < length)
                output_ids[seq_id * batch_size + batch_id] = input_ids[index];
            // output_ids[index] = input_ids[index];
        }

        // embedding lookup from word ids [batch, length] (part of [batch, max_length]) and [vocab, hidden] to generate
        // embedding [batch, length, hidden]
        const int word_index = index / hidden_units;
        const int word_index_row = word_index / length;
        const int word_index_col = word_index % length;
        const int real_word_index = word_index_row * max_length + word_index_col;
        const int step = start_step + word_index % length;
        const int col_index = index % hidden_units;
        T embedding = embedding_table[input_ids[real_word_index] * hidden_units + col_index];
        T pos_embed = pos_table==nullptr? (T)0 : pos_table[(step - 1) * hidden_units + col_index];
        from_tensor[index] = embedding + pos_embed;
    }
}

template<typename T>
void invokeInputIdsEmbeddingLookupPosEncoding(T* from_tensor,
                                              int* output_ids,
                                              const T* embedding_table,
                                              const T* pos_table,
                                              const int* input_ids,
                                              const int start_step,
                                              const int length,
                                              const int max_length,
                                              const int batch_size,
                                              const int hidden_units,
                                              cudaStream_t stream)
{
    dim3 grid(min(batch_size * length, 65536));
    dim3 block(min(hidden_units, 512));
    start_id_embedding_position_lookups_kernel<T><<<grid, block, 0, stream>>>(from_tensor,
                                                                              output_ids,
                                                                              embedding_table,
                                                                              pos_table,
                                                                              input_ids,
                                                                              start_step,
                                                                              length,
                                                                              max_length,
                                                                              batch_size,
                                                                              hidden_units);
}

template void invokeInputIdsEmbeddingLookupPosEncoding(float* from_tensor,
                                                       int* output_ids,
                                                       const float* embedding_table,
                                                       const float* pos_table,
                                                       const int* input_ids,
                                                       const int start_step,
                                                       const int length,
                                                       const int max_length,
                                                       const int batch_size,
                                                       const int hidden_units,
                                                       cudaStream_t stream);

template void invokeInputIdsEmbeddingLookupPosEncoding(half* from_tensor,
                                                       int* output_ids,
                                                       const half* embedding_table,
                                                       const half* pos_table,
                                                       const int* input_ids,
                                                       const int start_step,
                                                       const int length,
                                                       const int max_length,
                                                       const int batch_size,
                                                       const int hidden_units,
                                                       cudaStream_t stream);

// TODO Add half2 implementation
template<typename T>
__global__ void transposeAxis01(T* out, T* in, const int dim0, const int dim1, const int dim2)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < dim0 * dim1 * dim2) {
        const int input_dim2_index = index % dim2;
        index = (index - input_dim2_index) / dim2;
        const int input_dim1_index = index % dim1;
        index = (index - input_dim1_index) / dim1;
        const int input_dim0_index = index % dim0;

        out[input_dim1_index * dim0 * dim2 + input_dim0_index * dim2 + input_dim2_index] =
            in[input_dim0_index * dim1 * dim2 + input_dim1_index * dim2 + input_dim2_index];
    }
}

template<typename T>
void invokeTransposeAxis01(T* out, T* in, const int dim0, const int dim1, const int dim2, cudaStream_t stream)
{
    dim3 block(512);
    dim3 grid((int)(ceil(dim0 * dim1 * dim2 / 512.)));
    transposeAxis01<<<grid, block, 0, stream>>>(out, in, dim0, dim1, dim2);
}

template void
invokeTransposeAxis01(float* out, float* in, const int dim0, const int dim1, const int dim2, cudaStream_t stream);

template void
invokeTransposeAxis01(half* out, half* in, const int dim0, const int dim1, const int dim2, cudaStream_t stream);

template void
invokeTransposeAxis01(int* out, int* in, const int dim0, const int dim1, const int dim2, cudaStream_t stream);

template<typename T>
__global__ void buildDecoderAttentionMaskKernel(T* attention_mask, const int* sequence_lengths, const int max_seq_len)
{
    // sequence_lengths: [batch_size]
    // attention_mask: [batch_size, 1, max_seq_len, max_seq_len]
    attention_mask += blockIdx.x * max_seq_len * max_seq_len;
    const int length = sequence_lengths[blockIdx.x];
    for (int i = threadIdx.x; i < max_seq_len * max_seq_len; i += blockDim.x) {
        int row_id = i / max_seq_len;
        int col_id = i % max_seq_len;
        if (row_id < length && col_id <= row_id)
            attention_mask[i] = (T)(1.0f);
        else
            attention_mask[i] = (T)(0.0f);
    }
}

template<typename T>
void invokeBuildDecoderAttentionMask(
    T* attention_mask, const int* sequence_lengths, const int batch_size, const int max_seq_len, cudaStream_t stream)
{
    buildDecoderAttentionMaskKernel<<<batch_size, 256, 0, stream>>>(attention_mask, sequence_lengths, max_seq_len);
}

template void invokeBuildDecoderAttentionMask(float* attention_mask,
                                              const int* sequence_lengths,
                                              const int batch_size,
                                              const int max_seq_len,
                                              cudaStream_t stream);
template void invokeBuildDecoderAttentionMask(half* attention_mask,
                                              const int* sequence_lengths,
                                              const int batch_size,
                                              const int max_seq_len,
                                              cudaStream_t stream);

template<typename T>
__launch_bounds__(1024, 1) __global__ void lookupHiddenStateOfLastToken(T* from_tensor,
                                                                        const T* hidden_state,
                                                                        const int* input_lengths,
                                                                        const int max_input_length,
                                                                        const int batch_size,
                                                                        const int hidden_units)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * hidden_units;
         index += blockDim.x * gridDim.x) {
        const int col_index = index % hidden_units;
        const int batch_id = index / hidden_units;
        from_tensor[index] = hidden_state[batch_id * max_input_length * hidden_units
                                          + (input_lengths[batch_id] - 1) * hidden_units + col_index];
    }
}

template<typename T>
void invokeLookupHiddenStateOfLastToken(T* from_tensor,
                                        const T* hidden_state,
                                        const int* input_lengths,
                                        const int max_input_length,
                                        const int batch_size,
                                        const int hidden_units,
                                        cudaStream_t stream)
{
    const int grid_size = (int)(ceil(batch_size * hidden_units / 1024.));
    dim3 grid(min(grid_size, 65536));
    dim3 block(min(hidden_units, 1024));
    lookupHiddenStateOfLastToken<T><<<grid, block, 0, stream>>>(
        from_tensor, hidden_state, input_lengths, max_input_length, batch_size, hidden_units);
}

template void invokeLookupHiddenStateOfLastToken(float* from_tensor,
                                                 const float* hidden_state,
                                                 const int* input_lengths,
                                                 const int max_input_length,
                                                 const int batch_size,
                                                 const int hidden_units,
                                                 cudaStream_t stream);

template void invokeLookupHiddenStateOfLastToken(half* from_tensor,
                                                 const half* hidden_state,
                                                 const int* input_lengths,
                                                 const int max_input_length,
                                                 const int batch_size,
                                                 const int hidden_units,
                                                 cudaStream_t stream);

}  // namespace fastertransformer