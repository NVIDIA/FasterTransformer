/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
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
            if (seq_id < length) {
                output_ids[seq_id * batch_size + batch_id] = input_ids[index];
            }
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
        T pos_embed = pos_table == nullptr ? (T)0.f : pos_table[(step - 1) * hidden_units + col_index];
        from_tensor[index] = embedding + pos_embed;
    }
}

template<typename T>
__global__ void start_id_embedding_position_lookups_kernel(T* from_tensor,
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
        // embedding lookup from word ids [batch, length] (part of [batch, max_length]) and [vocab, hidden] to generate
        // embedding [batch, length, hidden]
        const int word_index = index / hidden_units;
        const int word_index_row = word_index / length;
        const int word_index_col = word_index % length;
        const int real_word_index = word_index_row * max_length + word_index_col;
        const int step = start_step + word_index % length;
        const int col_index = index % hidden_units;
        T embedding = embedding_table[input_ids[real_word_index] * hidden_units + col_index];
        T pos_embed = pos_table == nullptr ? (T)0.f : pos_table[(step - 1) * hidden_units + col_index];
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
    if (output_ids == nullptr) {
        start_id_embedding_position_lookups_kernel<T><<<grid, block, 0, stream>>>(from_tensor,
                                                                                  embedding_table,
                                                                                  pos_table,
                                                                                  input_ids,
                                                                                  start_step,
                                                                                  length,
                                                                                  max_length,
                                                                                  batch_size,
                                                                                  hidden_units);
    }
    else {
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

#ifdef ENABLE_BF16
template void invokeInputIdsEmbeddingLookupPosEncoding(__nv_bfloat16* from_tensor,
                                                       int* output_ids,
                                                       const __nv_bfloat16* embedding_table,
                                                       const __nv_bfloat16* pos_table,
                                                       const int* input_ids,
                                                       const int start_step,
                                                       const int length,
                                                       const int max_length,
                                                       const int batch_size,
                                                       const int hidden_units,
                                                       cudaStream_t stream);
#endif

template<typename T>
__global__ void inputIdsEmbeddingLookupPosEncodingSoftPrompt(inputIdsEmbeddingLookupPosEncodingSoftPromptParam<T> param)
{
    // 1. Copy the input ids to output ids and transpose ouptut ids to [seq_len, batch_size, beam_width].
    // 2. Embedding lookup by input ids and concat with soft prompt. The axis of concatenation is on axis of seq_len.

    // Assume batch size is 2 and prompts are [[t1, t2], [t3], [t4, t5]], input_ids are [[s1, s2], [s3], [s4]]
    // then the order of output_ids is
    // [ [?, ?, s1, s2]
    //   [?, s3, padding, padding]
    //   [?, ?, s4, padding] ]
    // and the order of embedding is
    // [ [t1, t2, s1, s2]
    //   [t3, s3, padding, padding]
    //   [t4, t5, s4, padding] ]
    // where "?" means undefined values and we should attach it.

    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < param.batch_size * param.beam_width * (param.max_prefix_soft_prompt_length + param.max_input_length)
                     * param.hidden_units;
         index += blockDim.x * gridDim.x) {
        // transpose the input_ids [batch, length] (part of [batch, beam, max_input_length]) to
        // output_ids [length, batch, beam].
        // ouptut_ids need to add padding in the beginning for soft prompting.

        if (index < param.batch_size * param.beam_width * param.max_input_length) {
            int tmp_index = index;
            const int seq_id = tmp_index % param.max_input_length;
            tmp_index = (tmp_index - seq_id) / param.max_input_length;
            const int beam_id = tmp_index % param.beam_width;
            tmp_index = (tmp_index - beam_id) / param.beam_width;
            const int batch_id = tmp_index % param.batch_size;
            if (seq_id < param.max_input_length) {
                param.output_ids[(param.prefix_soft_prompt_lengths[batch_id] + seq_id) * param.batch_size
                                     * param.beam_width
                                 + batch_id * param.beam_width + beam_id] = param.input_ids[index];
            }
        }

        // embedding lookup from word ids [batch, beam, length] (part of [batch, beam, max_input_length]), [vocab,
        // hidden] and [batch, max_prefix_soft_prompt_length, hidden] to generate embedding [batch, beam, length +
        // max_prefix_soft_prompt_length, hidden]
        int tmp_index = index;
        const int hidden_id = tmp_index % param.hidden_units;
        tmp_index = (tmp_index - hidden_id) / param.hidden_units;
        const int seq_id = tmp_index % (param.max_prefix_soft_prompt_length + param.max_input_length);
        tmp_index = (tmp_index - seq_id) / (param.max_prefix_soft_prompt_length + param.max_input_length);
        const int beam_id = tmp_index % param.beam_width;
        tmp_index = (tmp_index - beam_id) / param.beam_width;
        const int batch_id = tmp_index % param.batch_size;
        T embedding =
            (seq_id < param.prefix_soft_prompt_lengths[batch_id]) ?
                (T)param
                    .prefix_soft_prompt_embedding[batch_id * param.max_prefix_soft_prompt_length * param.hidden_units
                                                  + seq_id * param.hidden_units + hidden_id] :
                param.embedding_table[param.input_ids[batch_id * param.beam_width * param.max_input_length
                                                      + beam_id * param.max_input_length
                                                      + (seq_id - param.prefix_soft_prompt_lengths[batch_id])]
                                          * param.hidden_units
                                      + hidden_id];

        T pos_embed = param.pos_table == nullptr ?
                          (T)0.0f :
                          param.pos_table[(param.start_step + seq_id - 1) * param.hidden_units + hidden_id];
        param.from_tensor[index] = embedding + pos_embed;

        if (seq_id == 0 && hidden_id == 0) {
            param.input_lengths[batch_id * param.beam_width + beam_id] += param.prefix_soft_prompt_lengths[batch_id];
        }
    }
}

template<typename T>
void invokeInputIdsEmbeddingLookupPosEncodingSoftPrompt(inputIdsEmbeddingLookupPosEncodingSoftPromptParam<T> param)
{
    dim3 grid(min(param.batch_size * param.beam_width * (param.max_input_length + param.max_prefix_soft_prompt_length),
                  65536));
    dim3 block(min(param.hidden_units, 512));
    inputIdsEmbeddingLookupPosEncodingSoftPrompt<T><<<grid, block, 0, param.stream>>>(param);
}

template void
invokeInputIdsEmbeddingLookupPosEncodingSoftPrompt(inputIdsEmbeddingLookupPosEncodingSoftPromptParam<float> param);

template void
invokeInputIdsEmbeddingLookupPosEncodingSoftPrompt(inputIdsEmbeddingLookupPosEncodingSoftPromptParam<half> param);

#ifdef ENABLE_BF16
template void invokeInputIdsEmbeddingLookupPosEncodingSoftPrompt(
    inputIdsEmbeddingLookupPosEncodingSoftPromptParam<__nv_bfloat16> param);
#endif

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
__global__ void transposeAxis01(T* out, T* in, const int* in_skipping_dim1, const int dim0, const int dim1)
{
    // out: [dim1, dim0]
    // in: [dim0, dim1]
    // in_skipping_dim1: [dim1]

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < dim0 * dim1) {
        const int input_dim1_index = index % dim1;
        index = (index - input_dim1_index) / dim1;
        const int input_dim0_index = index % dim0;
        const int in_offset = in_skipping_dim1 == nullptr ? 0 : in_skipping_dim1[input_dim1_index] * dim1;

        out[input_dim1_index * dim0 + input_dim0_index] = in[in_offset + input_dim0_index * dim1 + input_dim1_index];
    }
}

template<typename T>
void invokeTransposeAxis01(
    T* out, T* in, const int* in_skipping_dim1, const int dim0, const int dim1, cudaStream_t stream)
{
    dim3 block(512);
    dim3 grid((int)(ceil(dim0 * dim1 / 512.)));
    transposeAxis01<<<grid, block, 0, stream>>>(out, in, in_skipping_dim1, dim0, dim1);
}

template void invokeTransposeAxis01(
    int* out, int* in, const int* in_skipping_dim1, const int dim0, const int dim1, cudaStream_t stream);

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
        if (row_id < length && col_id <= row_id) {
            attention_mask[i] = (T)(1.0f);
        }
        else {
            attention_mask[i] = (T)(0.0f);
        }
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
#ifdef ENABLE_BF16
template void invokeBuildDecoderAttentionMask(__nv_bfloat16* attention_mask,
                                              const int* sequence_lengths,
                                              const int batch_size,
                                              const int max_seq_len,
                                              cudaStream_t stream);
#endif

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

#ifdef ENABLE_BF16
template void invokeLookupHiddenStateOfLastToken(__nv_bfloat16* from_tensor,
                                                 const __nv_bfloat16* hidden_state,
                                                 const int* input_lengths,
                                                 const int max_input_length,
                                                 const int batch_size,
                                                 const int hidden_units,
                                                 cudaStream_t stream);
#endif

__global__ void tileGptInputs(int* tiled_input_ids,
                              int* tiled_input_lengths,
                              const int* input_ids,
                              const int* input_lengths,
                              const int max_input_length)
{
    if (threadIdx.x == 0) {
        tiled_input_lengths[blockIdx.x * gridDim.y + blockIdx.y] = input_lengths[blockIdx.x];
    }
    for (int index = threadIdx.x; index < max_input_length; index += blockDim.x) {
        tiled_input_ids[(blockIdx.x * gridDim.y + blockIdx.y) * max_input_length + index] =
            input_ids[blockIdx.x * max_input_length + index];
    }
}

void invokeTileGptInputs(int* tiled_input_ids,
                         int* tiled_input_lengths,
                         const int* input_ids,
                         const int* input_lengths,
                         const int batch_size,
                         const int beam_width,
                         const int max_input_length,
                         cudaStream_t stream)
{
    dim3 grid(batch_size, beam_width);
    dim3 block(min(1024, max_input_length));
    tileGptInputs<<<grid, block, 0, stream>>>(
        tiled_input_ids, tiled_input_lengths, input_ids, input_lengths, max_input_length);
}

bool hasDiffRuntimeArgs(const std::unordered_map<std::string, Tensor>* input_tensors)
{
    //      runtime_top_k [1] or [batch_size] on cpu, optional.
    //      runtime_top_p [1] or [batch_size] on cpu, optional
    //      beam_search_diversity_rate [1] or [batch_size] on cpu, optional
    //      temperature [1] or [batch_size] on cpu, optional
    //      len_penalty [1] or [batch_size] on cpu, optional
    //      repetition_penalty [1] or [batch_size] on cpu, optional
    //      random_seed [1] or [batch_size] on cpu, optional

    std::vector<std::string> check_list = {"runtime_top_k",
                                           "runtime_top_p",
                                           "beam_search_diversity_rate",
                                           "temperature",
                                           "len_penalty",
                                           "repetition_penalty",
                                           "random_seed"};

    for (int i = 0; i < (int)check_list.size(); i++) {
        if (input_tensors->count(check_list[i])) {
            auto tensor = input_tensors->at(check_list[i]);
            FT_CHECK(tensor.shape.size() == 1);
            for (int j = 1; j < (int)tensor.shape[0]; j++) {
                const void* data = tensor.data;
                switch (tensor.type) {
                    case TYPE_FP32:
                        if (((const float*)data)[0] != ((const float*)data)[j]) {
                            return true;
                        }
                        break;
                    case TYPE_INT32:
                        if (((const int*)data)[0] != ((const int*)data)[j]) {
                            return true;
                        }
                        break;
                    case TYPE_UINT32:
                        if (((const uint*)data)[0] != ((const uint*)data)[j]) {
                            return true;
                        }
                        break;
                    case TYPE_UINT64:
                        if (((const unsigned long long int*)data)[0] != ((const unsigned long long int*)data)[j]) {
                            return true;
                        }
                        break;
                    default:
                        FT_CHECK_WITH_INFO(false, check_list[i] + ": " + tensor.toString() + " is invalid.");
                        break;
                }
            }
        }
    }
    return false;
}

}  // namespace fastertransformer