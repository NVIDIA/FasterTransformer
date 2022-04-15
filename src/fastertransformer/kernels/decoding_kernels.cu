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

#include "src/fastertransformer/kernels/bfloat16_fallback_kenrels.cuh"
#include "src/fastertransformer/kernels/decoding_kernels.h"
#include "src/fastertransformer/utils/cuda_utils.h"

namespace fastertransformer {

static const float HALF_FLT_MAX = 65504.F;

template<typename T>
__global__ void decodingInitialize(bool* finished,
                                   int* sequence_length,
                                   int* word_ids,
                                   T* cum_log_probs,
                                   const int* sentence_ids,
                                   const int batch_size,
                                   const int beam_width,
                                   const int max_input_length)
{
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? (T)HALF_FLT_MAX : (T)1e20f;  // BF16 and FP32 have the same dynamic range
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * beam_width;
         index += blockDim.x * gridDim.x) {
        finished[index] = false;
        sequence_length[index] = max_input_length;
        if (word_ids != nullptr) {
            word_ids[index] = sentence_ids[index / beam_width];
        }
        cum_log_probs[index] = (index % beam_width == 0) ? (T)0.0f : (T)-MAX_T_VAL;
    }
}

template<typename T>
void invokeDecodingInitialize(bool* finished,
                              int* sequence_length,
                              int* word_ids,
                              T* cum_log_probs,
                              const int* sentence_ids,
                              const int batch_size,
                              const int beam_width,
                              const int max_input_length,
                              cudaStream_t stream)
{
    dim3 grid((int)ceil(batch_size * beam_width * 1.0 / 256));
    dim3 block(256);

    decodingInitialize<T><<<grid, block, 0, stream>>>(
        finished, sequence_length, word_ids, cum_log_probs, sentence_ids, batch_size, beam_width, max_input_length);
}

template void invokeDecodingInitialize(bool* finished,
                                       int* sequence_length,
                                       int* word_ids,
                                       float* cum_log_probs,
                                       const int* sentence_ids,
                                       const int batch_size,
                                       const int beam_width,
                                       const int max_input_length,
                                       cudaStream_t stream);

template void invokeDecodingInitialize(bool* finished,
                                       int* sequence_length,
                                       int* word_ids,
                                       half* cum_log_probs,
                                       const int* sentence_ids,
                                       const int batch_size,
                                       const int beam_width,
                                       const int max_input_length,
                                       cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeDecodingInitialize(bool* finished,
                                       int* sequence_length,
                                       int* word_ids,
                                       __nv_bfloat16* cum_log_probs,
                                       const int* sentence_ids,
                                       const int batch_size,
                                       const int beam_width,
                                       const int max_input_length,
                                       cudaStream_t stream);
#endif

template<typename T>
__global__ void embeddingLookupPosEncoding(T* from_tensor,
                                           const T* embedding_table,
                                           const T* position_encoding,
                                           const int* all_ids,
                                           const int* input_lengths,
                                           const int local_batch_size,
                                           const int hidden_units,
                                           const int step,
                                           const int max_input_length,
                                           const int batch_size,
                                           const int ite,
                                           const T scale)
{
    // 1. lookup from embedding table
    // 2. multiply scale
    // 3. add the position encoding
    const int id_offset = step * batch_size + ite * local_batch_size;

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < local_batch_size * hidden_units;
         index += blockDim.x * gridDim.x) {
        const int row_index = index / hidden_units;
        const int col_index = index % hidden_units;
        const int step_offset = input_lengths == nullptr ?
                                    step * hidden_units :
                                    (step - max_input_length + input_lengths[row_index]) * hidden_units;
        T val = embedding_table[all_ids[id_offset + row_index] * hidden_units + col_index] * scale;
        val = val + position_encoding[step_offset + col_index];

        from_tensor[index] = val;
    }
}

// No aboluste position embedding
template<typename T>
__global__ void embeddingLookup(T* from_tensor,
                                const T* embedding_table,
                                const int* all_ids,
                                const int local_batch_size,
                                const int hidden_units,
                                const int step,
                                const int max_input_length,
                                const int batch_size,
                                const int ite,
                                const T scale)
{
    // 1. lookup from embedding table
    // 2. multiply scale
    const int id_offset = step * batch_size + ite * local_batch_size;

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < local_batch_size * hidden_units;
         index += blockDim.x * gridDim.x) {
        const int row_index = index / hidden_units;
        const int col_index = index % hidden_units;
        T val = embedding_table[all_ids[id_offset + row_index] * hidden_units + col_index] * scale;
        from_tensor[index] = val;
    }
}

template<typename T>
void invokeEmbeddingLookupPosEncoding(T* from_tensor,
                                      const T* embedding_table,
                                      const T* position_encoding,
                                      const int* all_ids,
                                      const int* input_lengths,
                                      const int local_batch_size,
                                      const int hidden_units,
                                      const T scale,
                                      const int step,
                                      const int max_input_length,
                                      const int batch_size,
                                      const int ite,
                                      cudaStream_t stream)
{
    dim3 grid(min(local_batch_size, 65536));
    dim3 block(min(hidden_units, 1024));
    if (position_encoding != nullptr) {
        embeddingLookupPosEncoding<T><<<grid, block, 0, stream>>>(from_tensor,
                                                                  embedding_table,
                                                                  position_encoding,
                                                                  all_ids,
                                                                  input_lengths,
                                                                  local_batch_size,
                                                                  hidden_units,
                                                                  step,
                                                                  max_input_length,
                                                                  batch_size,
                                                                  ite,
                                                                  scale);
    }
    else {
        embeddingLookup<T><<<grid, block, 0, stream>>>(from_tensor,
                                                       embedding_table,
                                                       all_ids,
                                                       local_batch_size,
                                                       hidden_units,
                                                       step,
                                                       max_input_length,
                                                       batch_size,
                                                       ite,
                                                       scale);
    }
}

template void invokeEmbeddingLookupPosEncoding(float* from_tensor,
                                               const float* embedding_table,
                                               const float* position_encoding,
                                               const int* all_ids,
                                               const int* input_lengths,
                                               const int local_batch_size,
                                               const int hidden_units,
                                               const float scale,
                                               const int step,
                                               const int max_input_length,
                                               const int batch_size,
                                               const int ite,
                                               cudaStream_t stream);

template void invokeEmbeddingLookupPosEncoding(half* from_tensor,
                                               const half* embedding_table,
                                               const half* position_encoding,
                                               const int* all_ids,
                                               const int* input_lengths,
                                               const int local_batch_size,
                                               const int hidden_units,
                                               const half scale,
                                               const int step,
                                               const int max_input_length,
                                               const int batch_size,
                                               const int ite,
                                               cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeEmbeddingLookupPosEncoding(__nv_bfloat16* from_tensor,
                                               const __nv_bfloat16* embedding_table,
                                               const __nv_bfloat16* position_encoding,
                                               const int* all_ids,
                                               const int* input_lengths,
                                               const int local_batch_size,
                                               const int hidden_units,
                                               const __nv_bfloat16 scale,
                                               const int step,
                                               const int max_input_length,
                                               const int batch_size,
                                               const int ite,
                                               cudaStream_t stream);
#endif

template<typename T>
__global__ void paddingEmbedding(T* padded_embedding_kernel,
                                 T* padded_embedding_bias,
                                 const T* embedding_kernel,
                                 const T* embedding_bias,
                                 const int hidden_unit,
                                 const int vocab_size,
                                 const int vocab_size_padded)
{
    for (int id = threadIdx.x + blockIdx.x * blockDim.x; id < hidden_unit * vocab_size_padded;
         id += blockDim.x * gridDim.x) {
        int row_id = id / vocab_size_padded;
        int col_id = id % vocab_size_padded;
        if (col_id < vocab_size) {
            padded_embedding_kernel[id] = embedding_kernel[row_id * vocab_size + col_id];
        }
        else {
            padded_embedding_kernel[id] = (T)(0.0f);
        }
    }

    for (int id = threadIdx.x + blockIdx.x * blockDim.x; id < vocab_size_padded; id += blockDim.x * gridDim.x) {
        if (id < vocab_size) {
            padded_embedding_bias[id] = embedding_bias[id];
        }
        else {
            padded_embedding_bias[id] = (T)(0.0f);
        }
    }
}

template<typename T>
void invokePaddingEmbedding(T* padded_embedding_kernel,
                            T* padded_embedding_bias,
                            const T* embedding_kernel,
                            const T* embedding_bias,
                            const int hidden_unit,
                            const int vocab_size,
                            const int vocab_size_padded,
                            cudaStream_t stream)
{
    dim3 block(512);
    dim3 grid((int)(ceil(hidden_unit * vocab_size_padded / 512.)));
    paddingEmbedding<<<grid, block, 0, stream>>>(padded_embedding_kernel,
                                                 padded_embedding_bias,
                                                 embedding_kernel,
                                                 embedding_bias,
                                                 hidden_unit,
                                                 vocab_size,
                                                 vocab_size_padded);
}

template void invokePaddingEmbedding(float* padded_embedding_kernel,
                                     float* padded_embedding_bias,
                                     const float* embedding_kernel,
                                     const float* embedding_bias,
                                     const int hidden_unit,
                                     const int vocab_size,
                                     const int vocab_size_padded,
                                     cudaStream_t stream);

template void invokePaddingEmbedding(half* padded_embedding_kernel,
                                     half* padded_embedding_bias,
                                     const half* embedding_kernel,
                                     const half* embedding_bias,
                                     const int hidden_unit,
                                     const int vocab_size,
                                     const int vocab_size_padded,
                                     cudaStream_t stream);

template<typename T>
__global__ void paddingEmbeddingKernel(T* padded_embedding_kernel,
                                       const T* embedding_kernel,
                                       const int hidden_unit,
                                       const int vocab_size,
                                       const int vocab_size_padded)
{
    for (int id = threadIdx.x + blockIdx.x * blockDim.x; id < hidden_unit * vocab_size_padded;
         id += blockDim.x * gridDim.x) {
        int row_id = id / vocab_size_padded;
        int col_id = id % vocab_size_padded;
        if (col_id < vocab_size) {
            padded_embedding_kernel[id] = embedding_kernel[row_id * vocab_size + col_id];
        }
        else {
            padded_embedding_kernel[id] = (T)(0.0f);
        }
    }
}

template<typename T>
void invokePaddingEmbeddingKernel(T* padded_embedding_kernel,
                                  const T* embedding_kernel,
                                  const int hidden_unit,
                                  const int vocab_size,
                                  const int vocab_size_padded,
                                  cudaStream_t stream)
{
    dim3 block(512);
    dim3 grid((int)(ceil(hidden_unit * vocab_size_padded / 512.)));
    paddingEmbeddingKernel<<<grid, block, 0, stream>>>(
        padded_embedding_kernel, embedding_kernel, hidden_unit, vocab_size, vocab_size_padded);
}

template void invokePaddingEmbeddingKernel(float* padded_embedding_kernel,
                                           const float* embedding_kernel,
                                           const int hidden_unit,
                                           const int vocab_size,
                                           const int vocab_size_padded,
                                           cudaStream_t stream);

template void invokePaddingEmbeddingKernel(half* padded_embedding_kernel,
                                           const half* embedding_kernel,
                                           const int hidden_unit,
                                           const int vocab_size,
                                           const int vocab_size_padded,
                                           cudaStream_t stream);

__global__ void gatherTree(gatherTreeParam param)
{
    //  beam: have six parts
    //      [prompt | input | input_padding | prompt_padding | generated output | padding (use end_token)]
    //  parents: have five parts
    //      [prompt | input | input_padding | prompt_padding | generated output | padding (use 0)]
    //  step_ids: need to remove prompt, input_padding and prompt_padding
    //      the shape is [input_length + requested_output_length, bs, beam_width]
    //      need to transpose to output_ids [bs, beam_width, input_length + requested_output_length]
    //  max_input_length: input + input_padding + prompt_padding

    const int max_input_length = param.input_lengths == nullptr ? 0 : param.max_input_length;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < param.batch_size * param.beam_width;
         i += gridDim.x * blockDim.x) {
        const int batch = i / param.beam_width;
        const int beam = i % param.beam_width;
        const int prompt_len =
            param.prefix_soft_prompt_lengths == nullptr ? 0 : param.prefix_soft_prompt_lengths[batch];
        const int input_len = param.input_lengths == nullptr ? 0 : param.input_lengths[i];

        const int* parent_ids = param.parent_ids;
        const int* step_ids = param.step_ids;

        // TODO(bhsueh) optimize the reduce_max operation for large beam_width
        int max_len = -1;
        int selected_beam_index = 0;
        for (int j = 0; j < param.beam_width; j++) {
            int tmp_len = __ldg(param.max_sequence_lengths + batch * param.beam_width + j);
            if (tmp_len > max_len) {
                max_len = tmp_len;
                selected_beam_index = j;
            }
        }
        const int max_seq_len_b = min(param.max_time, max_len);
        if (max_seq_len_b <= 0) {
            continue;
        }

#define GET_IX(time_ix, beam_ix)                                                                                       \
    (param.batch_size * param.beam_width * (time_ix) + param.beam_width * batch + (beam_ix))

        const int padding_offset_and_prompt_offset = max_input_length - input_len + prompt_len;
        const int initial_tgt_ix = GET_IX(max_seq_len_b - 1 - padding_offset_and_prompt_offset, beam);
        const int initial_parent_ix = GET_IX(max_seq_len_b - 1, beam);
        param.beams[initial_tgt_ix] = __ldg(step_ids + initial_parent_ix);
        int parent = parent_ids == nullptr ? 0 : __ldg(parent_ids + initial_parent_ix) % param.beam_width;
        bool found_bad = false;

        for (int level = max_seq_len_b - 2; level >= 0; --level) {
            if (level < prompt_len || (level >= input_len && level < max_input_length)) {
                continue;
            }
            int tgt_level = level >= max_input_length ? level - padding_offset_and_prompt_offset : level - prompt_len;
            const int level_beam_ix = GET_IX(tgt_level, beam);
            const int level_parent_ix = GET_IX(level, parent);
            if (parent < 0 || parent > param.beam_width) {
                // param.beams[level_beam_ix] = -1;
                param.beams[level_beam_ix] = param.end_tokens[batch];
                parent = -1;
                found_bad = true;
            }
            else {
                param.beams[level_beam_ix] = __ldg(step_ids + level_parent_ix);
                parent = parent_ids == nullptr ? 0 : __ldg(parent_ids + level_parent_ix) % param.beam_width;
            }
        }

        // set the padded part as end_token
        // input_len
        for (int index = max_len - (max_input_length - input_len); index < param.max_time; ++index) {
            param.beams[GET_IX(index, beam)] = param.end_tokens[batch];
        }

        // Not necessary when using a BeamSearchDecoder, but necessary
        // when a user feeds in possibly broken trajectory (i.e., non-eos
        // entries in a beam following eos entries).
        if (!found_bad) {
            bool finished = false;
            // skip the step 0 because it is often the start token
            int start_step = max_input_length == 0 ? 1 : max_input_length;
            for (int time = start_step; time < max_seq_len_b; ++time) {
                const int level_beam_ix = GET_IX(time, beam);
                if (finished) {
                    param.beams[level_beam_ix] = param.end_tokens[batch];
                }
                else if (param.beams[level_beam_ix] == param.end_tokens[batch]) {
                    finished = true;
                }
            }
        }
#undef GET_IX

        // transpose on output_ids
        if (param.output_ids != nullptr) {
            for (int j = 0; j < (param.max_time - param.max_prefix_soft_prompt_length); j++) {
                param.output_ids[i * (param.max_time - param.max_prefix_soft_prompt_length) + j] =
                    param.beams[j * param.batch_size * param.beam_width + i];
            }
        }
    }
}

void invokeGatherTree(int* beams,
                      int* max_sequence_lengths,
                      const int max_time,
                      const int batch_size,
                      const int beam_width,
                      const int* step_ids,
                      const int* parent_ids,
                      const int* end_tokens,
                      cudaStream_t stream)
{
    gatherTreeParam param;
    param.beams = beams;
    param.max_sequence_lengths = max_sequence_lengths;
    param.max_time = max_time;
    param.batch_size = batch_size;
    param.beam_width = beam_width;
    param.step_ids = step_ids;
    param.parent_ids = parent_ids;
    param.end_tokens = end_tokens;
    param.max_input_length = 1;
    param.prefix_soft_prompt_lengths = nullptr;
    param.stream = stream;
    invokeGatherTree(param);
}

void invokeGatherTree(int* beams,
                      int* max_sequence_lengths,
                      const int max_time,
                      const int batch_size,
                      const int beam_width,
                      const int* step_ids,
                      const int* parent_ids,
                      const int* end_tokens,
                      const int max_input_length,
                      cudaStream_t stream)
{
    gatherTreeParam param;
    param.beams = beams;
    param.max_sequence_lengths = max_sequence_lengths;
    param.max_time = max_time;
    param.batch_size = batch_size;
    param.beam_width = beam_width;
    param.step_ids = step_ids;
    param.parent_ids = parent_ids;
    param.end_tokens = end_tokens;
    param.max_input_length = max_input_length;
    param.prefix_soft_prompt_lengths = nullptr;
    param.stream = stream;
    invokeGatherTree(param);
}

void invokeGatherTree(gatherTreeParam param)
{
    int batchbeam = param.batch_size * param.beam_width;
    dim3 grid(1), block(batchbeam);
    // though decoder do not support > 1024 for now
    if (batchbeam > 1024) {
        grid.x = ceil(param.batch_size * param.beam_width / 1024.);
        block.x = 1024;
    }
    gatherTree<<<grid, block, 0, param.stream>>>(param);
}

__global__ void minusUnfinishedSeqlen(int* sequence_lengths, const bool* finished, const int token_num)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < token_num; i += blockDim.x * gridDim.x) {
        if (finished[i] == false) {
            sequence_lengths[i] -= 1;
        }
    }
}

void invokeMinusUnfinishedSeqlen(int* sequence_lengths, const bool* finished, const int token_num, cudaStream_t stream)
{
    dim3 block(min(256, token_num));
    dim3 grid(ceil(token_num / 256.));
    minusUnfinishedSeqlen<<<block, grid, 0, stream>>>(sequence_lengths, finished, token_num);
}

__global__ void plusUnfinishedSeqlen(int* sequence_lengths, const bool* finished, const int token_num)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < token_num; i += blockDim.x * gridDim.x) {
        if (finished[i] == false) {
            sequence_lengths[i] += 1;
        }
    }
}

void invokePlusUnfinishedSeqlen(int* sequence_lengths, const bool* finished, const int token_num, cudaStream_t stream)
{
    dim3 block(min(256, token_num));
    dim3 grid(ceil(token_num / 256.));
    plusUnfinishedSeqlen<<<block, grid, 0, stream>>>(sequence_lengths, finished, token_num);
}

template<typename T>
__global__ void plusScalar(T* buf, const T val, const int size)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += blockDim.x * gridDim.x) {
        buf[i] += val;
    }
}

template<typename T>
void invokePlusScalar(T* buf, const T val, const int size, cudaStream_t stream)
{
    dim3 block(min(256, size));
    dim3 grid(ceil(size / 256.));
    plusScalar<<<block, grid, 0, stream>>>(buf, val, size);
}

template void invokePlusScalar(int* buf, const int val, const int size, cudaStream_t stream);

}  // namespace fastertransformer
