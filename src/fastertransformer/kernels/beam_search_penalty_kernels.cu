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

#include "src/fastertransformer/kernels/beam_search_penalty_kernels.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"

namespace fastertransformer {

template<typename T>
__global__ void add_bias_apply_logit_penalties_kernel(int step,
                                                      int vocab_size,
                                                      const int vocab_size_padded,
                                                      int beam_width,
                                                      T* logits,
                                                      const int* current_ids,
                                                      const int* previous_ids,
                                                      const int* parent_ids,
                                                      const int* input_lengths,
                                                      const T* bias,
                                                      const int ite,
                                                      const int max_input_length,
                                                      const int batch_size,
                                                      const int* end_ids,
                                                      float inv_temp,
                                                      float len_penalty,
                                                      float repeat_penalty)
{
    // TODO(bhsueh) Seems there are some problem for len_penalty implementation
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bbid = blockIdx.y;
    int bbsize = batch_size * beam_width;
    int batch_id = blockIdx.y / beam_width;

    const int vocab_size_padded_offset = bbid * vocab_size_padded;

    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    for (int i = tid + bid * blockDim.x; i < vocab_size_padded; i += blockDim.x * gridDim.x) {
        if (i < vocab_size) {
            T bias_val = bias == nullptr ? (T)(0.0f) : bias[i];
            logits[i + vocab_size_padded_offset] = (logits[i + vocab_size_padded_offset] + bias_val) * (T)(inv_temp);
        }
        else {
            logits[i + vocab_size_padded_offset] = -MAX_T_VAL;
        }
    }
    if (tid == 0 && bid == 0) {
        // TODO(bhsueh) apply repetition penalty (this can apply the penalty multiple times to a repeated word).
        int prev_id = current_ids[bbid];
        const int end_id = end_ids[batch_id];
        if (logits[prev_id + vocab_size_padded_offset] > T(0)) {
            logits[prev_id + vocab_size_padded_offset] =
                float(logits[prev_id + vocab_size_padded_offset]) / repeat_penalty;
            logits[end_id + vocab_size_padded_offset] = float(logits[end_id + vocab_size_padded_offset]) / len_penalty;
        }
        else {
            logits[prev_id + vocab_size_padded_offset] =
                float(logits[prev_id + vocab_size_padded_offset]) * repeat_penalty;
            logits[end_id + vocab_size_padded_offset] = float(logits[end_id + vocab_size_padded_offset]) * len_penalty;
        }
        if (step > 1) {
            int parent_beamid = parent_ids[bbsize * (step - 2) + ite * gridDim.y + blockIdx.y];
            for (int i = step - 2; i > 0; --i) {
                bool is_mask = input_lengths != nullptr && i >= input_lengths[bbid] && step < max_input_length;
                if (is_mask == false) {
                    prev_id = previous_ids[bbsize * i + ite * gridDim.y + batch_id * beam_width + parent_beamid];
                    if (logits[prev_id + vocab_size_padded_offset] > T(0)) {
                        logits[prev_id + vocab_size_padded_offset] =
                            float(logits[prev_id + vocab_size_padded_offset]) / repeat_penalty;
                    }
                    else {
                        logits[prev_id + vocab_size_padded_offset] =
                            float(logits[prev_id + vocab_size_padded_offset]) * repeat_penalty;
                    }
                }
                parent_beamid = parent_ids[bbsize * (i - 1) + ite * gridDim.y + batch_id * beam_width + parent_beamid];
            }
        }
    }
}

template<typename T>
void invokeAddBiasApplyPenalties(int step,
                                 T* logits,
                                 const int* current_ids,
                                 const int* previous_ids,
                                 const int* parent_ids,
                                 const int* input_lengths,
                                 const T* bias,
                                 const int ite,
                                 const int max_input_length,
                                 const int local_batch_size,
                                 const int batch_size,
                                 const int beam_width,
                                 const int vocab_size,
                                 const int vocab_size_padded,
                                 const int* end_ids,
                                 const float temperature,
                                 const float len_penalty,
                                 const float repeat_penalty,
                                 cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((vocab_size_padded + block.x - 1) / block.x, beam_width * local_batch_size);
    add_bias_apply_logit_penalties_kernel<T><<<grid, block, 0, stream>>>(step,
                                                                         vocab_size,
                                                                         vocab_size_padded,
                                                                         beam_width,
                                                                         logits,
                                                                         current_ids,
                                                                         previous_ids,
                                                                         parent_ids,
                                                                         input_lengths,
                                                                         bias,
                                                                         ite,
                                                                         max_input_length,
                                                                         batch_size,
                                                                         end_ids,
                                                                         1.f / temperature,
                                                                         len_penalty,
                                                                         repeat_penalty);
    sync_check_cuda_error();
}

template void invokeAddBiasApplyPenalties(int step,
                                          float* logits,
                                          const int* current_ids,
                                          const int* previous_ids,
                                          const int* parent_ids,
                                          const int* input_lengths,
                                          const float* bias,
                                          const int ite,
                                          const int max_input_length,
                                          const int local_batch_size,
                                          const int batch_size,
                                          const int beam_width,
                                          const int vocab_size,
                                          const int vocab_size_padded,
                                          const int* end_ids,
                                          const float temerature,
                                          const float len_penalty,
                                          const float repeat_penalty,
                                          cudaStream_t stream);

template void invokeAddBiasApplyPenalties(int step,
                                          half* logits,
                                          const int* current_ids,
                                          const int* previous_ids,
                                          const int* parent_ids,
                                          const int* input_lengths,
                                          const half* bias,
                                          const int ite,
                                          const int max_input_length,
                                          const int local_batch_size,
                                          const int batch_size,
                                          const int beam_width,
                                          const int vocab_size,
                                          const int vocab_size_padded,
                                          const int* end_ids,
                                          const float temerature,
                                          const float len_penalty,
                                          const float repeat_penalty,
                                          cudaStream_t stream);

}  // namespace fastertransformer