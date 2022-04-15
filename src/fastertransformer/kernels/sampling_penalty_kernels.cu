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

#include <float.h>

#include "src/fastertransformer/kernels/sampling_penalty_kernels.h"

namespace fastertransformer {

// TODO Add half2 implementation
template<typename T>
__global__ void applyTemperaturePenalty(T* logits,
                                        const T* bias,
                                        const float temperature_inverse,
                                        const int m,
                                        const int vocab_size,
                                        const int vocab_size_padd)
{
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? 65504.F : FLT_MAX;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < m * vocab_size_padd;
         index += blockDim.x * gridDim.x) {
        T bias_val = bias == nullptr ? (T)(0.0f) : bias[index % vocab_size_padd];
        if (index % vocab_size_padd < vocab_size) {
            logits[index] = (logits[index] + bias_val) * (T)temperature_inverse;
        }
        else {
            logits[index] = -MAX_T_VAL;
        }
    }
}

template<typename T>
void invokeApplyTemperaturePenalty(T* logits,
                                   const T* bias,
                                   const float temperature,
                                   const int m,
                                   const int vocab_size,
                                   const int vocab_size_padd,
                                   cudaStream_t stream)
{
    dim3 grid(min(m, 65536));
    dim3 block(min(vocab_size_padd, 1024));
    const T temperature_inverse = (T)(1.f / (float)temperature);
    applyTemperaturePenalty<T>
        <<<grid, block, 0, stream>>>(logits, bias, temperature_inverse, m, vocab_size, vocab_size_padd);
}

template void invokeApplyTemperaturePenalty(float* logits,
                                            const float* bias,
                                            const float temperature,
                                            const int m,
                                            const int vocab_size,
                                            const int vocab_size_padd,
                                            cudaStream_t stream);

template void invokeApplyTemperaturePenalty(half* logits,
                                            const half* bias,
                                            const float temperature,
                                            const int m,
                                            const int vocab_size,
                                            const int vocab_size_padd,
                                            cudaStream_t stream);

template<typename T>
__global__ void applyRepetitionPenalty(T* logits,
                                       const float penalty,
                                       const int* start_ids,
                                       int* output_ids,
                                       const int batch_size,
                                       const int local_batch_size,
                                       const int vocab_size,
                                       const int vocab_size_padd,
                                       const int* input_lengths,
                                       const int max_input_len,
                                       const int step,
                                       const int ite)
{
    extern __shared__ float penalty_logits[];
    int* penalty_indices = (int*)(penalty_logits + step);

    logits = logits + blockIdx.x * vocab_size_padd;
    const int input_length = input_lengths != nullptr ? input_lengths[blockIdx.x] : max_input_len;
    for (int index = threadIdx.x; index < step; index += blockDim.x) {

        if (index >= input_length && index < max_input_len) {
            continue;
        }

        // output_ids shape: (input_len + output_len, batch_size)
        int penalty_index = output_ids[index * batch_size + local_batch_size * ite + blockIdx.x];
        if (penalty_index >= vocab_size) {
            continue;
        }
        penalty_indices[index] = penalty_index;
        float logit = (float)logits[penalty_index];
        penalty_logits[index] = logit < 0.0f ? logit * penalty : logit / penalty;
    }

    if (blockDim.x > 32) {
        __syncthreads();
    }

    for (int index = threadIdx.x; index < step; index += blockDim.x) {

        if (index >= input_length && index < max_input_len) {
            continue;
        }

        // output_ids shape: (input_len + output_len, batch_size)
        if (penalty_indices[index] >= vocab_size) {
            continue;
        }
        logits[penalty_indices[index]] = penalty_logits[index];
    }
}

template<typename T>
void invokeApplyRepetitionPenalty(T* logits,
                                  const float penalty,
                                  const int* start_ids,
                                  int* output_ids,
                                  const int batch_size,
                                  const int local_batch_size,
                                  const int vocab_size,
                                  const int vocab_size_padd,
                                  const int* input_lengths,
                                  const int max_input_len,
                                  const int step,
                                  const int ite,
                                  cudaStream_t stream)
{
    dim3 block(min(512, step));
    dim3 grid((int)(local_batch_size));
    applyRepetitionPenalty<T><<<grid, block, step * (sizeof(float) + sizeof(int)), stream>>>(logits,
                                                                                             penalty,
                                                                                             start_ids,
                                                                                             output_ids,
                                                                                             batch_size,
                                                                                             local_batch_size,
                                                                                             vocab_size,
                                                                                             vocab_size_padd,
                                                                                             input_lengths,
                                                                                             max_input_len,
                                                                                             step,
                                                                                             ite);
}

template void invokeApplyRepetitionPenalty(float* logits,
                                           const float penalty,
                                           const int* start_ids,
                                           int* output_ids,
                                           const int batch_size,
                                           const int local_batch_size,
                                           const int vocab_size,
                                           const int vocab_size_padd,
                                           const int* input_lengths,
                                           const int max_input_len,
                                           const int step,
                                           const int ite,
                                           cudaStream_t stream);

template void invokeApplyRepetitionPenalty(half* logits,
                                           const float penalty,
                                           const int* start_ids,
                                           int* output_ids,
                                           const int batch_size,
                                           const int local_batch_size,
                                           const int vocab_size,
                                           const int vocab_size_padd,
                                           const int* input_lengths,
                                           const int max_input_len,
                                           const int step,
                                           const int ite,
                                           cudaStream_t stream);

}  // namespace fastertransformer