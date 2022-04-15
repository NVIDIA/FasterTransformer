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

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <unordered_map>

#include "src/fastertransformer/utils/Tensor.h"

namespace fastertransformer {

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
                                              cudaStream_t stream);

template<typename T>
struct inputIdsEmbeddingLookupPosEncodingSoftPromptParam {
    T* from_tensor;
    int* output_ids;
    int* input_lengths;
    const T* embedding_table;
    const T* pos_table;
    const float* prefix_soft_prompt_embedding;
    const int* prefix_soft_prompt_lengths;
    int* input_ids;
    int start_step;
    int max_input_length;
    int max_prefix_soft_prompt_length;
    int batch_size;
    int beam_width;
    int hidden_units;
    cudaStream_t stream;
};

template<typename T>
void invokeInputIdsEmbeddingLookupPosEncodingSoftPrompt(inputIdsEmbeddingLookupPosEncodingSoftPromptParam<T> param);

template<typename T>
void invokeTransposeAxis01(T* out, T* in, const int dim0, const int dim1, const int dim2, cudaStream_t stream);

template<typename T>
void invokeTransposeAxis01(
    T* out, T* in, const int* in_skipping_dim1, const int dim0, const int dim1, cudaStream_t stream);

template<typename T>
void invokeBuildDecoderAttentionMask(
    T* attention_mask, const int* sequence_lengths, const int batch_size, const int max_seq_len, cudaStream_t stream);

template<typename T>
void invokeLookupHiddenStateOfLastToken(T* from_tensor,
                                        const T* hidden_state,
                                        const int* input_lengths,
                                        const int max_input_length,
                                        const int batch_size,
                                        const int hidden_units,
                                        cudaStream_t stream);

void invokeTileGptInputs(int* tiled_input_ids,
                         int* tiled_input_lengths,
                         const int* input_ids,
                         const int* input_lengths,
                         const int batch_size,
                         const int beam_width,
                         const int max_input_length,
                         cudaStream_t stream);

bool hasDiffRuntimeArgs(const std::unordered_map<std::string, Tensor>* input_tensors);

template<typename T>
void handleOptArg(const std::unordered_map<std::string, Tensor>* input_tensors,
                  const std::string& arg_name,
                  T* d_ptr,
                  T default_value,
                  size_t size)
{
    if (input_tensors->find(arg_name) != input_tensors->end()) {
        FT_CHECK(input_tensors->at(arg_name).size() == size);
        cudaH2Dcpy(d_ptr, (const T*)input_tensors->at(arg_name).data, size);
    }
    else {
        deviceFill(d_ptr, size, default_value);
    }
}

}  // namespace fastertransformer
