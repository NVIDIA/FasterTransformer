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

#pragma once

#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/models/gpt/GptDecoderLayerWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
struct GptWeight {

    GptWeight() = default;
    GptWeight(const int hidden_units,
              const int inter_size,
              const int vocab_size,
              const int num_layer,
              const int max_seq_len);
    ~GptWeight();
    GptWeight(const GptWeight& other);
    GptWeight& operator=(const GptWeight& other);
    void loadModel(std::string dir_path);
#ifdef SPARSITY_ENABLED
    // Currently the name convention is followed by BERT's sparsity case,
    // but it will be fixed later.
    void compress_weights(cublasMMWrapper& cublas_wrapper);
#endif

    std::vector<GptDecoderLayerWeight<T>> decoder_layer_weights;
    const T* position_encoding_table = nullptr;
    const T* pre_decoder_embedding_table = nullptr;
    LayerNormWeight<T> post_decoder_layernorm;
    DenseWeight<T> post_decoder_embedding;

private:
    void setWeightPtr();
    void mallocWeights();

    int hidden_units_;
    int inter_size_;
    int vocab_size_;
    int num_layer_;
    int max_seq_len_;
    bool is_maintain_buffer = false;
    T* weights_ptr[5];
};

}  // namespace fastertransformer
