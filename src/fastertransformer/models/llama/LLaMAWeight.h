/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "src/fastertransformer/models/llama/LLaMADecoderLayerWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include "src/fastertransformer/utils/prompt_learning.h"

namespace fastertransformer {

template<typename T>
struct LLaMAWeight {

    LLaMAWeight() = default;
    LLaMAWeight(
        const int                                  hidden_units,
        const int                                  inter_size,
        const int                                  vocab_size,
        const int                                  num_layer,
        const int                                  tensor_para_size     = 1,
        const int                                  tensor_para_rank     = 0,
        const int                                  layer_para_size      = 1,
        const int                                  layer_para_rank      = 0);

    ~LLaMAWeight();
    LLaMAWeight(const LLaMAWeight& other);
    LLaMAWeight& operator=(const LLaMAWeight& other);

    void loadModel(std::string dir_path);

    void resizeLayer(const int num_layer);

    std::vector<LLaMADecoderLayerWeight<T>*> decoder_layer_weights;
    const T*                                   pre_decoder_embedding_table = nullptr;
    const T* position_encoding_table = nullptr;

    LayerNormWeight<T> post_decoder_layernorm;
    DenseWeight<T>     post_decoder_embedding;

private:
    void setWeightPtr();
    void mallocWeights();
    bool isValidLayerParallelId(int l);

    int hidden_units_;
    int inter_size_;
    int vocab_size_;
    int num_layer_;

    int tensor_para_size_;
    int tensor_para_rank_;
    int layer_para_size_;
    int layer_para_rank_;

    // prompt learning pair (task_name, (task_name_id, prompt_len))
    // each prompt token's weight size
    size_t prompt_token_weight_size_ = 0;

    bool            is_maintain_buffer = false;
    const size_t    num_base_weights   = 4;
    std::vector<T*> weights_ptr        = std::vector<T*>(num_base_weights);
};

}  // namespace fastertransformer
