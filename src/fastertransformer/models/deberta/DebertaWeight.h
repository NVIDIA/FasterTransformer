/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/deberta/DebertaLayerWeight.h"

namespace fastertransformer {

template<typename T>
struct DebertaWeight {

    DebertaWeight() = default;
    DebertaWeight(const size_t hidden_units,
                  const size_t inter_size,
                  const size_t max_relative_positions,
                  const size_t relative_position_buckets,
                  const size_t vocab_size,
                  const size_t num_layer,
                  const size_t tensor_para_size,
                  const size_t tensor_para_rank,
                  const size_t pipeline_para_size,
                  const size_t pipeline_para_rank);
    DebertaWeight(const int    hidden_units,
                  const int    inter_size,
                  const size_t max_relative_positions,
                  const size_t relative_position_buckets,
                  const size_t vocab_size,
                  const int    num_layer):
        DebertaWeight(hidden_units,
                      inter_size,
                      max_relative_positions,
                      relative_position_buckets,
                      vocab_size,
                      num_layer,
                      1,
                      0,
                      1,
                      0)
    {
    }
    ~DebertaWeight();
    DebertaWeight(const DebertaWeight& other);
    DebertaWeight&                     operator=(const DebertaWeight& other);
    std::vector<DebertaLayerWeight<T>> deberta_layer_weights;
    const T*                           word_embedding_table = nullptr;
    LayerNormWeight<T>                 word_embedding_layernorm_weights;
    const T*                           relative_embedding_table = nullptr;
    LayerNormWeight<T>                 relative_embedding_layernorm_weights;

    bool isValidLayerParallelId(int l);
    void loadModel(std::string dir_path);

private:
    void setWeightPtr();

    size_t hidden_units_;
    size_t inter_size_;
    size_t max_relative_positions_;
    size_t relative_position_buckets_;
    size_t vocab_size_;
    size_t num_layer_;
    size_t tensor_para_size_;
    size_t tensor_para_rank_;
    size_t pipeline_para_size_;
    size_t pipeline_para_rank_;
    bool   is_maintain_buffer = false;

    // 6: [1] word embedding weight [2] word-LN weight [3] word-LN bias [4] relative embedding weight [5] relative-LN
    // weight [6] relative-LN bias
    const static int weights_num_ = 6;
    T*               weights_ptr[weights_num_];
    size_t           weights_size[weights_num_];
};

}  // namespace fastertransformer
