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

#include "src/fastertransformer/models/bert/BertLayerWeight.h"

namespace fastertransformer {

template<typename T>
struct BertWeight {

    BertWeight() = default;
    BertWeight(const size_t hidden_units,
               const size_t inter_size,
               const size_t num_layer,
               const size_t tensor_para_size,
               const size_t tensor_para_rank,
               const size_t pipeline_para_size,
               const size_t pipeline_para_rank);
    BertWeight(const int hidden_units, const int inter_size, const int num_layer):
        BertWeight(hidden_units, inter_size, num_layer, 1, 0, 1, 0)
    {
    }
    ~BertWeight();
    BertWeight(const BertWeight& other);
    BertWeight&                     operator=(const BertWeight& other);
    std::vector<BertLayerWeight<T>> bert_layer_weights;
    LayerNormWeight<T>              post_transformer_layernorm_weights;

    bool isValidLayerParallelId(int l);
    void loadModel(std::string dir_path);

private:
    void setWeightPtr();

    size_t hidden_units_;
    size_t inter_size_;
    size_t num_layer_;
    size_t tensor_para_size_;
    size_t tensor_para_rank_;
    size_t pipeline_para_size_;
    size_t pipeline_para_rank_;
    bool   is_maintain_buffer = false;
    T*     weights_ptr[2];
};

}  // namespace fastertransformer
