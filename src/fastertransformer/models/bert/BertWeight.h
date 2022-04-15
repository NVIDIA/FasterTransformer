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

#include "src/fastertransformer/models/bert/BertLayerWeight.h"

namespace fastertransformer {

template<typename T>
struct BertWeight {

    BertWeight() = default;
    BertWeight(const int hidden_units, const int inter_size, const int num_layer):
        hidden_units_(hidden_units), inter_size_(inter_size), num_layer_(num_layer)
    {
        deviceMalloc(&weights_ptr[0], hidden_units_);
        deviceMalloc(&weights_ptr[1], hidden_units_);

        setWeightPtr();
        for (int i = 0; i < num_layer_; i++) {
            bert_layer_weights.push_back(BertLayerWeight<T>(hidden_units_, inter_size_));
        }
    }

    ~BertWeight()
    {
        if (is_maintain_buffer == true) {
            bert_layer_weights.clear();
            for (int i = 0; i < 2; i++) {
                deviceFree(weights_ptr[i]);
            }

            post_transformer_layernorm_weights.gamma = nullptr;
            post_transformer_layernorm_weights.beta = nullptr;
            is_maintain_buffer = false;
        }
    }

    BertWeight(const BertWeight& other):
        hidden_units_(other.hidden_units_), inter_size_(other.inter_size_), num_layer_(other.num_layer_)
    {
        bert_layer_weights.clear();
        for (int i = 0; i < num_layer_; i++) {
            bert_layer_weights.push_back(other.bert_layer_weights[i]);
        }
        deviceMalloc(&weights_ptr[0], hidden_units_);
        cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
        deviceMalloc(&weights_ptr[1], hidden_units_);
        cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);

        setWeightPtr();
    }

    BertWeight& operator=(const BertWeight& other)
    {
        hidden_units_ = other.hidden_units_;
        inter_size_ = other.inter_size_;
        num_layer_ = other.num_layer_;
        bert_layer_weights.clear();
        for (int i = 0; i < num_layer_; i++) {
            bert_layer_weights.push_back(other.bert_layer_weights[i]);
        }
        deviceMalloc(&weights_ptr[0], hidden_units_);
        cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
        deviceMalloc(&weights_ptr[1], hidden_units_);
        cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);

        setWeightPtr();
    }

    std::vector<BertLayerWeight<T>> bert_layer_weights;
    LayerNormWeight<T> post_transformer_layernorm_weights;

private:
    void setWeightPtr()
    {
        post_transformer_layernorm_weights.gamma = weights_ptr[0];
        post_transformer_layernorm_weights.beta = weights_ptr[1];

        is_maintain_buffer = true;
    }
    int hidden_units_;
    int inter_size_;
    int num_layer_;
    bool is_maintain_buffer = false;
    T* weights_ptr[2];
};

}  // namespace fastertransformer
