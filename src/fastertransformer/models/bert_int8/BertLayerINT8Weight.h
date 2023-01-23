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

#include "src/fastertransformer/layers/FfnINT8Weight.h"
#include "src/fastertransformer/layers/attention_layers_int8/AttentionINT8Weight.h"
#include "src/fastertransformer/models/bert/BertLayerWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
struct BertLayerINT8Weight: BertLayerWeight<T> {

    BertLayerINT8Weight() = default;
    BertLayerINT8Weight(const int hidden_units, const int inter_size);
    ~BertLayerINT8Weight();
    BertLayerINT8Weight(const BertLayerINT8Weight& other);
    BertLayerINT8Weight& operator=(const BertLayerINT8Weight& other);

    LayerNormWeight<T>     attn_layernorm_weights;
    LayerNormWeight<T>     ffn_layernorm_weights;
    AttentionINT8Weight<T> attention_weights;
    FfnINT8Weight<T>       ffn_weights;
    ScaleList              scale_list_;

private:
    void setWeightPtr();

    int    hidden_units_;
    int    inter_size_;
    bool   is_maintain_buffer = false;
    T*     weights_ptr[12];
    float* scale_list_ptr[2];
    T*     sp_weights_ptr[6];
    bool   is_maintain_sp_buffer = false;
};

}  // namespace fastertransformer
