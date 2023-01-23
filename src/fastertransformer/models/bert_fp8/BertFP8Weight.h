/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/bert_fp8/BertFP8LayerWeight.h"

namespace fastertransformer {

template<typename T1, typename T2>
struct BertFP8Weight {

    BertFP8Weight() = default;
    BertFP8Weight(const size_t d_model,
                  const size_t head_num,
                  const size_t size_per_head,
                  const size_t inter_size,
                  const size_t num_layer,
                  const size_t vocab_size,
                  const size_t max_position_embeddings,
                  const size_t token_type_vocab_size,
                  const size_t tensor_para_size,
                  const size_t pipeline_para_size,
                  const int    fp8_mode,
                  bool         is_load_model     = false,
                  bool         is_fused_qkv_gemm = true);
    ~BertFP8Weight();
    BertFP8Weight(const BertFP8Weight& other);
    BertFP8Weight& operator=(const BertFP8Weight& other) = delete;
    void           transposeWeight();
    void           loadModel(std::string dir_path);
    void           serialize(uint8_t*& buffer);
    void           deserialize(const uint8_t*& buffer);
    size_t         getSerializationSize() const;

    // Weights
    std::vector<BertFP8LayerWeight<T1, T2>> bert_layer_weights;
    const T2*                               word_embeddings;
    const T2*                               position_embeddings;
    const T2*                               token_type_embeddings;
    LayerNormWeight<T2>                     embeddings_layernorm;
    LayerNormWeight<T2>                     post_transformer_layernorm_weights;
    DenseWeight<T2, T2>                     pooler_dense;

private:
    void setWeightPtr();
    void mallocWeights();

    // model configs
    size_t d_model_       = 0;
    size_t head_num_      = 0;
    size_t size_per_head_ = 0;
    size_t inter_size_    = 0;
    size_t num_layer_     = 0;
    size_t vocab_size_    = 0;
    size_t max_position_embeddings_;
    size_t token_type_vocab_size_;
    size_t tensor_para_size_;
    size_t pipeline_para_size_;
    // mode 0: no fp8. Should use original bert directly
    // mode 1: per tensor scale for activation, per channel scale for weight
    // mode 2: per tensor scale for activation and weight
    int fp8_mode_;

    bool                                is_maintain_buffer = false;
    std::vector<std::pair<size_t, T2*>> weights_ptr;
};

}  // namespace fastertransformer
