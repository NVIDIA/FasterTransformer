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

#include "src/fastertransformer/kernels/gen_relative_pos_bias.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/models/bart/BartDecoderLayerWeight.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
struct BartDecodingWeight {

    BartDecodingWeight() = default;
    BartDecodingWeight(const size_t                head_num,
                       const size_t                size_per_head,
                       const size_t                d_model,
                       const size_t                inter_size,
                       const size_t                vocab_size,
                       const size_t                num_layer,
                       const size_t                mem_d_model,
                       const size_t                num_bucket_or_max_seq_len,
                       const size_t                tensor_para_size,
                       const size_t                tensor_para_rank,
                       const size_t                pipeline_para_size,
                       const size_t                pipeline_para_rank,
                       const bool                  bart_with_bias_para          = true,
                       const bool                  mbart_para                   = false,
                       const bool                  use_gated_activation_para    = false,
                       const PositionEmbeddingType position_embedding_type_para = PositionEmbeddingType::absolute);
    ~BartDecodingWeight();
    BartDecodingWeight(const BartDecodingWeight& other);
    BartDecodingWeight& operator=(const BartDecodingWeight& other);

    LayerNormWeight<T>                      pre_decoder_layernorm;
    std::vector<BartDecoderLayerWeight<T>*> decoder_layer_weights;
    const T*                                pre_decoder_embedding_table             = nullptr;
    const T*                                absolute_or_relative_position_embedding = nullptr;
    LayerNormWeight<T>                      post_decoder_layernorm;
    DenseWeight<T> post_decoder_embedding;  // Megatron embedding is weight + bias, so prefer to use a separate weight
                                            // class to store
    bool bart_with_bias       = true;
    bool mbart                = false;
    bool use_gated_activation = false;
    // 0 = relative_position_embedding,  1 = absolute_position_embedding
    PositionEmbeddingType position_embedding_type = PositionEmbeddingType::absolute;

    void loadModel(std::string dir_path);
    void resizeLayer(const int num_layer);

    void setBartStructureDiff(bool                  bart_with_bias_para,
                              bool                  mbart_para,
                              bool                  use_gated_activation_para,
                              PositionEmbeddingType position_embedding_type_para);

private:
    void setWeightPtr();
    void mallocWeights();
    bool isValidLayerParallelId(int l);
    void initialize();

    size_t head_num_;
    size_t size_per_head_;
    size_t d_model_;
    size_t inter_size_;
    size_t vocab_size_;
    size_t num_layer_;
    size_t mem_d_model_;
    // refer to num_buckt if using relative position embedding
    // refer to max_seq_len if using absolute position embedding
    size_t num_bucket_or_max_seq_len_;
    size_t tensor_para_size_;
    size_t tensor_para_rank_;
    size_t pipeline_para_size_;
    size_t pipeline_para_rank_;
    bool   is_maintain_buffer_ = false;

    int real_weights_num_;

    // 8: [0] absolute/relative positional embedding weight [1] word embedding weight [2] word embedding 2 weight [3]
    // pre-LN weight [4] post-LN weight [5] pre-LN bias [6] post-LN bias [7] word embedding 2 bias. Assuming both mBART
    // and bias
    const static int weights_num_ = 8;
    T*               weights_ptr[weights_num_];
    size_t           weights_size[weights_num_];
};

}  // namespace fastertransformer
