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

#include "src/fastertransformer/kernels/gen_relative_pos_bias.h"
#include "src/fastertransformer/models/t5/T5EncoderLayerWeight.h"
#include "src/fastertransformer/utils/prompt_learning.h"

namespace fastertransformer {

template<typename T>
struct T5EncoderWeight {

    T5EncoderWeight() = default;
    T5EncoderWeight(size_t                                     head_num,
                    size_t                                     size_per_head,
                    size_t                                     d_model,
                    size_t                                     inter_size,
                    size_t                                     vocab_size,
                    size_t                                     num_layer,
                    size_t                                     num_bucket_or_max_seq_len,
                    size_t                                     tensor_para_size,
                    size_t                                     tensor_para_rank,
                    size_t                                     pipeline_para_size,
                    size_t                                     pipeline_para_rank,
                    bool                                       t5_with_bias_para         = false,
                    bool                                       use_gated_activation_para = false,
                    PositionEmbeddingType                      pe_type              = PositionEmbeddingType::relative,
                    PromptLearningType                         prompt_learning_type = PromptLearningType::no_prompt,
                    std::map<std::string, std::pair<int, int>> prompt_learning_pair = {},
                    size_t                                     ia3_num_tasks        = 0,
                    size_t                                     adapter_inter_size   = 0);
    ~T5EncoderWeight();
    T5EncoderWeight(const T5EncoderWeight& other);
    T5EncoderWeight& operator=(const T5EncoderWeight& other);

    std::vector<T5EncoderLayerWeight<T>*> t5_encoder_layer_weights;
    LayerNormWeight<T>                    post_transformer_layernorm_weights;
    const T*                              absolute_or_relative_position_embedding = nullptr;
    const T*                              embedding_table                         = nullptr;
    bool                                  t5_with_bias                            = false;
    bool                                  use_gated_activation                    = false;
    PositionEmbeddingType                 position_embedding_type                 = PositionEmbeddingType::relative;
    std::vector<std::pair<const T*, int>> prompt_learning_table                   = {};

    void loadModel(std::string dir_path);
    void resizeLayer(const int num_layer);
    void setT5StructureDiff(bool                  t5_with_bias_para,
                            bool                  use_gated_activation_para,
                            PositionEmbeddingType position_embedding_type_para);

    inline size_t getNumIA3Tasks() const
    {
        return ia3_num_tasks_;
    };

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
    // refer to num_buckt if using relative position embedding
    // refer to max_seq_len if using absolute position embedding
    size_t num_bucket_or_max_seq_len_;
    size_t tensor_para_size_;
    size_t tensor_para_rank_;
    size_t pipeline_para_size_;
    size_t pipeline_para_rank_;
    size_t ia3_num_tasks_;
    size_t adapter_inter_size_;

    bool is_maintain_buffer = false;

    int                 real_weights_num_;
    const static int    weights_num_ = 4;
    std::vector<T*>     weights_ptr  = std::vector<T*>(weights_num_);
    std::vector<size_t> weights_size = std::vector<size_t>(weights_num_);

    // prompt learning pair (task_name, (task_name_id, prompt_len))
    PromptLearningType                         prompt_learning_type_;
    std::map<std::string, std::pair<int, int>> prompt_learning_pair_;
    bool                                       malloc_load_prompt_weights_ = false;
    // each prompt token's weight size
    size_t prompt_token_weight_size_ = 0;
};

}  // namespace fastertransformer