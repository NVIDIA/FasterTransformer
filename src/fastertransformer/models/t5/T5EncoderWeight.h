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

#include "src/fastertransformer/kernels/gen_relative_pos_bias.h"
#include "src/fastertransformer/models/t5/T5EncoderLayerWeight.h"

namespace fastertransformer {

template<typename T>
struct T5EncoderWeight {

    T5EncoderWeight() = default;
    T5EncoderWeight(const size_t head_num,
                    const size_t size_per_head,
                    const size_t d_model,
                    const size_t inter_size,
                    const size_t vocab_size,
                    const size_t num_layer,
                    const size_t num_bucket_or_max_seq_len,
                    const size_t tensor_para_size,
                    const size_t tensor_para_rank,
                    const size_t pipeline_para_size,
                    const size_t pipeline_para_rank,
                    const bool t5_with_bias_para = false,
                    const PositionEmbeddingType pe_type = PositionEmbeddingType::relative);
    ~T5EncoderWeight();
    T5EncoderWeight(const T5EncoderWeight& other);
    T5EncoderWeight& operator=(const T5EncoderWeight& other);

    std::vector<T5EncoderLayerWeight<T>*> t5_encoder_layer_weights;
    LayerNormWeight<T> post_transformer_layernorm_weights;
    T* absolute_or_relative_position_embedding = nullptr;
    T* embedding_table = nullptr;
    bool t5_with_bias = false;
    PositionEmbeddingType position_embedding_type = PositionEmbeddingType::relative;

    void loadModel(std::string dir_path);
    void resizeLayer(const int num_layer);
    void setT5StructureDiff(bool t5_with_bias_para, PositionEmbeddingType position_embedding_type_para);

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
    // refer to max_seq_len if using absoulte position embedding
    size_t num_bucket_or_max_seq_len_;
    size_t tensor_para_size_;
    size_t tensor_para_rank_;
    size_t pipeline_para_size_;
    size_t pipeline_para_rank_;

    bool is_maintain_buffer = false;

    int real_weights_num_;

    const static int weights_num_ = 4;
    T* weights_ptr[weights_num_];
    size_t weights_size[weights_num_];
};

}  // namespace fastertransformer