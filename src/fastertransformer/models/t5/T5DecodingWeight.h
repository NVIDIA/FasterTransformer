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
#include "src/fastertransformer/models/t5/T5DecoderLayerWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
struct T5DecodingWeight {

    T5DecodingWeight() = default;
    T5DecodingWeight(const size_t head_num,
                     const size_t size_per_head,
                     const size_t d_model,
                     const size_t inter_size,
                     const size_t vocab_size,
                     const size_t num_layer,
                     const size_t mem_d_model,
                     const size_t num_bucket,
                     const size_t tensor_para_size,
                     const size_t tensor_para_rank,
                     const size_t pipeline_para_size,
                     const size_t pipeline_para_rank);
    ~T5DecodingWeight();
    T5DecodingWeight(const T5DecodingWeight& other);
    T5DecodingWeight& operator=(const T5DecodingWeight& other);

    std::vector<T5DecoderLayerWeight<T>*> decoder_layer_weights;
    const T* pre_decoder_embedding_table = nullptr;
    const T* relative_attention_bias = nullptr;
    LayerNormWeight<T> post_decoder_layernorm;
    DenseWeight<T> post_decoder_embedding;

    void loadModel(std::string dir_path);
    void resizeLayer(const int num_layer);

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
    size_t num_bucket_;
    size_t tensor_para_size_;
    size_t tensor_para_rank_;
    size_t pipeline_para_size_;
    size_t pipeline_para_rank_;
    bool is_maintain_buffer = false;

    const static int weights_num_ = 4;
    T* weights_ptr[weights_num_];
    size_t weights_size[weights_num_];
};

}  // namespace fastertransformer
