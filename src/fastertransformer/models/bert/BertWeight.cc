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

#include "BertWeight.h"

namespace fastertransformer {

template<typename T>
BertWeight<T>::BertWeight(const size_t hidden_units,
                          const size_t inter_size,
                          const size_t num_layer,
                          const size_t tensor_para_size,
                          const size_t tensor_para_rank,
                          const size_t pipeline_para_size,
                          const size_t pipeline_para_rank):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    num_layer_(num_layer),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    pipeline_para_size_(pipeline_para_size),
    pipeline_para_rank_(pipeline_para_rank)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    deviceMalloc(&weights_ptr[0], hidden_units_);
    deviceMalloc(&weights_ptr[1], hidden_units_);

    setWeightPtr();
    bert_layer_weights.reserve(num_layer_);
    for (int i = 0; i < num_layer_; i++) {
        bert_layer_weights.push_back(
            BertLayerWeight<T>(hidden_units_, inter_size_, tensor_para_size_, tensor_para_rank_));
    }
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
BertWeight<T>::~BertWeight()
{
    if (is_maintain_buffer == true) {
        bert_layer_weights.clear();
        for (int i = 0; i < 2; i++) {
            deviceFree(weights_ptr[i]);
        }

        post_transformer_layernorm_weights.gamma = nullptr;
        post_transformer_layernorm_weights.beta  = nullptr;
        is_maintain_buffer                       = false;
    }
}

template<typename T>
BertWeight<T>::BertWeight(const BertWeight& other):
    BertWeight(other.hidden_units_,
               other.inter_size_,
               other.num_layer_,
               other.tensor_para_size_,
               other.tensor_para_rank_,
               other.pipeline_para_size_,
               other.pipeline_para_rank_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    bert_layer_weights.clear();
    bert_layer_weights.reserve(num_layer_);
    for (int i = 0; i < num_layer_; i++) {
        bert_layer_weights.push_back(other.bert_layer_weights[i]);
    }
    deviceMalloc(&weights_ptr[0], hidden_units_);
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
    deviceMalloc(&weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);

    setWeightPtr();
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
BertWeight<T>& BertWeight<T>::operator=(const BertWeight& other)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    hidden_units_       = other.hidden_units_;
    inter_size_         = other.inter_size_;
    num_layer_          = other.num_layer_;
    tensor_para_size_   = other.tensor_para_size_;
    tensor_para_rank_   = other.tensor_para_rank_;
    pipeline_para_size_ = other.pipeline_para_size_;
    pipeline_para_rank_ = other.pipeline_para_rank_;

    bert_layer_weights.clear();
    bert_layer_weights.reserve(num_layer_);
    for (int i = 0; i < num_layer_; i++) {
        bert_layer_weights.push_back(other.bert_layer_weights[i]);
    }
    deviceMalloc(&weights_ptr[0], hidden_units_);
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
    deviceMalloc(&weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);

    setWeightPtr();
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);

    return *this;
}

template<typename T>
void BertWeight<T>::loadModel(std::string dir_path)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FtCudaDataType model_file_type = getModelFileType(dir_path + "/config.ini", "bert");
    for (uint l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l)) {
            bert_layer_weights[l].loadModel(dir_path + "model.encoder.layer." + std::to_string(l) + ".",
                                            model_file_type);
        }
    }
    FT_LOG_DEBUG(__PRETTY_FUNCTION__, " stop");
}

template<typename T>
bool BertWeight<T>::isValidLayerParallelId(int l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_size_));
    return l < num_layer_ && (l >= local_num_layer * pipeline_para_rank_)
           && (l < local_num_layer * (pipeline_para_rank_ + 1));
}

template<typename T>
void BertWeight<T>::setWeightPtr()
{
    post_transformer_layernorm_weights.gamma = weights_ptr[0];
    post_transformer_layernorm_weights.beta  = weights_ptr[1];

    is_maintain_buffer = true;
}

template struct BertWeight<float>;
template struct BertWeight<half>;
#ifdef ENABLE_BF16
template struct BertWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
