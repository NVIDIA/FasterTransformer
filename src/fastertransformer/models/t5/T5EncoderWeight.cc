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

#include "src/fastertransformer/models/t5/T5EncoderWeight.h"
#include "src/fastertransformer/utils/logger.h"

namespace fastertransformer {

template<typename T>
T5EncoderWeight<T>::T5EncoderWeight(const size_t head_num,
                                    const size_t size_per_head,
                                    const size_t d_model,
                                    const size_t inter_size,
                                    const size_t vocab_size,
                                    const size_t num_layer,
                                    const size_t num_bucket,
                                    const size_t tensor_para_size,
                                    const size_t tensor_para_rank,
                                    const size_t pipeline_para_size,
                                    const size_t pipeline_para_rank):
    head_num_(head_num),
    size_per_head_(size_per_head),
    d_model_(d_model),
    inter_size_(inter_size),
    vocab_size_(vocab_size),
    num_layer_(num_layer),
    num_bucket_(num_bucket),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    pipeline_para_size_(pipeline_para_size),
    pipeline_para_rank_(pipeline_para_rank)
{
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " start");
    initialize();
    mallocWeights();
    setWeightPtr();
    t5_encoder_layer_weights.clear();
    for (int l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l)) {
            t5_encoder_layer_weights.push_back(
                new T5EncoderLayerWeight<T>(head_num_, size_per_head, d_model_, inter_size_, tensor_para_size_, tensor_para_rank_));
        }
        else {
            // Don't malloc and load these layers since we don't use them.
            t5_encoder_layer_weights.push_back(new T5EncoderLayerWeight<T>());
        }
    }
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
void T5EncoderWeight<T>::initialize()
{
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " start");
    weights_size[0] = d_model_;
    weights_size[1] = (head_num_ / tensor_para_size_) * num_bucket_;
    weights_size[2] = d_model_ * vocab_size_;
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " end");
}


template<typename T>
T5EncoderWeight<T>::~T5EncoderWeight()
{
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " start");
    
    if (is_maintain_buffer == true) {
        for (int i = 0; i < weights_num_; i++) {
            deviceFree(weights_ptr[i]);
        }

        post_transformer_layernorm_weights.gamma = nullptr;
        relative_attention_bias = nullptr;
        embedding_table = nullptr;
        is_maintain_buffer = false;
    }
    for (int i = 0; i < num_layer_; i++) {
        delete t5_encoder_layer_weights[i];
    }
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
T5EncoderWeight<T>::T5EncoderWeight(const T5EncoderWeight& other):
    head_num_(other.head_num_),
    size_per_head_(other.size_per_head_),
    d_model_(other.d_model_),
    inter_size_(other.inter_size_),
    vocab_size_(other.vocab_size_),
    num_layer_(other.num_layer_),
    num_bucket_(other.num_bucket_),
    tensor_para_size_(other.tensor_para_size_),
    tensor_para_rank_(other.tensor_para_rank_),
    pipeline_para_size_(other.pipeline_para_size_),
    pipeline_para_rank_(other.pipeline_para_rank_)
{
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " start");
    initialize();
    mallocWeights();
    for (int i = 0; i < weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    setWeightPtr();

    t5_encoder_layer_weights.clear();
    for (int i = 0; i < num_layer_; i++) {
        t5_encoder_layer_weights.push_back(new T5EncoderLayerWeight<T>(*other.t5_encoder_layer_weights[i]));
    }
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
T5EncoderWeight<T>& T5EncoderWeight<T>::operator=(const T5EncoderWeight& other)
{
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " start");

    head_num_ = other.head_num_;
    size_per_head_ = other.size_per_head_;
    d_model_ = other.d_model_;
    inter_size_ = other.inter_size_;
    vocab_size_ = other.vocab_size_;
    num_layer_ = other.num_layer_;
    num_bucket_ = other.num_bucket_;
    tensor_para_size_ = other.tensor_para_size_;
    tensor_para_rank_ = other.tensor_para_rank_;
    pipeline_para_size_ = other.pipeline_para_size_;
    pipeline_para_rank_ = other.pipeline_para_rank_;
    initialize();
    mallocWeights();
    for (int i = 0; i < weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    setWeightPtr();

    t5_encoder_layer_weights.clear();
    for (int i = 0; i < num_layer_; i++) {
        t5_encoder_layer_weights.push_back(new T5EncoderLayerWeight<T>(*other.t5_encoder_layer_weights[i]));
    }
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " end");

    return *this;
}

template<typename T>
void T5EncoderWeight<T>::setWeightPtr()
{
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " start");
    post_transformer_layernorm_weights.gamma = weights_ptr[0];
    relative_attention_bias = weights_ptr[1];
    embedding_table = weights_ptr[2];
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " end");

}

template<typename T>
void T5EncoderWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " start");
    for (int i = 0; i < weights_num_; i++) {
        deviceMalloc(&weights_ptr[i], weights_size[i]);
    }
    is_maintain_buffer = true;
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
void T5EncoderWeight<T>::loadModel(std::string dir_path)
{
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " start");
    FT_CHECK(is_maintain_buffer == true);

    loadWeightFromBin<T>(weights_ptr[0], {weights_size[0]}, dir_path + "/encoder.final_layer_norm.weight.bin");
    loadWeightFromBin<T>(weights_ptr[1],
                         {weights_size[1]},
                         dir_path + "/encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight."
                             + std::to_string(tensor_para_rank_) + ".bin");
    loadWeightFromBin<T>(weights_ptr[2], {weights_size[2]}, dir_path + "/shared.weight_T.bin");

    for (int l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l)) {
            t5_encoder_layer_weights[l]->loadModel(dir_path + "/encoder.block." + std::to_string(l) + ".");
        }
    }
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
bool T5EncoderWeight<T>::isValidLayerParallelId(int l)
{
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " start");
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_size_));
    return l < num_layer_ && (l >= local_num_layer * pipeline_para_rank_)
           && (l < local_num_layer * (pipeline_para_rank_ + 1));
}

template<typename T>
void T5EncoderWeight<T>::resizeLayer(const int num_layer)
{
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " start");
    t5_encoder_layer_weights.clear();
    num_layer_ = num_layer;
    for (int l = 0; l < num_layer_; l++) {
        t5_encoder_layer_weights.push_back(new T5EncoderLayerWeight<T>());
    }
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " end");
}

template struct T5EncoderWeight<float>;
template struct T5EncoderWeight<half>;

}  // namespace fastertransformer
