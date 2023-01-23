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

#include "src/fastertransformer/models/bart/BartEncoderWeight.h"
#include "src/fastertransformer/utils/logger.h"

namespace fastertransformer {

template<typename T>
BartEncoderWeight<T>::BartEncoderWeight(const size_t                head_num,
                                        const size_t                size_per_head,
                                        const size_t                d_model,
                                        const size_t                inter_size,
                                        const size_t                vocab_size,
                                        const size_t                num_layer,
                                        const size_t                num_bucket_or_max_seq_len,
                                        const size_t                tensor_para_size,
                                        const size_t                tensor_para_rank,
                                        const size_t                pipeline_para_size,
                                        const size_t                pipeline_para_rank,
                                        const bool                  bart_with_bias_para,
                                        const bool                  mbart_para,
                                        const bool                  use_gated_activation_para,
                                        const PositionEmbeddingType pe_type):
    head_num_(head_num),
    size_per_head_(size_per_head),
    d_model_(d_model),
    inter_size_(inter_size),
    vocab_size_(vocab_size),
    num_layer_(num_layer),
    num_bucket_or_max_seq_len_(num_bucket_or_max_seq_len),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    pipeline_para_size_(pipeline_para_size),
    pipeline_para_rank_(pipeline_para_rank),
    bart_with_bias(bart_with_bias_para),
    mbart(mbart_para),
    use_gated_activation(use_gated_activation_para),
    position_embedding_type(pe_type)
{
    // 2: absolute/relative positional embedding weight, word
    // embedding weight. mBART has two LN, BART has one LN
    real_weights_num_ = 2 + (mbart ? 2 : 1) * (bart_with_bias ? 2 : 1);

    FT_LOG_DEBUG("BartEncoderWeight " + std::string(__func__) + " start");
    FT_CHECK(num_layer_ % pipeline_para_size_ == 0);
    initialize();
    mallocWeights();
    setWeightPtr();
    bart_encoder_layer_weights.clear();
    bart_encoder_layer_weights.reserve(num_layer_);
    for (int l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l)) {
            bart_encoder_layer_weights.push_back(new BartEncoderLayerWeight<T>(head_num_,
                                                                               size_per_head,
                                                                               d_model_,
                                                                               inter_size_,
                                                                               tensor_para_size_,
                                                                               tensor_para_rank_,
                                                                               bart_with_bias,
                                                                               use_gated_activation));
        }
        else {
            // Don't malloc and load these layers since we don't use them.
            bart_encoder_layer_weights.push_back(new BartEncoderLayerWeight<T>());
        }
    }
    FT_LOG_DEBUG("BartEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
void BartEncoderWeight<T>::initialize()
{
    FT_LOG_DEBUG("BartEncoderWeight " + std::string(__func__) + " start");

    if (position_embedding_type == PositionEmbeddingType::absolute) {
        weights_size[0] = num_bucket_or_max_seq_len_ * d_model_;
    }
    else {
        weights_size[0] = (head_num_ / tensor_para_size_) * num_bucket_or_max_seq_len_;
    }
    weights_size[1] = d_model_ * vocab_size_;
    weights_size[2] = d_model_;
    if (mbart || bart_with_bias) {
        if (mbart && bart_with_bias) {
            weights_size[3] = d_model_;
            weights_size[4] = d_model_;
            weights_size[5] = d_model_;
        }
        else if (mbart && !bart_with_bias) {
            weights_size[3] = d_model_;
        }
        else if (!mbart && bart_with_bias) {
            weights_size[3] = d_model_;
        }
    }  // if none of the flags is on, there are only 3 weights

    FT_LOG_DEBUG("BartEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
BartEncoderWeight<T>::~BartEncoderWeight()
{
    FT_LOG_DEBUG("BartEncoderWeight " + std::string(__func__) + " start");

    if (is_maintain_buffer == true) {
        for (int i = 0; i < real_weights_num_; i++) {
            deviceFree(weights_ptr[i]);
        }

        pre_transformer_layernorm_weights.gamma  = nullptr;
        pre_transformer_layernorm_weights.beta   = nullptr;
        absolute_or_relative_position_embedding  = nullptr;
        embedding_table                          = nullptr;
        post_transformer_layernorm_weights.gamma = nullptr;
        post_transformer_layernorm_weights.beta  = nullptr;
        is_maintain_buffer                       = false;
    }
    for (int i = 0; i < num_layer_; i++) {
        delete bart_encoder_layer_weights[i];
    }
    FT_LOG_DEBUG("BartEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
BartEncoderWeight<T>::BartEncoderWeight(const BartEncoderWeight& other):
    head_num_(other.head_num_),
    size_per_head_(other.size_per_head_),
    d_model_(other.d_model_),
    inter_size_(other.inter_size_),
    vocab_size_(other.vocab_size_),
    num_layer_(other.num_layer_),
    num_bucket_or_max_seq_len_(other.num_bucket_or_max_seq_len_),
    tensor_para_size_(other.tensor_para_size_),
    tensor_para_rank_(other.tensor_para_rank_),
    pipeline_para_size_(other.pipeline_para_size_),
    pipeline_para_rank_(other.pipeline_para_rank_),
    bart_with_bias(other.bart_with_bias),
    mbart(other.mbart),
    use_gated_activation(other.use_gated_activation),
    position_embedding_type(other.position_embedding_type),
    real_weights_num_(other.real_weights_num_)
{
    FT_LOG_DEBUG("BartEncoderWeight " + std::string(__func__) + " start");
    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    setWeightPtr();

    bart_encoder_layer_weights.clear();
    bart_encoder_layer_weights.reserve(num_layer_);
    for (int i = 0; i < num_layer_; i++) {
        bart_encoder_layer_weights.push_back(new BartEncoderLayerWeight<T>(*other.bart_encoder_layer_weights[i]));
    }
    FT_LOG_DEBUG("BartEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
BartEncoderWeight<T>& BartEncoderWeight<T>::operator=(const BartEncoderWeight& other)
{
    FT_LOG_DEBUG("BartEncoderWeight " + std::string(__func__) + " start");

    head_num_                  = other.head_num_;
    size_per_head_             = other.size_per_head_;
    d_model_                   = other.d_model_;
    inter_size_                = other.inter_size_;
    vocab_size_                = other.vocab_size_;
    num_layer_                 = other.num_layer_;
    num_bucket_or_max_seq_len_ = other.num_bucket_or_max_seq_len_;
    tensor_para_size_          = other.tensor_para_size_;
    tensor_para_rank_          = other.tensor_para_rank_;
    pipeline_para_size_        = other.pipeline_para_size_;
    pipeline_para_rank_        = other.pipeline_para_rank_;
    bart_with_bias             = other.bart_with_bias;
    mbart                      = other.mbart;
    use_gated_activation       = other.use_gated_activation;
    position_embedding_type    = other.position_embedding_type;
    real_weights_num_          = other.real_weights_num_;
    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    setWeightPtr();

    bart_encoder_layer_weights.clear();
    bart_encoder_layer_weights.reserve(num_layer_);
    for (int i = 0; i < num_layer_; i++) {
        bart_encoder_layer_weights.push_back(new BartEncoderLayerWeight<T>(*other.bart_encoder_layer_weights[i]));
    }
    FT_LOG_DEBUG("BartEncoderWeight " + std::string(__func__) + " end");

    return *this;
}

template<typename T>
void BartEncoderWeight<T>::setWeightPtr()
{
    FT_LOG_DEBUG("BartEncoderWeight " + std::string(__func__) + " start");

    absolute_or_relative_position_embedding = weights_ptr[0];
    embedding_table                         = weights_ptr[1];
    pre_transformer_layernorm_weights.gamma = weights_ptr[2];
    if (mbart || bart_with_bias) {
        if (mbart && bart_with_bias) {
            pre_transformer_layernorm_weights.beta   = weights_ptr[3];
            post_transformer_layernorm_weights.gamma = weights_ptr[4];
            post_transformer_layernorm_weights.beta  = weights_ptr[5];
        }
        else if (mbart && !bart_with_bias) {
            post_transformer_layernorm_weights.gamma = weights_ptr[3];
        }
        else if (!mbart && bart_with_bias) {
            pre_transformer_layernorm_weights.beta = weights_ptr[3];
        }
    }

    FT_LOG_DEBUG("BartEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
void BartEncoderWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG("BartEncoderWeight " + std::string(__func__) + " start");
    for (int i = 0; i < real_weights_num_; i++) {
        deviceMalloc(&weights_ptr[i], weights_size[i]);
    }
    is_maintain_buffer = true;
    FT_LOG_DEBUG("BartEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
void BartEncoderWeight<T>::loadModel(std::string dir_path)
{
    FT_LOG_DEBUG("BartEncoderWeight " + std::string(__func__) + " start");

    FT_LOG_DEBUG("Megatron BART support TBD");

    FT_LOG_DEBUG("BartEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
bool BartEncoderWeight<T>::isValidLayerParallelId(int l)
{
    FT_LOG_DEBUG("BartEncoderWeight " + std::string(__func__) + " start");
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_size_));
    return l < num_layer_ && (l >= local_num_layer * pipeline_para_rank_)
           && (l < local_num_layer * (pipeline_para_rank_ + 1));
}

template<typename T>
void BartEncoderWeight<T>::resizeLayer(const int num_layer)
{
    FT_LOG_DEBUG("BartEncoderWeight " + std::string(__func__) + " start");
    bart_encoder_layer_weights.clear();
    num_layer_ = num_layer;
    bart_encoder_layer_weights.reserve(num_layer_);
    for (int l = 0; l < num_layer_; l++) {
        bart_encoder_layer_weights.push_back(new BartEncoderLayerWeight<T>());
    }
    FT_LOG_DEBUG("BartEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
void BartEncoderWeight<T>::setBartStructureDiff(bool                  bart_with_bias_para,
                                                bool                  mbart_para,
                                                bool                  use_gated_activation_para,
                                                PositionEmbeddingType position_embedding_type_para)
{
    bart_with_bias          = bart_with_bias_para;
    mbart                   = mbart_para;
    position_embedding_type = position_embedding_type_para;
    use_gated_activation    = use_gated_activation_para;
    for (int i = 0; i < num_layer_; i++) {
        bart_encoder_layer_weights[i]->setBartWithBias(bart_with_bias_para, use_gated_activation);
    }
}

template struct BartEncoderWeight<float>;
template struct BartEncoderWeight<half>;
#ifdef ENABLE_BF16
template struct BartEncoderWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
