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

#include "src/fastertransformer/models/m2m/M2MEncoderWeight.h"
#include "src/fastertransformer/utils/logger.h"

namespace fastertransformer {

template<typename T>
M2MEncoderWeight<T>::M2MEncoderWeight(const size_t                head_num,
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
                                        const bool                  m2m_with_bias_para,
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
    m2m_with_bias(m2m_with_bias_para),
    mbart(mbart_para),
    use_gated_activation(use_gated_activation_para),
    position_embedding_type(pe_type)
{
    // 2: absolute/relative positional embedding weight, word
    // embedding weight. mBART has two LN, BART has one LN
    real_weights_num_ = 2 + (mbart ? 2 : 1) * (m2m_with_bias ? 2 : 1);

    FT_LOG_DEBUG("M2MEncoderWeight " + std::string(__func__) + " start");
    FT_CHECK(num_layer_ % pipeline_para_size_ == 0);
    initialize();
    mallocWeights();
    setWeightPtr();
    m2m_encoder_layer_weights.clear();
    m2m_encoder_layer_weights.reserve(num_layer_);
    for (int l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l)) {
            m2m_encoder_layer_weights.push_back(new M2MEncoderLayerWeight<T>(head_num_,
                                                                               size_per_head,
                                                                               d_model_,
                                                                               inter_size_,
                                                                               tensor_para_size_,
                                                                               tensor_para_rank_,
                                                                               m2m_with_bias,
                                                                               use_gated_activation));
        }
        else {
            // Don't malloc and load these layers since we don't use them.
            m2m_encoder_layer_weights.push_back(new M2MEncoderLayerWeight<T>());
        }
    }
    FT_LOG_DEBUG("M2MEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
void M2MEncoderWeight<T>::initialize()
{
    FT_LOG_DEBUG("M2MEncoderWeight " + std::string(__func__) + " start");

    if (position_embedding_type == PositionEmbeddingType::absolute) {
        weights_size[0] = num_bucket_or_max_seq_len_ * d_model_;
    }
    else {
        weights_size[0] = (head_num_ / tensor_para_size_) * num_bucket_or_max_seq_len_;
    }
    weights_size[1] = d_model_ * vocab_size_;
    weights_size[2] = d_model_;
    if (mbart || m2m_with_bias) {
        if (mbart && m2m_with_bias) {
            weights_size[3] = d_model_;
            weights_size[4] = d_model_;
            weights_size[5] = d_model_;
        }
        else if (mbart && !m2m_with_bias) {
            weights_size[3] = d_model_;
        }
        else if (!mbart && m2m_with_bias) {
            weights_size[3] = d_model_;
        }
    }  // if none of the flags is on, there are only 3 weights

    FT_LOG_DEBUG("M2MEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
M2MEncoderWeight<T>::~M2MEncoderWeight()
{
    FT_LOG_DEBUG("M2MEncoderWeight " + std::string(__func__) + " start");

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
        delete m2m_encoder_layer_weights[i];
    }
    FT_LOG_DEBUG("M2MEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
M2MEncoderWeight<T>::M2MEncoderWeight(const M2MEncoderWeight& other):
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
    m2m_with_bias(other.m2m_with_bias),
    mbart(other.mbart),
    use_gated_activation(other.use_gated_activation),
    position_embedding_type(other.position_embedding_type),
    real_weights_num_(other.real_weights_num_)
{
    FT_LOG_DEBUG("M2MEncoderWeight " + std::string(__func__) + " start");
    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    setWeightPtr();

    m2m_encoder_layer_weights.clear();
    m2m_encoder_layer_weights.reserve(num_layer_);
    for (int i = 0; i < num_layer_; i++) {
        m2m_encoder_layer_weights.push_back(new M2MEncoderLayerWeight<T>(*other.m2m_encoder_layer_weights[i]));
    }
    FT_LOG_DEBUG("M2MEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
M2MEncoderWeight<T>& M2MEncoderWeight<T>::operator=(const M2MEncoderWeight& other)
{
    FT_LOG_DEBUG("M2MEncoderWeight " + std::string(__func__) + " start");

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
    m2m_with_bias             = other.m2m_with_bias;
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

    m2m_encoder_layer_weights.clear();
    m2m_encoder_layer_weights.reserve(num_layer_);
    for (int i = 0; i < num_layer_; i++) {
        m2m_encoder_layer_weights.push_back(new M2MEncoderLayerWeight<T>(*other.m2m_encoder_layer_weights[i]));
    }
    FT_LOG_DEBUG("M2MEncoderWeight " + std::string(__func__) + " end");

    return *this;
}

template<typename T>
void M2MEncoderWeight<T>::setWeightPtr()
{
    FT_LOG_DEBUG("M2MEncoderWeight " + std::string(__func__) + " start");

    absolute_or_relative_position_embedding = weights_ptr[0];
    embedding_table                         = weights_ptr[1];
    pre_transformer_layernorm_weights.gamma = weights_ptr[2];
    if (mbart || m2m_with_bias) {
        if (mbart && m2m_with_bias) {
            pre_transformer_layernorm_weights.beta   = weights_ptr[3];
            post_transformer_layernorm_weights.gamma = weights_ptr[4];
            post_transformer_layernorm_weights.beta  = weights_ptr[5];
        }
        else if (mbart && !m2m_with_bias) {
            post_transformer_layernorm_weights.gamma = weights_ptr[3];
        }
        else if (!mbart && m2m_with_bias) {
            pre_transformer_layernorm_weights.beta = weights_ptr[3];
        }
    }

    FT_LOG_DEBUG("M2MEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
void M2MEncoderWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG("M2MEncoderWeight " + std::string(__func__) + " start");
    for (int i = 0; i < real_weights_num_; i++) {
        deviceMalloc(&weights_ptr[i], weights_size[i]);
    }
    is_maintain_buffer = true;
    FT_LOG_DEBUG("M2MEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
void M2MEncoderWeight<T>::loadModel(std::string dir_path)
{
    FT_LOG_DEBUG("M2MEncoderWeight " + std::string(__func__) + " start");

    FtCudaDataType model_file_type = getModelFileType(dir_path + "/config.ini", "encoder");
    FT_CHECK(is_maintain_buffer == true);

    loadWeightFromBin<T>(weights_ptr[0], {(size_t)weights_size[0]}, dir_path + "/encoder.embed_positions.weight.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[1], {(size_t)weights_size[1]}, dir_path + "/encoder.embed_tokens.weight.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[2], {(size_t)weights_size[2]}, dir_path + "/encoder.final_layer_norm.weight.bin", model_file_type);
    if (m2m_with_bias) {
        if (mbart) {
            loadWeightFromBin<T>(weights_ptr[3],
                                {(size_t)weights_size[3]},
                                dir_path + "/encoder.final_layer_norm.bias.bin",
                                model_file_type);
            loadWeightFromBin<T>(weights_ptr[4],
                                {(size_t)weights_size[4]},
                                dir_path + "/encoder.layer_norm.weight.bin",
                                model_file_type);
            loadWeightFromBin<T>(weights_ptr[5],
                                {(size_t)weights_size[5]},
                                dir_path + "/encoder.layer_norm.bias.bin",
                                model_file_type);    
        } else {
            loadWeightFromBin<T>(weights_ptr[3],
                                {(size_t)weights_size[3]},
                                dir_path + "/encoder.final_layer_norm.bias.bin",
                                model_file_type);
        }
    }

    for (int l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l)) {
            m2m_encoder_layer_weights[l]->loadModel(dir_path + "/encoder." + std::to_string(l) + ".",
                                                   model_file_type);
        }
    }

    FT_LOG_DEBUG("M2MEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
bool M2MEncoderWeight<T>::isValidLayerParallelId(int l)
{
    FT_LOG_DEBUG("M2MEncoderWeight " + std::string(__func__) + " start");
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_size_));
    return l < num_layer_ && (l >= local_num_layer * pipeline_para_rank_)
           && (l < local_num_layer * (pipeline_para_rank_ + 1));
}

template<typename T>
void M2MEncoderWeight<T>::resizeLayer(const int num_layer)
{
    FT_LOG_DEBUG("M2MEncoderWeight " + std::string(__func__) + " start");
    m2m_encoder_layer_weights.clear();
    num_layer_ = num_layer;
    m2m_encoder_layer_weights.reserve(num_layer_);
    for (int l = 0; l < num_layer_; l++) {
        m2m_encoder_layer_weights.push_back(new M2MEncoderLayerWeight<T>());
    }
    FT_LOG_DEBUG("M2MEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
void M2MEncoderWeight<T>::setM2MStructureDiff(bool                  m2m_with_bias_para,
                                                bool                  mbart_para,
                                                bool                  use_gated_activation_para,
                                                PositionEmbeddingType position_embedding_type_para)
{
    m2m_with_bias          = m2m_with_bias_para;
    mbart                   = mbart_para;
    position_embedding_type = position_embedding_type_para;
    use_gated_activation    = use_gated_activation_para;
    for (int i = 0; i < num_layer_; i++) {
        m2m_encoder_layer_weights[i]->setM2MWithBias(m2m_with_bias_para, use_gated_activation);
    }
}

template struct M2MEncoderWeight<float>;
template struct M2MEncoderWeight<half>;
#ifdef ENABLE_BF16
template struct M2MEncoderWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
