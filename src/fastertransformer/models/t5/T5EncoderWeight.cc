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

#include "src/fastertransformer/models/t5/T5EncoderWeight.h"
#include "src/fastertransformer/utils/logger.h"

namespace fastertransformer {

template<typename T>
T5EncoderWeight<T>::T5EncoderWeight(const size_t                               head_num,
                                    const size_t                               size_per_head,
                                    const size_t                               d_model,
                                    const size_t                               inter_size,
                                    const size_t                               vocab_size,
                                    const size_t                               num_layer,
                                    const size_t                               num_bucket_or_max_seq_len,
                                    const size_t                               tensor_para_size,
                                    const size_t                               tensor_para_rank,
                                    const size_t                               pipeline_para_size,
                                    const size_t                               pipeline_para_rank,
                                    const bool                                 t5_with_bias_para,
                                    const bool                                 use_gated_activation_para,
                                    const PositionEmbeddingType                pe_type,
                                    PromptLearningType                         prompt_learning_type,
                                    std::map<std::string, std::pair<int, int>> prompt_learning_pair,
                                    const size_t                               ia3_num_tasks,
                                    const size_t                               adapter_inter_size):
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
    t5_with_bias(t5_with_bias_para),
    use_gated_activation(use_gated_activation_para),
    position_embedding_type(pe_type),
    prompt_learning_type_(prompt_learning_type),
    prompt_learning_pair_(prompt_learning_pair),
    real_weights_num_(t5_with_bias ? 4 : 3),
    ia3_num_tasks_(ia3_num_tasks),
    adapter_inter_size_{adapter_inter_size}
{
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " start");
    FT_CHECK(num_layer_ % pipeline_para_size_ == 0);
    initialize();
    mallocWeights();
    setWeightPtr();
    t5_encoder_layer_weights.clear();
    t5_encoder_layer_weights.reserve(num_layer_);
    for (int l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l)) {
            t5_encoder_layer_weights.push_back(new T5EncoderLayerWeight<T>(head_num_,
                                                                           size_per_head,
                                                                           d_model_,
                                                                           inter_size_,
                                                                           tensor_para_size_,
                                                                           tensor_para_rank_,
                                                                           t5_with_bias,
                                                                           use_gated_activation,
                                                                           ia3_num_tasks_,
                                                                           adapter_inter_size_));
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
    if (prompt_learning_type_ == PromptLearningType::prefix_prompt) {
        prompt_token_weight_size_ = 2 * num_layer_ * head_num_ * size_per_head_ / tensor_para_size_;
    }
    else if (prompt_learning_type_ == PromptLearningType::p_prompt_tuning) {
        prompt_token_weight_size_ = d_model_;
    }
    malloc_load_prompt_weights_ = !prompt_learning_pair_.empty()
                                  && (prompt_learning_type_ == PromptLearningType::p_prompt_tuning
                                      || prompt_learning_type_ == PromptLearningType::prefix_prompt);
    weights_ptr.reserve(weights_num_ + prompt_learning_pair_.size());
    weights_size.reserve(weights_num_ + prompt_learning_pair_.size());

    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " start");
    weights_size[0] = d_model_;
    if (position_embedding_type == PositionEmbeddingType::absolute) {
        weights_size[1] = num_bucket_or_max_seq_len_ * d_model_;
    }
    else {
        weights_size[1] = (head_num_ / tensor_para_size_) * num_bucket_or_max_seq_len_;
    }
    weights_size[2] = d_model_ * vocab_size_;
    weights_size[3] = d_model_;

    if (malloc_load_prompt_weights_) {
        for (auto const& prompt : prompt_learning_pair_) {
            int    task_name_id   = prompt.second.first;
            int    prompt_length  = prompt.second.second;
            size_t task_weight_id = weights_num_ + (size_t)task_name_id;

            // set weight size
            weights_size[task_weight_id] = prompt_length * prompt_token_weight_size_;
        }
    }
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
T5EncoderWeight<T>::~T5EncoderWeight()
{
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " start");

    if (is_maintain_buffer == true) {
        for (int i = 0; i < real_weights_num_; i++) {
            deviceFree(weights_ptr[i]);
        }

        post_transformer_layernorm_weights.gamma = nullptr;
        absolute_or_relative_position_embedding  = nullptr;
        embedding_table                          = nullptr;
        post_transformer_layernorm_weights.beta  = nullptr;

        if (malloc_load_prompt_weights_) {
            for (auto const& prompt : prompt_learning_pair_) {
                int    task_name_id   = prompt.second.first;
                size_t task_weight_id = weights_num_ + (size_t)task_name_id;

                deviceFree(weights_ptr[task_weight_id]);
            }
        }
        prompt_learning_table.clear();

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
    num_bucket_or_max_seq_len_(other.num_bucket_or_max_seq_len_),
    tensor_para_size_(other.tensor_para_size_),
    tensor_para_rank_(other.tensor_para_rank_),
    pipeline_para_size_(other.pipeline_para_size_),
    pipeline_para_rank_(other.pipeline_para_rank_),
    t5_with_bias(other.t5_with_bias),
    use_gated_activation(other.use_gated_activation),
    position_embedding_type(other.position_embedding_type),
    prompt_learning_type_(other.prompt_learning_type_),
    prompt_learning_pair_(other.prompt_learning_pair_),
    real_weights_num_(other.real_weights_num_),
    ia3_num_tasks_(other.ia3_num_tasks_),
    adapter_inter_size_(other.adapter_inter_size_)
{
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " start");
    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    // prompt learning table: malloc weights and set weight ptr
    if (malloc_load_prompt_weights_) {
        for (auto const& prompt : prompt_learning_pair_) {
            std::string task_name    = prompt.first;
            int         task_name_id = prompt.second.first;
            size_t      prompt_id    = weights_num_ + (size_t)task_name_id;

            // cuda device to device memcpy prompt table weights buffer memory
            cudaD2Dcpy(weights_ptr[prompt_id], other.weights_ptr[prompt_id], weights_size[prompt_id]);
        }
    }

    setWeightPtr();

    t5_encoder_layer_weights.clear();
    t5_encoder_layer_weights.reserve(num_layer_);
    for (int i = 0; i < num_layer_; i++) {
        t5_encoder_layer_weights.push_back(new T5EncoderLayerWeight<T>(*other.t5_encoder_layer_weights[i]));
    }
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
T5EncoderWeight<T>& T5EncoderWeight<T>::operator=(const T5EncoderWeight& other)
{
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " start");

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
    t5_with_bias               = other.t5_with_bias;
    use_gated_activation       = other.use_gated_activation;
    position_embedding_type    = other.position_embedding_type;
    prompt_learning_type_      = other.prompt_learning_type_;
    prompt_learning_pair_      = other.prompt_learning_pair_;
    real_weights_num_          = other.real_weights_num_;
    ia3_num_tasks_             = other.ia3_num_tasks_;
    adapter_inter_size_        = other.adapter_inter_size_;
    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    // prompt learning table: malloc weights and set weight ptr
    if (malloc_load_prompt_weights_) {
        for (auto const& prompt : prompt_learning_pair_) {
            std::string task_name    = prompt.first;
            int         task_name_id = prompt.second.first;
            size_t      prompt_id    = weights_num_ + (size_t)task_name_id;

            // cuda device to device memcpy prompt table weights buffer memory
            cudaD2Dcpy(weights_ptr[prompt_id], other.weights_ptr[prompt_id], weights_size[prompt_id]);
        }
    }

    setWeightPtr();

    t5_encoder_layer_weights.clear();
    t5_encoder_layer_weights.reserve(num_layer_);
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
    absolute_or_relative_position_embedding  = weights_ptr[1];
    embedding_table                          = weights_ptr[2];
    if (t5_with_bias) {
        post_transformer_layernorm_weights.beta = weights_ptr[3];
    }

    prompt_learning_table.resize(prompt_learning_pair_.size());
    // prompt learning tables: set weight ptr
    if (malloc_load_prompt_weights_) {
        for (auto const& prompt : prompt_learning_pair_) {
            int    task_name_id   = prompt.second.first;
            int    prompt_length  = prompt.second.second;
            size_t task_weight_id = weights_num_ + (size_t)task_name_id;

            // set weight ptr
            prompt_learning_table[task_name_id] = {weights_ptr[task_weight_id], prompt_length};
        }
    }
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
void T5EncoderWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " start");
    for (int i = 0; i < real_weights_num_; i++) {
        deviceMalloc(&weights_ptr[i], weights_size[i]);
    }
    // prompt learning tables: malloc weights
    if (malloc_load_prompt_weights_) {
        for (auto const& prompt : prompt_learning_pair_) {
            int    task_name_id   = prompt.second.first;
            size_t task_weight_id = weights_num_ + (size_t)task_name_id;

            // malloc weights
            deviceMalloc(&weights_ptr[task_weight_id], weights_size[task_weight_id]);
        }
    }
    is_maintain_buffer = true;
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
void T5EncoderWeight<T>::loadModel(std::string dir_path)
{
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " start");
    FtCudaDataType model_file_type = getModelFileType(dir_path + "/config.ini", "encoder");
    FT_CHECK(is_maintain_buffer == true);

    loadWeightFromBin<T>(
        weights_ptr[0], {(size_t)weights_size[0]}, dir_path + "/encoder.final_layer_norm.weight.bin", model_file_type);
    if (position_embedding_type == PositionEmbeddingType::absolute) {
        loadWeightFromBin<T>(weights_ptr[1], {(size_t)weights_size[1]}, dir_path + "/shared.ape.bin", model_file_type);
    }
    else {
        loadWeightFromBin<T>(weights_ptr[1],
                             {(size_t)weights_size[1]},
                             dir_path + "/encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight."
                                 + std::to_string(tensor_para_rank_) + ".bin",
                             model_file_type);
    }
    loadWeightFromBin<T>(weights_ptr[2], {(size_t)weights_size[2]}, dir_path + "/shared.weight_T.bin", model_file_type);
    if (t5_with_bias) {
        loadWeightFromBin<T>(weights_ptr[3],
                             {(size_t)weights_size[3]},
                             dir_path + "/encoder.final_layer_norm.bias.bin",
                             model_file_type);
    }

    // prompt table: load weights from bin
    if (malloc_load_prompt_weights_) {
        for (auto const& prompt : prompt_learning_pair_) {
            std::string task_name      = prompt.first;
            int         task_name_id   = prompt.second.first;
            int         prompt_length  = prompt.second.second;
            size_t      task_weight_id = weights_num_ + (size_t)task_name_id;

            std::string prompt_weight_path_name = (prompt_learning_type_ == PromptLearningType::p_prompt_tuning) ?
                                                      (dir_path + "/model.prompt_table." + task_name + ".weight.bin") :
                                                      (dir_path + "/model.prefix_prompt." + task_name + ".weight."
                                                       + std::to_string(tensor_para_rank_) + ".bin");
            FT_LOG_DEBUG("load prompt_weight_path_name: %s", prompt_weight_path_name.c_str());
            if (prompt_length > 0) {
                loadWeightFromBin<T>(weights_ptr[task_weight_id],
                                     {prompt_length * prompt_token_weight_size_},
                                     prompt_weight_path_name,
                                     model_file_type);
            }
        }
    }

    for (int l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l)) {
            t5_encoder_layer_weights[l]->loadModel(dir_path + "/encoder.block." + std::to_string(l) + ".",
                                                   model_file_type);
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
    t5_encoder_layer_weights.reserve(num_layer_);
    for (int l = 0; l < num_layer_; l++) {
        t5_encoder_layer_weights.push_back(new T5EncoderLayerWeight<T>());
    }
    FT_LOG_DEBUG("T5EncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
void T5EncoderWeight<T>::setT5StructureDiff(bool                  t5_with_bias_para,
                                            bool                  use_gated_activation_para,
                                            PositionEmbeddingType position_embedding_type_para)
{
    t5_with_bias            = t5_with_bias_para;
    position_embedding_type = position_embedding_type_para;
    use_gated_activation    = use_gated_activation_para;
    for (int i = 0; i < num_layer_; i++) {
        t5_encoder_layer_weights[i]->setT5WithBias(t5_with_bias_para, use_gated_activation);
    }
}

template struct T5EncoderWeight<float>;
template struct T5EncoderWeight<half>;
#ifdef ENABLE_BF16
template struct T5EncoderWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
