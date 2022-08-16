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

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptWeight.h"

namespace fastertransformer {

template<typename T>
ParallelGptWeight<T>::ParallelGptWeight(const int                                  hidden_units,
                                        const int                                  inter_size,
                                        const int                                  vocab_size,
                                        const int                                  num_layer,
                                        const int                                  max_seq_len,
                                        const int                                  tensor_para_size,
                                        const int                                  tensor_para_rank,
                                        const int                                  layer_para_size,
                                        const int                                  layer_para_rank,
                                        const int                                  int8_mode,
                                        PromptLearningType                         prompt_learning_type,
                                        std::map<std::string, std::pair<int, int>> prompt_learning_pair,
                                        gptVariantParams                           gpt_variant_params):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    vocab_size_(vocab_size),
    num_layer_(num_layer),
    max_seq_len_(max_seq_len),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    layer_para_size_(layer_para_size),
    layer_para_rank_(layer_para_rank),
    int8_mode_(int8_mode),
    prompt_learning_type_(prompt_learning_type),
    prompt_learning_pair_(prompt_learning_pair),
    gpt_variant_params_(gpt_variant_params)
{
    FT_CHECK(num_layer_ % layer_para_size_ == 0);
    // set prompt weight size
    if (prompt_learning_type_ == PromptLearningType::prefix_prompt) {
        prompt_token_weight_size_ = 2 * num_layer_ * hidden_units_ / tensor_para_size_;
    }
    else if (prompt_learning_type_ == PromptLearningType::p_prompt_tuning) {
        prompt_token_weight_size_ = hidden_units_;
    }

    // set if load and malloc prompt weights
    malloc_load_prompt_weights_ = !prompt_learning_pair_.empty()
                                  && (prompt_learning_type_ == PromptLearningType::p_prompt_tuning
                                      || prompt_learning_type_ == PromptLearningType::prefix_prompt);
    decoder_layer_weights.clear();
    decoder_layer_weights.reserve(num_layer_);
    for (int l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l)) {
            decoder_layer_weights.push_back(new ParallelGptDecoderLayerWeight<T>(
                hidden_units_, inter_size_, tensor_para_size_, tensor_para_rank_, int8_mode_, gpt_variant_params_));
        }
        else {
            // Don't malloc and load these layers since we don't use them.
            decoder_layer_weights.push_back(new ParallelGptDecoderLayerWeight<T>());
        }
    }

    mallocWeights();
    setWeightPtr();
}

template<typename T>
ParallelGptWeight<T>::~ParallelGptWeight()
{
    if (is_maintain_buffer == true) {
        for (int i = 0; i < weights_ptr.size(); i++) {
            deviceFree(weights_ptr[i]);
        }

        position_encoding_table       = nullptr;
        pre_decoder_embedding_table   = nullptr;
        post_decoder_layernorm.beta   = nullptr;
        post_decoder_layernorm.gamma  = nullptr;
        post_decoder_embedding.kernel = nullptr;
        post_decoder_embedding.bias   = nullptr;
        is_maintain_buffer            = false;
    }

    for (int i = 0; i < num_layer_; i++) {
        delete decoder_layer_weights[i];
    }
}

template<typename T>
ParallelGptWeight<T>::ParallelGptWeight(const ParallelGptWeight& other):
    hidden_units_(other.hidden_units_),
    inter_size_(other.inter_size_),
    vocab_size_(other.vocab_size_),
    num_layer_(other.num_layer_),
    max_seq_len_(other.max_seq_len_),
    tensor_para_size_(other.tensor_para_size_),
    tensor_para_rank_(other.tensor_para_rank_),
    layer_para_size_(other.layer_para_size_),
    layer_para_rank_(other.layer_para_rank_),
    int8_mode_(other.int8_mode_),
    prompt_token_weight_size_(other.prompt_token_weight_size_),
    malloc_load_prompt_weights_(other.malloc_load_prompt_weights_),
    prompt_learning_type_(other.prompt_learning_type_),
    prompt_learning_pair_(other.prompt_learning_pair_),
    gpt_variant_params_(other.gpt_variant_params_)
{
    mallocWeights();
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], max_seq_len_ * vocab_size_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], vocab_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], hidden_units_);
    cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ * vocab_size_);

    // prompt learning table: malloc weights and set weight ptr
    if (malloc_load_prompt_weights_) {
        for (auto const& prompt : prompt_learning_pair_) {
            std::string task_name     = prompt.first;
            int         task_name_id  = prompt.second.first;
            int         prompt_length = prompt.second.second;
            size_t      prompt_id     = num_base_weights + (size_t)task_name_id;

            // cuda device to device memcpy prompt table weights buffer memory
            cudaD2Dcpy(weights_ptr[prompt_id], other.weights_ptr[prompt_id], prompt_length * prompt_token_weight_size_);
        }
    }

    setWeightPtr();

    decoder_layer_weights.clear();
    decoder_layer_weights.reserve(num_layer_);
    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights.push_back(other.decoder_layer_weights[l]);
    }
}

template<typename T>
ParallelGptWeight<T>& ParallelGptWeight<T>::operator=(const ParallelGptWeight& other)
{
    hidden_units_               = other.hidden_units_;
    inter_size_                 = other.inter_size_;
    num_layer_                  = other.num_layer_;
    vocab_size_                 = other.vocab_size_;
    max_seq_len_                = other.max_seq_len_;
    tensor_para_size_           = other.tensor_para_size_;
    tensor_para_rank_           = other.tensor_para_rank_;
    layer_para_size_            = other.layer_para_size_;
    layer_para_rank_            = other.layer_para_rank_;
    int8_mode_                  = other.int8_mode_;
    prompt_token_weight_size_   = other.prompt_token_weight_size_;
    malloc_load_prompt_weights_ = other.malloc_load_prompt_weights_;
    prompt_learning_type_       = other.prompt_learning_type_;
    prompt_learning_pair_       = other.prompt_learning_pair_;
    gpt_variant_params_         = other.gpt_variant_params_;

    mallocWeights();
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], max_seq_len_ * vocab_size_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], vocab_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], hidden_units_);
    cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ * vocab_size_);

    // prompt learning tables: malloc weights and set weight ptr
    if (malloc_load_prompt_weights_) {
        for (auto const& prompt : prompt_learning_pair_) {
            int    task_name_id  = prompt.second.first;
            int    prompt_length = prompt.second.second;
            size_t prompt_id     = num_base_weights + (size_t)task_name_id;

            // cuda device to device memcpy prompt weights buffer memory
            cudaD2Dcpy(weights_ptr[prompt_id], other.weights_ptr[prompt_id], prompt_length * prompt_token_weight_size_);
        }
    }

    setWeightPtr();

    decoder_layer_weights.clear();
    decoder_layer_weights.reserve(num_layer_);
    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights.push_back(other.decoder_layer_weights[l]);
    }
    return *this;
}

template<typename T>
void ParallelGptWeight<T>::setWeightPtr()
{
    prompt_learning_table.resize(prompt_learning_pair_.size());

    position_encoding_table       = weights_ptr[0];
    pre_decoder_embedding_table   = weights_ptr[1];
    post_decoder_layernorm.beta   = weights_ptr[2];
    post_decoder_layernorm.gamma  = weights_ptr[3];
    post_decoder_embedding.kernel = weights_ptr[4];
    post_decoder_embedding.bias   = nullptr;

    // prompt learning tables: set weight ptr
    if (malloc_load_prompt_weights_) {
        for (auto const& prompt : prompt_learning_pair_) {
            int    task_name_id   = prompt.second.first;
            int    prompt_length  = prompt.second.second;
            size_t task_weight_id = num_base_weights + (size_t)task_name_id;

            // set weight ptr
            prompt_learning_table[task_name_id] = {weights_ptr[task_weight_id], prompt_length};
        }
    }
}

template<typename T>
void ParallelGptWeight<T>::mallocWeights()
{
    weights_ptr.resize(num_base_weights + prompt_learning_pair_.size());

    deviceMalloc(&weights_ptr[0], max_seq_len_ * vocab_size_);
    deviceMalloc(&weights_ptr[1], vocab_size_ * hidden_units_);
    deviceMalloc(&weights_ptr[2], hidden_units_);
    deviceMalloc(&weights_ptr[3], hidden_units_);
    deviceMalloc(&weights_ptr[4], hidden_units_ * vocab_size_);

    // prompt learning tables: malloc weights
    if (malloc_load_prompt_weights_) {
        for (auto const& prompt : prompt_learning_pair_) {
            int    task_name_id   = prompt.second.first;
            int    prompt_length  = prompt.second.second;
            size_t task_weight_id = num_base_weights + (size_t)task_name_id;

            // malloc weights
            T* prompt_weights_ptr = nullptr;
            deviceMalloc(&prompt_weights_ptr, prompt_length * prompt_token_weight_size_);
            weights_ptr[task_weight_id] = prompt_weights_ptr;
        }
    }
    is_maintain_buffer = true;
}

template<typename T>
void ParallelGptWeight<T>::loadModel(std::string dir_path)
{
    FtCudaDataType model_file_type = getModelFileType(dir_path + "/config.ini", "gpt");
    FT_CHECK(is_maintain_buffer == true);
    loadWeightFromBin<T>(weights_ptr[0], {max_seq_len_, hidden_units_}, dir_path + "/model.wpe.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[1], {vocab_size_ * hidden_units_}, dir_path + "/model.wte.bin", model_file_type);
    if (gpt_variant_params_.has_post_decoder_layernorm) {
        loadWeightFromBin<T>(
            weights_ptr[2], {hidden_units_}, dir_path + "/model.final_layernorm.bias.bin", model_file_type);
        loadWeightFromBin<T>(
            weights_ptr[3], {hidden_units_}, dir_path + "/model.final_layernorm.weight.bin", model_file_type);
    }
    if (checkIfFileExist(dir_path + "/model.lm_head.weight.bin")) {
        loadWeightFromBin<T>(
            weights_ptr[4], {vocab_size_ * hidden_units_}, dir_path + "/model.lm_head.weight.bin", model_file_type);
    }
    else {
        loadWeightFromBin<T>(
            weights_ptr[4], {vocab_size_ * hidden_units_}, dir_path + "/model.wte.bin", model_file_type);
    }

    // prompt table: load weights from bin
    if (malloc_load_prompt_weights_) {
        for (auto const& prompt : prompt_learning_pair_) {
            std::string task_name      = prompt.first;
            int         task_name_id   = prompt.second.first;
            int         prompt_length  = prompt.second.second;
            size_t      task_weight_id = num_base_weights + (size_t)task_name_id;

            std::string prompt_weight_path_name = (prompt_learning_type_ == PromptLearningType::p_prompt_tuning) ?
                                                      (dir_path + "/model.prompt_table." + task_name + ".weight.bin") :
                                                      (dir_path + "/model.prefix_prompt." + task_name + ".weight."
                                                       + std::to_string(tensor_para_rank_) + ".bin");

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
            decoder_layer_weights[l]->loadModel(dir_path + "/model.layers." + std::to_string(l), model_file_type);
        }
    }
}

template<typename T>
bool ParallelGptWeight<T>::isValidLayerParallelId(int l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / layer_para_size_));
    return l < num_layer_ && (l >= local_num_layer * layer_para_rank_)
           && (l < local_num_layer * (layer_para_rank_ + 1));
}

template<typename T>
void ParallelGptWeight<T>::resizeLayer(const int num_layer, const int int8_mode)
{
    int8_mode_ = int8_mode;
    num_layer_ = num_layer;
    decoder_layer_weights.reserve(num_layer_);
    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights.push_back(new ParallelGptDecoderLayerWeight<T>(int8_mode_));
    }
}

#ifdef SPARSITY_ENABLED
template<typename T>
void ParallelGptWeight<T>::compress_weights(cublasMMWrapper& cublas_wrapper)
{
    FT_CHECK(decoder_layer_weights.size() == static_cast<size_t>(num_layer_));
    for (int i = 0; i < num_layer_; ++i) {
        if (isValidLayerParallelId(i)) {
            decoder_layer_weights[i]->compress_weights(cublas_wrapper, hidden_units_);
        }
    }
}
#endif

template struct ParallelGptWeight<float>;
template struct ParallelGptWeight<half>;
#ifdef ENABLE_BF16
template struct ParallelGptWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
