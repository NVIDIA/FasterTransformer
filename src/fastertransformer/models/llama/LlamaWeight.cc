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

#include "src/fastertransformer/models/llama/LlamaWeight.h"

namespace fastertransformer {

template<typename T>
LlamaWeight<T>::LlamaWeight(const int                                  hidden_units,
                            const int                                  inter_size,
                            const int                                  vocab_size,
                            const int                                  num_layer,
                            const int                                  max_seq_len,
                            const int                                  tensor_para_size,
                            const int                                  tensor_para_rank,
                            const int                                  layer_para_size,
                            const int                                  layer_para_rank,
                            const bool                                 use_gptj_residual,
                            PromptLearningType                         prompt_learning_type,
                            std::map<std::string, std::pair<int, int>> prompt_learning_pair):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    vocab_size_(vocab_size),
    num_layer_(num_layer),
    max_seq_len_(max_seq_len),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    layer_para_size_(layer_para_size),
    layer_para_rank_(layer_para_rank),
    use_gptj_residual_(use_gptj_residual),
    prompt_learning_type_(prompt_learning_type),
    prompt_learning_pair_(prompt_learning_pair)
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

    decoder_layer_weights.reserve(num_layer_);
    for (int l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l)) {
            decoder_layer_weights.push_back(new LlamaDecoderLayerWeight<T>(
                hidden_units_, inter_size_, tensor_para_size_, tensor_para_rank_, use_gptj_residual_));
        }
        else {
            // Layer-parallelism: allocate empty layer because
            // this rank does not compute it:
            decoder_layer_weights.push_back(new LlamaDecoderLayerWeight<T>(0, 0));
        }
    }

    mallocWeights();
    setWeightPtr();
}

template<typename T>
LlamaWeight<T>::~LlamaWeight()
{
    if (is_maintain_buffer == true) {
        for (int i = 0; i < weights_ptr.size(); i++) {
            deviceFree(weights_ptr[i]);
        }

        pre_decoder_embedding_table   = nullptr;
        post_decoder_layernorm.beta   = nullptr;
        post_decoder_layernorm.gamma  = nullptr;
        post_decoder_embedding.kernel = nullptr;
        is_maintain_buffer            = false;
    }
}

template<typename T>
LlamaWeight<T>::LlamaWeight(const LlamaWeight& other):
    hidden_units_(other.hidden_units_),
    inter_size_(other.inter_size_),
    vocab_size_(other.vocab_size_),
    num_layer_(other.num_layer_),
    max_seq_len_(other.max_seq_len_),
    tensor_para_size_(other.tensor_para_size_),
    tensor_para_rank_(other.tensor_para_rank_),
    layer_para_size_(other.layer_para_size_),
    layer_para_rank_(other.layer_para_rank_),
    use_gptj_residual_(other.use_gptj_residual_),
    prompt_token_weight_size_(other.prompt_token_weight_size_),
    malloc_load_prompt_weights_(other.malloc_load_prompt_weights_),
    prompt_learning_type_(other.prompt_learning_type_),
    prompt_learning_pair_(other.prompt_learning_pair_)
{
    mallocWeights();
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], vocab_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], hidden_units_ * vocab_size_);

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
LlamaWeight<T>& LlamaWeight<T>::operator=(const LlamaWeight& other)
{
    hidden_units_               = other.hidden_units_;
    inter_size_                 = other.inter_size_;
    vocab_size_                 = other.vocab_size_;
    num_layer_                  = other.num_layer_;
    max_seq_len_                = other.max_seq_len_;
    tensor_para_size_           = other.tensor_para_size_;
    tensor_para_rank_           = other.tensor_para_rank_;
    layer_para_size_            = other.layer_para_size_;
    layer_para_rank_            = other.layer_para_rank_;
    use_gptj_residual_          = other.use_gptj_residual_;
    prompt_token_weight_size_   = other.prompt_token_weight_size_;
    malloc_load_prompt_weights_ = other.malloc_load_prompt_weights_;
    prompt_learning_type_       = other.prompt_learning_type_;
    prompt_learning_pair_       = other.prompt_learning_pair_;

    mallocWeights();
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], vocab_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], hidden_units_ * vocab_size_);

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
    return *this;
}

template<typename T>
void LlamaWeight<T>::setWeightPtr()
{
    prompt_learning_table.resize(prompt_learning_pair_.size());

    pre_decoder_embedding_table   = weights_ptr[0];
    post_decoder_layernorm.beta   = weights_ptr[1];
    post_decoder_layernorm.gamma  = weights_ptr[2];
    post_decoder_embedding.kernel = weights_ptr[3];

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
void LlamaWeight<T>::mallocWeights()
{
    weights_ptr.resize(num_base_weights + prompt_learning_pair_.size());

    deviceMalloc(&weights_ptr[0], vocab_size_ * hidden_units_);
    deviceMalloc(&weights_ptr[1], hidden_units_);
    deviceMalloc(&weights_ptr[2], hidden_units_);
    deviceMalloc(&weights_ptr[3], hidden_units_ * vocab_size_);

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
void LlamaWeight<T>::loadModel(std::string dir_path)
{
    FtCudaDataType model_file_type = getModelFileType(dir_path + "/config.ini", "llama");
    FT_CHECK(is_maintain_buffer == true);

    loadWeightFromBin<T>(
        weights_ptr[0], {(size_t)(vocab_size_ * hidden_units_)}, dir_path + "/model.wte.weight.bin", model_file_type);
    deviceFill(weights_ptr[1], (size_t)hidden_units_, (T)0.0);
    loadWeightFromBin<T>(
        weights_ptr[2], {(size_t)hidden_units_}, dir_path + "/model.final_layernorm.weight.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[3],
                         {(size_t)(vocab_size_ * hidden_units_)},
                         dir_path + "/model.lm_head.weight.bin",
                         model_file_type);

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
                                     {(size_t)(prompt_length * (int)prompt_token_weight_size_)},
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
void LlamaWeight<T>::resizeLayer(const int num_layer)
{
    num_layer_ = num_layer;
    decoder_layer_weights.reserve(num_layer_);
    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights.push_back(new LlamaDecoderLayerWeight<T>());
    }
}

template<typename T>
bool LlamaWeight<T>::isValidLayerParallelId(int l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / layer_para_size_));
    return l < num_layer_ && (l >= local_num_layer * layer_para_rank_)
           && (l < local_num_layer * (layer_para_rank_ + 1));
}

template struct LlamaWeight<float>;
template struct LlamaWeight<half>;
#ifdef ENABLE_BF16
template class LlamaWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
