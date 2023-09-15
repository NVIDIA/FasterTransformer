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

#include "src/fastertransformer/models/llama/LLaMAWeight.h"

namespace fastertransformer {

template<typename T>
LLaMAWeight<T>::LLaMAWeight(const int hidden_units,
                            const int inter_size,
                            const int vocab_size,
                            const int num_layer,
                            const int layer_para_size,
                            const int layer_para_rank):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    vocab_size_(vocab_size),
    num_layer_(num_layer),
    layer_para_size_(layer_para_size),
    layer_para_rank_(layer_para_rank)
{
    FT_CHECK(num_layer_ % layer_para_size_ == 0);

    decoder_layer_weights.reserve(num_layer_);
    for (int l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l)) {
            decoder_layer_weights.push_back(new LLaMADecoderLayerWeight<T>(hidden_units_, inter_size_));
        }
        else {
            // Layer-parallelism: allocate empty layer because
            // this rank does not compute it:
            decoder_layer_weights.push_back(new LLaMADecoderLayerWeight<T>(0, 0));
        }
    }

    mallocWeights();
    setWeightPtr();
}

template<typename T>
LLaMAWeight<T>::~LLaMAWeight()
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
LLaMAWeight<T>::LLaMAWeight(const LLaMAWeight& other):
    hidden_units_(other.hidden_units_),
    inter_size_(other.inter_size_),
    vocab_size_(other.vocab_size_),
    num_layer_(other.num_layer_),
    layer_para_size_(other.layer_para_size_),
    layer_para_rank_(other.layer_para_rank_),
    prompt_token_weight_size_(other.prompt_token_weight_size_)
{
    mallocWeights();
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], vocab_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], hidden_units_ * vocab_size_);

    // prompt learning table: malloc weights and set weight ptr
    setWeightPtr();

    decoder_layer_weights.clear();
    decoder_layer_weights.reserve(num_layer_);
    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights.push_back(other.decoder_layer_weights[l]);
    }
}

template<typename T>
LLaMAWeight<T>& LLaMAWeight<T>::operator=(const LLaMAWeight& other)
{
    hidden_units_             = other.hidden_units_;
    inter_size_               = other.inter_size_;
    vocab_size_               = other.vocab_size_;
    num_layer_                = other.num_layer_;
    layer_para_size_          = other.layer_para_size_;
    layer_para_rank_          = other.layer_para_rank_;
    prompt_token_weight_size_ = other.prompt_token_weight_size_;

    mallocWeights();
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], vocab_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], hidden_units_ * vocab_size_);

    setWeightPtr();

    decoder_layer_weights.clear();
    decoder_layer_weights.reserve(num_layer_);
    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights.push_back(other.decoder_layer_weights[l]);
    }
    return *this;
}

template<typename T>
void LLaMAWeight<T>::setWeightPtr()
{
    pre_decoder_embedding_table   = weights_ptr[0];
    post_decoder_layernorm.beta   = weights_ptr[1];
    post_decoder_layernorm.beta   = nullptr;
    post_decoder_layernorm.gamma  = weights_ptr[2];
    post_decoder_embedding.kernel = weights_ptr[3];
}

template<typename T>
void LLaMAWeight<T>::mallocWeights()
{
    weights_ptr.resize(num_base_weights);

    deviceMalloc(&weights_ptr[0], vocab_size_ * hidden_units_);
    deviceMalloc(&weights_ptr[1], hidden_units_);
    deviceMalloc(&weights_ptr[2], hidden_units_);
    deviceMalloc(&weights_ptr[3], hidden_units_ * vocab_size_);

    is_maintain_buffer = true;
}

template<typename T>
void LLaMAWeight<T>::loadModel(std::string dir_path)
{
    FtCudaDataType model_file_type = getModelFileType(dir_path + "/config.ini", "llama");
    FT_CHECK(is_maintain_buffer == true);

    loadWeightFromBin<T>(weights_ptr[0],
                         {(size_t)(vocab_size_ * hidden_units_)},
                         dir_path + "/model.tok_embeddings.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[1], {(size_t)hidden_units_}, dir_path + "/model.norm.bias.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[2], {(size_t)hidden_units_}, dir_path + "/model.norm.weight.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[3],
                         {(size_t)(vocab_size_ * hidden_units_)},
                         dir_path + "/model.output.weight.bin",
                         model_file_type);

    for (int l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l)) {
            decoder_layer_weights[l]->loadModel(dir_path + "/model.layers." + std::to_string(l), model_file_type);
        }
    }
}

template<typename T>
void LLaMAWeight<T>::resizeLayer(const int num_layer)
{
    num_layer_ = num_layer;
    decoder_layer_weights.reserve(num_layer_);
    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights.push_back(new LLaMADecoderLayerWeight<T>());
    }
}

template<typename T>
bool LLaMAWeight<T>::isValidLayerParallelId(int l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / layer_para_size_));
    return l < num_layer_ && (l >= local_num_layer * layer_para_rank_)
           && (l < local_num_layer * (layer_para_rank_ + 1));
}

template struct LLaMAWeight<float>;
template struct LLaMAWeight<half>;

}  // namespace fastertransformer
