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

#include "src/fastertransformer/models/gpt/GptWeight.h"

namespace fastertransformer {

template<typename T>
GptWeight<T>::GptWeight(
    const int hidden_units, const int inter_size, const int vocab_size, const int num_layer, const int max_seq_len):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    vocab_size_(vocab_size),
    num_layer_(num_layer),
    max_seq_len_(max_seq_len)
{
    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights.push_back(GptDecoderLayerWeight<T>(hidden_units_, inter_size_));
    }

    mallocWeights();
    setWeightPtr();
}

template<typename T>
GptWeight<T>::~GptWeight()
{
    if (is_maintain_buffer == true) {
        for (int i = 0; i < 5; i++) {
            deviceFree(weights_ptr[i]);
        }

        position_encoding_table = nullptr;
        pre_decoder_embedding_table = nullptr;
        post_decoder_layernorm.beta = nullptr;
        post_decoder_layernorm.gamma = nullptr;
        post_decoder_embedding.kernel = nullptr;
        post_decoder_embedding.bias = nullptr;
        is_maintain_buffer = false;
    }
}

template<typename T>
GptWeight<T>::GptWeight(const GptWeight& other):
    hidden_units_(other.hidden_units_),
    inter_size_(other.inter_size_),
    vocab_size_(other.vocab_size_),
    num_layer_(other.num_layer_),
    max_seq_len_(other.max_seq_len_)
{
    mallocWeights();
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], max_seq_len_ * vocab_size_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], vocab_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], hidden_units_);
    cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ * vocab_size_);
    setWeightPtr();

    decoder_layer_weights.clear();
    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights.push_back(other.decoder_layer_weights[l]);
    }
}

template<typename T>
GptWeight<T>& GptWeight<T>::operator=(const GptWeight& other)
{
    hidden_units_ = other.hidden_units_;
    inter_size_ = other.inter_size_;
    num_layer_ = other.num_layer_;
    vocab_size_ = other.vocab_size_;
    max_seq_len_ = other.max_seq_len_;

    mallocWeights();
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], max_seq_len_ * vocab_size_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], vocab_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], hidden_units_);
    cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ * vocab_size_);
    setWeightPtr();

    decoder_layer_weights.clear();
    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights.push_back(other.decoder_layer_weights[l]);
    }
    return *this;
}

template<typename T>
void GptWeight<T>::setWeightPtr()
{
    position_encoding_table = weights_ptr[0];
    pre_decoder_embedding_table = weights_ptr[1];
    post_decoder_layernorm.beta = weights_ptr[2];
    post_decoder_layernorm.gamma = weights_ptr[3];
    post_decoder_embedding.kernel = weights_ptr[4];
    post_decoder_embedding.bias = nullptr;
}

template<typename T>
void GptWeight<T>::mallocWeights()
{
    deviceMalloc(&weights_ptr[0], max_seq_len_ * vocab_size_);
    deviceMalloc(&weights_ptr[1], vocab_size_ * hidden_units_);
    deviceMalloc(&weights_ptr[2], hidden_units_);
    deviceMalloc(&weights_ptr[3], hidden_units_);
    deviceMalloc(&weights_ptr[4], hidden_units_ * vocab_size_);
    is_maintain_buffer = true;
}

template<typename T>
void GptWeight<T>::loadModel(std::string dir_path)
{
    FT_CHECK(is_maintain_buffer == true);

    loadWeightFromBin<T>(weights_ptr[0], {max_seq_len_, hidden_units_}, dir_path + "/model.wpe.bin");
    loadWeightFromBin<T>(weights_ptr[1], {vocab_size_ * hidden_units_}, dir_path + "/model.wte.bin");
    loadWeightFromBin<T>(weights_ptr[2], {hidden_units_}, dir_path + "/model.final_layernorm.bias.bin");
    loadWeightFromBin<T>(weights_ptr[3], {hidden_units_}, dir_path + "/model.final_layernorm.weight.bin");
    loadWeightFromBin<T>(weights_ptr[4], {vocab_size_ * hidden_units_}, dir_path + "/model.wte.bin");

    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights[l].loadModel(dir_path + "/model.layers." + std::to_string(l));
    }
}

#ifdef SPARSITY_ENABLED
template<typename T>
void GptWeight<T>::compress_weights(cublasMMWrapper& cublas_wrapper)
{
    // Assertion to prevent invalid attributes. By now, gpt_weight may not
    // have proper attribute values, because one can directly modify decoder
    // layer weights from outside.
    FT_CHECK(decoder_layer_weights.size() == static_cast<size_t>(num_layer_));
    for (int i = 0; i < num_layer_; ++i) {
        decoder_layer_weights[i].compress_weights(cublas_wrapper, hidden_units_);
    }
}
#endif

template struct GptWeight<float>;
template struct GptWeight<half>;

}  // namespace fastertransformer
