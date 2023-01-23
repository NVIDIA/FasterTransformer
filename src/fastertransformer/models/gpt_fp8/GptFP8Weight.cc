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

#include "src/fastertransformer/models/gpt_fp8/GptFP8Weight.h"

namespace fastertransformer {

template<typename T1, typename T2>
GptFP8Weight<T1, T2>::GptFP8Weight(const int hidden_units,
                                   const int inter_size,
                                   const int vocab_size,
                                   const int num_layer,
                                   const int max_seq_len,
                                   const int tensor_para_size,
                                   const int tensor_para_rank,
                                   const int layer_para_size,
                                   const int layer_para_rank):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    vocab_size_(vocab_size),
    num_layer_(num_layer),
    max_seq_len_(max_seq_len),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    layer_para_size_(layer_para_size),
    layer_para_rank_(layer_para_rank),
    table_ptr(3),
    vec_ptr(2)
{
    decoder_layer_weights.clear();
    for (int l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l)) {
            decoder_layer_weights.push_back(
                new GptFP8DecoderLayerWeight<T1, T2>(hidden_units_, inter_size_, tensor_para_size_, tensor_para_rank_));
        }
        else {
            // Don't malloc and load these layers since we don't use them.
            decoder_layer_weights.push_back(new GptFP8DecoderLayerWeight<T1, T2>());
        }
    }

    mallocWeights();
    setWeightPtr();
}

template<typename T1, typename T2>
GptFP8Weight<T1, T2>::~GptFP8Weight()
{
    if (is_maintain_buffer == true) {
        for (int i = 0; i < table_ptr.size(); ++i) {
            table_ptr[i].first = 0;
            deviceFree(table_ptr[i].second);
        }

        for (int i = 0; i < vec_ptr.size(); ++i) {
            vec_ptr[i].first = 0;
            deviceFree(vec_ptr[i].second);
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

template<typename T1, typename T2>
GptFP8Weight<T1, T2>::GptFP8Weight(const GptFP8Weight& other):
    hidden_units_(other.hidden_units_),
    inter_size_(other.inter_size_),
    vocab_size_(other.vocab_size_),
    num_layer_(other.num_layer_),
    max_seq_len_(other.max_seq_len_),
    tensor_para_size_(other.tensor_para_size_),
    tensor_para_rank_(other.tensor_para_rank_),
    layer_para_size_(other.layer_para_size_),
    layer_para_rank_(other.layer_para_rank_),
    table_ptr(3),
    vec_ptr(2)
{
    mallocWeights();
    for (int i = 0; i < table_ptr.size(); ++i) {
        cudaD2Dcpy(table_ptr[i].second, other.table_ptr[i].second, table_ptr[i].first);
    }
    for (int i = 0; i < table_ptr.size(); ++i) {
        cudaD2Dcpy(vec_ptr[i].second, other.vec_ptr[i].second, vec_ptr[i].first);
    }
    setWeightPtr();

    decoder_layer_weights.clear();
    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights.push_back(other.decoder_layer_weights[l]);
    }
}

template<typename T1, typename T2>
GptFP8Weight<T1, T2>& GptFP8Weight<T1, T2>::operator=(const GptFP8Weight& other)
{
    hidden_units_     = other.hidden_units_;
    inter_size_       = other.inter_size_;
    num_layer_        = other.num_layer_;
    vocab_size_       = other.vocab_size_;
    max_seq_len_      = other.max_seq_len_;
    tensor_para_size_ = other.tensor_para_size_;
    tensor_para_rank_ = other.tensor_para_rank_;
    layer_para_size_  = other.layer_para_size_;
    layer_para_rank_  = other.layer_para_rank_;

    mallocWeights();
    for (int i = 0; i < table_ptr.size(); ++i) {
        cudaD2Dcpy(table_ptr[i].second, other.table_ptr[i].second, table_ptr[i].first);
    }
    for (int i = 0; i < table_ptr.size(); ++i) {
        cudaD2Dcpy(vec_ptr[i].second, other.vec_ptr[i].second, vec_ptr[i].first);
    }
    setWeightPtr();

    decoder_layer_weights.clear();
    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights.push_back(other.decoder_layer_weights[l]);
    }
    return *this;
}

template<typename T1, typename T2>
void GptFP8Weight<T1, T2>::setWeightPtr()
{
    position_encoding_table       = table_ptr[0].second;
    pre_decoder_embedding_table   = table_ptr[1].second;
    post_decoder_layernorm.beta   = vec_ptr[0].second;
    post_decoder_layernorm.gamma  = vec_ptr[1].second;
    post_decoder_embedding.kernel = table_ptr[2].second;
    post_decoder_embedding.bias   = nullptr;
}

template<typename T1, typename T2>
void GptFP8Weight<T1, T2>::mallocWeights()
{
    table_ptr[0].first = max_seq_len_ * hidden_units_;
    table_ptr[1].first = vocab_size_ * hidden_units_;
    table_ptr[2].first = hidden_units_ * vocab_size_;
    vec_ptr[0].first   = hidden_units_;
    vec_ptr[1].first   = hidden_units_;

    deviceMalloc(&table_ptr[0].second, table_ptr[0].first);
    deviceMalloc(&table_ptr[1].second, table_ptr[1].first);
    deviceMalloc(&table_ptr[2].second, table_ptr[2].first);
    deviceMalloc(&vec_ptr[0].second, vec_ptr[0].first);
    deviceMalloc(&vec_ptr[1].second, vec_ptr[1].first);
    is_maintain_buffer = true;
}

template<typename T1, typename T2>
void GptFP8Weight<T1, T2>::loadModel(std::string dir_path)
{
    FT_CHECK(is_maintain_buffer == true);

    loadWeightFromBin<T2>(table_ptr[0].second, {table_ptr[0].first}, dir_path + "/model.wpe.bin");
    loadWeightFromBin<T2>(table_ptr[1].second, {table_ptr[1].first}, dir_path + "/model.wte.bin");
    loadWeightFromBin<T2>(vec_ptr[0].second, {vec_ptr[0].first}, dir_path + "/model.final_layernorm.bias.bin");
    loadWeightFromBin<T2>(vec_ptr[1].second, {vec_ptr[1].first}, dir_path + "/model.final_layernorm.weight.bin");
    loadWeightFromBin<T2>(table_ptr[2].second, {table_ptr[2].first}, dir_path + "/model.wte.bin");

    for (int l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l)) {
            decoder_layer_weights[l]->loadModel(dir_path + "/model.layers." + std::to_string(l));
        }
    }
}

template<typename T1, typename T2>
bool GptFP8Weight<T1, T2>::isValidLayerParallelId(int l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / layer_para_size_));
    return l < num_layer_ && (l >= local_num_layer * layer_para_rank_)
           && (l < local_num_layer * (layer_para_rank_ + 1));
}

template<typename T1, typename T2>
void GptFP8Weight<T1, T2>::resizeLayer(const int num_layer)
{
    num_layer_ = num_layer;
    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights.push_back(new GptFP8DecoderLayerWeight<T1, T2>());
    }
}

#ifdef SPARSITY_ENABLED
template<typename T1, typename T2>
void GptFP8Weight<T1, T2>::compress_weights(cublasMMWrapper& cublas_wrapper)
{
    FT_CHECK(decoder_layer_weights.size() == static_cast<size_t>(num_layer_));
    for (int i = 0; i < num_layer_; ++i) {
        if (isValidLayerParallelId(i)) {
            decoder_layer_weights[i]->compress_weights(cublas_wrapper, hidden_units_);
        }
    }
}
#endif

template<typename T1, typename T2>
void GptFP8Weight<T1, T2>::transposeWeight()
{
    // only transpose the weight of transformer layer
    for (int i = 0; i < num_layer_; ++i) {
        if (isValidLayerParallelId(i)) {
            decoder_layer_weights[i]->transposeWeight();
        }
    }
}

template struct GptFP8Weight<__nv_fp8_e4m3, __nv_bfloat16>;

}  // namespace fastertransformer
