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

#include "src/fastertransformer/models/bert_fp8/BertFP8Weight.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include <numeric>

namespace fastertransformer {

template<typename T1, typename T2>
BertFP8Weight<T1, T2>::BertFP8Weight(const size_t d_model,
                                     const size_t head_num,
                                     const size_t size_per_head,
                                     const size_t inter_size,
                                     const size_t num_layer,
                                     const size_t vocab_size,
                                     const size_t max_position_embeddings,
                                     const size_t token_type_vocab_size,
                                     const size_t tensor_para_size,
                                     const size_t pipeline_para_size,
                                     const int    fp8_mode,
                                     bool         is_load_model,
                                     bool         is_fused_qkv_gemm):
    d_model_(d_model),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    vocab_size_(vocab_size),
    max_position_embeddings_(max_position_embeddings),
    token_type_vocab_size_(token_type_vocab_size),
    tensor_para_size_(tensor_para_size),
    pipeline_para_size_(pipeline_para_size),
    fp8_mode_(fp8_mode),
    weights_ptr(9)
{
    mallocWeights();
    setWeightPtr();

    bool use_qgmma = false;
#ifdef USE_QGMMA
    use_qgmma = true;
#endif

    bert_layer_weights.clear();
    for (int i = 0; i < num_layer_; i++) {
        bert_layer_weights.push_back(BertFP8LayerWeight<T1, T2>(d_model_,
                                                                head_num_,
                                                                size_per_head_,
                                                                inter_size_,
                                                                tensor_para_size_,
                                                                pipeline_para_size_,
                                                                fp8_mode_,
                                                                is_load_model,
                                                                (is_fused_qkv_gemm && use_qgmma)));
    }
}

template<typename T1, typename T2>
BertFP8Weight<T1, T2>::~BertFP8Weight()
{
    if (is_maintain_buffer == true) {
        bert_layer_weights.clear();
        for (uint i = 0; i < weights_ptr.size(); i++) {
            deviceFree(weights_ptr[i].second);
            weights_ptr[i].first = 0;
        }

        word_embeddings                          = nullptr;  // weights_ptr[0]
        position_embeddings                      = nullptr;  // weights_ptr[1]
        token_type_embeddings                    = nullptr;  // weights_ptr[2]
        embeddings_layernorm.gamma               = nullptr;  // weights_ptr[3]
        embeddings_layernorm.beta                = nullptr;  // weights_ptr[4]
        post_transformer_layernorm_weights.gamma = nullptr;  // weights_ptr[5]
        post_transformer_layernorm_weights.beta  = nullptr;  // weights_ptr[6]
        pooler_dense.kernel                      = nullptr;  // weights_ptr[7]
        pooler_dense.bias                        = nullptr;  // weights_ptr[8]

        is_maintain_buffer = false;
    }
}

template<typename T1, typename T2>
BertFP8Weight<T1, T2>::BertFP8Weight(const BertFP8Weight& other):
    BertFP8Weight(other.d_model_,
                  other.head_num_,
                  other.size_per_head_,
                  other.inter_size_,
                  other.num_layer_,
                  other.vocab_size_,
                  other.max_position_embeddings_,
                  other.token_type_vocab_size_,
                  other.tensor_para_size_,
                  other.pipeline_para_size_,
                  other.fp8_mode_,
                  true)
{
    for (uint i = 0; i < weights_ptr.size(); i++) {
        cudaD2Dcpy(weights_ptr[i].second, other.weights_ptr[i].second, weights_ptr[i].first);
    }
}

template<typename T1, typename T2>
void BertFP8Weight<T1, T2>::transposeWeight()
{
    for (int i = 0; i < bert_layer_weights.size(); i++) {
        bert_layer_weights[i].transposeWeight();
    }

    // We use bfloat 16 with NN for this gemm, don't need to transpose this weight.
    // T2* workspace;
    // deviceMalloc(&workspace, d_model_ * d_model_);
    // invokeInPlaceTranspose(weights_ptr[7].second, workspace, d_model_, d_model_);
    // deviceFree(workspace);
}

template<typename T1, typename T2>
void BertFP8Weight<T1, T2>::loadModel(std::string dir_path)
{
    for (int i = 0; i < bert_layer_weights.size(); i++) {
        bert_layer_weights[i].loadModel(dir_path + "bert.encoder.layer." + std::to_string(i));
    }

    loadWeightFromBin<T2>(
        weights_ptr[0].second, {(int)weights_ptr[0].first}, dir_path + "bert.embeddings.word_embeddings.weight.bin");
    loadWeightFromBin<T2>(weights_ptr[1].second,
                          {(int)weights_ptr[1].first},
                          dir_path + "bert.embeddings.position_embeddings.weight.bin");
    loadWeightFromBin<T2>(weights_ptr[2].second,
                          {(int)weights_ptr[2].first},
                          dir_path + "bert.embeddings.token_type_embeddings.weight.bin");
    loadWeightFromBin<T2>(
        weights_ptr[3].second, {(int)weights_ptr[3].first}, dir_path + "bert.embeddings.LayerNorm.weight.bin");
    loadWeightFromBin<T2>(
        weights_ptr[4].second, {(int)weights_ptr[4].first}, dir_path + "bert.embeddings.LayerNorm.bias.bin");
    // loadWeightFromBin<T2>(weights_ptr[5].second, {(int)weights_ptr[5].first},
    // "bert.embeddings.post_transformer_layernorm_weights.gamma"); loadWeightFromBin<T2>(weights_ptr[6].second,
    // {(int)weights_ptr[6].first}, dir_path + "bert.embeddings.post_transformer_layernorm_weights.beta");
    loadWeightFromBin<T2>(
        weights_ptr[7].second, {(int)weights_ptr[7].first}, dir_path + "bert.pooler.dense.weight.bin");
    loadWeightFromBin<T2>(weights_ptr[8].second, {(int)weights_ptr[8].first}, dir_path + "bert.pooler.dense.bias.bin");
}

template<typename T1, typename T2>
void BertFP8Weight<T1, T2>::serialize(uint8_t*& buffer)
{
    uint8_t* base = buffer;
    for (const auto& layer_weight : bert_layer_weights) {
        layer_weight.serialize(buffer);
    }

    serialize_d2h(buffer, weights_ptr);
}

template<typename T1, typename T2>
void BertFP8Weight<T1, T2>::deserialize(const uint8_t*& buffer)
{
    const uint8_t* base = buffer;
    for (auto& layer_weight : bert_layer_weights) {
        layer_weight.deserialize(buffer);
    }

    deserialize_h2d(buffer, weights_ptr);
}

template<typename T1, typename T2>
size_t BertFP8Weight<T1, T2>::getSerializationSize() const
{
    size_t sz = 0;
    sz        = std::accumulate(std::begin(bert_layer_weights),
                         std::end(bert_layer_weights),
                         sz,
                         [](auto acc, const auto& layer) { return acc + layer.getSerializationSize(); });
    sz = std::accumulate(std::begin(weights_ptr), std::end(weights_ptr), sz, [](auto acc, const auto& weight_ptr) {
        return acc + weight_ptr.first * sizeof(T2);
    });
    return sz;
}

template<typename T1, typename T2>
void BertFP8Weight<T1, T2>::setWeightPtr()
{
    word_embeddings                          = weights_ptr[0].second;
    position_embeddings                      = weights_ptr[1].second;
    token_type_embeddings                    = weights_ptr[2].second;
    embeddings_layernorm.gamma               = weights_ptr[3].second;
    embeddings_layernorm.beta                = weights_ptr[4].second;
    post_transformer_layernorm_weights.gamma = weights_ptr[5].second;
    post_transformer_layernorm_weights.beta  = weights_ptr[6].second;
    pooler_dense.kernel                      = weights_ptr[7].second;
    pooler_dense.bias                        = weights_ptr[8].second;
}

template<typename T1, typename T2>
void BertFP8Weight<T1, T2>::mallocWeights()
{
    weights_ptr[0].first = vocab_size_ * d_model_;
    weights_ptr[1].first = max_position_embeddings_ * d_model_;
    weights_ptr[2].first = token_type_vocab_size_ * d_model_;
    weights_ptr[3].first = d_model_;
    weights_ptr[4].first = d_model_;
    weights_ptr[5].first = d_model_;
    weights_ptr[6].first = d_model_;
    weights_ptr[7].first = d_model_ * d_model_;
    weights_ptr[8].first = d_model_;

    for (uint i = 0; i < weights_ptr.size(); i++) {
        deviceMalloc(&weights_ptr[i].second, weights_ptr[i].first);
    }
}

template struct BertFP8Weight<__nv_fp8_e4m3, __nv_bfloat16>;

}  // namespace fastertransformer
