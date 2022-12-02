/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2022.  Authored by Yuqing Ding.
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

#include "src/fastertransformer/models/wenet/WenetDecoderLayerWeight.h"
#include "src/fastertransformer/models/wenet/WenetKernels.h"

namespace fastertransformer {

template<typename T>
WenetDecoderLayerWeight<T>::WenetDecoderLayerWeight(const int layer_id,
                                                    const int hidden_units,
                                                    const int inter_size,
                                                    const int mem_hidden_units):
    layer_id_(layer_id),
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    mem_hidden_units_(mem_hidden_units),
    real_weights_num_(26)
{
    initialize();
    mallocWeights();
    setWeightPtr();
}

template<typename T>
void WenetDecoderLayerWeight<T>::initialize()
{
    int hidden_size = hidden_units_;
    int idx         = 0;

    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size;

    weights_size[idx++] = hidden_size * hidden_size;
    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size * hidden_size;
    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size * hidden_size;
    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size * hidden_size;
    weights_size[idx++] = hidden_size;

    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size;

    weights_size[idx++] = hidden_size * hidden_size;
    weights_size[idx++] = hidden_size;
    weights_size[idx++] = mem_hidden_units_ * hidden_size;
    weights_size[idx++] = hidden_size;
    weights_size[idx++] = mem_hidden_units_ * hidden_size;
    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size * hidden_size;
    weights_size[idx++] = hidden_size;

    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size;

    weights_size[idx++] = hidden_size * inter_size_;
    weights_size[idx++] = inter_size_;
    weights_size[idx++] = hidden_size * inter_size_;
    weights_size[idx++] = hidden_size;
}

template<typename T>
WenetDecoderLayerWeight<T>::~WenetDecoderLayerWeight()
{
    if (is_maintain_buffer == true) {
        for (int i = 0; i < real_weights_num_; i++) {
            deviceFree(weights_ptr[i]);
        }

        pre_layernorm_weights.beta  = nullptr;
        pre_layernorm_weights.gamma = nullptr;

        self_attention_weights.query_weight.kernel            = nullptr;
        self_attention_weights.query_weight.bias              = nullptr;
        self_attention_weights.key_weight.kernel              = nullptr;
        self_attention_weights.key_weight.bias                = nullptr;
        self_attention_weights.value_weight.kernel            = nullptr;
        self_attention_weights.value_weight.bias              = nullptr;
        self_attention_weights.attention_output_weight.kernel = nullptr;
        self_attention_weights.attention_output_weight.bias   = nullptr;

        self_attn_layernorm_weights.beta  = nullptr;
        self_attn_layernorm_weights.gamma = nullptr;

        cross_attention_weights.query_weight.kernel            = nullptr;
        cross_attention_weights.query_weight.bias              = nullptr;
        cross_attention_weights.key_weight.kernel              = nullptr;
        cross_attention_weights.key_weight.bias                = nullptr;
        cross_attention_weights.value_weight.kernel            = nullptr;
        cross_attention_weights.value_weight.bias              = nullptr;
        cross_attention_weights.attention_output_weight.kernel = nullptr;
        cross_attention_weights.attention_output_weight.bias   = nullptr;

        cross_attn_layernorm_weights.beta  = nullptr;
        cross_attn_layernorm_weights.gamma = nullptr;

        ffn_weights.intermediate_weight.kernel = nullptr;
        ffn_weights.intermediate_weight.bias   = nullptr;
        ffn_weights.output_weight.kernel       = nullptr;
        ffn_weights.output_weight.bias         = nullptr;

        is_maintain_buffer = false;
    }
}

template<typename T>
WenetDecoderLayerWeight<T>::WenetDecoderLayerWeight(const WenetDecoderLayerWeight& other):
    layer_id_(other.layer_id_),
    hidden_units_(other.hidden_units_),
    inter_size_(other.inter_size_),
    mem_hidden_units_(other.mem_hidden_units_),
    real_weights_num_(other.real_weights_num_)
{
    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    setWeightPtr();
}

template<typename T>
WenetDecoderLayerWeight<T>& WenetDecoderLayerWeight<T>::operator=(const WenetDecoderLayerWeight& other)
{
    layer_id_         = other.layer_id_;
    hidden_units_     = other.hidden_units_;
    inter_size_       = other.inter_size_;
    mem_hidden_units_ = other.mem_hidden_units_;
    real_weights_num_ = other.real_weights_num_;

    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    setWeightPtr();

    return *this;
}

template<typename T>
void WenetDecoderLayerWeight<T>::setWeightPtr()
{
    int idx = 0;

    pre_layernorm_weights.gamma = weights_ptr[idx++];
    pre_layernorm_weights.beta  = weights_ptr[idx++];

    self_attention_weights.query_weight.kernel            = weights_ptr[idx++];
    self_attention_weights.query_weight.bias              = weights_ptr[idx++];
    self_attention_weights.key_weight.kernel              = weights_ptr[idx++];
    self_attention_weights.key_weight.bias                = weights_ptr[idx++];
    self_attention_weights.value_weight.kernel            = weights_ptr[idx++];
    self_attention_weights.value_weight.bias              = weights_ptr[idx++];
    self_attention_weights.attention_output_weight.kernel = weights_ptr[idx++];
    self_attention_weights.attention_output_weight.bias   = weights_ptr[idx++];

    self_attn_layernorm_weights.gamma = weights_ptr[idx++];
    self_attn_layernorm_weights.beta  = weights_ptr[idx++];

    cross_attention_weights.query_weight.kernel            = weights_ptr[idx++];
    cross_attention_weights.query_weight.bias              = weights_ptr[idx++];
    cross_attention_weights.key_weight.kernel              = weights_ptr[idx++];
    cross_attention_weights.key_weight.bias                = weights_ptr[idx++];
    cross_attention_weights.value_weight.kernel            = weights_ptr[idx++];
    cross_attention_weights.value_weight.bias              = weights_ptr[idx++];
    cross_attention_weights.attention_output_weight.kernel = weights_ptr[idx++];
    cross_attention_weights.attention_output_weight.bias   = weights_ptr[idx++];

    cross_attn_layernorm_weights.gamma = weights_ptr[idx++];
    cross_attn_layernorm_weights.beta  = weights_ptr[idx++];

    ffn_weights.intermediate_weight.kernel = weights_ptr[idx++];
    ffn_weights.intermediate_weight.bias   = weights_ptr[idx++];
    ffn_weights.output_weight.kernel       = weights_ptr[idx++];
    ffn_weights.output_weight.bias         = weights_ptr[idx++];
}

template<typename T>
void WenetDecoderLayerWeight<T>::mallocWeights()
{
    for (int i = 0; i < real_weights_num_; i++) {
        deviceMalloc(&weights_ptr[i], weights_size[i]);
    }
    is_maintain_buffer = true;
}

template<typename T>
void WenetDecoderLayerWeight<T>::loadModel(std::string dir_path, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG("WenetDecoderLayerWeight " + std::string(__func__) + " start");
    FT_CHECK(is_maintain_buffer == true);

    std::vector<std::string> weights_name;
    std::string              name_prefix = "decoder.decoders.";
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".norm1.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".norm1.bias");

    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.linear_q.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.linear_q.bias");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.linear_k.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.linear_k.bias");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.linear_v.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.linear_v.bias");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.linear_out.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.linear_out.bias");

    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".norm2.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".norm2.bias");

    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".src_attn.linear_q.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".src_attn.linear_q.bias");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".src_attn.linear_k.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".src_attn.linear_k.bias");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".src_attn.linear_v.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".src_attn.linear_v.bias");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".src_attn.linear_out.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".src_attn.linear_out.bias");

    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".norm3.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".norm3.bias");

    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".feed_forward.w_1.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".feed_forward.w_1.bias");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".feed_forward.w_2.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".feed_forward.w_2.bias");

    for (size_t i = 0; i < weights_name.size(); ++i) {
        loadWeightFromBin<T>(weights_ptr[i], {weights_size[i]}, dir_path + weights_name[i] + ".bin", model_file_type);
    }

    FT_LOG_DEBUG("WenetDecoderLayerWeight " + std::string(__func__) + " end");
}

template struct WenetDecoderLayerWeight<float>;
template struct WenetDecoderLayerWeight<half>;

}  // namespace fastertransformer
