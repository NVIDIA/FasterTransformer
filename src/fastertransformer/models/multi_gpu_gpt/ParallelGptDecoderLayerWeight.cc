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

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoderLayerWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
ParallelGptDecoderLayerWeight<T>::ParallelGptDecoderLayerWeight(const int hidden_units,
                                                                const int inter_size,
                                                                const int tensor_para_size,
                                                                const int tensor_para_rank):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank)
{
    mallocWeights();
    setWeightPtr();
}

template<typename T>
ParallelGptDecoderLayerWeight<T>::~ParallelGptDecoderLayerWeight()
{
    if (is_maintain_buffer == true) {
        for (int i = 0; i < 12; i++) {
            deviceFree(weights_ptr[i]);
        }

        pre_layernorm_weights.beta = nullptr;
        pre_layernorm_weights.gamma = nullptr;
        self_attention_weights.query_weight.kernel = nullptr;
        self_attention_weights.query_weight.bias = nullptr;
        self_attention_weights.attention_output_weight.kernel = nullptr;
        self_attention_weights.attention_output_weight.bias = nullptr;
        self_attn_layernorm_weights.beta = nullptr;
        self_attn_layernorm_weights.gamma = nullptr;

        ffn_weights.intermediate_weight.kernel = nullptr;
        ffn_weights.intermediate_weight.bias = nullptr;
        ffn_weights.output_weight.kernel = nullptr;
        ffn_weights.output_weight.bias = nullptr;
        is_maintain_buffer = false;
    }
}

template<typename T>
ParallelGptDecoderLayerWeight<T>::ParallelGptDecoderLayerWeight(const ParallelGptDecoderLayerWeight& other):
    hidden_units_(other.hidden_units_),
    inter_size_(other.inter_size_),
    tensor_para_size_(other.tensor_para_size_),
    tensor_para_rank_(other.tensor_para_rank_)
{
    mallocWeights();
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
    cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_);
    cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], hidden_units_);

    cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], hidden_units_);
    setWeightPtr();
}

template<typename T>
ParallelGptDecoderLayerWeight<T>&
ParallelGptDecoderLayerWeight<T>::operator=(const ParallelGptDecoderLayerWeight& other)
{
    hidden_units_ = other.hidden_units_;
    inter_size_ = other.inter_size_;
    tensor_para_size_ = other.tensor_para_size_;
    tensor_para_rank_ = other.tensor_para_rank_;

    mallocWeights();
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
    cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_);
    cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], hidden_units_);

    cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], hidden_units_);
    setWeightPtr();
    return *this;
}

template<typename T>
void ParallelGptDecoderLayerWeight<T>::loadModel(std::string dir_path)
{
    FT_CHECK(is_maintain_buffer == true);

    loadWeightFromBin<T>(weights_ptr[0], {(int)hidden_units_}, dir_path + ".input_layernorm.bias.bin");
    loadWeightFromBin<T>(weights_ptr[1], {(int)hidden_units_}, dir_path + ".input_layernorm.weight.bin");
    loadWeightFromBin<T>(weights_ptr[2],
                         {(int)hidden_units_, (int)(3 * hidden_units_ / tensor_para_size_)},
                         dir_path + ".attention.query_key_value.weight." + std::to_string(tensor_para_rank_) + ".bin");
    loadWeightFromBin<T>(weights_ptr[3],
                         {3, (int)(hidden_units_ / tensor_para_size_)},
                         dir_path + ".attention.query_key_value.bias." + std::to_string(tensor_para_rank_) + ".bin");
    loadWeightFromBin<T>(weights_ptr[4],
                         {(int)(hidden_units_ / tensor_para_size_), (int)hidden_units_},
                         dir_path + ".attention.dense.weight." + std::to_string(tensor_para_rank_) + ".bin");
    loadWeightFromBin<T>(weights_ptr[5], {(int)hidden_units_}, dir_path + ".attention.dense.bias.bin");
    loadWeightFromBin<T>(weights_ptr[6], {(int)hidden_units_}, dir_path + ".post_attention_layernorm.bias.bin");
    loadWeightFromBin<T>(weights_ptr[7], {(int)hidden_units_}, dir_path + ".post_attention_layernorm.weight.bin");

    loadWeightFromBin<T>(weights_ptr[8],
                         {(int)hidden_units_, (int)(inter_size_ / tensor_para_size_)},
                         dir_path + ".mlp.dense_h_to_4h.weight." + std::to_string(tensor_para_rank_) + ".bin");
    loadWeightFromBin<T>(weights_ptr[9],
                         {(int)(inter_size_ / tensor_para_size_)},
                         dir_path + ".mlp.dense_h_to_4h.bias." + std::to_string(tensor_para_rank_) + ".bin");
    loadWeightFromBin<T>(weights_ptr[10],
                         {(int)(inter_size_ / tensor_para_size_), (int)hidden_units_},
                         dir_path + ".mlp.dense_4h_to_h.weight." + std::to_string(tensor_para_rank_) + ".bin");
    loadWeightFromBin<T>(weights_ptr[11], {(int)hidden_units_}, dir_path + ".mlp.dense_4h_to_h.bias.bin");
}

template<typename T>
void ParallelGptDecoderLayerWeight<T>::setWeightPtr()
{
    pre_layernorm_weights.beta = weights_ptr[0];
    pre_layernorm_weights.gamma = weights_ptr[1];
    self_attention_weights.query_weight.kernel = weights_ptr[2];
    self_attention_weights.query_weight.bias = weights_ptr[3];
    self_attention_weights.attention_output_weight.kernel = weights_ptr[4];
    self_attention_weights.attention_output_weight.bias = weights_ptr[5];
    self_attn_layernorm_weights.beta = weights_ptr[6];
    self_attn_layernorm_weights.gamma = weights_ptr[7];

    ffn_weights.intermediate_weight.kernel = weights_ptr[8];
    ffn_weights.intermediate_weight.bias = weights_ptr[9];
    ffn_weights.output_weight.kernel = weights_ptr[10];
    ffn_weights.output_weight.bias = weights_ptr[11];

    is_maintain_buffer = true;
}

template<typename T>
void ParallelGptDecoderLayerWeight<T>::mallocWeights()
{
    deviceMalloc(&weights_ptr[0], hidden_units_);
    deviceMalloc(&weights_ptr[1], hidden_units_);
    deviceMalloc(&weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
    deviceMalloc(&weights_ptr[3], 3 * hidden_units_ / tensor_para_size_);
    deviceMalloc(&weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_);
    deviceMalloc(&weights_ptr[5], hidden_units_);
    deviceMalloc(&weights_ptr[6], hidden_units_);
    deviceMalloc(&weights_ptr[7], hidden_units_);

    deviceMalloc(&weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);
    deviceMalloc(&weights_ptr[9], inter_size_ / tensor_para_size_);
    deviceMalloc(&weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_);
    deviceMalloc(&weights_ptr[11], hidden_units_);
}

template struct ParallelGptDecoderLayerWeight<float>;
template struct ParallelGptDecoderLayerWeight<half>;

}  // namespace fastertransformer
