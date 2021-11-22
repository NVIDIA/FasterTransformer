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

#include "src/fastertransformer/models/gpt/GptDecoderLayerWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
GptDecoderLayerWeight<T>::GptDecoderLayerWeight(const int hidden_units, const int inter_size):
    hidden_units_(hidden_units), inter_size_(inter_size)
{
    mallocWeights();
    setWeightPtr();
}

template<typename T>
GptDecoderLayerWeight<T>::~GptDecoderLayerWeight()
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
GptDecoderLayerWeight<T>::GptDecoderLayerWeight(const GptDecoderLayerWeight& other):
    hidden_units_(other.hidden_units_), inter_size_(other.inter_size_)
{
    mallocWeights();
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ * 3 * hidden_units_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], 3 * hidden_units_);
    cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
    cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_);
    cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], hidden_units_);

    cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], hidden_units_ * inter_size_);
    cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], inter_size_);
    cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], inter_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], hidden_units_);

    setWeightPtr();
}

template<typename T>
GptDecoderLayerWeight<T>& GptDecoderLayerWeight<T>::operator=(const GptDecoderLayerWeight& other)
{
    hidden_units_ = other.hidden_units_;
    inter_size_ = other.inter_size_;

    mallocWeights();
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ * 3 * hidden_units_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], 3 * hidden_units_);
    cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
    cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_);
    cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], hidden_units_);

    cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], hidden_units_ * inter_size_);
    cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], inter_size_);
    cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], inter_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], hidden_units_);

    setWeightPtr();
    return *this;
}

template<typename T>
void GptDecoderLayerWeight<T>::loadModel(std::string dir_path)
{
    FT_CHECK(is_maintain_buffer == true);

    loadWeightFromBin<T>(weights_ptr[0], {hidden_units_}, dir_path + ".input_layernorm.bias.bin");
    loadWeightFromBin<T>(weights_ptr[1], {hidden_units_}, dir_path + ".input_layernorm.weight.bin");
    loadWeightFromBin<T>(
        weights_ptr[2], {hidden_units_, 3 * hidden_units_}, dir_path + ".attention.query_key_value.weight.0.bin");
    loadWeightFromBin<T>(weights_ptr[3], {3, hidden_units_}, dir_path + ".attention.query_key_value.bias.0.bin");
    loadWeightFromBin<T>(weights_ptr[4], {hidden_units_, hidden_units_}, dir_path + ".attention.dense.weight.0.bin");
    loadWeightFromBin<T>(weights_ptr[5], {hidden_units_}, dir_path + ".attention.dense.bias.bin");
    loadWeightFromBin<T>(weights_ptr[6], {hidden_units_}, dir_path + ".post_attention_layernorm.bias.bin");
    loadWeightFromBin<T>(weights_ptr[7], {hidden_units_}, dir_path + ".post_attention_layernorm.weight.bin");

    loadWeightFromBin<T>(
        weights_ptr[8], {hidden_units_, 4 * hidden_units_}, dir_path + ".mlp.dense_h_to_4h.weight.0.bin");
    loadWeightFromBin<T>(weights_ptr[9], {4 * hidden_units_}, dir_path + ".mlp.dense_h_to_4h.bias.0.bin");
    loadWeightFromBin<T>(
        weights_ptr[10], {4 * hidden_units_, hidden_units_}, dir_path + ".mlp.dense_4h_to_h.weight.0.bin");
    loadWeightFromBin<T>(weights_ptr[11], {hidden_units_}, dir_path + ".mlp.dense_4h_to_h.bias.bin");
}

template<typename T>
void GptDecoderLayerWeight<T>::setWeightPtr()
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
void GptDecoderLayerWeight<T>::mallocWeights()
{
    deviceMalloc(&weights_ptr[0], hidden_units_);
    deviceMalloc(&weights_ptr[1], hidden_units_);
    deviceMalloc(&weights_ptr[2], hidden_units_ * 3 * hidden_units_);
    deviceMalloc(&weights_ptr[3], 3 * hidden_units_);
    deviceMalloc(&weights_ptr[4], hidden_units_ * hidden_units_);
    deviceMalloc(&weights_ptr[5], hidden_units_);
    deviceMalloc(&weights_ptr[6], hidden_units_);
    deviceMalloc(&weights_ptr[7], hidden_units_);

    deviceMalloc(&weights_ptr[8], hidden_units_ * inter_size_);
    deviceMalloc(&weights_ptr[9], inter_size_);
    deviceMalloc(&weights_ptr[10], inter_size_ * hidden_units_);
    deviceMalloc(&weights_ptr[11], hidden_units_);
}

#ifdef SPARSITY_ENABLED
template<typename T>
void GptDecoderLayerWeight<T>::compress_weights(cublasMMWrapper& cublas_wrapper, int hidden_dim)
{
    hidden_units_ = hidden_dim;
    inter_size_ = 4 * hidden_units_;

    const size_t num_sparse_weights = 4;
    int shapes[num_sparse_weights][2] = {
        {hidden_units_, 3 * hidden_units_},
        {hidden_units_, hidden_units_},
        {hidden_units_, inter_size_},
        {inter_size_, hidden_units_}
    };
    const T* dense_weights[num_sparse_weights] = {
        self_attention_weights.query_weight.kernel,
        self_attention_weights.attention_output_weight.kernel,
        ffn_weights.intermediate_weight.kernel,
        ffn_weights.output_weight.kernel
    };

    for (size_t i = 0; i < num_sparse_weights; ++i) {
        int m = shapes[i][1];
        int k = shapes[i][0];
        size_t compressed_size = cublas_wrapper.getSparseMatrixSize(m, k);
        deviceMalloc(&sp_weights_ptr[i], static_cast<int>(compressed_size), false);
        cublas_wrapper.compressMatrix(dense_weights[i], sp_weights_ptr[i], m, k);
    }

    self_attention_weights.query_weight.sp_kernel = sp_weights_ptr[0];
    self_attention_weights.attention_output_weight.sp_kernel = sp_weights_ptr[1];
    ffn_weights.intermediate_weight.sp_kernel = sp_weights_ptr[2];
    ffn_weights.output_weight.sp_kernel = sp_weights_ptr[3];
    is_maintain_sp_buffer = true;
}
#endif


template struct GptDecoderLayerWeight<float>;
template struct GptDecoderLayerWeight<half>;

}  // namespace fastertransformer
