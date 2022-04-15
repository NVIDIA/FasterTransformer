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

#include "src/fastertransformer/models/t5/T5EncoderLayerWeight.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
T5EncoderLayerWeight<T>::T5EncoderLayerWeight(const size_t head_num,
                                              const size_t size_per_head,
                                              const size_t d_model,
                                              const size_t inter_size,
                                              const size_t tensor_para_size,
                                              const size_t tensor_para_rank,
                                              const bool t5_with_bias):
    head_num_(head_num),
    size_per_head_(size_per_head),
    d_model_(d_model),
    inter_size_(inter_size),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    t5_with_bias_(t5_with_bias),
    real_weights_num_(t5_with_bias ? 16 : 8)
{
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " start");
    initialize();
    mallocWeights();
    setWeightPtr();
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
void T5EncoderLayerWeight<T>::initialize()
{
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " start");
    weights_size[0] = d_model_ * (head_num_ / tensor_para_size_) * size_per_head_;
    weights_size[1] = d_model_ * (head_num_ / tensor_para_size_) * size_per_head_;
    weights_size[2] = d_model_ * (head_num_ / tensor_para_size_) * size_per_head_;
    weights_size[3] = (head_num_ / tensor_para_size_) * size_per_head_ * d_model_;
    weights_size[4] = d_model_;
    weights_size[5] = d_model_ * (inter_size_ / tensor_para_size_);
    weights_size[6] = (inter_size_ / tensor_para_size_) * d_model_;
    weights_size[7] = d_model_;
    if (t5_with_bias_) {
        weights_size[8] = (head_num_ / tensor_para_size_) * size_per_head_;
        weights_size[9] = (head_num_ / tensor_para_size_) * size_per_head_;
        weights_size[10] = (head_num_ / tensor_para_size_) * size_per_head_;
        weights_size[11] = d_model_;
        weights_size[12] = d_model_;
        weights_size[13] = (inter_size_ / tensor_para_size_);
        weights_size[14] = d_model_;
        weights_size[15] = d_model_;
    }

    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
T5EncoderLayerWeight<T>::~T5EncoderLayerWeight()
{
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " start");
    if (is_maintain_buffer == true) {
        for (int i = 0; i < real_weights_num_; i++) {
            deviceFree(weights_ptr[i]);
        }

        attention_weights.query_weight.kernel = nullptr;
        attention_weights.key_weight.kernel = nullptr;
        attention_weights.value_weight.kernel = nullptr;
        attention_weights.attention_output_weight.kernel = nullptr;
        attn_layernorm_weights.gamma = nullptr;
        ffn_weights.intermediate_weight.kernel = nullptr;
        ffn_weights.output_weight.kernel = nullptr;
        ffn_layernorm_weights.gamma = nullptr;
        if (t5_with_bias_) {
            attention_weights.query_weight.bias = nullptr;
            attention_weights.key_weight.bias = nullptr;
            attention_weights.value_weight.bias = nullptr;
            attention_weights.attention_output_weight.bias = nullptr;
            attn_layernorm_weights.beta = nullptr;
            ffn_weights.intermediate_weight.bias = nullptr;
            ffn_weights.output_weight.bias = nullptr;
            ffn_layernorm_weights.beta = nullptr;
            is_maintain_buffer = false;
        }
    }

    if (is_maintain_sp_buffer == true) {
        for (int i = 0; i < 6; i++) {
            deviceFree(sp_weights_ptr[i]);
        }
        attention_weights.query_weight.sp_kernel = nullptr;
        attention_weights.key_weight.sp_kernel = nullptr;
        attention_weights.value_weight.sp_kernel = nullptr;
        attention_weights.attention_output_weight.sp_kernel = nullptr;
        ffn_weights.intermediate_weight.sp_kernel = nullptr;
        ffn_weights.output_weight.sp_kernel = nullptr;
        is_maintain_sp_buffer = false;
    }
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
T5EncoderLayerWeight<T>::T5EncoderLayerWeight(const T5EncoderLayerWeight& other):
    head_num_(other.head_num_),
    size_per_head_(other.size_per_head_),
    d_model_(other.d_model_),
    inter_size_(other.inter_size_),
    tensor_para_size_(other.tensor_para_size_),
    tensor_para_rank_(other.tensor_para_rank_),
    t5_with_bias_(other.t5_with_bias_),
    real_weights_num_(other.real_weights_num_)
{
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " start");
    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    setWeightPtr();
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
T5EncoderLayerWeight<T>& T5EncoderLayerWeight<T>::operator=(const T5EncoderLayerWeight<T>& other)
{
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " start");

    head_num_ = other.head_num_;
    size_per_head_ = other.size_per_head_;
    d_model_ = other.d_model_;
    inter_size_ = other.inter_size_;
    tensor_para_size_ = other.tensor_para_size_;
    tensor_para_rank_ = other.tensor_para_rank_;
    t5_with_bias_ = other.t5_with_bias_;
    real_weights_num_ = other.real_weights_num_;
    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    setWeightPtr();
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " end");

    return *this;
}

#ifdef SPARSITY_ENABLED
template<typename T>
void T5EncoderLayerWeight<T>::compress_weights(cublasMMWrapper& cublas_wrapper, int hidden_dim)
{
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " start");
    int inter_size = hidden_dim * 4;
    deviceMalloc(&sp_weights_ptr[0], weights_size[0]);
    deviceMalloc(&sp_weights_ptr[1], weights_size[1]);
    deviceMalloc(&sp_weights_ptr[2], weights_size[2]);
    deviceMalloc(&sp_weights_ptr[3], weights_size[3]);
    deviceMalloc(&sp_weights_ptr[4], weights_size[5]);
    deviceMalloc(&sp_weights_ptr[5], weights_size[6]);

    cublas_wrapper.compressMatrix(attention_weights.query_weight.kernel,
                                  sp_weights_ptr[0],
                                  d_model_,
                                  (head_num_ / tensor_para_size_) * size_per_head_);
    cublas_wrapper.compressMatrix(attention_weights.key_weight.kernel,
                                  sp_weights_ptr[1],
                                  d_model_,
                                  (head_num_ / tensor_para_size_) * size_per_head_);
    cublas_wrapper.compressMatrix(attention_weights.value_weight.kernel,
                                  sp_weights_ptr[2],
                                  d_model_,
                                  (head_num_ / tensor_para_size_) * size_per_head_);
    cublas_wrapper.compressMatrix(attention_weights.attention_output_weight.kernel,
                                  sp_weights_ptr[3],
                                  (head_num_ / tensor_para_size_) * size_per_head_,
                                  d_model_);
    cublas_wrapper.compressMatrix(
        ffn_weights.intermediate_weight.kernel, sp_weights_ptr[4], inter_size / tensor_para_size_, d_model_);
    cublas_wrapper.compressMatrix(
        ffn_weights.output_weight.kernel, sp_weights_ptr[5], d_model_, inter_size / tensor_para_size_);
    attention_weights.query_weight.sp_kernel = sp_weights_ptr[0];
    attention_weights.key_weight.sp_kernel = sp_weights_ptr[1];
    attention_weights.value_weight.sp_kernel = sp_weights_ptr[2];
    attention_weights.attention_output_weight.sp_kernel = sp_weights_ptr[3];
    ffn_weights.intermediate_weight.sp_kernel = sp_weights_ptr[4];
    ffn_weights.output_weight.sp_kernel = sp_weights_ptr[5];
    is_maintain_sp_buffer = true;
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " end");
}
#endif

template<typename T>
void T5EncoderLayerWeight<T>::setWeightPtr()
{
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " start");
    attention_weights.query_weight.kernel = weights_ptr[0];
    attention_weights.key_weight.kernel = weights_ptr[1];
    attention_weights.value_weight.kernel = weights_ptr[2];
    attention_weights.attention_output_weight.kernel = weights_ptr[3];
    attn_layernorm_weights.gamma = weights_ptr[4];
    ffn_weights.intermediate_weight.kernel = weights_ptr[5];
    ffn_weights.output_weight.kernel = weights_ptr[6];
    ffn_layernorm_weights.gamma = weights_ptr[7];

    if (t5_with_bias_) {
        attention_weights.query_weight.bias = weights_ptr[8];
        attention_weights.key_weight.bias = weights_ptr[9];
        attention_weights.value_weight.bias = weights_ptr[10];
        attention_weights.attention_output_weight.bias = weights_ptr[11];
        attn_layernorm_weights.beta = weights_ptr[12];
        ffn_weights.intermediate_weight.bias = weights_ptr[13];
        ffn_weights.output_weight.bias = weights_ptr[14];
        ffn_layernorm_weights.beta = weights_ptr[15];
        is_maintain_buffer = false;
    }

    is_maintain_buffer = true;
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
void T5EncoderLayerWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " start");
    for (int i = 0; i < real_weights_num_; i++) {
        deviceMalloc(&weights_ptr[i], weights_size[i]);
    }
    is_maintain_buffer = true;
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
void T5EncoderLayerWeight<T>::loadModel(std::string dir_path, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " start");

    FT_CHECK(is_maintain_buffer == true);

    loadWeightFromBin<T>(weights_ptr[0],
                         {(int)weights_size[0]},
                         dir_path + "layer.0.SelfAttention.q.weight." + std::to_string(tensor_para_rank_) + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[1],
                         {(int)weights_size[1]},
                         dir_path + "layer.0.SelfAttention.k.weight." + std::to_string(tensor_para_rank_) + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[2],
                         {(int)weights_size[2]},
                         dir_path + "layer.0.SelfAttention.v.weight." + std::to_string(tensor_para_rank_) + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[3],
                         {(int)weights_size[3]},
                         dir_path + "layer.0.SelfAttention.o.weight." + std::to_string(tensor_para_rank_) + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[4], {(int)weights_size[4]}, dir_path + "layer.0.layer_norm.weight.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[5],
                         {(int)weights_size[5]},
                         dir_path + "layer.1.DenseReluDense.wi.weight." + std::to_string(tensor_para_rank_) + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[6],
                         {(int)weights_size[6]},
                         dir_path + "layer.1.DenseReluDense.wo.weight." + std::to_string(tensor_para_rank_) + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[7], {(int)weights_size[7]}, dir_path + "layer.1.layer_norm.weight.bin", model_file_type);

    if (t5_with_bias_) {
        loadWeightFromBin<T>(weights_ptr[8],
                             {(int)weights_size[8]},
                             dir_path + "layer.0.SelfAttention.q.bias." + std::to_string(tensor_para_rank_) + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(weights_ptr[9],
                             {(int)weights_size[9]},
                             dir_path + "layer.0.SelfAttention.k.bias." + std::to_string(tensor_para_rank_) + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(weights_ptr[10],
                             {(int)weights_size[10]},
                             dir_path + "layer.0.SelfAttention.v.bias." + std::to_string(tensor_para_rank_) + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(
            weights_ptr[11], {(int)weights_size[11]}, dir_path + "layer.0.SelfAttention.o.bias.bin", model_file_type);
        loadWeightFromBin<T>(
            weights_ptr[12], {(int)weights_size[12]}, dir_path + "layer.0.layer_norm.bias.bin", model_file_type);
        loadWeightFromBin<T>(weights_ptr[13],
                             {(int)weights_size[13]},
                             dir_path + "layer.1.DenseReluDense.wi.bias." + std::to_string(tensor_para_rank_) + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(
            weights_ptr[14], {(int)weights_size[14]}, dir_path + "layer.1.DenseReluDense.wo.bias.bin", model_file_type);
        loadWeightFromBin<T>(
            weights_ptr[15], {(int)weights_size[15]}, dir_path + "layer.1.layer_norm.bias.bin", model_file_type);
    }
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
void T5EncoderLayerWeight<T>::setT5WithBias(bool t5_with_bias_para)
{
    t5_with_bias_ = t5_with_bias_para;
}

template struct T5EncoderLayerWeight<float>;
template struct T5EncoderLayerWeight<half>;

}  // namespace fastertransformer
