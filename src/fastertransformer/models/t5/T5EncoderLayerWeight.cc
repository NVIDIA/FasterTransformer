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
#include "src/fastertransformer/utils/IA3.h"
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
                                              const bool   t5_with_bias,
                                              const bool   use_gated_activation,
                                              const size_t ia3_num_tasks):
    head_num_(head_num),
    size_per_head_(size_per_head),
    d_model_(d_model),
    inter_size_(inter_size),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    t5_with_bias_(t5_with_bias),
    use_gated_activation_(use_gated_activation),
    ia3_num_tasks_(ia3_num_tasks)
{
    real_weights_num_ = (8 + (use_gated_activation_ ? 1 : 0)) * (t5_with_bias_ ? 2 : 1);
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
    weights_size_[0] = d_model_ * (head_num_ / tensor_para_size_) * size_per_head_;
    weights_size_[1] = d_model_ * (head_num_ / tensor_para_size_) * size_per_head_;
    weights_size_[2] = d_model_ * (head_num_ / tensor_para_size_) * size_per_head_;
    weights_size_[3] = (head_num_ / tensor_para_size_) * size_per_head_ * d_model_;
    weights_size_[4] = d_model_;
    if (use_gated_activation_) {
        weights_size_[5] = d_model_ * (inter_size_ / tensor_para_size_);
        weights_size_[6] = d_model_ * (inter_size_ / tensor_para_size_);  // for gated activation
        weights_size_[7] = (inter_size_ / tensor_para_size_) * d_model_;
        weights_size_[8] = d_model_;
    }
    else {
        weights_size_[5] = d_model_ * (inter_size_ / tensor_para_size_);
        weights_size_[6] = (inter_size_ / tensor_para_size_) * d_model_;
        weights_size_[7] = d_model_;
    }
    if (t5_with_bias_) {
        if (use_gated_activation_) {
            weights_size_[9]  = (head_num_ / tensor_para_size_) * size_per_head_;
            weights_size_[10] = (head_num_ / tensor_para_size_) * size_per_head_;
            weights_size_[11] = (head_num_ / tensor_para_size_) * size_per_head_;
            weights_size_[12] = d_model_;
            weights_size_[13] = d_model_;
            weights_size_[14] = (inter_size_ / tensor_para_size_);
            weights_size_[15] = (inter_size_ / tensor_para_size_);  // for gated activation
            weights_size_[16] = d_model_;
            weights_size_[17] = d_model_;
        }
        else {
            weights_size_[8]  = (head_num_ / tensor_para_size_) * size_per_head_;
            weights_size_[9]  = (head_num_ / tensor_para_size_) * size_per_head_;
            weights_size_[10] = (head_num_ / tensor_para_size_) * size_per_head_;
            weights_size_[11] = d_model_;
            weights_size_[12] = d_model_;
            weights_size_[13] = (inter_size_ / tensor_para_size_);
            weights_size_[14] = d_model_;
            weights_size_[15] = d_model_;
        }
    }

    if (ia3_num_tasks_ > 0) {
        const size_t attention_adapter_size = ia3_num_tasks_ * (head_num_ / tensor_para_size_) * size_per_head_;
        const size_t mlp_adapter_size       = ia3_num_tasks_ * (inter_size_ / tensor_para_size_);

        ia3_weights_size_[0] = attention_adapter_size;
        ia3_weights_size_[1] = attention_adapter_size;
        ia3_weights_size_[2] = mlp_adapter_size;
    }

    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
T5EncoderLayerWeight<T>::~T5EncoderLayerWeight()
{
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " start");
    if (is_maintain_buffer_) {
        for (int i = 0; i < real_weights_num_; i++) {
            deviceFree(weights_ptr_[i]);
        }

        attention_weights_.query_weight.kernel            = nullptr;
        attention_weights_.key_weight.kernel              = nullptr;
        attention_weights_.value_weight.kernel            = nullptr;
        attention_weights_.attention_output_weight.kernel = nullptr;
        attn_layernorm_weights_.gamma                     = nullptr;
        ffn_weights_.intermediate_weight.kernel           = nullptr;
        ffn_weights_.intermediate_weight2.kernel          = nullptr;
        ffn_weights_.output_weight.kernel                 = nullptr;
        ffn_layernorm_weights_.gamma                      = nullptr;
        attention_weights_.query_weight.bias              = nullptr;
        attention_weights_.key_weight.bias                = nullptr;
        attention_weights_.value_weight.bias              = nullptr;
        attention_weights_.attention_output_weight.bias   = nullptr;
        attn_layernorm_weights_.beta                      = nullptr;
        ffn_weights_.intermediate_weight.bias             = nullptr;
        ffn_weights_.intermediate_weight2.bias            = nullptr;
        ffn_weights_.output_weight.bias                   = nullptr;
        ffn_layernorm_weights_.beta                       = nullptr;
        is_maintain_buffer_                               = false;
    }

    if (is_maintain_sp_buffer_) {
        for (int i = 0; i < 6; i++) {
            deviceFree(sp_weights_ptr_[i]);
        }
        attention_weights_.query_weight.sp_kernel            = nullptr;
        attention_weights_.key_weight.sp_kernel              = nullptr;
        attention_weights_.value_weight.sp_kernel            = nullptr;
        attention_weights_.attention_output_weight.sp_kernel = nullptr;
        ffn_weights_.intermediate_weight.sp_kernel           = nullptr;
        ffn_weights_.output_weight.sp_kernel                 = nullptr;
        is_maintain_sp_buffer_                               = false;
    }

    if (maintain_ia3_buffer_) {
        for (int i = 0; i < IA3_ADAPTER_MAX_NUM_ENCODER; i++) {
            deviceFree(ia3_weights_ptr_[i]);
        }
        attention_weights_.ia3_key_weight.kernel   = nullptr;
        ffn_weights_.ia3_weight.kernel             = nullptr;
        attention_weights_.ia3_value_weight.kernel = nullptr;
        maintain_ia3_buffer_                       = false;
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
    real_weights_num_(other.real_weights_num_),
    ia3_num_tasks_(other.ia3_num_tasks_)
{
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " start");
    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr_[i], other.weights_ptr_[i], weights_size_[i]);
    }
    if (ia3_num_tasks_ > 0) {
        for (int i = 0; i < IA3_ADAPTER_MAX_NUM_ENCODER; i++) {
            cudaD2Dcpy(ia3_weights_ptr_[i], other.ia3_weights_ptr_[i], ia3_weights_size_[i]);
        }
    }
    setWeightPtr();
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
T5EncoderLayerWeight<T>& T5EncoderLayerWeight<T>::operator=(const T5EncoderLayerWeight<T>& other)
{
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " start");

    head_num_         = other.head_num_;
    size_per_head_    = other.size_per_head_;
    d_model_          = other.d_model_;
    inter_size_       = other.inter_size_;
    tensor_para_size_ = other.tensor_para_size_;
    tensor_para_rank_ = other.tensor_para_rank_;
    t5_with_bias_     = other.t5_with_bias_;
    real_weights_num_ = other.real_weights_num_;
    ia3_num_tasks_    = other.ia3_num_tasks_;
    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr_[i], other.weights_ptr_[i], weights_size_[i]);
    }
    if (ia3_num_tasks_ > 0) {
        for (int i = 0; i < IA3_ADAPTER_MAX_NUM_ENCODER; i++) {
            cudaD2Dcpy(ia3_weights_ptr_[i], other.ia3_weights_ptr_[i], ia3_weights_size_[i]);
        }
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
    deviceMalloc(&sp_weights_ptr_[0], weights_size_[0]);
    deviceMalloc(&sp_weights_ptr_[1], weights_size_[1]);
    deviceMalloc(&sp_weights_ptr_[2], weights_size_[2]);
    deviceMalloc(&sp_weights_ptr_[3], weights_size_[3]);
    deviceMalloc(&sp_weights_ptr_[4], weights_size_[5]);
    deviceMalloc(&sp_weights_ptr_[5], weights_size_[6]);

    cublas_wrapper.compressMatrix(attention_weights_.query_weight.kernel,
                                  sp_weights_ptr_[0],
                                  d_model_,
                                  (head_num_ / tensor_para_size_) * size_per_head_);
    cublas_wrapper.compressMatrix(attention_weights_.key_weight.kernel,
                                  sp_weights_ptr_[1],
                                  d_model_,
                                  (head_num_ / tensor_para_size_) * size_per_head_);
    cublas_wrapper.compressMatrix(attention_weights_.value_weight.kernel,
                                  sp_weights_ptr_[2],
                                  d_model_,
                                  (head_num_ / tensor_para_size_) * size_per_head_);
    cublas_wrapper.compressMatrix(attention_weights_.attention_output_weight.kernel,
                                  sp_weights_ptr_[3],
                                  (head_num_ / tensor_para_size_) * size_per_head_,
                                  d_model_);
    cublas_wrapper.compressMatrix(
        ffn_weights_.intermediate_weight.kernel, sp_weights_ptr_[4], inter_size / tensor_para_size_, d_model_);
    cublas_wrapper.compressMatrix(
        ffn_weights_.output_weight.kernel, sp_weights_ptr_[5], d_model_, inter_size / tensor_para_size_);
    attention_weights_.query_weight.sp_kernel            = sp_weights_ptr_[0];
    attention_weights_.key_weight.sp_kernel              = sp_weights_ptr_[1];
    attention_weights_.value_weight.sp_kernel            = sp_weights_ptr_[2];
    attention_weights_.attention_output_weight.sp_kernel = sp_weights_ptr_[3];
    ffn_weights_.intermediate_weight.sp_kernel           = sp_weights_ptr_[4];
    ffn_weights_.output_weight.sp_kernel                 = sp_weights_ptr_[5];
    is_maintain_sp_buffer_                               = true;
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " end");
}
#endif

template<typename T>
void T5EncoderLayerWeight<T>::setWeightPtr()
{
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " start");
    attention_weights_.query_weight.kernel            = weights_ptr_[0];
    attention_weights_.key_weight.kernel              = weights_ptr_[1];
    attention_weights_.value_weight.kernel            = weights_ptr_[2];
    attention_weights_.attention_output_weight.kernel = weights_ptr_[3];
    attn_layernorm_weights_.gamma                     = weights_ptr_[4];
    if (use_gated_activation_) {
        ffn_weights_.intermediate_weight.kernel  = weights_ptr_[5];
        ffn_weights_.intermediate_weight2.kernel = weights_ptr_[6];
        ffn_weights_.output_weight.kernel        = weights_ptr_[7];
        ffn_layernorm_weights_.gamma             = weights_ptr_[8];
    }
    else {
        ffn_weights_.intermediate_weight.kernel = weights_ptr_[5];
        ffn_weights_.output_weight.kernel       = weights_ptr_[6];
        ffn_layernorm_weights_.gamma            = weights_ptr_[7];
    }

    if (t5_with_bias_) {
        if (use_gated_activation_) {
            attention_weights_.query_weight.bias            = weights_ptr_[9];
            attention_weights_.key_weight.bias              = weights_ptr_[10];
            attention_weights_.value_weight.bias            = weights_ptr_[11];
            attention_weights_.attention_output_weight.bias = weights_ptr_[12];
            attn_layernorm_weights_.beta                    = weights_ptr_[13];
            ffn_weights_.intermediate_weight.bias           = weights_ptr_[14];
            ffn_weights_.intermediate_weight2.bias          = weights_ptr_[15];
            ffn_weights_.output_weight.bias                 = weights_ptr_[16];
            ffn_layernorm_weights_.beta                     = weights_ptr_[17];
        }
        else {
            attention_weights_.query_weight.bias            = weights_ptr_[8];
            attention_weights_.key_weight.bias              = weights_ptr_[9];
            attention_weights_.value_weight.bias            = weights_ptr_[10];
            attention_weights_.attention_output_weight.bias = weights_ptr_[11];
            attn_layernorm_weights_.beta                    = weights_ptr_[12];
            ffn_weights_.intermediate_weight.bias           = weights_ptr_[13];
            ffn_weights_.output_weight.bias                 = weights_ptr_[14];
            ffn_layernorm_weights_.beta                     = weights_ptr_[15];
        }
    }

    if (ia3_num_tasks_ > 0) {
        attention_weights_.ia3_key_weight.kernel   = ia3_weights_ptr_[0];
        attention_weights_.ia3_value_weight.kernel = ia3_weights_ptr_[1];
        ffn_weights_.ia3_weight.kernel             = ia3_weights_ptr_[2];
    }

    is_maintain_buffer_ = true;
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
void T5EncoderLayerWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " start");
    for (int i = 0; i < real_weights_num_; i++) {
        deviceMalloc(&weights_ptr_[i], weights_size_[i]);
    }
    if (ia3_num_tasks_ > 0) {
        for (int i = 0; i < IA3_ADAPTER_MAX_NUM_ENCODER; i++) {
            deviceMalloc(&ia3_weights_ptr_[i], ia3_weights_size_[i]);
        }
        maintain_ia3_buffer_ = true;
    }
    is_maintain_buffer_ = true;
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
void T5EncoderLayerWeight<T>::loadModel(std::string dir_path, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " start");

    FT_CHECK(is_maintain_buffer_);

    const auto tp_rank = std::to_string(tensor_para_rank_);
    loadWeightFromBin<T>(weights_ptr_[0],
                         {weights_size_[0]},
                         dir_path + "layer.0.SelfAttention.q.weight." + tp_rank + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr_[1],
                         {weights_size_[1]},
                         dir_path + "layer.0.SelfAttention.k.weight." + tp_rank + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr_[2],
                         {weights_size_[2]},
                         dir_path + "layer.0.SelfAttention.v.weight." + tp_rank + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr_[3],
                         {weights_size_[3]},
                         dir_path + "layer.0.SelfAttention.o.weight." + tp_rank + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(
        weights_ptr_[4], {weights_size_[4]}, dir_path + "layer.0.layer_norm.weight.bin", model_file_type);

    loadWeightFromBin<T>(weights_ptr_[5],
                         {weights_size_[5]},
                         dir_path + "layer.1.DenseReluDense.wi.weight." + tp_rank + ".bin",
                         model_file_type);
    const int gated_activation_weight_offset = use_gated_activation_ ? 1 : 0;
    if (use_gated_activation_) {
        loadWeightFromBin<T>(weights_ptr_[6],
                             {weights_size_[6]},
                             dir_path + "layer.1.DenseReluDense.wi2.weight." + tp_rank + ".bin",
                             model_file_type);
    }

    loadWeightFromBin<T>(weights_ptr_[6 + gated_activation_weight_offset],
                         {weights_size_[6 + gated_activation_weight_offset]},
                         dir_path + "layer.1.DenseReluDense.wo.weight." + tp_rank + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr_[7 + gated_activation_weight_offset],
                         {weights_size_[7 + gated_activation_weight_offset]},
                         dir_path + "layer.1.layer_norm.weight.bin",
                         model_file_type);

    if (t5_with_bias_) {
        loadWeightFromBin<T>(weights_ptr_[8 + gated_activation_weight_offset],
                             {weights_size_[8 + gated_activation_weight_offset]},
                             dir_path + "layer.0.SelfAttention.q.bias." + tp_rank + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(weights_ptr_[9 + gated_activation_weight_offset],
                             {weights_size_[9 + gated_activation_weight_offset]},
                             dir_path + "layer.0.SelfAttention.k.bias." + tp_rank + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(weights_ptr_[10 + gated_activation_weight_offset],
                             {weights_size_[10 + gated_activation_weight_offset]},
                             dir_path + "layer.0.SelfAttention.v.bias." + tp_rank + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(weights_ptr_[11 + gated_activation_weight_offset],
                             {weights_size_[11 + gated_activation_weight_offset]},
                             dir_path + "layer.0.SelfAttention.o.bias.bin",
                             model_file_type);
        loadWeightFromBin<T>(weights_ptr_[12 + gated_activation_weight_offset],
                             {weights_size_[12 + gated_activation_weight_offset]},
                             dir_path + "layer.0.layer_norm.bias.bin",
                             model_file_type);
        loadWeightFromBin<T>(weights_ptr_[13 + gated_activation_weight_offset],
                             {weights_size_[13 + gated_activation_weight_offset]},
                             dir_path + "layer.1.DenseReluDense.wi.bias." + tp_rank + ".bin",
                             model_file_type);
        if (use_gated_activation_) {
            loadWeightFromBin<T>(weights_ptr_[15],
                                 {weights_size_[15]},
                                 dir_path + "layer.1.DenseReluDense.wi2.bias." + tp_rank + ".bin",
                                 model_file_type);
            loadWeightFromBin<T>(weights_ptr_[16],
                                 {weights_size_[16]},
                                 dir_path + "layer.1.DenseReluDense.wo.bias.bin",
                                 model_file_type);
            loadWeightFromBin<T>(
                weights_ptr_[17], {weights_size_[17]}, dir_path + "layer.1.layer_norm.bias.bin", model_file_type);
        }
        else {
            loadWeightFromBin<T>(weights_ptr_[14],
                                 {weights_size_[14]},
                                 dir_path + "layer.1.DenseReluDense.wo.bias.bin",
                                 model_file_type);
            loadWeightFromBin<T>(
                weights_ptr_[15], {weights_size_[15]}, dir_path + "layer.1.layer_norm.bias.bin", model_file_type);
        }
    }

    if (ia3_num_tasks_ > 0) {
        loadWeightFromBin<T>(ia3_weights_ptr_[0],
                             {ia3_weights_size_[0]},
                             dir_path + "layer.0.SelfAttention.k.ia3.weight." + tp_rank + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(ia3_weights_ptr_[1],
                             {ia3_weights_size_[1]},
                             dir_path + "layer.0.SelfAttention.v.ia3.weight." + tp_rank + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(ia3_weights_ptr_[2],
                             {ia3_weights_size_[2]},
                             dir_path + "layer.1.DenseReluDense.ia3.weight." + tp_rank + ".bin",
                             model_file_type);
    }

    FT_LOG_DEBUG("T5EncoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
void T5EncoderLayerWeight<T>::setT5WithBias(bool t5_with_bias_para, bool use_gated_activation_para)
{
    t5_with_bias_         = t5_with_bias_para;
    use_gated_activation_ = use_gated_activation_para;
}

template struct T5EncoderLayerWeight<float>;
template struct T5EncoderLayerWeight<half>;
#ifdef ENABLE_BF16
template struct T5EncoderLayerWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
