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

#pragma once

#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
#include "src/fastertransformer/models/BaseWeight.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include <unordered_map>

namespace fastertransformer {

template<typename T>
struct BertLayerWeight {

    BertLayerWeight() = default;
    BertLayerWeight(const size_t hidden_units,
                    const size_t inter_size,
                    const size_t tensor_para_size,
                    const size_t tensor_para_rank):
        hidden_units_(hidden_units),
        inter_size_(inter_size),
        tensor_para_size_(tensor_para_size),
        tensor_para_rank_(tensor_para_rank)
    {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        std::string name;

        name = "attention.self.query.weight." + std::to_string(tensor_para_rank_) + ".bin";
        weights_ptr.insert({name, FtWeight<T>(name, {hidden_units_, hidden_units_ / tensor_para_size_}, nullptr)});
        name = "attention.self.query.bias." + std::to_string(tensor_para_rank_) + ".bin";
        weights_ptr.insert({name, FtWeight<T>(name, {hidden_units_ / tensor_para_size_}, nullptr)});
        name = "attention.self.key.weight." + std::to_string(tensor_para_rank_) + ".bin";
        weights_ptr.insert({name, FtWeight<T>(name, {hidden_units_, hidden_units_ / tensor_para_size_}, nullptr)});
        name = "attention.self.key.bias." + std::to_string(tensor_para_rank_) + ".bin";
        weights_ptr.insert({name, FtWeight<T>(name, {hidden_units_ / tensor_para_size_}, nullptr)});
        name = "attention.self.value.weight." + std::to_string(tensor_para_rank_) + ".bin";
        weights_ptr.insert({name, FtWeight<T>(name, {hidden_units_, hidden_units_ / tensor_para_size_}, nullptr)});
        name = "attention.self.value.bias." + std::to_string(tensor_para_rank_) + ".bin";
        weights_ptr.insert({name, FtWeight<T>(name, {hidden_units_ / tensor_para_size_}, nullptr)});
        name = "attention.output.dense.weight." + std::to_string(tensor_para_rank_) + ".bin";
        weights_ptr.insert({name, FtWeight<T>(name, {hidden_units_, hidden_units_ / tensor_para_size_}, nullptr)});
        name = "attention.output.dense.bias.bin";
        weights_ptr.insert({name, FtWeight<T>(name, {hidden_units_}, nullptr)});
        name = "attention.output.LayerNorm.weight.bin";
        weights_ptr.insert({name, FtWeight<T>(name, {hidden_units_}, nullptr)});
        name = "attention.output.LayerNorm.bias.bin";
        weights_ptr.insert({name, FtWeight<T>(name, {hidden_units_}, nullptr)});
        name = "intermediate.dense.weight." + std::to_string(tensor_para_rank_) + ".bin";
        weights_ptr.insert({name, FtWeight<T>(name, {hidden_units_, inter_size_ / tensor_para_size_}, nullptr)});
        name = "intermediate.dense.bias." + std::to_string(tensor_para_rank_) + ".bin";
        weights_ptr.insert({name, FtWeight<T>(name, {inter_size_ / tensor_para_size_}, nullptr)});
        name = "output.dense.weight." + std::to_string(tensor_para_rank_) + ".bin";
        weights_ptr.insert({name, FtWeight<T>(name, {inter_size_ / tensor_para_size_, hidden_units_}, nullptr)});
        name = "output.dense.bias.bin";
        weights_ptr.insert({name, FtWeight<T>(name, {hidden_units_}, nullptr)});
        name = "output.LayerNorm.weight.bin";
        weights_ptr.insert({name, FtWeight<T>(name, {hidden_units_}, nullptr)});
        name = "output.LayerNorm.bias.bin";
        weights_ptr.insert({name, FtWeight<T>(name, {hidden_units_}, nullptr)});

        for (auto it = weights_ptr.begin(); it != weights_ptr.end(); ++it) {
            deviceMalloc(&it->second.ptr_, it->second.size_);
        }
        setWeightPtr();
        FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    }

    BertLayerWeight(const int hidden_units, const int inter_size): BertLayerWeight(hidden_units, inter_size, 1, 0) {}

    ~BertLayerWeight()
    {
        if (is_maintain_buffer == true) {
            for (auto it = weights_ptr.begin(); it != weights_ptr.end(); ++it) {
                deviceFree(it->second.ptr_);
            }
            weights_ptr.clear();

            attention_weights.query_weight.kernel            = nullptr;
            attention_weights.query_weight.bias              = nullptr;
            attention_weights.key_weight.kernel              = nullptr;
            attention_weights.key_weight.bias                = nullptr;
            attention_weights.value_weight.kernel            = nullptr;
            attention_weights.value_weight.bias              = nullptr;
            attention_weights.attention_output_weight.kernel = nullptr;
            attention_weights.attention_output_weight.bias   = nullptr;
            attn_layernorm_weights.gamma                     = nullptr;
            attn_layernorm_weights.beta                      = nullptr;
            ffn_weights.intermediate_weight.kernel           = nullptr;
            ffn_weights.intermediate_weight.bias             = nullptr;
            ffn_weights.output_weight.kernel                 = nullptr;
            ffn_weights.output_weight.bias                   = nullptr;
            ffn_layernorm_weights.gamma                      = nullptr;
            ffn_layernorm_weights.beta                       = nullptr;
            is_maintain_buffer                               = false;
        }
        if (is_maintain_sp_buffer == true) {
            for (int i = 0; i < 6; i++) {
                deviceFree(sp_weights_ptr[i]);
            }
            attention_weights.query_weight.sp_kernel            = nullptr;
            attention_weights.key_weight.sp_kernel              = nullptr;
            attention_weights.value_weight.sp_kernel            = nullptr;
            attention_weights.attention_output_weight.sp_kernel = nullptr;
            ffn_weights.intermediate_weight.sp_kernel           = nullptr;
            ffn_weights.output_weight.sp_kernel                 = nullptr;
            is_maintain_sp_buffer                               = false;
        }
    }

    BertLayerWeight(const BertLayerWeight& other):
        BertLayerWeight(other.hidden_units_, other.inter_size_, other.tensor_para_size_, other.tensor_para_rank_)
    {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        for (auto it = other.weights_ptr.begin(); it != other.weights_ptr.end(); ++it) {
            cudaD2Dcpy(weights_ptr.at(it->first).ptr_, it->second.ptr_, it->second.size_);
        }
        FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    }

    BertLayerWeight& operator=(const BertLayerWeight& other)
    {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        hidden_units_     = other.hidden_units_;
        inter_size_       = other.inter_size_;
        tensor_para_size_ = other.tensor_para_size_;
        tensor_para_rank_ = other.tensor_para_rank_;

        for (auto it = other.weights_ptr.begin(); it != other.weights_ptr.end(); ++it) {
            weights_ptr.insert({it->first, it->second});
            weights_ptr.at(it->first).ptr_ = nullptr;
            deviceMalloc(weights_ptr.at(it->first).ptr_, it->second.size_);
            cudaD2Dcpy(weights_ptr.at(it->first).ptr_, it->second.ptr_, it->second.size_);
        }
        setWeightPtr();
        FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);

        return *this;
    }

#ifdef SPARSITY_ENABLED
    void compress_weights(cublasMMWrapper& cublas_wrapper, int hidden_dim)
    {
        int inter_size = hidden_dim * 4;
        deviceMalloc(&sp_weights_ptr[0], hidden_dim * hidden_dim);
        deviceMalloc(&sp_weights_ptr[1], hidden_dim * hidden_dim);
        deviceMalloc(&sp_weights_ptr[2], hidden_dim * hidden_dim);
        deviceMalloc(&sp_weights_ptr[3], hidden_dim * hidden_dim);
        deviceMalloc(&sp_weights_ptr[4], hidden_dim * inter_size);
        deviceMalloc(&sp_weights_ptr[5], inter_size * hidden_dim);
        cublas_wrapper.compressMatrix(attention_weights.query_weight.kernel, sp_weights_ptr[0], hidden_dim, hidden_dim);
        cublas_wrapper.compressMatrix(attention_weights.key_weight.kernel, sp_weights_ptr[1], hidden_dim, hidden_dim);
        cublas_wrapper.compressMatrix(attention_weights.value_weight.kernel, sp_weights_ptr[2], hidden_dim, hidden_dim);
        cublas_wrapper.compressMatrix(
            attention_weights.attention_output_weight.kernel, sp_weights_ptr[3], hidden_dim, hidden_dim);
        cublas_wrapper.compressMatrix(
            ffn_weights.intermediate_weight.kernel, sp_weights_ptr[4], inter_size, hidden_dim);
        cublas_wrapper.compressMatrix(ffn_weights.output_weight.kernel, sp_weights_ptr[5], hidden_dim, inter_size);
        attention_weights.query_weight.sp_kernel            = sp_weights_ptr[0];
        attention_weights.key_weight.sp_kernel              = sp_weights_ptr[1];
        attention_weights.value_weight.sp_kernel            = sp_weights_ptr[2];
        attention_weights.attention_output_weight.sp_kernel = sp_weights_ptr[3];
        ffn_weights.intermediate_weight.sp_kernel           = sp_weights_ptr[4];
        ffn_weights.output_weight.sp_kernel                 = sp_weights_ptr[5];
        is_maintain_sp_buffer                               = true;
    }
#endif

    AttentionWeight<T> attention_weights;
    LayerNormWeight<T> attn_layernorm_weights;
    FfnWeight<T>       ffn_weights;
    LayerNormWeight<T> ffn_layernorm_weights;

    void loadModel(std::string dir_path, FtCudaDataType model_file_type)
    {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        for (auto it = weights_ptr.begin(); it != weights_ptr.end(); ++it) {
            loadWeightFromBin<T>(it->second.ptr_, it->second.shape_, dir_path + it->first, model_file_type);
        }
        FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    }

private:
    void setWeightPtr()
    {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        attention_weights.query_weight.kernel =
            weights_ptr.at("attention.self.query.weight." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
        attention_weights.query_weight.bias =
            weights_ptr.at("attention.self.query.bias." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
        attention_weights.key_weight.kernel =
            weights_ptr.at("attention.self.key.weight." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
        attention_weights.key_weight.bias =
            weights_ptr.at("attention.self.key.bias." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
        attention_weights.value_weight.kernel =
            weights_ptr.at("attention.self.value.weight." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
        attention_weights.value_weight.bias =
            weights_ptr.at("attention.self.value.bias." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
        attention_weights.attention_output_weight.kernel =
            weights_ptr.at("attention.output.dense.weight." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
        attention_weights.attention_output_weight.bias = weights_ptr.at("attention.output.dense.bias.bin").ptr_;
        attn_layernorm_weights.gamma                   = weights_ptr.at("attention.output.LayerNorm.weight.bin").ptr_;
        attn_layernorm_weights.beta                    = weights_ptr.at("attention.output.LayerNorm.bias.bin").ptr_;
        ffn_weights.intermediate_weight.kernel =
            weights_ptr.at("intermediate.dense.weight." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
        ffn_weights.intermediate_weight.bias =
            weights_ptr.at("intermediate.dense.bias." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
        ffn_weights.output_weight.kernel =
            weights_ptr.at("output.dense.weight." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
        ffn_weights.output_weight.bias = weights_ptr.at("output.dense.bias.bin").ptr_;
        ffn_layernorm_weights.gamma    = weights_ptr.at("output.LayerNorm.weight.bin").ptr_;
        ffn_layernorm_weights.beta     = weights_ptr.at("output.LayerNorm.bias.bin").ptr_;

        is_maintain_buffer = true;
        FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    }
    size_t                                       hidden_units_;
    size_t                                       inter_size_;
    size_t                                       tensor_para_size_;
    size_t                                       tensor_para_rank_;
    bool                                         is_maintain_buffer = false;
    std::unordered_map<std::string, FtWeight<T>> weights_ptr;
    T*                                           sp_weights_ptr[6];
    bool                                         is_maintain_sp_buffer = false;
};

}  // namespace fastertransformer
