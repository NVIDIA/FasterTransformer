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

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoderLayerWeight.h"
#include "src/fastertransformer/kernels/transpose_int8_kernels.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
ParallelGptDecoderLayerWeight<T>::ParallelGptDecoderLayerWeight(const int        hidden_units,
                                                                const int        inter_size,
                                                                const int        tensor_para_size,
                                                                const int        tensor_para_rank,
                                                                const int        int8_mode,
                                                                gptVariantParams gpt_variant_params):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    int8_mode_(int8_mode),
    gpt_variant_params_(gpt_variant_params)
{
    mallocWeights();
    setWeightPtr();

    FT_CHECK_WITH_INFO(!(std::is_same<T, float>::value && int8_mode_ == 1),
                       "Weight only quant does not work with FP32 compute.");
}

template<typename T>
ParallelGptDecoderLayerWeight<T>::ParallelGptDecoderLayerWeight(const int int8_mode): int8_mode_(int8_mode)
{
}

template<typename T>
ParallelGptDecoderLayerWeight<T>::~ParallelGptDecoderLayerWeight()
{
    if (is_maintain_buffer == true) {
        for (int i = 0; i < weights_ptr.size(); i++) {
            if (weights_ptr[i] != nullptr) {
                deviceFree(weights_ptr[i]);
            }
        }

        pre_layernorm_weights.beta                            = nullptr;
        pre_layernorm_weights.gamma                           = nullptr;
        self_attention_weights.query_weight.kernel            = nullptr;
        self_attention_weights.query_weight.bias              = nullptr;
        self_attention_weights.attention_output_weight.kernel = nullptr;
        self_attention_weights.attention_output_weight.bias   = nullptr;
        self_attn_layernorm_weights.beta                      = nullptr;
        self_attn_layernorm_weights.gamma                     = nullptr;

        ffn_weights.intermediate_weight.kernel = nullptr;
        ffn_weights.intermediate_weight.bias   = nullptr;
        ffn_weights.output_weight.kernel       = nullptr;
        ffn_weights.output_weight.bias         = nullptr;

        after_attention_adapter_weights.intermediate_weight.kernel = nullptr;
        after_attention_adapter_weights.intermediate_weight.bias   = nullptr;
        after_attention_adapter_weights.output_weight.kernel       = nullptr;
        after_attention_adapter_weights.output_weight.bias         = nullptr;

        after_ffn_adapter_weights.intermediate_weight.kernel = nullptr;
        after_ffn_adapter_weights.intermediate_weight.bias   = nullptr;
        after_ffn_adapter_weights.output_weight.kernel       = nullptr;
        after_ffn_adapter_weights.output_weight.bias         = nullptr;

        if (int8_mode_ != 0) {
            for (int i = 0; i < int8_weights_ptr.size(); i++) {
                if (int8_weights_ptr[i] != nullptr) {
                    deviceFree(int8_weights_ptr[i]);
                }
            }

            if (int8_mode_ == 1) {
                for (int i = 0; i < scale_ptr.size(); i++) {
                    if (weight_only_scale_ptr[i] != nullptr) {
                        deviceFree(weight_only_scale_ptr[i]);
                    }
                }
            }
            else if (int8_mode_ == 2) {
                for (int i = 0; i < scale_ptr.size(); i++) {
                    if (scale_ptr[i] != nullptr) {
                        deviceFree(scale_ptr[i]);
                    }
                }
                for (int i = 0; i < scale_inter_ptr.size(); i++) {
                    if (scale_inter_ptr[i] != nullptr) {
                        deviceFree(scale_inter_ptr[i]);
                    }
                }
                for (int i = 0; i < scale_out_ptr.size(); i++) {
                    if (scale_out_ptr[i] != nullptr) {
                        deviceFree(scale_out_ptr[i]);
                    }
                }
            }
            self_attention_weights.query_weight.int8_kernel                             = nullptr;
            self_attention_weights.query_weight.weight_only_quant_scale                 = nullptr;
            self_attention_weights.query_weight.scale                                   = nullptr;
            self_attention_weights.query_weight.scale_inter                             = nullptr;
            self_attention_weights.query_weight.scale_out                               = nullptr;
            self_attention_weights.attention_output_weight.int8_kernel                  = nullptr;
            self_attention_weights.attention_output_weight.weight_only_quant_scale      = nullptr;
            self_attention_weights.attention_output_weight.scale                        = nullptr;
            self_attention_weights.attention_output_weight.scale_inter                  = nullptr;
            self_attention_weights.attention_output_weight.scale_out                    = nullptr;
            ffn_weights.intermediate_weight.int8_kernel                                 = nullptr;
            ffn_weights.intermediate_weight.weight_only_quant_scale                     = nullptr;
            ffn_weights.intermediate_weight.scale                                       = nullptr;
            ffn_weights.intermediate_weight.scale_inter                                 = nullptr;
            ffn_weights.intermediate_weight.scale_out                                   = nullptr;
            ffn_weights.output_weight.int8_kernel                                       = nullptr;
            ffn_weights.output_weight.weight_only_quant_scale                           = nullptr;
            ffn_weights.output_weight.scale                                             = nullptr;
            ffn_weights.output_weight.scale_inter                                       = nullptr;
            ffn_weights.output_weight.scale_out                                         = nullptr;
            after_attention_adapter_weights.intermediate_weight.int8_kernel             = nullptr;
            after_attention_adapter_weights.intermediate_weight.weight_only_quant_scale = nullptr;
            after_attention_adapter_weights.output_weight.int8_kernel                   = nullptr;
            after_attention_adapter_weights.output_weight.weight_only_quant_scale       = nullptr;
            after_ffn_adapter_weights.intermediate_weight.int8_kernel                   = nullptr;
            after_ffn_adapter_weights.intermediate_weight.weight_only_quant_scale       = nullptr;
            after_ffn_adapter_weights.output_weight.int8_kernel                         = nullptr;
            after_ffn_adapter_weights.output_weight.weight_only_quant_scale             = nullptr;
        }

        is_maintain_buffer = false;
    }
}

template<typename T>
void ParallelGptDecoderLayerWeight<T>::copyFrom(const ParallelGptDecoderLayerWeight& other)
{
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
    cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_);
    cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], hidden_units_);
    cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], hidden_units_);

    if (gpt_variant_params_.has_adapters) {
        // Copy adapter biases regardless of int8 mode
        cudaD2Dcpy(weights_ptr[13], other.weights_ptr[13], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[15], other.weights_ptr[15], hidden_units_);
        cudaD2Dcpy(weights_ptr[17], other.weights_ptr[17], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[19], other.weights_ptr[19], hidden_units_);
    }

    if (int8_mode_ == 0) {
        cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_);

        if (gpt_variant_params_.has_adapters) {
            cudaD2Dcpy(weights_ptr[12],
                       other.weights_ptr[12],
                       hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(weights_ptr[14],
                       other.weights_ptr[14],
                       gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
            cudaD2Dcpy(weights_ptr[16],
                       other.weights_ptr[16],
                       hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(weights_ptr[18],
                       other.weights_ptr[18],
                       gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
        }
    }
    else {
        cudaD2Dcpy(
            int8_weights_ptr[0], other.int8_weights_ptr[0], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[1], other.int8_weights_ptr[1], hidden_units_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(int8_weights_ptr[2], other.int8_weights_ptr[2], hidden_units_ * inter_size_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[3], other.int8_weights_ptr[3], inter_size_ / tensor_para_size_ * hidden_units_);

        if (gpt_variant_params_.has_adapters) {
            // Copy weights for FFN adapters after attn and regular FFN
            cudaD2Dcpy(int8_weights_ptr[4],
                       other.int8_weights_ptr[4],
                       hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(int8_weights_ptr[5],
                       other.int8_weights_ptr[5],
                       gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
            cudaD2Dcpy(int8_weights_ptr[6],
                       other.int8_weights_ptr[6],
                       hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(int8_weights_ptr[7],
                       other.int8_weights_ptr[7],
                       gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
        }

        if (int8_mode_ == 1) {
            cudaD2Dcpy(weight_only_scale_ptr[0], other.weight_only_scale_ptr[0], 3 * hidden_units_ / tensor_para_size_);
            cudaD2Dcpy(weight_only_scale_ptr[1], other.weight_only_scale_ptr[1], hidden_units_);
            cudaD2Dcpy(weight_only_scale_ptr[2], other.weight_only_scale_ptr[2], inter_size_ / tensor_para_size_);
            cudaD2Dcpy(weight_only_scale_ptr[3], other.weight_only_scale_ptr[3], hidden_units_);

            if (gpt_variant_params_.has_adapters) {
                cudaD2Dcpy(weight_only_scale_ptr[4],
                           other.weight_only_scale_ptr[4],
                           gpt_variant_params_.adapter_inter_size / tensor_para_size_);
                cudaD2Dcpy(weight_only_scale_ptr[5], other.weight_only_scale_ptr[5], hidden_units_);
                cudaD2Dcpy(weight_only_scale_ptr[6],
                           other.weight_only_scale_ptr[6],
                           gpt_variant_params_.adapter_inter_size / tensor_para_size_);
                cudaD2Dcpy(weight_only_scale_ptr[7], other.weight_only_scale_ptr[7], hidden_units_);
            }
        }
        else if (int8_mode_ == 2) {
            cudaD2Dcpy(scale_ptr[0], other.scale_out_ptr[0], 1);
            cudaD2Dcpy(scale_inter_ptr[0], other.scale_inter_ptr[0], 3 * hidden_units_ / tensor_para_size_);
            cudaD2Dcpy(scale_out_ptr[0], other.scale_out_ptr[0], 3);

            for (int i = 1; i < 4; i++) {
                cudaD2Dcpy(scale_ptr[i], other.scale_ptr[i], 1);
                cudaD2Dcpy(scale_inter_ptr[i], other.scale_inter_ptr[i], 1);
                cudaD2Dcpy(scale_out_ptr[i], other.scale_out_ptr[i], 1);
            }
        }
    }
}

template<typename T>
ParallelGptDecoderLayerWeight<T>::ParallelGptDecoderLayerWeight(const ParallelGptDecoderLayerWeight& other):
    hidden_units_(other.hidden_units_),
    inter_size_(other.inter_size_),
    tensor_para_size_(other.tensor_para_size_),
    tensor_para_rank_(other.tensor_para_rank_),
    int8_mode_(other.int8_mode_),
    gpt_variant_params_(other.gpt_variant_params_)
{
    mallocWeights();
    copyFrom(other);
    setWeightPtr();
}

template<typename T>
ParallelGptDecoderLayerWeight<T>&
ParallelGptDecoderLayerWeight<T>::operator=(const ParallelGptDecoderLayerWeight& other)
{
    hidden_units_       = other.hidden_units_;
    inter_size_         = other.inter_size_;
    tensor_para_size_   = other.tensor_para_size_;
    tensor_para_rank_   = other.tensor_para_rank_;
    int8_mode_          = other.int8_mode_;
    gpt_variant_params_ = other.gpt_variant_params_;

    mallocWeights();
    copyFrom(other);
    setWeightPtr();
    return *this;
}

template<typename T>
void ParallelGptDecoderLayerWeight<T>::loadModel(std::string dir_path, FtCudaDataType model_file_type)
{
    FT_CHECK(is_maintain_buffer == true);

    loadWeightFromBin<T>(weights_ptr[0], {hidden_units_}, dir_path + ".input_layernorm.bias.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[1], {hidden_units_}, dir_path + ".input_layernorm.weight.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[3],
                         {3, hidden_units_ / tensor_para_size_},
                         dir_path + ".attention.query_key_value.bias." + std::to_string(tensor_para_rank_) + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[5], {hidden_units_}, dir_path + ".attention.dense.bias.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[6], {hidden_units_}, dir_path + ".post_attention_layernorm.bias.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[7], {hidden_units_}, dir_path + ".post_attention_layernorm.weight.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[9],
                         {inter_size_ / tensor_para_size_},
                         dir_path + ".mlp.dense_h_to_4h.bias." + std::to_string(tensor_para_rank_) + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[11], {hidden_units_}, dir_path + ".mlp.dense_4h_to_h.bias.bin", model_file_type);

    if (gpt_variant_params_.has_adapters) {
        loadWeightFromBin<T>(weights_ptr[13],
                             {gpt_variant_params_.adapter_inter_size / tensor_para_size_},
                             dir_path + ".after_attention_adapter.dense_h_to_4h.bias."
                                 + std::to_string(tensor_para_rank_) + ".bin",
                             model_file_type);

        loadWeightFromBin<T>(weights_ptr[15],
                             {hidden_units_},
                             dir_path + ".after_attention_adapter.dense_4h_to_h.bias.bin",
                             model_file_type);

        loadWeightFromBin<T>(weights_ptr[17],
                             {gpt_variant_params_.adapter_inter_size / tensor_para_size_},
                             dir_path + ".after_ffn_adapter.dense_h_to_4h.bias." + std::to_string(tensor_para_rank_)
                                 + ".bin",
                             model_file_type);

        loadWeightFromBin<T>(
            weights_ptr[19], {hidden_units_}, dir_path + ".after_ffn_adapter.dense_4h_to_h.bias.bin", model_file_type);
    }

    // Load weights for GPT
    if (int8_mode_ == 0) {
        loadWeightFromBin<T>(weights_ptr[2],
                             {hidden_units_, 3 * hidden_units_ / tensor_para_size_},
                             dir_path + ".attention.query_key_value.weight." + std::to_string(tensor_para_rank_)
                                 + ".bin",
                             model_file_type);

        loadWeightFromBin<T>(weights_ptr[4],
                             {hidden_units_ / tensor_para_size_, hidden_units_},
                             dir_path + ".attention.dense.weight." + std::to_string(tensor_para_rank_) + ".bin",
                             model_file_type);

        loadWeightFromBin<T>(weights_ptr[8],
                             {hidden_units_, inter_size_ / tensor_para_size_},
                             dir_path + ".mlp.dense_h_to_4h.weight." + std::to_string(tensor_para_rank_) + ".bin",
                             model_file_type);

        loadWeightFromBin<T>(weights_ptr[10],
                             {inter_size_ / tensor_para_size_, hidden_units_},
                             dir_path + ".mlp.dense_4h_to_h.weight." + std::to_string(tensor_para_rank_) + ".bin",
                             model_file_type);

        // Load adapter weights if required.
        if (gpt_variant_params_.has_adapters) {
            loadWeightFromBin<T>(weights_ptr[12],
                                 {hidden_units_, gpt_variant_params_.adapter_inter_size / tensor_para_size_},
                                 dir_path + ".after_attention_adapter.dense_h_to_4h.weight."
                                     + std::to_string(tensor_para_rank_) + ".bin",
                                 model_file_type);

            loadWeightFromBin<T>(weights_ptr[14],
                                 {gpt_variant_params_.adapter_inter_size / tensor_para_size_, hidden_units_},
                                 dir_path + ".after_attention_adapter.dense_4h_to_h.weight."
                                     + std::to_string(tensor_para_rank_) + ".bin",
                                 model_file_type);

            loadWeightFromBin<T>(weights_ptr[16],
                                 {hidden_units_, gpt_variant_params_.adapter_inter_size / tensor_para_size_},
                                 dir_path + ".after_ffn_adapter.dense_h_to_4h.weight."
                                     + std::to_string(tensor_para_rank_) + ".bin",
                                 model_file_type);

            loadWeightFromBin<T>(weights_ptr[18],
                                 {gpt_variant_params_.adapter_inter_size / tensor_para_size_, hidden_units_},
                                 dir_path + ".after_ffn_adapter.dense_4h_to_h.weight."
                                     + std::to_string(tensor_para_rank_) + ".bin",
                                 model_file_type);
        }
    }
    else if (int8_mode_ == 1) {
        loadWeightFromBinAndQuantizeForWeightOnly<T>(int8_weights_ptr[0],
                                                     weight_only_scale_ptr[0],
                                                     {hidden_units_, 3 * hidden_units_ / tensor_para_size_},
                                                     dir_path + ".attention.query_key_value.weight."
                                                         + std::to_string(tensor_para_rank_) + ".bin",
                                                     model_file_type);

        loadWeightFromBinAndQuantizeForWeightOnly<T>(int8_weights_ptr[1],
                                                     weight_only_scale_ptr[1],
                                                     {hidden_units_ / tensor_para_size_, hidden_units_},
                                                     dir_path + ".attention.dense.weight."
                                                         + std::to_string(tensor_para_rank_) + ".bin",
                                                     model_file_type);

        loadWeightFromBinAndQuantizeForWeightOnly<T>(int8_weights_ptr[2],
                                                     weight_only_scale_ptr[2],
                                                     {hidden_units_, inter_size_ / tensor_para_size_},
                                                     dir_path + ".mlp.dense_h_to_4h.weight."
                                                         + std::to_string(tensor_para_rank_) + ".bin",
                                                     model_file_type);

        loadWeightFromBinAndQuantizeForWeightOnly<T>(int8_weights_ptr[3],
                                                     weight_only_scale_ptr[3],
                                                     {inter_size_ / tensor_para_size_, hidden_units_},
                                                     dir_path + ".mlp.dense_4h_to_h.weight."
                                                         + std::to_string(tensor_para_rank_) + ".bin",
                                                     model_file_type);

        // Load adapter weights if required.
        if (gpt_variant_params_.has_adapters) {
            loadWeightFromBinAndQuantizeForWeightOnly<T>(
                int8_weights_ptr[4],
                weight_only_scale_ptr[4],
                {hidden_units_, gpt_variant_params_.adapter_inter_size / tensor_para_size_},
                dir_path + ".after_attention_adapter.dense_h_to_4h.weight." + std::to_string(tensor_para_rank_)
                    + ".bin",
                model_file_type);

            loadWeightFromBinAndQuantizeForWeightOnly<T>(
                int8_weights_ptr[5],
                weight_only_scale_ptr[5],
                {gpt_variant_params_.adapter_inter_size / tensor_para_size_, hidden_units_},
                dir_path + ".after_attention_adapter.dense_4h_to_h.weight." + std::to_string(tensor_para_rank_)
                    + ".bin",
                model_file_type);

            loadWeightFromBinAndQuantizeForWeightOnly<T>(
                int8_weights_ptr[6],
                weight_only_scale_ptr[6],
                {hidden_units_, gpt_variant_params_.adapter_inter_size / tensor_para_size_},
                dir_path + ".after_ffn_adapter.dense_h_to_4h.weight." + std::to_string(tensor_para_rank_) + ".bin",
                model_file_type);

            loadWeightFromBinAndQuantizeForWeightOnly<T>(
                int8_weights_ptr[7],
                weight_only_scale_ptr[7],
                {gpt_variant_params_.adapter_inter_size / tensor_para_size_, hidden_units_},
                dir_path + ".after_ffn_adapter.dense_4h_to_h.weight." + std::to_string(tensor_para_rank_) + ".bin",
                model_file_type);
        }
    }
    else if (int8_mode_ == 2) {
        const auto                     tp_rank = std::to_string(tensor_para_rank_);
        const std::vector<std::string> weight_list{
            "attention.query_key_value", "attention.dense", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"};
        const std::vector<std::vector<size_t>> shape_list{{hidden_units_, 3 * hidden_units_ / tensor_para_size_},
                                                          {hidden_units_ / tensor_para_size_, hidden_units_},
                                                          {hidden_units_, inter_size_ / tensor_para_size_},
                                                          {inter_size_ / tensor_para_size_, hidden_units_}};
        for (int i = 0; i < weight_list.size(); i++) {
            loadWeightFromBin<int8_t>(int8_weights_ptr[i],
                                      shape_list[i],
                                      dir_path + "." + weight_list[i] + ".weight.int8." + tp_rank + ".bin",
                                      FtCudaDataType::INT8);

            const std::pair<std::vector<std::vector<float*>*>, std::vector<std::string>> arg_pair{
                {&scale_ptr, &scale_inter_ptr, &scale_out_ptr}, {"scale", "scale_inter", "scale_out"}};
            for (int j = 0; j < arg_pair.first.size(); j++) {
                size_t num_elems = 1;
                // attention.qkv scale_inter has 3 weights for Q, K and V
                // attention.qkv scale_out has 3 weights for Q, K and V, duplicated along hidden_units dim
                if (i == 0 && j == 1) {
                    num_elems = 3 * hidden_units_ / tensor_para_size_;
                }
                else if (i == 0 && j == 2) {
                    num_elems = 3;
                }

                loadWeightFromBin<float>((*arg_pair.first[j])[i],
                                         {num_elems},
                                         dir_path + "." + weight_list[i] + "." + arg_pair.second[j] + ".bin",
                                         FtCudaDataType::FP32);
            }
        }
        transposeWeight();
    }
}

template<typename T>
void ParallelGptDecoderLayerWeight<T>::setWeightPtr()
{
    pre_layernorm_weights.beta                            = weights_ptr[0];
    pre_layernorm_weights.gamma                           = weights_ptr[1];
    self_attention_weights.query_weight.kernel            = weights_ptr[2];
    self_attention_weights.query_weight.bias              = weights_ptr[3];
    self_attention_weights.attention_output_weight.kernel = weights_ptr[4];
    self_attention_weights.attention_output_weight.bias   = weights_ptr[5];
    self_attn_layernorm_weights.beta                      = weights_ptr[6];
    self_attn_layernorm_weights.gamma                     = weights_ptr[7];

    ffn_weights.intermediate_weight.kernel = weights_ptr[8];
    ffn_weights.intermediate_weight.bias   = weights_ptr[9];
    ffn_weights.output_weight.kernel       = weights_ptr[10];
    ffn_weights.output_weight.bias         = weights_ptr[11];

    after_attention_adapter_weights.intermediate_weight.kernel = weights_ptr[12];
    after_attention_adapter_weights.intermediate_weight.bias   = weights_ptr[13];
    after_attention_adapter_weights.output_weight.kernel       = weights_ptr[14];
    after_attention_adapter_weights.output_weight.bias         = weights_ptr[15];

    after_ffn_adapter_weights.intermediate_weight.kernel = weights_ptr[16];
    after_ffn_adapter_weights.intermediate_weight.bias   = weights_ptr[17];
    after_ffn_adapter_weights.output_weight.kernel       = weights_ptr[18];
    after_ffn_adapter_weights.output_weight.bias         = weights_ptr[19];

    if (int8_mode_ != 0) {
        self_attention_weights.query_weight.int8_kernel                 = int8_weights_ptr[0];
        self_attention_weights.attention_output_weight.int8_kernel      = int8_weights_ptr[1];
        ffn_weights.intermediate_weight.int8_kernel                     = int8_weights_ptr[2];
        ffn_weights.output_weight.int8_kernel                           = int8_weights_ptr[3];
        after_attention_adapter_weights.intermediate_weight.int8_kernel = int8_weights_ptr[4];
        after_attention_adapter_weights.output_weight.int8_kernel       = int8_weights_ptr[5];
        after_ffn_adapter_weights.intermediate_weight.int8_kernel       = int8_weights_ptr[6];
        after_ffn_adapter_weights.output_weight.int8_kernel             = int8_weights_ptr[7];

        if (int8_mode_ == 1) {
            self_attention_weights.query_weight.weight_only_quant_scale                 = weight_only_scale_ptr[0];
            self_attention_weights.attention_output_weight.weight_only_quant_scale      = weight_only_scale_ptr[1];
            ffn_weights.intermediate_weight.weight_only_quant_scale                     = weight_only_scale_ptr[2];
            ffn_weights.output_weight.weight_only_quant_scale                           = weight_only_scale_ptr[3];
            after_attention_adapter_weights.intermediate_weight.weight_only_quant_scale = weight_only_scale_ptr[4];
            after_attention_adapter_weights.output_weight.weight_only_quant_scale       = weight_only_scale_ptr[5];
            after_ffn_adapter_weights.intermediate_weight.weight_only_quant_scale       = weight_only_scale_ptr[6];
            after_ffn_adapter_weights.output_weight.weight_only_quant_scale             = weight_only_scale_ptr[7];
        }
        else if (int8_mode_ == 2) {
            self_attention_weights.query_weight.scale                  = scale_ptr[0];
            self_attention_weights.query_weight.scale_inter            = scale_inter_ptr[0];
            self_attention_weights.query_weight.scale_out              = scale_out_ptr[0];
            self_attention_weights.attention_output_weight.scale       = scale_ptr[1];
            self_attention_weights.attention_output_weight.scale_inter = scale_inter_ptr[1];
            self_attention_weights.attention_output_weight.scale_out   = scale_out_ptr[1];
            ffn_weights.intermediate_weight.scale                      = scale_ptr[2];
            ffn_weights.intermediate_weight.scale_inter                = scale_inter_ptr[2];
            ffn_weights.intermediate_weight.scale_out                  = scale_out_ptr[2];
            ffn_weights.output_weight.scale                            = scale_ptr[3];
            ffn_weights.output_weight.scale_inter                      = scale_inter_ptr[3];
            ffn_weights.output_weight.scale_out                        = scale_out_ptr[3];
        }
    }

    is_maintain_buffer = true;
}

template<typename T>
void ParallelGptDecoderLayerWeight<T>::mallocWeights()
{
    deviceMalloc(&weights_ptr[0], hidden_units_);                          // pre layer norm beta
    deviceMalloc(&weights_ptr[1], hidden_units_);                          // pre layer norm gamma
    deviceMalloc(&weights_ptr[3], 3 * hidden_units_ / tensor_para_size_);  // qkv biases
    deviceMalloc(&weights_ptr[5], hidden_units_);                          // attention output bias
    deviceMalloc(&weights_ptr[6], hidden_units_);                          // attn layer norm beta
    deviceMalloc(&weights_ptr[7], hidden_units_);                          // attn layer norm gamma
    deviceMalloc(&weights_ptr[9], inter_size_ / tensor_para_size_);        // ffn inter bias
    deviceMalloc(&weights_ptr[11], hidden_units_);                         // ffn output bias

    // Alloc biases adapters. They do not get quantized so are placed here.
    if (gpt_variant_params_.has_adapters) {
        deviceMalloc(&weights_ptr[13], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        deviceMalloc(&weights_ptr[15], hidden_units_);
        deviceMalloc(&weights_ptr[17], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        deviceMalloc(&weights_ptr[19], hidden_units_);
    }

    if (int8_mode_ == 0) {
        deviceMalloc(&weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);  // qkv weight
        deviceMalloc(&weights_ptr[4],
                     hidden_units_ / tensor_para_size_ * hidden_units_);                  // attention output weight
        deviceMalloc(&weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);   // ffn inter weight
        deviceMalloc(&weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_);  // ffn output weight

        // Alloc weights for adapters
        if (gpt_variant_params_.has_adapters) {
            deviceMalloc(&weights_ptr[12], hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            deviceMalloc(&weights_ptr[14], gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
            deviceMalloc(&weights_ptr[16], hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            deviceMalloc(&weights_ptr[18], gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
        }
    }
    else {
        // Alloc FFN and Attention int8 weights
        deviceMalloc(&int8_weights_ptr[0], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
        deviceMalloc(&int8_weights_ptr[1], hidden_units_ / tensor_para_size_ * hidden_units_);
        deviceMalloc(&int8_weights_ptr[2], hidden_units_ * inter_size_ / tensor_para_size_);
        deviceMalloc(&int8_weights_ptr[3], inter_size_ / tensor_para_size_ * hidden_units_);

        if (gpt_variant_params_.has_adapters) {
            // Alloc weights for FFN adapters after attn and regular FFN
            deviceMalloc(&int8_weights_ptr[4],
                         hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            deviceMalloc(&int8_weights_ptr[5],
                         gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
            deviceMalloc(&int8_weights_ptr[6],
                         hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            deviceMalloc(&int8_weights_ptr[7],
                         gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
        }

        if (int8_mode_ == 1) {
            // Alloc scales for weight only quant for attention and FFN weights
            deviceMalloc(&weight_only_scale_ptr[0], 3 * hidden_units_ / tensor_para_size_);
            deviceMalloc(&weight_only_scale_ptr[1], hidden_units_);
            deviceMalloc(&weight_only_scale_ptr[2], inter_size_ / tensor_para_size_);
            deviceMalloc(&weight_only_scale_ptr[3], hidden_units_);

            if (gpt_variant_params_.has_adapters) {
                // Alloc scales for FFN adapters after attn and regular FFN.
                deviceMalloc(&weight_only_scale_ptr[4], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
                deviceMalloc(&weight_only_scale_ptr[5], hidden_units_);
                deviceMalloc(&weight_only_scale_ptr[6], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
                deviceMalloc(&weight_only_scale_ptr[7], hidden_units_);
            }
        }
        else if (int8_mode_ == 2) {
            deviceMalloc(&scale_ptr[0], 1);
            deviceMalloc(&scale_inter_ptr[0], 3 * hidden_units_ / tensor_para_size_);
            deviceMalloc(&scale_out_ptr[0], 3);
            for (int i = 1; i < 4; i++) {
                deviceMalloc(&scale_ptr[i], 1);
                deviceMalloc(&scale_inter_ptr[i], 1);
                deviceMalloc(&scale_out_ptr[i], 1);
            }
        }
    }
}

#ifdef SPARSITY_ENABLED
template<typename T>
void ParallelGptDecoderLayerWeight<T>::compress_weights(cublasMMWrapper& cublas_wrapper, int hidden_dim)
{
    hidden_units_ = hidden_dim;
    inter_size_   = 4 * hidden_units_;

    const size_t num_sparse_weights            = 8;
    size_t       shapes[num_sparse_weights][2] = {
              {hidden_units_, 3 * hidden_units_ / tensor_para_size_},
              {hidden_units_ / tensor_para_size_, hidden_units_},
              {hidden_units_, inter_size_ / tensor_para_size_},
              {inter_size_ / tensor_para_size_, hidden_units_},
              {hidden_units_, gpt_variant_params_.adapter_inter_size / tensor_para_size_},
              {gpt_variant_params_.adapter_inter_size / tensor_para_size_, hidden_units_},
              {hidden_units_, gpt_variant_params_.adapter_inter_size / tensor_para_size_},
              {gpt_variant_params_.adapter_inter_size / tensor_para_size_, hidden_units_}};

    const T* dense_weights[num_sparse_weights] = {self_attention_weights.query_weight.kernel,
                                                  self_attention_weights.attention_output_weight.kernel,
                                                  ffn_weights.intermediate_weight.kernel,
                                                  ffn_weights.output_weight.kernel,
                                                  after_attention_adapter_weights.intermediate_weight.kernel,
                                                  after_attention_adapter_weights.output_weight.kernel,
                                                  after_ffn_adapter_weights.intermediate_weight.kernel,
                                                  after_ffn_adapter_weights.output_weight.kernel};

    size_t real_num_sparse_weights = gpt_variant_params_.has_adapters ? num_sparse_weights : (num_sparse_weights - 4);
    for (size_t i = 0; i < real_num_sparse_weights; ++i) {
        int    m               = shapes[i][1];
        int    k               = shapes[i][0];
        size_t compressed_size = cublas_wrapper.getSparseMatrixSize(m, k);
        deviceMalloc(&sp_weights_ptr[i], static_cast<int>(compressed_size), false);
        cublas_wrapper.compressMatrix(dense_weights[i], sp_weights_ptr[i], m, k);
    }

    self_attention_weights.query_weight.sp_kernel                 = sp_weights_ptr[0];
    self_attention_weights.attention_output_weight.sp_kernel      = sp_weights_ptr[1];
    ffn_weights.intermediate_weight.sp_kernel                     = sp_weights_ptr[2];
    ffn_weights.output_weight.sp_kernel                           = sp_weights_ptr[3];
    after_attention_adapter_weights.intermediate_weight.sp_kernel = sp_weights_ptr[4];
    after_attention_adapter_weights.output_weight.sp_kernel       = sp_weights_ptr[5];
    after_ffn_adapter_weights.intermediate_weight.sp_kernel       = sp_weights_ptr[6];
    after_ffn_adapter_weights.output_weight.sp_kernel             = sp_weights_ptr[7];
    is_maintain_sp_buffer                                         = true;
}
#endif

template<typename T>
void ParallelGptDecoderLayerWeight<T>::transposeWeight()
{
    const auto                             tp = tensor_para_size_;
    const std::vector<std::vector<size_t>> shape_list{{hidden_units_, 3 * hidden_units_ / tp},
                                                      {hidden_units_ / tp, hidden_units_},
                                                      {hidden_units_, inter_size_ / tp},
                                                      {inter_size_ / tp, hidden_units_}};

    const auto max_size =
        sizeof(int8_t) * std::max(3 * hidden_units_ * hidden_units_ / tp, hidden_units_ * inter_size_ / tp);

    int8_t* transpose_temp;
    cudaMalloc(&transpose_temp, max_size);

    for (int i = 0; i < shape_list.size(); i++) {
        invokeTransposeInt8Tensor({MEMORY_GPU, TYPE_INT8, {shape_list[i][1], shape_list[i][0]}, transpose_temp},
                                  {MEMORY_GPU, TYPE_INT8, shape_list[i], int8_weights_ptr[i]},
                                  stream_);
        cudaD2Dcpy(int8_weights_ptr[i], transpose_temp, shape_list[i][0] * shape_list[i][1]);
    }

    cudaFree(transpose_temp);
}

// This function is deprecated.
template<typename T>
void ParallelGptDecoderLayerWeight<T>::transposeCalibrateQuantizeWeight()
{
    invokeLdnCalibrateWeightPerChannel(
        scale_ptr[0], weights_ptr[2], hidden_units_, 3 * hidden_units_ / tensor_para_size_, stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(int8_weights_ptr[0],
                                               scale_ptr[0],
                                               weights_ptr[2],
                                               hidden_units_,
                                               3 * hidden_units_ / tensor_para_size_,
                                               stream_);

    invokeLdnCalibrateWeightPerChannel(
        scale_ptr[1], weights_ptr[4], hidden_units_ / tensor_para_size_, hidden_units_, stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(
        int8_weights_ptr[1], scale_ptr[1], weights_ptr[4], hidden_units_ / tensor_para_size_, hidden_units_, stream_);

    invokeLdnCalibrateWeightPerChannel(
        scale_ptr[2], weights_ptr[8], hidden_units_, inter_size_ / tensor_para_size_, stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(
        int8_weights_ptr[2], scale_ptr[2], weights_ptr[8], hidden_units_, inter_size_ / tensor_para_size_, stream_);

    invokeLdnCalibrateWeightPerChannel(
        scale_ptr[3], weights_ptr[10], inter_size_ / tensor_para_size_, hidden_units_, stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(
        int8_weights_ptr[3], scale_ptr[3], weights_ptr[10], inter_size_ / tensor_para_size_, hidden_units_, stream_);

    invokeLdnCalibrateWeightPerChannel(scale_ptr[4],
                                       weights_ptr[12],
                                       hidden_units_,
                                       gpt_variant_params_.adapter_inter_size / tensor_para_size_,
                                       stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(int8_weights_ptr[4],
                                               scale_ptr[4],
                                               weights_ptr[12],
                                               hidden_units_,
                                               gpt_variant_params_.adapter_inter_size / tensor_para_size_,
                                               stream_);

    invokeLdnCalibrateWeightPerChannel(scale_ptr[5],
                                       weights_ptr[14],
                                       gpt_variant_params_.adapter_inter_size / tensor_para_size_,
                                       hidden_units_,
                                       stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(int8_weights_ptr[5],
                                               scale_ptr[5],
                                               weights_ptr[14],
                                               gpt_variant_params_.adapter_inter_size / tensor_para_size_,
                                               hidden_units_,
                                               stream_);

    invokeLdnCalibrateWeightPerChannel(scale_ptr[6],
                                       weights_ptr[16],
                                       hidden_units_,
                                       gpt_variant_params_.adapter_inter_size / tensor_para_size_,
                                       stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(int8_weights_ptr[6],
                                               scale_ptr[6],
                                               weights_ptr[16],
                                               hidden_units_,
                                               gpt_variant_params_.adapter_inter_size / tensor_para_size_,
                                               stream_);

    invokeLdnCalibrateWeightPerChannel(scale_ptr[7],
                                       weights_ptr[18],
                                       gpt_variant_params_.adapter_inter_size / tensor_para_size_,
                                       hidden_units_,
                                       stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(int8_weights_ptr[7],
                                               scale_ptr[7],
                                               weights_ptr[18],
                                               gpt_variant_params_.adapter_inter_size / tensor_para_size_,
                                               hidden_units_,
                                               stream_);
}

template struct ParallelGptDecoderLayerWeight<float>;
template struct ParallelGptDecoderLayerWeight<half>;
#ifdef ENABLE_BF16
template struct ParallelGptDecoderLayerWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
