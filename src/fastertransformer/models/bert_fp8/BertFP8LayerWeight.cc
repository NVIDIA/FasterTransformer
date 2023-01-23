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

#include "BertFP8LayerWeight.h"
#include "3rdparty/fp8_qgmma_1x1/fp8_qgmma_1x1_utils.h"
#include "src/fastertransformer/utils/cuda_fp8_utils.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include <algorithm>
#include <functional>
#include <numeric>

namespace fastertransformer {

template<typename T1, typename T2>
BertFP8LayerWeight<T1, T2>::BertFP8LayerWeight(const size_t d_model,
                                               const size_t head_num,
                                               const size_t size_per_head,
                                               const size_t inter_size,
                                               const size_t tensor_para_size,
                                               const size_t pipeline_para_size,
                                               const int    fp8_mode,
                                               bool         is_load_model,
                                               bool         is_fused_qkv_gemm_bias):
    d_model_(d_model),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    inter_size_(inter_size),
    tensor_para_size_(tensor_para_size),
    pipeline_para_size_(pipeline_para_size),
    fp8_mode_(fp8_mode),
    is_fused_qkv_gemm_bias_(is_fused_qkv_gemm_bias),
    weights_ptr(4),
    vec_ptr(8),
    scale_ptr_(26),
    scale_h_ptr_(26)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_load_model) {
        mallocWeights();
        setWeightPtr();
    }
}

template<typename T1, typename T2>
BertFP8LayerWeight<T1, T2>::~BertFP8LayerWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer == true) {
        for (uint i = 0; i < weights_ptr.size(); i++) {
            weights_ptr[i].first = 0;
            deviceFree(weights_ptr[i].second);
        }
        for (uint i = 0; i < vec_ptr.size(); i++) {
            vec_ptr[i].first = 0;
            deviceFree(vec_ptr[i].second);
        }
        for (uint i = 0; i < scale_ptr_.size(); i++) {
            scale_ptr_[i].first = 0;
            deviceFree(scale_ptr_[i].second);
        }
        for (uint i = 0; i < scale_h_ptr_.size(); i++) {
            delete scale_h_ptr_[i];
        }

        attention_weights.query_weight.kernel                      = nullptr;
        attention_weights.query_weight.bias                        = nullptr;
        attention_weights.query_weight.input_scale                 = nullptr;
        attention_weights.query_weight.input_scale_inv             = nullptr;
        attention_weights.query_weight.output_scale                = nullptr;
        attention_weights.query_weight.output_scale_inv            = nullptr;
        attention_weights.query_weight.weight_scale_inv            = nullptr;
        attention_weights.query_weight.weight_scale                = nullptr;
        attention_weights.key_weight.kernel                        = nullptr;
        attention_weights.key_weight.bias                          = nullptr;
        attention_weights.value_weight.kernel                      = nullptr;
        attention_weights.value_weight.bias                        = nullptr;
        attention_weights.attention_output_weight.kernel           = nullptr;
        attention_weights.attention_output_weight.bias             = nullptr;
        attention_weights.attention_output_weight.input_scale      = nullptr;
        attention_weights.attention_output_weight.input_scale_inv  = nullptr;
        attention_weights.attention_output_weight.output_scale     = nullptr;
        attention_weights.attention_output_weight.output_scale_inv = nullptr;
        attention_weights.attention_output_weight.weight_scale_inv = nullptr;
        attention_weights.attention_output_weight.weight_scale     = nullptr;
        attn_layernorm_weights.gamma                               = nullptr;
        attn_layernorm_weights.beta                                = nullptr;
        ffn_weights.intermediate_weight.kernel                     = nullptr;
        ffn_weights.intermediate_weight.bias                       = nullptr;
        ffn_weights.intermediate_weight.input_scale                = nullptr;
        ffn_weights.intermediate_weight.input_scale_inv            = nullptr;
        ffn_weights.intermediate_weight.output_scale               = nullptr;
        ffn_weights.intermediate_weight.output_scale_inv           = nullptr;
        ffn_weights.intermediate_weight.weight_scale_inv           = nullptr;
        ffn_weights.intermediate_weight.weight_scale               = nullptr;
        ffn_weights.output_weight.kernel                           = nullptr;
        ffn_weights.output_weight.bias                             = nullptr;
        ffn_weights.output_weight.input_scale                      = nullptr;
        ffn_weights.output_weight.input_scale_inv                  = nullptr;
        ffn_weights.output_weight.output_scale                     = nullptr;
        ffn_weights.output_weight.output_scale_inv                 = nullptr;
        ffn_weights.output_weight.weight_scale_inv                 = nullptr;
        ffn_weights.output_weight.weight_scale                     = nullptr;
        ffn_layernorm_weights.gamma                                = nullptr;
        ffn_layernorm_weights.beta                                 = nullptr;
        is_maintain_buffer                                         = false;
    }
    if (is_maintain_sp_buffer == true) {
        for (uint i = 0; i < sp_weights_ptr.size(); i++) {
            sp_weights_ptr[i].first = 0;
            deviceFree(sp_weights_ptr[i].second);
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

template<typename T1, typename T2>
BertFP8LayerWeight<T1, T2>::BertFP8LayerWeight(const BertFP8LayerWeight& other):
    BertFP8LayerWeight(other.d_model_,
                       other.head_num_,
                       other.size_per_head_,
                       other.inter_size_,
                       other.tensor_para_size_,
                       other.pipeline_para_size_,
                       other.fp8_mode_,
                       true,
                       other.is_fused_qkv_gemm_bias_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    for (uint i = 0; i < weights_ptr.size(); i++) {
        cudaD2Dcpy(weights_ptr[i].second, other.weights_ptr[i].second, weights_ptr[i].first);
    }
    for (uint i = 0; i < vec_ptr.size(); i++) {
        cudaD2Dcpy(vec_ptr[i].second, other.vec_ptr[i].second, vec_ptr[i].first);
    }
    for (uint i = 0; i < sp_weights_ptr.size(); i++) {
        cudaD2Dcpy(sp_weights_ptr[i].second, other.sp_weights_ptr[i].second, sp_weights_ptr[i].first);
    }
    for (int i = 0; i < scale_ptr_.size(); i++) {
        cudaD2Dcpy(scale_ptr_[i].second, other.scale_ptr_[i].second, scale_ptr_[i].first);
    }
    for (int i = 0; i < scale_h_ptr_.size(); i++) {
        *(scale_h_ptr_[i]) = *(other.scale_h_ptr_[i]);
    }
}

template<typename T1, typename T2>
void BertFP8LayerWeight<T1, T2>::setWeightPtr()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    attention_weights.query_weight.kernel            = weights_ptr[0].second;
    attention_weights.attention_output_weight.kernel = weights_ptr[1].second;
    ffn_weights.intermediate_weight.kernel           = weights_ptr[2].second;
    ffn_weights.output_weight.kernel                 = weights_ptr[3].second;

    attention_weights.query_weight.bias            = vec_ptr[0].second;
    attention_weights.attention_output_weight.bias = vec_ptr[1].second;
    attn_layernorm_weights.gamma                   = vec_ptr[2].second;
    attn_layernorm_weights.beta                    = vec_ptr[3].second;
    ffn_weights.intermediate_weight.bias           = vec_ptr[4].second;
    ffn_weights.output_weight.bias                 = vec_ptr[5].second;
    ffn_layernorm_weights.gamma                    = vec_ptr[6].second;
    ffn_layernorm_weights.beta                     = vec_ptr[7].second;

    attention_weights.query_weight.input_scale                 = scale_ptr_[0].second;
    attention_weights.query_weight.input_scale_inv             = scale_ptr_[1].second;
    attention_weights.query_weight.output_scale                = scale_ptr_[2].second;
    attention_weights.query_weight.output_scale_inv            = scale_ptr_[3].second;
    attention_weights.query_weight.weight_scale                = scale_ptr_[4].second;
    attention_weights.query_weight.weight_scale_inv            = scale_ptr_[5].second;
    attention_weights.attention_output_weight.input_scale      = scale_ptr_[6].second;
    attention_weights.attention_output_weight.input_scale_inv  = scale_ptr_[7].second;
    attention_weights.attention_output_weight.output_scale     = scale_ptr_[8].second;
    attention_weights.attention_output_weight.output_scale_inv = scale_ptr_[9].second;
    attention_weights.attention_output_weight.weight_scale     = scale_ptr_[10].second;
    attention_weights.attention_output_weight.weight_scale_inv = scale_ptr_[11].second;
    ffn_weights.intermediate_weight.input_scale                = scale_ptr_[12].second;
    ffn_weights.intermediate_weight.input_scale_inv            = scale_ptr_[13].second;
    ffn_weights.intermediate_weight.output_scale               = scale_ptr_[14].second;
    ffn_weights.intermediate_weight.output_scale_inv           = scale_ptr_[15].second;
    ffn_weights.intermediate_weight.weight_scale               = scale_ptr_[16].second;
    ffn_weights.intermediate_weight.weight_scale_inv           = scale_ptr_[17].second;
    ffn_weights.output_weight.input_scale                      = scale_ptr_[18].second;
    ffn_weights.output_weight.input_scale_inv                  = scale_ptr_[19].second;
    ffn_weights.output_weight.output_scale                     = scale_ptr_[20].second;
    ffn_weights.output_weight.output_scale_inv                 = scale_ptr_[21].second;
    ffn_weights.output_weight.weight_scale                     = scale_ptr_[22].second;
    ffn_weights.output_weight.weight_scale_inv                 = scale_ptr_[23].second;
    attention_weights.qk_scale                                 = scale_ptr_[24].second;
    attention_weights.qk_scale_inv                             = scale_ptr_[25].second;

    attention_weights.query_weight.input_h_scale                 = scale_h_ptr_[0];
    attention_weights.query_weight.input_h_scale_inv             = scale_h_ptr_[1];
    attention_weights.query_weight.output_h_scale                = scale_h_ptr_[2];
    attention_weights.query_weight.output_h_scale_inv            = scale_h_ptr_[3];
    attention_weights.query_weight.weight_h_scale                = scale_h_ptr_[4];
    attention_weights.query_weight.weight_h_scale_inv            = scale_h_ptr_[5];
    attention_weights.attention_output_weight.input_h_scale      = scale_h_ptr_[6];
    attention_weights.attention_output_weight.input_h_scale_inv  = scale_h_ptr_[7];
    attention_weights.attention_output_weight.output_h_scale     = scale_h_ptr_[8];
    attention_weights.attention_output_weight.output_h_scale_inv = scale_h_ptr_[9];
    attention_weights.attention_output_weight.weight_h_scale     = scale_h_ptr_[10];
    attention_weights.attention_output_weight.weight_h_scale_inv = scale_h_ptr_[11];
    ffn_weights.intermediate_weight.input_h_scale                = scale_h_ptr_[12];
    ffn_weights.intermediate_weight.input_h_scale_inv            = scale_h_ptr_[13];
    ffn_weights.intermediate_weight.output_h_scale               = scale_h_ptr_[14];
    ffn_weights.intermediate_weight.output_h_scale_inv           = scale_h_ptr_[15];
    ffn_weights.intermediate_weight.weight_h_scale               = scale_h_ptr_[16];
    ffn_weights.intermediate_weight.weight_h_scale_inv           = scale_h_ptr_[17];
    ffn_weights.output_weight.input_h_scale                      = scale_h_ptr_[18];
    ffn_weights.output_weight.input_h_scale_inv                  = scale_h_ptr_[19];
    ffn_weights.output_weight.output_h_scale                     = scale_h_ptr_[20];
    ffn_weights.output_weight.output_h_scale_inv                 = scale_h_ptr_[21];
    ffn_weights.output_weight.weight_h_scale                     = scale_h_ptr_[22];
    ffn_weights.output_weight.weight_h_scale_inv                 = scale_h_ptr_[23];
    attention_weights.qk_h_scale                                 = scale_h_ptr_[24];
    attention_weights.qk_h_scale_inv                             = scale_h_ptr_[25];

    attention_weights.query_weight.fuse_gemm_bias = is_fused_qkv_gemm_bias_;
    is_maintain_buffer                            = true;
}

template<typename T1, typename T2>
void BertFP8LayerWeight<T1, T2>::mallocWeights()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(d_model_ != 0 && hidden_units_ != 0 && inter_size_ != 0);
    FT_CHECK(head_num_ % tensor_para_size_ == 0);
    FT_CHECK(inter_size_ % tensor_para_size_ == 0);

    weights_ptr[0].first = d_model_ * 3 * hidden_units_ / tensor_para_size_;  // attention qkv weight
    weights_ptr[1].first = hidden_units_ / tensor_para_size_ * d_model_;      // attention projection weight
    weights_ptr[2].first = d_model_ * inter_size_ / tensor_para_size_;        // ffn gemm 1 weight
    weights_ptr[3].first = inter_size_ / tensor_para_size_ * d_model_;        // ffn gemm 2 weight

    vec_ptr[0].first = 3 * hidden_units_ / tensor_para_size_;  // attention qkv bias
    vec_ptr[1].first = d_model_;                               // attention projection bias
    vec_ptr[2].first = d_model_;                               // attention_layernorm gamma
    vec_ptr[3].first = d_model_;                               // attention_layernorm beta
    vec_ptr[4].first = inter_size_ / tensor_para_size_;        // ffn gemm 1 bias
    vec_ptr[5].first = d_model_;                               // ffn gemm 2 bias
    vec_ptr[6].first = d_model_;                               // ffn_layernorm gamma
    vec_ptr[7].first = d_model_;                               // ffn_layernorm beta

    if (fp8_mode_ == 1) {
        scale_ptr_[0].first  = 1;
        scale_ptr_[1].first  = 1;
        scale_ptr_[2].first  = 1;
        scale_ptr_[3].first  = 1;
        scale_ptr_[4].first  = 3 * hidden_units_ / tensor_para_size_;
        scale_ptr_[5].first  = 3 * hidden_units_ / tensor_para_size_;
        scale_ptr_[6].first  = 1;
        scale_ptr_[7].first  = 1;
        scale_ptr_[8].first  = 1;
        scale_ptr_[9].first  = 1;
        scale_ptr_[10].first = d_model_;
        scale_ptr_[11].first = d_model_;
        scale_ptr_[12].first = 1;
        scale_ptr_[13].first = 1;
        scale_ptr_[14].first = 1;
        scale_ptr_[15].first = 1;
        scale_ptr_[16].first = inter_size_ / tensor_para_size_;
        scale_ptr_[17].first = inter_size_ / tensor_para_size_;
        scale_ptr_[18].first = 1;
        scale_ptr_[19].first = 1;
        scale_ptr_[20].first = 1;
        scale_ptr_[21].first = 1;
        scale_ptr_[22].first = d_model_;
        scale_ptr_[23].first = d_model_;
        scale_ptr_[24].first = 1;
        scale_ptr_[25].first = 1;
    }
    else if (fp8_mode_ == 2) {
        for (int i = 0; i < scale_ptr_.size(); i++) {
            scale_ptr_[i].first = 1;
        }
    }
    else {
        FT_CHECK(false);
    }

    for (uint i = 0; i < weights_ptr.size(); i++) {
        deviceMalloc(&weights_ptr[i].second, weights_ptr[i].first);
    }
    for (uint i = 0; i < vec_ptr.size(); i++) {
        deviceMalloc(&vec_ptr[i].second, vec_ptr[i].first);
    }
    for (uint i = 0; i < scale_ptr_.size(); i++) {
        deviceMalloc(&scale_ptr_[i].second, scale_ptr_[i].first, false);
        deviceFill(scale_ptr_[i].second, scale_ptr_[i].first, 1.0f);
    }
    for (uint i = 0; i < scale_h_ptr_.size(); i++) {
        scale_h_ptr_[i] = new float(1.0f);
    }
}

template<typename T1, typename T2>
void BertFP8LayerWeight<T1, T2>::loadModel(std::string dir_path)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(is_maintain_buffer == true);

    loadWeightFromBin<T2>(
        vec_ptr[0].second, {(int)vec_ptr[0].first}, dir_path + ".attention.self.query_key_value.bias.bin");
    loadWeightFromBin<T2>(vec_ptr[1].second, {(int)vec_ptr[1].first}, dir_path + ".attention.output.dense.bias.bin");
    loadWeightFromBin<T2>(
        vec_ptr[2].second, {(int)vec_ptr[2].first}, dir_path + ".attention.output.LayerNorm.weight.bin");
    loadWeightFromBin<T2>(
        vec_ptr[3].second, {(int)vec_ptr[3].first}, dir_path + ".attention.output.LayerNorm.bias.bin");
    loadWeightFromBin<T2>(vec_ptr[4].second, {(int)vec_ptr[4].first}, dir_path + ".intermediate.dense.bias.bin");
    loadWeightFromBin<T2>(vec_ptr[5].second, {(int)vec_ptr[5].first}, dir_path + ".output.dense.bias.bin");
    loadWeightFromBin<T2>(vec_ptr[6].second, {(int)vec_ptr[6].first}, dir_path + ".output.LayerNorm.weight.bin");
    loadWeightFromBin<T2>(vec_ptr[7].second, {(int)vec_ptr[7].first}, dir_path + ".output.LayerNorm.bias.bin");

    // Load scaling
    {
        std::string quantizer_list[13] = {
            ".attention.self.query_key_value._input_quantizer._amax.bin",
            ".attention.self.matmul_q_input_quantizer._amax.bin",
            ".attention.self.query_key_value._weight_quantizer._amax.bin",
            ".attention.output.dense._input_quantizer._amax.bin",
            ".attention.output.add_local_input_quantizer._amax.bin",
            ".attention.output.dense._weight_quantizer._amax.bin",
            ".intermediate.dense._input_quantizer._amax.bin",
            ".intermediate.intermediate_act_fn_input_quantizer._amax.bin",
            ".intermediate.dense._weight_quantizer._amax.bin",
            ".output.dense._input_quantizer._amax.bin",
            ".output.add_local_input_quantizer._amax.bin",
            ".output.dense._weight_quantizer._amax.bin",
            ".attention.self.matmul_a_input_quantizer._amax.bin",  // special amax for multi-head attention
        };

        float* d_amax = nullptr;
        deviceMalloc(&d_amax, inter_size_, false);
        float* h_amax      = new float[inter_size_];
        float* h_scale     = new float[inter_size_];
        float* h_scale_inv = new float[inter_size_];

        for (int i = 0; i < 13; i++) {
            // prevent random initialize the scales, make them to 1.0f
            if (checkIfFileExist(dir_path + quantizer_list[i])) {
                loadWeightFromBin<float>(d_amax, {(int)scale_ptr_[i * 2].first}, dir_path + quantizer_list[i]);
                cudaD2Hcpy(h_amax, d_amax, scale_ptr_[i * 2].first);
            }
            else {
                for (int j = 0; j < scale_ptr_[i * 2].first; j++) {
                    h_amax[j] = 480.f;
                }
            }
            for (int j = 0; j < scale_ptr_[i * 2].first; j++) {
                h_scale[j]     = h_amax[j] / 480.f;
                h_scale_inv[j] = 480.f / h_amax[j];
            }
            cudaH2Dcpy(scale_ptr_[i * 2 + 0].second, h_scale, scale_ptr_[i * 2 + 0].first);
            cudaH2Dcpy(scale_ptr_[i * 2 + 1].second, h_scale_inv, scale_ptr_[i * 2 + 1].first);

            if (fp8_mode_ == 2) {
                *(scale_h_ptr_[i * 2 + 0]) = h_scale[0];      // only for per tensor
                *(scale_h_ptr_[i * 2 + 1]) = h_scale_inv[0];  // only for per tensor
            }
        }
        deviceFree(d_amax);
        delete[] h_amax;
        delete[] h_scale;
        delete[] h_scale_inv;
    }

    {
        float* d_float_weight = nullptr;
        deviceMalloc(&d_float_weight, inter_size_ * inter_size_);

        if (fp8_mode_ == 1) {
            loadWeightFromBin<float>(
                d_float_weight, {(int)weights_ptr[0].first}, dir_path + ".attention.self.query_key_value.weight.bin");
            invokeQuantizeMatrix<T1, float, QUANTIZE_MODE::PER_CHANNEL>(weights_ptr[0].second,
                                                                        scale_ptr_[5].second,
                                                                        d_float_weight,
                                                                        (int)weights_ptr[0].first,
                                                                        scale_ptr_[4].first,
                                                                        (cudaStream_t)0);
            loadWeightFromBin<float>(
                d_float_weight, {(int)weights_ptr[1].first}, dir_path + ".attention.output.dense.weight.bin");
            invokeQuantizeMatrix<T1, float, QUANTIZE_MODE::PER_CHANNEL>(weights_ptr[1].second,
                                                                        scale_ptr_[11].second,
                                                                        d_float_weight,
                                                                        (int)weights_ptr[1].first,
                                                                        scale_ptr_[10].first,
                                                                        (cudaStream_t)0);
            loadWeightFromBin<float>(
                d_float_weight, {(int)weights_ptr[2].first}, dir_path + ".intermediate.dense.weight.bin");
            invokeQuantizeMatrix<T1, float, QUANTIZE_MODE::PER_CHANNEL>(weights_ptr[2].second,
                                                                        scale_ptr_[17].second,
                                                                        d_float_weight,
                                                                        (int)weights_ptr[2].first,
                                                                        scale_ptr_[16].first,
                                                                        (cudaStream_t)0);
            loadWeightFromBin<float>(
                d_float_weight, {(int)weights_ptr[3].first}, dir_path + ".output.dense.weight.bin");
            invokeQuantizeMatrix<T1, float, QUANTIZE_MODE::PER_CHANNEL>(weights_ptr[3].second,
                                                                        scale_ptr_[23].second,
                                                                        d_float_weight,
                                                                        (int)weights_ptr[3].first,
                                                                        scale_ptr_[22].first,
                                                                        (cudaStream_t)0);
        }
        else if (fp8_mode_ == 2) {
            loadWeightFromBin<float>(
                d_float_weight, {(int)weights_ptr[0].first}, dir_path + ".attention.self.query_key_value.weight.bin");
            invokeQuantizeMatrix<T1, float, QUANTIZE_MODE::PER_TENSOR>(weights_ptr[0].second,
                                                                       scale_ptr_[5].second,
                                                                       d_float_weight,
                                                                       (int)weights_ptr[0].first,
                                                                       1,
                                                                       (cudaStream_t)0);
            loadWeightFromBin<float>(
                d_float_weight, {(int)weights_ptr[1].first}, dir_path + ".attention.output.dense.weight.bin");
            invokeQuantizeMatrix<T1, float, QUANTIZE_MODE::PER_TENSOR>(weights_ptr[1].second,
                                                                       scale_ptr_[11].second,
                                                                       d_float_weight,
                                                                       (int)weights_ptr[1].first,
                                                                       1,
                                                                       (cudaStream_t)0);
            loadWeightFromBin<float>(
                d_float_weight, {(int)weights_ptr[2].first}, dir_path + ".intermediate.dense.weight.bin");
            invokeQuantizeMatrix<T1, float, QUANTIZE_MODE::PER_TENSOR>(weights_ptr[2].second,
                                                                       scale_ptr_[17].second,
                                                                       d_float_weight,
                                                                       (int)weights_ptr[2].first,
                                                                       1,
                                                                       (cudaStream_t)0);
            loadWeightFromBin<float>(
                d_float_weight, {(int)weights_ptr[3].first}, dir_path + ".output.dense.weight.bin");
            invokeQuantizeMatrix<T1, float, QUANTIZE_MODE::PER_TENSOR>(weights_ptr[3].second,
                                                                       scale_ptr_[23].second,
                                                                       d_float_weight,
                                                                       (int)weights_ptr[3].first,
                                                                       1,
                                                                       (cudaStream_t)0);
        }

        deviceFree(d_float_weight);
    }
}

template<typename T1, typename T2>
void BertFP8LayerWeight<T1, T2>::serialize(uint8_t*& buffer) const
{
    serialize_d2h(buffer, weights_ptr);
    serialize_d2h(buffer, vec_ptr);
    serialize_d2h(buffer, sp_weights_ptr);
    serialize_d2h(buffer, scale_ptr_);
    serialize_h2h(buffer, scale_h_ptr_);
}

template<typename T1, typename T2>
void BertFP8LayerWeight<T1, T2>::deserialize(const uint8_t*& buffer)
{
    if (!is_maintain_buffer) {
        return;
    }

    deserialize_h2d(buffer, weights_ptr);
    deserialize_h2d(buffer, vec_ptr);
    deserialize_h2d(buffer, sp_weights_ptr);
    deserialize_h2d(buffer, scale_ptr_);
    deserialize_h2h(buffer, scale_h_ptr_);
}

template<typename T1, typename T2>
int32_t BertFP8LayerWeight<T1, T2>::getSerializationSize() const
{
    int32_t    size       = 0;
    int32_t    dtype_size = sizeof(T1);
    const auto pair_accum = [&dtype_size](const auto acc, const auto& x) { return acc + x.first * dtype_size; };
    size                  = std::accumulate(std::begin(weights_ptr), std::end(weights_ptr), size, pair_accum);
    dtype_size            = sizeof(T2);
    size                  = std::accumulate(std::begin(vec_ptr), std::end(vec_ptr), size, pair_accum);
    dtype_size            = sizeof(T1);
    size                  = std::accumulate(std::begin(sp_weights_ptr), std::end(sp_weights_ptr), size, pair_accum);
    dtype_size            = sizeof(float);
    size                  = std::accumulate(std::begin(scale_ptr_), std::end(scale_ptr_), size, pair_accum);
    size += scale_h_ptr_.size() * sizeof(float);
    return size;
}

template<typename T1, typename T2>
void BertFP8LayerWeight<T1, T2>::transposeWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    T1* workspace;
    deviceMalloc(&workspace, hidden_units_ / tensor_para_size_ * std::max(3 * hidden_units_, inter_size_));

    if (is_fused_qkv_gemm_bias_) {
        // transpose the qkv gemm from [hidden, 3, head_num / tp, size_per_head] to [hidden, head_num / tp, 3,
        // size_per_head]
        invokeInPlaceTranspose0213<T1>(
            weights_ptr[0].second, workspace, d_model_, 3, head_num_ / tensor_para_size_, size_per_head_);
        // transpose the qkv bias from [3, head_num / tp, size_per_head] to [head_num / tp, 3, size_per_head]
        invokeInPlaceTranspose102<T2>(
            vec_ptr[0].second, (T2*)workspace, 3, head_num_ / tensor_para_size_, size_per_head_);
    }
    invokeInPlaceTranspose(weights_ptr[0].second, workspace, hidden_units_, 3 * hidden_units_ / tensor_para_size_);
    invokeInPlaceTranspose(weights_ptr[1].second, workspace, hidden_units_ / tensor_para_size_, hidden_units_);
    invokeInPlaceTranspose(weights_ptr[2].second, workspace, hidden_units_, inter_size_ / tensor_para_size_);
    invokeInPlaceTranspose(weights_ptr[3].second, workspace, inter_size_ / tensor_para_size_, hidden_units_);
    deviceFree(workspace);

    // swizzle for fused qkv gemm and bias
    if (is_fused_qkv_gemm_bias_) {
        {
            T1* h_B = new T1[weights_ptr[0].first];
            cudaD2Hcpy(h_B, weights_ptr[0].second, weights_ptr[0].first);
            T2* h_bias = new T2[vec_ptr[0].first];
            cudaD2Hcpy(h_bias, vec_ptr[0].second, vec_ptr[0].first);
            invokeSwizzleQgmmaWeights(d_model_,
                                      3 * hidden_units_ / tensor_para_size_,
                                      (uint8_t*)h_B,
                                      (uint8_t*)weights_ptr[0].second,
                                      (uint16_t*)h_bias,
                                      (uint16_t*)vec_ptr[0].second);
            delete h_B;
            delete h_bias;
        }
    }
#ifdef FUSE_GEMM_ACT
    {
        T1* h_B = new T1[weights_ptr[1].first];
        cudaD2Hcpy(h_B, weights_ptr[1].second, weights_ptr[1].first);
        T2* h_bias = new T2[vec_ptr[1].first];
        cudaD2Hcpy(h_bias, vec_ptr[1].second, vec_ptr[1].first);
        invokeSwizzleQgmmaWeights(hidden_units_ / tensor_para_size_,
                                  d_model_,
                                  (uint8_t*)h_B,
                                  (uint8_t*)weights_ptr[1].second,
                                  (uint16_t*)h_bias,
                                  (uint16_t*)vec_ptr[1].second);
        delete h_B;
        delete h_bias;
    }
    // swizzle for fused ffn gemm 1 and bias
    {
        T1* h_B = new T1[weights_ptr[2].first];
        cudaD2Hcpy(h_B, weights_ptr[2].second, weights_ptr[2].first);
        T2* h_bias = new T2[vec_ptr[4].first];
        cudaD2Hcpy(h_bias, vec_ptr[4].second, vec_ptr[4].first);
        invokeSwizzleQgmmaWeights(hidden_units_,
                                  inter_size_ / tensor_para_size_,
                                  (uint8_t*)h_B,
                                  (uint8_t*)weights_ptr[2].second,
                                  (uint16_t*)h_bias,
                                  (uint16_t*)vec_ptr[4].second);
        delete h_B;
        delete h_bias;
    }
    // swizzle for fused ffn gemm 2 and bias
    {
        // T1* h_B = new T1[weights_ptr[3].first];
        // cudaD2Hcpy(h_B, weights_ptr[3].second, weights_ptr[3].first);
        // T2* h_bias = new T2[vec_ptr[5].first];
        // cudaD2Hcpy(h_bias, vec_ptr[5].second, vec_ptr[5].first);
        // invokeSwizzleQgmmaWeights(inter_size_ / tensor_para_size_,
        //                           hidden_units_,
        //                           (uint8_t*)h_B,
        //                           (uint8_t*)weights_ptr[3].second,
        //                           (uint16_t*)h_bias,
        //                           (uint16_t*)vec_ptr[5].second);
        // delete h_B;
        // delete h_bias;
    }
#endif
}

template struct BertFP8LayerWeight<__nv_fp8_e4m3, __nv_bfloat16>;

}  // namespace fastertransformer
