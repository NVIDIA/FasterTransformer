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

#include "src/fastertransformer/models/gpt_fp8/GptFP8DecoderLayerWeight.h"
#include "3rdparty/fp8_qgmma_1x1/fp8_qgmma_1x1_utils.h"
#include "src/fastertransformer/utils/cuda_fp8_utils.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T1, typename T2>
GptFP8DecoderLayerWeight<T1, T2>::GptFP8DecoderLayerWeight(const int hidden_units,
                                                           const int inter_size,
                                                           const int tensor_para_size,
                                                           const int tensor_para_rank):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    weights_ptr(4),
    vec_ptr(8),
    scale_ptr(26),
    scale_h_ptr_(26),
    trans_ptr(4)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    mallocWeights();
    setWeightPtr();
}

template<typename T1, typename T2>
GptFP8DecoderLayerWeight<T1, T2>::~GptFP8DecoderLayerWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer == true) {
        for (int i = 0; i < weights_ptr.size(); i++) {
            weights_ptr[i].first = 0;
            deviceFree(weights_ptr[i].second);
        }
        for (int i = 0; i < vec_ptr.size(); i++) {
            vec_ptr[i].first = 0;
            deviceFree(vec_ptr[i].second);
        }
        for (int i = 0; i < scale_ptr.size(); i++) {
            scale_ptr[i].first = 0;
            deviceFree(scale_ptr[i].second);
        }
        for (uint i = 0; i < scale_h_ptr_.size(); i++) {
            delete scale_h_ptr_[i];
        }

        pre_layernorm_weights.beta  = nullptr;
        pre_layernorm_weights.gamma = nullptr;

        self_attention_weights.query_weight.kernel           = nullptr;
        self_attention_weights.query_weight.bias             = nullptr;
        self_attention_weights.query_weight.input_scale      = nullptr;
        self_attention_weights.query_weight.input_scale_inv  = nullptr;
        self_attention_weights.query_weight.output_scale     = nullptr;
        self_attention_weights.query_weight.output_scale_inv = nullptr;
        self_attention_weights.query_weight.weight_scale     = nullptr;
        self_attention_weights.query_weight.weight_scale_inv = nullptr;

        self_attention_weights.attention_output_weight.kernel           = nullptr;
        self_attention_weights.attention_output_weight.bias             = nullptr;
        self_attention_weights.attention_output_weight.input_scale      = nullptr;
        self_attention_weights.attention_output_weight.input_scale_inv  = nullptr;
        self_attention_weights.attention_output_weight.output_scale     = nullptr;
        self_attention_weights.attention_output_weight.output_scale_inv = nullptr;
        self_attention_weights.attention_output_weight.weight_scale     = nullptr;
        self_attention_weights.attention_output_weight.weight_scale_inv = nullptr;

        self_attn_layernorm_weights.beta  = nullptr;
        self_attn_layernorm_weights.gamma = nullptr;

        ffn_weights.intermediate_weight.kernel           = nullptr;
        ffn_weights.intermediate_weight.bias             = nullptr;
        ffn_weights.intermediate_weight.input_scale      = nullptr;
        ffn_weights.intermediate_weight.input_scale_inv  = nullptr;
        ffn_weights.intermediate_weight.output_scale     = nullptr;
        ffn_weights.intermediate_weight.output_scale_inv = nullptr;
        ffn_weights.intermediate_weight.weight_scale     = nullptr;
        ffn_weights.intermediate_weight.weight_scale_inv = nullptr;

        ffn_weights.output_weight.kernel           = nullptr;
        ffn_weights.output_weight.bias             = nullptr;
        ffn_weights.output_weight.input_scale      = nullptr;
        ffn_weights.output_weight.input_scale_inv  = nullptr;
        ffn_weights.output_weight.output_scale     = nullptr;
        ffn_weights.output_weight.output_scale_inv = nullptr;
        ffn_weights.output_weight.weight_scale     = nullptr;
        ffn_weights.output_weight.weight_scale_inv = nullptr;

        self_attention_weights.identity_h_scale = nullptr;
        ffn_weights.identity_h_scale            = nullptr;

        self_attention_weights.qk_scale     = nullptr;
        self_attention_weights.qk_scale_inv = nullptr;

        deviceFree(fp8_qkv_bias);
        fp8_qkv_bias = nullptr;
        deviceFree(identity_scale);
        identity_scale = nullptr;
        delete identity_h_scale;
        identity_h_scale   = nullptr;
        is_maintain_buffer = false;
    }
}

template<typename T1, typename T2>
GptFP8DecoderLayerWeight<T1, T2>::GptFP8DecoderLayerWeight(const GptFP8DecoderLayerWeight& other):
    hidden_units_(other.hidden_units_),
    inter_size_(other.inter_size_),
    tensor_para_size_(other.tensor_para_size_),
    tensor_para_rank_(other.tensor_para_rank_),
    weights_ptr(4),
    vec_ptr(8),
    scale_ptr(26),
    scale_h_ptr_(26),
    trans_ptr(4)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    mallocWeights();
    for (int i = 0; i < weights_ptr.size(); i++) {
        cudaD2Dcpy(weights_ptr[i].second, other.weights_ptr[i].second, weights_ptr[i].first);
    }
    for (int i = 0; i < vec_ptr.size(); i++) {
        cudaD2Dcpy(vec_ptr[i].second, other.vec_ptr[i].second, vec_ptr[i].first);
    }
    for (int i = 0; i < scale_ptr.size(); i++) {
        cudaD2Dcpy(scale_ptr[i].second, other.scale_ptr[i].second, scale_ptr[i].first);
    }
    for (int i = 0; i < scale_h_ptr_.size(); i++) {
        *(scale_h_ptr_[i]) = *(other.scale_h_ptr_[i]);
    }
    setWeightPtr();
}

template<typename T1, typename T2>
GptFP8DecoderLayerWeight<T1, T2>& GptFP8DecoderLayerWeight<T1, T2>::operator=(const GptFP8DecoderLayerWeight& other)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    hidden_units_     = other.hidden_units_;
    inter_size_       = other.inter_size_;
    tensor_para_size_ = other.tensor_para_size_;
    tensor_para_rank_ = other.tensor_para_rank_;

    mallocWeights();
    for (int i = 0; i < weights_ptr.size(); i++) {
        cudaD2Dcpy(weights_ptr[i].second, other.weights_ptr[i].second, weights_ptr[i].first);
    }
    for (int i = 0; i < vec_ptr.size(); i++) {
        cudaD2Dcpy(vec_ptr[i].second, other.vec_ptr[i].second, vec_ptr[i].first);
    }
    for (int i = 0; i < scale_ptr.size(); i++) {
        cudaD2Dcpy(scale_ptr[i].second, other.scale_ptr[i].second, scale_ptr[i].first);
    }
    for (int i = 0; i < scale_h_ptr_.size(); i++) {
        *(scale_h_ptr_[i]) = *(other.scale_h_ptr_[i]);
    }
    setWeightPtr();
    return *this;
}

template<typename T1, typename T2>
void GptFP8DecoderLayerWeight<T1, T2>::loadModel(std::string dir_path)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(is_maintain_buffer == true);

    loadWeightFromBin<T2>(vec_ptr[0].second, {(int)vec_ptr[0].first}, dir_path + ".input_layernorm.bias.bin");
    loadWeightFromBin<T2>(vec_ptr[1].second, {(int)vec_ptr[1].first}, dir_path + ".input_layernorm.weight.bin");
    loadWeightFromBin<T2>(vec_ptr[2].second,
                          {(int)vec_ptr[2].first},
                          dir_path + ".attention.query_key_value.bias." + std::to_string(tensor_para_rank_) + ".bin");
    loadWeightFromBin<T2>(vec_ptr[3].second, {(int)vec_ptr[3].first}, dir_path + ".attention.dense.bias.bin");
    loadWeightFromBin<T2>(vec_ptr[4].second, {(int)vec_ptr[4].first}, dir_path + ".post_attention_layernorm.bias.bin");
    loadWeightFromBin<T2>(
        vec_ptr[5].second, {(int)vec_ptr[5].first}, dir_path + ".post_attention_layernorm.weight.bin");
    loadWeightFromBin<T2>(vec_ptr[6].second,
                          {(int)vec_ptr[6].first},
                          dir_path + ".mlp.dense_h_to_4h.bias." + std::to_string(tensor_para_rank_) + ".bin");
    loadWeightFromBin<T2>(vec_ptr[7].second, {(int)vec_ptr[7].first}, dir_path + ".mlp.dense_4h_to_h.bias.bin");

    // Load scaling
    {
        std::string quantizer_list[13] = {
            ".attention.query_key_value.fp_linear.fi.bin",
            ".attention.query_key_value.fp_linear.fo.bin",
            ".attention.query_key_value.fp_linear.fw.bin",
            ".attention.dense.fp_linear.fi.bin",
            ".attention.dense.fp_linear.fo.bin",
            ".attention.dense.fp_linear.fw.bin",
            ".mlp.dense_h_to_4h.fp_linear.fi.bin",
            ".mlp.dense_h_to_4h.fp_linear.fo.bin",
            ".mlp.dense_h_to_4h.fp_linear.fw.bin",
            ".mlp.dense_4h_to_h.fp_linear.fi.bin",
            ".mlp.dense_4h_to_h.fp_linear.fo.bin",
            ".mlp.dense_4h_to_h.fp_linear.fw.bin",
            ".attention.softmax.fo.bin",
        };

        float* d_scale_inv = nullptr;
        deviceMalloc(&d_scale_inv, inter_size_, false);
        float* h_scale     = new float[inter_size_];
        float* h_scale_inv = new float[inter_size_];

        for (int i = 0; i < 13; i++) {
            if (checkIfFileExist(dir_path + quantizer_list[i])) {
                loadWeightFromBin<float>(d_scale_inv, {(int)scale_ptr[i * 2].first}, dir_path + quantizer_list[i]);
                cudaD2Hcpy(h_scale_inv, d_scale_inv, scale_ptr[i * 2].first);
            }
            else {
                // If we cannot find the scales, use identity scales.
                FT_LOG_WARNING(fmtstr("Cannot find %s, use identity scales.", (dir_path + quantizer_list[i]).c_str()));
                for (int j = 0; j < scale_ptr[i * 2].first; j++) {
                    h_scale_inv[j] = 1.0f;
                }
            }
            for (int j = 0; j < scale_ptr[i * 2].first; j++) {
                h_scale[j] = 1.0f / h_scale_inv[j];
            }
            cudaH2Dcpy(scale_ptr[i * 2 + 0].second, h_scale, scale_ptr[i * 2 + 0].first);
            cudaH2Dcpy(scale_ptr[i * 2 + 1].second, h_scale_inv, scale_ptr[i * 2 + 1].first);

            if (fp8_mode_ == 2) {
                *(scale_h_ptr_[i * 2 + 0]) = h_scale[0];      // only for per tensor
                *(scale_h_ptr_[i * 2 + 1]) = h_scale_inv[0];  // only for per tensor
            }
        }
        deviceFree(d_scale_inv);
        delete[] h_scale;
        delete[] h_scale_inv;
    }

    {
        float* d_float_weight = nullptr;
        deviceMalloc(&d_float_weight, hidden_units_ * std::max(3 * hidden_units_, inter_size_) / tensor_para_size_);

        if (fp8_mode_ == 1) {
            loadWeightFromBin<float>(d_float_weight,
                                     {(int)weights_ptr[0].first},
                                     dir_path + ".attention.query_key_value.weight." + std::to_string(tensor_para_rank_)
                                         + ".bin");
            invokeQuantizeMatrix<T1, float, QUANTIZE_MODE::PER_CHANNEL>(weights_ptr[0].second,
                                                                        scale_ptr[5].second,
                                                                        d_float_weight,
                                                                        (int)weights_ptr[0].first,
                                                                        scale_ptr[4].first,
                                                                        (cudaStream_t)0);
            loadWeightFromBin<float>(d_float_weight,
                                     {(int)weights_ptr[1].first},
                                     dir_path + ".attention.dense.weight." + std::to_string(tensor_para_rank_)
                                         + ".bin");
            invokeQuantizeMatrix<T1, float, QUANTIZE_MODE::PER_CHANNEL>(weights_ptr[1].second,
                                                                        scale_ptr[11].second,
                                                                        d_float_weight,
                                                                        (int)weights_ptr[1].first,
                                                                        scale_ptr[10].first,
                                                                        (cudaStream_t)0);
            loadWeightFromBin<float>(d_float_weight,
                                     {(int)weights_ptr[2].first},
                                     dir_path + ".mlp.dense_h_to_4h.weight." + std::to_string(tensor_para_rank_)
                                         + ".bin");
            invokeQuantizeMatrix<T1, float, QUANTIZE_MODE::PER_CHANNEL>(weights_ptr[2].second,
                                                                        scale_ptr[17].second,
                                                                        d_float_weight,
                                                                        (int)weights_ptr[2].first,
                                                                        scale_ptr[16].first,
                                                                        (cudaStream_t)0);
            loadWeightFromBin<float>(d_float_weight,
                                     {(int)weights_ptr[3].first},
                                     dir_path + ".mlp.dense_4h_to_h.weight." + std::to_string(tensor_para_rank_)
                                         + ".bin");
            invokeQuantizeMatrix<T1, float, QUANTIZE_MODE::PER_CHANNEL>(weights_ptr[3].second,
                                                                        scale_ptr[23].second,
                                                                        d_float_weight,
                                                                        (int)weights_ptr[3].first,
                                                                        scale_ptr[22].first,
                                                                        (cudaStream_t)0);
        }
        else if (fp8_mode_ == 2) {
            loadWeightFromBin<float>(d_float_weight,
                                     {(int)weights_ptr[0].first},
                                     dir_path + ".attention.query_key_value.weight." + std::to_string(tensor_para_rank_)
                                         + ".bin");
            invokeQuantizeMatrix<T1, float, QUANTIZE_MODE::PER_TENSOR>(weights_ptr[0].second,
                                                                       scale_ptr[5].second,
                                                                       d_float_weight,
                                                                       (int)weights_ptr[0].first,
                                                                       1,
                                                                       (cudaStream_t)0);
            loadWeightFromBin<float>(d_float_weight,
                                     {(int)weights_ptr[1].first},
                                     dir_path + ".attention.dense.weight." + std::to_string(tensor_para_rank_)
                                         + ".bin");
            invokeQuantizeMatrix<T1, float, QUANTIZE_MODE::PER_TENSOR>(weights_ptr[1].second,
                                                                       scale_ptr[11].second,
                                                                       d_float_weight,
                                                                       (int)weights_ptr[1].first,
                                                                       1,
                                                                       (cudaStream_t)0);
            loadWeightFromBin<float>(d_float_weight,
                                     {(int)weights_ptr[2].first},
                                     dir_path + ".mlp.dense_h_to_4h.weight." + std::to_string(tensor_para_rank_)
                                         + ".bin");
            invokeQuantizeMatrix<T1, float, QUANTIZE_MODE::PER_TENSOR>(weights_ptr[2].second,
                                                                       scale_ptr[17].second,
                                                                       d_float_weight,
                                                                       (int)weights_ptr[2].first,
                                                                       1,
                                                                       (cudaStream_t)0);
            loadWeightFromBin<float>(d_float_weight,
                                     {(int)weights_ptr[3].first},
                                     dir_path + ".mlp.dense_4h_to_h.weight." + std::to_string(tensor_para_rank_)
                                         + ".bin");
            invokeQuantizeMatrix<T1, float, QUANTIZE_MODE::PER_TENSOR>(weights_ptr[3].second,
                                                                       scale_ptr[23].second,
                                                                       d_float_weight,
                                                                       (int)weights_ptr[3].first,
                                                                       1,
                                                                       (cudaStream_t)0);
        }

        invokeQuantizeMatrix<T1, T2, QUANTIZE_MODE::PER_TENSOR>(
            fp8_qkv_bias, scale_ptr[3].second, vec_ptr[2].second, vec_ptr[2].first, 1, (cudaStream_t)0);

        deviceFree(d_float_weight);
    }
}

template<typename T1, typename T2>
void GptFP8DecoderLayerWeight<T1, T2>::setWeightPtr()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    self_attention_weights.query_weight.kernel            = weights_ptr[0].second;
    self_attention_weights.attention_output_weight.kernel = weights_ptr[1].second;
    ffn_weights.intermediate_weight.kernel                = weights_ptr[2].second;
    ffn_weights.output_weight.kernel                      = weights_ptr[3].second;

    self_attention_weights.query_weight.input_scale                 = scale_ptr[0].second;
    self_attention_weights.query_weight.input_scale_inv             = scale_ptr[1].second;
    self_attention_weights.query_weight.output_scale                = scale_ptr[2].second;
    self_attention_weights.query_weight.output_scale_inv            = scale_ptr[3].second;
    self_attention_weights.query_weight.weight_scale                = scale_ptr[4].second;
    self_attention_weights.query_weight.weight_scale_inv            = scale_ptr[5].second;
    self_attention_weights.attention_output_weight.input_scale      = scale_ptr[6].second;
    self_attention_weights.attention_output_weight.input_scale_inv  = scale_ptr[7].second;
    self_attention_weights.attention_output_weight.output_scale     = scale_ptr[8].second;
    self_attention_weights.attention_output_weight.output_scale_inv = scale_ptr[9].second;
    self_attention_weights.attention_output_weight.weight_scale     = scale_ptr[10].second;
    self_attention_weights.attention_output_weight.weight_scale_inv = scale_ptr[11].second;
    ffn_weights.intermediate_weight.input_scale                     = scale_ptr[12].second;
    ffn_weights.intermediate_weight.input_scale_inv                 = scale_ptr[13].second;
    ffn_weights.intermediate_weight.output_scale                    = scale_ptr[14].second;
    ffn_weights.intermediate_weight.output_scale_inv                = scale_ptr[15].second;
    ffn_weights.intermediate_weight.weight_scale                    = scale_ptr[16].second;
    ffn_weights.intermediate_weight.weight_scale_inv                = scale_ptr[17].second;
    ffn_weights.output_weight.input_scale                           = scale_ptr[18].second;
    ffn_weights.output_weight.input_scale_inv                       = scale_ptr[19].second;
    ffn_weights.output_weight.output_scale                          = scale_ptr[20].second;
    ffn_weights.output_weight.output_scale_inv                      = scale_ptr[21].second;
    ffn_weights.output_weight.weight_scale                          = scale_ptr[22].second;
    ffn_weights.output_weight.weight_scale_inv                      = scale_ptr[23].second;
    self_attention_weights.qk_scale                                 = scale_ptr[24].second;
    self_attention_weights.qk_scale_inv                             = scale_ptr[25].second;

    self_attention_weights.identity_scale   = identity_scale;
    ffn_weights.identity_scale              = identity_scale;
    self_attention_weights.identity_h_scale = identity_h_scale;
    ffn_weights.identity_h_scale            = identity_h_scale;

    self_attention_weights.query_weight.input_h_scale                 = scale_h_ptr_[0];
    self_attention_weights.query_weight.input_h_scale_inv             = scale_h_ptr_[1];
    self_attention_weights.query_weight.output_h_scale                = scale_h_ptr_[2];
    self_attention_weights.query_weight.output_h_scale_inv            = scale_h_ptr_[3];
    self_attention_weights.query_weight.weight_h_scale                = scale_h_ptr_[4];
    self_attention_weights.query_weight.weight_h_scale_inv            = scale_h_ptr_[5];
    self_attention_weights.attention_output_weight.input_h_scale      = scale_h_ptr_[6];
    self_attention_weights.attention_output_weight.input_h_scale_inv  = scale_h_ptr_[7];
    self_attention_weights.attention_output_weight.output_h_scale     = scale_h_ptr_[8];
    self_attention_weights.attention_output_weight.output_h_scale_inv = scale_h_ptr_[9];
    self_attention_weights.attention_output_weight.weight_h_scale     = scale_h_ptr_[10];
    self_attention_weights.attention_output_weight.weight_h_scale_inv = scale_h_ptr_[11];
    ffn_weights.intermediate_weight.input_h_scale                     = scale_h_ptr_[12];
    ffn_weights.intermediate_weight.input_h_scale_inv                 = scale_h_ptr_[13];
    ffn_weights.intermediate_weight.output_h_scale                    = scale_h_ptr_[14];
    ffn_weights.intermediate_weight.output_h_scale_inv                = scale_h_ptr_[15];
    ffn_weights.intermediate_weight.weight_h_scale                    = scale_h_ptr_[16];
    ffn_weights.intermediate_weight.weight_h_scale_inv                = scale_h_ptr_[17];
    ffn_weights.output_weight.input_h_scale                           = scale_h_ptr_[18];
    ffn_weights.output_weight.input_h_scale_inv                       = scale_h_ptr_[19];
    ffn_weights.output_weight.output_h_scale                          = scale_h_ptr_[20];
    ffn_weights.output_weight.output_h_scale_inv                      = scale_h_ptr_[21];
    ffn_weights.output_weight.weight_h_scale                          = scale_h_ptr_[22];
    ffn_weights.output_weight.weight_h_scale_inv                      = scale_h_ptr_[23];
    self_attention_weights.qk_h_scale                                 = scale_h_ptr_[24];
    self_attention_weights.qk_h_scale_inv                             = scale_h_ptr_[25];

    pre_layernorm_weights.beta                          = vec_ptr[0].second;
    pre_layernorm_weights.gamma                         = vec_ptr[1].second;
    self_attention_weights.query_weight.bias            = vec_ptr[2].second;
    self_attention_weights.attention_output_weight.bias = vec_ptr[3].second;
    self_attn_layernorm_weights.beta                    = vec_ptr[4].second;
    self_attn_layernorm_weights.gamma                   = vec_ptr[5].second;
    ffn_weights.intermediate_weight.bias                = vec_ptr[6].second;
    ffn_weights.output_weight.bias                      = vec_ptr[7].second;

    self_attention_weights.query_weight.fp8_bias = fp8_qkv_bias;
    is_maintain_buffer                           = true;
}

template<typename T1, typename T2>
void GptFP8DecoderLayerWeight<T1, T2>::mallocWeights()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    weights_ptr[0].first = hidden_units_ * 3 * hidden_units_ / tensor_para_size_;
    weights_ptr[1].first = hidden_units_ / tensor_para_size_ * hidden_units_;
    weights_ptr[2].first = hidden_units_ * inter_size_ / tensor_para_size_;
    weights_ptr[3].first = inter_size_ / tensor_para_size_ * hidden_units_;

    vec_ptr[0].first = hidden_units_;
    vec_ptr[1].first = hidden_units_;
    vec_ptr[2].first = 3 * hidden_units_ / tensor_para_size_;
    vec_ptr[3].first = hidden_units_;
    vec_ptr[4].first = hidden_units_;
    vec_ptr[5].first = hidden_units_;
    vec_ptr[6].first = inter_size_ / tensor_para_size_;
    vec_ptr[7].first = hidden_units_;

    if (fp8_mode_ == 1) {
        scale_ptr[0].first  = 1;
        scale_ptr[1].first  = 1;
        scale_ptr[2].first  = 1;
        scale_ptr[3].first  = 1;
        scale_ptr[4].first  = 3 * hidden_units_ / tensor_para_size_;
        scale_ptr[5].first  = 3 * hidden_units_ / tensor_para_size_;
        scale_ptr[6].first  = 1;
        scale_ptr[7].first  = 1;
        scale_ptr[8].first  = 1;
        scale_ptr[9].first  = 1;
        scale_ptr[10].first = hidden_units_;
        scale_ptr[11].first = hidden_units_;
        scale_ptr[12].first = 1;
        scale_ptr[13].first = 1;
        scale_ptr[14].first = 1;
        scale_ptr[15].first = 1;
        scale_ptr[16].first = inter_size_ / tensor_para_size_;
        scale_ptr[17].first = inter_size_ / tensor_para_size_;
        scale_ptr[18].first = 1;
        scale_ptr[19].first = 1;
        scale_ptr[20].first = 1;
        scale_ptr[21].first = 1;
        scale_ptr[22].first = hidden_units_;
        scale_ptr[23].first = hidden_units_;
        scale_ptr[24].first = 1;
        scale_ptr[25].first = 1;
    }
    else if (fp8_mode_ == 2) {
        for (int i = 0; i < scale_ptr.size(); i++) {
            scale_ptr[i].first = 1;
        }
    }
    else {
        FT_CHECK(false);
    }

    for (int i = 0; i < weights_ptr.size(); i++) {
        deviceMalloc(&weights_ptr[i].second, weights_ptr[i].first);
    }
    for (int i = 0; i < vec_ptr.size(); i++) {
        deviceMalloc(&vec_ptr[i].second, vec_ptr[i].first);
    }
    for (int i = 0; i < scale_ptr.size(); i++) {
        deviceMalloc(&scale_ptr[i].second, scale_ptr[i].first);
        deviceFill(scale_ptr[i].second, scale_ptr[i].first, 1.0f);
    }
    deviceMalloc(&fp8_qkv_bias, vec_ptr[2].first);
    for (uint i = 0; i < scale_h_ptr_.size(); i++) {
        scale_h_ptr_[i] = new float(1.0f);
    }

    // identity scalar
    deviceMalloc(&identity_scale, hidden_units_);
    deviceFill(identity_scale, hidden_units_, 1.0f);
    identity_h_scale = new float{1.0f};
}

// #ifdef SPARSITY_ENABLED
// template<typename T1, typename T2>
// void GptFP8DecoderLayerWeight<T1, T2>::compress_weights(cublasMMWrapper& cublas_wrapper, int hidden_dim)
// {
//     hidden_units_ = hidden_dim;
//     inter_size_ = 4 * hidden_units_;

//     const size_t num_sparse_weights = 4;
//     size_t shapes[num_sparse_weights][2] = {{hidden_units_, 3 * hidden_units_ / tensor_para_size_},
//                                             {hidden_units_ / tensor_para_size_, hidden_units_},
//                                             {hidden_units_, inter_size_ / tensor_para_size_},
//                                             {inter_size_ / tensor_para_size_, hidden_units_}};

//     const T* dense_weights[num_sparse_weights] = {self_attention_weights.query_weight.kernel,
//                                                   self_attention_weights.attention_output_weight.kernel,
//                                                   ffn_weights.intermediate_weight.kernel,
//                                                   ffn_weights.output_weight.kernel};

//     for (size_t i = 0; i < num_sparse_weights; ++i) {
//         int m = shapes[i][1];
//         int k = shapes[i][0];
//         size_t compressed_size = cublas_wrapper.getSparseMatrixSize(m, k);
//         deviceMalloc(&sp_weights_ptr[i], static_cast<int>(compressed_size), false);
//         cublas_wrapper.compressMatrix(dense_weights[i], sp_weights_ptr[i], m, k);
//     }

//     self_attention_weights.query_weight.sp_kernel = sp_weights_ptr[0];
//     self_attention_weights.attention_output_weight.sp_kernel = sp_weights_ptr[1];
//     ffn_weights.intermediate_weight.sp_kernel = sp_weights_ptr[2];
//     ffn_weights.output_weight.sp_kernel = sp_weights_ptr[3];
//     is_maintain_sp_buffer = true;
// }
// #endif

template<typename T1, typename T2>
void GptFP8DecoderLayerWeight<T1, T2>::transposeWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    T1* workspace;
    deviceMalloc(&workspace, hidden_units_ / tensor_para_size_ * std::max(3 * hidden_units_, inter_size_));
    invokeInPlaceTranspose(weights_ptr[0].second, workspace, hidden_units_, 3 * hidden_units_ / tensor_para_size_);
    invokeInPlaceTranspose(weights_ptr[1].second, workspace, hidden_units_ / tensor_para_size_, hidden_units_);
    invokeInPlaceTranspose(weights_ptr[2].second, workspace, hidden_units_, inter_size_ / tensor_para_size_);
    invokeInPlaceTranspose(weights_ptr[3].second, workspace, inter_size_ / tensor_para_size_, hidden_units_);
    deviceFree(workspace);

#ifdef FUSE_GEMM_ACT
#ifdef USE_QGMMA
    {
        T1* h_B = new T1[weights_ptr[2].first];
        cudaD2Hcpy(h_B, weights_ptr[2].second, weights_ptr[2].first);
        T2* h_bias = new T2[vec_ptr[6].first];
        cudaD2Hcpy(h_bias, vec_ptr[6].second, vec_ptr[6].first);
        invokeSwizzleQgmmaWeights(hidden_units_,
                                  inter_size_ / tensor_para_size_,
                                  (uint8_t*)h_B,
                                  (uint8_t*)weights_ptr[2].second,
                                  (uint16_t*)h_bias,
                                  (uint16_t*)vec_ptr[6].second);
        delete h_B;
        delete h_bias;
    }
#endif
#endif
}

template struct GptFP8DecoderLayerWeight<__nv_fp8_e4m3, __nv_bfloat16>;

}  // namespace fastertransformer
