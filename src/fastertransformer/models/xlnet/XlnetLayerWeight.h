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
#include "src/fastertransformer/layers/xlnet_attention_layers/XlnetAttentionWeight.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {
#define NUM_WEIGHTS 15
template<typename T>
struct XlnetLayerWeight {
    XlnetAttentionWeight<T> attention_weights;
    LayerNormWeight<T> attn_layernorm_weights;
    FfnWeight<T> ffn_weights;
    LayerNormWeight<T> ffn_layernorm_weights;

    XlnetLayerWeight() = default;
    XlnetLayerWeight(const int hidden_units, const int inter_size): hidden_units_(hidden_units), inter_size_(inter_size)
    {
        setWeightSize();
        for (int i = 0; i < NUM_WEIGHTS; i++) {
            deviceMalloc(&weights_ptr[i], weights_size[i]);
        }

        setWeightPtr();
    }

    ~XlnetLayerWeight()
    {
        if (is_maintain_buffer == true) {
            for (int i = 0; i < NUM_WEIGHTS; i++) {
                deviceFree(weights_ptr[i]);
            }

            attention_weights.attr_kernel_Q = nullptr;
            attention_weights.attr_kernel_K = nullptr;
            attention_weights.attr_kernel_V = nullptr;
            attention_weights.attr_pos_emb = nullptr;
            attention_weights.attr_bias_Q_w = nullptr;
            attention_weights.attr_bias_Q_r = nullptr;
            attention_weights.attr_bias_Q_s = nullptr;
            attention_weights.attr_seg_embed = nullptr;
            attention_weights.attr_proj_o = nullptr;

            attn_layernorm_weights.gamma = nullptr;
            attn_layernorm_weights.beta = nullptr;
            ffn_weights.intermediate_weight.kernel = nullptr;
            ffn_weights.intermediate_weight.bias = nullptr;
            ffn_weights.output_weight.kernel = nullptr;
            ffn_weights.output_weight.bias = nullptr;
            ffn_layernorm_weights.gamma = nullptr;
            ffn_layernorm_weights.beta = nullptr;
            is_maintain_buffer = false;
        }
    }

    XlnetLayerWeight(const XlnetLayerWeight& other): hidden_units_(other.hidden_units_), inter_size_(other.inter_size_)
    {
        setWeightSize();
        for (int i = 0; i < NUM_WEIGHTS; i++) {
            deviceMalloc(&weights_ptr[i], weights_size[i]);
            cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
        }
        setWeightPtr();
    }

    XlnetLayerWeight& operator=(const XlnetLayerWeight& other)
    {
        hidden_units_ = other.hidden_units_;
        inter_size_ = other.inter_size_;
        setWeightSize();
        for (int i = 0; i < NUM_WEIGHTS; i++) {
            deviceMalloc(&weights_ptr[i], weights_size[i]);
            cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
        }
        setWeightPtr();
    }

    T** getWeightPtrs()
    {
        return weights_ptr;
    }

    int* getWeightSizes()
    {
        return weights_size;
    }

private:
    void setWeightPtr()
    {
        attention_weights.attr_kernel_Q = weights_ptr[0];
        attention_weights.attr_kernel_K = weights_ptr[0] + hidden_units_ * hidden_units_;
        attention_weights.attr_kernel_V = weights_ptr[0] + hidden_units_ * hidden_units_ * 2;
        attention_weights.attr_pos_emb = weights_ptr[1];
        attention_weights.attr_bias_Q_w = weights_ptr[2];
        attention_weights.attr_bias_Q_r = weights_ptr[3];
        attention_weights.attr_bias_Q_s = weights_ptr[4];
        attention_weights.attr_seg_embed = weights_ptr[5];
        attention_weights.attr_proj_o = weights_ptr[6];

        attn_layernorm_weights.gamma = weights_ptr[7];
        attn_layernorm_weights.beta = weights_ptr[8];
        ffn_weights.intermediate_weight.kernel = weights_ptr[9];
        ffn_weights.intermediate_weight.bias = weights_ptr[10];
        ffn_weights.output_weight.kernel = weights_ptr[11];
        ffn_weights.output_weight.bias = weights_ptr[12];
        ffn_layernorm_weights.gamma = weights_ptr[13];
        ffn_layernorm_weights.beta = weights_ptr[14];

        is_maintain_buffer = true;
    }

    void setWeightSize()
    {
        weights_size[0] = hidden_units_ * hidden_units_ * 3;
        weights_size[1] = hidden_units_ * hidden_units_;
        weights_size[2] = hidden_units_;
        weights_size[3] = hidden_units_;
        weights_size[4] = hidden_units_;
        weights_size[5] = hidden_units_ * 2;
        weights_size[6] = hidden_units_ * hidden_units_;
        weights_size[7] = hidden_units_;
        weights_size[8] = hidden_units_;
        weights_size[9] = hidden_units_ * inter_size_;
        weights_size[10] = inter_size_;
        weights_size[11] = hidden_units_ * inter_size_;
        weights_size[12] = hidden_units_;
        weights_size[13] = hidden_units_;
        weights_size[14] = hidden_units_;
    }

    int hidden_units_;
    int inter_size_;
    bool is_maintain_buffer = false;
    T* weights_ptr[NUM_WEIGHTS];
    int weights_size[NUM_WEIGHTS];
};

}  // namespace fastertransformer
