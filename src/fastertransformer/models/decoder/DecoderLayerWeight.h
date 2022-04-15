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
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
struct DecoderLayerWeight {

    DecoderLayerWeight() = default;
    DecoderLayerWeight(const int hidden_units, const int inter_size, const int mem_hidden_units):
        hidden_units_(hidden_units), inter_size_(inter_size), mem_hidden_units_(mem_hidden_units)
    {
        mallocWeights();
        setWeightPtr();
    }

    ~DecoderLayerWeight()
    {
        if (is_maintain_buffer == true) {
            for (int i = 0; i < 21; i++) {
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

            cross_attention_weights.query_weight.kernel = nullptr;
            cross_attention_weights.query_weight.bias = nullptr;
            cross_attention_weights.key_weight.kernel = nullptr;
            cross_attention_weights.key_weight.bias = nullptr;
            cross_attention_weights.value_weight.kernel = nullptr;
            cross_attention_weights.value_weight.bias = nullptr;
            cross_attention_weights.attention_output_weight.kernel = nullptr;
            cross_attention_weights.attention_output_weight.bias = nullptr;
            cross_attn_layernorm_weights.beta = nullptr;
            cross_attn_layernorm_weights.gamma = nullptr;

            ffn_weights.intermediate_weight.kernel = nullptr;
            ffn_weights.intermediate_weight.bias = nullptr;
            ffn_weights.output_weight.kernel = nullptr;
            ffn_weights.output_weight.bias = nullptr;
            is_maintain_buffer = false;
        }
    }

    DecoderLayerWeight(const DecoderLayerWeight& other):
        hidden_units_(other.hidden_units_), inter_size_(other.inter_size_), mem_hidden_units_(other.mem_hidden_units_)
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

        cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], hidden_units_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], hidden_units_);
        cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], mem_hidden_units_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], hidden_units_);
        cudaD2Dcpy(weights_ptr[12], other.weights_ptr[12], mem_hidden_units_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[13], other.weights_ptr[13], hidden_units_);
        cudaD2Dcpy(weights_ptr[14], other.weights_ptr[14], hidden_units_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[15], other.weights_ptr[15], hidden_units_);
        cudaD2Dcpy(weights_ptr[16], other.weights_ptr[16], hidden_units_);
        cudaD2Dcpy(weights_ptr[17], other.weights_ptr[17], hidden_units_);

        cudaD2Dcpy(weights_ptr[18], other.weights_ptr[18], hidden_units_ * inter_size_);
        cudaD2Dcpy(weights_ptr[19], other.weights_ptr[19], inter_size_);
        cudaD2Dcpy(weights_ptr[20], other.weights_ptr[20], inter_size_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[21], other.weights_ptr[21], hidden_units_);

        setWeightPtr();
    }

    DecoderLayerWeight& operator=(const DecoderLayerWeight& other)
    {
        hidden_units_ = other.hidden_units_;
        inter_size_ = other.inter_size_;
        mem_hidden_units_ = other.mem_hidden_units_;

        mallocWeights();
        cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
        cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
        cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ * 3 * hidden_units_);
        cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], 3 * hidden_units_);
        cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
        cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_);
        cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], hidden_units_);

        cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], hidden_units_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], hidden_units_);
        cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], mem_hidden_units_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], hidden_units_);
        cudaD2Dcpy(weights_ptr[12], other.weights_ptr[12], mem_hidden_units_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[13], other.weights_ptr[13], hidden_units_);
        cudaD2Dcpy(weights_ptr[14], other.weights_ptr[14], hidden_units_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[15], other.weights_ptr[15], hidden_units_);
        cudaD2Dcpy(weights_ptr[16], other.weights_ptr[16], hidden_units_);
        cudaD2Dcpy(weights_ptr[17], other.weights_ptr[17], hidden_units_);

        cudaD2Dcpy(weights_ptr[18], other.weights_ptr[18], hidden_units_ * inter_size_);
        cudaD2Dcpy(weights_ptr[19], other.weights_ptr[19], inter_size_);
        cudaD2Dcpy(weights_ptr[20], other.weights_ptr[20], inter_size_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[21], other.weights_ptr[21], hidden_units_);

        setWeightPtr();
    }

    LayerNormWeight<T> pre_layernorm_weights;
    AttentionWeight<T> self_attention_weights;
    LayerNormWeight<T> self_attn_layernorm_weights;
    AttentionWeight<T> cross_attention_weights;
    LayerNormWeight<T> cross_attn_layernorm_weights;
    FfnWeight<T> ffn_weights;

private:
    void setWeightPtr()
    {
        pre_layernorm_weights.beta = weights_ptr[0];
        pre_layernorm_weights.gamma = weights_ptr[1];
        self_attention_weights.query_weight.kernel = weights_ptr[2];
        self_attention_weights.query_weight.bias = weights_ptr[3];
        self_attention_weights.attention_output_weight.kernel = weights_ptr[4];
        self_attention_weights.attention_output_weight.bias = weights_ptr[5];
        self_attn_layernorm_weights.beta = weights_ptr[6];
        self_attn_layernorm_weights.gamma = weights_ptr[7];

        cross_attention_weights.query_weight.kernel = weights_ptr[8];
        cross_attention_weights.query_weight.bias = weights_ptr[9];
        cross_attention_weights.key_weight.kernel = weights_ptr[10];
        cross_attention_weights.key_weight.bias = weights_ptr[11];
        cross_attention_weights.value_weight.kernel = weights_ptr[12];
        cross_attention_weights.value_weight.bias = weights_ptr[13];
        cross_attention_weights.attention_output_weight.kernel = weights_ptr[14];
        cross_attention_weights.attention_output_weight.bias = weights_ptr[15];
        cross_attn_layernorm_weights.beta = weights_ptr[16];
        cross_attn_layernorm_weights.gamma = weights_ptr[17];

        ffn_weights.intermediate_weight.kernel = weights_ptr[18];
        ffn_weights.intermediate_weight.bias = weights_ptr[19];
        ffn_weights.output_weight.kernel = weights_ptr[20];
        ffn_weights.output_weight.bias = weights_ptr[21];
    }

    void mallocWeights()
    {
        deviceMalloc(&weights_ptr[0], hidden_units_);
        deviceMalloc(&weights_ptr[1], hidden_units_);
        deviceMalloc(&weights_ptr[2], hidden_units_ * 3 * hidden_units_);
        deviceMalloc(&weights_ptr[3], 3 * hidden_units_);
        deviceMalloc(&weights_ptr[4], hidden_units_ * hidden_units_);
        deviceMalloc(&weights_ptr[5], hidden_units_);
        deviceMalloc(&weights_ptr[6], hidden_units_);
        deviceMalloc(&weights_ptr[7], hidden_units_);

        deviceMalloc(&weights_ptr[8], hidden_units_ * hidden_units_);
        deviceMalloc(&weights_ptr[9], hidden_units_);
        deviceMalloc(&weights_ptr[10], mem_hidden_units_ * hidden_units_);
        deviceMalloc(&weights_ptr[11], hidden_units_);
        deviceMalloc(&weights_ptr[12], mem_hidden_units_ * hidden_units_);
        deviceMalloc(&weights_ptr[13], hidden_units_);
        deviceMalloc(&weights_ptr[14], hidden_units_ * hidden_units_);
        deviceMalloc(&weights_ptr[15], hidden_units_);
        deviceMalloc(&weights_ptr[16], hidden_units_);
        deviceMalloc(&weights_ptr[17], hidden_units_);

        deviceMalloc(&weights_ptr[18], hidden_units_ * inter_size_);
        deviceMalloc(&weights_ptr[19], inter_size_);
        deviceMalloc(&weights_ptr[20], inter_size_ * hidden_units_);
        deviceMalloc(&weights_ptr[21], hidden_units_);
        is_maintain_buffer = true;
    }

    int hidden_units_;
    int inter_size_;
    int mem_hidden_units_;
    bool is_maintain_buffer = false;
    T* weights_ptr[22];
};

}  // namespace fastertransformer
