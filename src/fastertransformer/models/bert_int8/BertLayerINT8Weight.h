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

#include "src/fastertransformer/layers/FfnINT8Weight.h"
#include "src/fastertransformer/layers/attention_layers_int8/AttentionINT8Weight.h"
#include "src/fastertransformer/models/bert/BertLayerWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
struct BertLayerINT8Weight: BertLayerWeight<T> {

    BertLayerINT8Weight() = default;
    BertLayerINT8Weight(const int hidden_units, const int inter_size):
        hidden_units_(hidden_units), inter_size_(inter_size)
    {

        deviceMalloc(&weights_ptr[0], hidden_units_ * hidden_units_ * 3);  // fuse kernel of qkv
        deviceMalloc(&weights_ptr[1], hidden_units_ * 3);                  // fuse bias of qkv
        deviceMalloc(&weights_ptr[2], hidden_units_ * hidden_units_);
        deviceMalloc(&weights_ptr[3], hidden_units_);
        deviceMalloc(&weights_ptr[4], hidden_units_);
        deviceMalloc(&weights_ptr[5], hidden_units_);
        deviceMalloc(&weights_ptr[6], hidden_units_ * inter_size_);
        deviceMalloc(&weights_ptr[7], inter_size_);
        deviceMalloc(&weights_ptr[8], inter_size_ * hidden_units_);
        deviceMalloc(&weights_ptr[9], hidden_units_);
        deviceMalloc(&weights_ptr[10], hidden_units_);
        deviceMalloc(&weights_ptr[11], hidden_units_);

        scale_list_.size_ = ACTIVATION_AMAX_NUM + 9 * hidden_units + INT8O_GEMM_NUM + TRT_AMAX_NUM + SCALE_RESERVE_NUM;
        scale_list_.p3_offset_ = ACTIVATION_AMAX_NUM + 9 * hidden_units;
        scale_list_.p4_offset_ = ACTIVATION_AMAX_NUM + 9 * hidden_units + INT8O_GEMM_NUM;
        deviceMalloc(&scale_list_ptr[0], scale_list_.size_);
        scale_list_ptr[1] = (float*)malloc(sizeof(float) * scale_list_.size_);

        setWeightPtr();
    }

    ~BertLayerINT8Weight()
    {
        if (is_maintain_buffer == true) {
            for (int i = 0; i < 12; i++) {
                deviceFree(weights_ptr[i]);
            }

            deviceFree(scale_list_ptr[0]);
            free(scale_list_ptr[1]);

            attention_weights.query_weight.kernel = nullptr;
            attention_weights.query_weight.bias = nullptr;
            attention_weights.key_weight.kernel = nullptr;
            attention_weights.key_weight.bias = nullptr;
            attention_weights.value_weight.kernel = nullptr;
            attention_weights.value_weight.bias = nullptr;
            attention_weights.attention_output_weight.kernel = nullptr;
            attention_weights.attention_output_weight.bias = nullptr;
            attention_weights.scale_list_ptr = nullptr;
            attn_layernorm_weights.gamma = nullptr;
            attn_layernorm_weights.beta = nullptr;
            ffn_weights.intermediate_weight.kernel = nullptr;
            ffn_weights.intermediate_weight.bias = nullptr;
            ffn_weights.output_weight.kernel = nullptr;
            ffn_weights.output_weight.bias = nullptr;
            ffn_weights.scale_list_ptr = nullptr;
            ffn_layernorm_weights.gamma = nullptr;
            ffn_layernorm_weights.beta = nullptr;
            is_maintain_buffer = false;
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
    }

    BertLayerINT8Weight(const BertLayerINT8Weight& other):
        hidden_units_(other.hidden_units_), inter_size_(other.inter_size_)
    {
        deviceMalloc(&weights_ptr[0], hidden_units_ * hidden_units_ * 3);
        cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_ * hidden_units_ * 3);
        deviceMalloc(&weights_ptr[1], hidden_units_ * 3);
        cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_ * 3);
        deviceMalloc(&weights_ptr[2], hidden_units_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ * hidden_units_);
        deviceMalloc(&weights_ptr[3], hidden_units_);
        cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], hidden_units_);
        deviceMalloc(&weights_ptr[4], hidden_units_);
        cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_);
        deviceMalloc(&weights_ptr[5], hidden_units_);
        cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
        deviceMalloc(&weights_ptr[6], hidden_units_ * inter_size_);
        cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_ * inter_size_);
        deviceMalloc(&weights_ptr[7], inter_size_);
        cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], inter_size_);
        deviceMalloc(&weights_ptr[8], inter_size_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], inter_size_ * hidden_units_);
        deviceMalloc(&weights_ptr[9], hidden_units_);
        cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], hidden_units_);
        deviceMalloc(&weights_ptr[10], hidden_units_);
        cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], hidden_units_);
        deviceMalloc(&weights_ptr[11], hidden_units_);
        cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], hidden_units_);

        scale_list_.size_ = other.scale_list_.size_;
        scale_list_.p3_offset_ = other.scale_list_.p3_offset_;
        scale_list_.p4_offset_ = other.scale_list_.p4_offset_;
        deviceMalloc(&scale_list_ptr[0], scale_list_.size_);
        cudaD2Dcpy(scale_list_ptr[0], other.scale_list_ptr[0], scale_list_.size_);
        scale_list_ptr[1] = (float*)malloc(sizeof(float) * scale_list_.size_);
        memcpy(scale_list_ptr[1], other.scale_list_ptr[1], sizeof(float) * scale_list_.size_);

        setWeightPtr();
    }

    BertLayerINT8Weight& operator=(const BertLayerINT8Weight& other)
    {

        // to be confirmed， free buffer before =
        /*
        if (is_maintain_buffer) {
            for(int i = 0; i < 16; i++) {
                deviceFree(weights_ptr[i]);
            }
            deviceFree(scale_list_.d_scale_list_);
            free(scale_list_.h_scale_list_);
        }
        */

        hidden_units_ = other.hidden_units_;
        inter_size_ = other.inter_size_;
        deviceMalloc(&weights_ptr[0], hidden_units_ * hidden_units_ * 3);
        cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_ * hidden_units_ * 3);
        deviceMalloc(&weights_ptr[1], hidden_units_ * 3);
        cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_ * 3);
        deviceMalloc(&weights_ptr[2], hidden_units_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ * hidden_units_);
        deviceMalloc(&weights_ptr[3], hidden_units_);
        cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], hidden_units_);
        deviceMalloc(&weights_ptr[4], hidden_units_);
        cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_);
        deviceMalloc(&weights_ptr[5], hidden_units_);
        cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
        deviceMalloc(&weights_ptr[6], hidden_units_ * inter_size_);
        cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_ * inter_size_);
        deviceMalloc(&weights_ptr[7], inter_size_);
        cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], inter_size_);
        deviceMalloc(&weights_ptr[8], inter_size_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], inter_size_ * hidden_units_);
        deviceMalloc(&weights_ptr[9], hidden_units_);
        cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], hidden_units_);
        deviceMalloc(&weights_ptr[10], hidden_units_);
        cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], hidden_units_);
        deviceMalloc(&weights_ptr[11], hidden_units_);
        cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], hidden_units_);

        scale_list_.size_ = other.scale_list_.size_;
        scale_list_.p3_offset_ = other.scale_list_.p3_offset_;
        scale_list_.p4_offset_ = other.scale_list_.p4_offset_;
        deviceMalloc(&scale_list_ptr[0], scale_list_.size_);
        cudaD2Dcpy(scale_list_ptr[0], other.scale_list_ptr[0], scale_list_.size_);
        scale_list_ptr[1] = (float*)malloc(sizeof(float) * scale_list_.size_);
        memcpy(scale_list_ptr[1], other.scale_list_ptr[1], sizeof(float) * scale_list_.size_);

        setWeightPtr();
    }

    LayerNormWeight<T> attn_layernorm_weights;
    LayerNormWeight<T> ffn_layernorm_weights;
    AttentionINT8Weight<T> attention_weights;
    FfnINT8Weight<T> ffn_weights;
    ScaleList scale_list_;

private:
    void setWeightPtr()
    {
        attention_weights.query_weight.kernel = weights_ptr[0];
        attention_weights.query_weight.bias = weights_ptr[1];
        attention_weights.key_weight.kernel = weights_ptr[0] + hidden_units_ * hidden_units_;
        attention_weights.key_weight.bias = weights_ptr[1] + hidden_units_;
        attention_weights.value_weight.kernel = weights_ptr[0] + hidden_units_ * hidden_units_ * 2;
        attention_weights.value_weight.bias = weights_ptr[1] + hidden_units_ * 2;
        attention_weights.attention_output_weight.kernel = weights_ptr[2];
        attention_weights.attention_output_weight.bias = weights_ptr[3];
        attn_layernorm_weights.gamma = weights_ptr[4];
        attn_layernorm_weights.beta = weights_ptr[5];
        ffn_weights.intermediate_weight.kernel = weights_ptr[6];
        ffn_weights.intermediate_weight.bias = weights_ptr[7];
        ffn_weights.output_weight.kernel = weights_ptr[8];
        ffn_weights.output_weight.bias = weights_ptr[9];
        ffn_layernorm_weights.gamma = weights_ptr[10];
        ffn_layernorm_weights.beta = weights_ptr[11];

        scale_list_.d_scale_list_ = scale_list_ptr[0];
        scale_list_.h_scale_list_ = scale_list_ptr[1];
        attention_weights.scale_list_ptr = &scale_list_;
        ffn_weights.scale_list_ptr = &scale_list_;

        is_maintain_buffer = true;
    }

    int hidden_units_;
    int inter_size_;
    bool is_maintain_buffer = false;
    T* weights_ptr[12];
    float* scale_list_ptr[2];
    T* sp_weights_ptr[6];
    bool is_maintain_sp_buffer = false;
};

}  // namespace fastertransformer
