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
#include "src/fastertransformer/layers/FfnINT8Weight.h"
#include "src/fastertransformer/layers/attention_layers_int8/AttentionINT8Weight.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
struct ViTLayerINT8Weight {

    ViTLayerINT8Weight() = default;
    ViTLayerINT8Weight(const int embed_dim, const int inter_size): embed_dim_(embed_dim), inter_size_(inter_size)
    {
        deviceMalloc(&weights_ptr[0], embed_dim_ * embed_dim_);
        deviceMalloc(&weights_ptr[1], embed_dim_);
        deviceMalloc(&weights_ptr[2], embed_dim_ * embed_dim_);
        deviceMalloc(&weights_ptr[3], embed_dim_);
        deviceMalloc(&weights_ptr[4], embed_dim_ * embed_dim_);
        deviceMalloc(&weights_ptr[5], embed_dim_);
        deviceMalloc(&weights_ptr[6], embed_dim_ * embed_dim_);
        deviceMalloc(&weights_ptr[7], embed_dim_);
        deviceMalloc(&weights_ptr[8], embed_dim_);
        deviceMalloc(&weights_ptr[9], embed_dim_);
        deviceMalloc(&weights_ptr[10], embed_dim_ * inter_size_);
        deviceMalloc(&weights_ptr[11], inter_size_);
        deviceMalloc(&weights_ptr[12], inter_size_ * embed_dim_);
        deviceMalloc(&weights_ptr[13], embed_dim_);
        deviceMalloc(&weights_ptr[14], embed_dim_);
        deviceMalloc(&weights_ptr[15], embed_dim_);

        scale_list_.size_ = ACTIVATION_AMAX_NUM + 9 * embed_dim + INT8O_GEMM_NUM + TRT_AMAX_NUM + SCALE_RESERVE_NUM;
        scale_list_.p3_offset_ = ACTIVATION_AMAX_NUM + 9 * embed_dim;
        scale_list_.p4_offset_ = ACTIVATION_AMAX_NUM + 9 * embed_dim + INT8O_GEMM_NUM;
        deviceMalloc(&scale_list_ptr[0], scale_list_.size_);
        scale_list_ptr[1] = (float*)malloc(sizeof(float) * scale_list_.size_);

        setWeightPtr();
    }

    ~ViTLayerINT8Weight()
    {
        if (is_maintain_buffer == true) {
            for (int i = 0; i < 16; i++) {
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

    ViTLayerINT8Weight(const ViTLayerINT8Weight& other): embed_dim_(other.embed_dim_), inter_size_(other.inter_size_)
    {
        deviceMalloc(&weights_ptr[0], embed_dim_ * embed_dim_);
        cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], embed_dim_ * embed_dim_);
        deviceMalloc(&weights_ptr[1], embed_dim_);
        cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], embed_dim_);
        deviceMalloc(&weights_ptr[2], embed_dim_ * embed_dim_);
        cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], embed_dim_ * embed_dim_);
        deviceMalloc(&weights_ptr[3], embed_dim_);
        cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], embed_dim_);
        deviceMalloc(&weights_ptr[4], embed_dim_ * embed_dim_);
        cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], embed_dim_ * embed_dim_);
        deviceMalloc(&weights_ptr[5], embed_dim_);
        cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], embed_dim_);
        deviceMalloc(&weights_ptr[6], embed_dim_ * embed_dim_);
        cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], embed_dim_ * embed_dim_);
        deviceMalloc(&weights_ptr[7], embed_dim_);
        cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], embed_dim_);
        deviceMalloc(&weights_ptr[8], embed_dim_);
        cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], embed_dim_);
        deviceMalloc(&weights_ptr[9], embed_dim_);
        cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], embed_dim_);
        deviceMalloc(&weights_ptr[10], embed_dim_ * inter_size_);
        cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], embed_dim_ * inter_size_);
        deviceMalloc(&weights_ptr[11], inter_size_);
        cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], inter_size_);
        deviceMalloc(&weights_ptr[12], inter_size_ * embed_dim_);
        cudaD2Dcpy(weights_ptr[12], other.weights_ptr[12], inter_size_ * embed_dim_);
        deviceMalloc(&weights_ptr[13], embed_dim_);
        cudaD2Dcpy(weights_ptr[13], other.weights_ptr[13], embed_dim_);
        deviceMalloc(&weights_ptr[14], embed_dim_);
        cudaD2Dcpy(weights_ptr[14], other.weights_ptr[14], embed_dim_);
        deviceMalloc(&weights_ptr[15], embed_dim_);
        cudaD2Dcpy(weights_ptr[15], other.weights_ptr[15], embed_dim_);

        scale_list_.size_ = other.scale_list_.size_;
        scale_list_.p3_offset_ = other.scale_list_.p3_offset_;
        scale_list_.p4_offset_ = other.scale_list_.p4_offset_;
        deviceMalloc(&scale_list_ptr[0], scale_list_.size_);
        cudaD2Dcpy(scale_list_ptr[0], other.scale_list_ptr[0], scale_list_.size_);
        scale_list_ptr[1] = (float*)malloc(sizeof(float) * scale_list_.size_);
        memcpy(scale_list_ptr[1], other.scale_list_ptr[1], sizeof(float) * scale_list_.size_);

        setWeightPtr();
    }

    ViTLayerINT8Weight& operator=(const ViTLayerINT8Weight& other)
    {
        embed_dim_ = other.embed_dim_;
        inter_size_ = other.inter_size_;
        deviceMalloc(&weights_ptr[0], embed_dim_ * embed_dim_);
        cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], embed_dim_ * embed_dim_);
        deviceMalloc(&weights_ptr[1], embed_dim_);
        cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], embed_dim_);
        deviceMalloc(&weights_ptr[2], embed_dim_ * embed_dim_);
        cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], embed_dim_ * embed_dim_);
        deviceMalloc(&weights_ptr[3], embed_dim_);
        cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], embed_dim_);
        deviceMalloc(&weights_ptr[4], embed_dim_ * embed_dim_);
        cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], embed_dim_ * embed_dim_);
        deviceMalloc(&weights_ptr[5], embed_dim_);
        cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], embed_dim_);
        deviceMalloc(&weights_ptr[6], embed_dim_ * embed_dim_);
        cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], embed_dim_ * embed_dim_);
        deviceMalloc(&weights_ptr[7], embed_dim_);
        cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], embed_dim_);
        deviceMalloc(&weights_ptr[8], embed_dim_);
        cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], embed_dim_);
        deviceMalloc(&weights_ptr[9], embed_dim_);
        cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], embed_dim_);
        deviceMalloc(&weights_ptr[10], embed_dim_ * inter_size_);
        cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], embed_dim_ * inter_size_);
        deviceMalloc(&weights_ptr[11], inter_size_);
        cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], inter_size_);
        deviceMalloc(&weights_ptr[12], inter_size_ * embed_dim_);
        cudaD2Dcpy(weights_ptr[12], other.weights_ptr[12], inter_size_ * embed_dim_);
        deviceMalloc(&weights_ptr[13], embed_dim_);
        cudaD2Dcpy(weights_ptr[13], other.weights_ptr[13], embed_dim_);
        deviceMalloc(&weights_ptr[14], embed_dim_);
        cudaD2Dcpy(weights_ptr[14], other.weights_ptr[14], embed_dim_);
        deviceMalloc(&weights_ptr[15], embed_dim_);
        cudaD2Dcpy(weights_ptr[15], other.weights_ptr[15], embed_dim_);

        scale_list_.size_ = other.scale_list_.size_;
        scale_list_.p3_offset_ = other.scale_list_.p3_offset_;
        scale_list_.p4_offset_ = other.scale_list_.p4_offset_;
        deviceMalloc(&scale_list_ptr[0], scale_list_.size_);
        cudaD2Dcpy(scale_list_ptr[0], other.scale_list_ptr[0], scale_list_.size_);
        scale_list_ptr[1] = (float*)malloc(sizeof(float) * scale_list_.size_);
        memcpy(scale_list_ptr[1], other.scale_list_ptr[1], sizeof(float) * scale_list_.size_);

        setWeightPtr();
    }

    AttentionINT8Weight<T> attention_weights;
    LayerNormWeight<T> attn_layernorm_weights;
    FfnINT8Weight<T> ffn_weights;
    LayerNormWeight<T> ffn_layernorm_weights;
    ScaleList scale_list_;

private:
    void setWeightPtr()
    {
        attention_weights.query_weight.kernel = weights_ptr[0];
        attention_weights.query_weight.bias = weights_ptr[1];
        attention_weights.key_weight.kernel = weights_ptr[2];
        attention_weights.key_weight.bias = weights_ptr[3];
        attention_weights.value_weight.kernel = weights_ptr[4];
        attention_weights.value_weight.bias = weights_ptr[5];
        attention_weights.attention_output_weight.kernel = weights_ptr[6];
        attention_weights.attention_output_weight.bias = weights_ptr[7];
        attn_layernorm_weights.gamma = weights_ptr[8];
        attn_layernorm_weights.beta = weights_ptr[9];
        ffn_weights.intermediate_weight.kernel = weights_ptr[10];
        ffn_weights.intermediate_weight.bias = weights_ptr[11];
        ffn_weights.output_weight.kernel = weights_ptr[12];
        ffn_weights.output_weight.bias = weights_ptr[13];
        ffn_layernorm_weights.gamma = weights_ptr[14];
        ffn_layernorm_weights.beta = weights_ptr[15];

        scale_list_.d_scale_list_ = scale_list_ptr[0];
        scale_list_.h_scale_list_ = scale_list_ptr[1];
        attention_weights.scale_list_ptr = &scale_list_;
        ffn_weights.scale_list_ptr = &scale_list_;

        is_maintain_buffer = true;
    }
    int embed_dim_;
    int inter_size_;
    bool is_maintain_buffer = false;
    T* weights_ptr[16];
    float* scale_list_ptr[2];
    T* sp_weights_ptr[6];
};

}  // namespace fastertransformer
