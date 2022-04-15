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
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
struct BertLayerWeight {

    BertLayerWeight() = default;
    BertLayerWeight(const int hidden_units, const int inter_size): hidden_units_(hidden_units), inter_size_(inter_size)
    {
        deviceMalloc(&weights_ptr[0], hidden_units_ * hidden_units_);
        deviceMalloc(&weights_ptr[1], hidden_units_);
        deviceMalloc(&weights_ptr[2], hidden_units_ * hidden_units_);
        deviceMalloc(&weights_ptr[3], hidden_units_);
        deviceMalloc(&weights_ptr[4], hidden_units_ * hidden_units_);
        deviceMalloc(&weights_ptr[5], hidden_units_);
        deviceMalloc(&weights_ptr[6], hidden_units_ * hidden_units_);
        deviceMalloc(&weights_ptr[7], hidden_units_);
        deviceMalloc(&weights_ptr[8], hidden_units_);
        deviceMalloc(&weights_ptr[9], hidden_units_);
        deviceMalloc(&weights_ptr[10], hidden_units_ * inter_size_);
        deviceMalloc(&weights_ptr[11], inter_size_);
        deviceMalloc(&weights_ptr[12], inter_size_ * hidden_units_);
        deviceMalloc(&weights_ptr[13], hidden_units_);
        deviceMalloc(&weights_ptr[14], hidden_units_);
        deviceMalloc(&weights_ptr[15], hidden_units_);

        setWeightPtr();
    }

    ~BertLayerWeight()
    {
        if (is_maintain_buffer == true) {
            for (int i = 0; i < 16; i++) {
                deviceFree(weights_ptr[i]);
            }

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

    BertLayerWeight(const BertLayerWeight& other): hidden_units_(other.hidden_units_), inter_size_(other.inter_size_)
    {
        deviceMalloc(&weights_ptr[0], hidden_units_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_ * hidden_units_);
        deviceMalloc(&weights_ptr[1], hidden_units_);
        cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
        deviceMalloc(&weights_ptr[2], hidden_units_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ * hidden_units_);
        deviceMalloc(&weights_ptr[3], hidden_units_);
        cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], hidden_units_);
        deviceMalloc(&weights_ptr[4], hidden_units_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ * hidden_units_);
        deviceMalloc(&weights_ptr[5], hidden_units_);
        cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
        deviceMalloc(&weights_ptr[6], hidden_units_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_ * hidden_units_);
        deviceMalloc(&weights_ptr[7], hidden_units_);
        cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], hidden_units_);
        deviceMalloc(&weights_ptr[8], hidden_units_);
        cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], hidden_units_);
        deviceMalloc(&weights_ptr[9], hidden_units_);
        cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], hidden_units_);
        deviceMalloc(&weights_ptr[10], hidden_units_ * inter_size_);
        cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], hidden_units_ * inter_size_);
        deviceMalloc(&weights_ptr[11], inter_size_);
        cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], inter_size_);
        deviceMalloc(&weights_ptr[12], inter_size_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[12], other.weights_ptr[12], inter_size_ * hidden_units_);
        deviceMalloc(&weights_ptr[13], hidden_units_);
        cudaD2Dcpy(weights_ptr[13], other.weights_ptr[13], hidden_units_);
        deviceMalloc(&weights_ptr[14], hidden_units_);
        cudaD2Dcpy(weights_ptr[14], other.weights_ptr[14], hidden_units_);
        deviceMalloc(&weights_ptr[15], hidden_units_);
        cudaD2Dcpy(weights_ptr[15], other.weights_ptr[15], hidden_units_);

        setWeightPtr();
    }

    BertLayerWeight& operator=(const BertLayerWeight& other)
    {
        hidden_units_ = other.hidden_units_;
        inter_size_ = other.inter_size_;
        deviceMalloc(&weights_ptr[0], hidden_units_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_ * hidden_units_);
        deviceMalloc(&weights_ptr[1], hidden_units_);
        cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
        deviceMalloc(&weights_ptr[2], hidden_units_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ * hidden_units_);
        deviceMalloc(&weights_ptr[3], hidden_units_);
        cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], hidden_units_);
        deviceMalloc(&weights_ptr[4], hidden_units_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ * hidden_units_);
        deviceMalloc(&weights_ptr[5], hidden_units_);
        cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
        deviceMalloc(&weights_ptr[6], hidden_units_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_ * hidden_units_);
        deviceMalloc(&weights_ptr[7], hidden_units_);
        cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], hidden_units_);
        deviceMalloc(&weights_ptr[8], hidden_units_);
        cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], hidden_units_);
        deviceMalloc(&weights_ptr[9], hidden_units_);
        cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], hidden_units_);
        deviceMalloc(&weights_ptr[10], hidden_units_ * inter_size_);
        cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], hidden_units_ * inter_size_);
        deviceMalloc(&weights_ptr[11], inter_size_);
        cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], inter_size_);
        deviceMalloc(&weights_ptr[12], inter_size_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[12], other.weights_ptr[12], inter_size_ * hidden_units_);
        deviceMalloc(&weights_ptr[13], hidden_units_);
        cudaD2Dcpy(weights_ptr[13], other.weights_ptr[13], hidden_units_);
        deviceMalloc(&weights_ptr[14], hidden_units_);
        cudaD2Dcpy(weights_ptr[14], other.weights_ptr[14], hidden_units_);
        deviceMalloc(&weights_ptr[15], hidden_units_);
        cudaD2Dcpy(weights_ptr[15], other.weights_ptr[15], hidden_units_);

        setWeightPtr();
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
        attention_weights.query_weight.sp_kernel = sp_weights_ptr[0];
        attention_weights.key_weight.sp_kernel = sp_weights_ptr[1];
        attention_weights.value_weight.sp_kernel = sp_weights_ptr[2];
        attention_weights.attention_output_weight.sp_kernel = sp_weights_ptr[3];
        ffn_weights.intermediate_weight.sp_kernel = sp_weights_ptr[4];
        ffn_weights.output_weight.sp_kernel = sp_weights_ptr[5];
        is_maintain_sp_buffer = true;
    }
#endif

    AttentionWeight<T> attention_weights;
    LayerNormWeight<T> attn_layernorm_weights;
    FfnWeight<T> ffn_weights;
    LayerNormWeight<T> ffn_layernorm_weights;

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

        is_maintain_buffer = true;
    }
    int hidden_units_;
    int inter_size_;
    bool is_maintain_buffer = false;
    T* weights_ptr[16];
    T* sp_weights_ptr[6];
    bool is_maintain_sp_buffer = false;
};

}  // namespace fastertransformer
