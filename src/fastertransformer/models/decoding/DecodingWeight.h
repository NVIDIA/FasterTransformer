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
#include "src/fastertransformer/models/decoder/DecoderLayerWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
struct DecodingWeight {

    DecodingWeight() = default;
    DecodingWeight(const int hidden_units,
                   const int inter_size,
                   const int vocab_size,
                   const int num_layer,
                   const int max_seq_len,
                   const int mem_hidden_units):
        hidden_units_(hidden_units),
        inter_size_(inter_size),
        vocab_size_(vocab_size),
        num_layer_(num_layer),
        max_seq_len_(max_seq_len),
        mem_hidden_units_(mem_hidden_units)
    {
        for (int l = 0; l < num_layer_; l++) {
            decoder_layer_weights.push_back(DecoderLayerWeight<T>(hidden_units_, inter_size_, mem_hidden_units_));
        }

        mallocWeights();
        setWeightPtr();
    }

    ~DecodingWeight()
    {
        if (is_maintain_buffer == true) {
            decoder_layer_weights.clear();
            for (int i = 0; i < 6; i++) {
                deviceFree(weights_ptr[i]);
            }

            position_encoding_table = nullptr;
            pre_decoder_embedding_table = nullptr;
            post_decoder_layernorm.beta = nullptr;
            post_decoder_layernorm.gamma = nullptr;
            post_decoder_embedding.kernel = nullptr;
            post_decoder_embedding.bias = nullptr;
            is_maintain_buffer = false;
        }
    }

    DecodingWeight(const DecodingWeight& other):
        hidden_units_(other.hidden_units_),
        inter_size_(other.inter_size_),
        num_layer_(other.num_layer_),
        vocab_size_(other.vocab_size_),
        max_seq_len_(other.max_seq_len_),
        mem_hidden_units_(other.mem_hidden_units_)
    {
        mallocWeights();
        cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], max_seq_len_ * vocab_size_);
        cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], vocab_size_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_);
        cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], hidden_units_);
        cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ * vocab_size_);
        cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], vocab_size_);
        setWeightPtr();

        decoder_layer_weights.clear();
        for (int l = 0; l < num_layer_; l++) {
            decoder_layer_weights.push_back(other.decoder_layer_weights[l]);
        }
    }

    DecodingWeight& operator=(const DecodingWeight& other)
    {
        hidden_units_ = other.hidden_units_;
        inter_size_ = other.inter_size_;
        num_layer_ = other.num_layer_;
        vocab_size_ = other.vocab_size_;
        max_seq_len_ = other.max_seq_len_;
        mem_hidden_units_ = other.mem_hidden_units_;

        mallocWeights();
        cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], max_seq_len_ * vocab_size_);
        cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], vocab_size_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_);
        cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], hidden_units_);
        cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ * vocab_size_);
        cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], vocab_size_);
        setWeightPtr();

        decoder_layer_weights.clear();
        for (int l = 0; l < num_layer_; l++) {
            decoder_layer_weights.push_back(other.decoder_layer_weights[l]);
        }
        return *this;
    }

    void mallocWeights()
    {
        deviceMalloc(&weights_ptr[0], max_seq_len_ * vocab_size_);
        deviceMalloc(&weights_ptr[1], vocab_size_ * hidden_units_);
        deviceMalloc(&weights_ptr[2], hidden_units_);
        deviceMalloc(&weights_ptr[3], hidden_units_);
        deviceMalloc(&weights_ptr[4], hidden_units_ * vocab_size_);
        deviceMalloc(&weights_ptr[5], vocab_size_);
        is_maintain_buffer = true;
    }

    std::vector<DecoderLayerWeight<T>> decoder_layer_weights;
    const T* position_encoding_table = nullptr;
    const T* pre_decoder_embedding_table = nullptr;
    LayerNormWeight<T> post_decoder_layernorm;
    DenseWeight<T> post_decoder_embedding;

private:
    void setWeightPtr()
    {
        position_encoding_table = weights_ptr[0];
        pre_decoder_embedding_table = weights_ptr[1];
        post_decoder_layernorm.beta = weights_ptr[2];
        post_decoder_layernorm.gamma = weights_ptr[3];
        post_decoder_embedding.kernel = weights_ptr[4];
        post_decoder_embedding.bias = weights_ptr[5];
    }

    int hidden_units_;
    int inter_size_;
    int vocab_size_;
    int num_layer_;
    int max_seq_len_;
    int mem_hidden_units_;
    bool is_maintain_buffer = false;
    T* weights_ptr[6];
};

}  // namespace fastertransformer
