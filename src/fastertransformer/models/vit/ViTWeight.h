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

#include "src/fastertransformer/layers/DenseWeight.h"
#include "src/fastertransformer/models/vit/ViTLayerWeight.h"

namespace fastertransformer {

template<typename T>
struct ViTEmbeds {
    const T* class_embed;
    const T* position_embed;
};

#define WEIGHT_N 6
template<typename T>
struct ViTWeight {

    ViTWeight() = delete;
    ViTWeight(const int embed_dim,
              const int inter_size,
              const int num_layer,
              const int img_size,
              const int patch_size,
              const int chn_num,
              const bool with_cls_token,
              const bool hold_buffer = true):
        with_cls_token_(with_cls_token),
        embed_dim_(embed_dim),
        inter_size_(inter_size),
        num_layer_(num_layer),
        img_size_(img_size),
        patch_size_(patch_size),
        chn_num_(chn_num),
        seq_len_(img_size_ * img_size_ / (patch_size_ * patch_size_) + (with_cls_token_ ? 1 : 0))
    {
        weights_size[0] = chn_num_ * patch_size_ * patch_size_ * embed_dim_;
        weights_size[1] = embed_dim_;
        weights_size[2] = with_cls_token_ ? embed_dim_ : 0;
        weights_size[3] = embed_dim_ * seq_len_;
        weights_size[4] = embed_dim_;
        weights_size[5] = embed_dim_;

        if (hold_buffer) {
            for (int i = 0; i < WEIGHT_N; i++) {
                if (weights_size[i] == 0) {
                    continue;
                }

                deviceMalloc(&weights_ptr[i], weights_size[i]);
            }

            setWeightPtr();
        }

        for (int i = 0; i < num_layer_; i++) {
            vit_layer_weights.push_back(ViTLayerWeight<T>(embed_dim_, inter_size_, i, hold_buffer));
        }
    }

    ~ViTWeight()
    {
        if (is_maintain_buffer == true) {
            vit_layer_weights.clear();
            for (int i = 0; i < WEIGHT_N; i++) {
                if (weights_ptr[i] != nullptr) {
                    deviceFree(weights_ptr[i]);
                }
            }

            post_transformer_layernorm_weights.gamma = nullptr;
            post_transformer_layernorm_weights.beta = nullptr;
            pre_transform_embeds.class_embed = nullptr;
            pre_transform_embeds.position_embed = nullptr;
            pre_encoder_conv_weights.kernel = nullptr;
            pre_encoder_conv_weights.bias = nullptr;
            is_maintain_buffer = false;
        }
    }

    ViTWeight(const ViTWeight& other):
        with_cls_token_(other.with_cls_token_),
        embed_dim_(other.embed_dim_),
        inter_size_(other.inter_size_),
        num_layer_(other.num_layer_),
        img_size_(other.img_size_),
        patch_size_(other.patch_size_),
        chn_num_(other.chn_num_),
        seq_len_(other.seq_len_)
    {
        memcpy(weights_size, other.weights_size, sizeof(size_t) * WEIGHT_N);
        if (other.is_maintain_buffer) {
            for (int i = 0; i < WEIGHT_N; i++) {
                if (!is_maintain_buffer) {
                    deviceMalloc(&weights_ptr[i], weights_size[i]);
                }
                cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
            }
            setWeightPtr();
        }

        vit_layer_weights.clear();
        for (int i = 0; i < num_layer_; i++) {
            vit_layer_weights.push_back(other.vit_layer_weights[i]);
        }
    }

    ViTWeight& operator=(const ViTWeight& other)
    {
        embed_dim_ = other.embed_dim_;
        inter_size_ = other.inter_size_;
        num_layer_ = other.num_layer_;
        img_size_ = other.img_size_;
        patch_size_ = other.patch_size_;
        chn_num_ = other.chn_num_;
        seq_len_ = other.seq_len_;
        with_cls_token_ = other.with_cls_token_;
        memcpy(weights_size, other.weights_size, sizeof(size_t) * WEIGHT_N);

        if (other.is_maintain_buffer) {
            for (int i = 0; i < WEIGHT_N; i++) {
                if (!is_maintain_buffer) {
                    deviceMalloc(&weights_ptr[i], weights_size[i]);
                }
                cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
            }
            setWeightPtr();
        }

        vit_layer_weights.clear();
        for (int i = 0; i < num_layer_; i++) {
            vit_layer_weights.push_back(other.vit_layer_weights[i]);
        }

        return *this;
    }

    size_t GetSerializeSize()
    {
        size_t count;
        for (int i = 0; i < WEIGHT_N; i++) {
            count += weights_size[i];
        }
        count *= sizeof(T);

        for (auto& lw : vit_layer_weights) {
            count += lw.GetSerializeSize();
        }

        return count;
    }

    void serialize(void* buffer)
    {
        char* tmp_buf = (char*)buffer;
        for (int i = 0; i < WEIGHT_N; i++) {
            cudaMemcpy(tmp_buf, weights_ptr[i], sizeof(T) * weights_size[i], cudaMemcpyDeviceToHost);
            tmp_buf += sizeof(T) * weights_size[i];
        }

        for (auto& lw : vit_layer_weights) {
            lw.serialize(tmp_buf);
            tmp_buf += lw.GetSerializeSize();
        }
    }

    void deserialize(const void* buffer)
    {
        if (!is_maintain_buffer) {
            return;
        }

        char* tmp_buf = (char*)buffer;
        for (int i = 0; i < WEIGHT_N; i++) {
            cudaMemcpy(weights_ptr[i], tmp_buf, sizeof(T) * weights_size[i], cudaMemcpyHostToDevice);
            tmp_buf += sizeof(T) * weights_size[i];
        }

        for (auto& lw : vit_layer_weights) {
            lw.deserialize(tmp_buf);
            tmp_buf += lw.GetSerializeSize();
        }
    }

    void CopyWeightsFromHostBuffers(const T* const*& w)
    {
        cudaMemcpy(
            const_cast<T*>(pre_encoder_conv_weights.kernel), *w++, sizeof(T) * weights_size[0], cudaMemcpyHostToDevice);
        cudaMemcpy(
            const_cast<T*>(pre_encoder_conv_weights.bias), *w++, sizeof(T) * weights_size[1], cudaMemcpyHostToDevice);
        if (with_cls_token_) {
            cudaMemcpy(const_cast<T*>(pre_transform_embeds.class_embed),
                       *w++,
                       sizeof(T) * weights_size[2],
                       cudaMemcpyHostToDevice);
            with_cls_token_ = true;
        }
        cudaMemcpy(const_cast<T*>(pre_transform_embeds.position_embed),
                   *w++,
                   sizeof(T) * weights_size[3],
                   cudaMemcpyHostToDevice);

        for (int i = 0; i < num_layer_; i++) {
            auto& layer_weight = vit_layer_weights[i];
            layer_weight.CopyWeightsFromHostBuffers(w);
        }

        cudaMemcpy(const_cast<T*>(post_transformer_layernorm_weights.gamma),
                   *w++,
                   sizeof(T) * weights_size[4],
                   cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(post_transformer_layernorm_weights.beta),
                   *w++,
                   sizeof(T) * weights_size[5],
                   cudaMemcpyHostToDevice);
    }

    inline size_t GetWeightCount()
    {
        size_t weight_count = WEIGHT_N;
        weight_count += num_layer_ * vit_layer_weights[0].GetWeightCount();

        return weight_count;
    }

    void ExportWeights() const
    {
        DataType dtype = DataType::TYPE_INVALID;
        if (std::is_same<T, half>::value) {
            dtype = DataType::TYPE_FP16;
        }
        else if (std::is_same<T, float>::value) {
            dtype = DataType::TYPE_FP32;
        }

        Tensor conv_kernel{MEMORY_GPU,
                           dtype,
                           std::vector<size_t>{embed_dim_, chn_num_, patch_size_, patch_size_},
                           pre_encoder_conv_weights.kernel};  // OIHW
        conv_kernel.save<T>("./weights/conv_kernel.npy");
        Tensor conv_bias{MEMORY_GPU, dtype, std::vector<size_t>{embed_dim_}, pre_encoder_conv_weights.bias};  // OIHW
        conv_bias.save<T>("./weights/conv_bias.npy");
        Tensor cls_token{MEMORY_GPU, dtype, std::vector<size_t>{embed_dim_}, pre_transform_embeds.class_embed};
        if (with_cls_token_) {
            cls_token.save<T>("./weights/cls_token.npy");
        }
        Tensor pos_embed{
            MEMORY_GPU, dtype, std::vector<size_t>{1, seq_len_, embed_dim_}, pre_transform_embeds.position_embed};
        pos_embed.save<T>("./weights/pos_embed.npy");
        Tensor ln_gamma{MEMORY_GPU, dtype, std::vector<size_t>{embed_dim_}, post_transformer_layernorm_weights.gamma};
        ln_gamma.save<T>("./weights/enc_ln_scale.npy");
        Tensor ln_beta{MEMORY_GPU, dtype, std::vector<size_t>{embed_dim_}, post_transformer_layernorm_weights.beta};
        ln_beta.save<T>("./weights/enc_ln_bias.npy");

        for (int i = 0; i < num_layer_; i++) {
            vit_layer_weights[i].ExportWeights(i);
        }
    }

    std::vector<ViTLayerWeight<T>> vit_layer_weights;
    LayerNormWeight<T> post_transformer_layernorm_weights;
    ViTEmbeds<T> pre_transform_embeds;
    DenseWeight<T> pre_encoder_conv_weights;
    bool with_cls_token_;

private:
    void setWeightPtr()
    {
        pre_encoder_conv_weights.kernel = weights_ptr[0];
        pre_encoder_conv_weights.bias = weights_ptr[1];
        pre_transform_embeds.class_embed = weights_ptr[2];
        pre_transform_embeds.position_embed = weights_ptr[3];
        post_transformer_layernorm_weights.gamma = weights_ptr[4];
        post_transformer_layernorm_weights.beta = weights_ptr[5];

        is_maintain_buffer = true;
    }
    int embed_dim_;
    int inter_size_;
    int num_layer_;
    int img_size_;
    int patch_size_;
    int chn_num_;
    int seq_len_;
    bool is_maintain_buffer = false;
    T* weights_ptr[WEIGHT_N]{nullptr};
    size_t weights_size[WEIGHT_N];
};

#undef WEIGHT_N
}  // namespace fastertransformer
