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
#include "src/fastertransformer/models/vit_int8/ViTLayerINT8Weight.h"

namespace fastertransformer {

template<typename T>
struct ViTEmbeds {
    const T* class_embed;
    const T* position_embed;
};

template<typename T>
struct ViTINT8Weight {

    ViTINT8Weight() = default;
    ViTINT8Weight(const int embed_dim,
                  const int inter_size,
                  const int num_layer,
                  const int img_size,
                  const int patch_size,
                  const int chn_num,
                  const bool with_cls_token):
        with_cls_token_(with_cls_token),
        embed_dim_(embed_dim),
        inter_size_(inter_size),
        num_layer_(num_layer),
        img_size_(img_size),
        patch_size_(patch_size),
        chn_num_(chn_num)
    {
        deviceMalloc(&weights_ptr[0], embed_dim_);
        deviceMalloc(&weights_ptr[1], embed_dim_);
        if (with_cls_token) {
            deviceMalloc(&weights_ptr[2], embed_dim_);  // pre_transform_embeds.class_embed
        }
        deviceMalloc(&weights_ptr[3],
                     embed_dim_
                         * (img_size_ * img_size_ / (patch_size_ * patch_size_)
                            + (with_cls_token ? 1 : 0)));  // pre_transform_embeds.position_embed
        deviceMalloc(&weights_ptr[4],
                     chn_num_ * patch_size_ * patch_size_ * embed_dim_);  // pre_encoder_conv_weights.kernel
        deviceMalloc(&weights_ptr[5], embed_dim_);                        // pre_encoder_conv_weights.bias

        setWeightPtr();
        for (int i = 0; i < num_layer_; i++) {
            vit_layer_weights.push_back(ViTLayerINT8Weight<T>(embed_dim_, inter_size_));
        }
    }

    ~ViTINT8Weight()
    {
        if (is_maintain_buffer == true) {
            vit_layer_weights.clear();
            for (int i = 0; i < 6; i++) {
                deviceFree(weights_ptr[i]);
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

    ViTINT8Weight(const ViTINT8Weight& other):
        with_cls_token_(other.with_cls_token_),
        embed_dim_(other.embed_dim_),
        inter_size_(other.inter_size_),
        num_layer_(other.num_layer_),
        img_size_(other.img_size_),
        patch_size_(other.patch_size_),
        chn_num_(other.chn_num_),
        cls_num_(other.cls_num_)
    {
        vit_layer_weights.clear();
        for (int i = 0; i < num_layer_; i++) {
            vit_layer_weights.push_back(other.vit_layer_weights[i]);
        }
        deviceMalloc(&weights_ptr[0], embed_dim_);
        cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], embed_dim_);
        deviceMalloc(&weights_ptr[1], embed_dim_);
        cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], embed_dim_);
        if (other.weights_ptr[2] != nullptr) {
            deviceMalloc(&weights_ptr[2], embed_dim_);  // pre_transform_embeds.class_embed
            cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], embed_dim_);
        }
        deviceMalloc(&weights_ptr[3],
                     embed_dim_
                         * (img_size_ * img_size_ / (patch_size_ * patch_size_)
                            + (with_cls_token_ ? 1 : 0)));  // pre_transform_embeds.position_embed
        cudaD2Dcpy(weights_ptr[3],
                   other.weights_ptr[3],
                   embed_dim_ * (img_size_ * img_size_ / (patch_size_ * patch_size_) + (with_cls_token_ ? 1 : 0)));
        deviceMalloc(&weights_ptr[4],
                     chn_num_ * patch_size_ * patch_size_ * embed_dim_);  // pre_encoder_conv_weights.kernel
        cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], chn_num_ * patch_size_ * patch_size_ * embed_dim_);
        deviceMalloc(&weights_ptr[5], embed_dim_);  // pre_encoder_conv_weights.bias
        cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], embed_dim_);

        setWeightPtr();
    }

    ViTINT8Weight& operator=(const ViTINT8Weight& other)
    {
        with_cls_token_ = other.with_cls_token_;
        embed_dim_ = other.embed_dim_;
        inter_size_ = other.inter_size_;
        num_layer_ = other.num_layer_;
        img_size_ = other.img_size_;
        patch_size_ = other.patch_size_;
        chn_num_ = other.chn_num_;
        cls_num_ = other.cls_num_;
        vit_layer_weights.clear();
        for (int i = 0; i < num_layer_; i++) {
            vit_layer_weights.push_back(other.vit_layer_weights[i]);
        }
        deviceMalloc(&weights_ptr[0], embed_dim_);
        cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], embed_dim_);
        deviceMalloc(&weights_ptr[1], embed_dim_);
        cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], embed_dim_);
        if (other.weights_ptr[2] != nullptr) {
            deviceMalloc(&weights_ptr[2], embed_dim_);  // pre_transform_embeds.class_embed
            cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], embed_dim_);
        }
        deviceMalloc(&weights_ptr[3],
                     embed_dim_
                         * (img_size_ * img_size_ / (patch_size_ * patch_size_)
                            + (with_cls_token_ ? 1 : 0)));  // pre_transform_embeds.position_embed
        cudaD2Dcpy(weights_ptr[3],
                   other.weights_ptr[3],
                   embed_dim_ * (img_size_ * img_size_ / (patch_size_ * patch_size_) + (with_cls_token_ ? 1 : 0)));
        deviceMalloc(&weights_ptr[4],
                     chn_num_ * patch_size_ * patch_size_ * embed_dim_);  // pre_encoder_conv_weights.kernel
        cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], chn_num_ * patch_size_ * patch_size_ * embed_dim_);
        deviceMalloc(&weights_ptr[5], embed_dim_);  // pre_encoder_conv_weights.bias
        cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], embed_dim_);

        setWeightPtr();
    }

    std::vector<ViTLayerINT8Weight<T>> vit_layer_weights;
    LayerNormWeight<T> post_transformer_layernorm_weights;
    ViTEmbeds<T> pre_transform_embeds;
    DenseWeight<T> pre_encoder_conv_weights;
    bool with_cls_token_;

private:
    void setWeightPtr()
    {
        post_transformer_layernorm_weights.gamma = weights_ptr[0];
        post_transformer_layernorm_weights.beta = weights_ptr[1];
        pre_transform_embeds.class_embed = weights_ptr[2];
        pre_transform_embeds.position_embed = weights_ptr[3];
        pre_encoder_conv_weights.kernel = weights_ptr[4];
        pre_encoder_conv_weights.bias = weights_ptr[5];

        is_maintain_buffer = true;
    }
    int embed_dim_;
    int inter_size_;
    int num_layer_;
    int img_size_;
    int patch_size_;
    int chn_num_;
    int cls_num_;
    bool is_maintain_buffer = false;
    T* weights_ptr[8]{nullptr};
};

}  // namespace fastertransformer
