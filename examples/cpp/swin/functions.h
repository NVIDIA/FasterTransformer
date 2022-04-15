/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include <vector>

namespace fastertransformer {

static int getWeightNum(const int layer_num, const int* depths)
{
    // We arrange weights layer by layer and block by block inside each layer;
    // each block has 13 weights
    // each layer has a block list && 4 weights
    // each swin transformer has a layer list && 6 weights && 3 handles

    int weight_num = 6;
    for (int l = 0; l < layer_num; l++) {
        for (int di = 0; di < depths[l]; di++) {
            weight_num += 13;
        }
        weight_num += 4;
    }
    return weight_num;
}

static void generateWeightSize(std::vector<size_t>& weight_size,
                               const int layer_num,
                               const int embed_dim,
                               const float mlp_ratio,
                               const int window_size,
                               const int img_size,
                               const int patch_size,
                               const int in_chans,
                               const int* depths,
                               const int* num_heads)
{
    size_t l_pow2 = 1;
    for (int l = 0; l < layer_num; l++) {
        for (int di = 0; di < depths[l]; di++) {
            size_t attention_qkv_kernel_size = (l_pow2 * embed_dim) * (l_pow2 * embed_dim * 3);
            weight_size.push_back(attention_qkv_kernel_size);
            size_t attention_qkv_bias_size = l_pow2 * embed_dim * 3;
            weight_size.push_back(attention_qkv_bias_size);
            size_t attention_proj_kernel_size = (l_pow2 * embed_dim) * (l_pow2 * embed_dim);
            weight_size.push_back(attention_proj_kernel_size);
            size_t attention_proj_bias_size = l_pow2 * embed_dim;
            weight_size.push_back(attention_proj_bias_size);
            size_t mlp_linear_kernel_size = (mlp_ratio * l_pow2 * embed_dim) * (l_pow2 * embed_dim);
            weight_size.push_back(mlp_linear_kernel_size);
            size_t mlp_linear_bias_size = mlp_ratio * l_pow2 * embed_dim;
            weight_size.push_back(mlp_linear_bias_size);
            size_t mlp_linear2_kernel_size = (l_pow2 * embed_dim) * (mlp_ratio * l_pow2 * embed_dim);
            weight_size.push_back(mlp_linear2_kernel_size);
            size_t mlp_linear2_bias_size = l_pow2 * embed_dim;
            weight_size.push_back(mlp_linear2_bias_size);
            size_t block_norm_gamma_size = l_pow2 * embed_dim;
            weight_size.push_back(block_norm_gamma_size);
            size_t block_norm_beta_size = l_pow2 * embed_dim;
            weight_size.push_back(block_norm_beta_size);
            size_t block_norm2_gamma_size = l_pow2 * embed_dim;
            weight_size.push_back(block_norm2_gamma_size);
            size_t block_norm2_beta_size = l_pow2 * embed_dim;
            weight_size.push_back(block_norm2_beta_size);
            size_t attention_relative_pos_bias_size =
                num_heads[l] * window_size * window_size * window_size * window_size;
            weight_size.push_back(attention_relative_pos_bias_size);
        }
        size_t patchMerge_norm_gamma_size = (l != layer_num - 1) ? 4 * l_pow2 * embed_dim : 0;
        weight_size.push_back(patchMerge_norm_gamma_size);
        size_t patchMerge_norm_beta_size = (l != layer_num - 1) ? 4 * l_pow2 * embed_dim : 0;
        weight_size.push_back(patchMerge_norm_beta_size);
        size_t patchMerge_linear_kernel_size =
            (l != layer_num - 1) ? (4 * l_pow2 * embed_dim) * (2 * l_pow2 * embed_dim) : 0;
        weight_size.push_back(patchMerge_linear_kernel_size);
        int window_num = img_size / patch_size / window_size / l_pow2;
        window_num *= window_num;
        size_t attn_mask_size =
            (window_num > 1) ? window_num * (window_size * window_size) * (window_size * window_size) : 0;
        weight_size.push_back(attn_mask_size);
        l_pow2 *= 2;
    }
    size_t patchEmbed_proj_kernel_size = embed_dim * in_chans * patch_size * patch_size;
    weight_size.push_back(patchEmbed_proj_kernel_size);
    size_t patchEmbed_proj_bias_size = embed_dim;
    weight_size.push_back(patchEmbed_proj_bias_size);
    size_t patchEmbed_norm_gamma_size = embed_dim;
    weight_size.push_back(patchEmbed_norm_gamma_size);
    size_t patchEmbed_norm_beta_size = embed_dim;
    weight_size.push_back(patchEmbed_norm_beta_size);
    size_t norm_gamma_size = pow(2, layer_num - 1) * embed_dim;
    weight_size.push_back(norm_gamma_size);
    size_t norm_beta_size = norm_gamma_size;
    weight_size.push_back(norm_beta_size);
}

}  // namespace fastertransformer
