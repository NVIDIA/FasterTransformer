/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "swinTransformerINT8Plugin.h"
#include "NvInfer.h"
#include "serialize.hpp"
#include <cuda.h>
#include <math.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <tuple>
#include <vector>

using namespace nvinfer1;
using namespace std;

namespace fastertransformer {

// Static class fields initialization
PluginFieldCollection SwinTransformerINT8PluginCreator::mFC{};
std::vector<PluginField> SwinTransformerINT8PluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(SwinTransformerINT8PluginCreator);

template<typename T>
SwinTransformerINT8Plugin<T>::SwinTransformerINT8Plugin(const std::string& name,
                                                        const int int8_mode,
                                                        const int max_batch_size,
                                                        const int img_size,
                                                        const int patch_size,
                                                        const int in_chans,
                                                        const int embed_dim,
                                                        const int window_size,
                                                        int* depths,
                                                        int* num_heads,
                                                        const bool ape,
                                                        const bool patch_norm,
                                                        const int layer_num,
                                                        const float mlp_ratio,
                                                        const bool qkv_bias,
                                                        const float qk_scale,
                                                        const std::vector<T*>& w,
                                                        const std::vector<float*>& d_amax,
                                                        const std::vector<float*>& h_amax):
    int8_mode_(int8_mode),
    layer_name_(name),
    max_batch_size_(max_batch_size),
    img_size_(img_size),
    patch_size_(patch_size),
    in_chans_(in_chans),
    embed_dim_(embed_dim),
    window_size_(window_size),
    ape_(ape),
    patch_norm_(patch_norm),
    layer_num_(layer_num),
    mlp_ratio_(mlp_ratio),
    qkv_bias_(qkv_bias),
    qk_scale_(qk_scale)
{
    check_cuda_error(cublasCreate(&cublas_handle_));
    check_cuda_error(cublasLtCreate(&cublaslt_handle_));
    checkCUDNN(cudnnCreate(&cudnn_handle_));
    sm_ = getSMVersion();
    bool _use_ORDER_COL32_2R_4R4 = false;
#if (CUDART_VERSION >= 11000)
    if (sm_ >= 80) {
        _use_ORDER_COL32_2R_4R4 = true;
    }
#endif

    weight_num_ = getWeightNum(layer_num, depths);
    if (weight_num_ != w.size()) {
        printf("[ERROR][SwinTransformerINT8Plugin](T) weights number %lu does not match expected number %d!\n",
               w.size(),
               weight_num_);
        exit(-1);
    }

    depths_ = (int*)malloc(layer_num * sizeof(int));
    num_heads_ = (int*)malloc(layer_num * sizeof(int));
    memcpy(depths_, depths, layer_num * sizeof(int));
    memcpy(num_heads_, num_heads, layer_num * sizeof(int));

    output_dim_ = int(pow(2, layer_num - 1)) * embed_dim;

    // calculate the size of each weight
    generateWeightSize(
        weight_size_, layer_num, embed_dim, mlp_ratio, window_size, img_size, patch_size, in_chans, depths, num_heads);

    int weight_idx = 0;
    int amax_idx = 0;
    int hidden_dim = embed_dim;
    for (int l = 0; l < layer_num; l++) {
        SwinTransformerINT8BasicLayerWeight<T> bl;
        for (int di = 0; di < depths[l]; di++) {
            SwinTransformerINT8BlockWeight<T> p;
            p.attention_weights.query_weight.kernel =
                cudaMallocAndCopy(weights_, w[weight_idx], weight_size_, weight_idx);
            weight_idx++;
            p.attention_weights.query_weight.bias =
                cudaMallocAndCopy(weights_, w[weight_idx], weight_size_, weight_idx);
            weight_idx++;
            p.attention_weights.attention_output_weight.kernel =
                cudaMallocAndCopy(weights_, w[weight_idx], weight_size_, weight_idx);
            weight_idx++;
            p.attention_weights.attention_output_weight.bias =
                cudaMallocAndCopy(weights_, w[weight_idx], weight_size_, weight_idx);
            weight_idx++;
            p.ffn_weights.intermediate_weight.kernel =
                cudaMallocAndCopy(weights_, w[weight_idx], weight_size_, weight_idx);
            weight_idx++;
            p.ffn_weights.intermediate_weight.bias =
                cudaMallocAndCopy(weights_, w[weight_idx], weight_size_, weight_idx);
            weight_idx++;
            p.ffn_weights.output_weight.kernel = cudaMallocAndCopy(weights_, w[weight_idx], weight_size_, weight_idx);
            weight_idx++;
            p.ffn_weights.output_weight.bias = cudaMallocAndCopy(weights_, w[weight_idx], weight_size_, weight_idx);
            weight_idx++;
            p.attn_layernorm_weights.gamma = cudaMallocAndCopy(weights_, w[weight_idx], weight_size_, weight_idx);
            weight_idx++;
            p.attn_layernorm_weights.beta = cudaMallocAndCopy(weights_, w[weight_idx], weight_size_, weight_idx);
            weight_idx++;
            p.ffn_layernorm_weights.gamma = cudaMallocAndCopy(weights_, w[weight_idx], weight_size_, weight_idx);
            weight_idx++;
            p.ffn_layernorm_weights.beta = cudaMallocAndCopy(weights_, w[weight_idx], weight_size_, weight_idx);
            weight_idx++;
            p.scalelist.size_ = ACTIVATION_AMAX_NUM + 5 + INT8O_GEMM_NUM + TRT_AMAX_NUM;
            p.scalelist.p2_offset_ = ACTIVATION_AMAX_NUM;
            p.scalelist.p3_offset_ = ACTIVATION_AMAX_NUM + 5;
            p.scalelist.p4_offset_ = ACTIVATION_AMAX_NUM + 5 + INT8O_GEMM_NUM;
            p.scalelist.d_scale_list_ = d_amaxCopy(d_amaxlist_, d_amax[amax_idx], 96, amax_idx);
            p.scalelist.h_scale_list_ = h_amaxCopy(h_amaxlist_, h_amax[amax_idx], 96, amax_idx);
            amax_idx++;
            p.attention_relative_pos_bias = cudaMallocAndCopy(weights_, w[weight_idx], weight_size_, weight_idx);
            weight_idx++;
            bl.block_weight_list.push_back(p);
        }
        bl.merge_layernorm_weights.gamma = cudaMallocAndCopy(weights_, w[weight_idx], weight_size_, weight_idx);
        weight_idx++;
        bl.merge_layernorm_weights.beta = cudaMallocAndCopy(weights_, w[weight_idx], weight_size_, weight_idx);
        weight_idx++;
        bl.merge_linear_weights.kernel = cudaMallocAndCopy(weights_, w[weight_idx], weight_size_, weight_idx);
        weight_idx++;
        bl.attn_mask = cudaMallocAndCopy(weights_, w[weight_idx], weight_size_, weight_idx);
        weight_idx++;
        // if(l != layer_num - 1){
        //   bl.quantize_weights(hidden_dim, _use_ORDER_COL32_2R_4R4, nullptr);
        // }
        params_.basic_layer_weight_list.push_back(bl);
        hidden_dim *= 2;
    }
    params_.patchEmbed_linear_weights.kernel = cudaMallocAndCopy(weights_, w[weight_idx], weight_size_, weight_idx);
    weight_idx++;
    params_.patchEmbed_linear_weights.bias = cudaMallocAndCopy(weights_, w[weight_idx], weight_size_, weight_idx);
    weight_idx++;
    params_.patchEmbed_norm_weights.gamma = cudaMallocAndCopy(weights_, w[weight_idx], weight_size_, weight_idx);
    weight_idx++;
    params_.patchEmbed_norm_weights.beta = cudaMallocAndCopy(weights_, w[weight_idx], weight_size_, weight_idx);
    weight_idx++;
    params_.norm_weights.gamma = cudaMallocAndCopy(weights_, w[weight_idx], weight_size_, weight_idx);
    weight_idx++;
    params_.norm_weights.beta = cudaMallocAndCopy(weights_, w[weight_idx], weight_size_, weight_idx);
    weight_idx++;

    cublasAlgoMap_ = new cublasAlgoMap("igemm.config", "");
    cublasWrapperMutex_ = new std::mutex();
    allocator_ = new Allocator<AllocatorType::CUDA>(getDevice());

    cublasINT8MMWrapper* cublas_wrapper = new cublasINT8MMWrapper(
        cublas_handle_, cublaslt_handle_, nullptr, cublasAlgoMap_, cublasWrapperMutex_, _use_ORDER_COL32_2R_4R4);
    if (std::is_same<T, half>::value) {
        cublas_wrapper->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper->setFP32GemmConfig();
    }

    swin_transformer_ = new SwinTransformerINT8<T>(int8_mode,
                                                   max_batch_size,
                                                   img_size,
                                                   patch_size,
                                                   in_chans,
                                                   embed_dim,
                                                   window_size,
                                                   depths,
                                                   num_heads,
                                                   ape,
                                                   patch_norm,
                                                   layer_num,
                                                   mlp_ratio,
                                                   cudnn_handle_,
                                                   nullptr,
                                                   cublas_wrapper,
                                                   allocator_,
                                                   false,
                                                   qkv_bias,
                                                   qk_scale);
}

template<typename T>
SwinTransformerINT8Plugin<T>::SwinTransformerINT8Plugin(const std::string& name,
                                                        const int int8_mode,
                                                        const int max_batch_size,
                                                        const int img_size,
                                                        const int patch_size,
                                                        const int in_chans,
                                                        const int embed_dim,
                                                        const int window_size,
                                                        int* depths,
                                                        int* num_heads,
                                                        const bool ape,
                                                        const bool patch_norm,
                                                        const int layer_num,
                                                        const float mlp_ratio,
                                                        const bool qkv_bias,
                                                        const float qk_scale,
                                                        const std::vector<Weights>& w,
                                                        const std::vector<Weights>& d_amax,
                                                        const std::vector<Weights>& h_amax):
    int8_mode_(int8_mode),
    layer_name_(name),
    max_batch_size_(max_batch_size),
    img_size_(img_size),
    patch_size_(patch_size),
    in_chans_(in_chans),
    embed_dim_(embed_dim),
    window_size_(window_size),
    ape_(ape),
    patch_norm_(patch_norm),
    layer_num_(layer_num),
    mlp_ratio_(mlp_ratio),
    qkv_bias_(qkv_bias),
    qk_scale_(qk_scale)
{

    check_cuda_error(cublasCreate(&cublas_handle_));
    check_cuda_error(cublasLtCreate(&cublaslt_handle_));
    checkCUDNN(cudnnCreate(&cudnn_handle_));

    sm_ = getSMVersion();
    bool _use_ORDER_COL32_2R_4R4 = false;
#if (CUDART_VERSION >= 11000)
    if (sm_ >= 80) {
        _use_ORDER_COL32_2R_4R4 = true;
    }
#endif

    weight_num_ = getWeightNum(layer_num, depths);
    if (weight_num_ != w.size()) {
        printf("[ERROR][SwinTransformerINT8Plugin](Weights) weights number %lu does not match expected number %d!\n",
               w.size(),
               weight_num_);
        exit(-1);
    }

    depths_ = (int*)malloc(layer_num * sizeof(int));
    num_heads_ = (int*)malloc(layer_num * sizeof(int));
    memcpy(depths_, depths, layer_num * sizeof(int));
    memcpy(num_heads_, num_heads, layer_num * sizeof(int));

    output_dim_ = int(pow(2, layer_num - 1)) * embed_dim;

    int weight_idx = 0;
    int amax_idx = 0;
    for (int l = 0; l < layer_num; l++) {
        SwinTransformerINT8BasicLayerWeight<T> bl;
        for (int di = 0; di < depths[l]; di++) {
            SwinTransformerINT8BlockWeight<T> p;
            p.attention_weights.query_weight.kernel = cudaMallocAndCopy(weights_, w[weight_idx++]);
            p.attention_weights.query_weight.bias = cudaMallocAndCopy(weights_, w[weight_idx++]);
            p.attention_weights.attention_output_weight.kernel = cudaMallocAndCopy(weights_, w[weight_idx++]);
            p.attention_weights.attention_output_weight.bias = cudaMallocAndCopy(weights_, w[weight_idx++]);
            p.ffn_weights.intermediate_weight.kernel = cudaMallocAndCopy(weights_, w[weight_idx++]);
            p.ffn_weights.intermediate_weight.bias = cudaMallocAndCopy(weights_, w[weight_idx++]);
            p.ffn_weights.output_weight.kernel = cudaMallocAndCopy(weights_, w[weight_idx++]);
            p.ffn_weights.output_weight.bias = cudaMallocAndCopy(weights_, w[weight_idx++]);
            p.attn_layernorm_weights.gamma = cudaMallocAndCopy(weights_, w[weight_idx++]);
            p.attn_layernorm_weights.beta = cudaMallocAndCopy(weights_, w[weight_idx++]);
            p.ffn_layernorm_weights.gamma = cudaMallocAndCopy(weights_, w[weight_idx++]);
            p.ffn_layernorm_weights.beta = cudaMallocAndCopy(weights_, w[weight_idx++]);
            p.scalelist.size_ = ACTIVATION_AMAX_NUM + 5 + INT8O_GEMM_NUM + TRT_AMAX_NUM;
            p.scalelist.p2_offset_ = ACTIVATION_AMAX_NUM;
            p.scalelist.p3_offset_ = ACTIVATION_AMAX_NUM + 5;
            p.scalelist.p4_offset_ = ACTIVATION_AMAX_NUM + 5 + INT8O_GEMM_NUM;
            p.scalelist.d_scale_list_ = d_amaxCopy(d_amaxlist_, d_amax[amax_idx]);
            p.scalelist.h_scale_list_ = h_amaxCopy(h_amaxlist_, h_amax[amax_idx]);
            amax_idx++;
            p.attention_relative_pos_bias = cudaMallocAndCopy(weights_, w[weight_idx++]);
            bl.block_weight_list.push_back(p);
        }
        bl.merge_layernorm_weights.gamma = cudaMallocAndCopy(weights_, w[weight_idx++]);
        bl.merge_layernorm_weights.beta = cudaMallocAndCopy(weights_, w[weight_idx++]);
        bl.merge_linear_weights.kernel = cudaMallocAndCopy(weights_, w[weight_idx++]);
        bl.attn_mask = cudaMallocAndCopy(weights_, w[weight_idx++]);
        params_.basic_layer_weight_list.push_back(bl);
    }
    params_.patchEmbed_linear_weights.kernel = cudaMallocAndCopy(weights_, w[weight_idx++]);
    params_.patchEmbed_linear_weights.bias = cudaMallocAndCopy(weights_, w[weight_idx++]);
    params_.patchEmbed_norm_weights.gamma = cudaMallocAndCopy(weights_, w[weight_idx++]);
    params_.patchEmbed_norm_weights.beta = cudaMallocAndCopy(weights_, w[weight_idx++]);
    params_.norm_weights.gamma = cudaMallocAndCopy(weights_, w[weight_idx++]);
    params_.norm_weights.beta = cudaMallocAndCopy(weights_, w[weight_idx++]);

    cublasAlgoMap_ = new cublasAlgoMap("igemm.config", "");
    cublasWrapperMutex_ = new std::mutex();
    allocator_ = new Allocator<AllocatorType::CUDA>(getDevice());

    cublasINT8MMWrapper* cublas_wrapper = new cublasINT8MMWrapper(
        cublas_handle_, cublaslt_handle_, nullptr, cublasAlgoMap_, cublasWrapperMutex_, _use_ORDER_COL32_2R_4R4);
    if (std::is_same<T, half>::value) {
        cublas_wrapper->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper->setFP32GemmConfig();
    }

    swin_transformer_ = new SwinTransformerINT8<T>(int8_mode,
                                                   max_batch_size,
                                                   img_size,
                                                   patch_size,
                                                   in_chans,
                                                   embed_dim,
                                                   window_size,
                                                   depths,
                                                   num_heads,
                                                   ape,
                                                   patch_norm,
                                                   layer_num,
                                                   mlp_ratio,
                                                   cudnn_handle_,
                                                   nullptr,
                                                   cublas_wrapper,
                                                   allocator_,
                                                   false,
                                                   qkv_bias,
                                                   qk_scale);
}

template<typename T>
SwinTransformerINT8Plugin<T>::SwinTransformerINT8Plugin(const std::string& name, const void* data, size_t length):
    layer_name_(name)
{
    check_cuda_error(cublasCreate(&cublas_handle_));
    check_cuda_error(cublasLtCreate(&cublaslt_handle_));
    checkCUDNN(cudnnCreate(&cudnn_handle_));

    sm_ = getSMVersion();
    bool _use_ORDER_COL32_2R_4R4 = false;
#if (CUDART_VERSION >= 11000)
    if (sm_ >= 80) {
        _use_ORDER_COL32_2R_4R4 = true;
    }
#endif

    deserialize_value(&data, &length, &int8_mode_);
    deserialize_value(&data, &length, &output_dim_);
    deserialize_value(&data, &length, &max_batch_size_);
    deserialize_value(&data, &length, &img_size_);
    deserialize_value(&data, &length, &patch_size_);
    deserialize_value(&data, &length, &in_chans_);
    deserialize_value(&data, &length, &embed_dim_);
    deserialize_value(&data, &length, &window_size_);
    deserialize_value(&data, &length, &ape_);
    deserialize_value(&data, &length, &patch_norm_);
    deserialize_value(&data, &length, &layer_num_);
    deserialize_value(&data, &length, &mlp_ratio_);
    deserialize_value(&data, &length, &qkv_bias_);
    deserialize_value(&data, &length, &qk_scale_);
    deserialize_value(&data, &length, &weight_num_);
    for (int i = 0; i < weight_num_; i++) {
        size_t tmp;
        deserialize_value(&data, &length, &tmp);
        weight_size_.push_back(tmp);
    }

    depths_ = (int*)malloc(layer_num_ * sizeof(int));
    num_heads_ = (int*)malloc(layer_num_ * sizeof(int));
    const char* d = static_cast<const char*>(data);
    memcpy(depths_, d, layer_num_ * sizeof(int));
    d = d + layer_num_ * sizeof(int);
    memcpy(num_heads_, d, layer_num_ * sizeof(int));
    d = d + layer_num_ * sizeof(int);
    for (int i = 0; i < weight_size_.size(); i++) {
        T* tmp;
        check_cuda_error(cudaMalloc((void**)&tmp, weight_size_[i] * sizeof(T)));
        check_cuda_error(cudaMemcpy(tmp, d, weight_size_[i] * sizeof(T), cudaMemcpyHostToDevice));
        d = d + weight_size_[i] * sizeof(T);
        weights_.push_back(tmp);
    }

    int all_depth = 0;
    for (int i = 0; i < layer_num_; i++) {
        all_depth += depths_[i];
    }
    for (int i = 0; i < all_depth; i++) {
        float* tmp;
        check_cuda_error(cudaMalloc((void**)&tmp, 96 * sizeof(float)));
        check_cuda_error(cudaMemcpy(tmp, d, 96 * sizeof(float), cudaMemcpyHostToDevice));
        d = d + 96 * sizeof(float);
        d_amaxlist_.push_back(tmp);
    }
    for (int i = 0; i < all_depth; i++) {
        float* tmp;
        tmp = (float*)malloc(96 * sizeof(float));
        check_cuda_error(cudaMemcpy(tmp, d, 96 * sizeof(float), cudaMemcpyHostToHost));
        // printf("[[%d]]:\n", i);
        // for(int i = 0; i < 96; i ++){
        //   printf("%.6f ", tmp[i]);
        // }
        // printf("\n");
        d = d + 96 * sizeof(float);
        h_amaxlist_.push_back(tmp);
    }

    int weight_idx = 0;
    int amax_idx = 0;
    for (int l = 0; l < layer_num_; l++) {
        SwinTransformerINT8BasicLayerWeight<T> bl;
        for (int di = 0; di < depths_[l]; di++) {
            SwinTransformerINT8BlockWeight<T> p;
            p.attention_weights.query_weight.kernel = weights_[weight_idx++];
            p.attention_weights.query_weight.bias = weights_[weight_idx++];
            p.attention_weights.attention_output_weight.kernel = weights_[weight_idx++];
            p.attention_weights.attention_output_weight.bias = weights_[weight_idx++];
            p.ffn_weights.intermediate_weight.kernel = weights_[weight_idx++];
            p.ffn_weights.intermediate_weight.bias = weights_[weight_idx++];
            p.ffn_weights.output_weight.kernel = weights_[weight_idx++];
            p.ffn_weights.output_weight.bias = weights_[weight_idx++];
            p.attn_layernorm_weights.gamma = weights_[weight_idx++];
            p.attn_layernorm_weights.beta = weights_[weight_idx++];
            p.ffn_layernorm_weights.gamma = weights_[weight_idx++];
            p.ffn_layernorm_weights.beta = weights_[weight_idx++];
            p.scalelist.size_ = ACTIVATION_AMAX_NUM + 5 + INT8O_GEMM_NUM + TRT_AMAX_NUM;
            p.scalelist.p2_offset_ = ACTIVATION_AMAX_NUM;
            p.scalelist.p3_offset_ = ACTIVATION_AMAX_NUM + 5;
            p.scalelist.p4_offset_ = ACTIVATION_AMAX_NUM + 5 + INT8O_GEMM_NUM;
            p.scalelist.d_scale_list_ = d_amaxlist_[amax_idx];
            p.scalelist.h_scale_list_ = h_amaxlist_[amax_idx];
            amax_idx++;
            p.attention_relative_pos_bias = weights_[weight_idx++];
            bl.block_weight_list.push_back(p);
        }
        bl.merge_layernorm_weights.gamma = weights_[weight_idx++];
        bl.merge_layernorm_weights.beta = weights_[weight_idx++];
        bl.merge_linear_weights.kernel = weights_[weight_idx++];
        bl.attn_mask = weights_[weight_idx++];
        params_.basic_layer_weight_list.push_back(bl);
    }
    params_.patchEmbed_linear_weights.kernel = weights_[weight_idx++];
    params_.patchEmbed_linear_weights.bias = weights_[weight_idx++];
    params_.patchEmbed_norm_weights.gamma = weights_[weight_idx++];
    params_.patchEmbed_norm_weights.beta = weights_[weight_idx++];
    params_.norm_weights.gamma = weights_[weight_idx++];
    params_.norm_weights.beta = weights_[weight_idx++];

    cublasAlgoMap_ = new cublasAlgoMap(IGEMM_CONFIG, "");
    cublasWrapperMutex_ = new std::mutex();
    allocator_ = new Allocator<AllocatorType::CUDA>(getDevice());

    cublasINT8MMWrapper* cublas_wrapper = new cublasINT8MMWrapper(
        cublas_handle_, cublaslt_handle_, nullptr, cublasAlgoMap_, cublasWrapperMutex_, _use_ORDER_COL32_2R_4R4);
    if (std::is_same<T, half>::value) {
        cublas_wrapper->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper->setFP32GemmConfig();
    }

    swin_transformer_ = new SwinTransformerINT8<T>(int8_mode_,
                                                   max_batch_size_,
                                                   img_size_,
                                                   patch_size_,
                                                   in_chans_,
                                                   embed_dim_,
                                                   window_size_,
                                                   depths_,
                                                   num_heads_,
                                                   ape_,
                                                   patch_norm_,
                                                   layer_num_,
                                                   mlp_ratio_,
                                                   cudnn_handle_,
                                                   nullptr,
                                                   cublas_wrapper,
                                                   allocator_,
                                                   false,
                                                   qkv_bias_,
                                                   qk_scale_);
}

template<typename T>
SwinTransformerINT8Plugin<T>::~SwinTransformerINT8Plugin()
{
    for (int i = 0; i < weights_.size(); i++) {
        check_cuda_error(cudaFree(weights_[i]));
    }
    for (int i = 0; i < d_amaxlist_.size(); i++) {
        check_cuda_error(cudaFree(d_amaxlist_[i]));
    }
    for (int i = 0; i < h_amaxlist_.size(); i++) {
        delete h_amaxlist_[i];
    }
    check_cuda_error(cublasDestroy(cublas_handle_));
    checkCUDNN(cudnnDestroy(cudnn_handle_));
    delete cublasWrapperMutex_;
    delete cublasAlgoMap_;
    delete swin_transformer_;
    delete allocator_;
    free(num_heads_);
    free(depths_);
    weights_.clear();
    weight_size_.clear();
}

// IPluginV2DynamicExt Methods
template<typename T>
nvinfer1::IPluginV2DynamicExt* SwinTransformerINT8Plugin<T>::clone() const noexcept
{
    printf("clone(): size is %lu\n", weights_.size());
    SwinTransformerINT8Plugin* ret = new SwinTransformerINT8Plugin<T>(layer_name_,
                                                                      int8_mode_,
                                                                      max_batch_size_,
                                                                      img_size_,
                                                                      patch_size_,
                                                                      in_chans_,
                                                                      embed_dim_,
                                                                      window_size_,
                                                                      depths_,
                                                                      num_heads_,
                                                                      ape_,
                                                                      patch_norm_,
                                                                      layer_num_,
                                                                      mlp_ratio_,
                                                                      qkv_bias_,
                                                                      qk_scale_,
                                                                      weights_,
                                                                      d_amaxlist_,
                                                                      h_amaxlist_);
    return ret;
}

template<typename T>
DimsExprs SwinTransformerINT8Plugin<T>::getOutputDimensions(int outputIndex,
                                                            const DimsExprs* inputs,
                                                            int nbInputs,
                                                            IExprBuilder& exprBuilder) noexcept
{
    // Input is B*in_chans*H*W, output should be B*dim*1*1 for fc layer
    assert(outputIndex == 0);
    // Copy over everything
    DimsExprs output;
    output.nbDims = 4;
    output.d[0] = inputs[0].d[0];
    output.d[1] = exprBuilder.constant(output_dim_);
    output.d[2] = exprBuilder.constant(1);
    output.d[3] = exprBuilder.constant(1);
    return output;
}

template<typename T>
bool SwinTransformerINT8Plugin<T>::supportsFormatCombination(int pos,
                                                             const PluginTensorDesc* inOut,
                                                             int nbInputs,
                                                             int nbOutputs) noexcept
{
    assert(pos >= 0);
    assert(pos < 2);
    assert(nbInputs == 1);

    return true;
}

template<typename T>
void SwinTransformerINT8Plugin<T>::configurePlugin(const DynamicPluginTensorDesc* in,
                                                   int nbInputs,
                                                   const DynamicPluginTensorDesc* out,
                                                   int nbOutputs) noexcept
{
    assert(nbInputs == 1);
    assert(nbOutputs == 1);
}

template<typename T>
size_t SwinTransformerINT8Plugin<T>::getWorkspaceSize(const PluginTensorDesc* inputs,
                                                      int nbInputs,
                                                      const PluginTensorDesc* outputs,
                                                      int nbOutputs) const noexcept
{
    return 0;
}

// IPluginV2Ext Methods
template<typename T>
nvinfer1::DataType SwinTransformerINT8Plugin<T>::getOutputDataType(int index,
                                                                   const nvinfer1::DataType* inputTypes,
                                                                   int nbInputs) const noexcept
{
    assert(index == 0);
    assert(inputTypes[0] == nvinfer1::DataType::kFLOAT || inputTypes[0] == nvinfer1::DataType::kHALF);
    return inputTypes[0];
}

// IPluginV2 Methods
template<typename T>
const char* SwinTransformerINT8Plugin<T>::getPluginType() const noexcept
{
    return SWIN_TRANSFORMER_PLUGIN_NAME;
}

template<typename T>
const char* SwinTransformerINT8Plugin<T>::getPluginVersion() const noexcept
{
    return SWIN_TRANSFORMER_PLUGIN_VERSION;
}

template<typename T>
int SwinTransformerINT8Plugin<T>::getNbOutputs() const noexcept
{
    return 1;
}

template<typename T>
int SwinTransformerINT8Plugin<T>::initialize() noexcept
{
    return 0;
}

template<typename T>
void SwinTransformerINT8Plugin<T>::terminate() noexcept
{
}

template<typename T>
size_t SwinTransformerINT8Plugin<T>::getSerializationSize() const noexcept
{

    size_t size = sizeof(int) + sizeof(int8_mode_) + sizeof(output_dim_) + sizeof(max_batch_size_) + sizeof(img_size_)
                  + sizeof(patch_size_) + sizeof(in_chans_) + sizeof(embed_dim_) + sizeof(window_size_) + sizeof(ape_)
                  + sizeof(patch_norm_) + sizeof(layer_num_) + sizeof(mlp_ratio_) + sizeof(qkv_bias_)
                  + sizeof(qk_scale_) + sizeof(weight_num_) + weight_num_ * sizeof(size_t) + layer_num_ * sizeof(int)
                  + layer_num_ * sizeof(int);
    for (int i = 0; i < weight_size_.size(); i++) {
        size += weight_size_[i] * sizeof(T);
    }
    for (int i = 0; i < d_amaxlist_.size(); i++) {
        size += 96 * sizeof(float);
    }
    for (int i = 0; i < h_amaxlist_.size(); i++) {
        size += 96 * sizeof(float);
    }
    return size;
}

template<typename T>
void SwinTransformerINT8Plugin<T>::serialize(void* buffer) const noexcept
{
    int type_id = 0;
    if (std::is_same<T, half>::value) {
        type_id = 1;
    }
    serialize_value(&buffer, type_id);
    serialize_value(&buffer, int8_mode_);
    serialize_value(&buffer, output_dim_);
    serialize_value(&buffer, max_batch_size_);
    serialize_value(&buffer, img_size_);
    serialize_value(&buffer, patch_size_);
    serialize_value(&buffer, in_chans_);
    serialize_value(&buffer, embed_dim_);
    serialize_value(&buffer, window_size_);
    serialize_value(&buffer, ape_);
    serialize_value(&buffer, patch_norm_);
    serialize_value(&buffer, layer_num_);
    serialize_value(&buffer, mlp_ratio_);
    serialize_value(&buffer, qkv_bias_);
    serialize_value(&buffer, qk_scale_);
    serialize_value(&buffer, weight_num_);
    for (int i = 0; i < weight_size_.size(); i++)
        serialize_value(&buffer, weight_size_[i]);

    char* d = static_cast<char*>(buffer);
    memcpy(d, depths_, layer_num_ * sizeof(int));
    d += layer_num_ * sizeof(int);
    memcpy(d, num_heads_, layer_num_ * sizeof(int));
    d += layer_num_ * sizeof(int);
    for (int i = 0; i < weight_size_.size(); i++) {
        check_cuda_error(cudaMemcpy(d, weights_[i], weight_size_[i] * sizeof(T), cudaMemcpyDeviceToHost));
        d += weight_size_[i] * sizeof(T);
    }
    for (int i = 0; i < d_amaxlist_.size(); i++) {
        // float* ptr = (float*)d;
        // printf("[[%d]]: ", i);
        // for(int j = 0; j < 96; j ++){
        //   printf("%.6f ", ptr[j]);
        // }
        // printf("\n");
        check_cuda_error(cudaMemcpy(d, d_amaxlist_[i], 96 * sizeof(float), cudaMemcpyDeviceToHost));
        d += 96 * sizeof(float);
    }

    for (int i = 0; i < h_amaxlist_.size(); i++) {
        // float* ptr = h_amaxlist_[i];
        // printf("[[%d]]: ", i);
        // for(int j = 0; j < 96; j ++){
        //   printf("%.6f ", ptr[j]);
        // }
        // printf("\n");
        check_cuda_error(cudaMemcpy(d, h_amaxlist_[i], 96 * sizeof(float), cudaMemcpyHostToHost));
        d += 96 * sizeof(float);
    }
}

template<typename T>
void SwinTransformerINT8Plugin<T>::destroy() noexcept
{
    delete this;
}

template<typename T>
void SwinTransformerINT8Plugin<T>::setPluginNamespace(const char* libNamespace) noexcept
{
    namespace_ = libNamespace;
}

template<typename T>
const char* SwinTransformerINT8Plugin<T>::getPluginNamespace() const noexcept
{
    return namespace_.c_str();
}

template<typename T>
int SwinTransformerINT8Plugin<T>::enqueue(const PluginTensorDesc* inputDesc,
                                          const PluginTensorDesc* outputDesc,
                                          const void* const* inputs,
                                          void* const* outputs,
                                          void* workspace,
                                          cudaStream_t stream) noexcept
{
    int batch_size = inputDesc->dims.d[0];
    assert(batch_size <= max_batch_size_);
    assert(in_chans_ == inputDesc->dims.d[1]);
    assert(img_size_ == inputDesc->dims.d[2]);
    assert(img_size_ == inputDesc->dims.d[3]);

    int sm_ptr[1] = {sm_};
    std::vector<Tensor> input_tensors = std::vector<Tensor>{
        Tensor{MEMORY_GPU,
               getTensorType<T>(),
               std::vector<size_t>{(size_t)batch_size, (size_t)img_size_ * img_size_, (size_t)in_chans_},
               (const T*)(inputs[0])},
        Tensor{MEMORY_CPU, TYPE_INT8, std::vector<size_t>{1}, sm_ptr}};

    std::vector<Tensor> output_tensors = std::vector<Tensor>{
        Tensor{MEMORY_GPU,
               getTensorType<T>(),
               std::vector<size_t>{(size_t)batch_size, (size_t)img_size_ * img_size_, (size_t)in_chans_},
               (T*)(outputs[0])}};

    swin_transformer_->forward(&output_tensors, &input_tensors, params_);
    return 0;
}

SwinTransformerINT8PluginCreator::SwinTransformerINT8PluginCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* SwinTransformerINT8PluginCreator::getPluginName() const noexcept
{
    return SWIN_TRANSFORMER_PLUGIN_NAME;
}

const char* SwinTransformerINT8PluginCreator::getPluginVersion() const noexcept
{
    return SWIN_TRANSFORMER_PLUGIN_VERSION;
}

const PluginFieldCollection* SwinTransformerINT8PluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* SwinTransformerINT8PluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    int int8_mode;
    int max_batch_size;
    int img_size;
    int patch_size;
    int in_chans;
    int embed_dim;
    int window_size;
    int* depths = nullptr;
    int* num_heads = nullptr;
    bool ape;
    bool patch_norm;
    int layer_num;
    float mlp_ratio;
    bool qkv_bias;
    float qk_scale;
    std::vector<Weights> w;
    std::vector<Weights> d_amax;
    std::vector<Weights> h_amax;

    for (int i = 0; i < fc->nbFields; i++) {
        std::string field_name(fc->fields[i].name);
        if (field_name.compare("int8_mode") == 0) {
            int8_mode = *static_cast<const int*>(fc->fields[i].data);
        }
        if (field_name.compare("max_batch_size") == 0) {
            max_batch_size = *static_cast<const int*>(fc->fields[i].data);
        }
        if (field_name.compare("img_size") == 0) {
            img_size = *static_cast<const int*>(fc->fields[i].data);
        }
        if (field_name.compare("patch_size") == 0) {
            patch_size = *static_cast<const int*>(fc->fields[i].data);
        }
        if (field_name.compare("in_chans") == 0) {
            in_chans = *static_cast<const int*>(fc->fields[i].data);
        }
        if (field_name.compare("embed_dim") == 0) {
            embed_dim = *static_cast<const int*>(fc->fields[i].data);
        }
        if (field_name.compare("window_size") == 0) {
            window_size = *static_cast<const int*>(fc->fields[i].data);
        }
        if (field_name.compare("ape") == 0) {
            int tmp = *static_cast<const int*>(fc->fields[i].data);
            if (tmp == 1)
                ape = true;
            else
                ape = false;
        }
        if (field_name.compare("patch_norm") == 0) {
            int tmp = *static_cast<const int*>(fc->fields[i].data);
            if (tmp == 1)
                patch_norm = true;
            else
                patch_norm = false;
        }
        if (field_name.compare("layer_num") == 0) {
            layer_num = *static_cast<const int*>(fc->fields[i].data);
        }
        if (field_name.compare("mlp_ratio") == 0) {
            mlp_ratio = *static_cast<const float*>(fc->fields[i].data);
        }
        if (field_name.compare("qkv_bias") == 0) {
            int tmp = *static_cast<const int*>(fc->fields[i].data);
            if (tmp == 1)
                qkv_bias = true;
            else
                qkv_bias = false;
        }
        if (field_name.compare("qk_scale") == 0) {
            qk_scale = *static_cast<const float*>(fc->fields[i].data);
        }
        if (field_name.compare("depths") == 0) {
            depths = (int*)malloc(fc->fields[i].length * sizeof(int));
            memcpy(depths, fc->fields[i].data, fc->fields[i].length * sizeof(int));
        }
        if (field_name.compare("num_heads") == 0) {
            num_heads = (int*)malloc(fc->fields[i].length * sizeof(int));
            memcpy(num_heads, fc->fields[i].data, fc->fields[i].length * sizeof(int));
        }
    }

    if (depths == nullptr || num_heads == nullptr) {
        printf("[ERROR][SwinTransformerINT8PluginCreator::createPlugin] empty depths or num_heads!\n");
        exit(-1);
    }

    char weight_name[1024];
    for (int l = 0; l < layer_num; l++) {
        for (int b = 0; b < depths[l]; b++) {
            sprintf(weight_name, "attention_qkv_kernel_%d_%d", l, b);
            getWeightsFromFC(weight_name, fc, w);
            sprintf(weight_name, "attention_qkv_bias_%d_%d", l, b);
            getWeightsFromFC(weight_name, fc, w);
            sprintf(weight_name, "attention_proj_kernel_%d_%d", l, b);
            getWeightsFromFC(weight_name, fc, w);
            sprintf(weight_name, "attention_proj_bias_%d_%d", l, b);
            getWeightsFromFC(weight_name, fc, w);
            sprintf(weight_name, "mlp_linear_kernel_%d_%d", l, b);
            getWeightsFromFC(weight_name, fc, w);
            sprintf(weight_name, "mlp_linear_bias_%d_%d", l, b);
            getWeightsFromFC(weight_name, fc, w);
            sprintf(weight_name, "mlp_linear2_kernel_%d_%d", l, b);
            getWeightsFromFC(weight_name, fc, w);
            sprintf(weight_name, "mlp_linear2_bias_%d_%d", l, b);
            getWeightsFromFC(weight_name, fc, w);
            sprintf(weight_name, "block_norm_gamma_%d_%d", l, b);
            getWeightsFromFC(weight_name, fc, w);
            sprintf(weight_name, "block_norm_beta_%d_%d", l, b);
            getWeightsFromFC(weight_name, fc, w);
            sprintf(weight_name, "block_norm2_gamma_%d_%d", l, b);
            getWeightsFromFC(weight_name, fc, w);
            sprintf(weight_name, "block_norm2_beta_%d_%d", l, b);
            getWeightsFromFC(weight_name, fc, w);
            sprintf(weight_name, "block_d_amaxlist_%d_%d", l, b);
            getWeightsFromFC(weight_name, fc, d_amax);
            sprintf(weight_name, "block_h_amaxlist_%d_%d", l, b);
            getWeightsFromFC(weight_name, fc, h_amax);

            sprintf(weight_name, "attention_relative_pos_bias_%d_%d", l, b);
            getWeightsFromFC(weight_name, fc, w);
        }
        sprintf(weight_name, "patchMerge_norm_gamma_%d", l);
        getWeightsFromFC(weight_name, fc, w);
        sprintf(weight_name, "patchMerge_norm_beta_%d", l);
        getWeightsFromFC(weight_name, fc, w);
        sprintf(weight_name, "patchMerge_linear_kernel_%d", l);
        getWeightsFromFC(weight_name, fc, w);
        sprintf(weight_name, "attn_mask_%d", l);
        getWeightsFromFC(weight_name, fc, w);
    }
    sprintf(weight_name, "patchEmbed_proj_kernel");
    getWeightsFromFC(weight_name, fc, w);
    sprintf(weight_name, "patchEmbed_proj_bias");
    getWeightsFromFC(weight_name, fc, w);
    sprintf(weight_name, "patchEmbed_norm_gamma");
    getWeightsFromFC(weight_name, fc, w);
    sprintf(weight_name, "patchEmbed_norm_beta");
    getWeightsFromFC(weight_name, fc, w);
    sprintf(weight_name, "norm_gamma");
    getWeightsFromFC(weight_name, fc, w);
    sprintf(weight_name, "norm_beta");
    getWeightsFromFC(weight_name, fc, w);

    if (w[0].type == nvinfer1::DataType::kFLOAT) {
        SwinTransformerINT8Plugin<float>* p = new SwinTransformerINT8Plugin<float>(name,
                                                                                   int8_mode,
                                                                                   max_batch_size,
                                                                                   img_size,
                                                                                   patch_size,
                                                                                   in_chans,
                                                                                   embed_dim,
                                                                                   window_size,
                                                                                   depths,
                                                                                   num_heads,
                                                                                   ape,
                                                                                   patch_norm,
                                                                                   layer_num,
                                                                                   mlp_ratio,
                                                                                   qkv_bias,
                                                                                   qk_scale,
                                                                                   w,
                                                                                   d_amax,
                                                                                   h_amax);

        if (depths != nullptr)
            free(depths);
        if (num_heads != nullptr)
            free(num_heads);
        return p;
    }
    else if (w[0].type == nvinfer1::DataType::kHALF) {
        SwinTransformerINT8Plugin<half>* p = new SwinTransformerINT8Plugin<half>(name,
                                                                                 int8_mode,
                                                                                 max_batch_size,
                                                                                 img_size,
                                                                                 patch_size,
                                                                                 in_chans,
                                                                                 embed_dim,
                                                                                 window_size,
                                                                                 depths,
                                                                                 num_heads,
                                                                                 ape,
                                                                                 patch_norm,
                                                                                 layer_num,
                                                                                 mlp_ratio,
                                                                                 qkv_bias,
                                                                                 qk_scale,
                                                                                 w,
                                                                                 d_amax,
                                                                                 h_amax);
        if (depths != nullptr)
            free(depths);
        if (num_heads != nullptr)
            free(num_heads);
        return p;
    }
    else {
        printf("[ERROR][SwinTransformerINT8PluginCreator::createPlugin] unsupport datatype.\n");
        exit(-1);
    }
}

IPluginV2* SwinTransformerINT8PluginCreator::deserializePlugin(const char* name,
                                                               const void* serialData,
                                                               size_t serialLength) noexcept
{
    int type_id;
    size_t int_length = sizeof(int);
    deserialize_value(&serialData, &int_length, &type_id);
    // This object will be deleted when the network is destroyed, which will
    // call SwinTransformerINT8Plugin::destroy()
    if (type_id == 0)
        return new SwinTransformerINT8Plugin<float>(name, serialData, serialLength);
    else if (type_id == 1)
        return new SwinTransformerINT8Plugin<half>(name, serialData, serialLength);
    else {
        printf("[ERROR][SwinTransformerINT8PluginCreator::deserializePlugin] unsupport data type %d\n", type_id);
        exit(-1);
    }
}

void SwinTransformerINT8PluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    namespace_ = libNamespace;
}

const char* SwinTransformerINT8PluginCreator::getPluginNamespace() const noexcept
{
    return namespace_.c_str();
}

template class SwinTransformerINT8Plugin<half>;
template class SwinTransformerINT8Plugin<float>;

}  // namespace fastertransformer
