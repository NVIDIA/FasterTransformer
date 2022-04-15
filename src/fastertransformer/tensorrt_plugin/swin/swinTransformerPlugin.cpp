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

#include "swinTransformerPlugin.h"
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
PluginFieldCollection SwinTransformerPluginCreator::mFC{};
std::vector<PluginField> SwinTransformerPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(SwinTransformerPluginCreator);

template<typename T>
SwinTransformerPlugin<T>::SwinTransformerPlugin(const std::string& name,
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
                                                const std::vector<T*>& w):
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

    weight_num_ = getWeightNum(layer_num, depths);
    if (weight_num_ != w.size()) {
        printf("[ERROR][SwinTransformerPlugin] weights number %lu does not match expected number %d!\n",
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
    for (int l = 0; l < layer_num; l++) {
        SwinTransformerBasicLayerWeight<T> bl;
        for (int di = 0; di < depths[l]; di++) {
            SwinTransformerBlockWeight<T> p;
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
        params_.basic_layer_weight_list.push_back(bl);
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

    cublasAlgoMap_ = new cublasAlgoMap(GEMM_CONFIG, "");
    cublasWrapperMutex_ = new std::mutex();
    allocator_ = new Allocator<AllocatorType::CUDA>(getDevice());

    cublasMMWrapper* cublas_wrapper =
        new cublasMMWrapper(cublas_handle_, cublaslt_handle_, nullptr, cublasAlgoMap_, cublasWrapperMutex_, nullptr);
    if (std::is_same<T, half>::value) {
        cublas_wrapper->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper->setFP32GemmConfig();
    }

    swin_transformer_ = new SwinTransformer<T>(max_batch_size,
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
SwinTransformerPlugin<T>::SwinTransformerPlugin(const std::string& name,
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
                                                const std::vector<Weights>& w):
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
    weight_num_ = getWeightNum(layer_num, depths);
    if (weight_num_ != w.size()) {
        printf("[ERROR][SwinTransformerPlugin] weights number %lu does not match expected number %d!\n",
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
    for (int l = 0; l < layer_num; l++) {
        SwinTransformerBasicLayerWeight<T> bl;
        for (int di = 0; di < depths[l]; di++) {
            SwinTransformerBlockWeight<T> p;
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

    cublasMMWrapper* cublas_wrapper =
        new cublasMMWrapper(cublas_handle_, cublaslt_handle_, nullptr, cublasAlgoMap_, cublasWrapperMutex_, nullptr);
    if (std::is_same<T, half>::value) {
        cublas_wrapper->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper->setFP32GemmConfig();
    }

    swin_transformer_ = new SwinTransformer<T>(max_batch_size_,
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
SwinTransformerPlugin<T>::SwinTransformerPlugin(const std::string& name, const void* data, size_t length):
    layer_name_(name)
{
    check_cuda_error(cublasCreate(&cublas_handle_));
    check_cuda_error(cublasLtCreate(&cublaslt_handle_));
    checkCUDNN(cudnnCreate(&cudnn_handle_));

    sm_ = getSMVersion();
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

    int weight_idx = 0;
    for (int l = 0; l < layer_num_; l++) {
        SwinTransformerBasicLayerWeight<T> bl;
        for (int di = 0; di < depths_[l]; di++) {
            SwinTransformerBlockWeight<T> p;
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

    cublasAlgoMap_ = new cublasAlgoMap("igemm.config", "");
    cublasWrapperMutex_ = new std::mutex();
    allocator_ = new Allocator<AllocatorType::CUDA>(getDevice());

    cublasMMWrapper* cublas_wrapper =
        new cublasMMWrapper(cublas_handle_, cublaslt_handle_, nullptr, cublasAlgoMap_, cublasWrapperMutex_, nullptr);
    if (std::is_same<T, half>::value) {
        cublas_wrapper->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper->setFP32GemmConfig();
    }

    swin_transformer_ = new SwinTransformer<T>(max_batch_size_,
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
SwinTransformerPlugin<T>::~SwinTransformerPlugin()
{
    for (int i = 0; i < weights_.size(); i++) {
        check_cuda_error(cudaFree(weights_[i]));
    }
    check_cuda_error(cublasDestroy(cublas_handle_));
    check_cuda_error(cublasLtDestroy(cublaslt_handle_));
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
nvinfer1::IPluginV2DynamicExt* SwinTransformerPlugin<T>::clone() const noexcept
{

    SwinTransformerPlugin* ret = new SwinTransformerPlugin<T>(layer_name_,
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
                                                              weights_);
    return ret;
}

template<typename T>
DimsExprs SwinTransformerPlugin<T>::getOutputDimensions(int outputIndex,
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
bool SwinTransformerPlugin<T>::supportsFormatCombination(int pos,
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
void SwinTransformerPlugin<T>::configurePlugin(const DynamicPluginTensorDesc* in,
                                               int nbInputs,
                                               const DynamicPluginTensorDesc* out,
                                               int nbOutputs) noexcept
{
    assert(nbInputs == 1);
    assert(nbOutputs == 1);
}

template<typename T>
size_t SwinTransformerPlugin<T>::getWorkspaceSize(const PluginTensorDesc* inputs,
                                                  int nbInputs,
                                                  const PluginTensorDesc* outputs,
                                                  int nbOutputs) const noexcept
{
    return 0;
}

// IPluginV2Ext Methods
template<typename T>
nvinfer1::DataType SwinTransformerPlugin<T>::getOutputDataType(int index,
                                                               const nvinfer1::DataType* inputTypes,
                                                               int nbInputs) const noexcept
{
    assert(index == 0);
    assert(inputTypes[0] == nvinfer1::DataType::kFLOAT || inputTypes[0] == nvinfer1::DataType::kHALF);
    return inputTypes[0];
}

// IPluginV2 Methods
template<typename T>
const char* SwinTransformerPlugin<T>::getPluginType() const noexcept
{
    return SWIN_TRANSFORMER_PLUGIN_NAME;
}

template<typename T>
const char* SwinTransformerPlugin<T>::getPluginVersion() const noexcept
{
    return SWIN_TRANSFORMER_PLUGIN_VERSION;
}

template<typename T>
int SwinTransformerPlugin<T>::getNbOutputs() const noexcept
{
    return 1;
}

template<typename T>
int SwinTransformerPlugin<T>::initialize() noexcept
{
    return 0;
}

template<typename T>
void SwinTransformerPlugin<T>::terminate() noexcept
{
}

template<typename T>
size_t SwinTransformerPlugin<T>::getSerializationSize() const noexcept
{

    size_t size = sizeof(int) + sizeof(output_dim_) + sizeof(max_batch_size_) + sizeof(img_size_) + sizeof(patch_size_)
                  + sizeof(in_chans_) + sizeof(embed_dim_) + sizeof(window_size_) + sizeof(ape_) + sizeof(patch_norm_)
                  + sizeof(layer_num_) + sizeof(mlp_ratio_) + sizeof(qkv_bias_) + sizeof(qk_scale_)
                  + sizeof(weight_num_) + weight_num_ * sizeof(size_t) + layer_num_ * sizeof(int)
                  + layer_num_ * sizeof(int);
    for (int i = 0; i < weight_size_.size(); i++) {
        size += weight_size_[i] * sizeof(T);
    }
    return size;
}

template<typename T>
void SwinTransformerPlugin<T>::serialize(void* buffer) const noexcept
{
    int type_id = 0;
    if (std::is_same<T, half>::value) {
        type_id = 1;
    }
    serialize_value(&buffer, type_id);
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
}

template<typename T>
void SwinTransformerPlugin<T>::destroy() noexcept
{
    delete this;
}

template<typename T>
void SwinTransformerPlugin<T>::setPluginNamespace(const char* libNamespace) noexcept
{
    namespace_ = libNamespace;
}

template<typename T>
const char* SwinTransformerPlugin<T>::getPluginNamespace() const noexcept
{
    return namespace_.c_str();
}

template<typename T>
int SwinTransformerPlugin<T>::enqueue(const PluginTensorDesc* inputDesc,
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

SwinTransformerPluginCreator::SwinTransformerPluginCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* SwinTransformerPluginCreator::getPluginName() const noexcept
{
    return SWIN_TRANSFORMER_PLUGIN_NAME;
}

const char* SwinTransformerPluginCreator::getPluginVersion() const noexcept
{
    return SWIN_TRANSFORMER_PLUGIN_VERSION;
}

const PluginFieldCollection* SwinTransformerPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* SwinTransformerPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
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

    for (int i = 0; i < fc->nbFields; i++) {
        std::string field_name(fc->fields[i].name);
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
        printf("[ERROR][SwinTransformerPluginCreator::createPlugin] empty depths or num_heads!\n");
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
        SwinTransformerPlugin<float>* p = new SwinTransformerPlugin<float>(name,
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
                                                                           w);

        if (depths != nullptr)
            free(depths);
        if (num_heads != nullptr)
            free(num_heads);
        return p;
    }
    else if (w[0].type == nvinfer1::DataType::kHALF) {
        SwinTransformerPlugin<half>* p = new SwinTransformerPlugin<half>(name,
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
                                                                         w);
        if (depths != nullptr)
            free(depths);
        if (num_heads != nullptr)
            free(num_heads);
        return p;
    }
    else {
        printf("[ERROR][SwinTransformerPluginCreator::createPlugin] unsupport datatype.\n");
        exit(-1);
    }
}

IPluginV2*
SwinTransformerPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    int type_id;
    size_t int_length = sizeof(int);
    deserialize_value(&serialData, &int_length, &type_id);
    // This object will be deleted when the network is destroyed, which will
    // call SwinTransformerPlugin::destroy()
    if (type_id == 0)
        return new SwinTransformerPlugin<float>(name, serialData, serialLength);
    else if (type_id == 1)
        return new SwinTransformerPlugin<half>(name, serialData, serialLength);
    else {
        printf("[ERROR][SwinTransformerPluginCreator::deserializePlugin] unsupport data type %d\n", type_id);
        exit(-1);
    }
}

void SwinTransformerPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    namespace_ = libNamespace;
}

const char* SwinTransformerPluginCreator::getPluginNamespace() const noexcept
{
    return namespace_.c_str();
}

template class SwinTransformerPlugin<half>;
template class SwinTransformerPlugin<float>;

}  // namespace fastertransformer
