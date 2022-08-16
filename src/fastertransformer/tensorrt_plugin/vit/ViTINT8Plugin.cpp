/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include "ViTINT8Plugin.h"
#include "NvInfer.h"
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
PluginFieldCollection    VisionTransformerINT8PluginCreator::mFC{};
std::vector<PluginField> VisionTransformerINT8PluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(VisionTransformerINT8PluginCreator);

template<typename T>
VisionTransformerINT8Plugin<T>::VisionTransformerINT8Plugin(const std::string&           name,
                                                            const int                    max_batch,
                                                            const int                    img_size,
                                                            const int                    patch_size,
                                                            const int                    in_chans,
                                                            const int                    embed_dim,
                                                            const int                    num_heads,
                                                            const int                    inter_size,
                                                            const int                    layer_num,
                                                            const float                  q_scaling,
                                                            const bool                   with_cls_token,
                                                            const int                    int8_mode,
                                                            const std::vector<const T*>& w):
    layer_name_(name)
{

    settings_.max_batch_size = max_batch;
    settings_.img_size       = img_size;
    settings_.chn_num        = in_chans;
    settings_.patch_size     = patch_size;
    settings_.embed_dim      = embed_dim;
    settings_.head_num       = num_heads;
    settings_.inter_size     = inter_size;
    settings_.num_layer      = layer_num;
    settings_.with_cls_token = with_cls_token;
    settings_.sm             = getSMVersion();
    settings_.q_scaling      = q_scaling;
    settings_.int8_mode      = int8_mode;
    settings_.seq_len        = (img_size / patch_size) * (img_size / patch_size) + (with_cls_token ? 1 : 0);
    settings_.attention_type = getAttentionType<T>(embed_dim / num_heads, settings_.sm, true, settings_.seq_len);

    Init(w);
}

template<typename T>
VisionTransformerINT8Plugin<T>::VisionTransformerINT8Plugin(const std::string& name, const void* data, size_t length):
    layer_name_(name)
{
    ::memcpy(&settings_, data, sizeof(settings_));
    const char* w_buffer = static_cast<const char*>(data) + sizeof(settings_);

    std::vector<const T*> dummy;
    Init(dummy);

    params_->deserialize(w_buffer);
}

template<typename T>
VisionTransformerINT8Plugin<T>::VisionTransformerINT8Plugin(const VisionTransformerINT8Plugin<T>& plugin):
    layer_name_(plugin.layer_name_), settings_(plugin.settings_)
{
    std::vector<const T*> dummy;
    Init(dummy);
    *params_ = *plugin.params_;
}

template<typename T>
void VisionTransformerINT8Plugin<T>::Init(const std::vector<const T*>& w)
{
    params_ = new ViTINT8Weight<T>(settings_.embed_dim,
                                   settings_.inter_size,
                                   settings_.num_layer,
                                   settings_.img_size,
                                   settings_.patch_size,
                                   settings_.chn_num,
                                   settings_.with_cls_token);

    if (w.size() > 0) {
        size_t weight_num = params_->GetWeightCount();

        if (weight_num != w.size()) {
            printf("[ERROR][VisionTransformerINT8Plugin] weights number %lu does not match expected number %lu!\n",
                   w.size(),
                   weight_num);
            exit(-1);
        }
        const T* const* pp_buf = &w[0];
        params_->CopyWeightsFromHostBuffers(pp_buf);
    }

    check_cuda_error(cublasCreate(&cublas_handle_));
    check_cuda_error(cublasLtCreate(&cublaslt_handle_));
    checkCUDNN(cudnnCreate(&cudnn_handle_));

    bool _use_ORDER_COL32_2R_4R4 = false;
#if (CUDART_VERSION >= 11000)
    if (settings_.sm >= 80) {
        _use_ORDER_COL32_2R_4R4 = true;
    }
#endif

    cublasAlgoMap_      = new cublasAlgoMap("igemm.config", "");
    cublasWrapperMutex_ = new std::mutex();
    allocator_          = new Allocator<AllocatorType::CUDA>(getDevice());

    cublas_wrapper_ = new cublasINT8MMWrapper(
        cublas_handle_, cublaslt_handle_, nullptr, cublasAlgoMap_, cublasWrapperMutex_, _use_ORDER_COL32_2R_4R4);
    if (std::is_same<T, half>::value) {
        cublas_wrapper_->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper_->setFP32GemmConfig();
    }

    vit_transformer_ = new ViTTransformerINT8<T>(settings_.max_batch_size,
                                                 settings_.img_size,
                                                 settings_.chn_num,
                                                 settings_.patch_size,
                                                 settings_.embed_dim,
                                                 settings_.head_num,
                                                 settings_.inter_size,
                                                 settings_.num_layer,
                                                 settings_.with_cls_token,
                                                 settings_.sm,
                                                 settings_.q_scaling,
                                                 settings_.int8_mode,
                                                 0,
                                                 cudnn_handle_,
                                                 cublas_wrapper_,
                                                 allocator_,
                                                 false,
                                                 settings_.attention_type);
}

template<typename T>
VisionTransformerINT8Plugin<T>::~VisionTransformerINT8Plugin()
{
    check_cuda_error(cublasDestroy(cublas_handle_));
    check_cuda_error(cublasLtDestroy(cublaslt_handle_));
    checkCUDNN(cudnnDestroy(cudnn_handle_));
    delete vit_transformer_;
    delete cublas_wrapper_;
    delete allocator_;
    delete cublasWrapperMutex_;
    delete cublasAlgoMap_;
    delete params_;
}

// IPluginV2DynamicExt Methods
template<typename T>
nvinfer1::IPluginV2DynamicExt* VisionTransformerINT8Plugin<T>::clone() const noexcept
{

    VisionTransformerINT8Plugin* ret = new VisionTransformerINT8Plugin<T>(*this);
    return ret;
}

template<typename T>
DimsExprs VisionTransformerINT8Plugin<T>::getOutputDimensions(int              outputIndex,
                                                              const DimsExprs* inputs,
                                                              int              nbInputs,
                                                              IExprBuilder&    exprBuilder) noexcept
{
    // Input is B*in_chans*H*W, output should be B*seq_len*embed_dim*1
    assert(outputIndex == 0);
    DimsExprs output;
    output.nbDims = 3;
    output.d[0]   = inputs[0].d[0];
    output.d[1]   = exprBuilder.constant(settings_.seq_len);
    output.d[2]   = exprBuilder.constant(settings_.embed_dim);
    return output;
}

template<typename T>
bool VisionTransformerINT8Plugin<T>::supportsFormatCombination(int                     pos,
                                                               const PluginTensorDesc* inOut,
                                                               int                     nbInputs,
                                                               int                     nbOutputs) noexcept
{
    bool res = false;
    assert(pos >= 0 && pos < 2);
    assert(nbInputs == 1);
    switch (pos) {
        case 0:  // input
        case 1:  // output
            res = (inOut[pos].type
                   == (std::is_same<T, half>::value ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT))
                  && (inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
            break;
        default:
            break;
    }

    return res;
}

template<typename T>
void VisionTransformerINT8Plugin<T>::configurePlugin(const DynamicPluginTensorDesc* in,
                                                     int                            nbInputs,
                                                     const DynamicPluginTensorDesc* out,
                                                     int                            nbOutputs) noexcept
{
    assert(nbInputs == 1);
    assert(nbOutputs == 1);
}

template<typename T>
size_t VisionTransformerINT8Plugin<T>::getWorkspaceSize(const PluginTensorDesc* inputs,
                                                        int                     nbInputs,
                                                        const PluginTensorDesc* outputs,
                                                        int                     nbOutputs) const noexcept
{
    return 0;
}

// IPluginV2Ext Methods
template<typename T>
nvinfer1::DataType VisionTransformerINT8Plugin<T>::getOutputDataType(int                       index,
                                                                     const nvinfer1::DataType* inputTypes,
                                                                     int                       nbInputs) const noexcept
{
    assert(index == 0);
    assert(inputTypes[0] == nvinfer1::DataType::kFLOAT || inputTypes[0] == nvinfer1::DataType::kHALF);
    return inputTypes[0];
}

// IPluginV2 Methods
template<typename T>
const char* VisionTransformerINT8Plugin<T>::getPluginType() const noexcept
{
    return VIT_PLUGIN_NAME;
}

template<typename T>
const char* VisionTransformerINT8Plugin<T>::getPluginVersion() const noexcept
{
    return VIT_PLUGIN_VERSION;
}

template<typename T>
int VisionTransformerINT8Plugin<T>::getNbOutputs() const noexcept
{
    return 1;
}

template<typename T>
int VisionTransformerINT8Plugin<T>::initialize() noexcept
{
    return 0;
}

template<typename T>
void VisionTransformerINT8Plugin<T>::terminate() noexcept
{
}

template<typename T>
size_t VisionTransformerINT8Plugin<T>::getSerializationSize() const noexcept
{

    size_t size = sizeof(int) + sizeof(settings_);
    size += params_->GetSerializeSize();
    return size;
}

template<typename T>
void VisionTransformerINT8Plugin<T>::serialize(void* buffer) const noexcept
{
    FT_LOG_INFO("start serialize vit...");

    int type_id = 0;
    if (std::is_same<T, half>::value) {
        type_id = 1;
    }
    ::memcpy(buffer, &type_id, sizeof(type_id));
    char* serial_buffer = (char*)buffer + sizeof(type_id);
    ::memcpy(serial_buffer, &settings_, sizeof(settings_));
    serial_buffer += sizeof(settings_);
    params_->serialize(serial_buffer);
}

template<typename T>
void VisionTransformerINT8Plugin<T>::destroy() noexcept
{
    delete this;
}

template<typename T>
void VisionTransformerINT8Plugin<T>::setPluginNamespace(const char* libNamespace) noexcept
{
    namespace_ = libNamespace;
}

template<typename T>
const char* VisionTransformerINT8Plugin<T>::getPluginNamespace() const noexcept
{
    return namespace_.c_str();
}

template<typename T>
int VisionTransformerINT8Plugin<T>::enqueue(const PluginTensorDesc* inputDesc,
                                            const PluginTensorDesc* outputDesc,
                                            const void* const*      inputs,
                                            void* const*            outputs,
                                            void*                   workspace,
                                            cudaStream_t            stream) noexcept
{
    int batch_size = inputDesc->dims.d[0];
    assert(batch_size <= settings_.max_batch_size);
    assert(settings_.chn_num == inputDesc->dims.d[1]);
    assert(settings_.img_size == inputDesc->dims.d[2]);
    assert(settings_.img_size == inputDesc->dims.d[3]);

    std::vector<Tensor> input_tensors = std::vector<Tensor>{Tensor{
        MEMORY_GPU,
        getTensorType<T>(),
        std::vector<size_t>{
            (size_t)batch_size, (size_t)settings_.chn_num, (size_t)settings_.img_size, (size_t)settings_.img_size},
        (const T*)(inputs[0])}};

    std::vector<Tensor> output_tensors = std::vector<Tensor>{
        Tensor{MEMORY_GPU,
               getTensorType<T>(),
               std::vector<size_t>{(size_t)batch_size, (size_t)settings_.seq_len, (size_t)settings_.embed_dim},
               (T*)(outputs[0])}};

    vit_transformer_->forward(&output_tensors, &input_tensors, params_);
    return 0;
}

VisionTransformerINT8PluginCreator::VisionTransformerINT8PluginCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields   = mPluginAttributes.data();
}

const char* VisionTransformerINT8PluginCreator::getPluginName() const noexcept
{
    return VIT_PLUGIN_NAME;
}

const char* VisionTransformerINT8PluginCreator::getPluginVersion() const noexcept
{
    return VIT_PLUGIN_VERSION;
}

const PluginFieldCollection* VisionTransformerINT8PluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

// Creator
#define L_ROOT "transformer.encoder.layer.%d"
#define ATT_Q "attn.query"
#define ATT_K "attn.key"
#define ATT_V "attn.value"
#define ATT_OUT "attn.out"
#define ATT_NORM "attention_norm"
#define FFN_NORM "ffn_norm"
#define FFN_IN "ffn.fc1"
#define FFN_OUT "ffn.fc2"

const std::vector<const char*> layer_weight_names = {L_ROOT "." ATT_NORM ".weight",
                                                     L_ROOT "." ATT_NORM ".bias",
                                                     L_ROOT "." ATT_Q ".weight",
                                                     L_ROOT "." ATT_Q ".bias",
                                                     L_ROOT "." ATT_K ".weight",
                                                     L_ROOT "." ATT_K ".bias",
                                                     L_ROOT "." ATT_V ".weight",
                                                     L_ROOT "." ATT_V ".bias",
                                                     L_ROOT "." ATT_OUT ".weight",
                                                     L_ROOT "." ATT_OUT ".bias",
                                                     L_ROOT "." FFN_NORM ".weight",
                                                     L_ROOT "." FFN_NORM ".bias",
                                                     L_ROOT "." FFN_IN ".weight",
                                                     L_ROOT "." FFN_IN ".bias",
                                                     L_ROOT "." FFN_OUT ".weight",
                                                     L_ROOT "." FFN_OUT ".bias",
                                                     L_ROOT ".amaxList",
                                                     L_ROOT ".h_amaxList"};

const std::vector<std::string> pre_layer_weight_names  = {"transformer.embeddings.patch_embeddings.weight",
                                                          "transformer.embeddings.patch_embeddings.bias",
                                                          "transformer.embeddings.cls_token",
                                                          "transformer.embeddings.position_embeddings"};
const std::vector<std::string> post_layer_weight_names = {"transformer.encoder.encoder_norm.weight",
                                                          "transformer.encoder.encoder_norm.bias"};

nvinfer1::PluginFieldType getFieldCollectionTypeINT8(std::string name, const nvinfer1::PluginFieldCollection* fc)
{
    for (int i = 0; i < fc->nbFields; i++) {
        std::string field_name(fc->fields[i].name);
        if (field_name.compare(name) == 0) {
            return fc->fields[i].type;
        }
    }
    return nvinfer1::PluginFieldType::kUNKNOWN;
}

template<typename T>
void loadWeightsPtrINT8(std::vector<const T*>&                 w,
                        const nvinfer1::PluginFieldCollection* fc,
                        int                                    layer_num,
                        bool                                   with_cls_token = true)
{
    int idx = 0;
    for (auto& name : pre_layer_weight_names) {
        if (!with_cls_token && name == "transformer.embeddings.cls_token") {
            continue;
        }

        for (int i = 0; i < fc->nbFields; i++) {
            std::string field_name(fc->fields[i].name);
            if (field_name.compare(name) == 0) {
                w[idx++] = (const T*)fc->fields[i].data;
            }
        }
    }

    for (int i = 0; i < layer_num; i++) {
        for (auto& name : layer_weight_names) {
            char str_buf[1024];
            sprintf(str_buf, name, i);

            for (int j = 0; j < fc->nbFields; j++) {
                std::string field_name(fc->fields[j].name);
                if (field_name.compare(str_buf) == 0) {
                    w[idx++] = (const T*)fc->fields[j].data;
                }
            }
        }
    }

    for (auto& name : post_layer_weight_names) {
        for (int i = 0; i < fc->nbFields; i++) {
            std::string field_name(fc->fields[i].name);
            if (field_name.compare(name) == 0) {
                w[idx++] = (const T*)fc->fields[i].data;
            }
        }
    }

    FT_CHECK(idx == w.size());
}

IPluginV2* VisionTransformerINT8PluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    int max_batch;
    int img_size;
    int patch_size;
    int in_chans;
    int embed_dim;
    int num_heads;
    int inter_size;
    int layer_num;
    int int8_mode;
    int with_cls_token = true;

    std::map<std::string, int*> name2pint = {{"max_batch", &max_batch},
                                             {"img_size", &img_size},
                                             {"patch_size", &patch_size},
                                             {"in_chans", &in_chans},
                                             {"embed_dim", &embed_dim},
                                             {"num_heads", &num_heads},
                                             {"inter_size", &inter_size},
                                             {"layer_num", &layer_num},
                                             {"int8_mode", &int8_mode},
                                             {"with_cls_token", &with_cls_token}};

    for (int i = 0; i < fc->nbFields; i++) {
        auto iter = name2pint.find(fc->fields[i].name);
        if (iter != name2pint.end()) {
            *(iter->second) = *((int*)fc->fields[i].data);
            printf("name=%s, value=%d\n", iter->first.c_str(), *((int*)fc->fields[i].data));
            continue;
        }
    }

    size_t weights_num =
        pre_layer_weight_names.size() + post_layer_weight_names.size() + layer_num * layer_weight_names.size();

    auto weights_type = getFieldCollectionTypeINT8(pre_layer_weight_names[0], fc);

    std::vector<const half*>  w_fp16;
    std::vector<const float*> w_fp32;
    IPluginV2*                p;
    switch (weights_type) {
        case nvinfer1::PluginFieldType::kFLOAT16:
            w_fp16.resize(weights_num);
            loadWeightsPtrINT8(w_fp16, fc, layer_num);
            p = new VisionTransformerINT8Plugin<half>(name,
                                                      max_batch,
                                                      img_size,
                                                      patch_size,
                                                      in_chans,
                                                      embed_dim,
                                                      num_heads,
                                                      inter_size,
                                                      layer_num,
                                                      1.0,
                                                      with_cls_token,
                                                      int8_mode,
                                                      w_fp16);

            break;
        case nvinfer1::PluginFieldType::kFLOAT32:
            w_fp32.resize(weights_num);
            loadWeightsPtrINT8(w_fp32, fc, layer_num);
            p = new VisionTransformerINT8Plugin<float>(name,
                                                       max_batch,
                                                       img_size,
                                                       patch_size,
                                                       in_chans,
                                                       embed_dim,
                                                       num_heads,
                                                       inter_size,
                                                       layer_num,
                                                       1.0,
                                                       with_cls_token,
                                                       int8_mode,
                                                       w_fp32);
            break;
        default:
            FT_CHECK_WITH_INFO(false, "[VisionTransformerINT8PluginCreator::createPlugin] unsupported datatype.");
    }

    return p;
}

IPluginV2* VisionTransformerINT8PluginCreator::deserializePlugin(const char* name,
                                                                 const void* serialData,
                                                                 size_t      serialLength) noexcept
{
    int type_id;
    ::memcpy(&type_id, serialData, sizeof(int));
    char* modelData = (char*)serialData + sizeof(int);

    // This object will be deleted when the network is destroyed, which will
    // call VisionTransformerINT8Plugin::destroy()
    if (type_id == 0)
        return new VisionTransformerINT8Plugin<float>(name, modelData, serialLength);
    else if (type_id == 1)
        return new VisionTransformerINT8Plugin<half>(name, modelData, serialLength);
    else {
        FT_LOG_ERROR("[VisionTransformerINT8PluginCreator::deserializePlugin] unsupported data type %d\n", type_id);
        FT_CHECK(false);
    }
}

void VisionTransformerINT8PluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    namespace_ = libNamespace;
}

const char* VisionTransformerINT8PluginCreator::getPluginNamespace() const noexcept
{
    return namespace_.c_str();
}

template class VisionTransformerINT8Plugin<half>;
template class VisionTransformerINT8Plugin<float>;

}  // namespace fastertransformer
