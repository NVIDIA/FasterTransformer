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

#include "ViTPlugin.h"
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
PluginFieldCollection VisionTransformerPluginCreator::mFC{};
std::vector<PluginField> VisionTransformerPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(VisionTransformerPluginCreator);

template<typename T>
VisionTransformerPlugin<T>::VisionTransformerPlugin(const std::string& name,
                                                    const int max_batch,
                                                    const int img_size,
                                                    const int patch_size,
                                                    const int in_chans,
                                                    const int embed_dim,
                                                    const int num_heads,
                                                    const int inter_size,
                                                    const int layer_num,
                                                    const float q_scaling,
                                                    const bool with_cls_token,
                                                    const std::vector<const T*>& w):
    layer_name_(name)
{

    settings_.max_batch_size = max_batch;
    settings_.img_size = img_size;
    settings_.chn_num = in_chans;
    settings_.patch_size = patch_size;
    settings_.embed_dim = embed_dim;
    settings_.head_num = num_heads;
    settings_.inter_size = inter_size;
    settings_.num_layer = layer_num;
    settings_.with_cls_token = with_cls_token;
    settings_.sm = getSMVersion();
    settings_.q_scaling = q_scaling;
    settings_.seq_len = (img_size / patch_size) * (img_size / patch_size) + (with_cls_token ? 1 : 0);
    settings_.attention_type = getAttentionType<T>(embed_dim / num_heads, settings_.sm, true, settings_.seq_len);

    Init(w);
}

template<typename T>
VisionTransformerPlugin<T>::VisionTransformerPlugin(const std::string& name, const void* data, size_t length):
    layer_name_(name)
{
    ::memcpy(&settings_, data, sizeof(settings_));
    const char* w_buffer = static_cast<const char*>(data) + sizeof(settings_);

    std::vector<const T*> dummy;
    Init(dummy);

    params_->deserialize(w_buffer);
}

template<typename T>
VisionTransformerPlugin<T>::VisionTransformerPlugin(const VisionTransformerPlugin<T>& plugin):
    layer_name_(plugin.layer_name_), settings_(plugin.settings_)
{
    std::vector<const T*> dummy;
    Init(dummy);
    *params_ = *plugin.params_;
}

template<typename T>
void VisionTransformerPlugin<T>::Init(const std::vector<const T*>& w)
{
    params_ = new ViTWeight<T>(settings_.embed_dim,
                               settings_.inter_size,
                               settings_.num_layer,
                               settings_.img_size,
                               settings_.patch_size,
                               settings_.chn_num,
                               settings_.with_cls_token);

    if (w.size() > 0) {
        size_t weight_num = params_->GetWeightCount();

        if (weight_num != w.size()) {
            printf("[ERROR][VisionTransformerPlugin] weights number %lu does not match expected number %lu!\n",
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

    cublasAlgoMap_ = new cublasAlgoMap("igemm.config", "");
    cublasWrapperMutex_ = new std::mutex();
    allocator_ = new Allocator<AllocatorType::CUDA>(getDevice());

    cublas_wrapper_ =
        new cublasMMWrapper(cublas_handle_, cublaslt_handle_, nullptr, cublasAlgoMap_, cublasWrapperMutex_, allocator_);
    if (std::is_same<T, half>::value) {
        cublas_wrapper_->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper_->setFP32GemmConfig();
    }

    vit_transformer_ = new ViTTransformer<T>(settings_.max_batch_size,
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
                                             0,
                                             cudnn_handle_,
                                             cublas_wrapper_,
                                             allocator_,
                                             false,
                                             settings_.attention_type);
}

template<typename T>
VisionTransformerPlugin<T>::~VisionTransformerPlugin()
{
    check_cuda_error(cublasDestroy(cublas_handle_));
    check_cuda_error(cublasLtDestroy(cublaslt_handle_));
    checkCUDNN(cudnnDestroy(cudnn_handle_));
    delete cublasWrapperMutex_;
    delete cublasAlgoMap_;
    delete vit_transformer_;
    delete allocator_;
    delete params_;
}

// IPluginV2DynamicExt Methods
template<typename T>
nvinfer1::IPluginV2DynamicExt* VisionTransformerPlugin<T>::clone() const noexcept
{

    VisionTransformerPlugin* ret = new VisionTransformerPlugin<T>(*this);
    return ret;
}

template<typename T>
DimsExprs VisionTransformerPlugin<T>::getOutputDimensions(int outputIndex,
                                                          const DimsExprs* inputs,
                                                          int nbInputs,
                                                          IExprBuilder& exprBuilder) noexcept
{
    // Input is B*in_chans*H*W, output should be B*seq_len*embed_dim*1
    assert(outputIndex == 0);
    DimsExprs output;
    output.nbDims = 3;
    output.d[0] = inputs[0].d[0];
    output.d[1] = exprBuilder.constant(settings_.seq_len);
    output.d[2] = exprBuilder.constant(settings_.embed_dim);
    return output;
}

template<typename T>
bool VisionTransformerPlugin<T>::supportsFormatCombination(int pos,
                                                           const PluginTensorDesc* inOut,
                                                           int nbInputs,
                                                           int nbOutputs) noexcept
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
void VisionTransformerPlugin<T>::configurePlugin(const DynamicPluginTensorDesc* in,
                                                 int nbInputs,
                                                 const DynamicPluginTensorDesc* out,
                                                 int nbOutputs) noexcept
{
    assert(nbInputs == 1);
    assert(nbOutputs == 1);
}

template<typename T>
size_t VisionTransformerPlugin<T>::getWorkspaceSize(const PluginTensorDesc* inputs,
                                                    int nbInputs,
                                                    const PluginTensorDesc* outputs,
                                                    int nbOutputs) const noexcept
{
    return 0;
}

// IPluginV2Ext Methods
template<typename T>
nvinfer1::DataType VisionTransformerPlugin<T>::getOutputDataType(int index,
                                                                 const nvinfer1::DataType* inputTypes,
                                                                 int nbInputs) const noexcept
{
    assert(index == 0);
    assert(inputTypes[0] == nvinfer1::DataType::kFLOAT || inputTypes[0] == nvinfer1::DataType::kHALF);
    return inputTypes[0];
}

// IPluginV2 Methods
template<typename T>
const char* VisionTransformerPlugin<T>::getPluginType() const noexcept
{
    return VIT_PLUGIN_NAME;
}

template<typename T>
const char* VisionTransformerPlugin<T>::getPluginVersion() const noexcept
{
    return VIT_PLUGIN_VERSION;
}

template<typename T>
int VisionTransformerPlugin<T>::getNbOutputs() const noexcept
{
    return 1;
}

template<typename T>
int VisionTransformerPlugin<T>::initialize() noexcept
{
    return 0;
}

template<typename T>
void VisionTransformerPlugin<T>::terminate() noexcept
{
}

template<typename T>
size_t VisionTransformerPlugin<T>::getSerializationSize() const noexcept
{

    size_t size = sizeof(int) + sizeof(settings_);
    size += params_->GetSerializeSize();
    return size;
}

template<typename T>
void VisionTransformerPlugin<T>::serialize(void* buffer) const noexcept
{
    printf("start serialize vit...\n");

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
void VisionTransformerPlugin<T>::destroy() noexcept
{
    delete this;
}

template<typename T>
void VisionTransformerPlugin<T>::setPluginNamespace(const char* libNamespace) noexcept
{
    namespace_ = libNamespace;
}

template<typename T>
const char* VisionTransformerPlugin<T>::getPluginNamespace() const noexcept
{
    return namespace_.c_str();
}

template<typename T>
int VisionTransformerPlugin<T>::enqueue(const PluginTensorDesc* inputDesc,
                                        const PluginTensorDesc* outputDesc,
                                        const void* const* inputs,
                                        void* const* outputs,
                                        void* workspace,
                                        cudaStream_t stream) noexcept
{
    int batch_size = inputDesc->dims.d[0];
    assert(batch_size <= settings_.max_batch_size);
    assert(settings_.chn_num == inputDesc->dims.d[1]);
    assert(settings_.img_size == inputDesc->dims.d[2]);
    assert(settings_.img_size == inputDesc->dims.d[3]);

    int sm_ptr[1] = {sm_};
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

VisionTransformerPluginCreator::VisionTransformerPluginCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* VisionTransformerPluginCreator::getPluginName() const noexcept
{
    return VIT_PLUGIN_NAME;
}

const char* VisionTransformerPluginCreator::getPluginVersion() const noexcept
{
    return VIT_PLUGIN_VERSION;
}

const PluginFieldCollection* VisionTransformerPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

// Creator
#define L_ROOT "Transformer/encoderblock_%d"
#define ATT_Q "MultiHeadDotProductAttention_1/query"
#define ATT_K "MultiHeadDotProductAttention_1/key"
#define ATT_V "MultiHeadDotProductAttention_1/value"
#define ATT_OUT "MultiHeadDotProductAttention_1/out"
#define ATT_NORM "LayerNorm_0"
#define FFN_NORM "LayerNorm_2"
#define FFN_IN "MlpBlock_3/Dense_0"
#define FFN_OUT "MlpBlock_3/Dense_1"

const std::vector<const char*> layer_weight_names = {L_ROOT "/" ATT_NORM "/scale",
                                                     L_ROOT "/" ATT_NORM "/bias",
                                                     L_ROOT "/" ATT_Q "/kernel",
                                                     L_ROOT "/" ATT_Q "/bias",
                                                     L_ROOT "/" ATT_K "/kernel",
                                                     L_ROOT "/" ATT_K "/bias",
                                                     L_ROOT "/" ATT_V "/kernel",
                                                     L_ROOT "/" ATT_V "/bias",
                                                     L_ROOT "/" ATT_OUT "/kernel",
                                                     L_ROOT "/" ATT_OUT "/bias",
                                                     L_ROOT "/" FFN_NORM "/scale",
                                                     L_ROOT "/" FFN_NORM "/bias",
                                                     L_ROOT "/" FFN_IN "/kernel",
                                                     L_ROOT "/" FFN_IN "/bias",
                                                     L_ROOT "/" FFN_OUT "/kernel",
                                                     L_ROOT "/" FFN_OUT "/bias"};

const std::vector<std::string> pre_layer_weight_names = {
    "embedding/kernel", "embedding/bias", "cls", "Transformer/posembed_input/pos_embedding"};
const std::vector<std::string> post_layer_weight_names = {"Transformer/encoder_norm/scale",
                                                          "Transformer/encoder_norm/bias"};

nvinfer1::PluginFieldType getFieldCollectionType(std::string name, const nvinfer1::PluginFieldCollection* fc)
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
void loadWeightsPtr(std::vector<const T*>& w,
                    const nvinfer1::PluginFieldCollection* fc,
                    int layer_num,
                    bool with_cls_token = true)
{
    int idx = 0;
    for (auto& name : pre_layer_weight_names) {
        if (!with_cls_token && name == "cls")
            continue;

        for (int i = 0; i < fc->nbFields; i++) {
            std::string field_name(fc->fields[i].name);
            if (field_name.compare(name) == 0) {
                w[idx++] = (const T*)fc->fields[i].data;
            }
        }
    }

    for (int i = 0; i < layer_num; i++){
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

IPluginV2* VisionTransformerPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    int max_batch;
    int img_size;
    int patch_size;
    int in_chans;
    int embed_dim;
    int num_heads;
    int inter_size;
    int layer_num;
    int with_cls_token = true;

    std::map<std::string, int*> name2pint = {{"max_batch", &max_batch},
                                             {"img_size", &img_size},
                                             {"patch_size", &patch_size},
                                             {"in_chans", &in_chans},
                                             {"embed_dim", &embed_dim},
                                             {"num_heads", &num_heads},
                                             {"inter_size", &inter_size},
                                             {"layer_num", &layer_num},
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

    auto weights_type = getFieldCollectionType(pre_layer_weight_names[0], fc);

    std::vector<const half*> w_fp16;
    std::vector<const float*> w_fp32;
    IPluginV2* p;
    switch (weights_type) {
        case nvinfer1::PluginFieldType::kFLOAT16:
            w_fp16.resize(weights_num);
            loadWeightsPtr(w_fp16, fc, layer_num);
            p = new VisionTransformerPlugin<half>(name,
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
                                                  w_fp16);

            break;
        case nvinfer1::PluginFieldType::kFLOAT32:
            w_fp32.resize(weights_num);
            loadWeightsPtr(w_fp32, fc, layer_num);
            p = new VisionTransformerPlugin<float>(name,
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
                                                   w_fp32);
            break;
        default:
            printf("[ERROR][VisionTransformerPluginCreator::createPlugin] unsupport datatype.\n");
            exit(-1);
    }

    return p;
}

IPluginV2* VisionTransformerPluginCreator::deserializePlugin(const char* name,
                                                             const void* serialData,
                                                             size_t serialLength) noexcept
{
    int type_id;
    ::memcpy(&type_id, serialData, sizeof(int));
    char* modelData = (char*)serialData + sizeof(int);

    // This object will be deleted when the network is destroyed, which will
    // call VisionTransformerPlugin::destroy()
    if (type_id == 0)
        return new VisionTransformerPlugin<float>(name, modelData, serialLength);
    else if (type_id == 1)
        return new VisionTransformerPlugin<half>(name, modelData, serialLength);
    else {
        printf("[ERROR][VisionTransformerPluginCreator::deserializePlugin] unsupport data type %d\n", type_id);
        exit(-1);
    }
}

void VisionTransformerPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    namespace_ = libNamespace;
}

const char* VisionTransformerPluginCreator::getPluginNamespace() const noexcept
{
    return namespace_.c_str();
}

template class VisionTransformerPlugin<half>;
template class VisionTransformerPlugin<float>;

}  // namespace fastertransformer
