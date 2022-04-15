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

#include <cuda.h>

#ifndef SWIN_TRANSFORMER_PLUGIN_H
#define SWIN_TRANSFORMER_PLUGIN_H

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntime.h"
#include "cublas_v2.h"
#include <string>
#include <vector>

#include "src/fastertransformer/models/vit/ViT.h"
#include "src/fastertransformer/utils/allocator.h"

namespace fastertransformer {

namespace {
static const char* VIT_PLUGIN_VERSION{"1"};
static const char* VIT_PLUGIN_NAME{"CustomVisionTransformerPlugin"};
}  // namespace

struct ViTSettings {
    size_t max_batch_size = 32;
    size_t img_size = 224;
    size_t chn_num = 3;
    size_t patch_size = 16;
    size_t embed_dim = 768;
    size_t head_num = 12;
    size_t inter_size = embed_dim * 4;
    size_t num_layer = 12;
    bool with_cls_token = true;
    bool is_fp16 = false;
    int sm = -1;
    float q_scaling = 1.0f;
    AttentionType attention_type = AttentionType::UNFUSED_MHA;
    // runtime param
    size_t seq_len = 0;
};

template<typename T>
class VisionTransformerPlugin: public nvinfer1::IPluginV2DynamicExt {
private:
    const std::string layer_name_;
    std::string namespace_;

    cublasHandle_t cublas_handle_ = nullptr;
    cublasLtHandle_t cublaslt_handle_ = nullptr;
    cudnnHandle_t cudnn_handle_ = nullptr;
    ViTWeight<T>* params_ = nullptr;
    ViTTransformer<T>* vit_transformer_ = nullptr;
    std::mutex* cublasWrapperMutex_ = nullptr;
    cublasAlgoMap* cublasAlgoMap_ = nullptr;
    fastertransformer::Allocator<AllocatorType::CUDA>* allocator_ = nullptr;
    cublasMMWrapper* cublas_wrapper_ = nullptr;

    ViTSettings settings_;

public:
    int sm_;
    VisionTransformerPlugin(const std::string& name,
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
                            const std::vector<const T*>& w);

    VisionTransformerPlugin(const std::string& name, const void* data, size_t length);
    VisionTransformerPlugin(const VisionTransformerPlugin<T>& plugin);
    VisionTransformerPlugin() = delete;

    ~VisionTransformerPlugin();

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex,
                                            const nvinfer1::DimsExprs* inputs,
                                            int nbInputs,
                                            nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int pos,
                                   const nvinfer1::PluginTensorDesc* inOut,
                                   int nbInputs,
                                   int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                         int nbInputs,
                         const nvinfer1::DynamicPluginTensorDesc* out,
                         int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                            int nbInputs,
                            const nvinfer1::PluginTensorDesc* outputs,
                            int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                const nvinfer1::PluginTensorDesc* outputDesc,
                const void* const* inputs,
                void* const* outputs,
                void* workspace,
                cudaStream_t stream) noexcept override;

    // IPluginV2Ext Methods
    nvinfer1::DataType
    getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    // IPluginV2 Methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    void Init(const std::vector<const T*>& w);
};

class VisionTransformerPluginCreator: public nvinfer1::IPluginCreator {
public:
    VisionTransformerPluginCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2*
    deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string namespace_;
};

}  // namespace fastertransformer
#endif  // TRT_SWIN_TRANSFORMER_H
