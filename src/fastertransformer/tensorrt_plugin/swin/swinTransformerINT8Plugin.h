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

#include "examples/cpp/swin/functions.h"
#include "src/fastertransformer/models/swin_int8/SwinINT8.h"
#include "src/fastertransformer/utils/allocator.h"
using namespace std;

namespace fastertransformer {

namespace {
static const char* SWIN_TRANSFORMER_PLUGIN_VERSION{"1"};
static const char* SWIN_TRANSFORMER_PLUGIN_NAME{"CustomSwinTransformerINT8Plugin"};
}  // namespace

inline nvinfer1::DataType fieldTypeToDataType(const nvinfer1::PluginFieldType ftype)
{
    switch (ftype) {
        case nvinfer1::PluginFieldType::kFLOAT32: {
            return nvinfer1::DataType::kFLOAT;
        }
        case nvinfer1::PluginFieldType::kFLOAT16: {
            return nvinfer1::DataType::kHALF;
        }
        case nvinfer1::PluginFieldType::kINT32: {
            return nvinfer1::DataType::kINT32;
        }
        case nvinfer1::PluginFieldType::kINT8: {
            return nvinfer1::DataType::kINT8;
        }
        default:
            throw std::invalid_argument("No corresponding datatype for plugin field type");
    }
}

template<typename T>
class SwinTransformerINT8Plugin: public nvinfer1::IPluginV2DynamicExt {
private:
    const std::string layer_name_;
    std::string namespace_;

    std::vector<T*> weights_;         // in device memory
    std::vector<float*> d_amaxlist_;  // in device memory
    std::vector<float*> h_amaxlist_;  // in host memory
    std::vector<size_t> weight_size_;
    cublasHandle_t cublas_handle_ = nullptr;
    cublasLtHandle_t cublaslt_handle_ = nullptr;
    cudnnHandle_t cudnn_handle_ = nullptr;
    SwinTransformerINT8Weight<T> params_;
    SwinTransformerINT8<T>* swin_transformer_ = nullptr;
    std::mutex* cublasWrapperMutex_ = nullptr;
    cublasAlgoMap* cublasAlgoMap_ = nullptr;
    fastertransformer::Allocator<AllocatorType::CUDA>* allocator_ = nullptr;
    int int8_mode_;
    int output_dim_;
    int weight_num_;
    int max_batch_size_;
    int img_size_;
    int patch_size_;
    int in_chans_;
    int embed_dim_;
    int window_size_;
    bool ape_;
    int patch_norm_;
    int layer_num_;
    float mlp_ratio_;
    bool qkv_bias_;
    float qk_scale_;
    int* depths_;
    int* num_heads_;

public:
    int sm_;
    SwinTransformerINT8Plugin(const std::string& name,
                              const int int8_mode,
                              const int max_batch,
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
                              const std::vector<float*>& h_amax);

    SwinTransformerINT8Plugin(const std::string& name,
                              const int int8_mode,
                              const int max_batch,
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
                              const std::vector<nvinfer1::Weights>& w,
                              const std::vector<nvinfer1::Weights>& d_amax,
                              const std::vector<nvinfer1::Weights>& h_amax);

    SwinTransformerINT8Plugin(const std::string& name, const void* data, size_t length);

    // It doesn't make sense to make SwinTransformerINT8Plugin without arguments, so we
    // delete default constructor.
    SwinTransformerINT8Plugin() = delete;

    ~SwinTransformerINT8Plugin();

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

    // Host To Device
    static T* cudaMallocAndCopy(vector<T*>& weights, const nvinfer1::Weights& w)
    {
        T* dpWeight;
        size_t nValue = w.count;
        check_cuda_error(cudaMalloc(&dpWeight, nValue * sizeof(T)));
        check_cuda_error(cudaMemcpy(dpWeight, w.values, nValue * sizeof(T), cudaMemcpyHostToDevice));

        weights.push_back(dpWeight);
        return dpWeight;
    }

    // Device to Device
    static T* cudaMallocAndCopy(vector<T*>& weights,
                                const T* dpWeightOld,
                                const vector<size_t>& weight_size,
                                const int weight_idx,
                                bool is_float = false)
    {
        T* dpWeight;
        size_t nValue = weight_size[weight_idx];
        check_cuda_error(cudaMalloc((void**)&dpWeight, nValue * sizeof(T)));
        check_cuda_error(cudaMemcpy(dpWeight, dpWeightOld, nValue * sizeof(T), cudaMemcpyDeviceToDevice));
        weights.push_back(dpWeight);
        return dpWeight;
    }

    static float* d_amaxCopy(vector<float*>& d_amaxlist, const nvinfer1::Weights& w)
    {
        float* dpWeight;
        size_t nValue = w.count;
        check_cuda_error(cudaMalloc(&dpWeight, nValue * sizeof(float)));
        check_cuda_error(cudaMemcpy(dpWeight, w.values, nValue * sizeof(float), cudaMemcpyHostToDevice));

        d_amaxlist.push_back(dpWeight);
        return dpWeight;
    }

    // Device to Device
    static float*
    d_amaxCopy(vector<float*>& d_amaxlist, const float* dpWeightOld, size_t weight_size, const int weight_idx)
    {
        float* dpWeight;
        size_t nValue = weight_size;
        // printf("[DEVICE]dpWeight is float, allocate %ld val, each %lu bytes\n", nValue, sizeof(float));
        check_cuda_error(cudaMalloc((void**)&dpWeight, nValue * sizeof(float)));
        check_cuda_error(cudaMemcpy(dpWeight, dpWeightOld, nValue * sizeof(float), cudaMemcpyDeviceToDevice));

        d_amaxlist.push_back(dpWeight);
        return dpWeight;
    }

    // Host To Host
    static float* h_amaxCopy(vector<float*>& h_amaxlist, const nvinfer1::Weights& w)
    {
        float* dpWeight;
        size_t nValue = w.count;
        // printf("[Host]tmp is float, malloc %ld val, each %lu bytes\n", nValue, sizeof(float));
        dpWeight = (float*)malloc(nValue * sizeof(float));
        check_cuda_error(cudaMemcpy(dpWeight, w.values, nValue * sizeof(float), cudaMemcpyHostToHost));

        h_amaxlist.push_back(dpWeight);
        return dpWeight;
    }

    // Host To Host
    static float*
    h_amaxCopy(vector<float*>& h_amaxlist, const float* dpWeightOld, size_t weight_size, const int weight_idx)
    {
        float* dpWeight;
        size_t nValue = weight_size;
        // printf("[Host]dpWeight is float, malloc %ld val, each %lu bytes\n", nValue, sizeof(float));
        dpWeight = (float*)malloc(nValue * sizeof(float));
        check_cuda_error(cudaMemcpy(dpWeight, dpWeightOld, nValue * sizeof(float), cudaMemcpyHostToHost));

        h_amaxlist.push_back(dpWeight);
        return dpWeight;
    }

    /*protected:
        // To prevent compiler warnings.
        using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
        using nvinfer1::IPluginV2DynamicExt::configurePlugin;
        using nvinfer1::IPluginV2DynamicExt::enqueue;
        using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
        using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
        using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
        using nvinfer1::IPluginV2DynamicExt::supportsFormat;*/
};

class SwinTransformerINT8PluginCreator: public nvinfer1::IPluginCreator {
public:
    SwinTransformerINT8PluginCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2*
    deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

    static void
    getWeightsFromFC(std::string name, const nvinfer1::PluginFieldCollection* fc, std::vector<nvinfer1::Weights>& w)
    {
        for (int i = 0; i < fc->nbFields; i++) {
            std::string field_name(fc->fields[i].name);
            if (field_name.compare(name) == 0) {
                nvinfer1::Weights tmp;
                tmp.values = fc->fields[i].data;
                tmp.count = fc->fields[i].length;
                tmp.type = fieldTypeToDataType(fc->fields[i].type);
                w.push_back(tmp);
                break;
            }
        }
    }

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string namespace_;
};

}  // namespace fastertransformer
#endif  // TRT_SWIN_TRANSFORMER_H
