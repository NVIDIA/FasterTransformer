/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#define ENABLE_FP8

#include "NvInferPlugin.h"
#include <NvInfer.h>
#include <cublasLt.h>
#include <cublas_v2.h>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#include <assert.h>
#include <stdio.h>

#include <iostream>
#include <string>

#include "src/fastertransformer/models/bert_fp8/BertFP8.h"

namespace fastertransformer {

using bf16_t = __nv_bfloat16;
using fp8_t  = __nv_fp8_e4m3;

struct CublasCtx {
    cudaStream_t                                           mStream;
    cublasHandle_t                                         mCublasHandle;
    cublasLtHandle_t                                       mCublasLtHandle;
    std::unique_ptr<fastertransformer::cublasFP8MMWrapper> mCublasWrapper;
    std::unique_ptr<fastertransformer::cublasAlgoMap>      mCublasAlgoMap;
    std::mutex                                             mCublasWrapperMutex;

    CublasCtx(fastertransformer::Allocator<fastertransformer::AllocatorType::CUDA>* allocator)
    {
        cudaStreamCreate(&mStream);
        cublasCreate(&mCublasHandle);
        cublasLtCreate(&mCublasLtHandle);
        cublasSetStream(mCublasHandle, mStream);
        mCublasAlgoMap.reset(new fastertransformer::cublasAlgoMap("gemm_config.in", ""));
        mCublasWrapper.reset(new fastertransformer::cublasFP8MMWrapper(
            mCublasHandle, mCublasLtHandle, mStream, mCublasAlgoMap.get(), &mCublasWrapperMutex, allocator));
    }

    ~CublasCtx()
    {
        cudaStreamDestroy(mStream);
        cublasDestroy(mCublasHandle);
        cublasLtDestroy(mCublasLtHandle);
    }
};

struct ScopedCudaEvent {
    ScopedCudaEvent()
    {
        FT_CHECK(cudaEventCreate(&mCudaEvent) == cudaSuccess);
    }
    ~ScopedCudaEvent()
    {
        FT_CHECK(cudaEventDestroy(mCudaEvent) == cudaSuccess);
    }
    cudaEvent_t get()
    {
        return mCudaEvent;
    }

    cudaEvent_t mCudaEvent;
};

template<typename T>
void write(uint8_t*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template<typename T>
void read(const uint8_t*& buffer, T& val)
{
    val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
}

struct BertFp8Config {
    BertFp8Config() = default;

    BertFp8Config(int32_t num_heads,
                  int32_t size_per_head,
                  int32_t num_layers,
                  int32_t max_seq_len,
                  int32_t vocab_size,
                  int32_t max_position_embeddings,
                  int32_t token_type_vocab_size,
                  int32_t remove_padding,
                  int32_t fp8_mode):
        num_heads(num_heads),
        size_per_head(size_per_head),
        num_layers(num_layers),
        max_seq_len(max_seq_len),
        vocab_size(vocab_size),
        max_position_embeddings(max_position_embeddings),
        token_type_vocab_size(token_type_vocab_size),
        remove_padding(remove_padding),
        fp8_mode(fp8_mode),
        hidden_units(num_heads * size_per_head),
        intermediate_size(4 * hidden_units)
    {
    }

    BertFp8Config(const uint8_t*& buffer)
    {
        read(buffer, num_heads);
        read(buffer, size_per_head);
        read(buffer, num_layers);
        read(buffer, max_seq_len);
        read(buffer, vocab_size);
        read(buffer, max_position_embeddings);
        read(buffer, token_type_vocab_size);
        read(buffer, remove_padding);
        read(buffer, fp8_mode);
        hidden_units      = num_heads * size_per_head;
        intermediate_size = 4 * hidden_units;
    }

    void serialize(uint8_t*& buffer) const
    {
        write(buffer, num_heads);
        write(buffer, size_per_head);
        write(buffer, num_layers);
        write(buffer, max_seq_len);
        write(buffer, vocab_size);
        write(buffer, max_position_embeddings);
        write(buffer, token_type_vocab_size);
        write(buffer, remove_padding);
        write(buffer, fp8_mode);
    }

    size_t getSerializationSize() const
    {
        return 9 * sizeof(int32_t);
    }

    int32_t num_heads;
    int32_t size_per_head;
    int32_t num_layers;
    int32_t max_seq_len;
    int32_t vocab_size;
    int32_t max_position_embeddings;
    int32_t token_type_vocab_size;
    int32_t remove_padding;
    int32_t fp8_mode;
    int32_t hidden_units;
    int32_t intermediate_size;
};

class BertFp8Plugin: public nvinfer1::IPluginV2DynamicExt {
public:
    BertFp8Plugin(BertFp8Config cfg, std::string weightDirPath);
    BertFp8Plugin(const nvinfer1::PluginFieldCollection* fc);
    BertFp8Plugin(const void* data, size_t length);
    ~BertFp8Plugin() noexcept override = default;

    // IPluginV2Ext fields
    const char*  getPluginType() const noexcept override;
    const char*  getPluginVersion() const noexcept override;
    void         setPluginNamespace(const char* libNamespace) noexcept override;
    const char*  getPluginNamespace() const noexcept override;
    void         destroy() noexcept override;
    int          getNbOutputs() const noexcept override;
    int          initialize() noexcept override;
    virtual void terminate() noexcept override;
    size_t       getSerializationSize() const noexcept override;
    void         serialize(void* buffer) const noexcept override;
    nvinfer1::DataType
    getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    // IPluginV2Ext fields
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    void                           configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                                                   int                                      nbInputs,
                                                   const nvinfer1::DynamicPluginTensorDesc* out,
                                                   int                                      nbOutputs) noexcept override;
    bool                           supportsFormatCombination(int                               pos,
                                                             const nvinfer1::PluginTensorDesc* inOut,
                                                             int                               nbInputs,
                                                             int                               nbOutputs) noexcept override;
    nvinfer1::DimsExprs            getOutputDimensions(int                        outputIndex,
                                                       const nvinfer1::DimsExprs* inputs,
                                                       int                        nbInputs,
                                                       nvinfer1::IExprBuilder&    exprBuilder) noexcept override;
    virtual size_t                 getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                                                    int                               nbInputs,
                                                    const nvinfer1::PluginTensorDesc* outputs,
                                                    int                               nbOutputs) const noexcept override;
    virtual int                    enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                           const nvinfer1::PluginTensorDesc* outputDesc,
                                           const void* const*                inputs,
                                           void* const*                      outputs,
                                           void*                             workspace,
                                           cudaStream_t                      stream) noexcept override;

    virtual void attachToContext(cudnnContext*, cublasContext*, nvinfer1::IGpuAllocator*) noexcept override;

    void initModel();

    template<typename T>
    void write(char*& buffer, const T& val) const;

    template<typename T>
    void read(const char*& buffer, T& val) const;

private:
    // create in constructor
    BertFp8Config                mBertFp8Config;
    fastertransformer::NcclParam mTensorPara;
    fastertransformer::NcclParam mPipelinePara;
    std::string                  mWeightDirPath;
    ScopedCudaEvent              mSyncEvent;
    bool                         mInitialized{false};
    bool                         mModelCreated{false};
    std::string                  mNamespace;

    // defer creation until initialize()
    std::unique_ptr<fastertransformer::Allocator<fastertransformer::AllocatorType::CUDA>> mAllocator;
    std::unique_ptr<CublasCtx> mCublasCtx;  // needs to be destroyed before mAllocator
    // can share weights among different execution contexts
    std::shared_ptr<fastertransformer::BertFP8Weight<fp8_t, bf16_t>> mBertWeights;
    std::unique_ptr<fastertransformer::BertFP8<fp8_t, bf16_t>>       mBertModel;
};

class BertFp8PluginCreator: public nvinfer1::IPluginCreator {
public:
    BertFp8PluginCreator() = default;

    ~BertFp8PluginCreator() override = default;

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    void setPluginNamespace(const char* libNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

    nvinfer1::IPluginV2DynamicExt* createPlugin(const char*                            name,
                                                const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2DynamicExt*
    deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    std::string mNamespace;
};

}  // namespace fastertransformer
