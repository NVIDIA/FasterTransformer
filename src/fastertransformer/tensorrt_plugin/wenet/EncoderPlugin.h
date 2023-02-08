/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2022.  Authored by Yuqing Ding.
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

#include "src/fastertransformer/models/wenet/WenetEncoder.h"
#include "src/fastertransformer/models/wenet/WenetEncoderWeight.h"
#include "src/fastertransformer/utils/Tensor.h"
#include <NvInfer.h>
#include <cstdio>
#include <cstring>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <vector>

#ifndef DEBUG_ENABLE
#define DEBUG_ENABLE 0
#endif
#if DEBUG_ENABLE == 1
#define WHERE_AM_I() printf("[%s]: this->%p\n", __func__, this);
#define PRINT_ENCODER(DATA_TYPE)                                                                                       \
    printf("[encoder::%s]Info:\n\tdatatype=%d\n", __func__, DATA_TYPE);                                                \
    printf("\tmax_batch_size=%ld\n", m_.max_batch_size);                                                               \
    printf("\tmax_seq_len=%ld\n", m_.max_seq_len);                                                                     \
    printf("\thead_num=%ld\n", m_.head_num);                                                                           \
    printf("\tsize_per_head=%ld\n", m_.size_per_head);                                                                 \
    printf("\tfeature_size=%ld\n", m_.feature_size);                                                                   \
    printf("\tmax_len=%ld\n", m_.max_len);                                                                             \
    printf("\tinter_size=%ld\n", m_.inter_size);                                                                       \
    printf("\td_model=%ld\n", m_.d_model);                                                                             \
    printf("\tnum_layer=%ld\n", m_.num_layer);                                                                         \
    printf("\tsm=%d\n", m_.sm);                                                                                        \
    printf("\tq_scaling=%f\n", m_.q_scaling);                                                                          \
    printf("\tuse_layernorm_in_cnn_module=%d\n", m_.use_layernorm_in_cnn_module);                                      \
    printf("\tuseFP16=%d\n", m_.useFP16);                                                                              \
    printf("\tweightFilePath=%s\n", m_.weightFilePath);                                                                \
    printf("\tvocab_size=%ld\n", m_.vocab_size);                                                                       \
    printf("\tis_remove_padding=%d\n", m_.is_remove_padding);                                                          \
    printf("\tis_free_buffer_after_forward=%d\n", m_.is_free_buffer_after_forward);                                    \
    printf("\tis_sparse=%d\n", m_.is_sparse);                                                                          \
    printf("\tattention_type=%d\n", m_.attention_type);                                                                \
    printf("\tactivation_type=%d\n", m_.activation_type);                                                              \
    printf("\tlayernorm_type=%d\n", m_.layernorm_type);                                                                \
    printf("\tbatch_size=%ld\n", m_.batch_size);                                                                       \
    printf("\tseq_len=%ld\n", m_.seq_len);

#else
#define WHERE_AM_I()
#define PRINT_ENCODER(DATA_TYPE)
#define PRINT_DECODING(DATA_TYPE)
#endif  // DEBUG_ENABLE==1

namespace {
static const char* ENCODER_NAME{"WenetEncoderPlugin"};
static const char* ENCODER_VERSION{"1"};
}  // namespace

using namespace fastertransformer;

namespace nvinfer1 {

// class WenetEncoderPlugin ---------------------------------------------------------------------------
class WenetEncoderPlugin: public IPluginV2DynamicExt {
private:
    using IPluginV2Ext::configurePlugin;
    using IPluginV2::getOutputDimensions;
    using IPluginV2::getWorkspaceSize;
    using IPluginV2::enqueue;

    const std::string name_;
    std::string       namespace_;
    cublasHandle_t    cublasHandle_;
    cublasLtHandle_t  cublasltHandle_;
#ifdef SPARSITY_ENABLED
    cusparseLtHandle_t cusparseltHandle_;
#endif
    cublasAlgoMap*                  pCublasAlgoMap_           = nullptr;
    std::mutex*                     pCublasWrapperMutex_      = nullptr;
    WenetEncoderWeight<half>*       pWenetEncoderWeightHalf_  = nullptr;
    WenetEncoderWeight<float>*      pWenetEncoderWeightFloat_ = nullptr;
    Allocator<AllocatorType::CUDA>* pAllocator_               = nullptr;
    cublasMMWrapper*                pCublasWrapper_           = nullptr;
    cudnnHandle_t                   cudnn_handle_             = nullptr;
    WenetEncoder<half>*             pWenetEncoderHalf_        = nullptr;
    WenetEncoder<float>*            pWenetEncoderFloat_       = nullptr;
    struct {
        // constructor parameter
        size_t max_batch_size               = 16;
        size_t max_seq_len                  = 256;
        size_t head_num                     = 8;
        size_t size_per_head                = 32;
        size_t feature_size                 = 80;
        size_t max_len                      = 5000;
        size_t inter_size                   = head_num * size_per_head * 4;
        size_t d_model                      = head_num * size_per_head;
        size_t num_layer                    = 12;
        size_t vocab_size                   = 4233;
        size_t conv_module_kernel_size      = 15;
        int    sm                           = -1;  // assign later
        float  q_scaling                    = 1.0f / (1.0f * sqrt(size_per_head));
        bool   use_layernorm_in_conv_module = false;
        bool   useFP16                      = false;
        // internal parameter
        bool                              is_remove_padding            = false;
        bool                              is_free_buffer_after_forward = false;
        bool                              is_sparse                    = false;
        AttentionType                     attention_type               = AttentionType::UNFUSED_MHA;
        fastertransformer::ActivationType activation_type              = fastertransformer::ActivationType::Silu;
        LayerNormType                     layernorm_type               = LayerNormType::pre_layernorm;
        // runtime parameter
        size_t batch_size          = 0;
        size_t seq_len             = 0;
        char   weightFilePath[256] = "";
    } m_;

    void CreateFT();

public:
    WenetEncoderPlugin() = delete;
    WenetEncoderPlugin(const std::string& name,
                       size_t             max_batch_size,
                       size_t             max_seq_len,
                       size_t             head_num,
                       size_t             size_per_head,
                       size_t             feature_size,
                       size_t             max_len,
                       size_t             inter_size,
                       size_t             d_model,
                       size_t             num_layer,
                       size_t             vocab_size,
                       size_t             conv_module_kernel_size,
                       int                sm,
                       float              q_scaling,
                       const std::string& weightFilePath,
                       int                use_layernorm_in_conv_module,
                       int                useFP16);
    WenetEncoderPlugin(const std::string& name, const void* buffer, size_t length);
    ~WenetEncoderPlugin();

    virtual size_t       getSerializationSize() const noexcept override;
    virtual void         serialize(void* buffer) const noexcept override;
    IPluginV2DynamicExt* clone() const noexcept override;
    int                  getNbOutputs() const noexcept override;
    DataType             getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override;
    bool
    supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    DimsExprs    getOutputDimensions(int              index,
                                     const DimsExprs* pInputDim,
                                     int              nInputDim,
                                     IExprBuilder&    exprBuilder) noexcept override;
    virtual void configurePlugin(const DynamicPluginTensorDesc* in,
                                 int                            nbInput,
                                 const DynamicPluginTensorDesc* out,
                                 int                            nbOutput) noexcept override;
    size_t       getWorkspaceSize(const PluginTensorDesc* inputs,
                                  int32_t                 nbInputs,
                                  const PluginTensorDesc* outputs,
                                  int32_t                 nbOutputs) const noexcept override;

    int enqueue(const PluginTensorDesc* inputDesc,
                const PluginTensorDesc* outputDesc,
                const void* const*      inputs,
                void* const*            outputs,
                void*                   workspace,
                cudaStream_t            stream) noexcept override;

    void        setPluginNamespace(const char* szNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int         initialize() noexcept override;
    void        terminate() noexcept override;
    void        destroy() noexcept override;
    void attachToContext(cudnnContext* /*cudnn*/, cublasContext* /*cublas*/, IGpuAllocator* /*allocator*/) noexcept;
    void detachFromContext() noexcept;
};

// class WenetEncoderPluginCreator --------------------------------------------------------------------
class WenetEncoderPluginCreator: public IPluginCreator {
private:
    static PluginFieldCollection    fc_;
    static std::vector<PluginField> attr_;
    std::string                     namespace_;

public:
    WenetEncoderPluginCreator();
    ~WenetEncoderPluginCreator();
    IPluginV2*  createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2*  deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void        setPluginNamespace(const char* szNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;
};

}  // namespace nvinfer1
