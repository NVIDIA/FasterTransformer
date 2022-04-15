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

#include "T5PluginGemm.h"
#include "src/fastertransformer/models/t5/T5Decoding.h"
#include "src/fastertransformer/models/t5/T5DecodingWeight.h"
#include "src/fastertransformer/models/t5/T5Encoder.h"
#include "src/fastertransformer/models/t5/T5EncoderWeight.h"
#include "src/fastertransformer/utils/Tensor.h"
#include <NvInfer.h>
#include <cstdio>
#include <cstring>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <vector>

#if DEBUG_ENABLE == 1
#define WHERE_AM_I() printf("[%s]: this->%p\n", __func__, this);
#define PRINT_ENCODER(DATA_TYPE)                                                                                       \
    printf("[encoder::%s]Info:\n\tdatatype=%d\n", __func__, DATA_TYPE);                                                \
    printf("\tmax_batch_size=%d\n", m_.max_batch_size);                                                                \
    printf("\tmax_seq_len=%d\n", m_.max_seq_len);                                                                      \
    printf("\tbeam_width=%d\n", m_.beam_width);                                                                        \
    printf("\thead_num=%d\n", m_.head_num);                                                                            \
    printf("\tsize_per_head=%d\n", m_.size_per_head);                                                                  \
    printf("\td_model=%d\n", m_.d_model);                                                                              \
    printf("\tinter_size=%d\n", m_.inter_size);                                                                        \
    printf("\tnum_layer=%d\n", m_.num_layer);                                                                          \
    printf("\tnum_bucket=%d\n", m_.num_bucket);                                                                        \
    printf("\tmax_distance=%d\n", m_.max_distance);                                                                    \
    printf("\tsm=%d\n", m_.sm);                                                                                        \
    printf("\tq_scaling=%f\n", m_.q_scaling);                                                                          \
    printf("\tuseFP16=%d\n", m_.useFP16);                                                                              \
    printf("\tvocab_size=%d\n", m_.vocab_size);                                                                        \
    printf("\tis_remove_padding=%d\n", m_.is_remove_padding);                                                          \
    printf("\tis_free_buffer_after_forward=%d\n", m_.is_free_buffer_after_forward);                                    \
    printf("\tis_sparse=%d\n", m_.is_sparse);                                                                          \
    printf("\tattention_type=%d\n", m_.attention_type);                                                                \
    printf("\tactivation_type=%d\n", m_.activation_type);                                                              \
    printf("\tlayernorm_type=%d\n", m_.layernorm_type);                                                                \
    printf("\tbatch_size=%d\n", m_.batch_size);                                                                        \
    printf("\tseq_len=%d\n", m_.seq_len);

#define PRINT_DECODING(DATA_TYPE)                                                                                      \
    printf("[decoding::%s]Info:\n\tdatatype=%d\n", __func__, DATA_TYPE);                                               \
    printf("\tmax_batch_size=%d\n", m_.max_batch_size);                                                                \
    printf("\tmax_seq_len=%d\n", m_.max_seq_len);                                                                      \
    printf("\tmem_max_seq_len=%d\n", m_.mem_max_seq_len);                                                              \
    printf("\tbeam_width=%d\n", m_.beam_width);                                                                        \
    printf("\thead_num=%d\n", m_.head_num);                                                                            \
    printf("\tsize_per_head=%d\n", m_.size_per_head);                                                                  \
    printf("\td_model=%d\n", m_.d_model);                                                                              \
    printf("\tinter_size=%d\n", m_.inter_size);                                                                        \
    printf("\tnum_layer=%d\n", m_.num_layer);                                                                          \
    printf("\tvocab_size=%d\n", m_.vocab_size);                                                                        \
    printf("\tnum_bucket=%d\n", m_.num_bucket);                                                                        \
    printf("\tmax_distance=%d\n", m_.max_distance);                                                                    \
    printf("\tq_scaling=%f\n", m_.q_scaling);                                                                          \
    printf("\tstart_id=%d\n", m_.start_id);                                                                            \
    printf("\tend_id=%d\n", m_.end_id);                                                                                \
    printf("\tbeam_search_diversity_rate=%f\n", m_.beam_search_diversity_rate);                                        \
    printf("\ttop_k=%d\n", m_.top_k);                                                                                  \
    printf("\ttop_p=%f\n", m_.top_p);                                                                                  \
    printf("\ttemperature=%f\n", m_.temperature);                                                                      \
    printf("\tlen_penalty=%f\n", m_.len_penalty);                                                                      \
    printf("\trepetition_penalty=%f\n", m_.repetition_penalty);                                                        \
    printf("\tuseFP16=%d\n", m_.useFP16);                                                                              \
    printf("\tmem_d_model=%d\n", m_.mem_d_model);                                                                      \
    printf("\tmem_hidden_units=%d\n", m_.mem_hidden_units);                                                            \
    printf("\tis_free_buffer_after_forward=%d\n", m_.is_free_buffer_after_forward);                                    \
    printf("\tbatch_size=%d\n", m_.batch_size);                                                                        \
    printf("\tseq_len=%d\n", m_.seq_len);

#else
#define WHERE_AM_I()
#define PRINT_ENCODER(DATA_TYPE)
#define PRINT_DECODING(DATA_TYPE)
#endif  // DEBUG_ENABLE==1

namespace {
static const char* ENCODER_NAME{"T5EncoderPlugin"};
static const char* ENCODER_VERSION{"1"};
static const char* DECODING_NAME{"T5DecodingPlugin"};
static const char* DECODING_VERSION{"1"};
}  // namespace

using namespace fastertransformer;

namespace nvinfer1 {

// class T5EncoderPlugin ---------------------------------------------------------------------------
class T5EncoderPlugin: public IPluginV2DynamicExt {
private:
    using IPluginV2Ext::configurePlugin;
    using IPluginV2::getOutputDimensions;
    using IPluginV2::getWorkspaceSize;
    using IPluginV2::enqueue;

    const std::string name_;
    std::string namespace_;
    cublasHandle_t cublasHandle_;
    cublasLtHandle_t cublasltHandle_;
#ifdef SPARSITY_ENABLED
    cusparseLtHandle_t cusparseltHandle_;
#endif
    cublasAlgoMap* pCublasAlgoMap_ = nullptr;
    std::mutex* pCublasWrapperMutex_ = nullptr;
    T5EncoderWeight<half>* pT5EncoderWeightHalf_ = nullptr;
    T5EncoderWeight<float>* pT5EncoderWeightFloat_ = nullptr;
    Allocator<AllocatorType::CUDA>* pAllocator_ = nullptr;
    cublasMMWrapper* pCublasWrapper_ = nullptr;
    T5Encoder<half>* pT5EncoderHalf_ = nullptr;
    T5Encoder<float>* pT5EncoderFloat_ = nullptr;
    struct {
        // constructor parameter
        size_t max_batch_size = 128;
        size_t max_seq_len = 384;
        size_t beam_width = 1;
        size_t head_num = 8;
        size_t size_per_head = 512 / 8;
        size_t d_model = head_num * size_per_head;
        size_t inter_size = d_model * 4;
        size_t num_layer = 6;
        size_t num_bucket = 32;
        size_t max_distance = 128;
        int sm = -1;  // assign later
        float q_scaling = 1.0f / (1.0f * sqrt(size_per_head));
        bool useFP16 = false;
        // internal parameter
        size_t vocab_size = 32128;
        bool is_remove_padding = true;
        bool is_free_buffer_after_forward = false;
        bool is_sparse = false;
        AttentionType attention_type = AttentionType::UNFUSED_MHA;
        fastertransformer::ActivationType activation_type = fastertransformer::ActivationType::Relu;
        LayerNormType layernorm_type = LayerNormType::pre_layernorm;
        // runtime parameter
        size_t batch_size = 0;
        size_t seq_len = 0;
    } m_;

public:
    T5EncoderPlugin() = delete;
    T5EncoderPlugin(const std::string& name,
                    size_t max_batch_size,
                    size_t max_seq_len,
                    size_t beam_width,
                    size_t head_num,
                    size_t size_per_head,
                    size_t inter_size,
                    size_t d_model,
                    size_t num_layer,
                    size_t num_bucket,
                    size_t max_distance,
                    int sm,
                    float q_scaling,
                    int useFP16);
    T5EncoderPlugin(const std::string& name, const void* buffer, size_t length);
    ~T5EncoderPlugin();

    virtual size_t getSerializationSize() const noexcept override;
    virtual void serialize(void* buffer) const noexcept override;
    IPluginV2DynamicExt* clone() const noexcept override;
    int getNbOutputs() const noexcept override;
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override;
    bool
    supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    DimsExprs getOutputDimensions(int index,
                                  const DimsExprs* pInputDim,
                                  int nInputDim,
                                  IExprBuilder& exprBuilder) noexcept override;
    virtual void configurePlugin(const DynamicPluginTensorDesc* in,
                                 int nbInput,
                                 const DynamicPluginTensorDesc* out,
                                 int nbOutput) noexcept override;
    size_t getWorkspaceSize(const PluginTensorDesc* inputs,
                            int32_t nbInputs,
                            const PluginTensorDesc* outputs,
                            int32_t nbOutputs) const noexcept override;
    int enqueue(const PluginTensorDesc* inputDesc,
                const PluginTensorDesc* outputDesc,
                const void* const* inputs,
                void* const* outputs,
                void* workspace,
                cudaStream_t stream) noexcept override;
    void setPluginNamespace(const char* szNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    void destroy() noexcept override;
    void attachToContext(cudnnContext* /*cudnn*/, cublasContext* /*cublas*/, IGpuAllocator* /*allocator*/) noexcept;
    void detachFromContext() noexcept;
};

// class T5EncoderPluginCreator --------------------------------------------------------------------
class T5EncoderPluginCreator: public IPluginCreator {
private:
    static PluginFieldCollection fc_;
    static std::vector<PluginField> attr_;
    std::string namespace_;

public:
    T5EncoderPluginCreator();
    ~T5EncoderPluginCreator();
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* szNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;
};
// class T5DecodingPlugin --------------------------------------------------------------------------
class T5DecodingPlugin: public IPluginV2DynamicExt {
private:
    using IPluginV2Ext::configurePlugin;
    using IPluginV2::getOutputDimensions;
    using IPluginV2::getWorkspaceSize;
    using IPluginV2::enqueue;

    const std::string name_;
    std::string namespace_;
    cublasHandle_t cublasHandle_;
    cublasLtHandle_t cublasltHandle_;
    cudaDeviceProp cuda_device_prop_;
    cublasAlgoMap* pCublasAlgoMap_ = nullptr;
    Allocator<AllocatorType::CUDA>* pAllocator_ = nullptr;
    std::mutex* pCublasWrapperMutex_ = nullptr;
    cublasMMWrapper* pCublasWrapper_ = nullptr;
    T5DecodingWeight<half>* pT5DecodingWeightHalf_ = nullptr;
    T5DecodingWeight<float>* pT5DecodingWeightFloat_ = nullptr;
    T5Decoding<half>* pT5DecodingHalf_ = nullptr;
    T5Decoding<float>* pT5DecodingFloat_ = nullptr;
    struct {
        // constructor parameter
        size_t max_batch_size = 128;
        size_t max_seq_len = 384;
        size_t mem_max_seq_len = max_seq_len;
        size_t beam_width = 4;
        size_t head_num = 8;
        size_t size_per_head = 512 / 8;
        size_t d_model = head_num * size_per_head;
        size_t inter_size = d_model * 4;
        size_t num_layer = 6;
        size_t vocab_size = 32128;
        size_t num_bucket = 32;
        size_t max_distance = 128;
        float q_scaling = 1.0f / (1.0f * sqrt(size_per_head));
        int start_id = 0;
        int end_id = 1;
        float beam_search_diversity_rate = 0.0f;
        size_t top_k = beam_width;
        float top_p = 0.0f;
        float temperature = 1.0f;
        float len_penalty = 2.0f;
        float repetition_penalty = 1.0f;
        bool useFP16 = false;
        // internal parameter
        size_t mem_d_model = d_model;
        size_t mem_hidden_units = d_model;
        bool is_free_buffer_after_forward = false;
        // runtime parameter
        size_t batch_size = 128;
        size_t seq_len = 384;
    } m_;

public:
    T5DecodingPlugin() = delete;
    T5DecodingPlugin(const std::string& name,
                     size_t max_batch_size,
                     size_t max_seq_len,
                     size_t mem_max_seq_len,
                     size_t beam_width,
                     size_t head_num,
                     size_t size_per_head,
                     size_t inter_size,
                     size_t d_model,
                     size_t num_layer,
                     size_t vocab_size,
                     size_t num_bucket,
                     size_t max_distance,
                     float q_scaling,
                     int start_id,
                     int end_id,
                     float beam_search_diversity_rate,
                     size_t top_k,
                     float top_p,
                     float temperature,
                     float len_penalty,
                     float repetition_penalty,
                     int useFP16);
    T5DecodingPlugin(const std::string& name, const void* buffer, size_t length);
    ~T5DecodingPlugin();

    virtual size_t getSerializationSize() const noexcept override;
    virtual void serialize(void* buffer) const noexcept override;
    IPluginV2DynamicExt* clone() const noexcept override;
    int getNbOutputs() const noexcept override;
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override;
    bool
    supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    DimsExprs getOutputDimensions(int index,
                                  const DimsExprs* pInputDim,
                                  int nInputDim,
                                  IExprBuilder& exprBuilder) noexcept override;
    virtual void configurePlugin(const DynamicPluginTensorDesc* in,
                                 int nbInput,
                                 const DynamicPluginTensorDesc* out,
                                 int nbOutput) noexcept override;
    size_t getWorkspaceSize(const PluginTensorDesc* inputs,
                            int32_t nbInputs,
                            const PluginTensorDesc* outputs,
                            int32_t nbOutputs) const noexcept override;
    int enqueue(const PluginTensorDesc* inputDesc,
                const PluginTensorDesc* outputDesc,
                const void* const* inputs,
                void* const* outputs,
                void* workspace,
                cudaStream_t stream) noexcept override;
    void setPluginNamespace(const char* szNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    void destroy() noexcept override;
    void attachToContext(cudnnContext* /*cudnn*/, cublasContext* /*cublas*/, IGpuAllocator* /*allocator*/) noexcept;
    void detachFromContext() noexcept;
};

// class T5DecodingPluginCreator -------------------------------------------------------------------
class T5DecodingPluginCreator: public IPluginCreator {
private:
    static PluginFieldCollection fc_;
    static std::vector<PluginField> attr_;
    std::string namespace_;

public:
    T5DecodingPluginCreator();
    ~T5DecodingPluginCreator();
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* szNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;
};

}  // namespace nvinfer1
