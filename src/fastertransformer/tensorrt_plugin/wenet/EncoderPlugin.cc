/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include "EncoderPlugin.h"
#include <cuda_profiler_api.h>

using namespace fastertransformer;

namespace nvinfer1 {

// class WenetEncoderPlugin ---------------------------------------------------------------------------
WenetEncoderPlugin::WenetEncoderPlugin(const std::string& name,
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
                                       int                useFP16):
    name_(name)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    m_.max_batch_size               = max_batch_size;
    m_.max_seq_len                  = max_seq_len;
    m_.head_num                     = head_num;
    m_.size_per_head                = size_per_head;
    m_.feature_size                 = feature_size;
    m_.max_len                      = max_len;
    m_.inter_size                   = inter_size;
    m_.d_model                      = d_model;
    m_.num_layer                    = num_layer;
    m_.vocab_size                   = vocab_size;
    m_.conv_module_kernel_size      = conv_module_kernel_size;
    m_.sm                           = sm;
    m_.q_scaling                    = q_scaling;
    m_.use_layernorm_in_conv_module = (bool)use_layernorm_in_conv_module;
    m_.useFP16                      = (bool)useFP16;
    m_.batch_size                   = m_.max_batch_size;
    m_.seq_len                      = m_.max_seq_len;
    strcpy(m_.weightFilePath, weightFilePath.c_str());

    CreateFT();
}

void WenetEncoderPlugin::CreateFT()
{
    cublasCreate(&cublasHandle_);
    cublasLtCreate(&cublasltHandle_);
#ifdef SPARSITY_ENABLED
    CHECK_CUSPARSE(cusparseLtInit(&cusparseltHandle_));
#endif
    cudnnCreate(&cudnn_handle_);

    // Wenet EncoderWeight
    std::string weightFilePath = m_.weightFilePath;
    FT_LOG_WARNING("The default weight file path is %s. Change it accordingly, otherwise model will fail to load! \n",
                   weightFilePath.c_str());
    if (m_.useFP16) {
        m_.attention_type        = AttentionType::UNFUSED_MHA;
        pWenetEncoderWeightHalf_ = new WenetEncoderWeight<half>(m_.head_num,
                                                                m_.size_per_head,
                                                                m_.inter_size,
                                                                m_.d_model,
                                                                m_.vocab_size,
                                                                m_.conv_module_kernel_size,
                                                                m_.feature_size,
                                                                m_.max_len,
                                                                m_.num_layer,
                                                                m_.use_layernorm_in_conv_module);
        pWenetEncoderWeightHalf_->loadModel(weightFilePath);
    }
    else {
        m_.attention_type         = AttentionType::UNFUSED_MHA;
        pWenetEncoderWeightFloat_ = new WenetEncoderWeight<float>(m_.head_num,
                                                                  m_.size_per_head,
                                                                  m_.inter_size,
                                                                  m_.d_model,
                                                                  m_.vocab_size,
                                                                  m_.conv_module_kernel_size,
                                                                  m_.feature_size,
                                                                  m_.max_len,
                                                                  m_.num_layer,
                                                                  m_.use_layernorm_in_conv_module);
        pWenetEncoderWeightFloat_->loadModel(weightFilePath);
    }

    // Gemm file selection
    pCublasAlgoMap_      = new cublasAlgoMap("gemm_config.in", "");
    pCublasWrapperMutex_ = new std::mutex();
    pAllocator_          = new Allocator<AllocatorType::CUDA>(getDevice());

    // cublas wrapper and WenetEncoder
#ifdef SPARSITY_ENABLED
    pCublasWrapper_ = new cublasMMWrapper(
        cublasHandle_, cublasltHandle_, cusparseltHandle_, 0, pCublasAlgoMap_, pCublasWrapperMutex_, pAllocator_);
    m_.is_sparse = true;
#else
    pCublasWrapper_ =
        new cublasMMWrapper(cublasHandle_, cublasltHandle_, 0, pCublasAlgoMap_, pCublasWrapperMutex_, pAllocator_);
    m_.is_sparse = false;
#endif
    if (m_.useFP16) {
        pCublasWrapper_->setFP16GemmConfig();

        pWenetEncoderHalf_ = new WenetEncoder<half>(0,  // max_batch_size_, deprecated
                                                    0,  // max_seq_len_, deprecated
                                                    m_.head_num,
                                                    m_.size_per_head,
                                                    m_.feature_size,
                                                    m_.max_len,
                                                    m_.inter_size,
                                                    m_.d_model,
                                                    m_.num_layer,
                                                    m_.vocab_size,
                                                    m_.conv_module_kernel_size,
                                                    m_.sm,
                                                    m_.q_scaling,
                                                    cudnn_handle_,
                                                    0,  // stream placeholder
                                                    pCublasWrapper_,
                                                    pAllocator_,
                                                    m_.is_free_buffer_after_forward,
                                                    m_.attention_type,
                                                    m_.is_sparse,
                                                    m_.activation_type,
                                                    m_.use_layernorm_in_conv_module);
    }
    else {
        pCublasWrapper_->setFP32GemmConfig();

        pWenetEncoderFloat_ = new WenetEncoder<float>(0,  // max_batch_size_, deprecated
                                                      0,  // max_seq_len_, deprecated
                                                      m_.head_num,
                                                      m_.size_per_head,
                                                      m_.feature_size,
                                                      m_.max_len,
                                                      m_.inter_size,
                                                      m_.d_model,
                                                      m_.num_layer,
                                                      m_.vocab_size,
                                                      m_.conv_module_kernel_size,
                                                      m_.sm,
                                                      m_.q_scaling,
                                                      cudnn_handle_,
                                                      0,  // stream placeholder
                                                      pCublasWrapper_,
                                                      pAllocator_,
                                                      m_.is_free_buffer_after_forward,
                                                      m_.attention_type,
                                                      m_.is_sparse,
                                                      m_.activation_type,
                                                      m_.use_layernorm_in_conv_module);
    }
    PRINT_ENCODER(m_.useFP16)
}

WenetEncoderPlugin::WenetEncoderPlugin(const std::string& name, const void* buffer, size_t length): name_(name)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    memcpy(&m_, buffer, sizeof(m_));
    CreateFT();
}

WenetEncoderPlugin::~WenetEncoderPlugin()
{
    WHERE_AM_I();
    if (pWenetEncoderWeightHalf_ != nullptr) {
        delete pWenetEncoderWeightHalf_;
    }
    if (pWenetEncoderWeightFloat_ != nullptr) {
        delete pWenetEncoderWeightFloat_;
    }
    if (pWenetEncoderHalf_ != nullptr) {
        delete pWenetEncoderHalf_;
    }
    if (pWenetEncoderFloat_ != nullptr) {
        delete pWenetEncoderFloat_;
    }
    delete pCublasAlgoMap_;
    delete pCublasWrapperMutex_;
    delete pCublasWrapper_;
    delete pAllocator_;
}

size_t WenetEncoderPlugin::getSerializationSize() const noexcept
{
    WHERE_AM_I();
    return sizeof(m_);
}

void WenetEncoderPlugin::serialize(void* buffer) const noexcept
{
    WHERE_AM_I();
    memcpy(buffer, &m_, sizeof(m_));
}

IPluginV2DynamicExt* WenetEncoderPlugin::clone() const noexcept
{
    WHERE_AM_I();
    auto p = new WenetEncoderPlugin(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int WenetEncoderPlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 3;
}

DataType WenetEncoderPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept
{
    WHERE_AM_I();
    if (index == 0)  // encoder_out
        return m_.useFP16 ? DataType::kHALF : DataType::kFLOAT;
    else if (index == 1)  // encoder_out_length
        return DataType::kINT32;
    else if (index == 2)  // ctc_log_probs
        return DataType::kFLOAT;
}

bool WenetEncoderPlugin::supportsFormatCombination(int                     pos,
                                                   const PluginTensorDesc* inOut,
                                                   int                     nbInputs,
                                                   int                     nbOutputs) noexcept
{
    WHERE_AM_I();
    bool res = false;

    switch (pos) {
        case 0:  // speech
            res = (inOut[pos].type == (m_.useFP16 ? DataType::kHALF : DataType::kFLOAT))
                  && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
        case 1:  // speech_length
            res = (inOut[pos].type == DataType::kINT32) && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
        case 2:  // encoder_out
            res = (inOut[pos].type == (m_.useFP16 ? DataType::kHALF : DataType::kFLOAT))
                  && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
        case 3:  // encoder_out_length
            res = (inOut[pos].type == DataType::kINT32) && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
        case 4:  // ctc_log_probs
            res = (inOut[pos].type == DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
        default:  // should NOT be here!
            ;
    }
    return res;
}

DimsExprs WenetEncoderPlugin::getOutputDimensions(int              index,
                                                  const DimsExprs* pInputDim,
                                                  int              nInputDim,
                                                  IExprBuilder&    exprBuilder) noexcept
{
    WHERE_AM_I();
    const int             kernel_size = 3;
    const int             stride      = 2;
    const IDimensionExpr* seq_len1    = exprBuilder.operation(
        DimensionOperation::kSUM,
        *exprBuilder.operation(
            DimensionOperation::kFLOOR_DIV,
            *exprBuilder.operation(DimensionOperation::kSUB, *pInputDim[0].d[1], *exprBuilder.constant(kernel_size)),
            *exprBuilder.constant(stride)),
        *exprBuilder.constant(1));
    const IDimensionExpr* seq_len2 = exprBuilder.operation(
        DimensionOperation::kSUM,
        *exprBuilder.operation(
            DimensionOperation::kFLOOR_DIV,
            *exprBuilder.operation(DimensionOperation::kSUB, *seq_len1, *exprBuilder.constant(kernel_size)),
            *exprBuilder.constant(stride)),
        *exprBuilder.constant(1));
    if (index == 0) {
        DimsExprs ret;
        ret.nbDims = 3;
        ret.d[0]   = pInputDim[0].d[0];
        ret.d[1]   = seq_len2;
        ret.d[2]   = exprBuilder.constant(m_.d_model);
        return ret;
    }
    else if (index == 1)
        return pInputDim[index];
    else if (index == 2) {
        DimsExprs ret;
        ret.nbDims = 3;
        ret.d[0]   = pInputDim[0].d[0];
        ret.d[1]   = seq_len2;
        ret.d[2]   = exprBuilder.constant(m_.vocab_size);
        return ret;
    }
}

void WenetEncoderPlugin::configurePlugin(const DynamicPluginTensorDesc* in,
                                         int                            nbInput,
                                         const DynamicPluginTensorDesc* out,
                                         int                            nbOutput) noexcept
{
    WHERE_AM_I();
    PRINT_ENCODER(int(out[0].desc.type))
}

size_t WenetEncoderPlugin::getWorkspaceSize(const PluginTensorDesc* inputs,
                                            int32_t                 nbInputs,
                                            const PluginTensorDesc* outputs,
                                            int32_t                 nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

void WenetEncoderPlugin::setPluginNamespace(const char* szNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = szNamespace;
}

const char* WenetEncoderPlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char* WenetEncoderPlugin::getPluginType() const noexcept
{
    WHERE_AM_I();
    return ENCODER_NAME;
}

const char* WenetEncoderPlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return ENCODER_VERSION;
}

int WenetEncoderPlugin::initialize() noexcept
{
    WHERE_AM_I();
    return 0;
}

void WenetEncoderPlugin::terminate() noexcept
{
    WHERE_AM_I();
}

void WenetEncoderPlugin::destroy() noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    cudnnDestroy(cudnn_handle_);
    cublasDestroy(cublasHandle_);
    cublasLtDestroy(cublasltHandle_);
#ifdef SPARSITY_ENABLED
    cusparseLtDestroy(&cusparseltHandle_);
#endif
    delete this;
}

void WenetEncoderPlugin::attachToContext(cudnnContext*  cudnnContext,
                                         cublasContext* cublasContext,
                                         IGpuAllocator* gpuAllocator) noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    // cudnn_handle_ = cudnnContext;
}

void WenetEncoderPlugin::detachFromContext() noexcept
{
    WHERE_AM_I();
}

int WenetEncoderPlugin::enqueue(const PluginTensorDesc* inputDesc,
                                const PluginTensorDesc* outputDesc,
                                const void* const*      inputs,
                                void* const*            outputs,
                                void*                   workspace,
                                cudaStream_t            stream) noexcept
{
    // cudaProfilerStart();

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    m_.batch_size      = inputDesc[0].dims.d[0];
    m_.seq_len         = inputDesc[0].dims.d[1];
    size_t kernel_size = 3;
    size_t stride      = 2;
    size_t seq_len2    = (((m_.seq_len - kernel_size) / stride + 1) - kernel_size) / stride + 1;
    PRINT_ENCODER(outputDesc[0].type)

    cublasSetStream(cublasHandle_, stream);
    cudnnSetStream(cudnn_handle_, stream);
    pCublasWrapper_->setStream(stream);

    if (m_.useFP16) {
        TensorMap inputTensor{
            {"speech",
             Tensor{MEMORY_GPU,
                    TYPE_FP32,
                    std::vector<size_t>{m_.batch_size, m_.seq_len, (size_t)inputDesc[0].dims.d[2]},
                    (float*)inputs[0]}},
            {"sequence_length", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{m_.batch_size}, (int*)inputs[1]}}};

        TensorMap outputTensor{
            {"output_hidden_state",
             Tensor{
                 MEMORY_GPU, TYPE_FP16, std::vector<size_t>{m_.batch_size, seq_len2, m_.d_model}, (half*)outputs[0]}},
            {"encoder_out_lens", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{m_.batch_size}, (int*)outputs[1]}},
            {"ctc_log_probs",
             Tensor{MEMORY_GPU,
                    TYPE_FP32,
                    std::vector<size_t>{m_.batch_size, seq_len2, m_.vocab_size},
                    (float*)outputs[2]}},
        };
        pWenetEncoderHalf_->setStream(stream);
        pWenetEncoderHalf_->forward(&outputTensor, &inputTensor, pWenetEncoderWeightHalf_);
    }
    else {
        TensorMap inputTensor{
            {"speech",
             Tensor{MEMORY_GPU,
                    TYPE_FP32,
                    std::vector<size_t>{m_.batch_size, m_.seq_len, (size_t)inputDesc[0].dims.d[2]},
                    (float*)inputs[0]}},
            {"sequence_length", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{m_.batch_size}, (int*)inputs[1]}}};

        TensorMap outputTensor{
            {"output_hidden_state",
             Tensor{
                 MEMORY_GPU, TYPE_FP32, std::vector<size_t>{m_.batch_size, seq_len2, m_.d_model}, (float*)outputs[0]}},
            {"encoder_out_lens", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{m_.batch_size}, (int*)outputs[1]}},
            {"ctc_log_probs",
             Tensor{MEMORY_GPU,
                    TYPE_FP32,
                    std::vector<size_t>{m_.batch_size, seq_len2, m_.vocab_size},
                    (float*)outputs[2]}},
        };

        pWenetEncoderFloat_->setStream(stream);
        pWenetEncoderFloat_->forward(&outputTensor, &inputTensor, pWenetEncoderWeightFloat_);
    }
    // cudaProfilerStop();
    return 0;
}

// class WenetEncoderPluginCreator --------------------------------------------------------------------
PluginFieldCollection    WenetEncoderPluginCreator::fc_{};
std::vector<PluginField> WenetEncoderPluginCreator::attr_{
    {"max_batch_size", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"max_seq_len", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"head_num", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"size_per_head", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"feature_size", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"max_len", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"inter_size", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"d_model", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"num_layer", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"vocab_size", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"conv_module_kernel_size", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"sm", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"use_layernorm_in_conv_module", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"useFP16", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"q_scaling", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0},
    {"weightFilePath", nullptr, nvinfer1::PluginFieldType::kCHAR, 0}};

WenetEncoderPluginCreator::WenetEncoderPluginCreator()
{
    WHERE_AM_I();
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

WenetEncoderPluginCreator::~WenetEncoderPluginCreator()
{
    WHERE_AM_I();
}

IPluginV2* WenetEncoderPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    int         max_batch_size               = 128;
    int         max_seq_len                  = 384;
    int         head_num                     = 8;
    int         size_per_head                = 32;
    int         feature_size                 = 80;
    int         max_len                      = 5000;
    int         d_model                      = head_num * size_per_head;
    int         inter_size                   = d_model * 4;
    int         num_layer                    = 12;
    int         vocab_size                   = 4233;
    int         conv_module_kernel_size      = 15;
    int         sm                           = -1;
    float       q_scaling                    = 1.0f / (sqrt(size_per_head) * 1.0f);
    std::string weightFilePath               = "";
    int         use_layernorm_in_conv_module = 0;
    int         useFP16                      = 0;

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    sm = prop.major * 10 + prop.minor;

    std::map<std::string, int*> name2p{
        {"max_batch_size", &max_batch_size},
        {"max_seq_len", &max_seq_len},
        {"head_num", &head_num},
        {"size_per_head", &size_per_head},
        {"feature_size", &feature_size},
        {"max_len", &max_len},
        {"inter_size", &inter_size},
        {"d_model", &d_model},
        {"num_layer", &num_layer},
        {"vocab_size", &vocab_size},
        {"conv_module_kernel_size", &conv_module_kernel_size},
        {"sm", &sm},
        {"use_layernorm_in_conv_module", &use_layernorm_in_conv_module},
        {"useFP16", &useFP16},
    };
    for (int i = 0; i < fc->nbFields; i++) {
        if (!strcmp(fc->fields[i].name, "q_scaling")) {
            q_scaling = *(float*)fc->fields[i].data;
        }
        else if (!strcmp(fc->fields[i].name, "weightFilePath")) {
            weightFilePath = std::string((char*)fc->fields[i].data);
        }
        else if (name2p.find(fc->fields[i].name) != name2p.end()) {
            *name2p[fc->fields[i].name] = *(int*)fc->fields[i].data;
        }
    }
    auto p = new WenetEncoderPlugin(name,
                                    max_batch_size,
                                    max_seq_len,
                                    head_num,
                                    size_per_head,
                                    feature_size,
                                    max_len,
                                    inter_size,
                                    d_model,
                                    num_layer,
                                    vocab_size,
                                    conv_module_kernel_size,
                                    sm,
                                    q_scaling,
                                    weightFilePath,
                                    use_layernorm_in_conv_module,
                                    useFP16);
    return p;
}

IPluginV2*
WenetEncoderPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    WHERE_AM_I();
    return new WenetEncoderPlugin(name, serialData, serialLength);
}

void WenetEncoderPluginCreator::setPluginNamespace(const char* szNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = szNamespace;
}

const char* WenetEncoderPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char* WenetEncoderPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return ENCODER_NAME;
}

const char* WenetEncoderPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return ENCODER_VERSION;
}

const PluginFieldCollection* WenetEncoderPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(WenetEncoderPluginCreator);

}  // namespace nvinfer1
