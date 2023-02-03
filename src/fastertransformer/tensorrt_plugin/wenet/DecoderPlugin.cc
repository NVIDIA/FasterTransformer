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

#include "DecoderPlugin.h"

using namespace fastertransformer;

namespace nvinfer1 {

// class WenetDecoderPlugin ---------------------------------------------------------------------------
WenetDecoderPlugin::WenetDecoderPlugin(const std::string& name,
                                       size_t             max_batch_size,
                                       size_t             max_seq_len,
                                       size_t             head_num,
                                       size_t             size_per_head,
                                       size_t             inter_size,
                                       size_t             d_model,
                                       size_t             num_layer,
                                       size_t             vocab_size,
                                       size_t             max_len,
                                       int                sm,
                                       float              q_scaling,
                                       const std::string& weightFilePath,
                                       int                useFP16):
    name_(name)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    m_.max_batch_size = max_batch_size;
    m_.max_seq_len    = max_seq_len;
    m_.head_num       = head_num;
    m_.size_per_head  = size_per_head;
    m_.inter_size     = inter_size;
    m_.d_model        = d_model;
    m_.num_layer      = num_layer;
    m_.vocab_size     = vocab_size;
    m_.max_len = max_len, m_.sm = sm;
    m_.q_scaling  = q_scaling;
    m_.useFP16    = (bool)useFP16;
    m_.batch_size = m_.max_batch_size;
    m_.seq_len    = m_.max_seq_len;
    strcpy(m_.weightFilePath, weightFilePath.c_str());

    CreateFT();
}

void WenetDecoderPlugin::CreateFT()
{
    cublasCreate(&cublasHandle_);
    cublasLtCreate(&cublasltHandle_);
#ifdef SPARSITY_ENABLED
    CHECK_CUSPARSE(cusparseLtInit(&cusparseltHandle_));
#endif

    // Wenet DecoderWeight
    std::string weightFilePath = m_.weightFilePath;
    FT_LOG_WARNING("The default weight file path is %s. Change it accordingly, otherwise model will fail to load! \n",
                   weightFilePath.c_str());
    if (m_.useFP16) {
        pWenetDecoderWeightHalf_ = new WenetDecoderWeight<half>(
            m_.head_num, m_.size_per_head, m_.inter_size, m_.num_layer, m_.vocab_size, m_.max_len);
        pWenetDecoderWeightHalf_->loadModel(weightFilePath);
    }
    else {
        pWenetDecoderWeightFloat_ = new WenetDecoderWeight<float>(
            m_.head_num, m_.size_per_head, m_.inter_size, m_.num_layer, m_.vocab_size, m_.max_len);
        pWenetDecoderWeightFloat_->loadModel(weightFilePath);
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
        pWenetDecoderHalf_ = new WenetDecoder<half>(0,  // max_batch_size_, deprecated
                                                    0,  // max_seq_len_, deprecated
                                                    m_.head_num,
                                                    m_.size_per_head,
                                                    m_.inter_size,
                                                    m_.num_layer,
                                                    m_.vocab_size,
                                                    m_.max_len,
                                                    m_.q_scaling,
                                                    0,  // stream placeholder
                                                    pCublasWrapper_,
                                                    pAllocator_,
                                                    m_.is_free_buffer_after_forward);
    }
    else {
        pCublasWrapper_->setFP32GemmConfig();

        pWenetDecoderFloat_ = new WenetDecoder<float>(0,  // max_batch_size_, deprecated
                                                      0,  // max_seq_len_, deprecated
                                                      m_.head_num,
                                                      m_.size_per_head,
                                                      m_.inter_size,
                                                      m_.num_layer,
                                                      m_.vocab_size,
                                                      m_.max_len,
                                                      m_.q_scaling,
                                                      0,  // stream placeholder
                                                      pCublasWrapper_,
                                                      pAllocator_,
                                                      m_.is_free_buffer_after_forward);
    }
    PRINT_DECODER(m_.useFP16)
}

WenetDecoderPlugin::WenetDecoderPlugin(const std::string& name, const void* buffer, size_t length): name_(name)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    memcpy(&m_, buffer, sizeof(m_));

    CreateFT();
}

WenetDecoderPlugin::~WenetDecoderPlugin()
{
    WHERE_AM_I();
    if (pWenetDecoderWeightHalf_ != nullptr) {
        delete pWenetDecoderWeightHalf_;
    }
    if (pWenetDecoderWeightFloat_ != nullptr) {
        delete pWenetDecoderWeightFloat_;
    }
    if (pWenetDecoderHalf_ != nullptr) {
        delete pWenetDecoderHalf_;
    }
    if (pWenetDecoderFloat_ != nullptr) {
        delete pWenetDecoderFloat_;
    }
    delete pCublasAlgoMap_;
    delete pCublasWrapperMutex_;
    delete pCublasWrapper_;
    delete pAllocator_;
}

size_t WenetDecoderPlugin::getSerializationSize() const noexcept
{
    WHERE_AM_I();
    return sizeof(m_);
}

void WenetDecoderPlugin::serialize(void* buffer) const noexcept
{
    WHERE_AM_I();
    memcpy(buffer, &m_, sizeof(m_));
}

IPluginV2DynamicExt* WenetDecoderPlugin::clone() const noexcept
{
    WHERE_AM_I();
    auto p = new WenetDecoderPlugin(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int WenetDecoderPlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 2;
}

DataType WenetDecoderPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept
{
    WHERE_AM_I();
    return index == 0 ? DataType::kFLOAT : DataType::kINT32;
}

bool WenetDecoderPlugin::supportsFormatCombination(int                     pos,
                                                   const PluginTensorDesc* inOut,
                                                   int                     nbInputs,
                                                   int                     nbOutputs) noexcept
{
    WHERE_AM_I();
    bool res = false;

    switch (pos) {
        case 0:
        case 1:
        case 3:
            res = (inOut[pos].type == DataType::kINT32) && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
        case 2:
        case 4:
            res = (inOut[pos].type == (m_.useFP16 ? DataType::kHALF : DataType::kFLOAT))
                  && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
        case 5:
            res = (inOut[pos].type == DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
        case 6:
            res = (inOut[pos].type == DataType::kINT32) && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
        default:  // should NOT be here!
            ;
    }
#if DEBUG_ENABLE == 1
    printf("Dim(");
    for (int i = 0; i < 3; i++) {
        printf("%d,", inOut[i].dims.nbDims);
    }
    printf("),");
    printf("pos=%d,res=%d,format(%d,%d,%d),type(%d,%d,%d),",
           pos,
           int(res),
           int(inOut[0].format),
           int(inOut[1].format),
           int(inOut[2].format),
           int(inOut[0].type),
           int(inOut[1].type),
           int(inOut[2].type));
    printf("kLINEAR=%d,float=%d,half=%d,int8=%d,int32=%d,bool=%d\n",
           int(TensorFormat::kLINEAR),
           int(DataType::kFLOAT),
           int(DataType::kHALF),
           int(DataType::kINT8),
           int(DataType::kINT32),
           int(DataType::kBOOL));
#endif
    return res;
}

DimsExprs WenetDecoderPlugin::getOutputDimensions(int              index,
                                                  const DimsExprs* pInputDim,
                                                  int              nInputDim,
                                                  IExprBuilder&    exprBuilder) noexcept
{
    WHERE_AM_I();
    DimsExprs ret;
    if (index == 0) {
        ret.nbDims = 4;
        ret.d[0]   = pInputDim[0].d[0];
        ret.d[1]   = pInputDim[0].d[1];
        ret.d[2]   = exprBuilder.operation(DimensionOperation::kSUB, *pInputDim[0].d[2], *exprBuilder.constant(1));
        ret.d[3]   = exprBuilder.constant(m_.vocab_size);
    }
    else {
        ret.nbDims = 1;
        ret.d[0]   = pInputDim[0].d[0];
    }
    return ret;
}

void WenetDecoderPlugin::configurePlugin(const DynamicPluginTensorDesc* in,
                                         int                            nbInput,
                                         const DynamicPluginTensorDesc* out,
                                         int                            nbOutput) noexcept
{
    WHERE_AM_I();
    PRINT_DECODER(m_.useFP16)
}

size_t WenetDecoderPlugin::getWorkspaceSize(const PluginTensorDesc* inputs,
                                            int32_t                 nbInputs,
                                            const PluginTensorDesc* outputs,
                                            int32_t                 nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

void WenetDecoderPlugin::setPluginNamespace(const char* szNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = szNamespace;
}

const char* WenetDecoderPlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char* WenetDecoderPlugin::getPluginType() const noexcept
{
    WHERE_AM_I();
    return DECODER_NAME;
}

const char* WenetDecoderPlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return DECODER_VERSION;
}

int WenetDecoderPlugin::initialize() noexcept
{
    WHERE_AM_I();
    return 0;
}

void WenetDecoderPlugin::terminate() noexcept
{
    WHERE_AM_I();
}

void WenetDecoderPlugin::destroy() noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    cublasDestroy(cublasHandle_);
    cublasLtDestroy(cublasltHandle_);
#ifdef SPARSITY_ENABLED
    cusparseLtDestroy(&cusparseltHandle_);
#endif
    delete this;
}

void WenetDecoderPlugin::attachToContext(cudnnContext* /*cudnn*/,
                                         cublasContext* /*cublas*/,
                                         IGpuAllocator* /*allocator*/) noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
}

void WenetDecoderPlugin::detachFromContext() noexcept
{
    WHERE_AM_I();
}

int WenetDecoderPlugin::enqueue(const PluginTensorDesc* inputDesc,
                                const PluginTensorDesc* outputDesc,
                                const void* const*      inputs,
                                void* const*            outputs,
                                void*                   workspace,
                                cudaStream_t            stream) noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    FT_CHECK(inputDesc[0].dims.nbDims == 3);
    FT_CHECK(inputDesc[2].dims.nbDims == 3);

    const size_t batch_size = inputDesc[0].dims.d[0];
    const size_t beam_width = inputDesc[0].dims.d[1];
    const size_t seq_len1   = inputDesc[0].dims.d[2] - 1;
    const size_t d_model    = inputDesc[2].dims.d[2];

    const size_t seq_len2 = inputDesc[2].dims.d[1];

    m_.batch_size = batch_size * beam_width;
    m_.seq_len    = seq_len1;
    m_.d_model    = d_model;

    PRINT_DECODER(m_.useFP16)

    cublasSetStream(cublasHandle_, stream);
    pCublasWrapper_->setStream(stream);

    if (m_.useFP16) {
        TensorMap inputTensor{
            {"decoder_input",
             Tensor{
                 MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size, beam_width, seq_len1 + 1}, (int*)inputs[0]}},
            {"decoder_sequence_length",
             Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size, beam_width}, (int*)inputs[1]}},
            {"encoder_output",
             Tensor{MEMORY_GPU, TYPE_FP16, std::vector<size_t>{batch_size, seq_len2, d_model}, (half*)inputs[2]}},
            {"encoder_sequence_length",
             Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size}, (int*)inputs[3]}},
            {"ctc_score",
             Tensor{MEMORY_GPU, TYPE_FP16, std::vector<size_t>{batch_size, beam_width}, (half*)inputs[4]}}};

        TensorMap outputTensor{
            {"decoder_output",
             Tensor{MEMORY_GPU,
                    TYPE_FP32,
                    std::vector<size_t>{batch_size, beam_width, seq_len1, m_.vocab_size},
                    (float*)outputs[0]}},
            {"best_index", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size}, (int*)outputs[1]}}};
        pWenetDecoderHalf_->setStream(stream);
        pWenetDecoderHalf_->forward(&outputTensor, &inputTensor, pWenetDecoderWeightHalf_);
    }
    else {
        TensorMap inputTensor{
            {"decoder_input",
             Tensor{
                 MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size, beam_width, seq_len1 + 1}, (int*)inputs[0]}},
            {"decoder_sequence_length",
             Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size, beam_width}, (int*)inputs[1]}},
            {"encoder_output",
             Tensor{MEMORY_GPU, TYPE_FP32, std::vector<size_t>{batch_size, seq_len2, d_model}, (float*)inputs[2]}},
            {"encoder_sequence_length",
             Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size}, (int*)inputs[3]}},
            {"ctc_score", Tensor{MEMORY_GPU, TYPE_FP32, std::vector<size_t>{batch_size, beam_width}, (int*)inputs[4]}}};

        TensorMap outputTensor{
            {"decoder_output",
             Tensor{MEMORY_GPU,
                    TYPE_FP32,
                    std::vector<size_t>{batch_size, beam_width, seq_len1, m_.vocab_size},
                    (float*)outputs[0]}},
            {"best_index", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size}, (int*)outputs[1]}}};

        pWenetDecoderFloat_->setStream(stream);
        pWenetDecoderFloat_->forward(&outputTensor, &inputTensor, pWenetDecoderWeightFloat_);
    }
    return 0;
}

// class WenetDecoderPluginCreator --------------------------------------------------------------------
PluginFieldCollection    WenetDecoderPluginCreator::fc_{};
std::vector<PluginField> WenetDecoderPluginCreator::attr_{
    {"max_batch_size", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"max_seq_len", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"head_num", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"size_per_head", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"inter_size", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"d_model", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"num_layer", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"vocab_size", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"sm", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"useFP16", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"q_scaling", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0},
    {"weightFilePath", nullptr, nvinfer1::PluginFieldType::kCHAR, 0}};

WenetDecoderPluginCreator::WenetDecoderPluginCreator()
{
    WHERE_AM_I();
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

WenetDecoderPluginCreator::~WenetDecoderPluginCreator()
{
    WHERE_AM_I();
}

IPluginV2* WenetDecoderPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    int         max_batch_size = 128;
    int         max_seq_len    = 384;
    int         head_num       = 8;
    int         size_per_head  = 32;
    int         max_len        = 5000;
    int         d_model        = head_num * size_per_head;
    int         inter_size     = d_model * 4;
    int         num_layer      = 12;
    int         vocab_size     = 4233;
    int         sm             = -1;
    float       q_scaling      = 1.0f / (sqrt(size_per_head) * 1.0f);
    std::string weightFilePath = "";
    int         useFP16        = 0;

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    sm = prop.major * 10 + prop.minor;

    std::map<std::string, int*> name2p{
        {"max_batch_size", &max_batch_size},
        {"max_seq_len", &max_seq_len},
        {"head_num", &head_num},
        {"size_per_head", &size_per_head},
        {"inter_size", &inter_size},
        {"d_model", &d_model},
        {"num_layer", &num_layer},
        {"vocab_size", &vocab_size},
        {"sm", &sm},
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
    return new WenetDecoderPlugin(name,
                                  max_batch_size,
                                  max_seq_len,
                                  head_num,
                                  size_per_head,
                                  inter_size,
                                  d_model,
                                  num_layer,
                                  vocab_size,
                                  max_len,
                                  sm,
                                  q_scaling,
                                  weightFilePath,
                                  useFP16);
}

IPluginV2*
WenetDecoderPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    WHERE_AM_I();
    return new WenetDecoderPlugin(name, serialData, serialLength);
}

void WenetDecoderPluginCreator::setPluginNamespace(const char* szNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = szNamespace;
}

const char* WenetDecoderPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char* WenetDecoderPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return DECODER_NAME;
}

const char* WenetDecoderPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return DECODER_VERSION;
}

const PluginFieldCollection* WenetDecoderPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(WenetDecoderPluginCreator);

}  // namespace nvinfer1
