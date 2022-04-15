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

#include "T5Plugin.h"

using namespace fastertransformer;

namespace nvinfer1 {

// class T5EncoderPlugin ---------------------------------------------------------------------------
T5EncoderPlugin::T5EncoderPlugin(const std::string& name,
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
                                 int useFP16):
    name_(name)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    m_.max_batch_size = max_batch_size;
    m_.max_seq_len = max_seq_len;
    m_.beam_width = beam_width;
    m_.head_num = head_num;
    m_.size_per_head = size_per_head;
    m_.inter_size = inter_size;
    m_.d_model = d_model;
    m_.num_layer = num_layer;
    m_.num_bucket = num_bucket;
    m_.max_distance = max_distance;
    m_.sm = sm;
    m_.q_scaling = q_scaling;
    m_.useFP16 = (bool)useFP16;
    m_.batch_size = m_.max_batch_size;
    m_.seq_len = m_.max_seq_len;

    cublasCreate(&cublasHandle_);
    cublasLtCreate(&cublasltHandle_);
#ifdef SPARSITY_ENABLED
    cusparseLtInit(&cusparseltHandle_));
#endif

    // T5 EncoderWeight
    std::string paraFilePath = "./para";
    if (m_.useFP16) {
        m_.attention_type = AttentionType::UNFUSED_MHA;  // when use FP16, only this type works till v5.0-dev
        pT5EncoderWeightHalf_ = new T5EncoderWeight<half>(m_.head_num,
                                                          m_.size_per_head,
                                                          m_.d_model,
                                                          m_.inter_size,
                                                          m_.vocab_size,
                                                          m_.num_layer,
                                                          m_.num_bucket,
                                                          1,  // tensor_para_size
                                                          0,  // tensor_para_rank
                                                          1,  // pipeline_para_size
                                                          0   // pipeline_para_rank
        );
        pT5EncoderWeightHalf_->loadModel(paraFilePath);
    }
    else {
        m_.attention_type =
            getAttentionType<float>(m_.size_per_head, getSMVersion(), m_.is_remove_padding, m_.max_seq_len);
        pT5EncoderWeightFloat_ = new T5EncoderWeight<float>(m_.head_num,
                                                            m_.size_per_head,
                                                            m_.d_model,
                                                            m_.inter_size,
                                                            m_.vocab_size,
                                                            m_.num_layer,
                                                            m_.num_bucket,
                                                            1,  // tensor_para_size
                                                            0,  // tensor_para_rank
                                                            1,  // pipeline_para_size
                                                            0   // pipeline_para_rank
        );
        pT5EncoderWeightFloat_->loadModel(paraFilePath);
    }

    // Gemm file selection
    std::string gemmFileName = std::string(GEMM_CONFIG).substr(0, 11) + std::string("-SM") + std::to_string(m_.sm)
                               + std::string("-FP") + std::to_string(m_.useFP16 ? 16 : 32) + std::string("-BS")
                               + std::to_string(m_.batch_size) + std::string("-SL") + std::to_string(m_.seq_len)
                               + std::string("-BM") + std::to_string(m_.beam_width) + std::string(".in");
    std::ifstream infile(gemmFileName);
    if (infile.good()) {
#if DEBUG_ENABLE == 1
        printf("Gemm file exist!\n");
#endif
    }
    else {
#if DEBUG_ENABLE == 1
        printf("Gemm file do not exist!\n");
#endif
        int argv[16] = {
            0,
            m_.max_batch_size,
            m_.beam_width,                                                   // useless for encoder
            (m_.batch_size == 128 && m_.seq_len == 384) ? 128 : m_.seq_len,  // seq_len, in case of OOM
            m_.d_model,
            m_.head_num,
            m_.size_per_head,
            m_.inter_size,
            m_.d_model,
            m_.head_num,
            m_.size_per_head,
            m_.inter_size,
            m_.vocab_size,
            m_.useFP16,  // is_fp16
            1,           // tensor_para_size
            m_.useFP16   // is_fp16_compute_type
        };
        t5_gemm(argv);
        rename(std::string(GEMM_CONFIG).c_str(), gemmFileName.c_str());
    }

    pCublasAlgoMap_ = new cublasAlgoMap(gemmFileName, "");
    pCublasWrapperMutex_ = new std::mutex();
    pAllocator_ = new Allocator<AllocatorType::CUDA>(getDevice());

    // cublas wrapper and T5Encoder
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

        pT5EncoderHalf_ = new T5Encoder<half>(m_.max_batch_size,
                                              m_.max_seq_len,
                                              m_.head_num,
                                              m_.size_per_head,
                                              m_.inter_size,
                                              m_.d_model,
                                              m_.num_layer,
                                              m_.num_bucket,
                                              m_.max_distance,
                                              m_.sm,
                                              m_.q_scaling,
                                              0,  // stream placeholder
                                              pCublasWrapper_,
                                              pAllocator_,
                                              m_.is_free_buffer_after_forward,
                                              m_.attention_type,
                                              m_.is_sparse,
                                              m_.activation_type,
                                              m_.layernorm_type,
                                              {0, 1, nullptr},  // tensor_para
                                              {0, 1, nullptr}   // pipeline_para
        );
    }
    else {
        pCublasWrapper_->setFP32GemmConfig();

        pT5EncoderFloat_ = new T5Encoder<float>(m_.max_batch_size,
                                                m_.max_seq_len,
                                                m_.head_num,
                                                m_.size_per_head,
                                                m_.inter_size,
                                                m_.d_model,
                                                m_.num_layer,
                                                m_.num_bucket,
                                                m_.max_distance,
                                                m_.sm,
                                                m_.q_scaling,
                                                0,  // stream placeholder
                                                pCublasWrapper_,
                                                pAllocator_,
                                                m_.is_free_buffer_after_forward,
                                                m_.attention_type,
                                                m_.is_sparse,
                                                m_.activation_type,
                                                m_.layernorm_type,
                                                {0, 1, nullptr},  // tensor_para
                                                {0, 1, nullptr}   // pipeline_para
        );
    }
    PRINT_ENCODER(m_.useFP16)
}

T5EncoderPlugin::T5EncoderPlugin(const std::string& name, const void* buffer, size_t length): name_(name)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    memcpy(&m_, buffer, sizeof(m_));

    cublasCreate(&cublasHandle_);
    cublasLtCreate(&cublasltHandle_);
#ifdef SPARSITY_ENABLED
    cusparseLtInit(&cusparseltHandle_));
#endif

    // T5 EncoderWeight
    std::string paraFilePath = "./para";
    if (m_.useFP16) {
        m_.attention_type = AttentionType::UNFUSED_MHA;  // when use FP16, only this type works till v5.0-dev
        pT5EncoderWeightHalf_ = new T5EncoderWeight<half>(m_.head_num,
                                                          m_.size_per_head,
                                                          m_.d_model,
                                                          m_.inter_size,
                                                          m_.vocab_size,
                                                          m_.num_layer,
                                                          m_.num_bucket,
                                                          1,  // tensor_para_size
                                                          0,  // tensor_para_rank
                                                          1,  // pipeline_para_size
                                                          0   // pipeline_para_rank
        );
        pT5EncoderWeightHalf_->loadModel(paraFilePath);
    }
    else {
        m_.attention_type =
            getAttentionType<float>(m_.size_per_head, getSMVersion(), m_.is_remove_padding, m_.max_seq_len);
        pT5EncoderWeightFloat_ = new T5EncoderWeight<float>(m_.head_num,
                                                            m_.size_per_head,
                                                            m_.d_model,
                                                            m_.inter_size,
                                                            m_.vocab_size,
                                                            m_.num_layer,
                                                            m_.num_bucket,
                                                            1,  // tensor_para_size
                                                            0,  // tensor_para_rank
                                                            1,  // pipeline_para_size
                                                            0   // pipeline_para_rank
        );
        pT5EncoderWeightFloat_->loadModel(paraFilePath);
    }

    // Gemm file selection, in constructor, we use max_batch_szie and seq_len as data size
    std::string gemmFileName = std::string(GEMM_CONFIG).substr(0, 11) + std::string("-SM") + std::to_string(m_.sm)
                               + std::string("-FP") + std::to_string(m_.useFP16 ? 16 : 32) + std::string("-BS")
                               + std::to_string(m_.batch_size) + std::string("-SL") + std::to_string(m_.seq_len)
                               + std::string("-BM") + std::to_string(m_.beam_width) + std::string(".in");
    std::ifstream infile(gemmFileName);
    if (infile.good()) {
#if DEBUG_ENABLE == 1
        printf("Gemm file exist!\n");
#endif
    }
    else {
#if DEBUG_ENABLE == 1
        printf("Gemm file do not exist!\n");
#endif
        int argv[16] = {
            0,
            m_.max_batch_size,
            m_.beam_width,                                                   // useless for encoder
            (m_.batch_size == 128 && m_.seq_len == 384) ? 128 : m_.seq_len,  // seq_len, in case of OOM
            m_.d_model,
            m_.head_num,
            m_.size_per_head,
            m_.inter_size,
            m_.d_model,
            m_.head_num,
            m_.size_per_head,
            m_.inter_size,
            m_.vocab_size,
            m_.useFP16,  // is_fp16
            1,           // tensor_para_size
            m_.useFP16   // is_fp16_compute_type
        };
        t5_gemm(argv);
        rename(std::string(GEMM_CONFIG).c_str(), gemmFileName.c_str());
    }

    pCublasAlgoMap_ = new cublasAlgoMap(gemmFileName, "");
    pCublasWrapperMutex_ = new std::mutex();
    pAllocator_ = new Allocator<AllocatorType::CUDA>(getDevice());

    // cublas wrapper and T5Encoder
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

        pT5EncoderHalf_ = new T5Encoder<half>(m_.max_batch_size,
                                              m_.max_seq_len,
                                              m_.head_num,
                                              m_.size_per_head,
                                              m_.inter_size,
                                              m_.d_model,
                                              m_.num_layer,
                                              m_.num_bucket,
                                              m_.max_distance,
                                              m_.sm,
                                              m_.q_scaling,
                                              0,  // stream placeholder
                                              pCublasWrapper_,
                                              pAllocator_,
                                              m_.is_free_buffer_after_forward,
                                              m_.attention_type,
                                              m_.is_sparse,
                                              m_.activation_type,
                                              m_.layernorm_type,
                                              {0, 1, nullptr},  // tensor_para
                                              {0, 1, nullptr}   // pipeline_para
        );
    }
    else {
        pCublasWrapper_->setFP32GemmConfig();

        pT5EncoderFloat_ = new T5Encoder<float>(m_.max_batch_size,
                                                m_.max_seq_len,
                                                m_.head_num,
                                                m_.size_per_head,
                                                m_.inter_size,
                                                m_.d_model,
                                                m_.num_layer,
                                                m_.num_bucket,
                                                m_.max_distance,
                                                m_.sm,
                                                m_.q_scaling,
                                                0,  // stream placeholder
                                                pCublasWrapper_,
                                                pAllocator_,
                                                m_.is_free_buffer_after_forward,
                                                m_.attention_type,
                                                m_.is_sparse,
                                                m_.activation_type,
                                                m_.layernorm_type,
                                                {0, 1, nullptr},  // tensor_para
                                                {0, 1, nullptr}   // pipeline_para
        );
    }
    PRINT_ENCODER(m_.useFP16)
}

T5EncoderPlugin::~T5EncoderPlugin()
{
    WHERE_AM_I();
    if (pT5EncoderWeightHalf_ != nullptr) {
        delete pT5EncoderWeightHalf_;
    }
    if (pT5EncoderWeightFloat_ != nullptr) {
        delete pT5EncoderWeightFloat_;
    }
    if (pT5EncoderHalf_ != nullptr) {
        delete pT5EncoderHalf_;
    }
    if (pT5EncoderFloat_ != nullptr) {
        delete pT5EncoderFloat_;
    }
    delete pCublasAlgoMap_;
    delete pCublasWrapperMutex_;
    delete pCublasWrapper_;
    delete pAllocator_;
}

size_t T5EncoderPlugin::getSerializationSize() const noexcept
{
    WHERE_AM_I();
    return sizeof(m_);
}

void T5EncoderPlugin::serialize(void* buffer) const noexcept
{
    WHERE_AM_I();
    memcpy(buffer, &m_, sizeof(m_));
}

IPluginV2DynamicExt* T5EncoderPlugin::clone() const noexcept
{
    WHERE_AM_I();
    auto p = new T5EncoderPlugin(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int T5EncoderPlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

DataType T5EncoderPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept
{
    WHERE_AM_I();
    return m_.useFP16 ? DataType::kHALF : DataType::kFLOAT;
}

bool T5EncoderPlugin::supportsFormatCombination(int pos,
                                                const PluginTensorDesc* inOut,
                                                int nbInputs,
                                                int nbOutputs) noexcept
{
    WHERE_AM_I();
    bool res = false;

    switch (pos) {
        case 0:
        case 1:
            res = (inOut[pos].type == DataType::kINT32) && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
        case 2:
            res = (inOut[2].type == (m_.useFP16 ? DataType::kHALF : DataType::kFLOAT))
                  && (inOut[2].format == TensorFormat::kLINEAR);
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

DimsExprs T5EncoderPlugin::getOutputDimensions(int index,
                                               const DimsExprs* pInputDim,
                                               int nInputDim,
                                               IExprBuilder& exprBuilder) noexcept
{
    WHERE_AM_I();
    DimsExprs ret;
    ret.nbDims = 3;
    ret.d[0] = pInputDim[0].d[0];
    ret.d[1] = pInputDim[0].d[1];
    ret.d[2] = exprBuilder.constant(512);
    return ret;
}

void T5EncoderPlugin::configurePlugin(const DynamicPluginTensorDesc* in,
                                      int nbInput,
                                      const DynamicPluginTensorDesc* out,
                                      int nbOutput) noexcept
{
    WHERE_AM_I();
    PRINT_ENCODER(int(out[0].desc.type))
}

size_t T5EncoderPlugin::getWorkspaceSize(const PluginTensorDesc* inputs,
                                         int32_t nbInputs,
                                         const PluginTensorDesc* outputs,
                                         int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

void T5EncoderPlugin::setPluginNamespace(const char* szNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = szNamespace;
}

const char* T5EncoderPlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char* T5EncoderPlugin::getPluginType() const noexcept
{
    WHERE_AM_I();
    return ENCODER_NAME;
}

const char* T5EncoderPlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return ENCODER_VERSION;
}

int T5EncoderPlugin::initialize() noexcept
{
    WHERE_AM_I();
    return 0;
}

void T5EncoderPlugin::terminate() noexcept
{
    WHERE_AM_I();
}

void T5EncoderPlugin::destroy() noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    cublasDestroy(cublasHandle_);
    cublasLtDestroy(cublasltHandle_);
#ifdef SPARSITY_ENABLED
    cusparseLtDestroy(&cusparseltHandle_);
#endif
    // delete this;
}

void T5EncoderPlugin::attachToContext(cudnnContext* /*cudnn*/,
                                      cublasContext* /*cublas*/,
                                      IGpuAllocator* /*allocator*/) noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
}

void T5EncoderPlugin::detachFromContext() noexcept
{
    WHERE_AM_I();
}

int T5EncoderPlugin::enqueue(const PluginTensorDesc* inputDesc,
                             const PluginTensorDesc* outputDesc,
                             const void* const* inputs,
                             void* const* outputs,
                             void* workspace,
                             cudaStream_t stream) noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    m_.batch_size = inputDesc[0].dims.d[0];
    m_.seq_len = inputDesc[0].dims.d[1];
    PRINT_ENCODER(outputDesc[0].type)

    cublasSetStream(cublasHandle_, stream);
    pCublasWrapper_->setStream(stream);

    std::unordered_map<std::string, Tensor> inputTensor{
        {"input_ids", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{m_.batch_size, m_.seq_len}, (int*)inputs[0]}},
        {"sequence_length", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{m_.batch_size}, (int*)inputs[1]}}};
    if (m_.useFP16) {
        std::unordered_map<std::string, Tensor> outputTensor{
            {"output_hidden_state",
             Tensor{MEMORY_GPU,
                    TYPE_FP16,
                    std::vector<size_t>{m_.batch_size, m_.seq_len, (size_t)(m_.head_num * m_.size_per_head)},
                    (half*)outputs[0]}}};
        pT5EncoderHalf_->setStream(stream);
        pT5EncoderHalf_->forward(&outputTensor, &inputTensor, pT5EncoderWeightHalf_);
    }
    else {
        std::unordered_map<std::string, Tensor> outputTensor{
            {"output_hidden_state",
             Tensor{MEMORY_GPU,
                    TYPE_FP32,
                    std::vector<size_t>{m_.batch_size, m_.seq_len, (size_t)(m_.head_num * m_.size_per_head)},
                    (float*)outputs[0]}}};
        pT5EncoderFloat_->setStream(stream);
        pT5EncoderFloat_->forward(&outputTensor, &inputTensor, pT5EncoderWeightFloat_);
    }
    return 0;
}

// class T5EncoderPluginCreator --------------------------------------------------------------------
PluginFieldCollection T5EncoderPluginCreator::fc_{};
std::vector<PluginField> T5EncoderPluginCreator::attr_;

T5EncoderPluginCreator::T5EncoderPluginCreator()
{
    WHERE_AM_I();
    fc_.nbFields = attr_.size();
    fc_.fields = attr_.data();
}

T5EncoderPluginCreator::~T5EncoderPluginCreator()
{
    WHERE_AM_I();
}

IPluginV2* T5EncoderPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    int max_batch_size = 128;
    int max_seq_len = 384;
    int beam_width = 1;
    int head_num = 8;
    int size_per_head = 512 / 8;
    int inter_size = 512 * 4;
    int d_model = 512;
    int num_layer = 6;
    int num_bucket = 32;
    int max_distance = 128;
    int sm = -1;
    float q_scaling = 1.0f / (sqrt(size_per_head) * 1.0f);
    int useFP16 = 0;

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    sm = prop.major * 10 + prop.minor;

    std::map<std::string, int*> name2p{
        {"max_batch_size", &max_batch_size},
        {"max_seq_len", &max_seq_len},
        {"beam_width", &beam_width},
        {"head_num", &head_num},
        {"size_per_head", &size_per_head},
        {"inter_size", &inter_size},
        {"d_model", &d_model},
        {"num_layer", &num_layer},
        {"num_bucket", &num_bucket},
        {"max_distance", &max_distance},
        {"sm", &sm},
        {"useFP16", &useFP16},
    };
    for (int i = 0; i < fc->nbFields; i++) {
        if (!strcmp(fc->fields[i].name, "q_scaling")) {
            q_scaling = *(float*)fc->fields[i].data;
        }
        else if (name2p.find(fc->fields[i].name) != name2p.end()) {
            *name2p[fc->fields[i].name] = *(int*)fc->fields[i].data;
        }
    }
    return new T5EncoderPlugin(name,
                               max_batch_size,
                               max_seq_len,
                               beam_width,
                               head_num,
                               size_per_head,
                               inter_size,
                               d_model,
                               num_layer,
                               num_bucket,
                               max_distance,
                               sm,
                               q_scaling,
                               useFP16);
}

IPluginV2*
T5EncoderPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    WHERE_AM_I();
    return new T5EncoderPlugin(name, serialData, serialLength);
}

void T5EncoderPluginCreator::setPluginNamespace(const char* szNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = szNamespace;
}

const char* T5EncoderPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char* T5EncoderPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return ENCODER_NAME;
}

const char* T5EncoderPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return ENCODER_VERSION;
}

const PluginFieldCollection* T5EncoderPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(T5EncoderPluginCreator);

// class T5DecodingPlugin --------------------------------------------------------------------------
T5DecodingPlugin::T5DecodingPlugin(const std::string& name,
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
                                   int useFP16):
    name_(name)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    m_.max_batch_size = max_batch_size;
    m_.max_seq_len = max_seq_len;
    m_.mem_max_seq_len = mem_max_seq_len;
    m_.beam_width = beam_width;
    m_.head_num = head_num;
    m_.size_per_head = size_per_head;
    m_.inter_size = inter_size;
    m_.d_model = d_model;
    m_.num_layer = num_layer;
    m_.vocab_size = vocab_size;
    m_.num_bucket = num_bucket;
    m_.max_distance = max_distance;
    m_.q_scaling = q_scaling;
    m_.start_id = start_id;
    m_.end_id = end_id;
    m_.beam_search_diversity_rate = beam_search_diversity_rate;
    m_.top_k = top_k;
    m_.top_p = top_p;
    m_.temperature = temperature;
    m_.len_penalty = len_penalty;
    m_.repetition_penalty = repetition_penalty;
    m_.useFP16 = useFP16;
    m_.batch_size = m_.max_batch_size;
    m_.seq_len = m_.max_seq_len;

    cublasCreate(&cublasHandle_);
    cublasLtCreate(&cublasltHandle_);

    // T5DecodingWeight
    std::string paraFilePath = "./para";
    if (m_.useFP16) {
        pT5DecodingWeightHalf_ = new T5DecodingWeight<half>(m_.head_num,
                                                            m_.size_per_head,
                                                            m_.d_model,
                                                            m_.inter_size,
                                                            m_.vocab_size,
                                                            m_.num_layer,
                                                            m_.mem_d_model,
                                                            m_.num_bucket,
                                                            1,  // tensor_para_size
                                                            0,  // tensor_para_rank
                                                            1,  // pipeline_para_size
                                                            0   // pipeline_para_rank
        );
        pT5DecodingWeightHalf_->loadModel(paraFilePath);
    }
    else {
        pT5DecodingWeightFloat_ = new T5DecodingWeight<float>(m_.head_num,
                                                              m_.size_per_head,
                                                              m_.d_model,
                                                              m_.inter_size,
                                                              m_.vocab_size,
                                                              m_.num_layer,
                                                              m_.mem_d_model,
                                                              m_.num_bucket,
                                                              1,  // tensor_para_size,
                                                              0,  // tensor_para_rank,
                                                              1,  // pipeline_para_size,
                                                              0   // pipeline_para_rank
        );
        pT5DecodingWeightFloat_->loadModel(paraFilePath);
    }

    // Gemm file selection
    check_cuda_error(cudaGetDeviceProperties(&cuda_device_prop_, 0));
    std::string gemmFileName = std::string(GEMM_CONFIG).substr(0, 11) + std::string("-SM")
                               + std::to_string(cuda_device_prop_.major * 10 + cuda_device_prop_.minor)
                               + std::string("-FP") + std::to_string(m_.useFP16 ? 16 : 32) + std::string("-BS")
                               + std::to_string(m_.batch_size) + std::string("-SL") + std::to_string(m_.seq_len)
                               + std::string("-BM") + std::to_string(m_.beam_width) + std::string(".in");
    std::ifstream infile(gemmFileName);
    if (infile.good()) {
#if DEBUG_ENABLE == 1
        printf("Gemm file exist!\n");
#endif
    }
    else {
#if DEBUG_ENABLE == 1
        printf("Gemm file do not exist!\n");
#endif
        int argv[16] = {
            0,
            m_.max_batch_size,
            m_.beam_width,
            (m_.batch_size == 128 && m_.seq_len == 384) ? 128 : m_.seq_len,  // seq_len, in case of OOM
            m_.d_model,
            m_.head_num,
            m_.size_per_head,
            m_.inter_size,
            m_.d_model,
            m_.head_num,
            m_.size_per_head,
            m_.inter_size,
            m_.vocab_size,
            m_.useFP16,  // is_fp16
            1,           // tensor_para_size
            m_.useFP16   // is_fp16_compute_type
        };
        t5_gemm(argv);
        rename(std::string(GEMM_CONFIG).c_str(), gemmFileName.c_str());
    }

    pCublasAlgoMap_ = new cublasAlgoMap(gemmFileName, "");
    pCublasWrapperMutex_ = new std::mutex();
    pAllocator_ = new Allocator<AllocatorType::CUDA>(getDevice());

    // cublas wrapper and T5Decoding
    pCublasWrapper_ =
        new cublasMMWrapper(cublasHandle_, cublasltHandle_, 0, pCublasAlgoMap_, pCublasWrapperMutex_, pAllocator_);

    if (m_.useFP16) {
        pCublasWrapper_->setFP16GemmConfig();

        pT5DecodingHalf_ = new T5Decoding<half>(m_.max_batch_size,
                                                m_.max_seq_len,
                                                m_.mem_max_seq_len,
                                                m_.beam_width,
                                                m_.head_num,
                                                m_.size_per_head,
                                                m_.inter_size,
                                                m_.d_model,
                                                m_.num_layer,
                                                m_.vocab_size,
                                                m_.num_bucket,
                                                m_.max_distance,
                                                m_.q_scaling,
                                                m_.start_id,
                                                m_.end_id,
                                                m_.beam_search_diversity_rate,
                                                m_.top_k,
                                                m_.top_p,
                                                m_.temperature,
                                                m_.len_penalty,
                                                m_.repetition_penalty,
                                                0,  // stream placeholder
                                                pCublasWrapper_,
                                                pAllocator_,
                                                m_.is_free_buffer_after_forward,
                                                &cuda_device_prop_,
                                                {0, 1, nullptr},  // tensor_para
                                                {0, 1, nullptr}   // pipeline_para
        );
    }
    else {
        pCublasWrapper_->setFP32GemmConfig();

        pT5DecodingFloat_ = new T5Decoding<float>(m_.max_batch_size,
                                                  m_.max_seq_len,
                                                  m_.mem_max_seq_len,
                                                  m_.beam_width,
                                                  m_.head_num,
                                                  m_.size_per_head,
                                                  m_.inter_size,
                                                  m_.d_model,
                                                  m_.num_layer,
                                                  m_.vocab_size,
                                                  m_.num_bucket,
                                                  m_.max_distance,
                                                  m_.q_scaling,
                                                  m_.start_id,
                                                  m_.end_id,
                                                  m_.beam_search_diversity_rate,
                                                  m_.top_k,
                                                  m_.top_p,
                                                  m_.temperature,
                                                  m_.len_penalty,
                                                  m_.repetition_penalty,
                                                  0,  // stream placeholder
                                                  pCublasWrapper_,
                                                  pAllocator_,
                                                  m_.is_free_buffer_after_forward,
                                                  &cuda_device_prop_,
                                                  {0, 1, nullptr},  // tensor_para
                                                  {0, 1, nullptr}   // pipeline_para
        );
    }
}

T5DecodingPlugin::T5DecodingPlugin(const std::string& name, const void* buffer, size_t length): name_(name)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    memcpy(&m_, buffer, sizeof(m_));

    cublasCreate(&cublasHandle_);
    cublasLtCreate(&cublasltHandle_);

    // T5DecodingWeight
    std::string paraFilePath = "./para";
    if (m_.useFP16) {
        pT5DecodingWeightHalf_ = new T5DecodingWeight<half>(m_.head_num,
                                                            m_.size_per_head,
                                                            m_.d_model,
                                                            m_.inter_size,
                                                            m_.vocab_size,
                                                            m_.num_layer,
                                                            m_.mem_d_model,
                                                            m_.num_bucket,
                                                            1,  // tensor_para_size
                                                            0,  // tensor_para_rank
                                                            1,  // pipeline_para_size
                                                            0   // pipeline_para_rank
        );
        pT5DecodingWeightHalf_->loadModel(paraFilePath);
    }
    else {
        pT5DecodingWeightFloat_ = new T5DecodingWeight<float>(m_.head_num,
                                                              m_.size_per_head,
                                                              m_.d_model,
                                                              m_.inter_size,
                                                              m_.vocab_size,
                                                              m_.num_layer,
                                                              m_.mem_d_model,
                                                              m_.num_bucket,
                                                              1,  // tensor_para_size,
                                                              0,  // tensor_para_rank,
                                                              1,  // pipeline_para_size,
                                                              0   // pipeline_para_rank
        );
        pT5DecodingWeightFloat_->loadModel(paraFilePath);
    }

    // Gemm file selection
    check_cuda_error(cudaGetDeviceProperties(&cuda_device_prop_, 0));
    std::string gemmFileName = std::string(GEMM_CONFIG).substr(0, 11) + std::string("-SM")
                               + std::to_string(cuda_device_prop_.major * 10 + cuda_device_prop_.minor)
                               + std::string("-FP") + std::to_string(m_.useFP16 ? 16 : 32) + std::string("-BS")
                               + std::to_string(m_.batch_size) + std::string("-SL") + std::to_string(m_.seq_len)
                               + std::string("-BM") + std::to_string(m_.beam_width) + std::string(".in");
    std::ifstream infile(gemmFileName);
    if (infile.good()) {
#if DEBUG_ENABLE == 1
        printf("Gemm file exist!\n");
#endif
    }
    else {
#if DEBUG_ENABLE == 1
        printf("Gemm file do not exist!\n");
#endif
        int argv[16] = {
            0,
            m_.max_batch_size,
            m_.beam_width,
            (m_.batch_size == 128 && m_.seq_len == 384) ? 128 : m_.seq_len,  // seq_len, in case of OOM
            m_.d_model,
            m_.head_num,
            m_.size_per_head,
            m_.inter_size,
            m_.d_model,
            m_.head_num,
            m_.size_per_head,
            m_.inter_size,
            m_.vocab_size,
            m_.useFP16,  // is_fp16
            1,           // tensor_para_size
            m_.useFP16   // is_fp16_compute_type
        };
        t5_gemm(argv);
        rename(std::string(GEMM_CONFIG).c_str(), gemmFileName.c_str());
    }

    pCublasAlgoMap_ = new cublasAlgoMap(gemmFileName, "");
    pCublasWrapperMutex_ = new std::mutex();
    pAllocator_ = new Allocator<AllocatorType::CUDA>(getDevice());

    // cublas wrapper and T5Decoding
    pCublasWrapper_ =
        new cublasMMWrapper(cublasHandle_, cublasltHandle_, 0, pCublasAlgoMap_, pCublasWrapperMutex_, pAllocator_);

    if (m_.useFP16) {
        pCublasWrapper_->setFP16GemmConfig();

        pT5DecodingHalf_ = new T5Decoding<half>(m_.max_batch_size,
                                                m_.max_seq_len,
                                                m_.mem_max_seq_len,
                                                m_.beam_width,
                                                m_.head_num,
                                                m_.size_per_head,
                                                m_.inter_size,
                                                m_.d_model,
                                                m_.num_layer,
                                                m_.vocab_size,
                                                m_.num_bucket,
                                                m_.max_distance,
                                                m_.q_scaling,
                                                m_.start_id,
                                                m_.end_id,
                                                m_.beam_search_diversity_rate,
                                                m_.top_k,
                                                m_.top_p,
                                                m_.temperature,
                                                m_.len_penalty,
                                                m_.repetition_penalty,
                                                0,  // stream placeholder
                                                pCublasWrapper_,
                                                pAllocator_,
                                                m_.is_free_buffer_after_forward,
                                                &cuda_device_prop_,
                                                {0, 1, nullptr},  // tensor_para
                                                {0, 1, nullptr}   // pipeline_para
        );
    }
    else {
        pCublasWrapper_->setFP32GemmConfig();

        pT5DecodingFloat_ = new T5Decoding<float>(m_.max_batch_size,
                                                  m_.max_seq_len,
                                                  m_.mem_max_seq_len,
                                                  m_.beam_width,
                                                  m_.head_num,
                                                  m_.size_per_head,
                                                  m_.inter_size,
                                                  m_.d_model,
                                                  m_.num_layer,
                                                  m_.vocab_size,
                                                  m_.num_bucket,
                                                  m_.max_distance,
                                                  m_.q_scaling,
                                                  m_.start_id,
                                                  m_.end_id,
                                                  m_.beam_search_diversity_rate,
                                                  m_.top_k,
                                                  m_.top_p,
                                                  m_.temperature,
                                                  m_.len_penalty,
                                                  m_.repetition_penalty,
                                                  0,  // stream placeholder
                                                  pCublasWrapper_,
                                                  pAllocator_,
                                                  m_.is_free_buffer_after_forward,
                                                  &cuda_device_prop_,
                                                  {0, 1, nullptr},  // tensor_para
                                                  {0, 1, nullptr}   // pipeline_para
        );
    }
}

T5DecodingPlugin::~T5DecodingPlugin()
{
    WHERE_AM_I();
    if (pT5DecodingWeightHalf_ != nullptr) {
        delete pT5DecodingWeightHalf_;
    }
    if (pT5DecodingWeightFloat_ != nullptr) {
        delete pT5DecodingWeightFloat_;
    }
    if (pT5DecodingHalf_ != nullptr) {
        delete pT5DecodingHalf_;
    }
    if (pT5DecodingFloat_ != nullptr) {
        delete pT5DecodingFloat_;
    }
    delete pCublasAlgoMap_;
    delete pCublasWrapperMutex_;
    delete pCublasWrapper_;
    delete pAllocator_;
}

size_t T5DecodingPlugin::getSerializationSize() const noexcept
{
    WHERE_AM_I();
    return sizeof(m_);
}

void T5DecodingPlugin::serialize(void* buffer) const noexcept
{
    WHERE_AM_I();
    memcpy(buffer, &m_, sizeof(m_));
}

IPluginV2DynamicExt* T5DecodingPlugin::clone() const noexcept
{
    WHERE_AM_I();
    auto p = new T5DecodingPlugin(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int T5DecodingPlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 2;
}

DataType T5DecodingPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept
{
    WHERE_AM_I();
    return DataType::kINT32;
}

bool T5DecodingPlugin::supportsFormatCombination(int pos,
                                                 const PluginTensorDesc* inOut,
                                                 int nbInputs,
                                                 int nbOutputs) noexcept
{
    WHERE_AM_I();
    bool res = false;

    switch (pos) {
        case 0:
            res = (inOut[0].type == (m_.useFP16 ? DataType::kHALF : DataType::kFLOAT))
                  && (inOut[0].format == TensorFormat::kLINEAR);
            break;
        case 1:
        case 2:
        case 3:
        case 4:
            res = (inOut[pos].type == DataType::kINT32) && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
        default:  // should NOT be here!
            ;
    }
#if DEBUG_ENABLE == 1
    printf("Dim(");
    for (int i = 0; i < 5; i++) {
        printf("%d,", inOut[i].dims.nbDims);
    }
    printf("),");
    printf("pos=%d,res=%d,format(%d,%d,%d,%d,%d),type(%d,%d,%d,%d,%d),",
           pos,
           int(res),
           int(inOut[0].format),
           int(inOut[1].format),
           int(inOut[2].format),
           int(inOut[3].format),
           int(inOut[4].format),
           int(inOut[0].type),
           int(inOut[1].type),
           int(inOut[2].type),
           int(inOut[3].type),
           int(inOut[4].type));
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

DimsExprs T5DecodingPlugin::getOutputDimensions(int index,
                                                const DimsExprs* pInputDim,
                                                int nInputDim,
                                                IExprBuilder& exprBuilder) noexcept
{
    WHERE_AM_I();
    DimsExprs ret;
    switch (index) {
        case 0:
            ret.nbDims = 3;
            ret.d[0] = pInputDim[0].d[0];
            ret.d[1] = exprBuilder.constant(m_.beam_width);
            ret.d[2] = exprBuilder.constant(m_.max_seq_len);
            break;
        case 1:
            ret.nbDims = 2;
            ret.d[0] = pInputDim[0].d[0];
            ret.d[1] = exprBuilder.constant(m_.beam_width);
            break;
        default:  // should NOT be here!
            ;
    }
    return ret;
}

void T5DecodingPlugin::configurePlugin(const DynamicPluginTensorDesc* in,
                                       int nbInput,
                                       const DynamicPluginTensorDesc* out,
                                       int nbOutput) noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    PRINT_DECODING(int(in[0].desc.type))
}

size_t T5DecodingPlugin::getWorkspaceSize(const PluginTensorDesc* inputs,
                                          int32_t nbInputs,
                                          const PluginTensorDesc* outputs,
                                          int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

void T5DecodingPlugin::setPluginNamespace(const char* szNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = szNamespace;
}

const char* T5DecodingPlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char* T5DecodingPlugin::getPluginType() const noexcept
{
    WHERE_AM_I();
    return DECODING_NAME;
}

const char* T5DecodingPlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return DECODING_VERSION;
}

int T5DecodingPlugin::initialize() noexcept
{
    WHERE_AM_I();
    return 0;
}

void T5DecodingPlugin::terminate() noexcept
{
    WHERE_AM_I();
}

void T5DecodingPlugin::destroy() noexcept
{
    WHERE_AM_I();
    cublasDestroy(cublasHandle_);
    cublasLtDestroy(cublasltHandle_);
    // delete this;
}

void T5DecodingPlugin::attachToContext(cudnnContext* /*cudnn*/,
                                       cublasContext* /*cublas*/,
                                       IGpuAllocator* /*allocator*/) noexcept
{
    WHERE_AM_I();
}

void T5DecodingPlugin::detachFromContext() noexcept
{
    WHERE_AM_I();
}

int T5DecodingPlugin::enqueue(const PluginTensorDesc* inputDesc,
                              const PluginTensorDesc* outputDesc,
                              const void* const* inputs,
                              void* const* outputs,
                              void* workspace,
                              cudaStream_t stream) noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    m_.batch_size = inputDesc[0].dims.d[0];
    m_.seq_len = inputDesc[0].dims.d[1];
    PRINT_DECODING(inputDesc[0].type)

    cublasSetStream(cublasHandle_, stream);
    pCublasWrapper_->setStream(stream);

    std::unordered_map<std::string, Tensor> outputTensor{
        {"output_ids",
         Tensor{MEMORY_GPU,
                TYPE_INT32,
                std::vector<size_t>{(size_t)m_.batch_size, (size_t)m_.beam_width, (size_t)m_.max_seq_len},
                (int*)outputs[0]}},
        {"sequence_length",
         Tensor{MEMORY_GPU,
                TYPE_INT32,
                std::vector<size_t>{(size_t)m_.batch_size, (size_t)m_.beam_width},
                (int*)outputs[1]}}};
    if (m_.useFP16) {
        std::unordered_map<std::string, Tensor> inputTensor{
            {"encoder_output",
             Tensor{MEMORY_GPU,
                    TYPE_FP16,
                    std::vector<size_t>{(size_t)m_.batch_size, (size_t)m_.seq_len, (size_t)m_.mem_hidden_units},
                    (half*)inputs[0]}},
            {"encoder_sequence_length",
             Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{(size_t)m_.batch_size}, (int*)inputs[1]}}};
        pT5DecodingHalf_->setStream(stream);
        pT5DecodingHalf_->forward(&outputTensor, &inputTensor, pT5DecodingWeightHalf_);
    }
    else {
        std::unordered_map<std::string, Tensor> inputTensor{
            {"encoder_output",
             Tensor{MEMORY_GPU,
                    TYPE_FP32,
                    std::vector<size_t>{(size_t)m_.batch_size, (size_t)m_.seq_len, (size_t)m_.mem_hidden_units},
                    (float*)inputs[0]}},
            {"encoder_sequence_length",
             Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{(size_t)m_.batch_size}, (int*)inputs[1]}}};
        pT5DecodingFloat_->setStream(stream);
        pT5DecodingFloat_->forward(&outputTensor, &inputTensor, pT5DecodingWeightFloat_);
    }
    return 0;
}
// class T5DecodingPluginCreator -------------------------------------------------------------------
PluginFieldCollection T5DecodingPluginCreator::fc_{};
std::vector<PluginField> T5DecodingPluginCreator::attr_;

T5DecodingPluginCreator::T5DecodingPluginCreator()
{
    WHERE_AM_I();
    fc_.nbFields = attr_.size();
    fc_.fields = attr_.data();
}

T5DecodingPluginCreator::~T5DecodingPluginCreator()
{
    WHERE_AM_I();
}

IPluginV2* T5DecodingPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    int max_batch_size = 128;
    int max_seq_len = 384;
    int mem_max_seq_len = max_seq_len;
    int beam_width = 4;
    int head_num = 8;
    int size_per_head = 512 / 8;
    int d_model = head_num * size_per_head;
    int inter_size = d_model * 4;
    int num_layer = 6;
    int vocab_size = 32128;
    int num_bucket = 32;
    int max_distance = 128;
    int start_id = 0;
    int end_id = 1;
    float beam_search_diversity_rate = 0.0f;
    int top_k = beam_width;
    float top_p = 0.0f;
    float temperature = 1.0f;
    float len_penalty = 2.0f;
    float q_scaling = 1.0f / (sqrt(size_per_head) * 1.0f);
    float repetition_penalty = 1.0f;
    int useFP16 = 0;

    std::map<std::string, int*> name2pint{
        {"max_batch_size", &max_batch_size},
        {"max_seq_len", &max_seq_len},
        {"mem_max_seq_len", &mem_max_seq_len},
        {"beam_width", &beam_width},
        {"head_num", &head_num},
        {"size_per_head", &size_per_head},
        {"inter_size", &inter_size},
        {"d_model", &d_model},
        {"num_layer", &num_layer},
        {"num_bucket", &num_bucket},
        {"max_distance", &max_distance},
        {"vocab_size", &vocab_size},
        {"start_id", &start_id},
        {"end_id", &end_id},
        {"top_k", &top_k},
        {"useFP16", &useFP16},
    };
    std::map<std::string, float*> name2pfloat{
        {"beam_search_diversity_rate", &beam_search_diversity_rate},
        {"top_p", &top_p},
        {"temperature", &temperature},
        {"len_penalty", &len_penalty},
        {"repetition_penalty", &repetition_penalty},
    };
    for (int i = 0; i < fc->nbFields; i++) {
        if (name2pint.find(fc->fields[i].name) != name2pint.end()) {
            *name2pint[fc->fields[i].name] = *(int*)fc->fields[i].data;
        }
        if (name2pfloat.find(fc->fields[i].name) != name2pfloat.end()) {
            *name2pfloat[fc->fields[i].name] = *(float*)fc->fields[i].data;
        }
    }

    return new T5DecodingPlugin(name,
                                max_batch_size,
                                max_seq_len,
                                mem_max_seq_len,
                                beam_width,
                                head_num,
                                size_per_head,
                                inter_size,
                                d_model,
                                num_layer,
                                vocab_size,
                                num_bucket,
                                max_distance,
                                q_scaling,
                                start_id,
                                end_id,
                                beam_search_diversity_rate,
                                top_k,
                                top_p,
                                temperature,
                                len_penalty,
                                repetition_penalty,
                                useFP16);
}

IPluginV2*
T5DecodingPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    WHERE_AM_I();
    return new T5DecodingPlugin(name, serialData, serialLength);
}

void T5DecodingPluginCreator::setPluginNamespace(const char* szNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = szNamespace;
}

const char* T5DecodingPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char* T5DecodingPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return DECODING_NAME;
}

const char* T5DecodingPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return DECODING_VERSION;
}

const PluginFieldCollection* T5DecodingPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(T5DecodingPluginCreator);

}  // namespace nvinfer1
