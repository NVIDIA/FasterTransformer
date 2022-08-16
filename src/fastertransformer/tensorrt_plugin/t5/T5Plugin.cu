/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

#include "3rdparty/INIReader.h"
#include "T5Plugin.h"

using namespace fastertransformer;

namespace nvinfer1 {

// class T5EncoderPlugin ---------------------------------------------------------------------------
T5EncoderPlugin::T5EncoderPlugin(const std::string& name,
                                 size_t             max_batch_size,
                                 size_t             max_seq_len,
                                 size_t             beam_width,
                                 int                sm,
                                 int                useFP16,
                                 const std::string& ckpt_path,
                                 bool               own_weight):
    name_(name)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();

    INIReader reader = INIReader(ckpt_path + "/config.ini");
    if (reader.ParseError() < 0) {
        FT_LOG_ERROR("Can't load %s/config.ini", ckpt_path.c_str());
        FT_CHECK(false);
    }

    bool t5_with_bias = reader.GetBoolean("structure", "t5_with_bias", false);
    m_.max_batch_size = max_batch_size;
    m_.max_seq_len    = max_seq_len;
    m_.beam_width     = beam_width;
    m_.head_num       = reader.GetInteger("encoder", "num_heads");
    m_.size_per_head  = reader.GetInteger("encoder", "d_kv");
    m_.inter_size     = reader.GetInteger("encoder", "d_ff");
    m_.d_model        = reader.GetInteger("encoder", "d_model");
    m_.num_layer      = reader.GetInteger("encoder", "num_layers");
    m_.num_bucket     = reader.GetInteger("encoder", "relative_attention_num_buckets_or_max_pos_seq_len");
    m_.max_distance   = reader.GetInteger("encoder", "relative_attention_max_distance");
    m_.sm             = sm;
    m_.q_scaling      = t5_with_bias ? 1.0f : (1.0f / (sqrt(m_.size_per_head) * 1.0f));
    m_.useFP16        = (bool)useFP16;
    m_.batch_size     = m_.max_batch_size;
    m_.seq_len        = m_.max_seq_len;
    strcpy(m_.ckpt_path, ckpt_path.c_str());

    cublasCreate(&cublasHandle_);
    cublasLtCreate(&cublasltHandle_);
#ifdef SPARSITY_ENABLED
    cusparseLtInit(&cusparseltHandle_);
#endif

    is_own_weight = own_weight;
    if (is_own_weight) {
        // T5 EncoderWeight
        if (m_.useFP16) {
            m_.attention_type =
                getAttentionType<half>(m_.size_per_head, getSMVersion(), m_.is_remove_padding, m_.max_seq_len, false);
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
            pT5EncoderWeightHalf_->loadModel(std::string(m_.ckpt_path));
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
            pT5EncoderWeightFloat_->loadModel(std::string(m_.ckpt_path));
        }
    }
    // Gemm file selection
    std::string gemmFileName = std::string(GEMM_CONFIG).substr(0, 11) + std::string("-SM") + std::to_string(m_.sm)
                               + std::string("-FP") + std::to_string(m_.useFP16 ? 16 : 32) + std::string("-BS")
                               + std::to_string(m_.batch_size) + std::string("-SL") + std::to_string(m_.seq_len)
                               + std::string("-BM") + std::to_string(m_.beam_width) + std::string(".in");
    std::ifstream infile(gemmFileName);
    if (infile.good()) {
#ifdef T5_PLUGIN_DEBUG
        printf("Gemm file exist!\n");
#endif
    }
    else {
#ifdef T5_PLUGIN_DEBUG
        printf("Gemm file do not exist!\n");
#endif
        int argv[16] = {
            0,
            (int)m_.max_batch_size,
            (int)m_.beam_width,                                                   // useless for encoder
            (m_.batch_size == 128 && m_.seq_len == 384) ? 128 : (int)m_.seq_len,  // seq_len, in case of OOM
            (int)m_.d_model,
            (int)m_.head_num,
            (int)m_.size_per_head,
            (int)m_.inter_size,
            (int)m_.d_model,
            (int)m_.head_num,
            (int)m_.size_per_head,
            (int)m_.inter_size,
            (int)m_.vocab_size,
            m_.useFP16 ? 1 : 0,  // is_fp16
            1,                   // tensor_para_size
            false                // always use fp32 compute type
        };
        t5_gemm(argv);
        rename(std::string(GEMM_CONFIG).c_str(), gemmFileName.c_str());
    }

    pCublasAlgoMap_      = new cublasAlgoMap(gemmFileName, "");
    pCublasWrapperMutex_ = new std::mutex();
    pAllocator_          = new Allocator<AllocatorType::CUDA>(getDevice());

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
                                              NcclParam(0, 1),  // tensor_para
                                              NcclParam(0, 1)   // pipeline_para
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
                                                NcclParam(0, 1),  // tensor_para
                                                NcclParam(0, 1)   // pipeline_para
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
    cusparseLtInit(&cusparseltHandle_);
#endif

    is_own_weight = true;
    if (is_own_weight) {
        // T5 EncoderWeight
        if (m_.useFP16) {
            m_.attention_type =
                getAttentionType<half>(m_.size_per_head, getSMVersion(), m_.is_remove_padding, m_.max_seq_len, false);
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
            pT5EncoderWeightHalf_->loadModel(std::string(m_.ckpt_path));
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
            pT5EncoderWeightFloat_->loadModel(std::string(m_.ckpt_path));
        }
    }
    // Gemm file selection, in constructor, we use max_batch_szie and seq_len as
    // data size
    std::string gemmFileName = std::string(GEMM_CONFIG).substr(0, 11) + std::string("-SM") + std::to_string(m_.sm)
                               + std::string("-FP") + std::to_string(m_.useFP16 ? 16 : 32) + std::string("-BS")
                               + std::to_string(m_.batch_size) + std::string("-SL") + std::to_string(m_.seq_len)
                               + std::string("-BM") + std::to_string(m_.beam_width) + std::string(".in");
    std::ifstream infile(gemmFileName);
    if (infile.good()) {
#ifdef T5_PLUGIN_DEBUG
        printf("Gemm file exist!\n");
#endif
    }
    else {
#ifdef T5_PLUGIN_DEBUG
        printf("Gemm file do not exist!\n");
#endif
        int argv[16] = {
            0,
            (int)m_.max_batch_size,
            (int)m_.beam_width,                                                   // useless for encoder
            (m_.batch_size == 128 && m_.seq_len == 384) ? 128 : (int)m_.seq_len,  // seq_len, in case of OOM
            (int)m_.d_model,
            (int)m_.head_num,
            (int)m_.size_per_head,
            (int)m_.inter_size,
            (int)m_.d_model,
            (int)m_.head_num,
            (int)m_.size_per_head,
            (int)m_.inter_size,
            (int)m_.vocab_size,
            m_.useFP16 ? 1 : 0,  // is_fp16
            1,                   // tensor_para_size
            false                // always use fp32 compute type
        };
        t5_gemm(argv);
        rename(std::string(GEMM_CONFIG).c_str(), gemmFileName.c_str());
    }

    pCublasAlgoMap_      = new cublasAlgoMap(gemmFileName, "");
    pCublasWrapperMutex_ = new std::mutex();
    pAllocator_          = new Allocator<AllocatorType::CUDA>(getDevice());

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
                                              NcclParam(0, 1),  // tensor_para
                                              NcclParam(0, 1)   // pipeline_para
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
                                                NcclParam(0, 1),  // tensor_para
                                                NcclParam(0, 1)   // pipeline_para
        );
    }
    PRINT_ENCODER(m_.useFP16)
}

T5EncoderPlugin::~T5EncoderPlugin()
{
    WHERE_AM_I();
    if (is_own_weight && pT5EncoderWeightHalf_ != nullptr) {
        delete pT5EncoderWeightHalf_;
    }
    if (is_own_weight && pT5EncoderWeightFloat_ != nullptr) {
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
    auto p = new T5EncoderPlugin(
        name_, m_.max_batch_size, m_.max_seq_len, m_.beam_width, m_.sm, m_.useFP16, std::string(m_.ckpt_path), false);
    p->setPluginNamespace(namespace_.c_str());
    p->pT5EncoderWeightHalf_  = this->pT5EncoderWeightHalf_;
    p->pT5EncoderWeightFloat_ = this->pT5EncoderWeightFloat_;
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

bool T5EncoderPlugin::supportsFormatCombination(int                     pos,
                                                const PluginTensorDesc* inOut,
                                                int                     nbInputs,
                                                int                     nbOutputs) noexcept
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
            res = false;
    }
#ifdef T5_PLUGIN_DEBUG
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

DimsExprs T5EncoderPlugin::getOutputDimensions(int              index,
                                               const DimsExprs* pInputDim,
                                               int              nInputDim,
                                               IExprBuilder&    exprBuilder) noexcept
{
    WHERE_AM_I();
    DimsExprs ret;
    ret.nbDims = 3;
    ret.d[0]   = pInputDim[0].d[0];
    ret.d[1]   = pInputDim[0].d[1];
    ret.d[2]   = exprBuilder.constant(512);
    return ret;
}

void T5EncoderPlugin::configurePlugin(const DynamicPluginTensorDesc* in,
                                      int                            nbInput,
                                      const DynamicPluginTensorDesc* out,
                                      int                            nbOutput) noexcept
{
    WHERE_AM_I();
    PRINT_ENCODER(int(out[0].desc.type))
}

size_t T5EncoderPlugin::getWorkspaceSize(const PluginTensorDesc* inputs,
                                         int32_t                 nbInputs,
                                         const PluginTensorDesc* outputs,
                                         int32_t                 nbOutputs) const noexcept
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
                             const void* const*      inputs,
                             void* const*            outputs,
                             void*                   workspace,
                             cudaStream_t            stream) noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    m_.batch_size = inputDesc[0].dims.d[0];
    m_.seq_len    = inputDesc[0].dims.d[1];
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
PluginFieldCollection    T5EncoderPluginCreator::fc_{};
std::vector<PluginField> T5EncoderPluginCreator::attr_;

T5EncoderPluginCreator::T5EncoderPluginCreator()
{
    WHERE_AM_I();
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

T5EncoderPluginCreator::~T5EncoderPluginCreator()
{
    WHERE_AM_I();
}

IPluginV2* T5EncoderPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    int         max_batch_size = 128;
    int         max_seq_len    = 384;
    int         beam_width     = 1;
    int         sm             = -1;
    int         useFP16        = 0;
    std::string ckpt_path      = std::string("");

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    sm = prop.major * 10 + prop.minor;

    std::map<std::string, int*> name2p{
        {"max_batch_size", &max_batch_size},
        {"max_seq_len", &max_seq_len},
        {"beam_width", &beam_width},
        {"sm", &sm},
        {"useFP16", &useFP16},
    };
    for (int i = 0; i < fc->nbFields; i++) {
        if (name2p.find(fc->fields[i].name) != name2p.end()) {
            *name2p[fc->fields[i].name] = *(int*)fc->fields[i].data;
        }
        else if (!strcmp(fc->fields[i].name, "ckpt_path")) {
            ckpt_path = std::string((char*)fc->fields[i].data);
        }
    }
    return new T5EncoderPlugin(name, max_batch_size, max_seq_len, beam_width, sm, useFP16, ckpt_path, true);
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
                                   size_t             max_batch_size,
                                   size_t             max_seq_len,
                                   size_t             mem_max_seq_len,
                                   size_t             beam_width,
                                   int                useFP16,
                                   const std::string& ckpt_path,
                                   bool               own_weight):
    name_(name)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();

    INIReader reader = INIReader(ckpt_path + "/config.ini");
    if (reader.ParseError() < 0) {
        FT_LOG_ERROR("Can't load %s/config.ini", ckpt_path.c_str());
        FT_CHECK(false);
    }

    bool t5_with_bias   = reader.GetBoolean("structure", "t5_with_bias", false);
    m_.max_batch_size   = max_batch_size;
    m_.max_seq_len      = max_seq_len;
    m_.mem_max_seq_len  = mem_max_seq_len;
    m_.beam_width       = beam_width;
    m_.head_num         = reader.GetInteger("decoder", "num_heads");
    m_.size_per_head    = reader.GetInteger("decoder", "d_kv");
    m_.inter_size       = reader.GetInteger("decoder", "d_ff");
    m_.d_model          = reader.GetInteger("decoder", "d_model");
    m_.num_layer        = reader.GetInteger("decoder", "num_layers");
    m_.vocab_size       = reader.GetInteger("decoder", "vocab_size");
    m_.num_bucket       = reader.GetInteger("decoder", "relative_attention_num_buckets_or_max_pos_seq_len");
    m_.max_distance     = reader.GetInteger("decoder", "relative_attention_max_distance");
    m_.q_scaling        = t5_with_bias ? 1.0f : (1.0f / (sqrt(m_.size_per_head) * 1.0f));
    m_.start_id         = reader.GetInteger("decoder", "decoder_start_token_id");
    m_.end_id           = reader.GetInteger("decoder", "eos_token_id");
    m_.useFP16          = (bool)useFP16;
    m_.batch_size       = m_.max_batch_size;
    m_.seq_len          = m_.max_seq_len;
    m_.mem_hidden_units = reader.GetInteger("encoder", "num_heads") * reader.GetInteger("encoder", "d_kv");
    m_.mem_d_model      = reader.GetInteger("encoder", "d_model");
    strcpy(m_.ckpt_path, ckpt_path.c_str());

    cublasCreate(&cublasHandle_);
    cublasLtCreate(&cublasltHandle_);

    is_own_weight = own_weight;
    if (is_own_weight) {
        // T5DecodingWeight
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
            pT5DecodingWeightHalf_->loadModel(std::string(m_.ckpt_path));
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
            pT5DecodingWeightFloat_->loadModel(std::string(m_.ckpt_path));
        }
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
#ifdef T5_PLUGIN_DEBUG
        printf("Gemm file exist!\n");
#endif
    }
    else {
#ifdef T5_PLUGIN_DEBUG
        printf("Gemm file do not exist!\n");
#endif
        int argv[16] = {
            0,
            (int)m_.max_batch_size,
            (int)m_.beam_width,
            (m_.batch_size == 128 && m_.seq_len == 384) ? 128 : (int)m_.seq_len,  // seq_len, in case of OOM
            (int)m_.d_model,
            (int)m_.head_num,
            (int)m_.size_per_head,
            (int)m_.inter_size,
            (int)m_.d_model,
            (int)m_.head_num,
            (int)m_.size_per_head,
            (int)m_.inter_size,
            (int)m_.vocab_size,
            m_.useFP16 ? 1 : 0,  // is_fp16
            1,                   // tensor_para_size
            false                // always use fp32 compute type
        };
        t5_gemm(argv);
        rename(std::string(GEMM_CONFIG).c_str(), gemmFileName.c_str());
    }

    pCublasAlgoMap_      = new cublasAlgoMap(gemmFileName, "");
    pCublasWrapperMutex_ = new std::mutex();
    pAllocator_          = new Allocator<AllocatorType::CUDA>(getDevice());

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
                                                0.0f,  // don't need to pass beam_search_diversity_rate in constructor
                                                0,     // don't need to pass top_k in constructor
                                                0.0f,  // don't need to pass top_p in constructor
                                                0.0f,  // don't need to pass temperature in constructor
                                                0.0f,  // don't need to pass len_penalty in constructor
                                                0.0f,  // don't need to pass repetition_penalty in constructor
                                                0,     // stream placeholder
                                                pCublasWrapper_,
                                                pAllocator_,
                                                m_.is_free_buffer_after_forward,
                                                &cuda_device_prop_,
                                                NcclParam(0, 1),  // tensor_para
                                                NcclParam(0, 1)   // pipeline_para
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
                                                  0.0f,  // don't need to pass beam_search_diversity_rate in constructor
                                                  0,     // don't need to pass top_k in constructor
                                                  0.0f,  // don't need to pass top_p in constructor
                                                  0.0f,  // don't need to pass temperature in constructor
                                                  0.0f,  // don't need to pass len_penalty in constructor
                                                  0.0f,  // don't need to pass repetition_penalty in constructor
                                                  0,     // stream placeholder
                                                  pCublasWrapper_,
                                                  pAllocator_,
                                                  m_.is_free_buffer_after_forward,
                                                  &cuda_device_prop_,
                                                  NcclParam(0, 1),  // tensor_para
                                                  NcclParam(0, 1)   // pipeline_para
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
    is_own_weight = true;
    if (is_own_weight) {
        // T5DecodingWeight
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
            pT5DecodingWeightHalf_->loadModel(std::string(m_.ckpt_path));
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
            pT5DecodingWeightFloat_->loadModel(std::string(m_.ckpt_path));
        }
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
#ifdef T5_PLUGIN_DEBUG
        FT_LOG_INFO("Gemm file exist!");
#endif
    }
    else {
#ifdef T5_PLUGIN_DEBUG
        FT_LOG_INFO("Gemm file do not exist!");
#endif
        int argv[16] = {
            0,
            (int)m_.max_batch_size,
            (int)m_.beam_width,
            (m_.batch_size == 128 && m_.seq_len == 384) ? 128 : (int)m_.seq_len,  // seq_len, in case of OOM
            (int)m_.d_model,
            (int)m_.head_num,
            (int)m_.size_per_head,
            (int)m_.inter_size,
            (int)m_.d_model,
            (int)m_.head_num,
            (int)m_.size_per_head,
            (int)m_.inter_size,
            (int)m_.vocab_size,
            m_.useFP16 ? 1 : 0,  // is_fp16
            1,                   // tensor_para_size
            false                // always use fp32 compute type
        };
        t5_gemm(argv);
        rename(std::string(GEMM_CONFIG).c_str(), gemmFileName.c_str());
    }

    pCublasAlgoMap_      = new cublasAlgoMap(gemmFileName, "");
    pCublasWrapperMutex_ = new std::mutex();
    pAllocator_          = new Allocator<AllocatorType::CUDA>(getDevice());

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
                                                NcclParam(0, 1),  // tensor_para
                                                NcclParam(0, 1)   // pipeline_para
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
                                                  NcclParam(0, 1),  // tensor_para
                                                  NcclParam(0, 1)   // pipeline_para
        );
    }
}

T5DecodingPlugin::~T5DecodingPlugin()
{
    WHERE_AM_I();
    if (is_own_weight && pT5DecodingWeightHalf_ != nullptr) {
        delete pT5DecodingWeightHalf_;
    }
    if (is_own_weight && pT5DecodingWeightFloat_ != nullptr) {
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
    auto p = new T5DecodingPlugin(name_,
                                  m_.max_batch_size,
                                  m_.max_seq_len,
                                  m_.mem_max_seq_len,
                                  m_.beam_width,
                                  m_.useFP16,
                                  std::string(m_.ckpt_path),
                                  false);
    p->setPluginNamespace(namespace_.c_str());
    p->pT5DecodingWeightHalf_  = this->pT5DecodingWeightHalf_;
    p->pT5DecodingWeightFloat_ = this->pT5DecodingWeightFloat_;
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

bool T5DecodingPlugin::supportsFormatCombination(int                     pos,
                                                 const PluginTensorDesc* inOut,
                                                 int                     nbInputs,
                                                 int                     nbOutputs) noexcept
{
    WHERE_AM_I();
    bool res = false;

    switch (pos) {
        case 0:  // encoder_output
            res = (inOut[pos].type == (m_.useFP16 ? DataType::kHALF : DataType::kFLOAT))
                  && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
        case 1:  // encoder_sequence_length
        case 2:  // runtime_top_k
        case 3:  // runtime_top_p
            res = (inOut[pos].type == DataType::kINT32) && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
            // res = (inOut[pos].type == DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR); break;
        case 4:  // beam_search_diversity_rate
        case 5:  // temperature
        case 6:  // len_penalty
        case 7:  // repetition_penalty
            res = (inOut[pos].type == DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
        case 8:  // output_ids
        case 9:  // sequence_length
            res = (inOut[pos].type == DataType::kINT32) && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
        default:  // should NOT be here!
            res = false;
    }
#ifdef T5_PLUGIN_DEBUG
    FT_LOG_INFO("Dim(");
    for (int i = 0; i < 5; i++) {
        FT_LOG_INFO("%d,", inOut[i].dims.nbDims);
    }
    FT_LOG_INFO("),");
    FT_LOG_INFO("pos=%d,res=%d,format(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d),type(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d),",
                pos,
                int(res),
                int(inOut[0].format),
                int(inOut[1].format),
                int(inOut[2].format),
                int(inOut[3].format),
                int(inOut[4].format),
                int(inOut[5].format),
                int(inOut[6].format),
                int(inOut[7].format),
                int(inOut[8].format),
                int(inOut[9].format),
                int(inOut[0].type),
                int(inOut[1].type),
                int(inOut[2].type),
                int(inOut[3].type),
                int(inOut[4].type),
                int(inOut[5].type),
                int(inOut[6].type),
                int(inOut[7].type),
                int(inOut[8].type),
                int(inOut[9].type));
    FT_LOG_INFO("kLINEAR=%d,float=%d,half=%d,int8=%d,int32=%d,bool=%d\n",
                int(TensorFormat::kLINEAR),
                int(DataType::kFLOAT),
                int(DataType::kHALF),
                int(DataType::kINT8),
                int(DataType::kINT32),
                int(DataType::kBOOL));
#endif
    return res;
}

DimsExprs T5DecodingPlugin::getOutputDimensions(int              index,
                                                const DimsExprs* pInputDim,
                                                int              nInputDim,
                                                IExprBuilder&    exprBuilder) noexcept
{
    WHERE_AM_I();
    DimsExprs ret;
    switch (index) {
        case 0:
            ret.nbDims = 3;
            ret.d[0]   = pInputDim[0].d[0];
            ret.d[1]   = exprBuilder.constant(m_.beam_width);
            ret.d[2]   = exprBuilder.constant(m_.max_seq_len);
            break;
        case 1:
            ret.nbDims = 2;
            ret.d[0]   = pInputDim[0].d[0];
            ret.d[1]   = exprBuilder.constant(m_.beam_width);
            break;
        default:  // should NOT be here!
            ;
    }
    return ret;
}

void T5DecodingPlugin::configurePlugin(const DynamicPluginTensorDesc* in,
                                       int                            nbInput,
                                       const DynamicPluginTensorDesc* out,
                                       int                            nbOutput) noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    PRINT_DECODING(int(in[0].desc.type))
}

size_t T5DecodingPlugin::getWorkspaceSize(const PluginTensorDesc* inputs,
                                          int32_t                 nbInputs,
                                          const PluginTensorDesc* outputs,
                                          int32_t                 nbOutputs) const noexcept
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
                              const void* const*      inputs,
                              void* const*            outputs,
                              void*                   workspace,
                              cudaStream_t            stream) noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    m_.batch_size = inputDesc[0].dims.d[0];
    m_.seq_len    = inputDesc[0].dims.d[1];
    PRINT_DECODING(inputDesc[0].type)

    cublasSetStream(cublasHandle_, stream);
    pCublasWrapper_->setStream(stream);

    int    nTopK                       = (inputDesc[2].dims.d[0] == 1) ? 1 : m_.batch_size;
    int    nTopP                       = (inputDesc[3].dims.d[0] == 1) ? 1 : m_.batch_size;
    int    nBeam_search_diversity_rate = (inputDesc[4].dims.d[0] == 1) ? 1 : m_.batch_size;
    int    nTemperature                = (inputDesc[5].dims.d[0] == 1) ? 1 : m_.batch_size;
    int    nLen_penalty                = (inputDesc[6].dims.d[0] == 1) ? 1 : m_.batch_size;
    int    nRepetition_penalty         = (inputDesc[7].dims.d[0] == 1) ? 1 : m_.batch_size;
    int*   pTopK                       = new int[m_.batch_size];
    float* pTopP                       = new float[m_.batch_size];
    float* pBeam_search_diversity_rate = new float[m_.batch_size];
    float* pTemperature                = new float[m_.batch_size];
    float* pLen_penalty                = new float[m_.batch_size];
    float* pRepetition_penalty         = new float[m_.batch_size];

    cudaMemcpyAsync(pTopK, (int*)inputs[2], sizeof(int) * nTopK, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(pTopP, (float*)inputs[3], sizeof(float) * nTopP, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(pBeam_search_diversity_rate,
                    (float*)inputs[4],
                    sizeof(float) * nBeam_search_diversity_rate,
                    cudaMemcpyDeviceToHost,
                    stream);
    cudaMemcpyAsync(pTemperature, (float*)inputs[5], sizeof(float) * nTemperature, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(pLen_penalty, (float*)inputs[6], sizeof(float) * nLen_penalty, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(
        pRepetition_penalty, (float*)inputs[7], sizeof(float) * nRepetition_penalty, cudaMemcpyDeviceToHost, stream);

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
                    std::vector<size_t>{(size_t)m_.batch_size, (size_t)m_.seq_len, (size_t)m_.mem_d_model},
                    (half*)inputs[0]}},
            {"encoder_sequence_length",
             Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{(size_t)m_.batch_size}, (int*)inputs[1]}},
            {"runtime_top_k", Tensor{MEMORY_CPU, TYPE_UINT32, std::vector<size_t>{(size_t)nTopK}, (int*)pTopK}},
            {"runtime_top_p", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{(size_t)nTopP}, (float*)pTopP}},
            {"beam_search_diversity_rate",
             Tensor{MEMORY_CPU,
                    TYPE_FP32,
                    std::vector<size_t>{(size_t)nBeam_search_diversity_rate},
                    (float*)pBeam_search_diversity_rate}},
            {"temperature",
             Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{(size_t)nTemperature}, (float*)pTemperature}},
            {"len_penalty",
             Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{(size_t)nLen_penalty}, (float*)pLen_penalty}},
            {"repetition_penalty",
             Tensor{MEMORY_CPU,
                    TYPE_FP32,
                    std::vector<size_t>{(size_t)nRepetition_penalty},
                    (float*)pRepetition_penalty}}};
        pT5DecodingHalf_->setStream(stream);
        pT5DecodingHalf_->forward(&outputTensor, &inputTensor, pT5DecodingWeightHalf_);
    }
    else {
        std::unordered_map<std::string, Tensor> inputTensor{
            {"encoder_output",
             Tensor{MEMORY_GPU,
                    TYPE_FP32,
                    std::vector<size_t>{(size_t)m_.batch_size, (size_t)m_.seq_len, (size_t)m_.mem_d_model},
                    (float*)inputs[0]}},
            {"encoder_sequence_length",
             Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{(size_t)m_.batch_size}, (int*)inputs[1]}},
            {"runtime_top_k", Tensor{MEMORY_CPU, TYPE_UINT32, std::vector<size_t>{(size_t)nTopK}, (int*)pTopK}},
            {"runtime_top_p", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{(size_t)nTopP}, (float*)pTopP}},
            {"beam_search_diversity_rate",
             Tensor{MEMORY_CPU,
                    TYPE_FP32,
                    std::vector<size_t>{(size_t)nBeam_search_diversity_rate},
                    (float*)pBeam_search_diversity_rate}},
            {"temperature",
             Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{(size_t)nTemperature}, (float*)pTemperature}},
            {"len_penalty",
             Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{(size_t)nLen_penalty}, (float*)pLen_penalty}},
            {"repetition_penalty",
             Tensor{MEMORY_CPU,
                    TYPE_FP32,
                    std::vector<size_t>{(size_t)nRepetition_penalty},
                    (float*)pRepetition_penalty}}};
        pT5DecodingFloat_->setStream(stream);
        pT5DecodingFloat_->forward(&outputTensor, &inputTensor, pT5DecodingWeightFloat_);
    }

    delete[] pTopK;
    delete[] pTopP;
    delete[] pBeam_search_diversity_rate;
    delete[] pTemperature;
    delete[] pLen_penalty;
    delete[] pRepetition_penalty;
    return 0;
}
// class T5DecodingPluginCreator -------------------------------------------------------------------
PluginFieldCollection    T5DecodingPluginCreator::fc_{};
std::vector<PluginField> T5DecodingPluginCreator::attr_;

T5DecodingPluginCreator::T5DecodingPluginCreator()
{
    WHERE_AM_I();
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

T5DecodingPluginCreator::~T5DecodingPluginCreator()
{
    WHERE_AM_I();
}

IPluginV2* T5DecodingPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    int         max_batch_size  = 128;
    int         max_seq_len     = 384;
    int         mem_max_seq_len = max_seq_len;
    int         beam_width      = 4;
    int         useFP16         = 0;
    std::string ckpt_path       = std::string("");

    std::map<std::string, int*> name2pint{
        {"max_batch_size", &max_batch_size},
        {"max_seq_len", &max_seq_len},
        {"mem_max_seq_len", &mem_max_seq_len},
        {"beam_width", &beam_width},
        {"useFP16", &useFP16},
    };
    for (int i = 0; i < fc->nbFields; i++) {
        if (name2pint.find(fc->fields[i].name) != name2pint.end()) {
            *name2pint[fc->fields[i].name] = *(int*)fc->fields[i].data;
        }
        else if (!strcmp(fc->fields[i].name, "ckpt_path")) {
            ckpt_path = std::string((char*)fc->fields[i].data);
        }
    }
    return new T5DecodingPlugin(
        name, max_batch_size, max_seq_len, mem_max_seq_len, beam_width, useFP16, ckpt_path, true);
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
