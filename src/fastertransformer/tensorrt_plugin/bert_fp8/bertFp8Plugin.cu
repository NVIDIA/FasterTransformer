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

#include <cuda.h>

#define ENABLE_FP8

#include "bertFp8Plugin.h"

#include "src/fastertransformer/utils/cuda_fp8_utils.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/logger.h"
#include <fstream>

using namespace nvinfer1;
namespace ft = fastertransformer;

namespace fastertransformer {

REGISTER_TENSORRT_PLUGIN(BertFp8PluginCreator);

BertFp8Plugin::BertFp8Plugin(BertFp8Config cfg, std::string weightDirPath):
    mBertFp8Config(cfg), mWeightDirPath(weightDirPath)
{
}

// assume this is called ONCE PER GPU, via a deserializeCudaEngine() call from the client
// since BertFp8Weights::deserialize() copies weights to the current GPU's device memory
BertFp8Plugin::BertFp8Plugin(const void* data, size_t length)
{
    const uint8_t* tmp = (const uint8_t*)data;
    mBertFp8Config     = BertFp8Config(tmp);
    const auto cfg     = mBertFp8Config;
    mBertWeights.reset(new ft::BertFP8Weight<fp8_t, bf16_t>(cfg.hidden_units,
                                                            cfg.num_heads,
                                                            cfg.size_per_head,
                                                            cfg.intermediate_size,
                                                            cfg.num_layers,
                                                            cfg.vocab_size,
                                                            cfg.max_position_embeddings,
                                                            cfg.token_type_vocab_size,
                                                            1,
                                                            1,
                                                            cfg.fp8_mode,
                                                            true,
                                                            true));
    mBertWeights->deserialize(tmp);  // copies weights to current GPU's device mem
}

int BertFp8Plugin::initialize() noexcept
{
    if (mInitialized) {
        return 0;
    }

    const auto cfg = mBertFp8Config;

    // load weights directly from weightDirPath
    // this should only be performed during engine build phase and then weights are serialized
    // at runtime, deserialize() is called which will load a copy of these weights
    if (mBertWeights == nullptr) {
        mBertWeights.reset(new ft::BertFP8Weight<fp8_t, bf16_t>(cfg.hidden_units,
                                                                cfg.num_heads,
                                                                cfg.size_per_head,
                                                                cfg.intermediate_size,
                                                                cfg.num_layers,
                                                                cfg.vocab_size,
                                                                cfg.max_position_embeddings,
                                                                cfg.token_type_vocab_size,
                                                                1,
                                                                1,
                                                                cfg.fp8_mode,
                                                                true,
                                                                true));

        mBertWeights->loadModel(mWeightDirPath);
        mBertWeights->transposeWeight();
    }

    mAllocator.reset(new ft::Allocator<ft::AllocatorType::CUDA>(ft::getDevice()));
    mCublasCtx.reset(new CublasCtx(mAllocator.get()));

    ft::AttentionType attention_type =
        ft::getAttentionType<fp8_t>(cfg.size_per_head, ft::getSMVersion(), cfg.remove_padding, cfg.max_seq_len);

    mBertModel.reset(new ft::BertFP8<fp8_t, bf16_t>(cfg.num_heads,
                                                    cfg.size_per_head,
                                                    cfg.hidden_units,
                                                    cfg.intermediate_size,
                                                    cfg.num_layers,
                                                    mTensorPara,
                                                    mPipelinePara,
                                                    ft::getSMVersion(),
                                                    1.0f,
                                                    mCublasCtx->mStream,
                                                    mCublasCtx->mCublasWrapper.get(),
                                                    mAllocator.get(),
                                                    false,
                                                    attention_type,
                                                    false,
                                                    ft::ActivationType::Gelu,
                                                    ft::LayerNormType::post_layernorm,
                                                    cfg.fp8_mode));

    mInitialized = true;
    return 0;
}

const char* BertFp8Plugin::getPluginType() const noexcept
{
    return "BertFp8Plugin";
}

const char* BertFp8Plugin::getPluginVersion() const noexcept
{
    return "1";
}

void BertFp8Plugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* BertFp8Plugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void BertFp8Plugin::destroy() noexcept
{
    delete this;
}

IPluginV2DynamicExt* BertFp8Plugin::clone() const noexcept
{
    BertFp8Plugin* ret = new BertFp8Plugin(mBertFp8Config, mWeightDirPath);
    ret->mBertWeights  = mBertWeights;
    ret->initialize();

    return ret;
}

void BertFp8Plugin::attachToContext(cudnnContext*, cublasContext*, IGpuAllocator*) noexcept
{
    // this->initialize();
}

int BertFp8Plugin::getNbOutputs() const noexcept
{
    return 1;
}

DimsExprs BertFp8Plugin::getOutputDimensions(int              outputIndex,
                                             const DimsExprs* inputs,
                                             int              nbInputs,
                                             IExprBuilder&    exprBuilder) noexcept
{

    FT_CHECK(outputIndex >= 0 && outputIndex < this->getNbOutputs());
    FT_CHECK(nbInputs == 3);

    DimsExprs output(inputs[0]);
    output.nbDims = 3;
    output.d[0]   = inputs[0].d[0];
    output.d[1]   = exprBuilder.constant(mBertFp8Config.max_seq_len);
    output.d[2]   = exprBuilder.constant(mBertFp8Config.hidden_units);

    return output;
}

bool BertFp8Plugin::supportsFormatCombination(int                     pos,
                                              const PluginTensorDesc* inOut,
                                              int                     nbInputs,
                                              int                     nbOutputs) noexcept
{
    if (inOut[pos].format != TensorFormat::kLINEAR) {
        return false;
    }

    if (nbInputs != 3 || nbOutputs != 1) {
        printf("Wrong input or output count %d %d\n", nbInputs, nbOutputs);
        return false;
    }

    // inputs
    if (pos == 0 && inOut[pos].type != nvinfer1::DataType::kINT32) {
        return false;
    }
    if (pos == 1 && inOut[pos].type != nvinfer1::DataType::kINT32) {
        return false;
    }
    if (pos == 2 && inOut[pos].type != nvinfer1::DataType::kINT32) {
        return false;
    }

    // outputs
    if (pos == 3 && inOut[pos].type != nvinfer1::DataType::kHALF) {
        return false;
    }

    return true;
}

void BertFp8Plugin::configurePlugin(const DynamicPluginTensorDesc* in,
                                    int                            nbInputs,
                                    const DynamicPluginTensorDesc* out,
                                    int                            nbOutputs) noexcept
{
}

void BertFp8Plugin::terminate() noexcept {}

size_t BertFp8Plugin::getWorkspaceSize(const PluginTensorDesc* inputs,
                                       int                     nbInputs,
                                       const PluginTensorDesc* outputs,
                                       int                     nbOutputs) const noexcept
{
    return 0;
}

int BertFp8Plugin::enqueue(const PluginTensorDesc* inputDesc,
                           const PluginTensorDesc* outputDesc,
                           const void* const*      inputs,
                           void* const*            outputs,
                           void*                   workspace,
                           cudaStream_t            stream) noexcept
{
    int32_t        batchSize        = inputDesc[0].dims.d[0];
    int32_t        maxSeqLenInBatch = inputDesc[0].dims.d[1];
    const int32_t* inputIds         = static_cast<const int32_t*>(inputs[0]);
    const int32_t* tokenTypeIds     = static_cast<const int32_t*>(inputs[1]);
    const int32_t* sequenceLengths  = static_cast<const int32_t*>(inputs[2]);

    size_t batchSize_s   = batchSize;
    size_t maxSeqLen_s   = maxSeqLenInBatch;
    size_t hiddenUnits_s = mBertFp8Config.hidden_units;

    auto inputTensors = ft::TensorMap(std::unordered_map<std::string, ft::Tensor>{
        {"input_ids",
         ft::Tensor{ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{batchSize_s, maxSeqLen_s}, inputIds}},
        {"sequence_lengths",
         ft::Tensor{ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{batchSize_s}, sequenceLengths}},
        {"token_type_ids",
         ft::Tensor{ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{batchSize_s, maxSeqLen_s}, tokenTypeIds}}});

    auto outputTensors = ft::TensorMap(std::unordered_map<std::string, ft::Tensor>{
        {"output_hidden_state",
         ft::Tensor{
             ft::MEMORY_GPU, ft::getTensorType<half>(), {batchSize_s, maxSeqLen_s, hiddenUnits_s}, outputs[0]}}});

    FT_CHECK(cudaEventRecord(mSyncEvent.get(), stream) == cudaSuccess);
    FT_CHECK(cudaStreamWaitEvent(mCublasCtx->mStream, mSyncEvent.get(), 0) == cudaSuccess);

    mBertModel->forward(&outputTensors, &inputTensors, mBertWeights.get());

    FT_CHECK(cudaEventRecord(mSyncEvent.get(), mCublasCtx->mStream) == cudaSuccess);
    FT_CHECK(cudaStreamWaitEvent(stream, mSyncEvent.get(), 0) == cudaSuccess);

    return 0;
}

size_t BertFp8Plugin::getSerializationSize() const noexcept
{
    auto sz = mBertWeights->getSerializationSize();
    sz += mBertFp8Config.getSerializationSize();
    return sz;
}

void BertFp8Plugin::serialize(void* buffer) const noexcept
{
    uint8_t* tmp = (uint8_t*)buffer;
    mBertFp8Config.serialize(tmp);
    mBertWeights->serialize(tmp);
}

nvinfer1::DataType
BertFp8Plugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    return nvinfer1::DataType::kHALF;
}

const char* BertFp8PluginCreator::getPluginName() const noexcept
{
    return "BertFp8Plugin";
}

const char* BertFp8PluginCreator::getPluginVersion() const noexcept
{
    return "1";
}

const PluginFieldCollection* BertFp8PluginCreator::getFieldNames() noexcept
{
    return nullptr;
}

void BertFp8PluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* BertFp8PluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

IPluginV2DynamicExt* BertFp8PluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    int32_t num_heads;
    int32_t size_per_head;
    int32_t num_layers;
    int32_t max_seq_len;
    int32_t vocab_size;
    int32_t max_position_embeddings;
    int32_t token_type_vocab_size;
    int32_t remove_padding;
    int32_t fp8_mode;

    std::map<std::string, int32_t*> name2pint = {{"num_heads", &num_heads},
                                                 {"size_per_head", &size_per_head},
                                                 {"num_layers", &num_layers},
                                                 {"max_seq_len", &max_seq_len},
                                                 {"vocab_size", &vocab_size},
                                                 {"max_position_embeddings", &max_position_embeddings},
                                                 {"token_type_vocab_size", &token_type_vocab_size},
                                                 {"remove_padding", &remove_padding},
                                                 {"fp8_mode", &fp8_mode}};

    size_t found = 0;
    std::for_each(fc->fields, fc->fields + fc->nbFields, [&name2pint, &found](const auto& f) {
        auto iter = name2pint.find(f.name);
        if (iter != name2pint.end()) {
            *(iter->second) = *(int32_t*)f.data;
            found++;
        }
    });

    std::string                         weightDirPath;
    std::map<std::string, std::string*> name2pstr = {{"weightDirPath", &weightDirPath}};

    std::for_each(fc->fields, fc->fields + fc->nbFields, [&name2pstr, &found](const auto& f) {
        auto iter = name2pstr.find(f.name);
        if (iter != name2pstr.end()) {
            *(iter->second) = std::string((const char*)f.data, f.length);
            found++;
        }
    });

    FT_CHECK(found == name2pint.size() + name2pstr.size());

    BertFp8Config cfg{num_heads,
                      size_per_head,
                      num_layers,
                      max_seq_len,
                      vocab_size,
                      max_position_embeddings,
                      token_type_vocab_size,
                      remove_padding,
                      fp8_mode};
    return new BertFp8Plugin(cfg, weightDirPath);
}

IPluginV2DynamicExt*
BertFp8PluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new BertFp8Plugin(serialData, serialLength);
}
}  // namespace fastertransformer
