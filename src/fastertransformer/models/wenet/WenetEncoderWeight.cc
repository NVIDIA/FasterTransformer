/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/wenet/WenetEncoderWeight.h"
#include "src/fastertransformer/utils/logger.h"

namespace fastertransformer {

template<typename T>
WenetEncoderWeight<T>::WenetEncoderWeight(const size_t head_num,
                                          const size_t size_per_head,
                                          const size_t inter_size,
                                          const size_t d_model,
                                          const size_t vocab_size,
                                          const size_t conv_module_kernel_size,
                                          const size_t feature_size,
                                          const size_t max_len,
                                          const size_t num_layer,
                                          const bool   use_layernorm_in_conv_module):
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    d_model_(d_model),
    vocab_size_(vocab_size),
    conv_module_kernel_size_(conv_module_kernel_size),
    feature_size_(feature_size),
    max_len_(max_len),
    num_layer_(num_layer),
    real_weights_num_(13),
    use_layernorm_in_conv_module_(use_layernorm_in_conv_module)
{
    FT_LOG_DEBUG("WenetEncoderWeight " + std::string(__func__) + " start");
    initialize();
    mallocWeights();
    setWeightPtr();
    encoder_layer_weights.clear();
    for (size_t l = 0; l < num_layer_; l++) {
        encoder_layer_weights.push_back(new WenetEncoderLayerWeight<T>(
            l, head_num_, size_per_head_, inter_size_, conv_module_kernel_size_, use_layernorm_in_conv_module_));
    }
    FT_LOG_DEBUG("WenetEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
void WenetEncoderWeight<T>::initialize()
{
    FT_LOG_DEBUG("WenetEncoderWeight " + std::string(__func__) + " start");
    int hidden_size  = head_num_ * size_per_head_;
    weights_size[0]  = hidden_size;
    weights_size[1]  = hidden_size;
    weights_size[2]  = hidden_size * vocab_size_;
    weights_size[3]  = vocab_size_;
    weights_size[4]  = feature_size_;
    weights_size[5]  = feature_size_;
    weights_size[6]  = hidden_size * 3 * 3;
    weights_size[7]  = hidden_size;
    weights_size[8]  = hidden_size * hidden_size * 3 * 3;
    weights_size[9]  = hidden_size;
    weights_size[10] = ((feature_size_ - 1) / 4) * hidden_size * hidden_size;
    weights_size[11] = hidden_size;
    weights_size[12] = max_len_ * hidden_size;

    FT_LOG_DEBUG("WenetEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
WenetEncoderWeight<T>::~WenetEncoderWeight()
{
    FT_LOG_DEBUG("WenetEncoderWeight " + std::string(__func__) + " start");

    if (is_maintain_buffer == true) {
        for (int i = 0; i < real_weights_num_; i++) {
            deviceFree(weights_ptr[i]);
        }
        post_transformer_layernorm_weights.gamma = nullptr;
        post_transformer_layernorm_weights.beta  = nullptr;
        ctc_lo_weight.kernel                     = nullptr;
        ctc_lo_weight.bias                       = nullptr;
        cmvn_weights.mean                        = nullptr;
        cmvn_weights.istd                        = nullptr;
        embed_conv1_weights.kernel               = nullptr;
        embed_conv1_weights.bias                 = nullptr;
        embed_conv2_weights.kernel               = nullptr;
        embed_conv2_weights.bias                 = nullptr;
        embed_out_weights.kernel                 = nullptr;
        embed_out_weights.bias                   = nullptr;
        positional_encoding_weights.data         = nullptr;

        is_maintain_buffer = false;
    }
    for (size_t i = 0; i < num_layer_; i++) {
        delete encoder_layer_weights[i];
    }
    FT_LOG_DEBUG("WenetEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
WenetEncoderWeight<T>::WenetEncoderWeight(const WenetEncoderWeight& other):
    head_num_(other.head_num_),
    size_per_head_(other.size_per_head_),
    inter_size_(other.inter_size_),
    d_model_(other.d_model_),
    vocab_size_(other.vocab_size_),
    conv_module_kernel_size_(other.conv_module_kernel_size_),
    feature_size_(other.feature_size_),
    max_len_(other.max_len_),
    num_layer_(other.num_layer_),
    real_weights_num_(other.real_weights_num_),
    use_layernorm_in_conv_module_(other.use_layernorm_in_conv_module_)
{
    FT_LOG_DEBUG("WenetEncoderWeight " + std::string(__func__) + " start");
    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    setWeightPtr();

    encoder_layer_weights.clear();
    for (size_t i = 0; i < num_layer_; i++) {
        encoder_layer_weights.push_back(new WenetEncoderLayerWeight<T>(*other.encoder_layer_weights[i]));
    }
    FT_LOG_DEBUG("WenetEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
WenetEncoderWeight<T>& WenetEncoderWeight<T>::operator=(const WenetEncoderWeight& other)
{
    FT_LOG_DEBUG("WenetEncoderWeight " + std::string(__func__) + " start");

    head_num_                     = other.head_num_;
    size_per_head_                = other.size_per_head_;
    inter_size_                   = other.inter_size_;
    d_model_                      = other.d_model_;
    vocab_size_                   = other.vocab_size_;
    conv_module_kernel_size_      = other.conv_module_kernel_size_;
    feature_size_                 = other.feature_size_;
    max_len_                      = other.max_len_;
    num_layer_                    = other.num_layer_;
    real_weights_num_             = other.real_weights_num_;
    use_layernorm_in_conv_module_ = other.use_layernorm_in_conv_module_;
    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    setWeightPtr();

    encoder_layer_weights.clear();
    for (size_t i = 0; i < num_layer_; i++) {
        encoder_layer_weights.push_back(new WenetEncoderLayerWeight<T>(*other.encoder_layer_weights[i]));
    }
    FT_LOG_DEBUG("WenetEncoderWeight " + std::string(__func__) + " end");

    return *this;
}

template<typename T>
void WenetEncoderWeight<T>::setWeightPtr()
{
    FT_LOG_DEBUG("WenetEncoderWeight " + std::string(__func__) + " start");
    post_transformer_layernorm_weights.gamma = weights_ptr[0];
    post_transformer_layernorm_weights.beta  = weights_ptr[1];
    ctc_lo_weight.kernel                     = weights_ptr[2];
    ctc_lo_weight.bias                       = weights_ptr[3];
    cmvn_weights.mean                        = weights_ptr[4];
    cmvn_weights.istd                        = weights_ptr[5];
    embed_conv1_weights.kernel               = weights_ptr[6];
    embed_conv1_weights.bias                 = weights_ptr[7];
    embed_conv2_weights.kernel               = weights_ptr[8];
    embed_conv2_weights.bias                 = weights_ptr[9];
    embed_out_weights.kernel                 = weights_ptr[10];
    embed_out_weights.bias                   = weights_ptr[11];
    positional_encoding_weights.data         = weights_ptr[12];
    FT_LOG_DEBUG("WenetEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
void WenetEncoderWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG("WenetEncoderWeight " + std::string(__func__) + " start");
    for (int i = 0; i < real_weights_num_; i++) {
        deviceMalloc(&weights_ptr[i], weights_size[i]);
    }
    is_maintain_buffer = true;
    FT_LOG_DEBUG("WenetEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
void WenetEncoderWeight<T>::loadModel(std::string dir_path)
{
    FT_LOG_DEBUG("WenetEncoderWeight " + std::string(__func__) + " start");
    FtCudaDataType model_file_type = getModelFileType(dir_path + "config.ini", "encoder");
    FT_CHECK(is_maintain_buffer == true);

    std::vector<std::string> weights_name;
    std::string              name_prefix = "encoder.";
    weights_name.push_back(name_prefix + "after_norm.weight");
    weights_name.push_back(name_prefix + "after_norm.bias");
    weights_name.push_back("ctc.ctc_lo.weight");
    weights_name.push_back("ctc.ctc_lo.bias");
    weights_name.push_back("encoder.global_cmvn.mean");
    weights_name.push_back("encoder.global_cmvn.istd");
    weights_name.push_back("encoder.embed.conv.0.weight");
    weights_name.push_back("encoder.embed.conv.0.bias");
    weights_name.push_back("encoder.embed.conv.2.weight");
    weights_name.push_back("encoder.embed.conv.2.bias");
    weights_name.push_back("encoder.embed.out.0.weight");
    weights_name.push_back("encoder.embed.out.0.bias");
    weights_name.push_back("encoder.positional.encoding.data");

    for (size_t i = 0; i < weights_name.size(); ++i) {
        loadWeightFromBin<T>(weights_ptr[i], {weights_size[i]}, dir_path + weights_name[i] + ".bin", model_file_type);
    }

    for (size_t l = 0; l < num_layer_; l++) {
        encoder_layer_weights[l]->loadModel(dir_path, model_file_type);
    }
    FT_LOG_DEBUG("WenetEncoderWeight " + std::string(__func__) + " end");
}

template struct WenetEncoderWeight<float>;
template struct WenetEncoderWeight<half>;

}  // namespace fastertransformer
