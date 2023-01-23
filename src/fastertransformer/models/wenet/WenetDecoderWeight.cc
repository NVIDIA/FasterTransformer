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

#include "src/fastertransformer/models/wenet/WenetDecoderWeight.h"
#include "src/fastertransformer/utils/logger.h"

namespace fastertransformer {

template<typename T>
WenetDecoderWeight<T>::WenetDecoderWeight(const size_t head_num,
                                          const size_t size_per_head,
                                          const size_t inter_size,
                                          const size_t num_layer,
                                          const size_t vocab_size,
                                          const size_t max_len):
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    vocab_size_(vocab_size),
    max_len_(max_len),
    real_weights_num_(6)
{
    FT_LOG_DEBUG("WenetDecoderWeight " + std::string(__func__) + " start");
    initialize();
    mallocWeights();
    setWeightPtr();

    int hidden_size = head_num_ * size_per_head_;
    decoder_layer_weights.clear();
    for (size_t l = 0; l < num_layer_; l++) {
        decoder_layer_weights.push_back(new WenetDecoderLayerWeight<T>(l, hidden_size, inter_size_, hidden_size));
    }
    FT_LOG_DEBUG("WenetDecoderWeight " + std::string(__func__) + " end");
}

template<typename T>
void WenetDecoderWeight<T>::initialize()
{
    FT_LOG_DEBUG("WenetDecoderWeight " + std::string(__func__) + " start");
    int hidden_size = head_num_ * size_per_head_;
    weights_size[0] = hidden_size;
    weights_size[1] = hidden_size;
    weights_size[2] = hidden_size * vocab_size_;
    weights_size[3] = vocab_size_;
    weights_size[4] = vocab_size_ * hidden_size;
    weights_size[5] = max_len_ * hidden_size;
    FT_LOG_DEBUG("WenetDecoderWeight " + std::string(__func__) + " end");
}

template<typename T>
WenetDecoderWeight<T>::~WenetDecoderWeight()
{
    FT_LOG_DEBUG("WenetDecoderWeight " + std::string(__func__) + " start");

    if (is_maintain_buffer == true) {
        for (int i = 0; i < real_weights_num_; i++) {
            deviceFree(weights_ptr[i]);
        }
        after_norm_weights.gamma         = nullptr;
        after_norm_weights.beta          = nullptr;
        output_layer_weights.kernel      = nullptr;
        output_layer_weights.bias        = nullptr;
        decoder_embed_weights.data       = nullptr;
        positional_encoding_weights.data = nullptr;
        is_maintain_buffer               = false;
    }
    for (size_t i = 0; i < num_layer_; i++) {
        delete decoder_layer_weights[i];
    }
    FT_LOG_DEBUG("WenetDecoderWeight " + std::string(__func__) + " end");
}

template<typename T>
WenetDecoderWeight<T>::WenetDecoderWeight(const WenetDecoderWeight& other):
    head_num_(other.head_num_),
    size_per_head_(other.size_per_head_),
    inter_size_(other.inter_size_),
    num_layer_(other.num_layer_),
    vocab_size_(other.vocab_size_),
    max_len_(other.max_len_),
    real_weights_num_(other.real_weights_num_)
{
    FT_LOG_DEBUG("WenetDecoderWeight " + std::string(__func__) + " start");
    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    setWeightPtr();

    decoder_layer_weights.clear();
    for (size_t i = 0; i < num_layer_; i++) {
        decoder_layer_weights.push_back(new WenetDecoderLayerWeight<T>(*other.decoder_layer_weights[i]));
    }
    FT_LOG_DEBUG("WenetDecoderWeight " + std::string(__func__) + " end");
}

template<typename T>
WenetDecoderWeight<T>& WenetDecoderWeight<T>::operator=(const WenetDecoderWeight& other)
{
    FT_LOG_DEBUG("WenetDecoderWeight " + std::string(__func__) + " start");

    head_num_         = other.head_num_;
    size_per_head_    = other.size_per_head_;
    inter_size_       = other.inter_size_;
    num_layer_        = other.num_layer_;
    vocab_size_       = other.vocab_size_;
    max_len_          = other.max_len_;
    real_weights_num_ = other.real_weights_num_;
    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    setWeightPtr();

    decoder_layer_weights.clear();
    for (size_t i = 0; i < num_layer_; i++) {
        decoder_layer_weights.push_back(new WenetDecoderLayerWeight<T>(*other.decoder_layer_weights[i]));
    }
    FT_LOG_DEBUG("WenetDecoderWeight " + std::string(__func__) + " end");

    return *this;
}

template<typename T>
void WenetDecoderWeight<T>::setWeightPtr()
{
    FT_LOG_DEBUG("WenetDecoderWeight " + std::string(__func__) + " start");
    after_norm_weights.gamma         = weights_ptr[0];
    after_norm_weights.beta          = weights_ptr[1];
    output_layer_weights.kernel      = weights_ptr[2];
    output_layer_weights.bias        = weights_ptr[3];
    decoder_embed_weights.data       = weights_ptr[4];
    positional_encoding_weights.data = weights_ptr[5];
    FT_LOG_DEBUG("WenetDecoderWeight " + std::string(__func__) + " end");
}

template<typename T>
void WenetDecoderWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG("WenetDecoderWeight " + std::string(__func__) + " start");
    for (int i = 0; i < real_weights_num_; i++) {
        deviceMalloc(&weights_ptr[i], weights_size[i]);
    }
    is_maintain_buffer = true;
    FT_LOG_DEBUG("WenetDecoderWeight " + std::string(__func__) + " end");
}

template<typename T>
void WenetDecoderWeight<T>::loadModel(std::string dir_path)
{
    FT_LOG_DEBUG("WenetDecoderWeight " + std::string(__func__) + " start");
    FtCudaDataType model_file_type = getModelFileType(dir_path + "/config.ini", "decoder");
    FT_CHECK(is_maintain_buffer == true);

    std::vector<std::string> weights_name;
    std::string              name_prefix = "decoder.";
    weights_name.push_back(name_prefix + "after_norm.weight");
    weights_name.push_back(name_prefix + "after_norm.bias");
    weights_name.push_back(name_prefix + "output_layer.weight");
    weights_name.push_back(name_prefix + "output_layer.bias");
    weights_name.push_back("decoder.embed.0.weight");
    weights_name.push_back("decoder.positional.encoding.data");

    for (size_t i = 0; i < weights_name.size(); ++i) {
        loadWeightFromBin<T>(weights_ptr[i], {weights_size[i]}, dir_path + weights_name[i] + ".bin", model_file_type);
    }

    for (size_t l = 0; l < num_layer_; l++) {
        decoder_layer_weights[l]->loadModel(dir_path, model_file_type);
    }
    FT_LOG_DEBUG("WenetDecoderWeight " + std::string(__func__) + " end");
}

template struct WenetDecoderWeight<float>;
template struct WenetDecoderWeight<half>;

}  // namespace fastertransformer
