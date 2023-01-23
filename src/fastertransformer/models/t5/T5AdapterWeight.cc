
/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/t5/T5AdapterWeight.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
T5AdapterWeight<T>::T5AdapterWeight(const size_t d_model,
                                    const size_t adapter_inter_size,
                                    const size_t tensor_para_size,
                                    const size_t tensor_para_rank):
    d_model_{d_model},
    adapter_inter_size_{adapter_inter_size},
    tensor_para_size_{tensor_para_size},
    tensor_para_rank_{tensor_para_rank}
{
    FT_LOG_DEBUG("T5AdapterWeight " + std::string(__func__) + " start");
    initialize();
    mallocWeights();
    setWeightPtr();
    FT_LOG_DEBUG("T5AdapterWeight " + std::string(__func__) + " end");
}

template<typename T>
void T5AdapterWeight<T>::initialize()
{
    FT_LOG_DEBUG("T5AdapterWeight " + std::string(__func__) + " start");

    if (enabled()) {
        auto const adapter_weights_in                           = d_model_ * (adapter_inter_size_ / tensor_para_size_);
        auto const adapter_weights_out                          = (adapter_inter_size_ / tensor_para_size_) * d_model_;
        adapter_weights_size_[AdapterWeights::kAttentionInput]  = adapter_weights_in;
        adapter_weights_size_[AdapterWeights::kAttentionOutput] = adapter_weights_out;
        adapter_weights_size_[AdapterWeights::kAttentionNormGamma] = d_model_;
        adapter_weights_size_[AdapterWeights::kAttentionNormBeta]  = d_model_;
        adapter_weights_size_[AdapterWeights::kFfnInput]           = adapter_weights_in;
        adapter_weights_size_[AdapterWeights::kFfnOutput]          = adapter_weights_out;
        adapter_weights_size_[AdapterWeights::kFfnNormGamma]       = d_model_;
        adapter_weights_size_[AdapterWeights::kFfnNormBeta]        = d_model_;
    }

    FT_LOG_DEBUG("T5AdapterWeight " + std::string(__func__) + " end");
}

template<typename T>
T5AdapterWeight<T>::~T5AdapterWeight()
{
    FT_LOG_DEBUG("T5AdapterWeight " + std::string(__func__) + " start");
    if (maintain_adapter_buffer_) {
        for (auto w : adapter_weights_ptr_) {
            deviceFree(w);
        }
        after_attention_adapter_weights_.input_weight().kernel   = nullptr;
        after_attention_adapter_weights_.output_weight().kernel  = nullptr;
        after_attention_adapter_weights_.layer_norm_weight.gamma = nullptr;
        after_attention_adapter_weights_.layer_norm_weight.beta  = nullptr;
        after_ffn_adapter_weights_.input_weight().kernel         = nullptr;
        after_ffn_adapter_weights_.output_weight().kernel        = nullptr;
        after_ffn_adapter_weights_.layer_norm_weight.gamma       = nullptr;
        after_ffn_adapter_weights_.layer_norm_weight.beta        = nullptr;
        maintain_adapter_buffer_                                 = false;
    }
    FT_LOG_DEBUG("T5AdapterWeight " + std::string(__func__) + " end");
}

template<typename T>
T5AdapterWeight<T>::T5AdapterWeight(const T5AdapterWeight& other):
    d_model_{other.d_model_},
    adapter_inter_size_{other.adapter_inter_size_},
    tensor_para_size_{other.tensor_para_size_},
    tensor_para_rank_{other.tensor_para_rank_}
{
    FT_LOG_DEBUG("T5AdapterWeight " + std::string(__func__) + " start");
    initialize();
    mallocWeights();
    if (enabled()) {
        for (int i = 0; i < AdapterWeights::kNum; i++) {
            cudaD2Dcpy(adapter_weights_ptr_[i], other.adapter_weights_ptr_[i], adapter_weights_size_[i]);
        }
    }
    setWeightPtr();
    FT_LOG_DEBUG("T5AdapterWeight " + std::string(__func__) + " end");
}

template<typename T>
T5AdapterWeight<T>& T5AdapterWeight<T>::operator=(const T5AdapterWeight<T>& other)
{
    if (this == &other) {
        return *this;
    }

    FT_LOG_DEBUG("T5AdapterWeight " + std::string(__func__) + " start");

    d_model_            = other.d_model_;
    adapter_inter_size_ = other.adapter_inter_size_;
    tensor_para_size_   = other.tensor_para_size_;
    tensor_para_rank_   = other.tensor_para_rank_;
    initialize();
    mallocWeights();
    if (enabled()) {
        for (int i = 0; i < AdapterWeights::kNum; i++) {
            cudaD2Dcpy(adapter_weights_ptr_[i], other.adapter_weights_ptr_[i], adapter_weights_size_[i]);
        }
    }
    setWeightPtr();
    FT_LOG_DEBUG("T5AdapterWeight " + std::string(__func__) + " end");

    return *this;
}

template<typename T>
void T5AdapterWeight<T>::setWeightPtr()
{
    FT_LOG_DEBUG("T5AdapterWeight " + std::string(__func__) + " start");
    if (enabled()) {
        after_attention_adapter_weights_.input_weight().kernel = adapter_weights_ptr_[AdapterWeights::kAttentionInput];
        after_attention_adapter_weights_.output_weight().kernel =
            adapter_weights_ptr_[AdapterWeights::kAttentionOutput];
        after_attention_adapter_weights_.layer_norm_weight.gamma =
            adapter_weights_ptr_[AdapterWeights::kAttentionNormGamma];
        after_attention_adapter_weights_.layer_norm_weight.beta =
            adapter_weights_ptr_[AdapterWeights::kAttentionNormBeta];
        after_ffn_adapter_weights_.input_weight().kernel   = adapter_weights_ptr_[AdapterWeights::kFfnInput];
        after_ffn_adapter_weights_.output_weight().kernel  = adapter_weights_ptr_[AdapterWeights::kFfnOutput];
        after_ffn_adapter_weights_.layer_norm_weight.gamma = adapter_weights_ptr_[AdapterWeights::kFfnNormGamma];
        after_ffn_adapter_weights_.layer_norm_weight.beta  = adapter_weights_ptr_[AdapterWeights::kFfnNormBeta];
    }

    FT_LOG_DEBUG("T5AdapterWeight " + std::string(__func__) + " end");
}

template<typename T>
void T5AdapterWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG("T5AdapterWeight " + std::string(__func__) + " start");
    if (enabled()) {
        for (int i = 0; i < AdapterWeights::kNum; i++) {
            deviceMalloc(&adapter_weights_ptr_[i], adapter_weights_size_[i]);
        }
        maintain_adapter_buffer_ = true;
    }
    FT_LOG_DEBUG("T5AdapterWeight " + std::string(__func__) + " end");
}

template<typename T>
void T5AdapterWeight<T>::loadModel(std::string const& dir_path, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG("T5AdapterWeight " + std::string(__func__) + " start");

    if (enabled()) {
        FT_CHECK(maintain_adapter_buffer_);

        const auto tp_rank = std::to_string(tensor_para_rank_);

        loadWeightFromBin<T>(adapter_weights_ptr_[AdapterWeights::kAttentionInput],
                             {adapter_weights_size_[AdapterWeights::kAttentionInput]},
                             dir_path + "after_attention_adapter.DenseSiluDense.wi.weight." + tp_rank + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(adapter_weights_ptr_[AdapterWeights::kAttentionOutput],
                             {adapter_weights_size_[AdapterWeights::kAttentionOutput]},
                             dir_path + "after_attention_adapter.DenseSiluDense.wo.weight." + tp_rank + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(adapter_weights_ptr_[AdapterWeights::kAttentionNormGamma],
                             {adapter_weights_size_[AdapterWeights::kAttentionNormGamma]},
                             dir_path + "after_attention_adapter.layer_norm.weight.bin",
                             model_file_type);
        loadWeightFromBin<T>(adapter_weights_ptr_[AdapterWeights::kAttentionNormBeta],
                             {adapter_weights_size_[AdapterWeights::kAttentionNormBeta]},
                             dir_path + "after_attention_adapter.layer_norm.bias.bin",
                             model_file_type);
        loadWeightFromBin<T>(adapter_weights_ptr_[AdapterWeights::kFfnInput],
                             {adapter_weights_size_[AdapterWeights::kFfnInput]},
                             dir_path + "after_ffn_adapter.DenseSiluDense.wi.weight." + tp_rank + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(adapter_weights_ptr_[AdapterWeights::kFfnOutput],
                             {adapter_weights_size_[AdapterWeights::kFfnOutput]},
                             dir_path + "after_ffn_adapter.DenseSiluDense.wo.weight." + tp_rank + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(adapter_weights_ptr_[AdapterWeights::kFfnNormGamma],
                             {adapter_weights_size_[AdapterWeights::kFfnNormGamma]},
                             dir_path + "after_ffn_adapter.layer_norm.weight.bin",
                             model_file_type);
        loadWeightFromBin<T>(adapter_weights_ptr_[AdapterWeights::kFfnNormBeta],
                             {adapter_weights_size_[AdapterWeights::kFfnNormBeta]},
                             dir_path + "after_ffn_adapter.layer_norm.bias.bin",
                             model_file_type);
    }

    FT_LOG_DEBUG("T5AdapterWeight " + std::string(__func__) + " end");
}

template struct T5AdapterWeight<float>;
template struct T5AdapterWeight<half>;
#ifdef ENABLE_BF16
template struct T5AdapterWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
