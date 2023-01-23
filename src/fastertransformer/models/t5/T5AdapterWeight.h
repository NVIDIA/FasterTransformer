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

#pragma once

#include "src/fastertransformer/layers/adapter_layers/LinearAdapterWeight.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include <string>

namespace fastertransformer {

template<typename T>
struct T5AdapterWeight {

    T5AdapterWeight() = default;
    T5AdapterWeight(size_t d_model, size_t tensor_para_size, size_t tensor_para_rank, size_t adapter_inter_size);
    ~T5AdapterWeight();
    T5AdapterWeight(const T5AdapterWeight& other);
    T5AdapterWeight& operator=(const T5AdapterWeight& other);

    LinearAdapterWeight<T> after_attention_adapter_weights_;
    LinearAdapterWeight<T> after_ffn_adapter_weights_;

    void loadModel(std::string const& dir_path, FtCudaDataType model_file_type);

    bool enabled() const
    {
        return adapter_inter_size_ > 0;
    }

    void setAdapterInterSize(size_t adapter_inter_size)
    {
        adapter_inter_size_ = adapter_inter_size;
    }

private:
    void setWeightPtr();
    void mallocWeights();
    void initialize();

    size_t d_model_{};
    size_t adapter_inter_size_{};
    size_t tensor_para_size_{};
    size_t tensor_para_rank_{};

    enum AdapterWeights : int {
        kAttentionInput,
        kAttentionOutput,
        kAttentionNormGamma,
        kAttentionNormBeta,
        kFfnInput,
        kFfnOutput,
        kFfnNormGamma,
        kFfnNormBeta,
        kNum  // Must be the last one
    };

    T*     adapter_weights_ptr_[AdapterWeights::kNum];
    size_t adapter_weights_size_[AdapterWeights::kNum];
    bool   maintain_adapter_buffer_ = false;
};

}  // namespace fastertransformer
