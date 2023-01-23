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

#pragma once

#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
struct WenetDecoderLayerWeight {

    WenetDecoderLayerWeight() = default;
    WenetDecoderLayerWeight(const int layer_id,
                            const int hidden_units,
                            const int inter_size,
                            const int mem_hidden_units);

    ~WenetDecoderLayerWeight();

    WenetDecoderLayerWeight(const WenetDecoderLayerWeight& other);

    WenetDecoderLayerWeight& operator=(const WenetDecoderLayerWeight& other);

    LayerNormWeight<T> pre_layernorm_weights;
    AttentionWeight<T> self_attention_weights;
    LayerNormWeight<T> self_attn_layernorm_weights;
    AttentionWeight<T> cross_attention_weights;
    LayerNormWeight<T> cross_attn_layernorm_weights;
    FfnWeight<T>       ffn_weights;

    void loadModel(std::string dir_path, FtCudaDataType model_file_type);

private:
    void setWeightPtr();
    void mallocWeights();
    void initialize();

    int  layer_id_;
    int  hidden_units_;
    int  inter_size_;
    int  mem_hidden_units_;
    bool is_maintain_buffer = false;

    const static int weights_num_ = 35;
    int              real_weights_num_;
    T*               weights_ptr[weights_num_];
    size_t           weights_size[weights_num_];
};

}  // namespace fastertransformer
