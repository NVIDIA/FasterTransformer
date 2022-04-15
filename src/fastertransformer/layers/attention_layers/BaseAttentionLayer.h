/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include <assert.h>
#include <vector>

#include "3rdparty/trt_fused_multihead_attention/fused_multihead_attention_common.h"
#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/allocator.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

enum class AttentionType {
    UNFUSED_MHA,
    UNFUSED_PADDED_MHA,
    FUSED_MHA,
    FUSED_PADDED_MHA
};

template<typename T>
AttentionType getAttentionType(
    size_t size_per_head, const int sm, const bool remove_padding, const int max_seq_len, const bool is_fuse = true)
{
    if (std::is_same<T, half>::value && (sm == kSM_70 || sm == kSM_86 || sm == kSM_80 || sm == kSM_75 || sm == kSM_72)
        && size_per_head == 64 && max_seq_len <= 384 && is_fuse == true) {
        return remove_padding ? AttentionType::FUSED_MHA : AttentionType::FUSED_PADDED_MHA;
    }
    else {
        return remove_padding ? AttentionType::UNFUSED_MHA : AttentionType::UNFUSED_PADDED_MHA;
    }
}

template<typename T>
AttentionType getAttentionTypeINT8(
    size_t size_per_head, const int sm, const bool remove_padding, const int max_seq_len, const int int8_mode)
{
    if ((int8_mode == 1 || int8_mode == 2) && (sm == kSM_86 || sm == kSM_80 || sm == kSM_75) && size_per_head == 64
        && max_seq_len <= 384) {
        return remove_padding ? AttentionType::FUSED_MHA : AttentionType::FUSED_PADDED_MHA;
    }
    else {
        return remove_padding ? AttentionType::UNFUSED_MHA : AttentionType::UNFUSED_PADDED_MHA;
    }
}

template<typename T>
class BaseAttentionLayer: public BaseLayer {

public:
    virtual void forward(std::vector<fastertransformer::Tensor>* output_tensors,
                         const std::vector<fastertransformer::Tensor>* input_tensors,
                         const AttentionWeight<T>* attention_weights) = 0;
    BaseAttentionLayer(cudaStream_t stream,
                       cublasMMWrapper* cublas_wrapper,
                       IAllocator* allocator,
                       bool is_free_buffer_after_forward,
                       bool sparse = false):
        BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse)
    {
    }
    virtual ~BaseAttentionLayer() = default;
};

}  // namespace fastertransformer
