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

#include <memory>

#include "3rdparty/trt_fused_multihead_attention/qkvToContext.h"
#include "src/fastertransformer/layers/attention_layers/BaseAttentionLayer.h"
#include "src/fastertransformer/layers/attention_layers_int8/AttentionINT8Weight.h"
#include "src/fastertransformer/utils/ScaleList.h"
#include "src/fastertransformer/utils/cublasINT8MMWrapper.h"

namespace fastertransformer {

// This class is only used when we satisfy the following conditions:
// 1. INT8
// 2. Temporally add seqlen <= 384 limitation because the
template<typename T>
class FusedAttentionLayerINT8: public BaseAttentionLayer<T> {
private:
    // metadata
    size_t head_num_;
    size_t size_per_head_;

    // calculated params
    size_t hidden_units_;

    // buffer handling
    size_t max_batch_size_ = 0;
    size_t max_seq_len_ = 0;

    void allocateBuffer() override;
    void freeBuffer() override;
    bool isValidBatchSize(size_t batch_size);
    bool isValidSeqLen(size_t seq_len);

    float q_scaling_;
    int sm_;
    int int8_mode_;
    std::unique_ptr<MHARunner> dispatcher_int8_;
    bool sparse_;

    using BaseAttentionLayer<T>::stream_;
    using BaseAttentionLayer<T>::is_free_buffer_after_forward_;
    using BaseAttentionLayer<T>::is_allocate_buffer_;
    using BaseAttentionLayer<T>::cublas_wrapper_;
    using BaseAttentionLayer<T>::allocator_;

protected:
    int32_t* Q_int_buf_;
    int32_t* K_int_buf_;
    int32_t* V_int_buf_;
    int8_t* qkv_buf_;
    int8_t* qkv_buf_2_;
    T* attn_workspace_;

public:
    FusedAttentionLayerINT8(size_t max_batch_size,
                            size_t max_seq_len,
                            size_t head_num,
                            size_t size_per_head,
                            int sm,
                            float q_scaling,
                            int int8_mode,
                            cudaStream_t stream,
                            cublasMMWrapper* cublas_wrapper,
                            IAllocator* allocator,
                            bool is_free_buffer_after_forward,
                            bool sparse = false);

    FusedAttentionLayerINT8(FusedAttentionLayerINT8<T> const& attention_layer);

    ~FusedAttentionLayerINT8();

    void forward(std::vector<fastertransformer::Tensor>* output_tensors,
                 const std::vector<fastertransformer::Tensor>* input_tensors,
                 const AttentionWeight<T>* attention_weights) override;

    void invokeTrtAddQkvBias(size_t token_num, const AttentionWeight<T>* attention_weights);
};

}  // namespace fastertransformer
