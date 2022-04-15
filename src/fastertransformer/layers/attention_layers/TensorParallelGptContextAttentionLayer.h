/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/layers/attention_layers/GptContextAttentionLayer.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"
#include "src/fastertransformer/utils/nccl_utils.h"

namespace fastertransformer {

template<typename T>
class TensorParallelGptContextAttentionLayer: public GptContextAttentionLayer<T> {
private:
    NcclParam tensor_para_;
    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm_;
    int enable_custom_all_reduce_;

public:
    TensorParallelGptContextAttentionLayer(size_t max_batch_size,
                                           size_t max_seq_len,
                                           size_t head_num,
                                           size_t size_per_head,
                                           NcclParam tensor_para,
                                           cudaStream_t stream,
                                           cublasMMWrapper* cublas_wrapper,
                                           IAllocator* allocator,
                                           bool is_free_buffer_after_forward,
                                           bool is_qk_buf_float,
                                           bool sparse = false,
                                           std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm = nullptr,
                                           int enable_custom_all_reduce = 0);

    TensorParallelGptContextAttentionLayer(size_t max_batch_size,
                                           size_t max_seq_len,
                                           size_t head_num,
                                           size_t size_per_head,
                                           size_t rotary_embedding_dim,
                                           NcclParam tensor_para,
                                           cudaStream_t stream,
                                           cublasMMWrapper* cublas_wrapper,
                                           IAllocator* allocator,
                                           bool is_free_buffer_after_forward,
                                           bool is_qk_buf_float,
                                           bool sparse = false,
                                           std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm = nullptr,
                                           int enable_custom_all_reduce = 0);

    TensorParallelGptContextAttentionLayer(TensorParallelGptContextAttentionLayer<T> const& attention_layer);

    ~TensorParallelGptContextAttentionLayer() = default;

    void forward(std::vector<fastertransformer::Tensor>* output_tensors,
                 const std::vector<fastertransformer::Tensor>* input_tensors,
                 const AttentionWeight<T>* attention_weights) override;
};

}  // namespace fastertransformer
