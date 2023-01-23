/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <vector>

#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/TensorParallelGeluFfnLayer.h"
#include "src/fastertransformer/layers/TensorParallelReluFfnLayer.h"
#include "src/fastertransformer/layers/attention_layers/TensorParallelDisentangledAttentionLayer.h"
#include "src/fastertransformer/models/deberta/DebertaWeight.h"
#include "src/fastertransformer/utils/nccl_utils.h"

namespace fastertransformer {

template<typename T>
class Deberta: public BaseLayer {
private:
    // meta data
    size_t                 head_num_;
    size_t                 size_per_head_;
    size_t                 inter_size_;
    size_t                 hidden_units_;
    size_t                 num_layer_;
    size_t                 max_relative_positions_;     // usually 2 * attention_span
    size_t                 relative_position_buckets_;  // i.e., attention_span
    static constexpr float layernorm_eps_ = 1e-7f;
    float                  q_scaling_;
    bool                   sparse_;

    BaseAttentionLayer<T>* disentangled_attention_layer_ = nullptr;
    FfnLayer<T>*           ffn_layer_;

    bool is_allocate_buffer_ = false;

    NcclParam                           tensor_para_;
    NcclParam                           pipeline_para_;
    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm_;
    bool                                enable_custom_all_reduce_;

    void allocateBuffer();
    void freeBuffer();
    void initialize();

    const ActivationType activation_type_;
    const LayerNormType  layernorm_type_;

    void allocateBuffer(size_t batch_size, size_t seq_len);
    bool isValidLayerParallelId(uint l);
    bool isFirstLayerParallelId(uint l);
    bool isLastLayerParallelId(uint l);
    int  getFirstLayerParallelId();

protected:
    // model params
    size_t* h_pinned_token_num_ptr_ = nullptr;
    int*    padding_offset_         = nullptr;
    T*      attention_mask_         = nullptr;
    T*      deberta_emb_buf_        = nullptr;
    T*      deberta_rel_emb_buf_    = nullptr;
    T*      deberta_in_buffer_      = nullptr;
    T*      attn_out_buf_           = nullptr;
    T*      deberta_out_buffer_     = nullptr;

    T* normed_from_tensor_  = nullptr;
    T* normed_attn_out_buf_ = nullptr;

public:
    Deberta(size_t                              max_batch_size,
            size_t                              max_seq_len,
            size_t                              head_num,
            size_t                              size_per_head,
            size_t                              max_relative_positions,
            size_t                              relative_position_buckets,
            size_t                              inter_size,
            size_t                              num_layer,
            float                               q_scaling,
            cudaStream_t                        stream,
            cublasMMWrapper*                    cublas_wrapper,
            IAllocator*                         allocator,
            bool                                is_free_buffer_after_forward,
            bool                                sparse,
            ActivationType                      activation_type,
            LayerNormType                       layernorm_type,
            NcclParam                           tensor_para,
            NcclParam                           pipeline_para,
            std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
            bool                                enable_custom_all_reduce);

    Deberta(size_t           max_batch_size,
            size_t           max_seq_len,
            size_t           head_num,
            size_t           size_per_head,
            size_t           max_relative_positions,
            size_t           relative_position_buckets,
            size_t           inter_size,
            size_t           num_layer,
            float            q_scaling,
            cudaStream_t     stream,
            cublasMMWrapper* cublas_wrapper,
            IAllocator*      allocator,
            bool             is_free_buffer_after_forward,
            bool             sparse,
            ActivationType   activation_type,
            LayerNormType    layernorm_type);

    Deberta(Deberta<T> const& deberta);

    ~Deberta();

    void forward(std::vector<Tensor>*       output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 const DebertaWeight<T>*    deberta_weights);
    void forward(TensorMap* output_tensors, TensorMap* input_tensors, const DebertaWeight<T>* deberta_weights);
};

}  // namespace fastertransformer
