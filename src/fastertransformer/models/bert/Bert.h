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

#include <vector>

#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/FfnLayer.h"
#include "src/fastertransformer/layers/attention_layers/FusedAttentionLayer.h"
#include "src/fastertransformer/layers/attention_layers/UnfusedAttentionLayer.h"
#include "src/fastertransformer/models/bert/BertWeight.h"

namespace fastertransformer {

template<typename T>
class Bert: public BaseLayer {
private:
    // buffer handling
    size_t max_batch_size_ = 0;
    size_t max_seq_len_ = 0;

    // meta data
    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t hidden_units_;
    size_t num_layer_;
    int sm_;
    float q_scaling_;
    AttentionType attention_type_;
    bool sparse_;

    BaseAttentionLayer<T>* attention_layer_;
    FfnLayer<T>* ffn_layer_;

    bool is_allocate_buffer_ = false;

    void allocateBuffer();
    void freeBuffer();
    bool isValidBatchSize(size_t batch_size);
    bool isValidSeqLen(size_t seq_len);
    void initialize();

    const ActivationType activation_type_;
    const LayerNormType layernorm_type_;

    void allocateBuffer(size_t batch_size, size_t seq_len);

protected:
    // model params
    size_t* token_num_ = nullptr;
    int* padding_offset_ = nullptr;
    int* trt_mha_padding_offset_ = nullptr;
    T* attention_mask_ = nullptr;
    T* bert_in_buffer_ = nullptr;
    T* attn_out_buf_ = nullptr;
    T* bert_out_buffer_ = nullptr;

    T* normed_from_tensor_ = nullptr;
    T* normed_attn_out_buf_ = nullptr;

public:
    Bert(size_t max_batch_size,
         size_t max_seq_len,
         size_t head_num,
         size_t size_per_head,
         size_t inter_size,
         size_t num_layer,
         int sm,
         float q_scaling,
         cudaStream_t stream,
         cublasMMWrapper* cublas_wrapper,
         IAllocator* allocator,
         bool is_free_buffer_after_forward,
         AttentionType attention_type,
         bool sparse,
         ActivationType activation_type,
         LayerNormType layernorm_type);

    Bert(Bert<T> const& bert_layer);

    ~Bert();

    void forward(std::vector<Tensor>* output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 const BertWeight<T>* bert_weights);
};

}  // namespace fastertransformer
