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

#include "src/fastertransformer/kernels/xlnet_preprocess_kernels.h"
#include "src/fastertransformer/layers/xlnet_attention_layers/XlnetAttentionLayer.h"
#include "src/fastertransformer/models/xlnet/XlnetLayerWeight.h"

#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/FfnLayer.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/allocator.h"

namespace fastertransformer {

template<typename T>
class Xlnet: public BaseLayer {
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
    float q_scaling_;

    bool is_allocate_buffer_ = false;
    FfnLayer<T>* ffn_layer_;

    void allocateBuffer();
    void freeBuffer();
    bool isValidBatchSize(size_t batch_size);
    bool isValidSeqLen(size_t seq_len);
    void initialize();

protected:
    // Preprocess data
    T* word_emb_k_;
    T* attn_mask_;
    T* seg_mat_;
    T* attr_k_head_r_;

    // Postprocess data
    T* attn_out_buf_;
    T* output_fc2_;

    XlnetAttentionLayer<T>* attention_layer_;

public:
    Xlnet(size_t max_batch_size,
          size_t max_seq_len,
          size_t head_num,
          size_t size_per_head,
          size_t inter_size,
          size_t num_layer,
          float q_scaling,
          cudaStream_t stream,
          cublasMMWrapper* cublas_wrapper,
          IAllocator* allocator,
          bool is_free_buffer_after_forward);

    Xlnet(Xlnet<T> const& xlnet_layer);

    ~Xlnet();

    void forward(std::vector<Tensor>* output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 const std::vector<XlnetLayerWeight<T>>* bert_layer_weights);
};

}  // namespace fastertransformer
