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

#include "BertLayerINT8.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"

#include <vector>

namespace fastertransformer {

template<typename T>
class BertINT8 {
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
    int int8_mode_;
    cudaStream_t stream_;
    cublasMMWrapper* cublas_wrapper_;
    IAllocator* allocator_;
    bool is_free_buffer_after_forward_;
    AttentionType attention_type_;
    bool sparse_;

    bool is_allocate_buffer_ = false;

    BertLayerINT8<T>* bert_layer_ = nullptr;

    void allocateBuffer();
    void freeBuffer();
    bool isValidBatchSize(size_t batch_size);
    bool isValidSeqLen(size_t seq_len);

protected:
    size_t* token_num_;
    int* padding_offset_;
    int* trt_mha_padding_offset_;
    T* attention_mask_;
    T* bert_in_buffer_;
    T* bert_out_buffer_;

public:
    BertINT8(size_t max_batch_size,
             size_t max_seq_len,
             size_t head_num,
             size_t size_per_head,
             size_t inter_size,
             size_t num_layer,
             int sm,
             float q_scaling,
             int int8_mode,
             cudaStream_t stream,
             cublasMMWrapper* cublas_wrapper,
             IAllocator* allocator,
             bool is_free_buffer_after_forward,
             AttentionType attention_type,
             bool sparse = false);

    BertINT8(BertINT8<T> const& bert_layer);

    ~BertINT8();

    void forward(std::vector<Tensor>* output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 const std::vector<BertLayerINT8Weight<T>>* bert_layer_weights);
    // friend class BertDebug<T>;
};

}  // namespace fastertransformer
