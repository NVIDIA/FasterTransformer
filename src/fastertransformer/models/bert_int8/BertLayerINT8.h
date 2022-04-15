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

#include "src/fastertransformer/kernels/layernorm_int8_kernels.h"
#include "src/fastertransformer/kernels/layout_transformer_int8_kernels.h"
#include "src/fastertransformer/kernels/quantization_int8_kernels.h"
#include "src/fastertransformer/layers/FfnLayerINT8.h"
#include "src/fastertransformer/layers/attention_layers_int8/FusedAttentionLayerINT8.h"
#include "src/fastertransformer/layers/attention_layers_int8/UnfusedAttentionLayerINT8.h"
#include "src/fastertransformer/models/bert_int8/BertLayerINT8Weight.h"
#include "src/fastertransformer/utils/ScaleList.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/allocator.h"

namespace fastertransformer {

template<typename T>
class BertLayerINT8: public BaseLayer {
private:
    // buffer handling
    size_t max_batch_size_ = 0;
    size_t max_seq_len_ = 0;

    // meta data
    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    int sm_;
    float q_scaling_;
    size_t hidden_units_;
    AttentionType attention_type_;
    int int8_mode_;
    bool sparse_;

    BaseAttentionLayer<T>* attention_layer_;
    FfnLayerINT8<T>* ffn_layer_;

    void allocateBuffer() override;
    void freeBuffer() override;
    bool isValidBatchSize(size_t batch_size);
    bool isValidSeqLen(size_t seq_len);

    void initialize();

protected:
    int32_t* attn_out_buf_;
    int8_t* int8_buf_;
    T* layer_norm_tmp_buf_;
    T* transformer_out_tmp_DataType_;
    T* col32_from_tensor_;

public:
    BertLayerINT8(size_t max_batch_size,
                  size_t max_seq_len,
                  size_t head_num,
                  size_t size_per_head,
                  size_t inter_size,
                  int sm,
                  float q_scaling,
                  int int8_mode,
                  cudaStream_t stream,
                  cublasMMWrapper* cublas_wrapper,
                  IAllocator* allocator,
                  bool is_free_buffer_after_forward,
                  AttentionType attention_type = AttentionType::UNFUSED_PADDED_MHA,
                  bool sparse = false);

    BertLayerINT8(BertLayerINT8<T> const& bert_layer);

    ~BertLayerINT8();

    void forward(std::vector<Tensor>* output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 const BertLayerWeight<T>* bert_layer_weight);
};

}  // namespace fastertransformer
