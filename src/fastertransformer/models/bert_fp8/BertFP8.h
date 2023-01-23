/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/TensorParallelGeluFfnFP8Layer.h"
#include "src/fastertransformer/layers/attention_layers_fp8/SelfAttentionFP8Layer.h"
#include "src/fastertransformer/models/bert_fp8/BertFP8Weight.h"
#include "src/fastertransformer/utils/cublasFP8MMWrapper.h"
#include "src/fastertransformer/utils/nccl_utils.h"

namespace fastertransformer {

template<typename T1, typename T2>
class BertFP8: public BaseLayer {
private:
    // meta data
    size_t    head_num_;
    size_t    size_per_head_;
    size_t    inter_size_;
    size_t    d_model_;
    size_t    num_layer_;
    NcclParam tensor_para_;
    NcclParam pipeline_para_;
    // mode 0: no fp8. Should use original bert directly
    // mode 1: per tensor scale for activation, per channel scale for weight
    // mode 2: per tensor scale for activation and weight
    int fp8_mode_;

    int           sm_;
    float         q_scaling_;
    AttentionType attention_type_;

    BaseAttentionFP8Layer<T1, T2>* attention_layer_;
    FfnFP8Layer<T1, T2>*           ffn_layer_;

    bool is_allocate_buffer_ = false;

    void allocateBuffer();
    void freeBuffer();
    void initialize();

    const ActivationType activation_type_;
    const LayerNormType  layernorm_type_;

    void allocateBuffer(size_t batch_size, size_t seq_len);

protected:
    // model params
    size_t* h_pinned_token_num_ptr_ = nullptr;
    int*    padding_offset_         = nullptr;
    int*    trt_mha_padding_offset_ = nullptr;
    T1*     attention_mask_         = nullptr;
    T1*     bert_in_buffer_         = nullptr;
    T1*     attn_out_buf_           = nullptr;
    T2*     bf16_out_tensor_        = nullptr;
    T1*     bert_out_buffer_        = nullptr;

    T1* normed_from_tensor_  = nullptr;
    T1* normed_attn_out_buf_ = nullptr;

    T2* first_token_tensor_ = nullptr;

public:
    BertFP8(size_t           head_num,
            size_t           size_per_head,
            size_t           d_model,
            size_t           inter_size,
            size_t           num_layer,
            NcclParam        tensor_para,
            NcclParam        pipeline_para,
            int              sm,
            float            q_scaling,
            cudaStream_t     stream,
            cublasMMWrapper* cublas_wrapper,
            IAllocator*      allocator,
            bool             is_free_buffer_after_forward,
            AttentionType    attention_type,
            bool             sparse,
            ActivationType   activation_type,
            LayerNormType    layernorm_type,
            int              fp8_mode);

    BertFP8(BertFP8<T1, T2> const& bert_layer);

    ~BertFP8();

    void forward(TensorMap* output_tensors, TensorMap* input_tensors, const BertFP8Weight<T1, T2>* bert_weights);
};

}  // namespace fastertransformer
