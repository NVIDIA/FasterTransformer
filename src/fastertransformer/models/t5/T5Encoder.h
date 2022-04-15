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

#include <unordered_map>
#include <vector>

#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/TensorParallelGeluFfnLayer.h"
#include "src/fastertransformer/layers/TensorParallelReluFfnLayer.h"
// #include "src/fastertransformer/layers/attention_layers/FusedAttentionLayer.h"
#include "src/fastertransformer/layers/attention_layers/TensorParallelUnfusedAttentionLayer.h"
#include "src/fastertransformer/models/t5/T5EncoderWeight.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"

namespace fastertransformer {

template<typename T>
class T5Encoder: public BaseLayer {
private:
    // buffer handling
    size_t max_batch_size_ = 0;
    size_t max_seq_len_ = 0;

    // meta data
    const size_t head_num_;
    const size_t size_per_head_;
    const size_t inter_size_;
    const size_t hidden_units_;
    const size_t d_model_;
    const size_t num_layer_;
    const size_t num_bucket_or_max_seq_len_;
    const size_t max_distance_;
    int sm_;
    float q_scaling_;
    AttentionType attention_type_;
    bool sparse_;

    BaseAttentionLayer<T>* attention_layer_;
    FfnLayer<T>* ffn_layer_;

    bool is_allocate_buffer_ = false;

    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size, size_t seq_len);
    void freeBuffer() override;
    void initialize();
    bool isValidLayerParallelId(uint l);
    bool isFirstLayerParallelId(uint l);
    bool isLastLayerParallelId(uint l);
    int getFirstLayerParallelId();

    const ActivationType activation_type_;
    const LayerNormType layernorm_type_;

    const NcclParam tensor_para_;
    const NcclParam pipeline_para_;

    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm_;
    int enable_custom_all_reduce_;

protected:
    // model params
    size_t* token_num_ = nullptr;
    int* padding_offset_ = nullptr;
    int* trt_mha_padding_offset_ = nullptr;
    T* attention_mask_ = nullptr;
    T* relative_attention_bias_ = nullptr;
    T* t5_encoder_emb_buf_ = nullptr;
    T* t5_encoder_in_buffer_ = nullptr;
    T* attn_out_buf_ = nullptr;
    T* t5_encoder_out_buffer_ = nullptr;

    T* normed_from_tensor_ = nullptr;
    T* normed_attn_out_buf_ = nullptr;

public:
    T5Encoder(size_t max_batch_size,
              size_t max_seq_len,
              size_t head_num,
              size_t size_per_head,
              size_t inter_size,
              size_t d_model,
              size_t num_layer,
              size_t num_bucket_or_max_seq_len,
              size_t max_distance,
              int sm,
              float q_scaling,
              cudaStream_t stream,
              cublasMMWrapper* cublas_wrapper,
              IAllocator* allocator,
              bool is_free_buffer_after_forward,
              AttentionType attention_type,
              bool sparse,
              ActivationType activation_type,
              LayerNormType layernorm_type,
              NcclParam tensor_para,
              NcclParam pipeline_para,
              std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm = nullptr,
              int enable_custom_all_reduce = 0);

    T5Encoder(T5Encoder<T> const& t5_layer);

    ~T5Encoder();

    void forward(std::vector<Tensor>* output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 const T5EncoderWeight<T>* t5_weights);

    void forward(std::unordered_map<std::string, Tensor>* output_tensors,
                 const std::unordered_map<std::string, Tensor>* input_tensors,
                 const T5EncoderWeight<T>* t5_weights);

    inline size_t getDModel()
    {
        return d_model_;
    }
    void setStream(cudaStream_t stream) override;
};

}  // namespace fastertransformer
