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

#include "src/fastertransformer/models/bert/Bert.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"

namespace fastertransformer {

template<typename T>
void Bert<T>::initialize()
{
    if ((attention_type_ == AttentionType::FUSED_MHA || attention_type_ == AttentionType::FUSED_PADDED_MHA)
        && std::is_same<T, half>::value == true && max_seq_len_ <= 384) {
        attention_layer_ = new FusedAttentionLayer<T>(max_batch_size_,
                                                      max_seq_len_,
                                                      head_num_,
                                                      size_per_head_,
                                                      sm_,
                                                      q_scaling_,
                                                      stream_,
                                                      cublas_wrapper_,
                                                      allocator_,
                                                      is_free_buffer_after_forward_,
                                                      sparse_);
    }
    else if (attention_type_ == AttentionType::UNFUSED_MHA || attention_type_ == AttentionType::UNFUSED_PADDED_MHA) {
        attention_layer_ = new UnfusedAttentionLayer<T>(max_batch_size_,
                                                        max_seq_len_,
                                                        head_num_,
                                                        size_per_head_,
                                                        q_scaling_,
                                                        stream_,
                                                        cublas_wrapper_,
                                                        allocator_,
                                                        is_free_buffer_after_forward_,
                                                        sparse_);
    }
    else {
        throw std::runtime_error(std::string("[FT][ERROR] Invalid attention type \n"));
    }

    if (activation_type_ == ActivationType::Gelu) {
        ffn_layer_ = new GeluFfnLayer<T>(max_batch_size_,
                                         max_seq_len_,
                                         head_num_,
                                         size_per_head_,
                                         inter_size_,
                                         stream_,
                                         cublas_wrapper_,
                                         allocator_,
                                         is_free_buffer_after_forward_,
                                         sparse_);
    }
    else if (activation_type_ == ActivationType::Relu) {
        ffn_layer_ = new ReluFfnLayer<T>(max_batch_size_,
                                         max_seq_len_,
                                         head_num_,
                                         size_per_head_,
                                         inter_size_,
                                         stream_,
                                         cublas_wrapper_,
                                         allocator_,
                                         is_free_buffer_after_forward_,
                                         sparse_);
    }
}

template<typename T>
Bert<T>::Bert(size_t max_batch_size,
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
              LayerNormType layernorm_type):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    hidden_units_(head_num_ * size_per_head_),
    num_layer_(num_layer),
    sm_(sm),
    q_scaling_(q_scaling),
    attention_type_(attention_type),
    sparse_(sparse),
    activation_type_(activation_type),
    layernorm_type_(layernorm_type)
{
    initialize();
}

template<typename T>
Bert<T>::Bert(Bert<T> const& bert):
    BaseLayer(bert),
    max_batch_size_(bert.max_batch_size_),
    max_seq_len_(bert.max_seq_len_),
    head_num_(bert.head_num_),
    size_per_head_(bert.size_per_head_),
    inter_size_(bert.inter_size_),
    hidden_units_(bert.hidden_units_),
    num_layer_(bert.num_layer_),
    sm_(bert.sm_),
    q_scaling_(bert.q_scaling_),
    attention_type_(bert.attention_type_),
    sparse_(bert.sparse_),
    activation_type_(bert.activation_type_),
    layernorm_type_(bert.layernorm_type_)
{
    initialize();
}

template<typename T>
Bert<T>::~Bert()
{
    delete attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void Bert<T>::allocateBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_ == false) {
        token_num_ = (size_t*)allocator_->malloc(sizeof(size_t) * 1, false);
        padding_offset_ = (int*)allocator_->malloc(sizeof(int) * max_batch_size_ * max_seq_len_, false);
        trt_mha_padding_offset_ = (int*)allocator_->malloc(sizeof(int) * (2 * max_batch_size_ + 1), false);

        attention_mask_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * max_seq_len_, false);

        bert_in_buffer_ =
            (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * head_num_ * size_per_head_, false);
        attn_out_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        bert_out_buffer_ =
            (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * head_num_ * size_per_head_, false);

        if (layernorm_type_ == LayerNormType::post_layernorm) {
            normed_from_tensor_ = nullptr;
            normed_attn_out_buf_ = nullptr;
        }
        else {
            normed_from_tensor_ =
                (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
            normed_attn_out_buf_ =
                (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        }
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void Bert<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    token_num_ = (size_t*)allocator_->reMalloc(token_num_, sizeof(size_t) * 1, false);
    padding_offset_ = (int*)allocator_->reMalloc(padding_offset_, sizeof(int) * batch_size * seq_len, false);
    trt_mha_padding_offset_ =
        (int*)allocator_->reMalloc(trt_mha_padding_offset_, sizeof(int) * (2 * batch_size + 1), false);

    attention_mask_ = (T*)allocator_->reMalloc(attention_mask_, sizeof(T) * batch_size * seq_len * seq_len, false);

    bert_in_buffer_ =
        (T*)allocator_->reMalloc(bert_in_buffer_, sizeof(T) * batch_size * seq_len * head_num_ * size_per_head_, false);
    attn_out_buf_ = (T*)allocator_->reMalloc(attn_out_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    bert_out_buffer_ = (T*)allocator_->reMalloc(
        bert_out_buffer_, sizeof(T) * batch_size * seq_len * head_num_ * size_per_head_, false);

    if (layernorm_type_ == LayerNormType::post_layernorm) {
        normed_from_tensor_ = nullptr;
        normed_attn_out_buf_ = nullptr;
    }
    else {
        normed_from_tensor_ =
            (T*)allocator_->reMalloc(normed_from_tensor_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
        normed_attn_out_buf_ =
            (T*)allocator_->reMalloc(normed_attn_out_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    }
}

template<typename T>
void Bert<T>::freeBuffer()
{
    allocator_->free(token_num_);
    allocator_->free(padding_offset_);
    allocator_->free(trt_mha_padding_offset_);

    allocator_->free(attention_mask_);
    allocator_->free(bert_in_buffer_);
    allocator_->free(attn_out_buf_);
    allocator_->free(bert_out_buffer_);

    if (layernorm_type_ == LayerNormType::post_layernorm) {
        normed_from_tensor_ = nullptr;
        normed_attn_out_buf_ = nullptr;
    }
    else {
        allocator_->free(normed_from_tensor_);
        allocator_->free(normed_attn_out_buf_);
    }
}

template<typename T>
void Bert<T>::forward(std::vector<Tensor>* output_tensors,
                      const std::vector<Tensor>* input_tensors,
                      const BertWeight<T>* bert_weights)
{
    // input_tensors:
    //      input_query [batch, seqlen, hidden]
    //      sequence_length [batch]
    // output tensors:
    //      output hidden state [batch, seqlen, hidden]

    const size_t request_batch_size = input_tensors->at(0).shape[0];
    const size_t request_seq_len = input_tensors->at(0).shape[1];
    FT_CHECK(input_tensors->size() == 2);
    FT_CHECK(isValidBatchSize(request_batch_size));
    FT_CHECK(isValidSeqLen(request_seq_len));
    FT_CHECK(request_batch_size == input_tensors->at(1).shape[0]);
    FT_CHECK(input_tensors->at(0).shape.size() == 3);
    FT_CHECK(input_tensors->at(1).shape.size() == 1);
    allocateBuffer(request_batch_size, request_seq_len);

    const int* sequence_lengths = reinterpret_cast<const int*>(input_tensors->at(1).data);

    size_t h_token_num;
    T* bert_input_ptr;
    T* bert_output_ptr;
    Tensor* padding_offset_tensor_ptr;

    // preprocess (remove padding and build mask)
    switch (attention_type_) {
        case AttentionType::UNFUSED_MHA: {
            invokeBuildEncoderAttentionMask(
                attention_mask_, sequence_lengths, request_batch_size, request_seq_len, stream_);
            sync_check_cuda_error();
            invokeGetPaddingOffset(&h_token_num,
                                   token_num_,
                                   padding_offset_,
                                   sequence_lengths,
                                   request_batch_size,
                                   request_seq_len,
                                   stream_);

            invokeRemovePadding(bert_in_buffer_,
                                (const T*)input_tensors->at(0).data,
                                padding_offset_,
                                h_token_num,
                                head_num_ * size_per_head_,
                                stream_);
            sync_check_cuda_error();
            bert_input_ptr = bert_in_buffer_;
            bert_output_ptr = bert_out_buffer_;

            padding_offset_tensor_ptr =
                new Tensor(MEMORY_GPU, TYPE_INT32, std::vector<size_t>{h_token_num}, padding_offset_);
            break;
        }
        case AttentionType::UNFUSED_PADDED_MHA: {
            invokeBuildEncoderAttentionMask(
                attention_mask_, sequence_lengths, request_batch_size, request_seq_len, stream_);
            sync_check_cuda_error();
            h_token_num = request_batch_size * request_seq_len;
            bert_input_ptr = (T*)input_tensors->at(0).data;
            bert_output_ptr = (T*)output_tensors->at(0).data;
            padding_offset_tensor_ptr = new Tensor(MEMORY_GPU, TYPE_INT32, std::vector<size_t>{0}, nullptr);
            break;
        }
        case AttentionType::FUSED_MHA: {
            invokeGetPaddingOffset(&h_token_num,
                                   token_num_,
                                   padding_offset_,
                                   sequence_lengths,
                                   request_batch_size,
                                   request_seq_len,
                                   stream_);

            invokeRemovePadding(bert_in_buffer_,
                                (const T*)input_tensors->at(0).data,
                                padding_offset_,
                                h_token_num,
                                head_num_ * size_per_head_,
                                stream_);
            sync_check_cuda_error();
            bert_input_ptr = bert_in_buffer_;
            bert_output_ptr = bert_out_buffer_;

            invokeGetTrtPaddingOffset(trt_mha_padding_offset_, sequence_lengths, request_batch_size, stream_);

            padding_offset_tensor_ptr = new Tensor(
                MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size + 1}, trt_mha_padding_offset_);
            break;
        }
        case AttentionType::FUSED_PADDED_MHA: {
            h_token_num = request_batch_size * request_seq_len;
            invokeGetTrtPaddingOffset(
                trt_mha_padding_offset_, sequence_lengths, request_batch_size, request_seq_len, stream_);
            padding_offset_tensor_ptr = new Tensor(
                MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size * 2 + 1}, trt_mha_padding_offset_);
            bert_input_ptr = (T*)input_tensors->at(0).data;
            bert_output_ptr = (T*)output_tensors->at(0).data;
            break;
        }
        default: {
            throw std::runtime_error(std::string("[FT][ERROR] Invalid attention type \n"));
        }
    }

    DataType data_type = getTensorType<T>();
    for (uint i = 0; i < num_layer_; i++) {
        const T* from_tensor = (const T*)(i == 0 ? bert_input_ptr : bert_output_ptr);
        T* out_tensor = bert_output_ptr;

        if (layernorm_type_ == LayerNormType::pre_layernorm) {
            invokeGeneralLayerNorm(normed_from_tensor_,
                                   from_tensor,
                                   bert_weights->bert_layer_weights[i].attn_layernorm_weights.gamma,
                                   bert_weights->bert_layer_weights[i].attn_layernorm_weights.beta,
                                   h_token_num,
                                   hidden_units_,
                                   stream_);
        }

        // Attention
        {
            std::vector<Tensor> attn_input_tensors{
                Tensor{MEMORY_GPU,
                       data_type,
                       std::vector<size_t>{h_token_num, hidden_units_},
                       layernorm_type_ == LayerNormType::pre_layernorm ? normed_from_tensor_ : from_tensor},
                Tensor{MEMORY_GPU,
                       data_type,
                       std::vector<size_t>{request_batch_size, 1, request_seq_len, request_seq_len},
                       attention_mask_},
                *padding_offset_tensor_ptr};
            std::vector<Tensor> attn_output_tensors{
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, hidden_units_}, attn_out_buf_}};

            attention_layer_->forward(
                &attn_output_tensors, &attn_input_tensors, &bert_weights->bert_layer_weights[i].attention_weights);
        }

        if (layernorm_type_ == LayerNormType::post_layernorm) {
            invokeAddBiasResidualLayerNorm(
                attn_out_buf_,
                from_tensor,
                bert_weights->bert_layer_weights[i].attention_weights.attention_output_weight.bias,
                bert_weights->bert_layer_weights[i].attn_layernorm_weights.gamma,
                bert_weights->bert_layer_weights[i].attn_layernorm_weights.beta,
                h_token_num,
                hidden_units_,
                stream_);
        }
        else if (layernorm_type_ == LayerNormType::pre_layernorm) {
            invokeGeneralAddBiasResidualPreLayerNorm(
                attn_out_buf_,
                normed_attn_out_buf_,
                from_tensor,
                bert_weights->bert_layer_weights[i].ffn_layernorm_weights.gamma,
                bert_weights->bert_layer_weights[i].ffn_layernorm_weights.beta,
                bert_weights->bert_layer_weights[i].attention_weights.attention_output_weight.bias,
                h_token_num,
                hidden_units_,
                stream_);
        }

        // FFN
        {
            std::vector<Tensor> ffn_input_tensors{
                Tensor{MEMORY_GPU,
                       data_type,
                       std::vector<size_t>{h_token_num, hidden_units_},
                       layernorm_type_ == LayerNormType::pre_layernorm ? normed_attn_out_buf_ : attn_out_buf_}};
            std::vector<Tensor> ffn_output_tensors{
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, hidden_units_}, out_tensor}};
            ffn_layer_->forward(
                &ffn_output_tensors, &ffn_input_tensors, &bert_weights->bert_layer_weights[i].ffn_weights);
        }

        if (layernorm_type_ == LayerNormType::post_layernorm) {
            invokeAddBiasResidualLayerNorm(out_tensor,
                                           attn_out_buf_,
                                           bert_weights->bert_layer_weights[i].ffn_weights.output_weight.bias,
                                           bert_weights->bert_layer_weights[i].ffn_layernorm_weights.gamma,
                                           bert_weights->bert_layer_weights[i].ffn_layernorm_weights.beta,
                                           h_token_num,
                                           hidden_units_,
                                           stream_);
        }
        else if (layernorm_type_ == LayerNormType::pre_layernorm) {
            invokeAddBiasResidual(out_tensor,
                                  attn_out_buf_,
                                  bert_weights->bert_layer_weights[i].ffn_weights.output_weight.bias,
                                  h_token_num,
                                  hidden_units_,
                                  stream_);
        }
        sync_check_cuda_error();
    }

    if (layernorm_type_ == LayerNormType::pre_layernorm) {
        invokeGeneralLayerNorm(bert_output_ptr,
                               bert_output_ptr,
                               bert_weights->post_transformer_layernorm_weights.gamma,
                               bert_weights->post_transformer_layernorm_weights.beta,
                               h_token_num,
                               hidden_units_,
                               stream_);
    }

    // post process (rebuild padding)
    switch (attention_type_) {
        case AttentionType::UNFUSED_MHA: {
            invokeRebuildPadding((T*)output_tensors->at(0).data,
                                 bert_out_buffer_,
                                 padding_offset_,
                                 h_token_num,
                                 head_num_ * size_per_head_,
                                 stream_);
            break;
        }
        case AttentionType::UNFUSED_PADDED_MHA: {
            break;
        }
        case AttentionType::FUSED_MHA: {
            invokeRebuildPadding((T*)output_tensors->at(0).data,
                                 bert_out_buffer_,
                                 padding_offset_,
                                 h_token_num,
                                 head_num_ * size_per_head_,
                                 stream_);
            break;
        }
        case AttentionType::FUSED_PADDED_MHA: {
            break;
        }
        default: {
            throw std::runtime_error(std::string("[FT][ERROR] Invalid attention type \n"));
        }
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();

    delete padding_offset_tensor_ptr;
}

template<typename T>
bool Bert<T>::isValidBatchSize(size_t batch_size)
{
    if (max_batch_size_ < batch_size) {
        max_batch_size_ = batch_size;
    }
    return true;
}

template<typename T>
bool Bert<T>::isValidSeqLen(size_t seq_len)
{
    if (max_seq_len_ < seq_len) {
        max_seq_len_ = seq_len;
    }
    return true;
}

template class Bert<float>;
template class Bert<half>;

}  // namespace fastertransformer
