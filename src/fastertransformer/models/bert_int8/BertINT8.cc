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

#include "BertINT8.h"

namespace fastertransformer {

template<typename T>
BertINT8<T>::BertINT8(size_t max_batch_size,
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
                      bool sparse):
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    hidden_units_(head_num_ * size_per_head),
    num_layer_(num_layer),
    sm_(sm),
    q_scaling_(q_scaling),
    int8_mode_(int8_mode),
    stream_(stream),
    cublas_wrapper_(cublas_wrapper),
    allocator_(allocator),
    is_free_buffer_after_forward_(is_free_buffer_after_forward),
    attention_type_(attention_type),
    sparse_(sparse)
{
    if (int8_mode_ != 1 && int8_mode_ != 2 && int8_mode_ != 3) {
        throw std::runtime_error(std::string("[FT][ERROR] int8_mode_ not support \n"));
    }
    if (sparse_ && int8_mode_ == 1) {
        throw std::runtime_error(std::string("[FT][ERROR] int8_mode 1 does not support sparsity \n"));
    }
    if (int8_mode_ == 1 && max_seq_len_ > 384 && (max_seq_len_ % 32 != 0)) {
        throw std::runtime_error(std::string(
            "[FT][ERROR] max_seq_len_ should be a multiple of 32 when int8_mode == 1 && max_seq_len_ > 384\n"));
    }
    bert_layer_ = new BertLayerINT8<T>(max_batch_size_,
                                       max_seq_len_,
                                       head_num_,
                                       size_per_head_,
                                       head_num_ * size_per_head_ * 4,
                                       sm_,
                                       q_scaling_,
                                       int8_mode_,
                                       stream_,
                                       cublas_wrapper_,
                                       allocator_,
                                       is_free_buffer_after_forward_,
                                       attention_type_,
                                       sparse_);
}

template<typename T>
BertINT8<T>::BertINT8(BertINT8<T> const& bert):
    max_batch_size_(bert.max_batch_size_),
    max_seq_len_(bert.max_seq_len_),
    head_num_(bert.head_num_),
    size_per_head_(bert.size_per_head_),
    inter_size_(bert.inter_size_),
    hidden_units_(bert.hidden_units_),
    num_layer_(bert.num_layer_),
    sm_(bert.sm_),
    q_scaling_(bert.q_scaling_),
    int8_mode_(bert.int8_mode_),
    stream_(bert.stream_),
    cublas_wrapper_(bert.cublas_wrapper_),
    allocator_(bert.allocator_),
    is_free_buffer_after_forward_(bert.is_free_buffer_after_forward_),
    attention_type_(bert.attention_type_),
    sparse_(bert.sparse_)
{

    bert_layer_ = new BertLayerINT8<T>(max_batch_size_,
                                       max_seq_len_,
                                       head_num_,
                                       size_per_head_,
                                       head_num_ * size_per_head_ * 4,
                                       sm_,
                                       q_scaling_,
                                       int8_mode_,
                                       stream_,
                                       cublas_wrapper_,
                                       allocator_,
                                       is_free_buffer_after_forward_,
                                       attention_type_,
                                       sparse_);
}

template<typename T>
BertINT8<T>::~BertINT8()
{
    freeBuffer();
    delete bert_layer_;
}

template<typename T>
void BertINT8<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        token_num_ = (size_t*)allocator_->malloc(sizeof(size_t) * 1, false);
        padding_offset_ = (int*)allocator_->malloc(sizeof(int) * max_batch_size_ * max_seq_len_, false);
        trt_mha_padding_offset_ = (int*)allocator_->malloc(sizeof(int) * (2 * max_batch_size_ + 1), false);

        attention_mask_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * max_seq_len_, false);

        bert_in_buffer_ =
            (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * head_num_ * size_per_head_, false);
        bert_out_buffer_ =
            (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * head_num_ * size_per_head_, false);
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void BertINT8<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free(token_num_);
        allocator_->free(padding_offset_);
        allocator_->free(trt_mha_padding_offset_);

        allocator_->free(attention_mask_);
        allocator_->free(bert_in_buffer_);
        allocator_->free(bert_out_buffer_);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
void BertINT8<T>::forward(std::vector<Tensor>* output_tensors,
                          const std::vector<Tensor>* input_tensors,
                          const std::vector<BertLayerINT8Weight<T>>* bert_layer_weights)
{
    // input_tensors: [input_query (batch, seqlen, hidden), sequence_length (batch)]
    // output_tensors: [output (batch, seqlen, size_per_head*head_num)]
    const size_t request_batch_size = input_tensors->at(0).shape[0];
    const size_t request_seq_len = input_tensors->at(0).shape[1];
    FT_CHECK(input_tensors->size() == 2);
    FT_CHECK(isValidBatchSize(request_batch_size));
    FT_CHECK(isValidSeqLen(request_seq_len));
    FT_CHECK(request_batch_size == input_tensors->at(1).shape[0]);
    FT_CHECK(input_tensors->at(0).shape.size() == 3);
    FT_CHECK(input_tensors->at(1).shape.size() == 1);
    allocateBuffer();

    const int* sequence_lengths = reinterpret_cast<const int*>(input_tensors->at(1).data);

    size_t h_token_num;
    T* bert_input_ptr;
    T* bert_output_ptr;

    Tensor* padding_offset_tensor_ptr;
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
    std::vector<Tensor> tmp_output_tensors = {
        Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, hidden_units_}, bert_output_ptr},
    };

    int num_layer_ptr[1] = {int(num_layer_)};
    int layer_idx_ptr[1] = {-1};
    for (uint i = 0; i < num_layer_; i++) {
        layer_idx_ptr[0] = i;
        std::vector<Tensor> tmp_input_tensors{
            Tensor{MEMORY_GPU,
                   data_type,
                   std::vector<size_t>{h_token_num, hidden_units_},
                   i == 0 ? bert_input_ptr : bert_output_ptr},
            Tensor{MEMORY_GPU,
                   data_type,
                   std::vector<size_t>{request_batch_size, 1, request_seq_len, request_seq_len},
                   attention_mask_},
            *padding_offset_tensor_ptr,
            Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{1}, layer_idx_ptr},
            Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{1}, num_layer_ptr},
        };
        bert_layer_->forward(&tmp_output_tensors, &tmp_input_tensors, &bert_layer_weights->at(i));
    }

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
bool BertINT8<T>::isValidBatchSize(size_t batch_size)
{
    if (max_batch_size_ == 0) {
        max_batch_size_ = batch_size;
        return true;
    }
    else {
        return batch_size <= max_batch_size_;
    }
}

template<typename T>
bool BertINT8<T>::isValidSeqLen(size_t seq_len)
{
    if (max_seq_len_ == 0) {
        max_seq_len_ = seq_len;
        return true;
    }
    else {
        return seq_len <= max_seq_len_;
    }
}

template class BertINT8<float>;
template class BertINT8<half>;

}  // namespace fastertransformer
