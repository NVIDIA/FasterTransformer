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

#include "src/fastertransformer/layers/xlnet_attention_layers/XlnetAttentionLayer.h"

namespace fastertransformer {

template<typename T>
void XlnetAttentionLayer<T>::forward(std::vector<fastertransformer::Tensor>* output_tensors,
                                     const std::vector<fastertransformer::Tensor>* input_tensors,
                                     const XlnetAttentionWeight<T>* attention_weights)
{
    const size_t request_batch_size = input_tensors->at(0).shape[0];
    const size_t request_seq_len = input_tensors->at(0).shape[1];

    FT_CHECK(isValidBatchSize(input_tensors->at(1).shape[0]));
    FT_CHECK(isValidSeqLen(input_tensors->at(1).shape[2]));

    FT_CHECK(input_tensors->size() == 4);
    FT_CHECK(input_tensors->at(0).shape.size() == 3);
    FT_CHECK(input_tensors->at(1).shape.size() == 3);
    FT_CHECK(input_tensors->at(2).shape.size() == 3);
    FT_CHECK(input_tensors->at(3).shape.size() == 2);

    FT_CHECK(input_tensors->at(1).shape[0] == request_batch_size);
    FT_CHECK(input_tensors->at(2).shape[0] == request_batch_size);
    FT_CHECK(input_tensors->at(1).shape[1] == request_seq_len);
    FT_CHECK(input_tensors->at(2).shape[1] == request_seq_len);

    T* out_tensor = (T*)output_tensors->at(0).data;
    T* in_tensor = (T*)input_tensors->at(0).data;
    T* attention_mask = (T*)input_tensors->at(1).data;
    T* seg_mat = (T*)input_tensors->at(2).data;
    T* attr_k_head_r = (T*)input_tensors->at(3).data;

    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        hidden_units_,
                                        request_seq_len * request_batch_size,
                                        hidden_units_,
                                        attention_weights->attr_kernel_Q,
                                        hidden_units_,
                                        hidden_units_ * hidden_units_,
                                        in_tensor,
                                        hidden_units_,
                                        0,
                                        query_buf_,
                                        hidden_units_,
                                        request_seq_len * request_batch_size * hidden_units_,
                                        3);

    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          hidden_units_,
                          request_seq_len * 2,
                          hidden_units_,
                          attention_weights->attr_pos_emb,
                          hidden_units_,
                          attr_k_head_r,
                          hidden_units_,
                          k_head_r_,
                          hidden_units_);

    invokePrepareMatrixes(request_batch_size,
                          request_seq_len,
                          hidden_units_,
                          size_per_head_,
                          q_buf_,
                          q_buf_bd_,
                          q_buf_ef_,
                          k_buf_,
                          k_buf_bd_,
                          k_buf_ef_,
                          query_buf_,
                          key_buf_,
                          k_head_r_,
                          attention_weights->attr_seg_embed,
                          attention_weights->attr_bias_Q_w,
                          attention_weights->attr_bias_Q_r,
                          attention_weights->attr_bias_Q_s,
                          stream_);

    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        request_seq_len,
                                        request_seq_len,
                                        size_per_head_,
                                        k_buf_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        q_buf_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        qk_buf_,
                                        request_seq_len,
                                        request_seq_len * request_seq_len,
                                        request_batch_size * head_num_);

    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        request_seq_len * 2,
                                        request_seq_len,
                                        size_per_head_,
                                        k_buf_bd_,
                                        size_per_head_,
                                        request_seq_len * 2 * size_per_head_,
                                        q_buf_bd_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        qk_buf_bd_,
                                        request_seq_len * 2,
                                        request_seq_len * request_seq_len * 2,
                                        request_batch_size * head_num_);

    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        2,
                                        request_seq_len,
                                        size_per_head_,
                                        k_buf_ef_,
                                        size_per_head_,
                                        2 * size_per_head_,
                                        q_buf_ef_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        qk_buf_ef_,
                                        2,
                                        request_seq_len * 2,
                                        request_batch_size * head_num_);

    invokeTranspose102(request_batch_size, request_seq_len, head_num_, qk_buf_ef_trans_, qk_buf_ef_, stream_);

    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        head_num_,
                                        request_seq_len,
                                        2,
                                        qk_buf_ef_trans_,
                                        2,
                                        2 * head_num_,
                                        seg_mat,
                                        2,
                                        request_seq_len * 2,
                                        qk_buf_ef_seg_,
                                        head_num_,
                                        request_seq_len * head_num_,
                                        request_batch_size * request_seq_len);

    invokeTranspose201(request_batch_size, request_seq_len, head_num_, qk_buf_ef_seg_trans_, qk_buf_ef_seg_, stream_);
    invokeRelShiftBd(request_batch_size, head_num_, request_seq_len, qk_buf_bd_shift_, qk_buf_bd_, stream_);

    invokeCalAttnScore(request_batch_size,
                       head_num_,
                       request_seq_len,
                       size_per_head_,
                       q_scaling_,
                       attn_score_,
                       qk_buf_,
                       qk_buf_bd_shift_,
                       qk_buf_ef_seg_trans_,
                       attention_mask,
                       value_buf_trans_,
                       value_buf_,
                       stream_);

    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        size_per_head_,
                                        request_seq_len,
                                        request_seq_len,
                                        value_buf_trans_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        attn_score_,
                                        request_seq_len,
                                        request_seq_len * request_seq_len,
                                        attn_vec_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        request_batch_size * head_num_);

    invokeTranspose102v2(
        request_batch_size, request_seq_len, head_num_, size_per_head_, attn_vec_trans_, attn_vec_, stream_);

    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                          CUBLAS_OP_N,
                          hidden_units_,
                          request_seq_len * request_batch_size,
                          hidden_units_,
                          attention_weights->attr_proj_o,
                          hidden_units_,
                          attn_vec_trans_,
                          hidden_units_,
                          out_tensor,
                          hidden_units_);
}

template<typename T>
XlnetAttentionLayer<T>::XlnetAttentionLayer(size_t max_batch_size,
                                            size_t max_seq_len,
                                            size_t head_num,
                                            size_t size_per_head,
                                            float q_scaling,
                                            cudaStream_t stream,
                                            cublasMMWrapper* cublas_wrapper,
                                            IAllocator* allocator,
                                            bool is_free_buffer_after_forward):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    q_scaling_(q_scaling)
{
    hidden_units_ = head_num_ * size_per_head_;
    allocateBuffer();
}

template<typename T>
void XlnetAttentionLayer<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        k_head_r_ = (T*)allocator_->malloc(sizeof(T) * max_seq_len_ * 2 * hidden_units_, false);
        query_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_ * 3, false);
        key_buf_ = query_buf_ + max_batch_size_ * max_seq_len_ * hidden_units_;
        value_buf_ = query_buf_ + 2 * max_batch_size_ * max_seq_len_ * hidden_units_;
        q_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        k_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        qk_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * max_seq_len_ * head_num_, false);
        q_buf_bd_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        k_buf_bd_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * 2 * hidden_units_, false);
        qk_buf_bd_ =
            (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * head_num_ * max_seq_len_ * 2, false);
        qk_buf_bd_shift_ =
            (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * head_num_ * max_seq_len_, false);
        q_buf_ef_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        k_buf_ef_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * hidden_units_ * 2, false);
        qk_buf_ef_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * head_num_ * 2, false);
        qk_buf_ef_trans_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * head_num_ * 2, false);
        qk_buf_ef_seg_ =
            (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * max_seq_len_ * head_num_, false);
        qk_buf_ef_seg_trans_ =
            (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * max_seq_len_ * head_num_, false);
        attn_score_ =
            (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * max_seq_len_ * head_num_, false);
        value_buf_trans_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        attn_vec_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        attn_vec_trans_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        attn_out_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);

        is_allocate_buffer_ = true;
    }
}

template<typename T>
bool XlnetAttentionLayer<T>::isValidBatchSize(size_t batch_size)
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
bool XlnetAttentionLayer<T>::isValidSeqLen(size_t seq_len)
{
    if (max_seq_len_ == 0) {
        max_seq_len_ = seq_len;
        return true;
    }
    else {
        return seq_len <= max_seq_len_;
    }
}

template<typename T>
void XlnetAttentionLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free(k_head_r_);
        allocator_->free(query_buf_);
        allocator_->free(q_buf_);
        allocator_->free(k_buf_);
        allocator_->free(qk_buf_);
        allocator_->free(q_buf_bd_);
        allocator_->free(k_buf_bd_);
        allocator_->free(qk_buf_bd_);
        allocator_->free(qk_buf_bd_shift_);
        allocator_->free(q_buf_ef_);
        allocator_->free(k_buf_ef_);
        allocator_->free(qk_buf_ef_);
        allocator_->free(qk_buf_ef_trans_);
        allocator_->free(qk_buf_ef_seg_);
        allocator_->free(qk_buf_ef_seg_trans_);
        allocator_->free(attn_score_);
        allocator_->free(value_buf_trans_);
        allocator_->free(attn_vec_);
        allocator_->free(attn_vec_trans_);
        allocator_->free(attn_out_);

        is_allocate_buffer_ = false;
    }
}

template<typename T>
XlnetAttentionLayer<T>::~XlnetAttentionLayer()
{
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template class XlnetAttentionLayer<float>;
template class XlnetAttentionLayer<half>;

}  // namespace fastertransformer
