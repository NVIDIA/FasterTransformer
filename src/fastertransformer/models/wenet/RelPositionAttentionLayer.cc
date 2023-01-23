/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2022.  Authored by Yuqing Ding.
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

#include "src/fastertransformer/models/wenet/RelPositionAttentionLayer.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"
#include "src/fastertransformer/models/wenet/WenetKernels.h"

namespace fastertransformer {

template<typename T>
void RelPositionAttentionLayer<T>::forward(TensorMap*                output_tensors,
                                           TensorMap*                input_tensors,
                                           const AttentionWeight<T>* attention_weights)
{
    FT_CHECK(false && "ERROR:Not supported.");
}

template<typename T>
void RelPositionAttentionLayer<T>::forward(TensorMap*                                      output_tensors,
                                           TensorMap*                                      input_tensors,
                                           const RelPositionMultiHeadedAttentionWeight<T>* attention_weights)
{
    // input_tensors:
    //      input_query (token_num, d_model),
    //      attention_mask (batch, 1, seqlen, seqlen),
    //      padding_offset (token_num)   Note: padding_offset.data must be nullptr
    //      pos_emb (token_num, d_model)
    //      relative_attention_bias (optional)
    // If padding_offset.data is nullptr, then not remove padding

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() == 3 || input_tensors->size() == 4 || input_tensors->size() == 5);
    const int request_batch_size = input_tensors->at("attention_mask").shape[0];
    const int request_seq_len    = input_tensors->at("attention_mask").shape[2];
    allocateBuffer(request_batch_size, request_seq_len);

    T*         attention_out           = output_tensors->getPtr<T>("attention_out");
    const T*   from_tensor             = input_tensors->getPtr<T>("normed_ffn_out");
    const T*   attention_mask          = input_tensors->getPtr<T>("attention_mask");
    const int* padding_offset          = input_tensors->getPtr<int>("padding_offset", nullptr);
    const T*   pos_emb                 = input_tensors->getPtr<T>("pos_emb");
    const T*   relative_attention_bias = input_tensors->getPtr<T>("relative_attention_bias", nullptr);

    bool use_relative_position_bias = relative_attention_bias != nullptr ? true : false;

    const int m = input_tensors->at("normed_ffn_out").shape[0];
    int       k = d_model_;
    int       n = hidden_units_;
#ifdef SPARSITY_ENABLED
    int m_tmp = m;
    if (m_tmp % 8 != 0) {
        m_tmp = (m_tmp / 8 + 1) * 8;
    }
    const int m_padded = m_tmp;

    if (sparse_ && cublas_wrapper_->isUseSparse(1, n, m, k)) {
        cublas_wrapper_->SpGemm(
            CUBLAS_OP_N, CUBLAS_OP_N, n, m_padded, k, attention_weights->query_weight.sp_kernel, from_tensor, q_buf_);
        cublas_wrapper_->SpGemm(
            CUBLAS_OP_N, CUBLAS_OP_N, n, m_padded, k, attention_weights->key_weight.sp_kernel, from_tensor, k_buf_);
        cublas_wrapper_->SpGemm(
            CUBLAS_OP_N, CUBLAS_OP_N, n, m_padded, k, attention_weights->value_weight.sp_kernel, from_tensor, v_buf_);
        cublas_wrapper_->SpGemm(
            CUBLAS_OP_N, CUBLAS_OP_N, n, m_padded, k, attention_weights->pos_weight.sp_kernel, pos_emb, p_buf_);
    }
    else {
#endif
        const bool is_batched_QKV_ = true;  // cublas_wrapper_->isFuseBatchGemm(3, n, m, k);  // enforce this option
        if (is_batched_QKV_) {
            const T* hA[]{attention_weights->query_weight.kernel,
                          attention_weights->key_weight.kernel,
                          attention_weights->value_weight.kernel,
                          nullptr,
                          from_tensor,
                          from_tensor,
                          from_tensor,
                          nullptr,
                          q_buf_,
                          k_buf_,
                          v_buf_,
                          nullptr};
            // Note: Here, we assume the weights of each time may be different.
            // If we can preprocess these weights before inference, we can reduce the overhead
            // caused by cudaMemcpyAsync
            cudaMemcpyAsync((void*)batch_qkv_kernel_ptr_, hA, sizeof(T*) * 12, cudaMemcpyHostToDevice, stream_);
            cublas_wrapper_->batchedGemm(CUBLAS_OP_N,
                                         CUBLAS_OP_N,
                                         n,
                                         m,
                                         k,
                                         (const void* const*)batch_qkv_kernel_ptr_,
                                         n,
                                         (const void* const*)batch_qkv_input_ptr_,
                                         k,
                                         (void* const*)batch_qkv_buf_ptr_,
                                         n,
                                         3);
        }
        else {
            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  n,
                                  m,
                                  k,
                                  attention_weights->query_weight.kernel,
                                  n,
                                  from_tensor,
                                  k,
                                  q_buf_,
                                  n);

            cublas_wrapper_->Gemm(
                CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, attention_weights->key_weight.kernel, n, from_tensor, k, k_buf_, n);

            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  n,
                                  m,
                                  k,
                                  attention_weights->value_weight.kernel,
                                  n,
                                  from_tensor,
                                  k,
                                  v_buf_,
                                  n);
        }
        cublas_wrapper_->Gemm(
            CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, attention_weights->pos_weight.kernel, n, pos_emb, k, p_buf_, n);

#ifdef SPARSITY_ENABLED
    }
#endif
    sync_check_cuda_error();

    if (padding_offset == nullptr) {
        invokeAddQKVPBiasTranspose(q_buf_2_,
                                   k_buf_2_,
                                   v_buf_2_,
                                   q_buf_,
                                   attention_weights->query_weight.bias,
                                   k_buf_,
                                   attention_weights->key_weight.bias,
                                   v_buf_,
                                   attention_weights->value_weight.bias,
                                   p_buf_2_,
                                   p_buf_,
                                   q_buf_bias_v_,
                                   attention_weights->pos_bias_u,
                                   attention_weights->pos_bias_v,
                                   request_batch_size,
                                   request_seq_len,
                                   head_num_,
                                   size_per_head_,
                                   stream_);
        sync_check_cuda_error();
    }
    else {
        // TODO: support padding_offset
        /*
        cudaMemsetAsync(q_buf_2_, 0, 3 * request_batch_size * request_seq_len * hidden_units_ * sizeof(T), stream_);
        sync_check_cuda_error();
        invokeAddQKVBiasRebuildPadding(q_buf_,
                                       attention_weights->query_weight.bias,
                                       k_buf_,
                                       attention_weights->key_weight.bias,
                                       v_buf_,
                                       attention_weights->value_weight.bias,
                                       q_buf_2_,
                                       k_buf_2_,
                                       v_buf_2_,
                                       request_batch_size,
                                       request_seq_len,
                                       head_num_,
                                       size_per_head_,
                                       m,
                                       padding_offset,
                                       stream_);
        sync_check_cuda_error();
        */
    }

    float scalar = 1 / (sqrtf(size_per_head_ * 1.0f) * q_scaling_);

    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        request_seq_len,
                                        request_seq_len,
                                        size_per_head_,
                                        k_buf_2_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        q_buf_2_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        qk_buf_,
                                        request_seq_len,
                                        request_seq_len * request_seq_len,
                                        request_batch_size * head_num_,
                                        scalar);

    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        request_seq_len,
                                        request_seq_len,
                                        size_per_head_,
                                        p_buf_2_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        q_buf_bias_v_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        qp_buf_,
                                        request_seq_len,
                                        request_seq_len * request_seq_len,
                                        request_batch_size * head_num_,
                                        scalar);

    if (use_relative_position_bias) {
        invokeAddRelativeAttentionBias(
            qk_buf_, relative_attention_bias, request_batch_size, head_num_, request_seq_len, stream_);
    }

    invokeAddMaskedSoftMax(
        qk_buf_, qk_buf_, qp_buf_, attention_mask, request_batch_size, request_seq_len, head_num_, (T)1.0f, stream_);
    sync_check_cuda_error();

    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        size_per_head_,
                                        request_seq_len,
                                        request_seq_len,
                                        v_buf_2_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        qk_buf_,
                                        request_seq_len,
                                        request_seq_len * request_seq_len,
                                        qkv_buf_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        request_batch_size * head_num_);

    if (padding_offset == nullptr) {

        invokeTransposeQKV(qkv_buf_2_,
                           qkv_buf_,
                           request_batch_size,
                           request_seq_len,
                           head_num_,
                           size_per_head_,
                           (float*)nullptr,
                           0,
                           stream_);
        sync_check_cuda_error();
    }
    else {
        // TODO: support padding_offset
        /*
                invokeTransposeAttentionOutRemovePadding(qkv_buf_,
                                                         qkv_buf_2_,
                                                         m,
                                                         request_batch_size,
                                                         request_seq_len,
                                                         head_num_,
                                                         size_per_head_,
                                                         padding_offset,
                                                         stream_);
                                                         */
    }

    k = hidden_units_;
    n = d_model_;

#ifdef SPARSITY_ENABLED
    if (sparse_ && cublas_wrapper_->isUseSparse(1, n, m, k)) {
        cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                n,
                                m_padded,
                                k,
                                attention_weights->attention_output_weight.sp_kernel,
                                qkv_buf_2_,
                                attention_out);
    }
    else {
#endif
        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              n,
                              m,
                              k,
                              attention_weights->attention_output_weight.kernel,
                              n,
                              qkv_buf_2_,
                              k,
                              attention_out,
                              n);
#ifdef SPARSITY_ENABLED
    }
#endif

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template<typename T>
RelPositionAttentionLayer<T>::RelPositionAttentionLayer(size_t           max_batch_size,
                                                        size_t           max_seq_len,
                                                        size_t           head_num,
                                                        size_t           size_per_head,
                                                        float            q_scaling,
                                                        cudaStream_t     stream,
                                                        cublasMMWrapper* cublas_wrapper,
                                                        IAllocator*      allocator,
                                                        bool             is_free_buffer_after_forward,
                                                        bool             sparse):
    RelPositionAttentionLayer(max_batch_size,
                              max_seq_len,
                              head_num,
                              size_per_head,
                              head_num * size_per_head,
                              q_scaling,
                              stream,
                              cublas_wrapper,
                              allocator,
                              is_free_buffer_after_forward,
                              sparse)
{
}

template<typename T>
RelPositionAttentionLayer<T>::RelPositionAttentionLayer(size_t           max_batch_size,
                                                        size_t           max_seq_len,
                                                        size_t           head_num,
                                                        size_t           size_per_head,
                                                        size_t           d_model,
                                                        float            q_scaling,
                                                        cudaStream_t     stream,
                                                        cublasMMWrapper* cublas_wrapper,
                                                        IAllocator*      allocator,
                                                        bool             is_free_buffer_after_forward,
                                                        bool             sparse):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num_ * size_per_head_),
    d_model_(d_model),
    sparse_(sparse),
    q_scaling_(q_scaling)
{
}

template<typename T>
RelPositionAttentionLayer<T>::RelPositionAttentionLayer(RelPositionAttentionLayer<T> const& attention_layer):
    BaseAttentionLayer<T>(attention_layer.stream_,
                          attention_layer.cublas_wrapper_,
                          attention_layer.allocator_,
                          attention_layer.is_free_buffer_after_forward_),
    head_num_(attention_layer.head_num_),
    size_per_head_(attention_layer.size_per_head_),
    hidden_units_(attention_layer.hidden_units_),
    d_model_(attention_layer.d_model_),
    sparse_(attention_layer.sparse_),
    q_scaling_(attention_layer.q_scaling_)
{
}

template<typename T>
RelPositionAttentionLayer<T>::~RelPositionAttentionLayer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void RelPositionAttentionLayer<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void RelPositionAttentionLayer<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    q_buf_   = (T*)allocator_->reMalloc(q_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    k_buf_   = (T*)allocator_->reMalloc(k_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    v_buf_   = (T*)allocator_->reMalloc(v_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    q_buf_2_ = (T*)allocator_->reMalloc(q_buf_2_, sizeof(T) * 3 * batch_size * seq_len * hidden_units_, false);
    k_buf_2_ = q_buf_2_ + batch_size * seq_len * hidden_units_;
    v_buf_2_ = k_buf_2_ + batch_size * seq_len * hidden_units_;

    p_buf_ = (T*)allocator_->reMalloc(p_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);  // (BS, N, H)
    p_buf_2_ =
        (T*)allocator_->reMalloc(p_buf_2_, sizeof(T) * batch_size * seq_len * hidden_units_, false);  // (B,N,H,S)
    q_buf_bias_v_ =
        (T*)allocator_->reMalloc(q_buf_bias_v_, sizeof(T) * batch_size * seq_len * hidden_units_, false);  // (B,N,S,H)
    qp_buf_ =
        (T*)allocator_->reMalloc(qp_buf_, sizeof(T) * batch_size * head_num_ * seq_len * seq_len, false);  // (B,N,S,S)

    qk_buf_    = (T*)allocator_->reMalloc(qk_buf_, sizeof(T) * batch_size * head_num_ * seq_len * seq_len, false);
    qkv_buf_   = (T*)allocator_->reMalloc(qkv_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    qkv_buf_2_ = (T*)allocator_->reMalloc(qkv_buf_2_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    batch_qkv_kernel_ptr_ = (T**)allocator_->reMalloc(batch_qkv_kernel_ptr_, sizeof(T*) * 12, false);
    batch_qkv_input_ptr_  = batch_qkv_kernel_ptr_ + 4;
    batch_qkv_buf_ptr_    = batch_qkv_input_ptr_ + 4;
}

template<typename T>
void RelPositionAttentionLayer<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    allocator_->free((void**)(&q_buf_));
    allocator_->free((void**)(&k_buf_));
    allocator_->free((void**)(&v_buf_));
    allocator_->free((void**)(&q_buf_2_));

    allocator_->free((void**)(&p_buf_));
    allocator_->free((void**)(&p_buf_2_));
    allocator_->free((void**)(&q_buf_bias_v_));
    allocator_->free((void**)(&qp_buf_));

    allocator_->free((void**)(&qk_buf_));
    allocator_->free((void**)(&qkv_buf_));
    allocator_->free((void**)(&qkv_buf_2_));
    allocator_->free((void**)(&batch_qkv_kernel_ptr_));
    sync_check_cuda_error();
}

template class RelPositionAttentionLayer<float>;
template class RelPositionAttentionLayer<half>;

}  // namespace fastertransformer