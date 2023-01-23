/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/layers/attention_layers/DisentangledAttentionLayer.h"
#include "src/fastertransformer/kernels/disentangled_attention_kernels.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"

namespace fastertransformer {

template<typename T>
void DisentangledAttentionLayer<T>::forward(TensorMap*                output_tensors,
                                            TensorMap*                input_tensors,
                                            const AttentionWeight<T>* attention_weights)
{
    // input_tensors:
    //      input_query [token_num, d_model],
    //      attention_mask [batch, 1, seqlen, seqlen],
    //      relative_embeddings [2*attention_span, hidden_size],
    //      padding_offset [token_num] (optional)
    //  output_tensors:
    //      hidden_features  [token_num, hidden_units]
    //      attentions [batch, num_layer, head_num, seqlen, seqlen] (optional)
    // If padding_offset.data is nullptr, then not remove padding

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    const size_t request_batch_size = input_tensors->at("attention_mask").shape[0];
    const size_t request_seq_len    = input_tensors->at("attention_mask").shape[2];
    const bool   output_attentions  = output_tensors->isExist("attentions");
    allocateBuffer(request_batch_size, request_seq_len);

    T*         hidden_features     = output_tensors->getPtr<T>("hidden_features");
    const T*   from_tensor         = input_tensors->getPtr<T>("input_query");
    const T*   attention_mask      = input_tensors->getPtr<T>("attention_mask");
    const T*   relative_embeddings = input_tensors->getPtr<T>("relative_embeddings");
    const int* padding_offset      = input_tensors->getPtr<int>("padding_offset", nullptr);

    const int m = input_tensors->at("input_query").shape[0];  // total_valid_tokens
    int       k = d_model_;                                   // hidden size
    int       n = hidden_units_;                              // num_heads * head_size
    int       s = attention_span_;                            // relative attention span ("k" in original paper)

    // Compute Q,K,V [token_num, hidden_size] --> [token_num, num_heads*head_size]
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
    }
    else {
#endif
        const bool is_batched_QKV_ = cublas_wrapper_->isFuseBatchGemm(3, n, m, k);
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
#ifdef SPARSITY_ENABLED
    }
#endif

    // add QKV bias (bias optional, can be nullptr) & permute
    // [batch, seq_len, num_heads*head_size] or [token_num, num_heads*head_size] --> [batch, num_heads, seq_len,
    // head_size] Note: aligned to padded seq len again
    if (padding_offset == nullptr) {
        invokeAddQKVBiasIA3Transpose(q_buf_2_,
                                     k_buf_2_,
                                     v_buf_2_,
                                     q_buf_,
                                     attention_weights->query_weight.bias,
                                     k_buf_,
                                     attention_weights->key_weight.bias,
                                     v_buf_,
                                     attention_weights->value_weight.bias,
                                     request_batch_size,
                                     request_seq_len,
                                     head_num_,
                                     size_per_head_,
                                     (int*)nullptr,  // suppress IA3 inputs
                                     (T*)nullptr,
                                     (T*)nullptr,
                                     stream_);
        sync_check_cuda_error();
    }
    else {
        cudaMemsetAsync(q_buf_2_, 0, 3 * request_batch_size * request_seq_len * hidden_units_ * sizeof(T), stream_);
        sync_check_cuda_error();
        invokeAddQKVBiasIA3RebuildPadding(q_buf_,
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
                                          (int*)nullptr,  // suppress IA3 inputs
                                          (T*)nullptr,
                                          (T*)nullptr,
                                          stream_);
        sync_check_cuda_error();
    }

    float scalar = 1 / (sqrtf(size_per_head_ * 1.0f) * q_scaling_);

    // compute Q*K [batch, num_heads, q_seq_len, k_seq_len]
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
                                        request_batch_size * head_num_, /* batch size */
                                        scalar /* alpha */);

    // above is content-to-content "c2c" attention, Qc*Kc^T
    // similarly, disentangled attention has two extra type of attentions (replacing the normal relative attention bias
    // w/ real attentions)

    // compute Qr, Kr [2*attention_span, hidden_size] --> [2*attention_span, num_heads*head_size]
#ifdef SPARSITY_ENABLED
    int tmp_2k = 2 * s;
    if (tmp_2k % 8 != 0) {
        tmp_2k = (tmp_2k / 8 + 1) * 8;
    }
    const int padded_2k = tmp_2k;

    if (sparse_ && cublas_wrapper_->isUseSparse(1, n, 2 * s, k)) {
        cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                n,
                                padded_2k,
                                k,
                                attention_weights->query_weight.sp_kernel,
                                relative_embeddings,
                                q_rel_buf_);
        cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                n,
                                padded_2k,
                                k,
                                attention_weights->key_weight.sp_kernel,
                                relative_embeddings,
                                k_rel_buf_);
    }
    else {
#endif
        const bool is_batched_QK_rel_ = cublas_wrapper_->isFuseBatchGemm(2, n, 2 * s, k);
        if (is_batched_QK_rel_) {
            const T* hA[]{attention_weights->query_weight.kernel,
                          attention_weights->key_weight.kernel,
                          nullptr,
                          relative_embeddings,
                          relative_embeddings,
                          nullptr,
                          q_rel_buf_,
                          k_rel_buf_,
                          nullptr};
            // Note: Here, we assume the weights of each time may be different.
            // If we can preprocess these weights before inference, we can reduce the overhead
            // caused by cudaMemcpyAsync
            cudaMemcpyAsync((void*)batch_qk_rel_kernel_ptr_, hA, sizeof(T*) * 9, cudaMemcpyHostToDevice, stream_);
            cublas_wrapper_->batchedGemm(CUBLAS_OP_N,
                                         CUBLAS_OP_N,
                                         n,
                                         2 * s,
                                         k,
                                         (const void* const*)batch_qk_rel_kernel_ptr_,
                                         n,
                                         (const void* const*)batch_qk_rel_input_ptr_,
                                         k,
                                         (void* const*)batch_qk_rel_buf_ptr_,
                                         n,
                                         2);
        }
        else {
            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  n,
                                  2 * s,
                                  k,
                                  attention_weights->query_weight.kernel,
                                  n,
                                  relative_embeddings,
                                  k,
                                  q_rel_buf_,
                                  n);

            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  n,
                                  2 * s,
                                  k,
                                  attention_weights->key_weight.kernel,
                                  n,
                                  relative_embeddings,
                                  k,
                                  k_rel_buf_,
                                  n);
        }
#ifdef SPARSITY_ENABLED
    }
#endif

    // add QK bias & permute & repeat [2*attention_span, num_heads*head_size] --> [batch, num_heads, 2*attention_span,
    // head_size]. Relative position embedding is constant across all batches, so need to repeat for later gemm
    invokeAddQKBiasTransposeRepeat(q_rel_buf_2_,
                                   k_rel_buf_2_,
                                   q_rel_buf_,
                                   attention_weights->query_weight.bias,
                                   k_rel_buf_,
                                   attention_weights->key_weight.bias,
                                   request_batch_size,
                                   2 * s,
                                   head_num_,
                                   size_per_head_,
                                   stream_);
    sync_check_cuda_error();

    // compute content-to-position "c2p" attention,  Qc*Kr^T [batch, num_heads, seq_len, 2*attention_span]
    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        2 * s,
                                        request_seq_len,
                                        size_per_head_,
                                        k_rel_buf_2_,
                                        size_per_head_,
                                        2 * s * size_per_head_,
                                        q_buf_2_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        QcKr_buf_,
                                        2 * s,
                                        request_seq_len * 2 * s,
                                        request_batch_size * head_num_, /* batch size */
                                        scalar /* alpha */);

    // compute position-to-content "p2c" attention,  Kc*Qr^T [batch, num_heads, seq_len, 2*attention_span]
    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        2 * s,
                                        request_seq_len,
                                        size_per_head_,
                                        q_rel_buf_2_,
                                        size_per_head_,
                                        2 * s * size_per_head_,
                                        k_buf_2_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        KcQr_buf_,
                                        2 * s,
                                        request_seq_len * 2 * s,
                                        request_batch_size * head_num_, /* batch size */
                                        scalar /* alpha */);

    // gather & add c2c+c2p+p2c. In-place operation
    invokeDisentangledAttention(
        qk_buf_, qk_buf_, QcKr_buf_, KcQr_buf_, request_batch_size * head_num_, request_seq_len, s, stream_);
    sync_check_cuda_error();

    // softmax(QK)
    MaskedSoftmaxParam<T, T> param;
    param.attention_score    = qk_buf_;         // (batch_size, head_num, q_length, k_length)
    param.qk                 = qk_buf_;         // (batch_size, head_num, q_length, k_length)
    param.attention_mask     = attention_mask;  // (batch_size, q_length, k_length)
    param.batch_size         = request_batch_size;
    param.q_length           = request_seq_len;
    param.k_length           = request_seq_len;
    param.num_heads          = head_num_;
    param.qk_scale           = 1.0f;
    param.linear_bias_slopes = nullptr;
    invokeMaskedSoftmax(param, stream_);
    sync_check_cuda_error();

    // save attention results
    // Note: "transpose" is not transpose, it's just writting attention results to certain layer, [B, M, S, S] --> [B,
    // L, M, S, S]
    if (output_attentions) {
        invokeTransposeAttentions<T>(output_tensors->at("attentions"),
                                     {MEMORY_GPU,
                                      getTensorType<T>(),
                                      {request_batch_size, head_num_, request_seq_len, request_seq_len},
                                      qk_buf_},
                                     stream_);
    }
    sync_check_cuda_error();

    // compute softmax(QK) * V
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

    // permute [batch_size, num_heads, seq_len, head_size] --> [batch_size, seq_len, num_heads, head_size] or
    // [token_num, num_heads, head_size] w/ padding removal
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
        invokeTransposeAttentionOutRemovePadding(qkv_buf_,
                                                 qkv_buf_2_,
                                                 m,
                                                 request_batch_size,
                                                 request_seq_len,
                                                 head_num_,
                                                 size_per_head_,
                                                 padding_offset,
                                                 (float*)nullptr,
                                                 0,
                                                 stream_);
    }

    // switch Linear dimension
    k = hidden_units_;
    n = d_model_;

    // attention output Linear layer (bias and layernorm are handled outside)
#ifdef SPARSITY_ENABLED
    if (sparse_ && cublas_wrapper_->isUseSparse(1, n, m, k)) {
        cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                n,
                                m_padded,
                                k,
                                attention_weights->attention_output_weight.sp_kernel,
                                qkv_buf_2_,
                                hidden_features);
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
                              hidden_features,
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
DisentangledAttentionLayer<T>::DisentangledAttentionLayer(size_t           max_batch_size,
                                                          size_t           max_seq_len,
                                                          size_t           head_num,
                                                          size_t           size_per_head,
                                                          size_t           attention_span,
                                                          float            q_scaling,
                                                          cudaStream_t     stream,
                                                          cublasMMWrapper* cublas_wrapper,
                                                          IAllocator*      allocator,
                                                          bool             is_free_buffer_after_forward,
                                                          bool             sparse):
    DisentangledAttentionLayer(max_batch_size,
                               max_seq_len,
                               head_num,
                               size_per_head,
                               attention_span,
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
DisentangledAttentionLayer<T>::DisentangledAttentionLayer(size_t           max_batch_size,
                                                          size_t           max_seq_len,
                                                          size_t           head_num,
                                                          size_t           size_per_head,
                                                          size_t           attention_span,
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
    attention_span_(attention_span),
    d_model_(d_model),
    sparse_(sparse),
    q_scaling_(q_scaling)
{
}

template<typename T>
DisentangledAttentionLayer<T>::DisentangledAttentionLayer(DisentangledAttentionLayer<T> const& attention_layer):
    BaseAttentionLayer<T>(attention_layer.stream_,
                          attention_layer.cublas_wrapper_,
                          attention_layer.allocator_,
                          attention_layer.is_free_buffer_after_forward_),
    head_num_(attention_layer.head_num_),
    size_per_head_(attention_layer.size_per_head_),
    hidden_units_(attention_layer.hidden_units_),
    attention_span_(attention_layer.attention_span_),
    d_model_(attention_layer.d_model_),
    sparse_(attention_layer.sparse_),
    q_scaling_(attention_layer.q_scaling_)
{
}

template<typename T>
DisentangledAttentionLayer<T>::~DisentangledAttentionLayer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void DisentangledAttentionLayer<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void DisentangledAttentionLayer<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    q_buf_   = (T*)allocator_->reMalloc(q_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    k_buf_   = (T*)allocator_->reMalloc(k_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    v_buf_   = (T*)allocator_->reMalloc(v_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    q_buf_2_ = (T*)allocator_->reMalloc(q_buf_2_, sizeof(T) * 3 * batch_size * seq_len * hidden_units_, false);
    k_buf_2_ = q_buf_2_ + batch_size * seq_len * hidden_units_;
    v_buf_2_ = k_buf_2_ + batch_size * seq_len * hidden_units_;
    qk_buf_  = (T*)allocator_->reMalloc(qk_buf_, sizeof(T) * batch_size * head_num_ * seq_len * seq_len, false);
    q_rel_buf_ =
        (T*)allocator_->reMalloc(q_rel_buf_, sizeof(T) * 2 * 2 * attention_span_ * head_num_ * size_per_head_, false);
    k_rel_buf_   = q_rel_buf_ + 2 * attention_span_ * head_num_ * size_per_head_;
    q_rel_buf_2_ = (T*)allocator_->reMalloc(
        q_rel_buf_2_, sizeof(T) * 2 * batch_size * head_num_ * 2 * attention_span_ * size_per_head_, false);
    k_rel_buf_2_ = q_rel_buf_2_ + batch_size * head_num_ * 2 * attention_span_ * size_per_head_;
    QcKr_buf_    = (T*)allocator_->reMalloc(
        QcKr_buf_, sizeof(T) * 2 * batch_size * head_num_ * seq_len * 2 * attention_span_, false);
    KcQr_buf_  = QcKr_buf_ + batch_size * head_num_ * seq_len * 2 * attention_span_;
    qkv_buf_   = (T*)allocator_->reMalloc(qkv_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    qkv_buf_2_ = (T*)allocator_->reMalloc(qkv_buf_2_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    batch_qkv_kernel_ptr_    = (T**)allocator_->reMalloc(batch_qkv_kernel_ptr_, sizeof(T*) * 12, false);
    batch_qkv_input_ptr_     = batch_qkv_kernel_ptr_ + 4;
    batch_qkv_buf_ptr_       = batch_qkv_input_ptr_ + 4;
    batch_qk_rel_kernel_ptr_ = (T**)allocator_->reMalloc(batch_qk_rel_kernel_ptr_, sizeof(T*) * 9, false);
    batch_qk_rel_input_ptr_  = batch_qk_rel_kernel_ptr_ + 3;
    batch_qk_rel_buf_ptr_    = batch_qk_rel_input_ptr_ + 3;

    is_allocate_buffer_ = true;
}

template<typename T>
void DisentangledAttentionLayer<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&q_buf_));
        allocator_->free((void**)(&k_buf_));
        allocator_->free((void**)(&v_buf_));
        allocator_->free((void**)(&q_buf_2_));
        allocator_->free((void**)(&qk_buf_));
        allocator_->free((void**)(&q_rel_buf_));
        allocator_->free((void**)(&q_rel_buf_2_));
        allocator_->free((void**)(&QcKr_buf_));
        allocator_->free((void**)(&qkv_buf_));
        allocator_->free((void**)(&qkv_buf_2_));
        allocator_->free((void**)(&batch_qkv_kernel_ptr_));
        allocator_->free((void**)(&batch_qk_rel_kernel_ptr_));
        sync_check_cuda_error();
        is_allocate_buffer_ = false;
    }
}

template class DisentangledAttentionLayer<float>;
template class DisentangledAttentionLayer<half>;
#ifdef ENABLE_BF16
template class DisentangledAttentionLayer<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
