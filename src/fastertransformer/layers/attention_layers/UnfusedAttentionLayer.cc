/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/layers/attention_layers/UnfusedAttentionLayer.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"

namespace fastertransformer {

template<typename T>
void UnfusedAttentionLayer<T>::forward(TensorMap*                output_tensors,
                                       TensorMap*                input_tensors,
                                       const AttentionWeight<T>* attention_weights)
{
    // input_tensors:
    //      input_query [token_num, d_model],
    //      attention_mask [batch, 1, seqlen, seqlen],
    //      padding_offset [token_num] (optional)
    //      relative_attention_bias [head_num, seq_len, seq_len] (optional)
    //      linear_bias_slopes [head_num] (optional)
    //      ia3_tasks [batch] (optional)
    //  output_tensors:
    //      hidden_features  [token_num, hidden_units]
    //      attentions [batch, num_layer, head_num, seqlen, seqlen] (optional)
    // If padding_offset.data is nullptr, then not remove padding

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    const size_t request_batch_size = input_tensors->at("attention_mask").shape[0];
    const size_t request_seq_len    = input_tensors->at("attention_mask").shape[2];
    const bool   output_attentions  = output_tensors->isExist("attentions");
    allocateBuffer(request_batch_size, request_seq_len);

    T*         hidden_features         = output_tensors->getPtr<T>("hidden_features");
    const T*   from_tensor             = input_tensors->getPtr<T>("input_query");
    const T*   attention_mask          = input_tensors->getPtr<T>("attention_mask");
    const int* padding_offset          = input_tensors->getPtr<int>("padding_offset", nullptr);
    const T*   relative_attention_bias = input_tensors->getPtr<T>("relative_attention_bias", nullptr);
    const T*   linear_bias_slopes      = input_tensors->getPtr<T>("linear_bias_slopes", nullptr);
    const int* ia3_tasks               = input_tensors->getPtr<int>("ia3_tasks", nullptr);

    bool with_bias                  = attention_weights->query_weight.bias != nullptr ? true : false;
    bool use_relative_position_bias = relative_attention_bias != nullptr ? true : false;

    const int m = input_tensors->at("input_query").shape[0];
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
                                     ia3_tasks,
                                     attention_weights->ia3_key_weight.kernel,
                                     attention_weights->ia3_value_weight.kernel,
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
                                          ia3_tasks,
                                          attention_weights->ia3_key_weight.kernel,
                                          attention_weights->ia3_value_weight.kernel,
                                          stream_);
        sync_check_cuda_error();
    }
    // print_abs_mean(q_buf_2_, m * hidden_units_, stream_, "q_buf_2_");
    // print_to_screen(q_buf_2_, 3);
    // print_abs_mean(k_buf_2_, m * hidden_units_, stream_, "k_buf_2_");
    // print_to_screen(k_buf_2_, 3);
    // print_abs_mean(v_buf_2_, m * hidden_units_, stream_, "v_buf_2_");
    // print_to_screen(v_buf_2_, 3);
    // exit(0);
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

    // TODO (fuse with softMax)
    if (use_relative_position_bias) {
        invokeAddRelativeAttentionBias(
            qk_buf_, relative_attention_bias, request_batch_size, head_num_, request_seq_len, stream_);
    }

    MaskedSoftmaxParam<T, T> param;
    param.attention_score    = qk_buf_;         // (batch_size, head_num, q_length, k_length)
    param.qk                 = qk_buf_;         // (batch_size, head_num, q_length, k_length)
    param.attention_mask     = attention_mask;  // (batch_size, q_length, k_length)
    param.batch_size         = request_batch_size;
    param.q_length           = request_seq_len;
    param.k_length           = request_seq_len;
    param.num_heads          = head_num_;
    param.qk_scale           = 1.0f;
    param.linear_bias_slopes = linear_bias_slopes;  // (head_num,), optional
    invokeMaskedSoftmax(param, stream_);
    sync_check_cuda_error();

    if (output_attentions) {
        invokeTransposeAttentions<T>(output_tensors->at("attentions"),
                                     {MEMORY_GPU,
                                      getTensorType<T>(),
                                      {request_batch_size, head_num_, request_seq_len, request_seq_len},
                                      qk_buf_},
                                     stream_);
    }
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
UnfusedAttentionLayer<T>::UnfusedAttentionLayer(size_t           max_batch_size,
                                                size_t           max_seq_len,
                                                size_t           head_num,
                                                size_t           size_per_head,
                                                float            q_scaling,
                                                cudaStream_t     stream,
                                                cublasMMWrapper* cublas_wrapper,
                                                IAllocator*      allocator,
                                                bool             is_free_buffer_after_forward,
                                                bool             sparse):
    UnfusedAttentionLayer(max_batch_size,
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
UnfusedAttentionLayer<T>::UnfusedAttentionLayer(size_t           max_batch_size,
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
    d_model_(d_model),
    hidden_units_(head_num_ * size_per_head_),
    sparse_(sparse),
    q_scaling_(q_scaling)
{
}

template<typename T>
UnfusedAttentionLayer<T>::UnfusedAttentionLayer(UnfusedAttentionLayer<T> const& attention_layer):
    BaseAttentionLayer<T>(attention_layer.stream_,
                          attention_layer.cublas_wrapper_,
                          attention_layer.allocator_,
                          attention_layer.is_free_buffer_after_forward_),
    head_num_(attention_layer.head_num_),
    size_per_head_(attention_layer.size_per_head_),
    d_model_(attention_layer.d_model_),
    hidden_units_(attention_layer.hidden_units_),
    sparse_(attention_layer.sparse_),
    q_scaling_(attention_layer.q_scaling_)
{
}

template<typename T>
UnfusedAttentionLayer<T>::~UnfusedAttentionLayer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void UnfusedAttentionLayer<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void UnfusedAttentionLayer<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    q_buf_     = (T*)allocator_->reMalloc(q_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    k_buf_     = (T*)allocator_->reMalloc(k_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    v_buf_     = (T*)allocator_->reMalloc(v_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    q_buf_2_   = (T*)allocator_->reMalloc(q_buf_2_, sizeof(T) * 3 * batch_size * seq_len * hidden_units_, false);
    k_buf_2_   = q_buf_2_ + batch_size * seq_len * hidden_units_;
    v_buf_2_   = k_buf_2_ + batch_size * seq_len * hidden_units_;
    qk_buf_    = (T*)allocator_->reMalloc(qk_buf_, sizeof(T) * batch_size * head_num_ * seq_len * seq_len, false);
    qkv_buf_   = (T*)allocator_->reMalloc(qkv_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    qkv_buf_2_ = (T*)allocator_->reMalloc(qkv_buf_2_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    batch_qkv_kernel_ptr_ = (T**)allocator_->reMalloc(batch_qkv_kernel_ptr_, sizeof(T*) * 12, false);
    batch_qkv_input_ptr_  = batch_qkv_kernel_ptr_ + 4;
    batch_qkv_buf_ptr_    = batch_qkv_input_ptr_ + 4;
    is_allocate_buffer_   = true;
}

template<typename T>
void UnfusedAttentionLayer<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&q_buf_));
        allocator_->free((void**)(&k_buf_));
        allocator_->free((void**)(&v_buf_));
        allocator_->free((void**)(&q_buf_2_));
        allocator_->free((void**)(&qk_buf_));
        allocator_->free((void**)(&qkv_buf_));
        allocator_->free((void**)(&qkv_buf_2_));
        allocator_->free((void**)(&batch_qkv_kernel_ptr_));
        sync_check_cuda_error();
        is_allocate_buffer_ = false;
    }
}

template class UnfusedAttentionLayer<float>;
template class UnfusedAttentionLayer<half>;
#ifdef ENABLE_BF16
template class UnfusedAttentionLayer<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
