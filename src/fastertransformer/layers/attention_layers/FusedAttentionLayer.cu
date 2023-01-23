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

#include "src/fastertransformer/layers/attention_layers/FusedAttentionLayer.h"

namespace fastertransformer {

__global__ void trt_add_QKV_bias(half2*       qkv_buf,
                                 const half2* Q,
                                 const half2* bias_Q,
                                 const half2* K,
                                 const half2* bias_K,
                                 const half2* V,
                                 const half2* bias_V,
                                 const int    valid_word_num,
                                 const int    head_num,
                                 const int    size_per_head)
{
    // Add bias, and then transpose from
    // [3, valid_word_num, head, size] -> [valid_word_num, head, 3, size]

    // const int seq_id = blockIdx.x % valid_word_num;
    // const int qkv_id = (blockIdx.x - seq_id) / valid_word_num;
    const int seq_id = blockIdx.x;

    for (int index = threadIdx.x; index < head_num * size_per_head; index += blockDim.x) {
        const int size_id = index % size_per_head;
        const int head_id = (index - size_id) / size_per_head;

        const int target_offset = blockIdx.x * head_num * 3 * size_per_head + head_id * 3 * size_per_head;
        const int src_id        = seq_id * head_num * size_per_head + index;

        qkv_buf[target_offset + 0 * size_per_head + size_id] = Q[src_id] + bias_Q[index];
        qkv_buf[target_offset + 1 * size_per_head + size_id] = K[src_id] + bias_K[index];
        qkv_buf[target_offset + 2 * size_per_head + size_id] = V[src_id] + bias_V[index];
    }
}

template<typename T>
void FusedAttentionLayer<T>::invokeTrtAddQkvBias(size_t token_num, const AttentionWeight<T>* attention_weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    dim3 grid(token_num);
    dim3 block(min((int)(head_num_ * size_per_head_ / 2), 512));

    trt_add_QKV_bias<<<grid, block, 0, stream_>>>((half2*)qkv_buf_,
                                                  (const half2*)q_buf_,
                                                  (const half2*)attention_weights->query_weight.bias,
                                                  (const half2*)k_buf_,
                                                  (const half2*)attention_weights->key_weight.bias,
                                                  (const half2*)v_buf_,
                                                  (const half2*)attention_weights->value_weight.bias,
                                                  token_num,
                                                  head_num_,
                                                  size_per_head_ / 2);
}

template<typename T>
void FusedAttentionLayer<T>::forward(TensorMap*                output_tensors,
                                     TensorMap*                input_tensors,
                                     const AttentionWeight<T>* attention_weights)
{
    // input_tensors: [input_query (h_token_num, d_model),
    //                 attention_mask (batch, 1, seqlen, seqlen),
    //                 padding_offset (batch + 1 or batch * 2 + 1))]
    // If padding_offset.data is nullptr, then not remove padding

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    const int request_batch_size = input_tensors->at("attention_mask").shape[0];
    const int request_seq_len    = input_tensors->at("attention_mask").shape[2];
    allocateBuffer(request_batch_size, request_seq_len);

    T*         attention_out  = output_tensors->getPtr<T>("hidden_features");
    const T*   from_tensor    = input_tensors->getPtr<T>("input_query");
    const T*   attention_mask = input_tensors->getPtr<T>("attention_mask");
    const int* padding_offset = input_tensors->getPtr<int>("padding_offset");

    size_t m_tmp = input_tensors->at("input_query").shape[0];
    if (m_tmp % 8 != 0) {
        m_tmp = (m_tmp / 8 + 1) * 8;
    }
    const size_t m = input_tensors->at("input_query").shape[0];
    int          k = d_model_;
    int          n = hidden_units_;

#ifdef SPARSITY_ENABLED
    const size_t m_padded = m_tmp;
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
            check_cuda_error(
                cudaMemcpyAsync((void*)batch_qkv_kernel_ptr_, hA, sizeof(T*) * 12, cudaMemcpyHostToDevice, stream_));
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

    invokeTrtAddQkvBias(m, attention_weights);
    sync_check_cuda_error();

    int S = dispatcher_fp16->getSFromMaxSeqLen(request_seq_len);
    FT_CHECK(dispatcher_fp16->isValid(S, false));
    const int B = input_tensors->at("padding_offset").shape[0] - 1;
    dispatcher_fp16->setup(S, B);
    dispatcher_fp16->run(qkv_buf_, nullptr, padding_offset, attn_workspace_, qkv_buf_2_, stream_);
    sync_check_cuda_error();

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
}

template<typename T>
FusedAttentionLayer<T>::FusedAttentionLayer(size_t           max_batch_size,
                                            size_t           max_seq_len,
                                            size_t           head_num,
                                            size_t           size_per_head,
                                            size_t           d_model,
                                            int              sm,
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
    sm_(sm),
    q_scaling_(q_scaling),
    sparse_(sparse)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (((sm_ == kSM_70 || sm_ == kSM_86 || sm_ == kSM_80 || sm_ == kSM_75 || sm_ == kSM_72) && size_per_head_ == 64)
        || ((sm_ == kSM_86 || sm_ == kSM_80 || sm_ == kSM_75) && size_per_head_ == 32)) {
        dispatcher_fp16.reset(new FusedMHARunnerFP16v2(head_num_, size_per_head_, sm_, q_scaling_));
    }
    else {
        throw std::runtime_error(std::string("[FT][ERROR] FusedAttentionLayer not support \n"));
    }
    hidden_units_ = head_num_ * size_per_head_;
}

template<typename T>
FusedAttentionLayer<T>::FusedAttentionLayer(FusedAttentionLayer<T> const& attention_layer):
    FusedAttentionLayer(0,
                        0,
                        attention_layer.head_num_,
                        attention_layer.size_per_head_,
                        attention_layer.d_model_,
                        attention_layer.sm_,
                        attention_layer.q_scaling_,
                        attention_layer.stream_,
                        attention_layer.cublas_wrapper_,
                        attention_layer.allocator_,
                        attention_layer.is_free_buffer_after_forward_,
                        attention_layer.sparse_)
{
}

template<typename T>
FusedAttentionLayer<T>::~FusedAttentionLayer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void FusedAttentionLayer<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void FusedAttentionLayer<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    q_buf_          = (T*)allocator_->reMalloc(q_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    k_buf_          = (T*)allocator_->reMalloc(k_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    v_buf_          = (T*)allocator_->reMalloc(v_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    qkv_buf_        = (T*)allocator_->reMalloc(qkv_buf_, sizeof(T) * 3 * batch_size * seq_len * hidden_units_, false);
    qkv_buf_2_      = (T*)allocator_->reMalloc(qkv_buf_2_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    attn_workspace_ = (T*)allocator_->reMalloc(attn_workspace_, dispatcher_fp16->getWorkspaceSize(), false);

    batch_qkv_kernel_ptr_ = (T**)allocator_->reMalloc(batch_qkv_kernel_ptr_, sizeof(T*) * 12, false);
    batch_qkv_input_ptr_  = batch_qkv_kernel_ptr_ + 4;
    batch_qkv_buf_ptr_    = batch_qkv_input_ptr_ + 4;
    is_allocate_buffer_   = true;
}

template<typename T>
void FusedAttentionLayer<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&q_buf_));
        allocator_->free((void**)(&k_buf_));
        allocator_->free((void**)(&v_buf_));
        allocator_->free((void**)(&qkv_buf_));
        allocator_->free((void**)(&qkv_buf_2_));
        allocator_->free((void**)(&attn_workspace_));
        allocator_->free((void**)(&batch_qkv_kernel_ptr_));
        sync_check_cuda_error();
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool FusedAttentionLayer<T>::isValidSeqLen(const size_t seq_len)
{
    return true;
}

template class FusedAttentionLayer<float>;
template class FusedAttentionLayer<half>;
#ifdef ENABLE_BF16
template class FusedAttentionLayer<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
