/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include "src/fastertransformer/layers/attention_layers/GptContextAttentionLayer.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"

namespace fastertransformer {

template<typename T>
void GptContextAttentionLayer<T>::forward(std::vector<fastertransformer::Tensor>*       output_tensors,
                                          const std::vector<fastertransformer::Tensor>* input_tensors,
                                          const AttentionWeight<T>*                     attention_weights)
{
    // input_tensors:
    //      input_query [token_num, hidden_dimension]
    //      attention_mask [batch_size, 1, seq_len, seq_len + max_prompt_length]
    //      is_final_layer [1], bool on cpu
    //      d_prefix_prompt_batch [global_batch_size],
    //          each element contains ptr with buffer shape[2, local_head_num_, prompt_length, size_per_head]
    //      d_prefix_prompt_lengths [batch_size], int
    //      layer_id [1], int on cpu
    //      padding_offset, int, [token_num]

    // output_tensors:
    //      attention_out [token_num, hidden_dimension]
    //      key_cache [batch, local_head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [batch, local_head_num, max_seq_len, size_per_head]

    FT_CHECK(input_tensors->size() >= 6);
    FT_CHECK(output_tensors->size() == 3);
    FT_CHECK(output_tensors->at(1).shape.size() == 5);
    FT_CHECK(output_tensors->at(2).shape.size() == 4 || output_tensors->at(2).shape.size() == 3);
    const int  request_batch_size      = input_tensors->at(1).shape[0];
    const int  request_seq_len         = input_tensors->at(1).shape[2];
    const int  max_prompt_length       = input_tensors->at(1).shape[3] - input_tensors->at(1).shape[2];
    const int  layer_id                = *(int*)input_tensors->at(5).data;
    const T**  d_prefix_prompt_batch   = (const T**)input_tensors->at(3).data;
    const int* d_prefix_prompt_lengths = (const int*)input_tensors->at(4).data;
    const int* padding_offset          = input_tensors->size() == 7 ? input_tensors->at(6).getPtr<int>() : nullptr;

    allocateBuffer(request_batch_size, request_seq_len + max_prompt_length);
    sync_check_cuda_error();

    T*         attention_out   = (T*)output_tensors->at(0).data;
    const T*   attention_input = (const T*)input_tensors->at(0).data;
    const T*   attention_mask  = (const T*)input_tensors->at(1).data;
    const bool is_final        = *((bool*)(input_tensors->at(2).data));

    const int m = input_tensors->at(0).shape[0];

#ifdef SPARSITY_ENABLED
    const int m_padded = 8 * div_up(m, 8);
    if (sparse_ && cublas_wrapper_->isUseSparse(1, 3 * local_hidden_units_, m_padded, hidden_units_)) {
        cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                3 * local_hidden_units_,
                                m_padded,
                                hidden_units_,
                                attention_weights->query_weight.sp_kernel,
                                attention_input,
                                qkv_buf_);
    }
    else {
#endif
        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              3 * local_hidden_units_,  // n
                              m,
                              hidden_units_,  // k
                              attention_weights->query_weight.kernel,
                              3 * local_hidden_units_,  // n
                              attention_input,
                              hidden_units_,  // k
                              qkv_buf_,
                              3 * local_hidden_units_ /* n */);
#ifdef SPARSITY_ENABLED
    }
#endif

    sync_check_cuda_error();

    // IDEA: append prefix prompt key value here
    PrefixPromptBatchWeightsParam<T> param{d_prefix_prompt_batch,
                                           d_prefix_prompt_lengths,
                                           max_prompt_length,
                                           (size_t)layer_id * 2 * local_head_num_ * size_per_head_};

    if (padding_offset != nullptr) {
        // q_buf_2_, k_buf_2_ and v_buf_2_ are continuous
        cudaMemsetAsync(
            q_buf_2_, 0, request_batch_size * request_seq_len * 3 * local_hidden_units_ * sizeof(T), stream_);
    }
    invokeAddFusedQKVBiasTranspose(q_buf_2_,
                                   k_buf_2_,
                                   v_buf_2_,
                                   param,  // prefix prompt
                                   qkv_buf_,
                                   attention_weights->query_weight.bias,
                                   padding_offset,
                                   request_batch_size,
                                   request_seq_len,
                                   m,
                                   local_head_num_,
                                   size_per_head_,
                                   rotary_embedding_dim_,
                                   neox_rotary_style_,
                                   stream_);
    sync_check_cuda_error();

    const int max_seq_len = (int)(output_tensors->at(1).shape[3]);  // max output seq length
    // Use batch major
    // put k/v_buf from shape [B, H, PL + L, Dh]
    // to cache [B, H, Dh/x, PL + L, x]  and [B, H, PL + L, Dh/x, x], PL denotes prompt length
    invokeTranspose4dBatchMajor((T*)output_tensors->at(1).data,
                                (T*)output_tensors->at(2).data,
                                k_buf_2_,
                                v_buf_2_,
                                request_batch_size,
                                max_prompt_length + request_seq_len,  // max input length + prefix prompt length
                                max_seq_len,
                                size_per_head_,
                                local_head_num_,
                                stream_);
    // IDEA : after this, k_cache = (batch_size, num_heads, Dh/x, prefix_prompt_len + L, x)
    // k_cache = (batch_size, num_heads, prefix_prompt_len + L, Dh)
    sync_check_cuda_error();

    // NOTE: qkv buffer shape (batch_size, num_heads,L or prompt_len + L, Dh)

    if (is_final == false) {
        const cudaDataType_t gemm_data_type      = getCudaDataType<T>();
        const int            attention_seq_len_1 = request_seq_len;                      // q length
        const int            attention_seq_len_2 = max_prompt_length + request_seq_len;  // kv length
        if (is_qk_buf_float_ == true && gemm_data_type != CUDA_R_32F) {
            cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                                CUBLAS_OP_N,
                                                attention_seq_len_2,  // n
                                                attention_seq_len_1,  // m
                                                size_per_head_,       // k
                                                1.0f,
                                                k_buf_2_,
                                                gemm_data_type,
                                                size_per_head_,                        // k
                                                attention_seq_len_2 * size_per_head_,  // n * k
                                                q_buf_2_,
                                                gemm_data_type,
                                                size_per_head_,                        // k
                                                attention_seq_len_1 * size_per_head_,  // m * k
                                                0.0f,
                                                qk_buf_float_,
                                                CUDA_R_32F,
                                                attention_seq_len_2,  // n
                                                attention_seq_len_2 * attention_seq_len_1,
                                                request_batch_size * local_head_num_,  // global batch size
                                                CUDA_R_32F);

            sync_check_cuda_error();
            T scalar = 1 / sqrtf(size_per_head_ * 1.0f);
            invokeMaskedSoftMax(qk_buf_,
                                qk_buf_float_,
                                attention_mask,
                                request_batch_size,
                                attention_seq_len_1,  // seq_len_1
                                attention_seq_len_2,  // seq_len_2
                                local_head_num_,
                                scalar,
                                stream_);
            sync_check_cuda_error();
        }
        else {
            cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                                CUBLAS_OP_N,
                                                attention_seq_len_2,
                                                attention_seq_len_1,
                                                size_per_head_,
                                                k_buf_2_,
                                                size_per_head_,
                                                attention_seq_len_2 * size_per_head_,
                                                q_buf_2_,
                                                size_per_head_,
                                                attention_seq_len_1 * size_per_head_,
                                                qk_buf_,
                                                attention_seq_len_2,
                                                attention_seq_len_2 * attention_seq_len_1,
                                                request_batch_size * local_head_num_);

            T scalar = 1 / sqrtf(size_per_head_ * 1.0f);
            invokeMaskedSoftMax(qk_buf_,
                                qk_buf_,
                                attention_mask,
                                request_batch_size,
                                attention_seq_len_1,
                                attention_seq_len_2,
                                local_head_num_,
                                scalar,
                                stream_);
            sync_check_cuda_error();
        }

        cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
                                            CUBLAS_OP_N,
                                            size_per_head_,
                                            attention_seq_len_1,
                                            attention_seq_len_2,
                                            v_buf_2_,
                                            size_per_head_,
                                            attention_seq_len_2 * size_per_head_,
                                            qk_buf_,
                                            attention_seq_len_2,
                                            attention_seq_len_1 * attention_seq_len_2,
                                            qkv_buf_2_,
                                            size_per_head_,
                                            attention_seq_len_1 * size_per_head_,
                                            request_batch_size * local_head_num_);

        // transpose (batch_size, num_heads, L, Dh) to (batch_size, L, num_heads * Dh)
        invokeTransposeQKV(
            qkv_buf_3_, qkv_buf_2_, request_batch_size, attention_seq_len_1, local_head_num_, size_per_head_, stream_);
        if (padding_offset == nullptr) {
            invokeTransposeQKV(qkv_buf_3_,
                               qkv_buf_2_,
                               request_batch_size,
                               attention_seq_len_1,
                               local_head_num_,
                               size_per_head_,
                               stream_);
            sync_check_cuda_error();
        }
        else {
            invokeTransposeAttentionOutRemovePadding(qkv_buf_2_,
                                                     qkv_buf_3_,
                                                     m,
                                                     request_batch_size,
                                                     attention_seq_len_1,
                                                     local_head_num_,
                                                     size_per_head_,
                                                     padding_offset,
                                                     stream_);
        }
        sync_check_cuda_error();

#ifdef SPARSITY_ENABLED
        if (sparse_ && cublas_wrapper_->isUseSparse(1, hidden_units_, m_padded, local_hidden_units_)) {
            cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    hidden_units_,
                                    m_padded,
                                    local_hidden_units_,
                                    attention_weights->attention_output_weight.sp_kernel,
                                    qkv_buf_3_,
                                    attention_out);
        }
        else {
#endif
            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  hidden_units_,
                                  m,
                                  local_hidden_units_,
                                  attention_weights->attention_output_weight.kernel,
                                  hidden_units_,
                                  qkv_buf_3_,
                                  local_hidden_units_,
                                  attention_out,
                                  hidden_units_);

#ifdef SPARSITY_ENABLED
        }
#endif
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template<typename T>
GptContextAttentionLayer<T>::GptContextAttentionLayer(size_t           max_batch_size,
                                                      size_t           max_seq_len,
                                                      size_t           head_num,
                                                      size_t           size_per_head,
                                                      cudaStream_t     stream,
                                                      cublasMMWrapper* cublas_wrapper,
                                                      IAllocator*      allocator,
                                                      bool             is_free_buffer_after_forward,
                                                      bool             is_qk_buf_float,
                                                      bool             sparse):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    local_head_num_(head_num),
    local_hidden_units_(local_head_num_ * size_per_head),
    rotary_embedding_dim_(0),
    neox_rotary_style_(false),
    is_qk_buf_float_(is_qk_buf_float)
{
}

template<typename T>
GptContextAttentionLayer<T>::GptContextAttentionLayer(size_t           max_batch_size,
                                                      size_t           max_seq_len,
                                                      size_t           head_num,
                                                      size_t           size_per_head,
                                                      size_t           local_head_num,
                                                      cudaStream_t     stream,
                                                      cublasMMWrapper* cublas_wrapper,
                                                      IAllocator*      allocator,
                                                      bool             is_free_buffer_after_forward,
                                                      bool             is_qk_buf_float,
                                                      bool             sparse):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    local_head_num_(local_head_num),
    local_hidden_units_(local_head_num_ * size_per_head),
    rotary_embedding_dim_(0),
    neox_rotary_style_(false),
    is_qk_buf_float_(is_qk_buf_float)
{
}

template<typename T>
GptContextAttentionLayer<T>::GptContextAttentionLayer(size_t           max_batch_size,
                                                      size_t           max_seq_len,
                                                      size_t           head_num,
                                                      size_t           size_per_head,
                                                      size_t           local_head_num,
                                                      size_t           rotary_embedding_dim,
                                                      bool             neox_rotary_style,
                                                      cudaStream_t     stream,
                                                      cublasMMWrapper* cublas_wrapper,
                                                      IAllocator*      allocator,
                                                      bool             is_free_buffer_after_forward,
                                                      bool             is_qk_buf_float,
                                                      bool             sparse):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    local_head_num_(local_head_num),
    local_hidden_units_(local_head_num_ * size_per_head),
    rotary_embedding_dim_(rotary_embedding_dim),
    neox_rotary_style_(neox_rotary_style),
    is_qk_buf_float_(is_qk_buf_float)
{
}

template<typename T>
GptContextAttentionLayer<T>::GptContextAttentionLayer(GptContextAttentionLayer<T> const& attention_layer):
    BaseAttentionLayer<T>(attention_layer.stream_,
                          attention_layer.cublas_wrapper_,
                          attention_layer.allocator_,
                          attention_layer.is_free_buffer_after_forward_,
                          attention_layer.sparse_),
    max_batch_size_(attention_layer.max_batch_size_),
    max_seq_len_(attention_layer.max_seq_len_),
    head_num_(attention_layer.head_num_),
    size_per_head_(attention_layer.size_per_head_),
    hidden_units_(attention_layer.hidden_units_),
    local_head_num_(attention_layer.local_head_num_),
    local_hidden_units_(attention_layer.local_hidden_units_),
    rotary_embedding_dim_(attention_layer.rotary_embedding_dim_),
    neox_rotary_style_(attention_layer.neox_rotary_style_),
    is_qk_buf_float_(attention_layer.is_qk_buf_float_)
{
}

template<typename T>
GptContextAttentionLayer<T>::~GptContextAttentionLayer()
{
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void GptContextAttentionLayer<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void GptContextAttentionLayer<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    qkv_buf_ = (T*)allocator_->reMalloc(qkv_buf_, sizeof(T) * 3 * batch_size * seq_len * local_hidden_units_, true);
    q_buf_2_ = (T*)allocator_->reMalloc(q_buf_2_, sizeof(T) * batch_size * seq_len * 3 * local_hidden_units_, true);
    k_buf_2_ = q_buf_2_ + batch_size * seq_len * local_hidden_units_;
    v_buf_2_ = k_buf_2_ + batch_size * seq_len * local_hidden_units_;

    qk_buf_    = (T*)allocator_->reMalloc(qk_buf_, sizeof(T) * batch_size * local_head_num_ * seq_len * seq_len, true);
    qkv_buf_2_ = (T*)allocator_->reMalloc(qkv_buf_2_, sizeof(T) * batch_size * seq_len * local_hidden_units_, true);
    qkv_buf_3_ = (T*)allocator_->reMalloc(qkv_buf_3_, sizeof(T) * batch_size * seq_len * local_hidden_units_, true);

    if (is_qk_buf_float_ == true) {
        qk_buf_float_ = (float*)allocator_->reMalloc(
            qk_buf_float_, sizeof(float) * batch_size * local_head_num_ * seq_len * seq_len, true);
    }
    is_allocate_buffer_ = true;
}

template<typename T>
void GptContextAttentionLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        allocator_->free((void**)(&qkv_buf_));
        allocator_->free((void**)(&q_buf_2_));
        allocator_->free((void**)(&qk_buf_));
        allocator_->free((void**)(&qkv_buf_2_));
        allocator_->free((void**)(&qkv_buf_3_));

        if (is_qk_buf_float_ == true) {
            allocator_->free((void**)(&qk_buf_float_));
        }
        is_allocate_buffer_ = false;
    }
}

template class GptContextAttentionLayer<float>;
template class GptContextAttentionLayer<half>;
#ifdef ENABLE_BF16
template class GptContextAttentionLayer<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
