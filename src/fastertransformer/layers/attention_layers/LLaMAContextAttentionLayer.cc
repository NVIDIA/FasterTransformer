/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/layers/attention_layers/LLaMAContextAttentionLayer.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"
#include "src/fastertransformer/utils/llama_utils.h"
#include "src/fastertransformer/utils/nvtx_utils.h"

namespace fastertransformer {

template<typename T>
void LLaMAContextAttentionLayer<T>::forward(TensorMap*                output_tensors,
                                            TensorMap*                input_tensors,
                                            const AttentionWeight<T>* attention_weights)
{
    // input_tensors:
    //      input_query [token_num, hidden_dimension]
    //      attention_mask [batch_size, 1, seq_len, max_length]
    //      attention_type [1]
    //      layer_id [1], int on cpu
    //      context_lengths, int, [batch_size]
    //      max_length, int, [batch_size] on cpu
    //      padding_offset, int, [token_num] (optional)
    //      cu_seqlens, int, [batch_size] (optional)

    // output_tensors:
    //      hidden_features [token_num, hidden_dimension]
    //      key_cache [batch, local_head_num, max_seq_len, size_per_head]
    //      value_cache [batch, local_head_num, max_seq_len, size_per_head]

    FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    FT_CHECK(output_tensors->at("key_cache").shape.size() == 4);
    FT_CHECK(output_tensors->at("value_cache").shape.size() == 4);
    const int  batch_size      = input_tensors->at("attention_mask").shape[0];
    const int  seq_len         = input_tensors->at("attention_mask").shape[2];
    const int  max_seq_len     = (int)(output_tensors->at("key_cache").shape[2]);
    const int  layer_id        = input_tensors->getVal<int>("layer_id");
    const int* padding_offset  = input_tensors->getPtr<int>("padding_offset", nullptr);
    const int* cu_seqlens      = input_tensors->getPtr<int>("cu_seqlens", nullptr);
    const int* context_lengths = input_tensors->at("context_lengths").getPtr<int>();
    const int  max_length      = input_tensors->at("max_length").getVal<int>();

    T* attention_out   = output_tensors->at("hidden_features").getPtr<T>();
    T* attention_input = input_tensors->at("input_query").getPtr<T>();
    T* attention_mask  = input_tensors->at("attention_mask").getPtr<T>();

    const AttentionType attention_type = input_tensors->getVal<AttentionType>("attention_type");
    FT_CHECK_WITH_INFO(attention_type != AttentionType::FUSED_PADDED_MHA,
                       "LLaMA Context FUSED_PADDED_MHA is not supported !");

    PUSH_RANGE("attention buffer alloc");
    allocateBuffer(batch_size, seq_len, max_seq_len, attention_type != AttentionType::FUSED_MHA);
    POP_RANGE;
    sync_check_cuda_error();

    const int m = input_tensors->at("input_query").shape[0];

    PUSH_RANGE("qkv_gemm");

    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          3 * hidden_units_,  // n
                          m,
                          hidden_units_,  // k
                          attention_weights->query_weight.kernel,
                          3 * hidden_units_,  // n
                          attention_input,
                          hidden_units_,  // k
                          qkv_buf_,
                          3 * hidden_units_ /* n */);
    sync_check_cuda_error();

    if (padding_offset != nullptr) {
        // q_buf_2_, k_buf_2_ and v_buf_2_ are continuous
        cudaMemsetAsync(q_buf_2_, 0, batch_size * max_seq_len * 3 * hidden_units_ * sizeof(T), stream_);
        sync_check_cuda_error();
    }
    invokeLLaMAAddFusedQKVBiasTranspose(q_buf_2_,
                                        k_buf_2_,
                                        v_buf_2_,
                                        qkv_buf_,
                                        padding_offset,
                                        batch_size,
                                        seq_len,
                                        m,
                                        head_num_,
                                        size_per_head_,
                                        rotary_embedding_dim_,
                                        context_lengths,
                                        stream_);
    sync_check_cuda_error();

    // key_cache [batch, local_head_num, max_seq_len, size_per_head]
    // value_cache [batch, local_head_num, max_seq_len, size_per_head]
    T* key_cache   = output_tensors->getPtr<T>("key_cache");
    T* value_cache = output_tensors->getPtr<T>("value_cache");
    invokeLLaMASaveToCache(key_cache,
                           value_cache,
                           k_buf_2_,
                           v_buf_2_,
                           batch_size,
                           seq_len,
                           max_seq_len,
                           size_per_head_,
                           head_num_,
                           context_lengths,
                           stream_);
    sync_check_cuda_error();
    POP_RANGE;

    invokeLLaMALoadFromCache(k_buf_2_,
                             v_buf_2_,
                             key_cache,
                             value_cache,
                             batch_size,
                             seq_len,
                             max_seq_len,
                             size_per_head_,
                             head_num_,
                             max_length,
                             stream_);

    if (attention_type == AttentionType::FUSED_MHA) {
        dispatcher_fp16->setup_causal_masked_fmha(seq_len, batch_size);
        dispatcher_fp16->run_causal_masked_fmha(qkv_buf_, cu_seqlens, qkv_buf_3_, true, stream_);
    }
    else {
        const cudaDataType_t gemm_data_type      = getCudaDataType<T>();
        const int            attention_seq_len_1 = seq_len;     // q length
        const int            attention_seq_len_2 = max_length;  // kv length
        const T              qk_scale            = static_cast<T>(1.0f / sqrtf(size_per_head_ * 1.0f));

        //
        // softmax(Q*K^T)
        //
        if (is_qk_buf_float_ == true && gemm_data_type != CUDA_R_32F) {
            PUSH_RANGE("Q*K batch gemm");

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
                                                batch_size * head_num_,  // global batch size
                                                CUDA_R_32F);
            sync_check_cuda_error();
            POP_RANGE;


            PUSH_RANGE("softmax");
            MaskedSoftmaxParam<T, float> param;
            param.attention_score    = qk_buf_;         // (batch_size, head_num, q_length, k_length)
            param.qk                 = qk_buf_float_;   // (batch_size, head_num, q_length, k_length)
            param.attention_mask     = attention_mask;  // (batch_size, q_length, k_length)
            param.batch_size         = batch_size;
            param.q_length           = attention_seq_len_1;
            param.k_length           = attention_seq_len_2;
            param.num_heads          = head_num_;
            param.qk_scale           = qk_scale;
            param.linear_bias_slopes = nullptr;
            invokeMaskedSoftmax(param, stream_);
            sync_check_cuda_error();
            POP_RANGE;
        }
        else {
            PUSH_RANGE("Q*K batch gemm");
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
                                                batch_size * head_num_);

            POP_RANGE;
            PUSH_RANGE("softmax");
            MaskedSoftmaxParam<T, T> param;
            param.attention_score    = qk_buf_;         // (batch_size, head_num, q_length, k_length)
            param.qk                 = qk_buf_;         // (batch_size, head_num, q_length, k_length)
            param.attention_mask     = attention_mask;  // (batch_size, q_length, k_length)
            param.batch_size         = batch_size;
            param.q_length           = attention_seq_len_1;
            param.k_length           = attention_seq_len_2;
            param.num_heads          = head_num_;
            param.qk_scale           = qk_scale;
            param.linear_bias_slopes = nullptr;
            invokeMaskedSoftmax(param, stream_);
            sync_check_cuda_error();
            POP_RANGE;
        }

        PUSH_RANGE("QK*V batch gemm");
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

                                            batch_size * head_num_);
        sync_check_cuda_error();

        // transpose (batch_size, num_heads, L, Dh) to (batch_size, L, num_heads * Dh)
        if (padding_offset == nullptr) {
            invokeTransposeQKV(qkv_buf_3_,
                               qkv_buf_2_,
                               batch_size,
                               attention_seq_len_1,
                               head_num_,
                               size_per_head_,
                               attention_weights->attention_output_weight.scale,
                               0,  // int8_mode
                               stream_);
            sync_check_cuda_error();
        }
        else {
            invokeTransposeAttentionOutRemovePadding(qkv_buf_2_,
                                                     qkv_buf_3_,
                                                     m,
                                                     batch_size,
                                                     attention_seq_len_1,
                                                     head_num_,
                                                     size_per_head_,
                                                     padding_offset,
                                                     attention_weights->attention_output_weight.scale,
                                                     0,  // int8_mode
                                                     stream_);
            sync_check_cuda_error();
        }
        POP_RANGE;
    }
    sync_check_cuda_error();

    PUSH_RANGE("proj gemm");
    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          hidden_units_,
                          m,
                          hidden_units_,
                          attention_weights->attention_output_weight.kernel,
                          hidden_units_,
                          qkv_buf_3_,
                          hidden_units_,
                          attention_out,
                          hidden_units_);
    sync_check_cuda_error();
    POP_RANGE;

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
LLaMAContextAttentionLayer<T>::LLaMAContextAttentionLayer(size_t           head_num,
                                                          size_t           size_per_head,
                                                          cudaStream_t     stream,
                                                          cublasMMWrapper* cublas_wrapper,
                                                          IAllocator*      allocator,
                                                          bool             is_free_buffer_after_forward,
                                                          bool             is_qk_buf_float):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, false),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    rotary_embedding_dim_(0),
    is_qk_buf_float_(is_qk_buf_float)
{
}

template<typename T>
LLaMAContextAttentionLayer<T>::LLaMAContextAttentionLayer(size_t           head_num,
                                                          size_t           size_per_head,
                                                          size_t           local_head_num,
                                                          cudaStream_t     stream,
                                                          cublasMMWrapper* cublas_wrapper,
                                                          IAllocator*      allocator,
                                                          bool             is_free_buffer_after_forward,
                                                          bool             is_qk_buf_float):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, false),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    rotary_embedding_dim_(0),
    is_qk_buf_float_(is_qk_buf_float)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    dispatcher_fp16.reset(new FusedMHARunnerFP16v2(head_num_, size_per_head_, sm_, 1.0f));
}

template<typename T>
LLaMAContextAttentionLayer<T>::LLaMAContextAttentionLayer(size_t           head_num,
                                                          size_t           size_per_head,
                                                          size_t           local_head_num,
                                                          size_t           rotary_embedding_dim,
                                                          cudaStream_t     stream,
                                                          cublasMMWrapper* cublas_wrapper,
                                                          IAllocator*      allocator,
                                                          bool             is_free_buffer_after_forward,
                                                          bool             is_qk_buf_float):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, false),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    rotary_embedding_dim_(rotary_embedding_dim),
    is_qk_buf_float_(is_qk_buf_float)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    dispatcher_fp16.reset(new FusedMHARunnerFP16v2(head_num_, size_per_head_, sm_, 1.0f));
}

template<typename T>
LLaMAContextAttentionLayer<T>::LLaMAContextAttentionLayer(LLaMAContextAttentionLayer<T> const& attention_layer):
    BaseAttentionLayer<T>(attention_layer.stream_,
                          attention_layer.cublas_wrapper_,
                          attention_layer.allocator_,
                          attention_layer.is_free_buffer_after_forward_),
    head_num_(attention_layer.head_num_),
    size_per_head_(attention_layer.size_per_head_),
    hidden_units_(attention_layer.hidden_units_),
    rotary_embedding_dim_(attention_layer.rotary_embedding_dim_),
    is_qk_buf_float_(attention_layer.is_qk_buf_float_)
{
}

template<typename T>
LLaMAContextAttentionLayer<T>::~LLaMAContextAttentionLayer()
{
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void LLaMAContextAttentionLayer<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void LLaMAContextAttentionLayer<T>::allocateBuffer(size_t batch_size,
                                                   size_t seq_len,
                                                   size_t max_seq_len,
                                                   bool   allocate_qk_buf)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    qkv_buf_ = (T*)allocator_->reMalloc(qkv_buf_, sizeof(T) * 3 * batch_size * seq_len * hidden_units_, false);
    q_buf_2_ = (T*)allocator_->reMalloc(q_buf_2_, sizeof(T) * batch_size * max_seq_len * 3 * hidden_units_, false);
    k_buf_2_ = q_buf_2_ + batch_size * max_seq_len * hidden_units_;
    v_buf_2_ = k_buf_2_ + batch_size * max_seq_len * hidden_units_;

    // save memory usage when using fmha
    if (allocate_qk_buf) {
        qk_buf_ =
            (T*)allocator_->reMalloc(qk_buf_, sizeof(T) * batch_size * head_num_ * max_seq_len * max_seq_len, false);
    }
    else {
        allocator_->free((void**)(&qk_buf_));
        qk_buf_ = nullptr;
    }
    qkv_buf_2_ = (T*)allocator_->reMalloc(qkv_buf_2_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    qkv_buf_3_ = (T*)allocator_->reMalloc(qkv_buf_3_, sizeof(T) * batch_size * seq_len * hidden_units_, false);

    if (is_qk_buf_float_ == true) {
        if (allocate_qk_buf) {
            qk_buf_float_ = (float*)allocator_->reMalloc(
                qk_buf_float_, sizeof(float) * batch_size * head_num_ * max_seq_len * max_seq_len, false);
        }
        else {
            allocator_->free((void**)(&qk_buf_float_));
            qk_buf_float_ = nullptr;
        }
    }

    is_allocate_buffer_ = true;
}

template<typename T>
void LLaMAContextAttentionLayer<T>::freeBuffer()
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

template class LLaMAContextAttentionLayer<float>;
template class LLaMAContextAttentionLayer<half>;
#ifdef ENABLE_BF16
template class LLaMAContextAttentionLayer<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
