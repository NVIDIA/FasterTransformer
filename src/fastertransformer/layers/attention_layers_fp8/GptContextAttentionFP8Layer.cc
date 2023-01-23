/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/layers/attention_layers_fp8/GptContextAttentionFP8Layer.h"
#include "src/fastertransformer/kernels/unfused_attention_fp8_kernels.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"

namespace fastertransformer {

template<typename T1, typename T2>
void GptContextAttentionFP8Layer<T1, T2>::forward(TensorMap*                 output_tensors,
                                                  TensorMap*                 input_tensors,
                                                  const AttentionWeight<T1>* attention_weights)
{
    // input_tensors:
    //      input_query [batch_size * seq_len, hidden_dimension]
    //      attention_mask [batch_size, 1, seq_len_padded, seq_len_padded]
    //      is_final_layer [1], bool on cpu

    // output_tensors:
    //      attention_out [batch_size * seq_len, hidden_dimension]
    //      key_cache [batch, local_head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [batch, local_head_num, max_seq_len, size_per_head]

    const AttentionFP8Weight<T1, T2>* attention_weights_ptr =
        reinterpret_cast<const AttentionFP8Weight<T1, T2>*>(attention_weights);
    FT_CHECK(input_tensors->size() == 3);
    FT_CHECK(output_tensors->size() == 3);
    FT_CHECK(output_tensors->at("key_cache").shape.size() == 5);
    FT_CHECK(output_tensors->at("value_cache").shape.size() == 4
             || output_tensors->at("value_cache").shape.size() == 3);
    const int request_batch_size = (int)(input_tensors->at("attention_mask").shape[0]);
    const int request_seq_len    = (int)(input_tensors->at("attention_mask").shape[2]);
    const int max_seq_len        = (int)(output_tensors->at("key_cache").shape[3]);
    allocateBuffer(request_batch_size, max_seq_len);
    sync_check_cuda_error();

    const int request_seq_len_padded = (request_seq_len + 15) / 16 * 16;

    T2*        attention_out   = output_tensors->at("attention_out").getPtr<T2>();
    const T1*  attention_input = input_tensors->at("input_query").getPtr<T1>();
    const T1*  attention_mask  = input_tensors->at("attention_mask").getPtr<T1>();
    const bool is_final        = input_tensors->at("is_final_layer").getVal<bool>();

    const int m = input_tensors->at("input_query").shape[0];

#ifdef SPARSITY_ENABLED
    FT_CHECK_WITH_INFO(false, "DecoderSelfAttentionFP8Layer does not support sparse now.");
    const int m_padded = 8 * div_up(m, 8);
    if (sparse_ && cublas_wrapper_->isUseSparse(1, 3 * local_hidden_units_, m_padded, hidden_units_)) {
        //     cublas_wrapper_->SpGemm(CUBLAS_OP_N,
        //                             CUBLAS_OP_N,
        //                             3 * local_hidden_units_,
        //                             m_padded,
        //                             hidden_units_,
        //                             attention_weights_ptr->query_weight.sp_kernel,
        //                             attention_input,
        //                             qkv_buf_);
    }
    else {
#endif
        // reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_)
        //     ->Gemm(qkv_buf_,
        //            (int)1,
        //            (int)m,
        //            (int)3 * local_hidden_units_,
        //            (int)hidden_units_,
        //            (int64_t)0,
        //            (int64_t)0,
        //            (int64_t)0,
        //            attention_weights_ptr->query_weight.input_scale,
        //            attention_input,
        //            attention_weights_ptr->query_weight.kernel);

        {
            const float alpha = 1.0f;
            const float beta  = 0.0f;
            reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_)
                ->Gemm(qkv_buf_,
                       (int)1,
                       (int)m,
                       (int)3 * local_hidden_units_,
                       (int)hidden_units_,
                       (int64_t)0,
                       (int64_t)0,
                       (int64_t)0,
                       &alpha,
                       &beta,
                       attention_input,
                       attention_weights_ptr->query_weight.kernel,
                       attention_weights_ptr->query_weight.input_scale,
                       attention_weights_ptr->query_weight.weight_scale,
#ifndef FP8_GEMM_OUTPUT_QUANT_DISABLE
                       attention_weights_ptr->query_weight.output_scale_inv,
#endif
                       stream_);
        }

#ifdef SPARSITY_ENABLED
    }
#endif
    sync_check_cuda_error();

    {
        FP8AddFusedQKVBiasRebuildPaddingParam<T1, T2> param{
            q_buf_2_,
            k_buf_2_,
            v_buf_2_,
#ifdef FP8_GEMM_OUTPUT_QUANT_DISABLE
            nullptr,   // T1
            qkv_buf_,  // T2
#else
            qkv_buf_,  // T1
            nullptr,   // T2
#endif
            attention_weights_ptr->query_weight.bias,
#ifdef FP8_GEMM_OUTPUT_QUANT_DISABLE
            attention_weights_ptr->identity_scale,
#else
            attention_weights_ptr->query_weight.output_scale,
#endif
            nullptr,  // fp8_mode_ == 1 ? attention_weights_ptr->query_weight.scale : nullptr,
            nullptr,  // fp8_mode_ == 1 ? attention_weights_ptr->query_weight.per_channel_scale_min : nullptr,
            attention_weights_ptr->query_weight.output_scale_inv,
            nullptr,
            nullptr,  // padding_offset
#ifdef FP8_MHA
            output_tensors->at("value_cache").getPtr<T1>(),
#else
            output_tensors->at("value_cache").getPtr<T2>(),
#endif
            (uint32_t)m,
            (uint32_t)request_batch_size,
            (uint32_t)request_seq_len,
            (uint32_t)request_seq_len_padded,
            (uint32_t)max_seq_len,
            (uint32_t)local_head_num_,
            (uint32_t)size_per_head_,
            0,  // rotary_embedding_dim_,
            stream_};
        invokeFP8AddFusedQKVBiasRebuildPadding(param);
        sync_check_cuda_error();
    }

    {
        const int max_seq_len = (int)(output_tensors->at("key_cache").shape[3]);
        // Use batch major
        // put k_buf from shape [B, H, L, Dh] to [B, H, Dh/x, L, x]
        // put v_buf from shape [B, H, Dh, L_padd] to [B, H, L_max, Dh/x, x]

#ifdef FP8_MHA
        // invokeMultiplyScale(k_buf_2_,
        //                   (1.0f / K_CACHE_SCALE),
        //                   request_batch_size * request_seq_len * size_per_head_ * local_head_num_,
        //                   stream_);
        FP8Transpose4dBatchMajorParam<T1, T1> param{output_tensors->at("key_cache").getPtr<T1>(),
                                                    output_tensors->at("value_cache").getPtr<T1>(),
                                                    k_buf_2_,
                                                    v_buf_2_,
                                                    (const float*)nullptr,
                                                    request_batch_size,
                                                    request_seq_len,
                                                    max_seq_len,
                                                    size_per_head_,
                                                    local_head_num_,
                                                    request_seq_len_padded,
                                                    stream_};
        invokeFP8Transpose4dBatchMajor(param);
#else
        // {
        FP8Transpose4dBatchMajorParam<T1, T2> param{output_tensors->at("key_cache").getPtr<T2>(),
                                                    output_tensors->at("value_cache").getPtr<T2>(),
                                                    k_buf_2_,
                                                    v_buf_2_,
                                                    attention_weights_ptr->query_weight.output_scale,
                                                    request_batch_size,
                                                    request_seq_len,
                                                    max_seq_len,
                                                    size_per_head_,
                                                    local_head_num_,
                                                    request_seq_len_padded,
                                                    stream_};
        invokeFP8Transpose4dBatchMajor(param);
        // TODO(bhsueh) fuse these two kernels into invokeTranspose4dBatchMajor
        //     invokeTmpHanldKCache(tmp_k_buf_,
        //                          k_buf_2_,
        //                          attention_weights_ptr->query_weight.output_scale,
        //                          request_batch_size,
        //                          request_seq_len,
        //                          request_seq_len_padded,
        //                          local_head_num_,
        //                          size_per_head_,
        //                          stream_);

        //     invokeTmpHanldVCache(tmp_v_buf_,
        //                          v_buf_2_,
        //                          attention_weights_ptr->query_weight.output_scale,
        //                          request_batch_size,
        //                          request_seq_len,
        //                          request_seq_len_padded,
        //                          local_head_num_,
        //                          size_per_head_,
        //                          stream_);

        //     invokeTranspose4dBatchMajor((T2*)output_tensors->at("key_cache").data,
        //                                 (T2*)output_tensors->at("value_cache").data,
        //                                 tmp_k_buf_,
        //                                 tmp_v_buf_,
        //                                 request_batch_size,
        //                                 request_seq_len,
        //                                 max_seq_len,
        //                                 size_per_head_,
        //                                 local_head_num_,
        //                                 stream_);
        //     sync_check_cuda_error();
        // }
#endif
    }

    if (is_final == false) {
        {
            const float alpha = 1.0f;
            const float beta  = 0.0f;

            reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_)
                ->Gemm(qk_buf_bfloat_,
                       (int)request_batch_size * local_head_num_,
                       (int)request_seq_len_padded,
                       (int)request_seq_len_padded,
                       (int)size_per_head_,
                       (int64_t)request_seq_len_padded * size_per_head_,
                       (int64_t)request_seq_len_padded * size_per_head_,
                       (int64_t)request_seq_len_padded * request_seq_len_padded,
                       &alpha,
                       &beta,
                       q_buf_2_,
                       k_buf_2_,
                       attention_weights_ptr->query_weight.output_scale,
                       attention_weights_ptr->query_weight.output_scale,
                       stream_);
        }
        sync_check_cuda_error();

        float                         scalar = (float)(1.0f / sqrtf(size_per_head_ * 1.0f));
        FP8MaskedSoftMaxParam<T1, T2> param{qk_buf_,
                                            qk_buf_bfloat_,
                                            attention_mask,
                                            nullptr,
                                            (uint32_t)request_batch_size,
                                            (uint32_t)request_seq_len_padded,
                                            (uint32_t)local_head_num_,
                                            scalar,
                                            nullptr,
                                            attention_weights_ptr->qk_scale_inv,
                                            stream_};
        invokeFP8MaskedSoftMax(param);
        sync_check_cuda_error();

        {
            const float alpha = 1.0f;
            const float beta  = 0.0f;
            reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_)
                ->Gemm(qkv_buf_2_,
                       (int)request_batch_size * local_head_num_,
                       (int)request_seq_len_padded,
                       (int)size_per_head_,
                       (int)request_seq_len_padded,
                       (int64_t)size_per_head_ * request_seq_len_padded,
                       (int64_t)request_seq_len_padded * request_seq_len_padded,
                       (int64_t)size_per_head_ * request_seq_len_padded,
                       &alpha,
                       &beta,
                       qk_buf_,
                       v_buf_2_,
                       attention_weights_ptr->qk_scale,
                       attention_weights_ptr->query_weight.output_scale,
#ifndef FP8_GEMM_OUTPUT_QUANT_DISABLE
                       attention_weights_ptr->attention_output_weight.input_scale_inv,
#endif
                       stream_);
            sync_check_cuda_error();
        }

        {
            // NOTE: add last gemm's output scale here
#ifdef FP8_GEMM_OUTPUT_QUANT_DISABLE
            FP8TransposeAttentionOutRemovePaddingParam<T2, T1> param{
                qkv_buf_3_,
                qkv_buf_2_,
                attention_weights_ptr->attention_output_weight.input_scale_inv,
                m,
                request_batch_size,
                request_seq_len_padded,
                (int)local_head_num_,
                (int)size_per_head_,
                nullptr,  // padding_offset
                stream_};
#else
            FP8TransposeAttentionOutRemovePaddingParam<T1, T1> param{qkv_buf_3_,
                                                                     qkv_buf_2_,
                                                                     (const float*)nullptr,
                                                                     m,
                                                                     request_batch_size,
                                                                     request_seq_len_padded,
                                                                     (int)local_head_num_,
                                                                     (int)size_per_head_,
                                                                     nullptr,  // padding_offset
                                                                     stream_};
#endif
            invokeFP8TransposeAttentionOutRemovePadding(param);
        }

#ifdef SPARSITY_ENABLED
        FT_CHECK_WITH_INFO(false, "DecoderSelfAttentionFP8Layer does not support sparse now.");
        if (sparse_ && cublas_wrapper_->isUseSparse(1, hidden_units_, m_padded, local_hidden_units_)) {
            // cublas_wrapper_->SpGemm(CUBLAS_OP_N,
            //                         CUBLAS_OP_N,
            //                         hidden_units_,
            //                         m_padded,
            //                         local_hidden_units_,
            //                         attention_weights_ptr->attention_output_weight.sp_kernel,
            //                         qkv_buf_3_,
            //                         attention_out);
        }
        else {
#endif
            const float alpha = 1.0f;
            const float beta  = 0.0f;
            reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_)
                ->Gemm(attention_out,
                       (int)1,
                       (int)m,
                       (int)hidden_units_,
                       (int)local_hidden_units_,
                       (int64_t)0,
                       (int64_t)0,
                       (int64_t)0,
                       &alpha,
                       &beta,
                       qkv_buf_3_,
                       attention_weights_ptr->attention_output_weight.kernel,
                       attention_weights_ptr->attention_output_weight.input_scale,
                       attention_weights_ptr->attention_output_weight.weight_scale,
                       stream_);
            sync_check_cuda_error();
#ifdef SPARSITY_ENABLED
        }
#endif
    }

    if (is_free_buffer_after_forward_ == true)
        freeBuffer();
    sync_check_cuda_error();
}

template<typename T1, typename T2>
GptContextAttentionFP8Layer<T1, T2>::GptContextAttentionFP8Layer(size_t           head_num,
                                                                 size_t           size_per_head,
                                                                 size_t           local_head_num,
                                                                 size_t           rotary_embedding_dim,
                                                                 cudaStream_t     stream,
                                                                 cublasMMWrapper* cublas_wrapper,
                                                                 IAllocator*      allocator,
                                                                 bool             is_free_buffer_after_forward,
                                                                 bool             is_qk_buf_float,
                                                                 bool             sparse):
    BaseAttentionLayer<T1>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    local_head_num_(local_head_num),
    local_hidden_units_(local_head_num_ * size_per_head),
    rotary_embedding_dim_(rotary_embedding_dim),
    is_qk_buf_float_(is_qk_buf_float)
{
}

template<typename T1, typename T2>
GptContextAttentionFP8Layer<T1, T2>::GptContextAttentionFP8Layer(
    GptContextAttentionFP8Layer<T1, T2> const& attention_layer):
    BaseAttentionLayer<T1>(attention_layer.stream_,
                           attention_layer.cublas_wrapper_,
                           attention_layer.allocator_,
                           attention_layer.is_free_buffer_after_forward_,
                           attention_layer.sparse_),
    head_num_(attention_layer.head_num_),
    size_per_head_(attention_layer.size_per_head_),
    hidden_units_(attention_layer.hidden_units_),
    local_head_num_(attention_layer.local_head_num_),
    local_hidden_units_(attention_layer.local_hidden_units_),
    rotary_embedding_dim_(attention_layer.rotary_embedding_dim_),
    is_qk_buf_float_(attention_layer.is_qk_buf_float_)
{
}

template<typename T1, typename T2>
GptContextAttentionFP8Layer<T1, T2>::~GptContextAttentionFP8Layer()
{
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T1, typename T2>
void GptContextAttentionFP8Layer<T1, T2>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T1, typename T2>
void GptContextAttentionFP8Layer<T1, T2>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    size_t seq_len_padded = (seq_len + 15) / 16 * 16;
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
#ifdef FP8_GEMM_OUTPUT_QUANT_DISABLE
    qkv_buf_ =
        (T2*)allocator_->reMalloc(qkv_buf_, sizeof(T2) * 3 * batch_size * seq_len_padded * local_hidden_units_, true);
    qkv_buf_2_ =
        (T2*)allocator_->reMalloc(qkv_buf_2_, sizeof(T2) * batch_size * seq_len_padded * local_hidden_units_, true);
#else
    qkv_buf_ =
        (T1*)allocator_->reMalloc(qkv_buf_, sizeof(T1) * 3 * batch_size * seq_len_padded * local_hidden_units_, true);
    qkv_buf_2_ =
        (T1*)allocator_->reMalloc(qkv_buf_2_, sizeof(T1) * batch_size * seq_len_padded * local_hidden_units_, true);
#endif
    q_buf_2_ =
        (T1*)allocator_->reMalloc(q_buf_2_, sizeof(T1) * batch_size * seq_len_padded * local_hidden_units_, true);
    k_buf_2_ =
        (T1*)allocator_->reMalloc(k_buf_2_, sizeof(T1) * batch_size * seq_len_padded * local_hidden_units_, true);
    v_buf_2_ =
        (T1*)allocator_->reMalloc(v_buf_2_, sizeof(T1) * batch_size * seq_len_padded * local_hidden_units_, true);

    qk_buf_ = (T1*)allocator_->reMalloc(
        qk_buf_, sizeof(T1) * batch_size * local_head_num_ * seq_len_padded * seq_len_padded, true);
    qkv_buf_3_ = (T1*)allocator_->reMalloc(qkv_buf_3_, sizeof(T1) * batch_size * seq_len * local_hidden_units_, true);

    if (is_qk_buf_float_ == true) {
        // qk_buf_float_ = (float*)allocator_->reMalloc(
        //     qk_buf_float_, sizeof(float) * batch_size * local_head_num_ * seq_len * seq_len, true);
    }
    qk_buf_bfloat_ = (T2*)allocator_->reMalloc(
        qk_buf_bfloat_, sizeof(T2) * batch_size * local_head_num_ * seq_len_padded * seq_len_padded, true);
    tmp_k_buf_ =
        (T2*)allocator_->reMalloc(tmp_k_buf_, sizeof(T2) * batch_size * seq_len_padded * local_hidden_units_, true);
    tmp_v_buf_ =
        (T2*)allocator_->reMalloc(tmp_v_buf_, sizeof(T2) * batch_size * seq_len_padded * local_hidden_units_, true);
    is_allocate_buffer_ = true;
}

template<typename T1, typename T2>
void GptContextAttentionFP8Layer<T1, T2>::freeBuffer()
{
    if (is_allocate_buffer_) {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        allocator_->free((void**)(&qkv_buf_));
        allocator_->free((void**)(&q_buf_2_));
        allocator_->free((void**)(&k_buf_2_));
        allocator_->free((void**)(&v_buf_2_));
        allocator_->free((void**)(&qk_buf_));
        allocator_->free((void**)(&qkv_buf_2_));
        allocator_->free((void**)(&qkv_buf_3_));

        if (is_qk_buf_float_ == true) {
            // allocator_->free(qk_buf_float_);
        }
        allocator_->free((void**)(&qk_buf_bfloat_));
        allocator_->free((void**)(&tmp_k_buf_));
        allocator_->free((void**)(&tmp_v_buf_));
        is_allocate_buffer_ = false;
    }
}

template class GptContextAttentionFP8Layer<__nv_fp8_e4m3, __nv_bfloat16>;

}  // namespace fastertransformer
