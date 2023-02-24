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

#include "src/fastertransformer/layers/attention_layers_fp8/SelfAttentionFP8Layer.h"
#include "3rdparty/trt_fp8_fmha/fused_multihead_attention.h"
#include "src/fastertransformer/kernels/unfused_attention_fp8_kernels.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"
#include "src/fastertransformer/layers/attention_layers_fp8/AttentionFP8Weight.h"
#include "src/fastertransformer/utils/nvtx_utils.h"

namespace fastertransformer {

template<typename T1, typename T2>
void SelfAttentionFP8Layer<T1, T2>::forward(TensorMap*                        output_tensors,
                                            TensorMap*                        input_tensors,
                                            const AttentionFP8Weight<T1, T2>* attention_weights)
{
    // input_tensors:
    //      input_hidden_state (token_num, d_model),
    //      attention_mask (batch, 1, seqlen, seqlen),
    //      padding_offset (token_num)
    //      relative_attention_bias (optional)
    //          If padding_offset.data is nullptr, then not remove padding
    //      trt_padding_offset (batch_size + 1 or 2 * batch_size + 1)

    // ouptut_tensors:
    //      output_hidden_state [token_num, d_model]

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() >= 2);
    const int request_batch_size = input_tensors->at("attention_mask").shape[0];
    const int request_seq_len    = input_tensors->at("attention_mask").shape[2];
    allocateBuffer(request_batch_size, request_seq_len);

    T1*        attention_out   = output_tensors->at("output_hidden_state").getPtr<T1>();
    const T1*  attention_input = input_tensors->at("input_hidden_state").getPtr<T1>();
    const T1*  attention_mask  = input_tensors->at("attention_mask").getPtr<T1>();
    const int* padding_offset =
        input_tensors->isExist("padding_offset") ? input_tensors->at("padding_offset").getPtr<int>() : nullptr;
    const int* trt_padding_offset =
        input_tensors->isExist("trt_padding_offset") ? input_tensors->at("trt_padding_offset").getPtr<int>() : nullptr;

    const T1* relative_attention_bias = input_tensors->isExist("relative_attention_bias") ?
                                            input_tensors->at("relative_attention_bias").getPtr<T1>() :
                                            nullptr;

    const int           m                      = input_tensors->at("input_hidden_state").shape[0];
    int                 k                      = d_model_;
    int                 n                      = hidden_units_;
    const int           request_seq_len_padded = (request_seq_len + 15) / 16 * 16;
    const AttentionType attention_type         = input_tensors->getVal<AttentionType>("attention_type");

    if (attention_weights->query_weight.fuse_gemm_bias) {
        PUSH_RANGE("qkv gemm bias");
        reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_)
            ->Conv1x1Gemm<false, false>(q_buf_2_,
                                        m,
                                        3 * hidden_units_,
                                        d_model_,
                                        attention_input,
                                        attention_weights->query_weight.kernel,
                                        attention_weights->query_weight.bias,
                                        *(attention_weights->query_weight.input_h_scale),       // scale_a,
                                        *(attention_weights->query_weight.weight_h_scale),      // scale_b,
                                        *(attention_weights->query_weight.output_h_scale_inv),  // scale_d,
                                        stream_);
        sync_check_cuda_error();
        POP_RANGE;
    }
    else {
        PUSH_RANGE("qkv_gemm");
        if (fp8_mode_ == 1) {
            FT_CHECK_WITH_INFO(false, fmtstr("%s not support fp8_mode 1 now", __PRETTY_FUNCTION__));
        }
        else if (fp8_mode_ == 2) {
            float alpha = 1.0f;
            float beta  = 0.0f;
            reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_)
                ->Gemm(qkv_buf_,
                       (int)1,
                       (int)m,
                       (int)3 * hidden_units_,
                       (int)d_model_,
                       (int64_t)0,
                       (int64_t)0,
                       (int64_t)0,
                       &alpha,
                       &beta,
                       attention_input,
                       attention_weights->query_weight.kernel,
                       attention_weights->query_weight.input_scale,
                       attention_weights->query_weight.weight_scale,
                       attention_weights->query_weight.output_scale_inv,
                       stream_);
            sync_check_cuda_error();
        }
        POP_RANGE;
    }

    if (isFusedMHA(attention_type)) {
        if (!attention_weights->query_weight.fuse_gemm_bias) {
            PUSH_RANGE("invokeFP8TrtAddQKVBias");
            FP8TrtAddQKVBiasParam<T1, T2> param{q_buf_2_,
                                                qkv_buf_,
                                                attention_weights->query_weight.bias,
                                                attention_weights->query_weight.output_scale,
                                                attention_weights->query_weight.output_scale_inv,
                                                (size_t)m,
                                                head_num_,
                                                size_per_head_,
                                                head_num_ * size_per_head_,
                                                stream_};
            invokeFP8TrtAddQKVBias(param);
            sync_check_cuda_error();
            POP_RANGE;
        }

        {
            PUSH_RANGE("run_fmha_v2");
            dispatcher_fp8->setup_flags(false, false, false, false, false);
            int s = dispatcher_fp8->getSFromMaxSeqLen(request_seq_len);
            FT_CHECK(dispatcher_fp8->isValid(s, false));
            float attn_scale_1 = attention_weights->query_weight.output_h_scale[0]
                                 * attention_weights->query_weight.output_h_scale[0] * 1.0f
                                 / q_scaling_;  // q and k and softmax scaling
            float attn_scale_2 =
                attention_weights->query_weight.output_h_scale[0]
                * attention_weights->attention_output_weight.input_h_scale_inv[0];  // v and output scale
            dispatcher_fp8->setScaleList(attn_scale_1, 1.0f, attn_scale_2);
            // For example, if a query is like
            // [[S_0, P_0], [S_1, P_1]], where S_i is real tokens and P_i is padded tokens.
            // In zero pad case, we remove the padding and the input looks like [S_0, S_1].
            // In padding case, we view padding as sentences and the input looks like [S_0, P_0, S_1, P_0]
            // to leverage the fused mha directly.
            dispatcher_fp8->setup(s, input_tensors->at("trt_padding_offset").size() - 1);
            dispatcher_fp8->run(q_buf_2_, nullptr, const_cast<int*>(trt_padding_offset), nullptr, qkv_buf_3_, stream_);
            sync_check_cuda_error();
            POP_RANGE;
        }
    }
    else {
        FT_CHECK_WITH_INFO(!attention_weights->query_weight.fuse_gemm_bias,
                           "Unfused MHA only support fuse_gemm_bias=false");
        {
            PUSH_RANGE("invokeFP8AddFusedQKVBiasRebuildPadding");
            FP8AddFusedQKVBiasRebuildPaddingParam<T1, T2> param{
                q_buf_2_,
                k_buf_2_,
                v_buf_2_,
                qkv_buf_,  // T1
                nullptr,   // T2
                attention_weights->query_weight.bias,
                attention_weights->query_weight.output_scale,
                fp8_mode_ == 1 ? attention_weights->query_weight.scale : nullptr,
                fp8_mode_ == 1 ? attention_weights->query_weight.per_channel_scale_min : nullptr,
                attention_weights->query_weight.output_scale_inv,
                padding_offset,
                trt_padding_offset,
                nullptr,
                (uint32_t)m,
                (uint32_t)request_batch_size,
                (uint32_t)request_seq_len,
                (uint32_t)request_seq_len_padded,
                (uint32_t)request_seq_len_padded,
                (uint32_t)head_num_,
                (uint32_t)size_per_head_,
                0,  // rotary_embedding_dim_,
                stream_};
            invokeFP8AddFusedQKVBiasRebuildPadding(param);
            sync_check_cuda_error();
            POP_RANGE;
        }

        {
            PUSH_RANGE("Q*K batch gemm");
            const float alpha = 1.0f;
            const float beta  = 0.0f;
            reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_)
                ->Gemm(qk_buf_bfloat_,
                       (int)request_batch_size * head_num_,
                       (int)request_seq_len,
                       (int)request_seq_len,
                       (int)size_per_head_,
                       (int64_t)request_seq_len * size_per_head_,
                       (int64_t)request_seq_len * size_per_head_,
                       (int64_t)request_seq_len * request_seq_len,
                       &alpha,
                       &beta,
                       q_buf_2_,
                       k_buf_2_,
                       attention_weights->query_weight.output_scale,
                       attention_weights->query_weight.output_scale,
                       stream_);
            sync_check_cuda_error();
            POP_RANGE;
        }
        FT_CHECK(relative_attention_bias == nullptr);
        // if (relative_attention_bias != nullptr) {
        //     invokeAddRelativeAttentionBias(
        //         qk_buf_, relative_attention_bias, request_batch_size, head_num_, request_seq_len, stream_);
        // }

        {
            PUSH_RANGE("softmax");
            float                         scalar = (float)(1.0f / (sqrtf(size_per_head_ * 1.0f) * q_scaling_));
            FP8MaskedSoftMaxParam<T1, T2> param{qk_buf_,
                                                qk_buf_bfloat_,
                                                attention_mask,
                                                trt_padding_offset,
                                                (uint32_t)request_batch_size,
                                                (uint32_t)request_seq_len,
                                                (uint32_t)head_num_,
                                                scalar,
                                                nullptr,
                                                attention_weights->qk_scale_inv,
                                                stream_};
            invokeFP8MaskedSoftMax(param);
            sync_check_cuda_error();
            POP_RANGE;
        }

        {
            PUSH_RANGE("QK*V batch gemm");

            const float alpha = 1.0f;
            const float beta  = 0.0f;
            reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_)
                ->Gemm(qkv_buf_2_,
                       (int)request_batch_size * head_num_,
                       (int)request_seq_len,
                       (int)size_per_head_,
                       (int)request_seq_len,
                       (int64_t)size_per_head_ * request_seq_len,
                       (int64_t)request_seq_len * request_seq_len,
                       (int64_t)size_per_head_ * request_seq_len,
                       &alpha,
                       &beta,
                       qk_buf_,
                       v_buf_2_,
                       attention_weights->qk_scale,                                 // qk_buf_ scale
                       attention_weights->query_weight.output_scale,                // v_buf_2_ scale
                       attention_weights->attention_output_weight.input_scale_inv,  // outupt_scale
                       stream_);

            sync_check_cuda_error();
            POP_RANGE
        }
        {
            PUSH_RANGE("invokeFP8TransposeAttentionOutRemovePadding");
            FP8TransposeAttentionOutRemovePaddingParam<T1, T1> param{qkv_buf_3_,
                                                                     qkv_buf_2_,
                                                                     (const float*)nullptr,
                                                                     m,
                                                                     request_batch_size,
                                                                     request_seq_len,
                                                                     (int)head_num_,
                                                                     (int)size_per_head_,
                                                                     padding_offset,
                                                                     stream_};
            invokeFP8TransposeAttentionOutRemovePadding(param);
            POP_RANGE;
        }
    }

    k = hidden_units_;
    n = d_model_;

    PUSH_RANGE("proj gemm");
    if (fp8_mode_ == 1) {
        FT_CHECK_WITH_INFO(false, fmtstr("%s not support fp8_mode 1 now", __PRETTY_FUNCTION__));
    }
    else if (fp8_mode_ == 2) {
#ifdef FUSE_GEMM_ACT
        reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_)
            ->Conv1x1Gemm<false, false>(attention_out,
                                        m,
                                        n,
                                        k,
                                        qkv_buf_3_,
                                        attention_weights->attention_output_weight.kernel,
                                        attention_weights->attention_output_weight.bias,
                                        *(attention_weights->attention_output_weight.input_h_scale),       // scale_a,
                                        *(attention_weights->attention_output_weight.weight_h_scale),      // scale_b,
                                        *(attention_weights->attention_output_weight.output_h_scale_inv),  // scale_d,
                                        stream_);
#else
        float alpha = 1.0f;
        float beta  = 0.0f;
        reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_)
            ->Gemm(attention_out,
                   (int)1,
                   (int)m,
                   (int)n,
                   (int)k,
                   (int64_t)0,
                   (int64_t)0,
                   (int64_t)0,
                   &alpha,
                   &beta,
                   qkv_buf_3_,
                   attention_weights->attention_output_weight.kernel,
                   attention_weights->attention_output_weight.input_scale,
                   attention_weights->attention_output_weight.weight_scale,
                   attention_weights->attention_output_weight.output_scale_inv,
                   stream_);
#endif
    }

    POP_RANGE;
    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template<typename T1, typename T2>
SelfAttentionFP8Layer<T1, T2>::SelfAttentionFP8Layer(size_t           head_num,
                                                     size_t           size_per_head,
                                                     size_t           d_model,
                                                     float            q_scaling,
                                                     int              fp8_mode,
                                                     int              sm,
                                                     cudaStream_t     stream,
                                                     cublasMMWrapper* cublas_wrapper,
                                                     IAllocator*      allocator,
                                                     bool             is_free_buffer_after_forward,
                                                     bool             sparse):
    BaseAttentionFP8Layer<T1, T2>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    head_num_(head_num),
    size_per_head_(size_per_head),
    d_model_(d_model),
    hidden_units_(head_num_ * size_per_head_),
    fp8_mode_(fp8_mode),
    sm_(sm),
    sparse_(sparse),
    q_scaling_(q_scaling)
{
    FT_CHECK_WITH_INFO(sparse_ == false, fmtstr("%s not support sparse gemm yet.", __PRETTY_FUNCTION__));
    if ((sm_ == kSM_90 || sm_ == kSM_89) && size_per_head_ == 64) {
        dispatcher_fp8.reset(new FusedMHARunnerFP8v2(head_num_, size_per_head_, sm_, q_scaling_));
    }
}

template<typename T1, typename T2>
SelfAttentionFP8Layer<T1, T2>::SelfAttentionFP8Layer(SelfAttentionFP8Layer<T1, T2> const& attention_layer):
    SelfAttentionFP8Layer(attention_layer.head_num_,
                          attention_layer.size_per_head_,
                          attention_layer.d_model_,
                          attention_layer.q_scaling_,
                          attention_layer.fp8_mode_,
                          attention_layer.sm_,
                          attention_layer.stream_,
                          attention_layer.cublas_wrapper_,
                          attention_layer.allocator_,
                          attention_layer.is_free_buffer_after_forward_,
                          attention_layer.sparse_)
{
}

template<typename T1, typename T2>
SelfAttentionFP8Layer<T1, T2>::~SelfAttentionFP8Layer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T1, typename T2>
void SelfAttentionFP8Layer<T1, T2>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T1, typename T2>
void SelfAttentionFP8Layer<T1, T2>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    qkv_buf_ = (T1*)(allocator_->reMalloc(qkv_buf_, sizeof(T1) * batch_size * seq_len * 3 * hidden_units_, false));
    qk_buf_bfloat_ =
        (T2*)(allocator_->reMalloc(qk_buf_bfloat_, sizeof(T2) * batch_size * head_num_ * seq_len * seq_len, false));
    qk_buf_ = (T1*)(allocator_->reMalloc(qk_buf_, sizeof(T1) * batch_size * head_num_ * seq_len * seq_len, false));

    q_buf_2_ = (T1*)(allocator_->reMalloc(q_buf_2_, sizeof(T1) * batch_size * seq_len * 3 * hidden_units_, false));
    k_buf_2_ = q_buf_2_ + batch_size * seq_len * hidden_units_;
    v_buf_2_ = k_buf_2_ + batch_size * seq_len * hidden_units_;

    qkv_buf_2_ = (T1*)(allocator_->reMalloc(qkv_buf_2_, sizeof(T1) * batch_size * seq_len * hidden_units_, false));
    qkv_buf_3_ = (T1*)(allocator_->reMalloc(qkv_buf_3_, sizeof(T1) * batch_size * seq_len * hidden_units_, false));
    qkv_buf_4_ = (T2*)(allocator_->reMalloc(qkv_buf_4_, sizeof(T2) * batch_size * seq_len * hidden_units_, false));

    is_allocate_buffer_ = true;
}

template<typename T1, typename T2>
void SelfAttentionFP8Layer<T1, T2>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&qkv_buf_));
        allocator_->free((void**)(&qk_buf_bfloat_));
        allocator_->free((void**)(&qk_buf_));
        allocator_->free((void**)(&q_buf_2_));
        allocator_->free((void**)(&qkv_buf_2_));
        allocator_->free((void**)(&qkv_buf_3_));
        allocator_->free((void**)(&qkv_buf_4_));
        sync_check_cuda_error();
        is_allocate_buffer_ = false;
    }
}

template class SelfAttentionFP8Layer<__nv_fp8_e4m3, __nv_bfloat16>;

}  // namespace fastertransformer
