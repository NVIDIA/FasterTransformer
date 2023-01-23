/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/layers/attention_layers/WindowAttention.h"

namespace fastertransformer {

template<typename T>
int WindowAttention<T>::trt_getS(const int actual_seqlen)
{
    int S = 384;
    if (actual_seqlen <= 64) {
        S = 64;
    }
    else if (actual_seqlen <= 128) {
        S = 128;
    }
    else if (actual_seqlen <= 256) {
        S = 256;
    }
    else if (actual_seqlen <= 384) {
        S = 384;
    }
    return S;
}

template<typename T>
void WindowAttention<T>::allocateBuffer()
{
    assert(false && "WindowAttentionINT8<T>::allocateBuffer() not implemented");
}

template<typename T>
void WindowAttention<T>::allocateBuffer(int batch, int window_num, int window_len, int embed_dim, int num_head)
{
    if (is_allocate_buffer_ == false) {
        if (use_trt_) {
            int S = trt_getS(window_len);
            qkv_buf_ =
                (T*)allocator_->reMalloc(qkv_buf_, 3 * batch * window_num * window_len * embed_dim * sizeof(T), false);
            q_buf_ =
                (T*)allocator_->reMalloc(q_buf_, 3 * batch * window_num * window_len * embed_dim * sizeof(T), false);
            k_buf_  = q_buf_ + batch * window_num * window_len * embed_dim;
            v_buf_  = k_buf_ + batch * window_num * window_len * embed_dim;
            qk_buf_ = nullptr;
        }
        else {
            qkv_buf_ =
                (T*)allocator_->reMalloc(qkv_buf_, 3 * batch * window_num * window_len * embed_dim * sizeof(T), false);
            q_buf_ =
                (T*)allocator_->reMalloc(q_buf_, 3 * batch * window_num * window_len * embed_dim * sizeof(T), false);
            k_buf_  = q_buf_ + batch * window_num * window_len * embed_dim;
            v_buf_  = k_buf_ + batch * window_num * window_len * embed_dim;
            qk_buf_ = (T*)allocator_->reMalloc(
                qk_buf_, batch * window_num * num_head * window_len * window_len * sizeof(T), false);
        }
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void WindowAttention<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        if (use_trt_) {
            allocator_->free((void**)(&qkv_buf_));
            allocator_->free((void**)(&q_buf_));
        }
        else {
            allocator_->free((void**)(&qkv_buf_));
            allocator_->free((void**)(&q_buf_));
            allocator_->free((void**)(&qk_buf_));
        }
        is_allocate_buffer_ = false;
    }
}

template<typename T>
WindowAttention<T>::WindowAttention(int              max_batch,
                                    int              window_size,
                                    cudaStream_t     stream,
                                    cublasMMWrapper* cublas_wrapper,
                                    IAllocator*      allocator,
                                    bool             is_free_buffer_after_forward,
                                    bool             qkv_bias,
                                    float            qk_scale,
                                    int              version):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_(max_batch),
    window_size_(window_size),
    qkv_bias_(qkv_bias),
    qk_scale_(qk_scale),
    version_(version)
{
}

template<typename T>
WindowAttention<T>::~WindowAttention()
{
}

template<typename T>
void WindowAttention<T>::forward(TensorMap*                output_tensors,
                                 TensorMap*                input_tensors,
                                 const AttentionWeight<T>* attention_weights)
{
    // input_tensors:
    //      input [batch * window_num * window_len, dim]
    //      attention_mask [window_num, window_len, window_len]
    //      trt_attention_mask [window_num, window_len, window_len]
    //      attention_relative_position_bias [num_head, window_len, window_len]
    //      trt_relative_position_bias [num_head, window_len, window_len]
    //      attention_logit_scale [num_head]
    //      additional_params [7] {batch, dim, input_resolution, num_head, shift_size, sm, window_size_in_use}
    // output_tensors:
    //      output [batch * window_num * window_len, dim]
    T*         output                      = output_tensors->getPtr<T>("hidden_features");
    const T*   input                       = input_tensors->getPtr<T>("input_query");
    const T*   attention_mask              = input_tensors->getPtr<T>("attention_mask", nullptr);
    const T*   trt_attention_mask          = input_tensors->getPtr<T>("trt_attention_mask", nullptr);
    const T*   attention_relative_pos_bias = input_tensors->getPtr<T>("attention_relative_position_bias");
    const T*   trt_relative_position_bias  = input_tensors->getPtr<T>("trt_relative_position_bias", nullptr);
    const int  window_len_in_use           = input_tensors->at("attention_relative_position_bias").shape[1];
    const T*   attention_logit_scale       = input_tensors->getPtr<T>("attn_logit_scale", nullptr);
    const int* additional_params           = input_tensors->getPtr<int>("additional_params");
    const int  batch                       = additional_params[0];
    const int  dim                         = additional_params[1];
    const int  input_resolution            = additional_params[2];
    const int  num_head                    = additional_params[3];
    const int  shift_size                  = additional_params[4];
    const int  sm                          = additional_params[5];
    const int  window_size_in_use          = additional_params[6];

    int size_per_head = dim / num_head;
    int trt_S         = 1024;
    // we should decide whether to use trt fmha based on window_size_ * window_size_
    if ((sm == 75 || sm == 80 || sm == 86) && size_per_head == 32 && window_size_ * window_size_ <= TRT_MAX_LEN
        && std::is_same<T, half>::value) {
        trt_S    = trt_getS(window_len_in_use);
        use_trt_ = true;
    }
    int window_num = (input_resolution / window_size_in_use) * (input_resolution / window_size_in_use);
    allocateBuffer(batch, window_num, window_len_in_use, dim, num_head);

    float scale = (version_ == 1) ? (1.0f / sqrt(size_per_head)) : 1.0f;
    if (fabs(qk_scale_ - 1.0f) > 0.0001) {
        scale = qk_scale_;
    }

    if (use_trt_) {
        if (dispatcher_fp16_.get() && num_head == dispatcher_fp16_num_head_) {}
        else {
            scale = 1.0f;
            if (fabs(qk_scale_ - 1.0f) > 0.0001) {
                scale = 1.0f / sqrt(size_per_head) / qk_scale_;
            }
            dispatcher_fp16_.reset(new FusedMHARunnerFP16v2(num_head, size_per_head, sm, scale));
            dispatcher_fp16_num_head_ = num_head;
        }
    }
    // NOTICE: We preprocess the attention_qkv_kernel and attention_qkv_bias for :
    //        1. gemm+bias fusion in cublasLt 11.x and
    //        2. fmha.
    //        So attention_qkv_kernel is [k, head*3*size], attention_qkv_bias is [head*3*size]

    int S;
    if (use_trt_ && dispatcher_fp16_.get()) {
        S = dispatcher_fp16_->getSFromMaxSeqLen(window_len_in_use, true);
    }

    if (use_trt_ && dispatcher_fp16_.get() && dispatcher_fp16_->isValid(S, true)) {
#if (CUDART_VERSION >= 11000)
        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              3 * dim,
                              batch * window_num * window_len_in_use,
                              dim,
                              attention_weights->query_weight.kernel,
                              3 * dim,
                              input,
                              dim,
                              attention_weights->query_weight.bias,
                              q_buf_,
                              3 * dim);
#else
        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              3 * dim,
                              batch * window_num * window_len_in_use,
                              dim,
                              attention_weights->query_weight.kernel,
                              3 * dim,
                              input,
                              dim,
                              q_buf_,
                              3 * dim);

        invokeGenericActivation<IdentityActivation, half, half>((half*)q_buf_,
                                                                (const half*)(attention_weights->query_weight.bias),
                                                                nullptr,
                                                                nullptr,
                                                                nullptr,
                                                                nullptr,
                                                                batch * window_num * window_len_in_use,
                                                                3 * dim,
                                                                0,
                                                                nullptr,
                                                                nullptr,
                                                                stream_);
#endif
        const int B = batch * window_num;
        dispatcher_fp16_->setup(S, B, window_num);

        if (version_ == 2) {
            invokeNormalizeForFMHA(
                q_buf_, attention_logit_scale, B, window_len_in_use, num_head, size_per_head, stream_);
        }
        dispatcher_fp16_->run(q_buf_,
                              shift_size != 0 ? trt_attention_mask : nullptr,
                              trt_relative_position_bias,
                              window_len_in_use,
                              nullptr,
                              qkv_buf_,
                              stream_);

        cublas_wrapper_->Gemm(CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              dim,
                              batch * window_num * window_len_in_use,
                              dim,
                              attention_weights->attention_output_weight.kernel,
                              dim,
                              qkv_buf_,
                              dim,
                              q_buf_,
                              dim);
    }
    else {
        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              3 * dim,
                              batch * window_num * window_len_in_use,
                              dim,
                              attention_weights->query_weight.kernel,
                              3 * dim,
                              input,
                              dim,
                              qkv_buf_,
                              3 * dim);
        invokeAddHead3SizeQKVBias(qkv_buf_,
                                  attention_weights->query_weight.bias,
                                  q_buf_,
                                  k_buf_,
                                  v_buf_,
                                  batch,
                                  window_num,
                                  window_len_in_use,
                                  num_head,
                                  size_per_head,
                                  stream_);
        if (version_ == 2) {
            invokeNormalize(
                q_buf_, attention_logit_scale, batch * window_num, window_len_in_use, num_head, size_per_head, stream_);
            invokeNormalize(
                k_buf_, (const T*)nullptr, batch * window_num, window_len_in_use, num_head, size_per_head, stream_);
        }
        cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                            CUBLAS_OP_N,
                                            window_len_in_use,
                                            window_len_in_use,
                                            size_per_head,
                                            k_buf_,
                                            size_per_head,
                                            window_len_in_use * size_per_head,
                                            q_buf_,
                                            size_per_head,
                                            window_len_in_use * size_per_head,
                                            qk_buf_,
                                            window_len_in_use,
                                            window_len_in_use * window_len_in_use,
                                            batch * window_num * num_head);
        if (shift_size != 0) {
            invokeMaskedSoftMaxWithRelPosBias(qk_buf_,
                                              attention_mask,
                                              attention_relative_pos_bias,
                                              batch,
                                              num_head,
                                              window_num,
                                              window_len_in_use,
                                              scale,
                                              stream_);
        }
        else {
            const T* attn_mask_tmp = nullptr;
            invokeMaskedSoftMaxWithRelPosBias(qk_buf_,
                                              attn_mask_tmp,
                                              attention_relative_pos_bias,
                                              batch,
                                              num_head,
                                              window_num,
                                              window_len_in_use,
                                              scale,
                                              stream_);
        }
        cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
                                            CUBLAS_OP_N,
                                            size_per_head,
                                            window_len_in_use,
                                            window_len_in_use,
                                            v_buf_,
                                            size_per_head,
                                            size_per_head * window_len_in_use,
                                            qk_buf_,
                                            window_len_in_use,
                                            window_len_in_use * window_len_in_use,
                                            qkv_buf_,
                                            size_per_head,
                                            size_per_head * window_len_in_use,
                                            batch * window_num * num_head);
        invokeTransposeQKV(v_buf_,
                           qkv_buf_,
                           batch * window_num,
                           window_len_in_use,
                           num_head,
                           size_per_head,
                           (float*)nullptr,
                           0,
                           stream_);
        cublas_wrapper_->Gemm(CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              dim,
                              batch * window_num * window_len_in_use,
                              dim,
                              attention_weights->attention_output_weight.kernel,
                              dim,
                              v_buf_,
                              dim,
                              q_buf_,
                              dim);
    }
    invokeReverseRoll(output,
                      q_buf_,
                      batch,
                      window_num,
                      window_len_in_use,
                      window_size_in_use,
                      input_resolution,
                      input_resolution,
                      dim,
                      shift_size,
                      stream_);
    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
}

template class WindowAttention<float>;
template class WindowAttention<half>;

}  // namespace fastertransformer
