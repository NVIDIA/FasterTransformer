/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
size_t WindowAttention<T>::roundByteSize(const size_t size, const int factor)
{
    return (size + factor - 1) / factor * factor;
}

template<typename T>
size_t WindowAttention<T>::getBufSize(
    const int batch, const int num_head, const int window_num, const int window_len, const int dim)
{
    int sm;
#if (__CUDA_ARCH__ < 750)
    sm = 70;
#else
    sm = 75;
#endif
    // in case we may use trt fused-multihead attention
    if ((sm == 75 || sm == 80 || sm == 86) && std::is_same<T, half>::value && window_len <= TRT_MAX_LEN) {
        int S = trt_getS(window_len);
        // transformed_attention_mask
        size_t buf_size = roundByteSize(window_num * S * S * sizeof(T), 4) +
                          // transformed_relative_position_bias
                          roundByteSize(num_head * S * S * sizeof(T), 4) +
                          // qkv_buf_ + q_buf_ + k_buf_ + v_buf_
                          6 * batch * window_num * window_len * dim * sizeof(T) +
                          // qk_buf_
                          0;
        return (buf_size + 31) / 32 * 32;
    }
    // we will not use trt fused-multihead attention
    {
        // qkv_buf_ + q_buf_ + k_buf_ + v_buf_
        size_t buf_size = 6 * batch * window_num * window_len * dim * sizeof(T) +
                          // qk_buf_
                          batch * window_num * num_head * window_len * window_len * sizeof(T);
        return (buf_size + 31) / 32 * 32;
    }
}

template<typename T>
void WindowAttention<T>::allocateBuffer()
{
    int num_head = num_head_;
    if (!is_free_buffer_after_forward_) {
        num_head = num_head * 8;
    }
    if (is_allocate_buffer_ == false) {
        if (use_trt_) {
            int S = trt_getS(window_len_);
            trt_attention_mask_ = (half*)allocator_->malloc(roundByteSize(window_num_ * S * S * sizeof(T), 4), false);
            trt_relative_position_bias_ =
                (half*)allocator_->malloc(roundByteSize(num_head * S * S * sizeof(T), 4), false);
            qkv_buf_ =
                (T*)allocator_->malloc(3 * max_batch_ * window_num_ * window_len_ * embed_dim_ * sizeof(T), false);
            q_buf_ = (T*)allocator_->malloc(3 * max_batch_ * window_num_ * window_len_ * embed_dim_ * sizeof(T), false);
            k_buf_ = q_buf_ + max_batch_ * window_num_ * window_len_ * embed_dim_;
            v_buf_ = k_buf_ + max_batch_ * window_num_ * window_len_ * embed_dim_;
            qk_buf_ = nullptr;
        }
        else {
            trt_attention_mask_ = nullptr;
            trt_relative_position_bias_ = nullptr;
            qkv_buf_ =
                (T*)allocator_->malloc(3 * max_batch_ * window_num_ * window_len_ * embed_dim_ * sizeof(T), false);
            q_buf_ = (T*)allocator_->malloc(3 * max_batch_ * window_num_ * window_len_ * embed_dim_ * sizeof(T), false);
            k_buf_ = q_buf_ + max_batch_ * window_num_ * window_len_ * embed_dim_;
            v_buf_ = k_buf_ + max_batch_ * window_num_ * window_len_ * embed_dim_;
            qk_buf_ = (T*)allocator_->malloc(
                3 * max_batch_ * window_num_ * num_head * window_len_ * window_len_ * sizeof(T), false);
        }
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void WindowAttention<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        if (use_trt_) {
            allocator_->free(trt_attention_mask_);
            allocator_->free(trt_relative_position_bias_);
            allocator_->free(qkv_buf_);
            allocator_->free(q_buf_);
        }
        else {
            allocator_->free(qkv_buf_);
            allocator_->free(q_buf_);
            allocator_->free(qk_buf_);
        }
        is_allocate_buffer_ = false;
    }
}

template<typename T>
WindowAttention<T>::WindowAttention(int max_batch,
                                    int window_size,
                                    cudaStream_t stream,
                                    cublasMMWrapper* cublas_wrapper,
                                    IAllocator* allocator,
                                    bool is_free_buffer_after_forward,
                                    bool qkv_bias,
                                    float qk_scale):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_(max_batch),
    window_size_(window_size),
    qkv_bias_(qkv_bias),
    qk_scale_(qk_scale)
{
    window_len_ = window_size_ * window_size_;
}

template<typename T>
WindowAttention<T>::~WindowAttention()
{
}

template<typename T>
void WindowAttention<T>::forward(std::vector<fastertransformer::Tensor>* output_tensors,
                                 const std::vector<fastertransformer::Tensor>* input_tensors,
                                 const AttentionWeight<T>* attention_weights)
{
    // input_tensors:
    //      input [batch * window_num * window_len, dim]
    //      attention_mask [window_num, window_len, window_len]
    //      attention_relative_position_bias [num_head, window_len, window_len]
    //      additional_params [6] {batch, dim, input_resolution, num_head, shift_size, sm}
    // output_tensors:
    //      output [batch * window_num * window_len, dim]

    T* output = (T*)output_tensors->at(0).data;
    const T* input = (const T*)input_tensors->at(0).data;
    const T* attention_mask = (const T*)input_tensors->at(1).data;
    const T* attention_relative_pos_bias = (const T*)input_tensors->at(2).data;
    const int* additional_params = (const int*)input_tensors->at(3).data;
    const int batch = additional_params[0];
    const int dim = additional_params[1];
    const int input_resolution = additional_params[2];
    const int num_head = additional_params[3];
    const int shift_size = additional_params[4];
    const int sm = additional_params[5];

    int size_per_head = dim / num_head;
    int trt_S = 1024;
    if ((sm == 75 || sm == 80 || sm == 86) && size_per_head == 32 && window_len_ <= TRT_MAX_LEN
        && std::is_same<T, half>::value) {
        trt_S = trt_getS(window_len_);
        use_trt_ = true;
    }
    num_head_ = num_head;
    window_num_ = (input_resolution / window_size_) * (input_resolution / window_size_);
    embed_dim_ = dim;
    allocateBuffer();

    float scale = 1.0f / sqrt(size_per_head);
    if (fabs(qk_scale_ - 1.0f) > 0.0001) {
        scale = qk_scale_;
    }

    if (use_trt_) {
        if (dispatcher_fp16_.get() && num_head == dispatcher_fp16_num_head_) {}
        else {
            dispatcher_fp16_.reset(new FusedMHARunnerFP16v2(num_head, size_per_head, sm, 1.0f));
            dispatcher_fp16_num_head_ = num_head;
        }
    }

    // NOTICE: We preprocess the attention_qkv_kernel and attention_qkv_bias for :
    //        1. gemm+bias fusion in cublasLt 11.x and
    //        2. fmha.
    //        So attention_qkv_kernel is [k, head*3*size], attention_qkv_bias is [head*3*size]

    int S;
    if (use_trt_ && dispatcher_fp16_.get()) {
        S = dispatcher_fp16_->getSFromMaxSeqLen(window_len_);
    }

    if (use_trt_ && dispatcher_fp16_.get() && dispatcher_fp16_->isValid(S)) {
#if (CUDART_VERSION >= 11000)
        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              3 * dim,
                              batch * window_num_ * window_len_,
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
                              batch * window_num_ * window_len_,
                              dim,
                              attention_weights->query_weight.kernel,
                              3 * dim,
                              input,
                              dim,
                              q_buf_,
                              3 * dim);

        invokeAddBias((half*)q_buf_,
                      (const half*)(attention_weights->query_weight.bias),
                      batch * window_num_ * window_len_,
                      3 * dim,
                      stream_);
#endif

        half* trt_attention_mask = nullptr;
        if (shift_size != 0) {
            invokeTransformMask(trt_attention_mask_, (const half*)attention_mask, window_num_, window_len_, stream_);
            trt_attention_mask = trt_attention_mask_;
        }

        invokeTransformMask(
            trt_relative_position_bias_, (const half*)attention_relative_pos_bias, num_head, window_len_, stream_);
        const int B = batch * window_num_;
        dispatcher_fp16_->setup(S, B, window_num_);
        dispatcher_fp16_->run(
            q_buf_, trt_attention_mask, trt_relative_position_bias_, window_len_, nullptr, qkv_buf_, stream_);
        cublas_wrapper_->Gemm(CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              dim,
                              batch * window_num_ * window_len_,
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
                              batch * window_num_ * window_len_,
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
                                  window_num_,
                                  window_len_,
                                  num_head,
                                  size_per_head,
                                  stream_);

        cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                            CUBLAS_OP_N,
                                            window_len_,
                                            window_len_,
                                            size_per_head,
                                            k_buf_,
                                            size_per_head,
                                            window_len_ * size_per_head,
                                            q_buf_,
                                            size_per_head,
                                            window_len_ * size_per_head,
                                            qk_buf_,
                                            window_len_,
                                            window_len_ * window_len_,
                                            batch * window_num_ * num_head);

        if (shift_size != 0) {
            invokeMaskedSoftMaxWithRelPosBias(qk_buf_,
                                              attention_mask,
                                              attention_relative_pos_bias,
                                              batch,
                                              num_head,
                                              window_num_,
                                              window_len_,
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
                                              window_num_,
                                              window_len_,
                                              scale,
                                              stream_);
        }

        cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
                                            CUBLAS_OP_N,
                                            size_per_head,
                                            window_len_,
                                            window_len_,
                                            v_buf_,
                                            size_per_head,
                                            size_per_head * window_len_,
                                            qk_buf_,
                                            window_len_,
                                            window_len_ * window_len_,
                                            qkv_buf_,
                                            size_per_head,
                                            size_per_head * window_len_,
                                            batch * window_num_ * num_head);

        invokeTransposeQKV(v_buf_, qkv_buf_, batch * window_num_, window_len_, num_head, size_per_head, stream_);

        cublas_wrapper_->Gemm(CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              dim,
                              batch * window_num_ * window_len_,
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
                      window_num_,
                      window_len_,
                      window_size_,
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