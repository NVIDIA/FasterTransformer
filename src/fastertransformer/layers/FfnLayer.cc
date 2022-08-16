/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/layers/FfnLayer.h"

namespace fastertransformer {

template<typename T>
void FfnLayer<T>::forward(std::vector<fastertransformer::Tensor>*       output_tensors,
                          const std::vector<fastertransformer::Tensor>* input_tensors,
                          const FfnWeight<T>*                           ffn_weights)
{
    // input tensors:
    //      ffn_input [token_num, hidden_dimension],

    // output tensors:
    //      ffn_output [token_num, hidden_dimension],

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() == 1);
    FT_CHECK(output_tensors->size() == 1);
    // FT_CHECK(isValidTokenNum(input_tensors->at(0).shape[0]));
    allocateBuffer(input_tensors->at(0).shape[0]);

    const int m             = input_tensors->at(0).shape[0];
    T*        output_tensor = (T*)output_tensors->at(0).data;
    const T*  input_tensor  = (const T*)input_tensors->at(0).data;

    // TODO: INT8 and Sparsity are currently not implemented (geglu or reglu)
    const bool use_gated_activation = use_gated_activation_ && ffn_weights->intermediate_weight2.kernel != nullptr;

#ifdef SPARSITY_ENABLED
    int m_tmp = input_tensors->at(0).shape[0];
    if (m_tmp % 8 != 0) {
        m_tmp = (m_tmp / 8 + 1) * 8;
    }
    const int m_padded = m_tmp;
    if (sparse_ && cublas_wrapper_->isUseSparse(1, inter_size_, m, hidden_units_)) {
        FT_CHECK(!use_gated_activation);
        cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                inter_size_,
                                m_padded,
                                hidden_units_,
                                ffn_weights->intermediate_weight.sp_kernel,
                                input_tensor,
                                inter_buf_);
    }
    else {
#endif
        if (int8_mode_ == 1 && m <= 2) {
            FT_CHECK(!use_gated_activation);
            FT_CHECK(ffn_weights->intermediate_weight.int8_kernel != NULL
                     && ffn_weights->intermediate_weight.scale != NULL);
            int8WeightPerChannelLdkMultiplicationLauncher(ffn_weights->intermediate_weight.int8_kernel,
                                                          input_tensor,
                                                          ffn_weights->intermediate_weight.scale,
                                                          inter_buf_,
                                                          m,
                                                          inter_size_,
                                                          hidden_units_,
                                                          stream_);
        }
        else {
            if (int8_mode_ == 1) {
                printf("[WARNING][FfnLayer<T>::forward] int8 gpt doesn't support m > 2, run fp gpt instead.\n");
            }
            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  inter_size_,
                                  m,
                                  hidden_units_,
                                  ffn_weights->intermediate_weight.kernel,
                                  inter_size_,
                                  input_tensor,
                                  hidden_units_,
                                  inter_buf_,
                                  inter_size_);
            if (use_gated_activation) {
                cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                      CUBLAS_OP_N,
                                      inter_size_,
                                      m,
                                      hidden_units_,
                                      ffn_weights->intermediate_weight2.kernel,
                                      inter_size_,
                                      input_tensor,
                                      hidden_units_,
                                      inter_buf_2_,
                                      inter_size_);
            }
        }
#ifdef SPARSITY_ENABLED
    }
#endif

    if (use_gated_activation) {
        invokeAddBiasGatedActivation(m, ffn_weights->intermediate_weight.bias, ffn_weights->intermediate_weight2.bias);
    }
    else {
        invokeAddBiasActivation(m, ffn_weights->intermediate_weight.bias);
    }
    sync_check_cuda_error();

#ifdef SPARSITY_ENABLED
    if (sparse_ && cublas_wrapper_->isUseSparse(1, hidden_units_, m, inter_size_)) {
        cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                hidden_units_,
                                m_padded,
                                inter_size_,
                                ffn_weights->output_weight.sp_kernel,
                                inter_buf_,
                                output_tensor);
    }
    else {
#endif
        if (int8_mode_ == 1 && m <= 2) {
            FT_CHECK(ffn_weights->output_weight.int8_kernel != NULL && ffn_weights->output_weight.scale != NULL);
            int8WeightPerChannelLdkMultiplicationLauncher(ffn_weights->output_weight.int8_kernel,
                                                          inter_buf_,
                                                          ffn_weights->output_weight.scale,
                                                          output_tensor,
                                                          m,
                                                          hidden_units_,
                                                          inter_size_,
                                                          stream_);
        }
        else {
            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  hidden_units_,
                                  m,
                                  inter_size_,
                                  ffn_weights->output_weight.kernel,
                                  hidden_units_,
                                  inter_buf_,
                                  inter_size_,
                                  output_tensor,
                                  hidden_units_);
        }
#ifdef SPARSITY_ENABLED
    }
#endif
    sync_check_cuda_error();
    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template<typename T>
FfnLayer<T>::FfnLayer(size_t           max_batch_size,
                      size_t           max_seq_len,
                      size_t           head_num,
                      size_t           size_per_head,
                      size_t           inter_size,
                      cudaStream_t     stream,
                      cublasMMWrapper* cublas_wrapper,
                      IAllocator*      allocator,
                      bool             is_free_buffer_after_forward,
                      bool             sparse,
                      int              int8_mode,
                      bool             use_gated_activation):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    max_token_num_(max_batch_size * max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    max_inter_size_(inter_size),
    inter_size_(inter_size),
    int8_mode_(int8_mode),
    use_gated_activation_(use_gated_activation)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
FfnLayer<T>::FfnLayer(FfnLayer<T> const& ffn_layer):
    BaseLayer(ffn_layer.stream_,
              ffn_layer.cublas_wrapper_,
              ffn_layer.allocator_,
              ffn_layer.is_free_buffer_after_forward_,
              ffn_layer.cuda_device_prop_,
              ffn_layer.sparse_),
    max_token_num_(ffn_layer.max_token_num_),
    head_num_(ffn_layer.head_num_),
    size_per_head_(ffn_layer.size_per_head_),
    hidden_units_(ffn_layer.hidden_units_),
    max_inter_size_(ffn_layer.max_inter_size_),
    inter_size_(ffn_layer.inter_size_),
    int8_mode_(ffn_layer.int8_mode_),
    use_gated_activation_(ffn_layer.use_gated_activation_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
FfnLayer<T>::~FfnLayer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void FfnLayer<T>::allocateBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_ == false) {
        inter_buf_ = (T*)allocator_->reMalloc(inter_buf_, sizeof(T) * max_token_num_ * max_inter_size_, false);
        if (use_gated_activation_) {
            inter_buf_2_ = (T*)allocator_->reMalloc(inter_buf_2_, sizeof(T) * max_token_num_ * max_inter_size_, false);
        }
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void FfnLayer<T>::allocateBuffer(size_t token_num)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    inter_buf_ = (T*)allocator_->reMalloc(inter_buf_, sizeof(T) * token_num * max_inter_size_, false);
    if (use_gated_activation_) {
        inter_buf_2_ = (T*)allocator_->reMalloc(inter_buf_2_, sizeof(T) * token_num * max_inter_size_, false);
    }
    is_allocate_buffer_ = true;
}

template<typename T>
void FfnLayer<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&inter_buf_));
        if (use_gated_activation_) {
            allocator_->free((void**)(&inter_buf_2_));
        }
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool FfnLayer<T>::isValidTokenNum(size_t token_num)
{
    if (max_token_num_ < token_num) {
        max_token_num_ = token_num;
    }
    return true;
}

template class FfnLayer<float>;
template class FfnLayer<half>;
#ifdef ENABLE_BF16
template class FfnLayer<__nv_bfloat16>;
#endif

template<typename T>
GeluFfnLayer<T>::GeluFfnLayer(size_t           max_batch_size,
                              size_t           max_seq_len,
                              size_t           head_num,
                              size_t           size_per_head,
                              size_t           inter_size,
                              cudaStream_t     stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator*      allocator,
                              bool             is_free_buffer_after_forward,
                              bool             sparse,
                              int              int8_mode,
                              bool             use_gated_activation):
    FfnLayer<T>(max_batch_size,
                max_seq_len,
                head_num,
                size_per_head,
                inter_size,
                stream,
                cublas_wrapper,
                allocator,
                is_free_buffer_after_forward,
                sparse,
                int8_mode,
                use_gated_activation)
{
}

template<typename T>
GeluFfnLayer<T>::GeluFfnLayer(GeluFfnLayer<T> const& gelu_ffn_layer): FfnLayer<T>(gelu_ffn_layer)
{
}

template<typename T>
void GeluFfnLayer<T>::invokeAddBiasActivation(const int m, const T* bias)
{
    invokeAddBiasGeluV2<T>(inter_buf_, bias, m, inter_size_, stream_);
}

template<typename T>
void GeluFfnLayer<T>::invokeAddBiasGatedActivation(const int m, const T* bias1, const T* bias2)
{
    invokeAddBiasGatedGelu<T>(inter_buf_, inter_buf_2_, bias1, bias2, m, inter_size_, stream_);
}

template class GeluFfnLayer<float>;
template class GeluFfnLayer<half>;
#ifdef ENABLE_BF16
template class GeluFfnLayer<__nv_bfloat16>;
#endif

template<typename T>
ReluFfnLayer<T>::ReluFfnLayer(size_t           max_batch_size,
                              size_t           max_seq_len,
                              size_t           head_num,
                              size_t           size_per_head,
                              size_t           inter_size,
                              cudaStream_t     stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator*      allocator,
                              bool             is_free_buffer_after_forward,
                              bool             sparse,
                              bool             use_gated_activation):
    FfnLayer<T>(max_batch_size,
                max_seq_len,
                head_num,
                size_per_head,
                inter_size,
                stream,
                cublas_wrapper,
                allocator,
                is_free_buffer_after_forward,
                sparse,
                0,
                use_gated_activation)
{
}

template<typename T>
ReluFfnLayer<T>::ReluFfnLayer(ReluFfnLayer<T> const& relu_ffn_layer): FfnLayer<T>(relu_ffn_layer)
{
}

template<typename T>
void ReluFfnLayer<T>::invokeAddBiasActivation(const int m, const T* bias)
{
    invokeAddBiasRelu<T>(inter_buf_, bias, m, inter_size_, stream_);
}

template<typename T>
void ReluFfnLayer<T>::invokeAddBiasGatedActivation(const int m, const T* bias1, const T* bias2)
{
    invokeAddBiasGatedRelu<T>(inter_buf_, inter_buf_2_, bias1, bias2, m, inter_size_, stream_);
}

template class ReluFfnLayer<float>;
template class ReluFfnLayer<half>;
#ifdef ENABLE_BF16
template class ReluFfnLayer<__nv_bfloat16>;
#endif

template<typename T>
SiluFfnLayer<T>::SiluFfnLayer(size_t           max_batch_size,
                              size_t           max_seq_len,
                              size_t           head_num,
                              size_t           size_per_head,
                              size_t           inter_size,
                              cudaStream_t     stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator*      allocator,
                              bool             is_free_buffer_after_forward,
                              bool             sparse,
                              bool             use_gated_activation):
    FfnLayer<T>(max_batch_size,
                max_seq_len,
                head_num,
                size_per_head,
                inter_size,
                stream,
                cublas_wrapper,
                allocator,
                is_free_buffer_after_forward,
                sparse,
                0,
                use_gated_activation)
{
}

template<typename T>
SiluFfnLayer<T>::SiluFfnLayer(SiluFfnLayer<T> const& gelu_ffn_layer): FfnLayer<T>(gelu_ffn_layer)
{
}

template<typename T>
void SiluFfnLayer<T>::invokeAddBiasActivation(const int m, const T* bias)
{
    invokeAddBiasSilu<T>(inter_buf_, bias, m, inter_size_, stream_);
}

template<typename T>
void SiluFfnLayer<T>::invokeAddBiasGatedActivation(const int m, const T* bias1, const T* bias2)
{
    invokeAddBiasGatedSilu<T>(inter_buf_, inter_buf_2_, bias1, bias2, m, inter_size_, stream_);
}

template class SiluFfnLayer<float>;
template class SiluFfnLayer<half>;
#ifdef ENABLE_BF16
template class SiluFfnLayer<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
