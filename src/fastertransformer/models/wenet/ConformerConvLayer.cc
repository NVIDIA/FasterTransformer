/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2022.  Authored by Yuqing Ding.
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

#include "src/fastertransformer/models/wenet/ConformerConvLayer.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/models/wenet/WenetKernels.h"

namespace fastertransformer {

#ifdef ENABLE_BF16
/*
template void invokeRemovePadding(__nv_bfloat16* dst,
                                  const __nv_bfloat16* src,
                                  const int* padding_offset,
                                  const int token_num,
                                  const int hidden_dim,
                                  cudaStream_t stream);

template void invokeRebuildPadding(__nv_bfloat16* dst,
                                   const __nv_bfloat16* src,
                                   const int* padding_offset,
                                   const int token_num,
                                   const int hidden_dim,
                                   cudaStream_t stream);
*/
#endif
namespace {
// ugly implementation
template<typename T>
struct PaddingType {
    using Type = T;
};
#ifdef ENABLE_BF16
template<>
struct PaddingType<__nv_bfloat16> {
    using Type = half;
};
#endif
}  // namespace

template<typename T>
void ConformerConvLayer<T>::forward(std::vector<Tensor>*          output_tensors,
                                    const std::vector<Tensor>*    input_tensors,
                                    const ConformerConvWeight<T>* conformer_conv_weights)
{
    TensorMap input_tensors_map = TensorMap({{"input_tensor", input_tensors->at(0)},
                                             {"attention_mask", input_tensors->at(1)},
                                             {"padding_offset", input_tensors->at(2)},
                                             {"bid_start_end", input_tensors->at(3)}});

    TensorMap output_tensors_map = TensorMap({
        {"output_tensor", output_tensors->at(0)},
    });

    forward(&output_tensors_map, &input_tensors_map, conformer_conv_weights);
}

template<typename T>
void ConformerConvLayer<T>::forward(TensorMap*                    output_tensors,
                                    TensorMap*                    input_tensors,
                                    const ConformerConvWeight<T>* conformer_conv_weights)
{
    // input tensors:
    //      input_tensor [batch_size, seq_len, hidden_dimension],
    //      attention_mask (batch_size, 1, seqlen, seqlen),
    //      padding_offset (h_var_token_num),
    //      bid_start_end  (h_var_token_num * 3)

    // output tensors:
    //      output_tensor [batch_size, seq_len hidden_dimension],

    bool use_varlen = false;

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() == 4);
    FT_CHECK(output_tensors->size() == 1);
    const int batch_size = input_tensors->at("input_tensor").shape[0];
    const int seq_len    = input_tensors->at("input_tensor").shape[1];
    int       m          = batch_size * seq_len;
    allocateBuffer(m);

    const T* input_tensor   = (const T*)input_tensors->at("input_tensor").data;
    const T* attr_mask_data = (const T*)input_tensors->at("attention_mask").data;
    T*       output_tensor  = (T*)output_tensors->at("output_tensor").data;

    const int* padding_offset = (const int*)input_tensors->at("padding_offset").data;
    const int* bid_start_end  = (const int*)input_tensors->at("bid_start_end").data;
    if (use_varlen) {
        FT_CHECK(false);
        //     m = input_tensors->at("padding_offset").shape[0];
        //     invokeRemovePadding((typename PaddingType<T>::Type*)input_remove_padding_,
        //                         (const typename PaddingType<T>::Type*)input_tensors->at("input_tensor").data,
        //                         padding_offset,
        //                         m,
        //                         head_num_ * size_per_head_,
        //                         stream_);
        //     sync_check_cuda_error();
        //     input_tensor = input_remove_padding_;
    }

#ifdef SPARSITY_ENABLED
    int m_tmp = m;
    if (m_tmp % 8 != 0) {
        m_tmp = (m_tmp / 8 + 1) * 8;
    }
    const int m_padded = m_tmp;
    if (sparse_ && cublas_wrapper_->isUseSparse(1, hidden_units_ * 2, m, hidden_units_)) {
        cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                hidden_units_ * 2,
                                m_padded,
                                hidden_units_,
                                conformer_conv_weights->pointwise_conv1_weight.sp_kernel,
                                input_tensor,
                                inter2_buf_);
    }
    else {
#endif

        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              hidden_units_ * 2,
                              m,
                              hidden_units_,
                              conformer_conv_weights->pointwise_conv1_weight.kernel,
                              hidden_units_ * 2,
                              input_tensor,
                              hidden_units_,
                              inter2_buf_,
                              hidden_units_ * 2);

#ifdef SPARSITY_ENABLED
    }
#endif
    // inter2_buf_ -> inter_buf_
    if (use_varlen) {
        FT_CHECK(false);
        // invokeBiasGlu(inter_buf_,  // input_remove_padding_,
        //               inter2_buf_,
        //               conformer_conv_weights->pointwise_conv1_weight.bias,
        //               m,
        //               hidden_units_,
        //               stream_);
        // sync_check_cuda_error();
    }
    else {
        invokeMaskBiasGlu(inter_buf_,
                          inter2_buf_,
                          conformer_conv_weights->pointwise_conv1_weight.bias,
                          m,
                          hidden_units_,
                          attr_mask_data,
                          seq_len,
                          stream_);
    }
    sync_check_cuda_error();

    // inter_buf_ -> inter2_buf_
    if (use_varlen) {
        FT_CHECK(false);
        // invokeVarLenConformerDepthwiseConvBiasSilu(inter2_buf_,
        //                                            inter_buf_,
        //                                            conformer_conv_weights->depthwise_conv_weight.kernel,
        //                                            conformer_conv_weights->depthwise_conv_weight.bias,
        //                                            bid_start_end,
        //                                            conformer_conv_weights->pointwise_conv1_weight.bias,
        //                                            m,
        //                                            batch_size,
        //                                            seq_len,
        //                                            hidden_units_,
        //                                            conv_module_kernel_size_,
        //                                            conv_module_kernel_size_ / 2,
        //                                            stream_);
    }
    else {
        if (use_layernorm_in_conv_module_) {
            invokeConformerDepthwiseConvBias(inter2_buf_,
                                             inter_buf_,
                                             conformer_conv_weights->depthwise_conv_weight.kernel,
                                             conformer_conv_weights->depthwise_conv_weight.bias,
                                             batch_size,
                                             seq_len,
                                             hidden_units_,
                                             conv_module_kernel_size_,
                                             conv_module_kernel_size_ / 2,
                                             stream_);
        }
        else {
            invokeConformerDepthwiseConvBiasSilu(inter2_buf_,
                                                 inter_buf_,
                                                 conformer_conv_weights->depthwise_conv_weight.kernel,
                                                 conformer_conv_weights->depthwise_conv_weight.bias,
                                                 batch_size,
                                                 seq_len,
                                                 hidden_units_,
                                                 conv_module_kernel_size_,
                                                 conv_module_kernel_size_ / 2,
                                                 stream_);
        }
    }

    // inter_buf2_ -> inter_buf_
    if (use_layernorm_in_conv_module_) {
        invokeGeneralLayerNorm(normed_inter_buf_,
                               inter2_buf_,
                               conformer_conv_weights->norm_weights.gamma,
                               conformer_conv_weights->norm_weights.beta,
                               1e-6f,
                               batch_size * seq_len,
                               hidden_units_,
                               (float*)nullptr,
                               0,
                               stream_);
        invokeGenericActivation<SiluActivation, T, T>(normed_inter_buf_,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      batch_size * seq_len,
                                                      hidden_units_,
                                                      0,
                                                      nullptr,
                                                      nullptr,
                                                      stream_);
        sync_check_cuda_error();
    }

#ifdef SPARSITY_ENABLED
    if (sparse_ && cublas_wrapper_->isUseSparse(1, hidden_units_, m, hidden_units_)) {
        if (use_layernorm_in_conv_module_) {
            cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    hidden_units_,
                                    m_padded,
                                    hidden_units_,
                                    conformer_conv_weights->pointwise_conv2_weight.sp_kernel,
                                    normed_inter_buf_,
                                    output_tensor);
        }
        else {
            cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    hidden_units_,
                                    m_padded,
                                    hidden_units_,
                                    conformer_conv_weights->pointwise_conv2_weight.sp_kernel,
                                    inter2_buf_,
                                    inter_buf_);
        }
    }
    else {
#endif
        if (use_layernorm_in_conv_module_) {
            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  hidden_units_,
                                  m,
                                  hidden_units_,
                                  conformer_conv_weights->pointwise_conv2_weight.kernel,
                                  hidden_units_,
                                  normed_inter_buf_,
                                  hidden_units_,
                                  inter_buf_,
                                  hidden_units_);
        }
        else {
            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  hidden_units_,
                                  m,
                                  hidden_units_,
                                  conformer_conv_weights->pointwise_conv2_weight.kernel,
                                  hidden_units_,
                                  inter2_buf_,
                                  hidden_units_,
                                  inter_buf_,
                                  hidden_units_);
        }

        if (use_varlen) {
            FT_CHECK(false);
            // cudaMemsetAsync(output_tensor, 0, batch_size * seq_len * hidden_units_ * sizeof(T), stream_);

            // invokeBiasRebuildPadding(output_tensor,
            //                          inter_buf_,
            //                          conformer_conv_weights->pointwise_conv2_weight.bias,
            //                          padding_offset,
            //                          m,
            //                          hidden_units_,
            //                          stream_);

            // m = batch_size * seq_len;
        }
        else {
            invokeMaskBias(output_tensor,
                           inter_buf_,
                           conformer_conv_weights->pointwise_conv2_weight.bias,
                           m,
                           hidden_units_,
                           attr_mask_data,
                           seq_len,
                           stream_);
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
ConformerConvLayer<T>::ConformerConvLayer(size_t           max_batch_size,
                                          size_t           max_seq_len,
                                          size_t           head_num,
                                          size_t           size_per_head,
                                          size_t           conv_module_kernel_size,
                                          cudaStream_t     stream,
                                          cublasMMWrapper* cublas_wrapper,
                                          IAllocator*      allocator,
                                          bool             is_free_buffer_after_forward,
                                          bool             sparse,
                                          int              int8_mode,
                                          bool             use_layernorm_in_conv_module):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    head_num_(head_num),
    size_per_head_(size_per_head),
    conv_module_kernel_size_(conv_module_kernel_size),
    hidden_units_(head_num * size_per_head),
    int8_mode_(int8_mode),
    use_layernorm_in_conv_module_(use_layernorm_in_conv_module)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
ConformerConvLayer<T>::ConformerConvLayer(ConformerConvLayer<T> const& conformer_conv_layer):
    BaseLayer(conformer_conv_layer.stream_,
              conformer_conv_layer.cublas_wrapper_,
              conformer_conv_layer.allocator_,
              conformer_conv_layer.is_free_buffer_after_forward_,
              conformer_conv_layer.cuda_device_prop_,
              conformer_conv_layer.sparse_),
    head_num_(conformer_conv_layer.head_num_),
    size_per_head_(conformer_conv_layer.size_per_head_),
    conv_module_kernel_size_(conformer_conv_layer.conv_module_kernel_size_),
    hidden_units_(conformer_conv_layer.hidden_units_),
    int8_mode_(conformer_conv_layer.int8_mode_),
    use_layernorm_in_conv_module_(conformer_conv_layer.use_layernorm_in_conv_module_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
ConformerConvLayer<T>::~ConformerConvLayer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void ConformerConvLayer<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void ConformerConvLayer<T>::allocateBuffer(size_t token_num)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    input_remove_padding_ =
        (T*)allocator_->reMalloc(input_remove_padding_, sizeof(T) * token_num * hidden_units_, false);
    inter_buf_  = (T*)allocator_->reMalloc(inter_buf_, sizeof(T) * token_num * hidden_units_, false);
    inter2_buf_ = (T*)allocator_->reMalloc(inter2_buf_, sizeof(T) * token_num * hidden_units_ * 2, false);
    if (use_layernorm_in_conv_module_) {
        normed_inter_buf_ = (T*)allocator_->reMalloc(normed_inter_buf_, sizeof(T) * token_num * hidden_units_, false);
    }
}

template<typename T>
void ConformerConvLayer<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    allocator_->free((void**)(&input_remove_padding_));
    allocator_->free((void**)(&inter_buf_));
    allocator_->free((void**)(&inter2_buf_));
    if (use_layernorm_in_conv_module_) {
        allocator_->free((void**)(&normed_inter_buf_));
    }
}

template class ConformerConvLayer<float>;
template class ConformerConvLayer<half>;
#ifdef ENABLE_BF16
template class ConformerConvLayer<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
