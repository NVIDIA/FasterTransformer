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

#include "SwinBlock.h"

namespace fastertransformer {

template<typename T>
void SwinTransformerBlock<T>::allocateBuffer()
{
    assert(false && "SwinTransformerBlock<T>::allocateBuffer() is not implemented\n");
}

template<typename T>
void SwinTransformerBlock<T>::allocateBuffer(int batch, int input_resolution, int dim)
{
    if (is_allocate_buffer_ == false) {
        attention_output_ = (T*)allocator_->reMalloc(
            attention_output_, batch * input_resolution * input_resolution * dim * sizeof(T), false);
        normed_attn_out_buf_ =
            (version_ == 1) ? (T*)allocator_->reMalloc(
                normed_attn_out_buf_, batch * input_resolution * input_resolution * dim * sizeof(T), false) :
                              nullptr;
        mlp_buf_ = (T*)allocator_->reMalloc(
            mlp_buf_, batch * input_resolution * input_resolution * int(dim * mlp_ratio_) * sizeof(T), false);

        normed_shifted_input_ = mlp_buf_;
        is_allocate_buffer_   = true;
    }
}

template<typename T>
void SwinTransformerBlock<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free((void**)(&attention_output_));
        allocator_->free((void**)(&normed_attn_out_buf_));
        allocator_->free((void**)(&mlp_buf_));
        is_allocate_buffer_ = false;
    }
}

template<typename T>
SwinTransformerBlock<T>::SwinTransformerBlock(int              max_batch,
                                              int              window_size,
                                              float            mlp_ratio,
                                              float            layernorm_eps,
                                              cudaStream_t     stream,
                                              cublasMMWrapper* cublas_wrapper,
                                              IAllocator*      allocator,
                                              bool             is_free_buffer_after_forward,
                                              bool             qkv_bias,
                                              float            qk_scale,
                                              int              version):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_(max_batch),
    window_size_(window_size),
    mlp_ratio_(mlp_ratio),
    layernorm_eps_(layernorm_eps),
    qkv_bias_(qkv_bias),
    qk_scale_(qk_scale),
    version_(version)
{
    atten_ = new WindowAttention<T>(max_batch_,
                                    window_size_,
                                    stream,
                                    cublas_wrapper,
                                    allocator,
                                    is_free_buffer_after_forward,
                                    qkv_bias_,
                                    qk_scale_,
                                    version_);
}

template<typename T>
SwinTransformerBlock<T>::~SwinTransformerBlock()
{
    if (atten_ != nullptr) {
        delete atten_;
        atten_ = nullptr;
    }
}

template<typename T>
void SwinTransformerBlock<T>::forward(TensorMap*                     output_tensors,
                                      TensorMap*                     input_tensors,
                                      SwinTransformerBlockWeight<T>& swin_block_weights)
{
    // input_tensors:
    //      input_query [batch, input_resolution, input_resolution, dim]
    //      attention_mask [window_num, window_len, window_len]
    //      trt_attention_mask  [window_num, trt_window_len, trt_window_len]
    //      additional_params [3] {number_of_head, shift_size, sm}
    // output_tensors:
    //      hidden_features [batch, input_resolution, input_resolution, dim]

    T*         output             = output_tensors->getPtr<T>("hidden_features");
    T*         input              = input_tensors->getPtr<T>("input_query");
    int        batch              = input_tensors->at("input_query").shape[0];
    int        input_resolution   = input_tensors->at("input_query").shape[1];
    int        dim                = input_tensors->at("input_query").shape[3];
    T*         attention_mask     = input_tensors->getPtr<T>("attention_mask", nullptr);
    T*         trt_attention_mask = input_tensors->getPtr<T>("trt_attention_mask", nullptr);
    const int* additional_params  = input_tensors->getPtr<const int>("additional_params");
    const int  num_head           = additional_params[0];
    int        shift_size         = additional_params[1];
    const int  sm                 = additional_params[2];
    FT_CHECK(input_resolution == input_tensors->at("input_query").shape[2]);

    int window_size_in_use = (input_resolution <= window_size_) ? input_resolution : window_size_;
    int window_len_in_use  = window_size_in_use * window_size_in_use;
    shift_size             = (input_resolution <= window_size_) ? 0 : shift_size;
    int window_num         = (input_resolution / window_size_in_use) * (input_resolution / window_size_in_use);
    int mlp_dim            = int(mlp_ratio_ * dim);
    allocateBuffer(batch, input_resolution, dim);

    if (version_ == 2) {
        invokeShiftPartition(normed_shifted_input_,
                             input,
                             batch,
                             input_resolution,
                             input_resolution,
                             dim,
                             shift_size,
                             window_size_in_use,
                             stream_);
    }
    else if (version_ == 1) {
        invokeLayernormShiftPartition(normed_shifted_input_,
                                      input,
                                      swin_block_weights.attn_layernorm_weights.gamma,
                                      swin_block_weights.attn_layernorm_weights.beta,
                                      layernorm_eps_,
                                      batch,
                                      input_resolution,
                                      input_resolution,
                                      dim,
                                      shift_size,
                                      window_size_in_use,
                                      stream_);
    }

    const size_t m                     = batch * input_resolution * input_resolution;
    const size_t n                     = dim;
    DataType     data_type             = getTensorType<T>();
    int       additional_parameters[7] = {batch, dim, input_resolution, num_head, shift_size, sm, window_size_in_use};
    TensorMap attn_output_tensors{
        {"hidden_features", Tensor{MEMORY_GPU, data_type, std::vector<size_t>{m, n}, attention_output_}}};
    TensorMap attn_input_tensors{
        {"input_query", Tensor{MEMORY_GPU, data_type, std::vector<size_t>{m, n}, normed_shifted_input_}},
        {"attention_mask",
         Tensor{MEMORY_GPU,
                data_type,
                std::vector<size_t>{(size_t)window_num, (size_t)window_len_in_use, (size_t)window_len_in_use},
                attention_mask}},
        {"trt_attention_mask",
         Tensor{MEMORY_GPU,
                data_type,
                std::vector<size_t>{(size_t)window_num, (size_t)window_len_in_use, (size_t)window_len_in_use},
                trt_attention_mask}},
        {"attention_relative_position_bias",
         Tensor{MEMORY_GPU,
                TYPE_FP16,
                std::vector<size_t>{(size_t)num_head, (size_t)window_len_in_use, (size_t)window_len_in_use},
                swin_block_weights.attention_relative_pos_bias}},
        {"trt_relative_position_bias",
         Tensor{MEMORY_GPU,
                data_type,
                std::vector<size_t>{(size_t)num_head, (size_t)window_len_in_use, (size_t)window_len_in_use},
                swin_block_weights.trt_relative_position_bias}},
        {"attn_logit_scale",
         Tensor{
             MEMORY_GPU, data_type, std::vector<size_t>{(size_t)num_head}, swin_block_weights.attention_logit_scale}},
        {"additional_params", Tensor{MEMORY_CPU, TYPE_INT32, {7}, additional_parameters}}};

    atten_->forward(&attn_output_tensors, &attn_input_tensors, &swin_block_weights.attention_weights);

    if (version_ == 2) {
        invokeAddBiasLayernormAddRes(attention_output_,
                                     input,
                                     swin_block_weights.attention_weights.attention_output_weight.bias,
                                     swin_block_weights.attn_layernorm_weights.gamma,
                                     swin_block_weights.attn_layernorm_weights.beta,
                                     layernorm_eps_,
                                     m,
                                     n,
                                     stream_);
    }
    else if (version_ == 1) {
        invokeGeneralAddBiasResidualPreLayerNorm(attention_output_,
                                                 normed_attn_out_buf_,
                                                 attention_output_,
                                                 input,
                                                 swin_block_weights.ffn_layernorm_weights.gamma,
                                                 swin_block_weights.ffn_layernorm_weights.beta,
                                                 swin_block_weights.attention_weights.attention_output_weight.bias,
                                                 layernorm_eps_,
                                                 batch * input_resolution * input_resolution,
                                                 dim,
                                                 (float*)nullptr,
                                                 (float*)nullptr,
                                                 (float*)nullptr,
                                                 0,
                                                 stream_);
    }

    // MLP
    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                          CUBLAS_OP_N,
                          mlp_dim,
                          batch * input_resolution * input_resolution,
                          dim,
                          swin_block_weights.ffn_weights.intermediate_weight.kernel,
                          dim,
                          version_ == 2 ? attention_output_ : normed_attn_out_buf_,
                          dim,
                          mlp_buf_,
                          mlp_dim);

    invokeAddBiasGeluV2(mlp_buf_,
                        swin_block_weights.ffn_weights.intermediate_weight.bias,
                        (const int*)nullptr,
                        (const T*)nullptr,
                        batch * input_resolution * input_resolution,
                        mlp_dim,
                        stream_);

    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                          CUBLAS_OP_N,
                          dim,
                          batch * input_resolution * input_resolution,
                          mlp_dim,
                          swin_block_weights.ffn_weights.output_weight.kernel,
                          mlp_dim,
                          mlp_buf_,
                          mlp_dim,
                          output,
                          dim);

    if (version_ == 2) {
        invokeAddBiasLayernormAddRes(output,
                                     attention_output_,
                                     swin_block_weights.ffn_weights.output_weight.bias,
                                     swin_block_weights.ffn_layernorm_weights.gamma,
                                     swin_block_weights.ffn_layernorm_weights.beta,
                                     layernorm_eps_,
                                     batch * input_resolution * input_resolution,
                                     dim,
                                     stream_);
    }
    else {
        invokeAddBiasResidual(output,
                              attention_output_,
                              swin_block_weights.ffn_weights.output_weight.bias,
                              batch * input_resolution * input_resolution,
                              dim,
                              stream_);
    }
    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
}

template class SwinTransformerBlock<float>;
template class SwinTransformerBlock<half>;
}  // namespace fastertransformer
