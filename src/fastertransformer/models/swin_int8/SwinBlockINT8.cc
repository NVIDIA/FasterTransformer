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

#include "src/fastertransformer/models/swin_int8/SwinBlockINT8.h"

namespace fastertransformer {

template<typename T>
void SwinTransformerINT8Block<T>::allocateBuffer()
{
    assert(false && "SwinTransformerINT8Block<T>::allocateBuffer() is not implemented\n");
}

template<typename T>
void SwinTransformerINT8Block<T>::allocateBuffer(int batch, int input_resolution, int dim)
{
    if (is_allocate_buffer_ == false) {
        attention_output_ = (int8_t*)allocator_->reMalloc(
            attention_output_, batch * input_resolution * input_resolution * dim * sizeof(int8_t), false);
        skip_buf_ = (int8_t*)allocator_->reMalloc(
            skip_buf_, batch * input_resolution * input_resolution * dim * sizeof(int8_t), false);
        mlp_buf_ = (int8_t*)allocator_->reMalloc(
            mlp_buf_, batch * input_resolution * input_resolution * int(dim * mlp_ratio_) * sizeof(int8_t), false);
        mlp_output_ = (int8_t*)allocator_->reMalloc(
            mlp_output_, batch * input_resolution * input_resolution * dim * sizeof(int32_t), false);

        normed_shifted_input_ = mlp_buf_;
        is_allocate_buffer_   = true;
    }
}

template<typename T>
void SwinTransformerINT8Block<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free((void**)(&attention_output_));
        allocator_->free((void**)(&skip_buf_));
        allocator_->free((void**)(&mlp_buf_));
        allocator_->free((void**)(&mlp_output_));
        is_allocate_buffer_ = false;
    }
}

template<typename T>
SwinTransformerINT8Block<T>::SwinTransformerINT8Block(int              int8_mode,
                                                      int              max_batch,
                                                      int              window_size,
                                                      float            mlp_ratio,
                                                      float            layernorm_eps,
                                                      bool             qkv_bias,
                                                      cudaStream_t     stream,
                                                      cublasMMWrapper* cublas_wrapper,
                                                      IAllocator*      allocator,
                                                      bool             is_free_buffer_after_forward,
                                                      float            qk_scale,
                                                      int              version):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    int8_mode(int8_mode),
    window_size_(window_size),
    mlp_ratio_(mlp_ratio),
    layernorm_eps_(layernorm_eps),
    version_(version)
{
    atten_ = new WindowAttentionINT8<T>(max_batch,
                                        window_size,
                                        stream,
                                        cublas_wrapper,
                                        allocator,
                                        is_free_buffer_after_forward,
                                        qkv_bias,
                                        qk_scale,
                                        version);
}

template<typename T>
SwinTransformerINT8Block<T>::~SwinTransformerINT8Block()
{
    if (atten_ != nullptr) {
        delete atten_;
        atten_ = nullptr;
    }
}

template<typename T>
void SwinTransformerINT8Block<T>::forward(TensorMap*                         output_tensors,
                                          TensorMap*                         input_tensors,
                                          SwinTransformerINT8BlockWeight<T>& swin_block_weights)
{
    // input_tensors:
    //      input [batch, input_resolution, input_resolution, dim]
    //      attention_mask [window_num, window_len, window_len]
    //      trt_attention_mask  [window_num, trt_window_len, trt_window_len]
    //      additional_params [5] {number_of_head, shift_size, sm, basic_layer_id, block_id}
    // output_tensors:
    //      output [batch, input_resolution, input_resolution, dim]

    cublasINT8MMWrapper* cublas_wrapper     = (cublasINT8MMWrapper*)cublas_wrapper_;
    T*                   out_tensor         = output_tensors->getPtr<T>("hidden_features");
    T*                   from_tensor        = input_tensors->getPtr<T>("input_query");
    const int            batch              = input_tensors->at("input_query").shape[0];
    const int            input_resolution   = input_tensors->at("input_query").shape[1];
    const int            dim                = input_tensors->at("input_query").shape[3];
    T*                   attention_mask     = input_tensors->getPtr<T>("attention_mask", nullptr);
    T*                   trt_attention_mask = input_tensors->getPtr<T>("trt_attention_mask", nullptr);
    const int*           additional_params  = input_tensors->getPtr<const int>("additional_params");
    const int            num_head           = additional_params[0];
    int                  shift_size         = additional_params[1];
    const int            sm                 = additional_params[2];
    const int            basic_layer_id     = additional_params[3];
    const int            block_id           = additional_params[4];

    int window_size_in_use = (input_resolution <= window_size_) ? input_resolution : window_size_;
    int window_len_in_use  = window_size_in_use * window_size_in_use;
    shift_size             = (input_resolution <= window_size_) ? 0 : shift_size;
    int window_num         = (input_resolution / window_size_in_use) * (input_resolution / window_size_in_use);
    int mlp_dim            = int(mlp_ratio_ * dim);
    allocateBuffer(batch, input_resolution, dim);

    const ScaleList* scalePtr = &(swin_block_weights.scalelist);
    if (version_ == 2) {
        invokeShiftPartitionCol32(normed_shifted_input_,
                                  from_tensor,
                                  batch,
                                  input_resolution,
                                  input_resolution,
                                  dim,
                                  &(scalePtr->d_scale_list_[0 + 3]),
                                  shift_size,
                                  window_size_in_use,
                                  stream_);
    }
    else if (version_ == 1) {
        invokeLayernormShiftPartitionCol32(normed_shifted_input_,
                                           (const half*)from_tensor,
                                           (const half*)swin_block_weights.attn_layernorm_weights.gamma,
                                           (const half*)swin_block_weights.attn_layernorm_weights.beta,
                                           batch,
                                           input_resolution,
                                           input_resolution,
                                           dim,
                                           scalePtr->h_scale_list_[0 + 3],
                                           shift_size,
                                           window_size_in_use,
                                           stream_);
    }

    const size_t m                        = batch * input_resolution * input_resolution;
    const size_t n                        = dim;
    DataType     data_type                = getTensorType<T>();
    int          additional_parameters[9] = {
                 batch, dim, input_resolution, num_head, shift_size, sm, window_size_in_use, basic_layer_id, block_id};
    TensorMap attn_output_tensors{
        {"hidden_features", Tensor{MEMORY_GPU, TYPE_INT8, std::vector<size_t>{m, n}, attention_output_}}};
    TensorMap attn_input_tensors{
        {"input_query", Tensor{MEMORY_GPU, TYPE_INT8, std::vector<size_t>{m, n}, normed_shifted_input_}},
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
                data_type,
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
        {"additional_params", Tensor{MEMORY_CPU, TYPE_INT8, std::vector<size_t>{9}, additional_parameters}}};

    swin_block_weights.attention_weights.scale_list_ptr = &(swin_block_weights.scalelist);
    atten_->forward(&attn_output_tensors, &attn_input_tensors, &swin_block_weights.attention_weights);
    sync_check_cuda_error();

    if (version_ == 2) {
        invokeAddBiasLayernormAddResCol32((int8_t*)attention_output_,
                                          (int8_t*)attention_output_,
                                          from_tensor,
                                          swin_block_weights.attention_weights.attention_output_weight.bias,
                                          swin_block_weights.attn_layernorm_weights.gamma,
                                          swin_block_weights.attn_layernorm_weights.beta,
                                          layernorm_eps_,
                                          m,
                                          dim,
                                          stream_,
                                          &(scalePtr->d_scale_list_[12 + 1]),
                                          &(scalePtr->d_scale_list_[36 + 3]));
    }
    else if (version_ == 1) {
        invokeAddBiasResidualPreLayerNormCol32(
            (int8_t*)attention_output_,
            (int8_t*)attention_output_,
            (__half*)from_tensor,
            (const __half*)swin_block_weights.attention_weights.attention_output_weight.bias,
            (const __half*)swin_block_weights.ffn_layernorm_weights.gamma,
            (const __half*)swin_block_weights.ffn_layernorm_weights.beta,
            m,
            dim,
            stream_,
            (scalePtr->h_scale_list_[12 + 1]),
            (scalePtr->h_scale_list_[36 + 3]));
    }

    cublas_wrapper->Gemm((int8_t*)mlp_buf_,
                         1,
                         m,
                         mlp_dim,
                         dim,
                         0,
                         0,
                         0,
                         scalePtr->h_scale_list_[scalePtr->p3_offset_ + 2],
                         (int8_t*)attention_output_,
                         (int8_t*)swin_block_weights.ffn_weights.intermediate_weight.kernel);

    invokeAddBiasGeluCol32_v2(mlp_buf_,
                              swin_block_weights.ffn_weights.intermediate_weight.bias,
                              m,
                              mlp_dim,
                              stream_,
                              &(scalePtr->d_scale_list_[40 + 1]),
                              &(scalePtr->d_scale_list_[44 + 3]));

    if (version_ == 2) {
        cublas_wrapper->Gemm((int8_t*)mlp_output_,
                             1,
                             m,
                             dim,
                             mlp_dim,
                             0,
                             0,
                             0,
                             scalePtr->h_scale_list_[scalePtr->p3_offset_ + 3],
                             (int8_t*)mlp_buf_,
                             (int8_t*)swin_block_weights.ffn_weights.output_weight.kernel);

        invokeAddBiasLayernormAddResCol32((int8_t*)out_tensor,
                                          (int8_t*)mlp_output_,
                                          from_tensor,
                                          swin_block_weights.ffn_weights.output_weight.bias,
                                          swin_block_weights.ffn_layernorm_weights.gamma,
                                          swin_block_weights.ffn_layernorm_weights.beta,
                                          layernorm_eps_,
                                          m,
                                          dim,
                                          stream_,
                                          &(scalePtr->d_scale_list_[48 + 1]));
    }
    else if (int8_mode == 1) {
        cublas_wrapper->Gemm((int8_t*)mlp_output_,
                             1,
                             m,
                             dim,
                             mlp_dim,
                             0,
                             0,
                             0,
                             scalePtr->h_scale_list_[scalePtr->p3_offset_ + 3],
                             (int8_t*)mlp_buf_,
                             (int8_t*)swin_block_weights.ffn_weights.output_weight.kernel);
        invokeAddBiasResidualCol32(out_tensor,
                                   (int8_t*)mlp_output_,
                                   from_tensor,
                                   swin_block_weights.ffn_weights.output_weight.bias,
                                   m,
                                   dim,
                                   stream_,
                                   &(scalePtr->d_scale_list_[48 + 1]));
    }
    else {
        cublas_wrapper->Gemm((int*)mlp_output_,
                             1,
                             m,
                             dim,
                             mlp_dim,
                             0,
                             0,
                             0,
                             (int8_t*)mlp_buf_,
                             (int8_t*)swin_block_weights.ffn_weights.output_weight.kernel);
        invokeAddBiasResidualCol32(out_tensor,
                                   (int*)mlp_output_,
                                   from_tensor,
                                   swin_block_weights.ffn_weights.output_weight.bias,
                                   m,
                                   dim,
                                   stream_,
                                   &scalePtr->d_scale_list_[scalePtr->p2_offset_ + 3],
                                   &scalePtr->d_scale_list_[44 + 0]);
    }

    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
}

template class SwinTransformerINT8Block<float>;
template class SwinTransformerINT8Block<half>;

}  // namespace fastertransformer