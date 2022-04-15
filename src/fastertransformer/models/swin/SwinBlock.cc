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
    if (is_allocate_buffer_ == false) {
        attention_output_ =
            (T*)allocator_->malloc(max_batch_ * window_num_ * window_len_ * embed_dim_ * sizeof(T), false);
        normed_attn_out_buf_ =
            (T*)allocator_->malloc(max_batch_ * window_num_ * window_len_ * embed_dim_ * sizeof(T), false);
        mlp_buf_ = (T*)allocator_->malloc(
            max_batch_ * window_num_ * window_len_ * int(embed_dim_ * mlp_ratio_) * sizeof(T), false);

        normed_shifted_input_ = mlp_buf_;
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void SwinTransformerBlock<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free(attention_output_);
        allocator_->free(normed_attn_out_buf_);
        allocator_->free(mlp_buf_);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
SwinTransformerBlock<T>::SwinTransformerBlock(int max_batch,
                                              int window_size,
                                              float mlp_ratio,
                                              cudaStream_t stream,
                                              cublasMMWrapper* cublas_wrapper,
                                              IAllocator* allocator,
                                              bool is_free_buffer_after_forward,
                                              bool qkv_bias,
                                              float qk_scale):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_(max_batch),
    window_size_(window_size),
    mlp_ratio_(mlp_ratio),
    qkv_bias_(qkv_bias),
    qk_scale_(qk_scale)
{
    window_len_ = window_size_ * window_size_;
    atten_ = new WindowAttention<T>(max_batch_,
                                    window_size_,
                                    stream,
                                    cublas_wrapper,
                                    allocator,
                                    is_free_buffer_after_forward,
                                    qkv_bias_,
                                    qk_scale_);
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
void SwinTransformerBlock<T>::forward(std::vector<Tensor>* output_tensors,
                                      const std::vector<Tensor>* input_tensors,
                                      SwinTransformerBlockWeight<T>& swin_block_weights)
{
    // input_tensors:
    //      input [batch, input_resolution, input_resolution, dim]
    //      attention_mask [window_num, window_len, window_len]
    //      input_parameters [3] {number_of_head, shift_size, sm}
    // output_tensors:
    //      output [batch, input_resolution, input_resolution, dim]

    T* input = (T*)input_tensors->at(0).data;
    T* output = (T*)(output_tensors->at(0).data);
    const int batch = input_tensors->at(0).shape[0];
    const int input_resolution = input_tensors->at(0).shape[1];
    assert(input_resolution == input_tensors->at(0).shape[2]);
    const int dim = input_tensors->at(0).shape[3];
    const int* input_parameters = (const int*)input_tensors->at(2).data;
    const int num_head = input_parameters[0];
    int shift_size = input_parameters[1];
    const int sm = input_parameters[2];

    shift_size = (input_resolution <= window_size_) ? 0 : shift_size;
    int window_num = (input_resolution / window_size_) * (input_resolution / window_size_);
    window_num_ = window_num;
    embed_dim_ = dim;
    int mlp_dim = int(mlp_ratio_ * dim);
    allocateBuffer();

    invokeLayernormShiftPartition(normed_shifted_input_,
                                  input,
                                  swin_block_weights.attn_layernorm_weights.gamma,
                                  swin_block_weights.attn_layernorm_weights.beta,
                                  batch,
                                  input_resolution,
                                  input_resolution,
                                  dim,
                                  shift_size,
                                  window_size_,
                                  stream_);

    const size_t m = batch * input_resolution * input_resolution;
    const size_t n = dim;
    DataType data_type = getTensorType<T>();
    int additional_parameters[6] = {batch, dim, input_resolution, num_head, shift_size, sm};
    std::vector<Tensor> attn_output_tensors{
        Tensor{MEMORY_GPU, data_type, std::vector<size_t>{m, n}, attention_output_}};
    std::vector<Tensor> attn_input_tensors{
        Tensor{MEMORY_GPU, data_type, std::vector<size_t>{m, n}, normed_shifted_input_},
        input_tensors->at(1),
        Tensor{MEMORY_GPU,
               TYPE_FP16,
               std::vector<size_t>{(size_t)num_head, (size_t)window_len_, (size_t)window_len_},
               swin_block_weights.attention_relative_pos_bias},
        Tensor{MEMORY_CPU, TYPE_INT8, std::vector<size_t>{6}, additional_parameters}};

    atten_->forward(&attn_output_tensors, &attn_input_tensors, &swin_block_weights.attention_weights);

    invokeGeneralAddBiasResidualPreLayerNorm(attention_output_,
                                             normed_attn_out_buf_,
                                             input,
                                             swin_block_weights.ffn_layernorm_weights.gamma,
                                             swin_block_weights.ffn_layernorm_weights.beta,
                                             swin_block_weights.attention_weights.attention_output_weight.bias,
                                             batch * input_resolution * input_resolution,
                                             dim,
                                             stream_);

    // MLP
    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                          CUBLAS_OP_N,
                          mlp_dim,
                          batch * input_resolution * input_resolution,
                          dim,
                          swin_block_weights.ffn_weights.intermediate_weight.kernel,
                          dim,
                          normed_attn_out_buf_,
                          dim,
                          mlp_buf_,
                          mlp_dim);

    invokeAddBiasGelu(mlp_buf_,
                      swin_block_weights.ffn_weights.intermediate_weight.bias,
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

    invokeAddBiasResidual(output,
                          attention_output_,
                          swin_block_weights.ffn_weights.output_weight.bias,
                          batch * input_resolution * input_resolution,
                          dim,
                          stream_);

    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
}

template class SwinTransformerBlock<float>;
template class SwinTransformerBlock<half>;

}  // namespace fastertransformer