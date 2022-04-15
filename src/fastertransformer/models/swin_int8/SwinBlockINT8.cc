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
    if (is_allocate_buffer_ == false) {
        attention_output_ = (int8_t*)allocator_->malloc(
            max_batch_ * patches_resolution_ * patches_resolution_ * embed_dim_ * sizeof(int8_t), false);
        skip_buf_ = (int8_t*)allocator_->malloc(
            max_batch_ * patches_resolution_ * patches_resolution_ * embed_dim_ * sizeof(int8_t), false);
        mlp_buf_ = (int8_t*)allocator_->malloc(max_batch_ * patches_resolution_ * patches_resolution_
                                                   * int(embed_dim_ * mlp_ratio_) * sizeof(int8_t),
                                               false);
        mlp_output_ = (int8_t*)allocator_->malloc(
            max_batch_ * patches_resolution_ * patches_resolution_ * embed_dim_ * sizeof(int32_t), false);

        normed_shifted_input_ = mlp_buf_;
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void SwinTransformerINT8Block<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free(attention_output_);
        allocator_->free(skip_buf_);
        allocator_->free(mlp_buf_);
        allocator_->free(mlp_output_);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
SwinTransformerINT8Block<T>::SwinTransformerINT8Block(int int8_mode,
                                                      int max_batch,
                                                      int window_size,
                                                      int patches_resolution,
                                                      int embed_dim,
                                                      float mlp_ratio,
                                                      bool qkv_bias,
                                                      cudaStream_t stream,
                                                      cublasMMWrapper* cublas_wrapper,
                                                      IAllocator* allocator,
                                                      bool is_free_buffer_after_forward,
                                                      float qk_scale):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    int8_mode(int8_mode),
    max_batch_(max_batch),
    window_size_(window_size),
    patches_resolution_(patches_resolution),
    embed_dim_(embed_dim),
    mlp_ratio_(mlp_ratio),
    qkv_bias_(qkv_bias),
    qk_scale_(qk_scale)
{
    window_len_ = window_size_ * window_size_;
    atten_ = new WindowAttentionINT8<T>(max_batch_,
                                        window_size_,
                                        patches_resolution,
                                        embed_dim,
                                        stream,
                                        cublas_wrapper,
                                        allocator,
                                        is_free_buffer_after_forward,
                                        qkv_bias_,
                                        qk_scale_);
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
void SwinTransformerINT8Block<T>::forward(std::vector<Tensor>* output_tensors,
                                          const std::vector<Tensor>* input_tensors,
                                          SwinTransformerINT8BlockWeight<T>& swin_block_weights)
{
    // input_tensors:
    //      input [batch, input_resolution, input_resolution, dim]
    //      attention_mask [window_num, window_len, window_len]
    //      input_paramters [3] {number_of_head, shift_size, sm}
    // output_tensors:
    //      output [batch, input_resolution, input_resolution, dim]

    cublasINT8MMWrapper* cublas_wrapper = (cublasINT8MMWrapper*)cublas_wrapper_;
    T* from_tensor = (T*)input_tensors->at(0).data;
    T* out_tensor = (T*)(output_tensors->at(0).data);
    const int batch = input_tensors->at(0).shape[0];
    const int input_resolution = input_tensors->at(0).shape[1];
    const int dim = input_tensors->at(0).shape[3];
    const int* input_parameters = (const int*)input_tensors->at(2).data;
    const int num_head = input_parameters[0];
    int shift_size = input_parameters[1];
    const int sm = input_parameters[2];

    shift_size = (input_resolution <= window_size_) ? 0 : shift_size;
    size_t window_num = (input_resolution / window_size_) * (input_resolution / window_size_);
    int mlp_dim = int(mlp_ratio_ * dim);
    allocateBuffer();

    const size_t m = batch * input_resolution * input_resolution;
    const size_t n = dim;

    // end of checking buf size
    const ScaleList* scalePtr = &(swin_block_weights.scalelist);

    invokeLayernormShiftPartitionCol32(normed_shifted_input_,
                                       from_tensor,
                                       swin_block_weights.attn_layernorm_weights.gamma,
                                       swin_block_weights.attn_layernorm_weights.beta,
                                       batch,
                                       input_resolution,
                                       input_resolution,
                                       dim,
                                       &(scalePtr->d_scale_list_[0 + 3]),
                                       shift_size,
                                       window_size_,
                                       stream_);
    sync_check_cuda_error();

    int additional_parameters[6] = {batch, dim, input_resolution, num_head, shift_size, sm};
    std::vector<Tensor> attn_output_tensors{
        Tensor{MEMORY_GPU, TYPE_INT8, std: vector<size_t>{m, n}, attention_output_}};
    std::vector<Tensor> int8_input_tensors{
        Tensor{MEMORY_GPU, TYPE_INT8, std::vector<size_t>{m, n}, normed_shifted_input_},
        input_tensors->at(1),
        Tensor{MEMORY_GPU,
               TYPE_FP16,
               std::vector<size_t>{(size_t)num_head, (size_t)window_len_, (size_t)window_len_},
               swin_block_weights.attention_relative_pos_bias},
        Tensor{MEMORY_CPU, TYPE_INT8, std::vector<size_t>{6}, additional_parameters}};

    swin_block_weights.attention_weights.scale_list_ptr = &(swin_block_weights.scalelist);
    atten_->forward(&attn_output_tensors, &int8_input_tensors, &swin_block_weights.attention_weights);
    sync_check_cuda_error();
    invokeAddBiasResidualLayerNormCol32_noRes((int8_t*)attention_output_,
                                              (int8_t*)attention_output_,
                                              from_tensor,
                                              swin_block_weights.attention_weights.attention_output_weight.bias,
                                              swin_block_weights.ffn_layernorm_weights.gamma,
                                              swin_block_weights.ffn_layernorm_weights.beta,
                                              m,
                                              dim,
                                              stream_,
                                              &(scalePtr->d_scale_list_[12 + 1]),
                                              &(scalePtr->d_scale_list_[36 + 3]));
    // }

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

    if (int8_mode == 1) {
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