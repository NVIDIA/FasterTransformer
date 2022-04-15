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

#include "SwinBasicLayer.h"

namespace fastertransformer {

template<typename T>
void SwinTransformerBasicLayer<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        block_output_ = (T*)allocator_->malloc(
            2 * max_batch_ * patches_resolution_ * patches_resolution_ * embed_dim_ * sizeof(T), false);

        is_allocate_buffer_ = true;
    }
}

template<typename T>
void SwinTransformerBasicLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free(block_output_);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
SwinTransformerBasicLayer<T>::SwinTransformerBasicLayer(int max_batch,
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
    block_ = new SwinTransformerBlock<T>(max_batch_,
                                         window_size_,
                                         mlp_ratio_,
                                         stream,
                                         cublas_wrapper,
                                         allocator,
                                         is_free_buffer_after_forward,
                                         qkv_bias_,
                                         qk_scale_);
}

template<typename T>
SwinTransformerBasicLayer<T>::~SwinTransformerBasicLayer()
{
    // unBindBuffer();
    if (block_ != nullptr) {
        delete block_;
        block_ = nullptr;
    }
}

template<typename T>
void SwinTransformerBasicLayer<T>::patchMerge(T* output,
                                              T* merge_layernorm_buf,
                                              const T* input,
                                              const T* gamma,
                                              const T* beta,
                                              const T* weight,
                                              int batch,
                                              int input_resolution,
                                              int dim)
{
    invokeMergeLayernorm(
        merge_layernorm_buf, input, gamma, beta, batch, input_resolution, input_resolution, dim, stream_);

    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                          CUBLAS_OP_N,
                          2 * dim,
                          batch * input_resolution * input_resolution / 4,
                          4 * dim,
                          weight,
                          4 * dim,
                          merge_layernorm_buf,
                          4 * dim,
                          output,
                          2 * dim);
}

template<typename T>
void SwinTransformerBasicLayer<T>::forward(std::vector<Tensor>* output_tensors,
                                           std::vector<Tensor>* input_tensors,
                                           SwinTransformerBasicLayerWeight<T>& swin_basic_layer_weights)
{
    // input_tensors:
    //      input [batch, input_resolution, input_resolution, dim]
    //      input_paramters [4] {basic_layer_depth, number_of_head, do_patch_merge, sm}
    // output_tensors:
    //      output [batch, output_resolution, output_resolution, output_dim]

    T* from_tensor = (T*)input_tensors->at(0).data;
    T* output_tensor = (T*)(output_tensors->at(0).data);
    size_t batch = input_tensors->at(0).shape[0];
    size_t input_resolution = input_tensors->at(0).shape[1];
    assert(input_resolution == input_tensors->at(0).shape[2]);
    size_t dim = input_tensors->at(0).shape[3];
    int* input_paramters = (int*)input_tensors->at(1).data;
    const int depth = input_paramters[0];
    const int num_head = input_paramters[1];
    bool do_patch_merge = (input_paramters[2] == 1) ? true : false;
    const int sm = input_paramters[3];

    patches_resolution_ = input_resolution;
    embed_dim_ = dim;
    allocateBuffer();

    int block_output_size = batch * input_resolution * input_resolution * dim;
    size_t m = batch * input_resolution * input_resolution;
    size_t n = dim;
    size_t window_num = (input_resolution / window_size_) * (input_resolution / window_size_);
    size_t window_len = window_size_ * window_size_;
    int shift_size = 0;
    DataType data_type = getTensorType<T>();

    if (do_patch_merge) {
        for (int i = 0; i < depth; i++) {
            std::vector<Tensor> tmp_output_tensors{
                Tensor{MEMORY_GPU,
                       data_type,
                       std::vector<size_t>{batch, input_resolution, input_resolution, dim},
                       block_output_ + (i % 2) * block_output_size}};
            shift_size = (i % 2 == 0) ? 0 : (window_size_ / 2);
            int additional_parameters[3] = {num_head, shift_size, sm};
            std::vector<Tensor> tmp_input_tensors{
                Tensor{MEMORY_GPU,
                       data_type,
                       std::vector<size_t>{batch, input_resolution, input_resolution, dim},
                       i == 0 ? from_tensor : block_output_ + ((i - 1) % 2) * block_output_size},
                Tensor{MEMORY_GPU,
                       TYPE_FP16,
                       std::vector<size_t>{window_num, window_len, window_len},
                       swin_basic_layer_weights.attn_mask},
                Tensor{MEMORY_CPU, TYPE_INT8, std::vector<size_t>{3}, additional_parameters}};
            block_->forward(&tmp_output_tensors, &tmp_input_tensors, swin_basic_layer_weights.block_weight_list[i]);
        }

        patchMerge(output_tensor,
                   block_output_ + (depth % 2) * block_output_size,
                   block_output_ + ((depth - 1) % 2) * block_output_size,
                   swin_basic_layer_weights.merge_layernorm_weights.gamma,
                   swin_basic_layer_weights.merge_layernorm_weights.beta,
                   swin_basic_layer_weights.merge_linear_weights.kernel,
                   batch,
                   input_resolution,
                   dim);
    }
    else {
        for (int i = 0; i < depth; i++) {
            std::vector<Tensor> tmp_output_tensors{
                Tensor{MEMORY_GPU,
                       data_type,
                       std::vector<size_t>{batch, input_resolution, input_resolution, dim},
                       i == depth - 1 ? output_tensor : block_output_ + (i % 2) * block_output_size}};
            shift_size = (i % 2 == 0) ? 0 : (window_size_ / 2);
            int additional_parameters[3] = {num_head, shift_size, sm};
            std::vector<Tensor> tmp_input_tensors{
                Tensor{MEMORY_GPU,
                       data_type,
                       std::vector<size_t>{batch, input_resolution, input_resolution, dim},
                       i == 0 ? from_tensor : block_output_ + ((i - 1) % 2) * block_output_size},
                Tensor{MEMORY_GPU,
                       TYPE_FP16,
                       std::vector<size_t>{window_num, window_len, window_len},
                       swin_basic_layer_weights.attn_mask},
                Tensor{MEMORY_CPU, TYPE_INT8, std::vector<size_t>{3}, additional_parameters}};
            block_->forward(&tmp_output_tensors, &tmp_input_tensors, swin_basic_layer_weights.block_weight_list[i]);
        }
    }

    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
}

template class SwinTransformerBasicLayer<float>;
template class SwinTransformerBasicLayer<half>;
}  // namespace fastertransformer