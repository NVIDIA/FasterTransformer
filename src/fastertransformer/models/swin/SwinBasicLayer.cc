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
    assert(false && "SwinTransformerBasicLayer<T>::allocateBuffer() is not implemented\n");
}

template<typename T>
void SwinTransformerBasicLayer<T>::allocateBuffer(int batch, int input_resolution, int dim)
{
    if (is_allocate_buffer_ == false) {
        block_output_ = (T*)allocator_->reMalloc(
            block_output_, 2 * batch * input_resolution * input_resolution * dim * sizeof(T), false);

        is_allocate_buffer_ = true;
    }
}

template<typename T>
void SwinTransformerBasicLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free((void**)(&block_output_));
        is_allocate_buffer_ = false;
    }
}

template<typename T>
SwinTransformerBasicLayer<T>::SwinTransformerBasicLayer(int              max_batch,
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
    block_ = new SwinTransformerBlock<T>(max_batch_,
                                         window_size_,
                                         mlp_ratio_,
                                         layernorm_eps_,
                                         stream,
                                         cublas_wrapper,
                                         allocator,
                                         is_free_buffer_after_forward,
                                         qkv_bias_,
                                         qk_scale_,
                                         version_);
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
void SwinTransformerBasicLayer<T>::patchMerge(T*       output,
                                              T*       merge_layernorm_buf,
                                              const T* input,
                                              const T* gamma,
                                              const T* beta,
                                              const T* weight,
                                              int      batch,
                                              int      input_resolution,
                                              int      dim)
{
    if (version_ == 1) {
        invokeMergeLayernorm(merge_layernorm_buf,
                             input,
                             gamma,
                             beta,
                             layernorm_eps_,
                             batch,
                             input_resolution,
                             input_resolution,
                             dim,
                             stream_);

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
    else if (version_ == 2) {
        invokeImageMerge(merge_layernorm_buf, input, batch, input_resolution, input_resolution, dim, stream_);

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

        invokeGeneralLayerNorm(output,
                               output,
                               gamma,
                               beta,
                               layernorm_eps_,
                               batch * input_resolution * input_resolution / 4,
                               2 * dim,
                               (float*)nullptr,
                               0,
                               stream_);
    }
}

template<typename T>
void SwinTransformerBasicLayer<T>::forward(TensorMap*                          output_tensors,
                                           TensorMap*                          input_tensors,
                                           SwinTransformerBasicLayerWeight<T>& swin_basic_layer_weights)
{
    // input_tensors:
    //      input_query [batch, input_resolution, input_resolution, dim]
    //      additional_params [4] {basic_layer_depth, number_of_head, do_patch_merge, sm}
    // output_tensors:
    //      hidden_features [batch, output_resolution, output_resolution, output_dim]

    T*        output_tensor     = output_tensors->getPtr<T>("hidden_features");
    T*        from_tensor       = input_tensors->getPtr<T>("input_query");
    size_t    batch             = input_tensors->at("input_query").shape[0];
    size_t    input_resolution  = input_tensors->at("input_query").shape[1];
    size_t    dim               = input_tensors->at("input_query").shape[3];
    int*      additional_params = input_tensors->getPtr<int>("additional_params");
    const int depth             = additional_params[0];
    const int num_head          = additional_params[1];
    bool      do_patch_merge    = (additional_params[2] == 1) ? true : false;
    const int sm                = additional_params[3];
    FT_CHECK(input_resolution == input_tensors->at("input_query").shape[2]);

    allocateBuffer(batch, input_resolution, dim);

    int      block_output_size = batch * input_resolution * input_resolution * dim;
    size_t   m                 = batch * input_resolution * input_resolution;
    size_t   n                 = dim;
    size_t   window_num        = (input_resolution / window_size_) * (input_resolution / window_size_);
    size_t   window_len        = window_size_ * window_size_;
    int      shift_size        = 0;
    DataType data_type         = getTensorType<T>();

    if (do_patch_merge) {
        for (int i = 0; i < depth; i++) {
            TensorMap tmp_output_tensors{{"hidden_features",
                                          Tensor{MEMORY_GPU,
                                                 data_type,
                                                 std::vector<size_t>{batch, input_resolution, input_resolution, dim},
                                                 block_output_ + (i % 2) * block_output_size}}};
            shift_size                         = (i % 2 == 0) ? 0 : (window_size_ / 2);
            int       additional_parameters[3] = {num_head, shift_size, sm};
            TensorMap tmp_input_tensors{
                {"input_query",
                 Tensor{MEMORY_GPU,
                        data_type,
                        std::vector<size_t>{batch, input_resolution, input_resolution, dim},
                        i == 0 ? from_tensor : block_output_ + ((i - 1) % 2) * block_output_size}},
                {"attention_mask",
                 Tensor{MEMORY_GPU,
                        TYPE_FP16,
                        std::vector<size_t>{window_num, window_len, window_len},
                        swin_basic_layer_weights.attn_mask}},
                {"trt_attention_mask",
                 Tensor{MEMORY_GPU,
                        TYPE_FP16,
                        std::vector<size_t>{window_num, window_len, window_len},
                        swin_basic_layer_weights.trt_attn_mask}},
                {"additional_params", Tensor{MEMORY_CPU, TYPE_INT8, std::vector<size_t>{3}, additional_parameters}}};
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
            TensorMap tmp_output_tensors{
                {"hidden_features",
                 Tensor{MEMORY_GPU,
                        data_type,
                        std::vector<size_t>{batch, input_resolution, input_resolution, dim},
                        i == depth - 1 ? output_tensor : block_output_ + (i % 2) * block_output_size}}};
            shift_size                         = (i % 2 == 0) ? 0 : (window_size_ / 2);
            int       additional_parameters[3] = {num_head, shift_size, sm};
            TensorMap tmp_input_tensors{
                {"input_query",
                 Tensor{MEMORY_GPU,
                        data_type,
                        std::vector<size_t>{batch, input_resolution, input_resolution, dim},
                        i == 0 ? from_tensor : block_output_ + ((i - 1) % 2) * block_output_size}},
                {"attention_mask",
                 Tensor{MEMORY_GPU,
                        TYPE_FP16,
                        std::vector<size_t>{window_num, window_len, window_len},
                        swin_basic_layer_weights.attn_mask}},
                {"trt_attention_mask",
                 Tensor{MEMORY_GPU,
                        TYPE_FP16,
                        std::vector<size_t>{window_num, window_len, window_len},
                        swin_basic_layer_weights.trt_attn_mask}},
                {"additional_params", Tensor{MEMORY_CPU, TYPE_INT8, std::vector<size_t>{3}, additional_parameters}}};
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
