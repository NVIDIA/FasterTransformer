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

#include "src/fastertransformer/models/swin/Swin.h"

namespace fastertransformer {

template<typename T>
size_t
SwinTransformer<T>::getBufSize(const int batch, const int patches_resolution, const int layer_num, const int embed_dim)
{
    int final_len = patches_resolution;
    for (int i = 0; i < layer_num - 1; i++) {
        final_len /= 2;
    }
    final_len = batch * final_len * final_len;
    // for x_patch_embed_
    size_t buf_size = batch * embed_dim * patches_resolution * patches_resolution * sizeof(T) +
                      // for 2 x basic_layer_output
                      2 * batch * patches_resolution * patches_resolution * embed_dim / 2 * sizeof(T) +
                      // for avg pool ones
                      (final_len + 3) / 4 * 4 * sizeof(T);
    return (buf_size + 31) / 32 * 32;
}

template<typename T>
SwinTransformer<T>::SwinTransformer(int max_batch,
                                    int img_size,
                                    int patch_size,
                                    int in_chans,
                                    int embed_dim,
                                    int window_size,
                                    int* depths,
                                    int* num_heads,
                                    bool ape,
                                    bool patch_norm,
                                    int layer_num,
                                    float mlp_ratio,
                                    cudnnHandle_t cudnn_handle,
                                    cudaStream_t stream,
                                    cublasMMWrapper* cublas_wrapper,
                                    IAllocator* allocator,
                                    bool is_free_buffer_after_forward,
                                    bool qkv_bias,
                                    float qk_scale):
    max_batch_(max_batch),
    img_size_(img_size),
    patch_size_(patch_size),
    in_chans_(in_chans),
    embed_dim_(embed_dim),
    window_size_(window_size),
    depths_(depths),
    num_heads_(num_heads),
    ape_(ape),
    patch_norm_(patch_norm),
    layer_num_(layer_num),
    mlp_ratio_(mlp_ratio),
    cudnn_handle_(cudnn_handle),
    stream_(stream),
    cublas_wrapper_(cublas_wrapper),
    allocator_(allocator),
    is_free_buffer_after_forward_(is_free_buffer_after_forward),
    qkv_bias_(qkv_bias),
    qk_scale_(qk_scale)
{

    patches_resolution_ = img_size / patch_size;
    max_buf_size_ = getBufSize(max_batch_, patches_resolution_, layer_num_, embed_dim_);

    basic_layer_ = new SwinTransformerBasicLayer<T>(max_batch_,
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
SwinTransformer<T>::~SwinTransformer()
{
    freeBuffer();
    if (basic_layer_ != nullptr) {
        delete basic_layer_;
        basic_layer_ = nullptr;
    }
}

template<typename T>
void SwinTransformer<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {

        buf_ = reinterpret_cast<T*>(allocator_->malloc(max_buf_size_, false));
        x_patch_embed_ = buf_;

        basic_layer_output_ = x_patch_embed_ + max_batch_ * embed_dim_ * patches_resolution_ * patches_resolution_;

        avg_pool_ones_ =
            basic_layer_output_ + 2 * max_batch_ * patches_resolution_ * patches_resolution_ * embed_dim_ / 2;

        int final_len = patches_resolution_;
        for (int i = 0; i < layer_num_ - 1; i++) {
            final_len /= 2;
        }
        final_len = max_batch_ * final_len * final_len;
        deviceFill(avg_pool_ones_, (final_len + 3) / 4 * 4, T(1.0f));

        is_allocate_buffer_ = true;
    }
}

template<typename T>
void SwinTransformer<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        if (allocator_ == NULL) {
            printf("[ERROR][SwinTransformer][freeBuffer] allocator_ is NULL!\n");
            exit(-1);
        }
        allocator_->free(buf_);
        allocator_ = nullptr;
        buf_ = nullptr;
        is_allocate_buffer_ = false;
    }
}

template<typename T>
void SwinTransformer<T>::patchEmbed(T* output,
                                    const T* input,
                                    const T* kernel,
                                    const T* bias,
                                    const T* gamma,
                                    const T* beta,
                                    const int batch,
                                    const int img_size,
                                    const int patch_size,
                                    const int patches_resolution,
                                    const int in_chans,
                                    const int embed_dim,
                                    const bool patch_norm)
{
    conv2d(
        output, input, kernel, batch, img_size, img_size, in_chans, embed_dim, patch_size, patch_size, cudnn_handle_);

    if (patch_norm) {
        invokeAddBiasLayernorm<T>(
            output, bias, gamma, beta, batch * patches_resolution * patches_resolution, embed_dim, stream_);
    }
    else {
        invokeAddBias<T>(output, bias, batch * patches_resolution * patches_resolution, embed_dim, stream_);
    }
}

template<typename T>
void SwinTransformer<T>::forward(std::vector<Tensor>* output_tensors,
                                 const std::vector<Tensor>* input_tensors,
                                 SwinTransformerWeight<T>& swin_weights)
{
    // input_tensors:
    //      input_images [batch, in_channels, input_resolution, input_resolution]
    //      sm [1]
    // output_tensors:
    //      output_embedding [batch, final_len]

    T* output = (T*)output_tensors->at(0).data;
    const T* input = (const T*)input_tensors->at(0).data;
    const size_t batch = input_tensors->at(0).shape[0];
    const int sm = *(const int*)input_tensors->at(1).data;
    allocateBuffer();
    patchEmbed(x_patch_embed_,
               input,
               swin_weights.patchEmbed_linear_weights.kernel,
               swin_weights.patchEmbed_linear_weights.bias,
               swin_weights.patchEmbed_norm_weights.gamma,
               swin_weights.patchEmbed_norm_weights.beta,
               batch,
               img_size_,
               patch_size_,
               patches_resolution_,
               in_chans_,
               embed_dim_,
               patch_norm_);

    size_t basic_layer_dim = embed_dim_;
    size_t basic_layer_input_resolution = patches_resolution_;
    int basic_layer_output_size = batch * patches_resolution_ * patches_resolution_ * embed_dim_ / 2;
    size_t m = batch * patches_resolution_ * patches_resolution_;
    size_t n = embed_dim_;
    DataType data_type = getTensorType<T>();

    bool do_patch_merge = true;
    if (layer_num_ == 1) {
        do_patch_merge = false;
    }

    for (int i = 0; i < layer_num_; i++) {
        if (i == layer_num_ - 1) {
            do_patch_merge = false;
        }

        std::vector<Tensor> tmp_output_tensors{Tensor{
            MEMORY_GPU,
            data_type,
            std::vector<size_t>{batch, basic_layer_input_resolution, basic_layer_input_resolution, basic_layer_dim},
            basic_layer_output_ + (i % 2) * basic_layer_output_size}};
        int additional_params[4] = {depths_[i], num_heads_[i], do_patch_merge ? 1 : 0, sm};
        std::vector<Tensor> tmp_input_tensors{
            Tensor{
                MEMORY_GPU,
                data_type,
                std::vector<size_t>{batch, basic_layer_input_resolution, basic_layer_input_resolution, basic_layer_dim},
                i == 0 ? x_patch_embed_ : basic_layer_output_ + ((i - 1) % 2) * basic_layer_output_size},
            Tensor{MEMORY_CPU, TYPE_INT8, std::vector<size_t>{4}, additional_params}};
        basic_layer_->forward(&tmp_output_tensors, &tmp_input_tensors, swin_weights.basic_layer_weight_list[i]);

        if (i != layer_num_ - 1) {
            basic_layer_dim *= 2;
            basic_layer_input_resolution /= 2;
        }
    }

    invokeGeneralLayerNorm(basic_layer_output_ + (layer_num_ % 2) * basic_layer_output_size,
                           basic_layer_output_ + ((layer_num_ - 1) % 2) * basic_layer_output_size,
                           swin_weights.norm_weights.gamma,
                           swin_weights.norm_weights.beta,
                           batch * basic_layer_input_resolution * basic_layer_input_resolution,
                           basic_layer_dim,
                           stream_);

    // avg pool
    int final_len = basic_layer_input_resolution * basic_layer_input_resolution;
    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        basic_layer_dim,
                                        1,
                                        final_len,
                                        basic_layer_output_ + (layer_num_ % 2) * basic_layer_output_size,
                                        basic_layer_dim,
                                        basic_layer_dim * final_len,
                                        avg_pool_ones_,
                                        final_len,
                                        1 * final_len,
                                        output,
                                        basic_layer_dim,
                                        basic_layer_dim * 1,
                                        batch,
                                        1.0f / final_len);

    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
}

template class SwinTransformer<float>;
template class SwinTransformer<half>;

}  // namespace fastertransformer
