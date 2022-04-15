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

#include "SwinINT8.h"

namespace fastertransformer {

template<typename T>
size_t SwinTransformerINT8<T>::getBufSize(const int batch,
                                          const int patches_resolution,
                                          const int layer_num,
                                          const int embed_dim)
{
    int final_len = patches_resolution;
    for (int i = 0; i < layer_num - 1; i++) {
        final_len /= 2;
    }
    final_len = batch * final_len * final_len;
    // for x_patch_embed_
    size_t buf_size = batch * embed_dim * patches_resolution * patches_resolution * sizeof(T) +
                      // for buffer which holds COL32 data
                      batch * embed_dim * patches_resolution * patches_resolution * sizeof(T) +
                      // for 2 x basic_layer_output
                      2 * batch * patches_resolution * patches_resolution * embed_dim / 2 * sizeof(T) +
                      // for avg pool ones
                      (final_len + 3) / 4 * 4 * sizeof(T);

    return (buf_size + 31) / 32 * 32;
}

template<typename T>
void SwinTransformerINT8<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        buf_ = (T*)(allocator_->malloc(max_buf_size_, false));

        x_patch_embed_ = buf_;

        buffer_COL32 = x_patch_embed_ + max_batch_ * embed_dim_ * patches_resolution_ * patches_resolution_;

        basic_layer_output_ = buffer_COL32 + max_batch_ * embed_dim_ * patches_resolution_ * patches_resolution_;

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
void SwinTransformerINT8<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free(buf_);
        is_allocate_buffer_ = false;
    }
}

// input is [B, C_in, H, W]
// output is [B, H, W, C_out]
template<typename T>
void SwinTransformerINT8<T>::patchEmbed(T* output,
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
SwinTransformerINT8<T>::SwinTransformerINT8(int int8_mode,
                                            int max_batch,
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
    int8_mode(int8_mode),
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

    basic_layer_ = new SwinTransformerINT8BasicLayer<T>(int8_mode,
                                                        max_batch_,
                                                        window_size_,
                                                        patches_resolution_,
                                                        embed_dim_,
                                                        mlp_ratio_,
                                                        qkv_bias_,
                                                        qk_scale_,
                                                        stream_,
                                                        cublas_wrapper_,
                                                        allocator_,
                                                        is_free_buffer_after_forward_);
}

template<typename T>
SwinTransformerINT8<T>::~SwinTransformerINT8()
{
    if (basic_layer_ != nullptr) {
        delete basic_layer_;
        basic_layer_ = nullptr;
    }
}

template<typename T>
void SwinTransformerINT8<T>::forward(std::vector<Tensor>* output_tensors,
                                     const std::vector<Tensor>* input_tensors,
                                     SwinTransformerINT8Weight<T>& swin_weights)
{
    // input_tensors:
    //      input_images [batch, in_channels, input_resolution, input_resolution]
    //      sm [1]
    // output_tensors:
    //      output_embedding [batch, final_len]

    T* from_tensor = (T*)input_tensors->at(0).data;
    T* output = (T*)(output_tensors->at(0).data);
    const size_t batch = input_tensors->at(0).shape[0];
    const int sm = *(const int*)input_tensors->at(1).data;
    allocateBuffer();
    patchEmbed(x_patch_embed_,
               from_tensor,
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

    invokeTransposeMatrixColMajorToCOL32(
        buffer_COL32, x_patch_embed_, embed_dim_, batch * patches_resolution_ * patches_resolution_, stream_);

    for (int i = 0; i < layer_num_; i++) {
        if (i == layer_num_ - 1) {
            do_patch_merge = false;
        }

        std::vector<Tensor> tmp_output_tensors{Tensor{
            MEMORY_GPU,
            data_type,
            std::vector<size_t>{batch, basic_layer_input_resolution, basic_layer_input_resolution, basic_layer_dim},
            basic_layer_output_ + (i % 2) * basic_layer_output_size}};
        int additional_parameters[4] = {depths_[i], num_heads_[i], do_patch_merge ? 1 : 0, sm};
        std::vector<Tensor> tmp_input_tensors{
            Tensor{
                MEMORY_GPU,
                data_type,
                std::vector<size_t>{batch, basic_layer_input_resolution, basic_layer_input_resolution, basic_layer_dim},
                i == 0 ? buffer_COL32 : basic_layer_output_ + ((i - 1) % 2) * basic_layer_output_size},
            Tensor{MEMORY_CPU, TYPE_INT8, std::vector<size_t>{4}, additional_parameters}};
        basic_layer_->forward(&tmp_output_tensors, &tmp_input_tensors, swin_weights.basic_layer_weight_list[i]);

        if (i != layer_num_ - 1) {
            basic_layer_dim *= 2;
            basic_layer_input_resolution /= 2;
        }
    }

    invokeTransposeMatrixCOL32ToColMajor(buffer_COL32,
                                         basic_layer_output_ + ((layer_num_ - 1) % 2) * basic_layer_output_size,
                                         batch * basic_layer_input_resolution * basic_layer_input_resolution,
                                         basic_layer_dim,
                                         stream_);
    invokeGeneralLayerNorm(basic_layer_output_ + (layer_num_ % 2) * basic_layer_output_size,
                           buffer_COL32,
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

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template class SwinTransformerINT8<float>;
template class SwinTransformerINT8<half>;

}  // namespace fastertransformer
