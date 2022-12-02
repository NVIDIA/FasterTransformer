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
void SwinTransformerINT8<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        x_patch_embed_ = (T*)(allocator_->reMalloc(
            x_patch_embed_, max_batch_ * embed_dim_ * patches_resolution_ * patches_resolution_ * sizeof(T), false));
        buffer_COL32   = (T*)(allocator_->reMalloc(
            buffer_COL32, max_batch_ * embed_dim_ * patches_resolution_ * patches_resolution_ * sizeof(T), false));

        basic_layer_output_ = (T*)(allocator_->reMalloc(
            buffer_COL32, max_batch_ * embed_dim_ * patches_resolution_ * patches_resolution_ * sizeof(T), false));

        avg_pool_ones_ =
            basic_layer_output_ + 2 * max_batch_ * patches_resolution_ * patches_resolution_ * embed_dim_ / 2;

        int final_len = patches_resolution_;
        for (int i = 0; i < layer_num_ - 1; i++) {
            final_len /= 2;
        }
        final_len      = (max_batch_ * final_len * final_len + 3) / 4 * 4;
        avg_pool_ones_ = reinterpret_cast<T*>(allocator_->reMalloc(avg_pool_ones_, final_len * sizeof(T), false));
        deviceFill(avg_pool_ones_, final_len, T(1.0f));

        is_allocate_buffer_ = true;
    }
}

template<typename T>
void SwinTransformerINT8<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free((void**)(&buf_));
        is_allocate_buffer_ = false;
    }
}

// input is [B, C_in, H, W]
// output is [B, H, W, C_out]
template<typename T>
void SwinTransformerINT8<T>::patchEmbed(
    T* output, const T* input, const T* kernel, const T* bias, const T* gamma, const T* beta, const int batch)
{
    conv2d(output,
           input,
           kernel,
           batch,
           img_size_,
           img_size_,
           in_chans_,
           embed_dim_,
           patch_size_,
           patch_size_,
           cudnn_handle_);

    if (patch_norm_) {
        invokeAddBiasLayernorm<T>(output,
                                  bias,
                                  gamma,
                                  beta,
                                  layernorm_eps_,
                                  batch * patches_resolution_ * patches_resolution_,
                                  embed_dim_,
                                  stream_);
    }
    else {
        invokeGenericActivation<IdentityActivation, T, T>(output,
                                                          bias,
                                                          nullptr,
                                                          nullptr,
                                                          nullptr,
                                                          nullptr,
                                                          batch * patches_resolution_ * patches_resolution_,
                                                          embed_dim_,
                                                          0,
                                                          nullptr,
                                                          nullptr,
                                                          stream_);
    }
}

template<typename T>
SwinTransformerINT8<T>::SwinTransformerINT8(int              int8_mode,
                                            int              max_batch,
                                            int              img_size,
                                            int              patch_size,
                                            int              in_chans,
                                            int              embed_dim,
                                            int              window_size,
                                            int*             depths,
                                            int*             num_heads,
                                            bool             ape,
                                            bool             patch_norm,
                                            int              layer_num,
                                            float            mlp_ratio,
                                            cudnnHandle_t    cudnn_handle,
                                            cudaStream_t     stream,
                                            cublasMMWrapper* cublas_wrapper,
                                            IAllocator*      allocator,
                                            bool             is_free_buffer_after_forward,
                                            bool             qkv_bias,
                                            float            qk_scale,
                                            int              version):
    int8_mode(int8_mode),
    max_batch_(max_batch),
    img_size_(img_size),
    patch_size_(patch_size),
    in_chans_(in_chans),
    embed_dim_(embed_dim),
    depths_(depths),
    num_heads_(num_heads),
    ape_(ape),
    patch_norm_(patch_norm),
    layer_num_(layer_num),
    cudnn_handle_(cudnn_handle),
    stream_(stream),
    cublas_wrapper_(cublas_wrapper),
    allocator_(allocator),
    is_free_buffer_after_forward_(is_free_buffer_after_forward)
{
    patches_resolution_ = img_size / patch_size;

    basic_layer_ = new SwinTransformerINT8BasicLayer<T>(int8_mode,
                                                        max_batch,
                                                        window_size,
                                                        mlp_ratio,
                                                        layernorm_eps_,
                                                        qkv_bias,
                                                        qk_scale,
                                                        version,
                                                        stream,
                                                        cublas_wrapper,
                                                        allocator,
                                                        is_free_buffer_after_forward);
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
void SwinTransformerINT8<T>::forward(TensorMap*                    output_tensors,
                                     TensorMap*                    input_tensors,
                                     SwinTransformerINT8Weight<T>& swin_weights)
{
    // input_tensors:
    //      input_query [batch, in_channels, input_resolution, input_resolution]
    //      additional_params[1]={sm}
    // output_tensors:
    //      hidden_features [batch, final_len]

    T*           output      = output_tensors->getPtr<T>("hidden_features");
    T*           from_tensor = input_tensors->getPtr<T>("input_query");
    const size_t batch       = input_tensors->at("input_query").shape[0];
    const int    sm          = input_tensors->getVal<const int>("additional_params");
    allocateBuffer();
    patchEmbed(x_patch_embed_,
               from_tensor,
               swin_weights.patchEmbed_linear_weights.kernel,
               swin_weights.patchEmbed_linear_weights.bias,
               swin_weights.patchEmbed_norm_weights.gamma,
               swin_weights.patchEmbed_norm_weights.beta,
               batch);

    size_t   basic_layer_dim              = embed_dim_;
    size_t   basic_layer_input_resolution = patches_resolution_;
    int      basic_layer_output_size      = batch * patches_resolution_ * patches_resolution_ * embed_dim_ / 2;
    size_t   m                            = batch * patches_resolution_ * patches_resolution_;
    size_t   n                            = embed_dim_;
    DataType data_type                    = getTensorType<T>();

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

        TensorMap tmp_output_tensors{
            {"hidden_features",
             Tensor{MEMORY_GPU,
                    data_type,
                    std::vector<size_t>{
                        batch, basic_layer_input_resolution, basic_layer_input_resolution, basic_layer_dim},
                    basic_layer_output_ + (i % 2) * basic_layer_output_size}}};
        int       additional_parameters[5] = {depths_[i], num_heads_[i], do_patch_merge ? 1 : 0, sm, i};
        TensorMap tmp_input_tensors{
            {"input_query",
             Tensor{MEMORY_GPU,
                    data_type,
                    std::vector<size_t>{
                        batch, basic_layer_input_resolution, basic_layer_input_resolution, basic_layer_dim},
                    i == 0 ? buffer_COL32 : basic_layer_output_ + ((i - 1) % 2) * basic_layer_output_size}},
            {"additional_params", Tensor{MEMORY_CPU, TYPE_INT8, std::vector<size_t>{5}, additional_parameters}}};
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
                           layernorm_eps_,
                           batch * basic_layer_input_resolution * basic_layer_input_resolution,
                           basic_layer_dim,
                           (float*)nullptr,
                           0,
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
