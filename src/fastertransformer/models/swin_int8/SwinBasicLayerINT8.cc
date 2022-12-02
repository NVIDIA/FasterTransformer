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

#include "src/fastertransformer/models/swin_int8/SwinBasicLayerINT8.h"

namespace fastertransformer {

template<typename T>
void SwinTransformerINT8BasicLayer<T>::allocateBuffer()
{
    assert(false && "SwinTransformerINT8BasicLayer<T>::allocateBuffer() is not implemented\n");
}

template<typename T>
void SwinTransformerINT8BasicLayer<T>::allocateBuffer(int batch, int input_resolution, int dim)
{
    if (is_allocate_buffer_ == false) {
        block_output_ = (T*)allocator_->reMalloc(
            block_output_, 2 * batch * input_resolution * input_resolution * dim * sizeof(T), false);

        gemm_out_buf_ = (int8_t*)allocator_->reMalloc(
            gemm_out_buf_, batch * input_resolution * input_resolution * dim / 2 * sizeof(int32_t), false);

        is_allocate_buffer_ = true;
    }
}

template<typename T>
void SwinTransformerINT8BasicLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free((void**)(&block_output_));
        allocator_->free((void**)(&gemm_out_buf_));
        is_allocate_buffer_ = false;
    }
}

// input is [B, H, W, C]
// merge_layernorm_buf is [B, H/2, W/2, 4*C]
// output is [B, H/2, W/2, 2*C]
template<typename T>
void SwinTransformerINT8BasicLayer<T>::patchMerge(T*               output,
                                                  int8_t*          gemm_out_buf,
                                                  int8_t*          merge_layernorm_buf,
                                                  const T*         input,
                                                  const T*         gamma,
                                                  const T*         beta,
                                                  const int8_t*    weight,
                                                  int              batch,
                                                  const ScaleList* scalePtr,
                                                  int              input_resolution,
                                                  int              dim,
                                                  int              sm)
{
    cublasINT8MMWrapper* cublas_wrapper = (cublasINT8MMWrapper*)cublas_wrapper_;
    if (version_ == 1) {
        invokeMergeLayerNormCol32(merge_layernorm_buf,
                                  input,
                                  gamma,
                                  beta,
                                  batch,
                                  &(scalePtr->d_scale_list_[60 + 3]),
                                  input_resolution,
                                  input_resolution,
                                  dim,
                                  stream_);

        if (int8_mode == 1) {
            cublas_wrapper->Gemm(gemm_out_buf,
                                 1,
                                 batch * input_resolution * input_resolution / 4,
                                 2 * dim,
                                 4 * dim,
                                 0,
                                 0,
                                 0,
                                 scalePtr->h_scale_list_[scalePtr->p3_offset_ + 6],
                                 merge_layernorm_buf,
                                 weight);

            invokeDequantization(output,
                                 gemm_out_buf,
                                 batch * input_resolution * input_resolution * dim / 2,
                                 &(scalePtr->d_scale_list_[64 + 1]),
                                 stream_);
        }
        else {
            cublas_wrapper->Gemm((int32_t*)gemm_out_buf,
                                 1,
                                 batch * input_resolution * input_resolution / 4,
                                 2 * dim,
                                 4 * dim,
                                 0,
                                 0,
                                 0,
                                 merge_layernorm_buf,
                                 weight);

            invokeDequantization_INT32(output,
                                       (int32_t*)gemm_out_buf,
                                       batch * input_resolution * input_resolution * dim / 2,
                                       stream_,
                                       &(scalePtr->d_scale_list_[60 + 3]),
                                       &(scalePtr->d_scale_list_[scalePtr->p2_offset_ + 4]));
        }
    }
    else if (version_ == 2) {
        invokeImageMergeCol32(merge_layernorm_buf,
                              input,
                              batch,
                              input_resolution,
                              input_resolution,
                              dim,
                              &(scalePtr->d_scale_list_[60 + 3]),
                              stream_);

        cublas_wrapper->Gemm(gemm_out_buf,
                             1,
                             batch * input_resolution * input_resolution / 4,
                             2 * dim,
                             4 * dim,
                             0,
                             0,
                             0,
                             scalePtr->h_scale_list_[scalePtr->p3_offset_ + 6],
                             merge_layernorm_buf,
                             weight);

        invokeLayernormCol32(output,
                             gemm_out_buf,
                             gamma,
                             beta,
                             batch * input_resolution * input_resolution / 4,
                             2 * dim,
                             &(scalePtr->d_scale_list_[64 + 1]),
                             stream_);
    }
}

template<typename T>
SwinTransformerINT8BasicLayer<T>::SwinTransformerINT8BasicLayer(int              int8_mode,
                                                                int              max_batch,
                                                                int              window_size,
                                                                float            mlp_ratio,
                                                                float            layernorm_eps,
                                                                bool             qkv_bias,
                                                                float            qk_scale,
                                                                int              version,
                                                                cudaStream_t     stream,
                                                                cublasMMWrapper* cublas_wrapper,
                                                                IAllocator*      allocator,
                                                                bool             is_free_buffer_after_forward):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    int8_mode(int8_mode),
    max_batch_(max_batch),
    window_size_(window_size),
    layernorm_eps_(layernorm_eps),
    version_(version)
{
    block_ = new SwinTransformerINT8Block<T>(int8_mode,
                                             max_batch,
                                             window_size,
                                             mlp_ratio,
                                             layernorm_eps,
                                             qkv_bias,
                                             stream,
                                             cublas_wrapper,
                                             allocator,
                                             is_free_buffer_after_forward,
                                             qk_scale,
                                             version);
}

template<typename T>
SwinTransformerINT8BasicLayer<T>::~SwinTransformerINT8BasicLayer()
{
    if (block_ != nullptr) {
        delete block_;
        block_ = nullptr;
    }
}

template<typename T>
void SwinTransformerINT8BasicLayer<T>::forward(TensorMap*                              output_tensors,
                                               TensorMap*                              input_tensors,
                                               SwinTransformerINT8BasicLayerWeight<T>& swin_basic_layer_weights)
{
    // input_tensors:
    //      input_query [batch, input_resolution, input_resolution, dim]
    //      additional_params [5] {basic_layer_depth, number_of_head, do_patch_merge, sm, basic_layer_id}
    // output_tensors:
    //      hidden_features [batch, output_resolution, output_resolution, output_dim]

    T*     out_tensor       = output_tensors->at("hidden_features").getPtr<T>();
    T*     from_tensor      = input_tensors->at("input_query").getPtr<T>();
    size_t batch            = input_tensors->at("input_query").shape[0];
    size_t input_resolution = input_tensors->at("input_query").shape[1];
    FT_CHECK(input_resolution == input_tensors->at("input_query").shape[2]);
    size_t    dim             = input_tensors->at("input_query").shape[3];
    int*      input_paramters = input_tensors->getPtr<int>("additional_params");
    const int depth           = input_paramters[0];
    const int num_head        = input_paramters[1];
    bool      do_patch_merge  = (input_paramters[2] == 1) ? true : false;
    const int sm              = input_paramters[3];
    const int basic_layer_id  = input_paramters[4];

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
            int       additional_parameters[5] = {num_head, shift_size, sm, basic_layer_id, i};
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
                {"additional_params", Tensor{MEMORY_CPU, TYPE_INT8, std::vector<size_t>{5}, additional_parameters}}};
            block_->forward(&tmp_output_tensors, &tmp_input_tensors, swin_basic_layer_weights.block_weight_list[i]);
        }

        const ScaleList* scalePtr = &(swin_basic_layer_weights.block_weight_list[0].scalelist);
        // gemm_out_buf_ = (int8_t*)(buf_ + 2 * batch * input_resolution * input_resolution * dim);
        patchMerge(out_tensor,
                   gemm_out_buf_,
                   (int8_t*)(block_output_ + (depth % 2) * block_output_size),
                   block_output_ + ((depth - 1) % 2) * block_output_size,
                   swin_basic_layer_weights.merge_layernorm_weights.gamma,
                   swin_basic_layer_weights.merge_layernorm_weights.beta,
                   (int8_t*)swin_basic_layer_weights.merge_linear_weights.kernel,
                   batch,
                   scalePtr,
                   input_resolution,
                   dim,
                   sm);
    }
    else {
        for (int i = 0; i < depth; i++) {
            TensorMap tmp_output_tensors{
                {"hidden_features",
                 Tensor{MEMORY_GPU,
                        data_type,
                        std::vector<size_t>{batch, input_resolution, input_resolution, dim},
                        i == depth - 1 ? out_tensor : block_output_ + (i % 2) * block_output_size}}};
            shift_size                         = (i % 2 == 0) ? 0 : (window_size_ / 2);
            int       additional_parameters[5] = {num_head, shift_size, sm, basic_layer_id, i};
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
                {"additional_params", Tensor{MEMORY_CPU, TYPE_INT8, std::vector<size_t>{5}, additional_parameters}}};

            block_->forward(&tmp_output_tensors, &tmp_input_tensors, swin_basic_layer_weights.block_weight_list[i]);
        }
    }

    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
}

template class SwinTransformerINT8BasicLayer<float>;
template class SwinTransformerINT8BasicLayer<half>;

}  // namespace fastertransformer