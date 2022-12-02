/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2022.  Authored by Yuqing Ding.
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

#pragma once

#include <unordered_map>
#include <vector>

#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/FfnLayer.h"
#include "src/fastertransformer/models/wenet/ConformerConvLayer.h"
#include "src/fastertransformer/models/wenet/RelPositionAttentionLayer.h"
#include "src/fastertransformer/models/wenet/WenetEncoderWeight.h"
#include "src/fastertransformer/models/wenet/WenetKernels.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/wenet_conv2d.h"

namespace fastertransformer {

template<typename T>
class WenetEncoder: public BaseLayer {
private:
    // meta data
    const size_t head_num_;
    const size_t size_per_head_;
    const size_t inter_size_;
    const size_t feature_size_;
    const size_t max_len_;
    const size_t d_model_;
    const size_t hidden_units_;
    const size_t num_layer_;
    const size_t vocab_size_;
    const size_t conv_module_kernel_size_;

    int           sm_;
    float         q_scaling_;
    AttentionType attention_type_;
    bool          sparse_;

    FfnLayer<T>*                  ffn_layer_;
    RelPositionAttentionLayer<T>* attention_layer_;
    ConformerConvLayer<T>*        conformer_conv_layer_;

    bool is_allocate_buffer_ = false;

    void allocateBuffer();
    void allocateBuffer(size_t batch_size, size_t seq_len, size_t feature_size, size_t kernel_size, size_t stride);
    void freeBuffer();
    void initialize();

    const ActivationType activation_type_;

    // for varlen
    size_t*       h_var_token_num_;
    cudnnHandle_t cudnn_handle_;
    cudaEvent_t   stream_finished_;
    cudaEvent_t   stream2_finished_;
    cudaStream_t  stream2_;

    // for model structure
    const bool use_layernorm_in_conv_module_ = false;

protected:
    T* input_hidden_state_ = nullptr;
    T* attention_mask_     = nullptr;
    T* pos_emb_tensor_     = nullptr;

    T*    inter_conv1_input_buf_  = nullptr;
    T*    inter_conv1_output_buf_ = nullptr;
    T*    inter_conv2_output_buf_ = nullptr;
    T*    inter_fc_input_buf_     = nullptr;
    void* conv_workspace_         = nullptr;

    size_t* token_num_      = nullptr;
    int*    padding_offset_ = nullptr;
    int*    bid_start_end_  = nullptr;

    T* normed_from_tensor_ = nullptr;

    T* ffn_out_buf_        = nullptr;
    T* normed_ffn_out_buf_ = nullptr;

    T* attn_out_buf_        = nullptr;
    T* normed_attn_out_buf_ = nullptr;

    T* conv_out_buf_        = nullptr;
    T* normed_conv_out_buf_ = nullptr;

    T* ffn2_out_buf_ = nullptr;

    T* ctc_lo_out_buf_ = nullptr;

    T* log_softmax_out_buf_ = nullptr;

public:
    WenetEncoder(size_t           max_batch_size,
                 size_t           max_seq_len,
                 size_t           head_num,
                 size_t           size_per_head,
                 size_t           feature_size,
                 size_t           max_len,
                 size_t           inter_size,
                 size_t           d_model,
                 size_t           num_layer,
                 size_t           vocab_size,
                 size_t           conv_module_kernel_size,
                 int              sm,
                 float            q_scaling,
                 cudnnHandle_t    cudnn_handle,
                 cudaStream_t     stream,
                 cublasMMWrapper* cublas_wrapper,
                 IAllocator*      allocator,
                 bool             is_free_buffer_after_forward,
                 AttentionType    attention_type,
                 bool             sparse,
                 ActivationType   activation_type,
                 bool             use_layernorm_in_conv_module = false);

    WenetEncoder(WenetEncoder<T> const& layer);

    ~WenetEncoder();

    void forward(std::vector<Tensor>*         output_tensors,
                 const std::vector<Tensor>*   input_tensors,
                 const WenetEncoderWeight<T>* weights);

    void forward(TensorMap* output_tensors, TensorMap* input_tensors, const WenetEncoderWeight<T>* weights);

    void setStream(cudaStream_t stream) override;
};

}  // namespace fastertransformer
