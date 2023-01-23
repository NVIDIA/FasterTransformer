/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <vector>

// #include "src/fastertransformer/kernels/add_residual_fp8_kernels.h"
#include "src/fastertransformer/kernels/layernorm_fp8_kernels.h"
#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/layers/TensorParallelGeluFfnFP8Layer.h"
#include "src/fastertransformer/layers/attention_layers_fp8/TensorParallelGptContextAttentionFP8Layer.h"
#include "src/fastertransformer/models/gpt_fp8/GptFP8DecoderLayerWeight.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/allocator.h"
#include "src/fastertransformer/utils/cublasFP8MMWrapper.h"

namespace fastertransformer {

template<typename T1, typename T2>
class GptFP8ContextDecoder: public BaseLayer {
private:
    // meta data
    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t num_layer_;

    // calculated data
    size_t hidden_units_;

    NcclParam tensor_para_;
    NcclParam pipeline_para_;

    bool is_qk_buf_float_;

    BaseAttentionLayer<T1>* self_attention_layer_;
    FfnFP8Layer<T1, T2>*    ffn_layer_;

    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size, size_t seq_len);
    void freeBuffer() override;
    bool isValidLayerParallelId(uint l);
    void initialize();
    bool isFirstLayerParallelId(uint l);
    bool isLastLayerParallelId(uint l);
    int  getFirstLayerParallelId();

    T1* decoder_normed_input_    = nullptr;
    T2* self_attn_output_        = nullptr;
    T1* normed_self_attn_output_ = nullptr;
    T2* decoder_layer_output_    = nullptr;

protected:
public:
    GptFP8ContextDecoder(size_t           head_num,
                         size_t           size_per_head,
                         size_t           inter_size,
                         size_t           num_layer,
                         NcclParam        tensor_para,
                         NcclParam        pipeline_para,
                         cudaStream_t     stream,
                         cublasMMWrapper* cublas_wrapper,
                         IAllocator*      allocator,
                         bool             is_free_buffer_after_forward,
                         bool             is_qk_buf_float,
                         bool             sparse = false);

    GptFP8ContextDecoder(GptFP8ContextDecoder<T1, T2> const& decoder);

    ~GptFP8ContextDecoder();

    void forward(std::vector<Tensor>*                                  output_tensors,
                 const std::vector<Tensor>*                            input_tensors,
                 const std::vector<GptFP8DecoderLayerWeight<T1, T2>*>* decoder_layer_weights);
};

}  // namespace fastertransformer
