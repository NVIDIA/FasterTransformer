/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/layers/TensorParallelGeluFfnLayer.h"
#include "src/fastertransformer/layers/TensorParallelReluFfnLayer.h"
#include "src/fastertransformer/layers/attention_layers/TensorParallelDecoderCrossAttentionLayer.h"
#include "src/fastertransformer/layers/attention_layers/TensorParallelDecoderSelfAttentionLayer.h"
#include "src/fastertransformer/models/t5/T5DecoderLayerWeight.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/allocator.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"

namespace fastertransformer {

template<typename T>
class T5Decoder: public BaseLayer {
private:
    // buffer handling
    size_t max_batch_size_ = 0;
    // meta data
    const size_t head_num_;
    const size_t size_per_head_;
    const size_t inter_size_;
    const size_t d_model_;
    const size_t num_layer_;
    const size_t hidden_units_;
    const ActivationType activation_type_;
    float q_scaling_;

    BaseAttentionLayer<T>* self_attention_layer_;
    BaseAttentionLayer<T>* cross_attention_layer_;
    FfnLayer<T>* ffn_layer_;

    void allocateBuffer() override;
    void freeBuffer() override;
    bool isValidBatchSize(size_t batch_size);
    void allocateBuffer(size_t batch_size);

    void initialize();

    NcclParam tensor_para_;
    NcclParam pipeline_para_;

    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm_;
    int enable_custom_all_reduce_;

    bool isValidLayerParallelId(uint l);
    bool isFirstLayerParallelId(uint l);
    bool isLastLayerParallelId(uint l);
    int getFirstLayerParallelId();

protected:
    T* decoder_normed_input_ = nullptr;
    T* self_attn_output_ = nullptr;
    T* normed_self_attn_output_ = nullptr;
    T* cross_attn_output_ = nullptr;
    T* normed_cross_attn_output_ = nullptr;
    T* decoder_layer_output_ = nullptr;

public:
    T5Decoder(size_t max_batch_size,
              size_t head_num,
              size_t size_per_head,
              size_t inter_size,
              size_t d_model,
              size_t num_layer,
              cudaStream_t stream,
              cublasMMWrapper* cublas_wrapper,
              IAllocator* allocator,
              bool is_free_buffer_after_forward,
              NcclParam tensor_para,
              NcclParam pipeline_para,
              ActivationType activation_type,
              float q_scaling = 1.0f,
              std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm = nullptr,
              int enable_custom_all_reduce = 0);

    T5Decoder(T5Decoder<T> const& decoder);

    ~T5Decoder();

    void forward(std::vector<Tensor>* output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 const std::vector<T5DecoderLayerWeight<T>*>* decoder_layer_weights);
    void setStream(cudaStream_t stream) override;
};

}  // namespace fastertransformer
