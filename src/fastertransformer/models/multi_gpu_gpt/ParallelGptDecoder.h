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

#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/layers/TensorParallelGeluFfnLayer.h"
#include "src/fastertransformer/layers/attention_layers/TensorParallelDecoderSelfAttentionLayer.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoderLayerWeight.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/allocator.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"
#include "src/fastertransformer/utils/nccl_utils.h"
namespace fastertransformer {

template<typename T>
class ParallelGptDecoder: public BaseLayer {
private:
    // buffer handling
    size_t max_batch_size_ = 0;
    // meta data
    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t num_layer_;

    int int8_mode_ = 0;

    // calculated data
    size_t hidden_units_;

    NcclParam tensor_para_;
    NcclParam pipeline_para_;

    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm_;
    int enable_custom_all_reduce_;

    // buffers
    T* decoder_normed_input_ = nullptr;
    T* self_attn_output_ = nullptr;
    T* normed_self_attn_output_ = nullptr;
    T* decoder_layer_output_ = nullptr;

    BaseAttentionLayer<T>* self_attention_layer_;
    FfnLayer<T>* ffn_layer_;

    void initialize();
    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size);
    void freeBuffer() override;
    bool isValidBatchSize(size_t batch_size);
    bool isValidLayerParallelId(uint l);
    bool isFirstLayerParallelId(uint l);
    bool isLastLayerParallelId(uint l);
    int getFirstLayerParallelId();

protected:
public:
    ParallelGptDecoder(size_t max_batch_size,
                       size_t head_num,
                       size_t size_per_head,
                       size_t inter_size,
                       size_t num_layer,
                       NcclParam tensor_para,
                       NcclParam pipeline_para,
                       cudaStream_t stream,
                       cublasMMWrapper* cublas_wrapper,
                       IAllocator* allocator,
                       bool is_free_buffer_after_forward,
                       bool sparse = false,
                       int int8_mode = 0,
                       std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm = nullptr,
                       int enable_custom_all_reduce_ = 0);

    ParallelGptDecoder(ParallelGptDecoder<T> const& decoder);

    ~ParallelGptDecoder();

    void forward(std::vector<Tensor>* output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 const std::vector<ParallelGptDecoderLayerWeight<T>*>* decoder_layer_weights);
};

}  // namespace fastertransformer
