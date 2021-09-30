/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/layers/DynamicDecodeBaseLayer.h"

namespace fastertransformer {

template<typename T>
class BaseBeamSearchLayer: public DynamicDecodeBaseLayer {
private:
    // calculated data
    size_t hidden_units_;

    bool isValidBatchSize(size_t batch_size);
    void freeBuffer();

protected:
    // buffer handling
    size_t max_batch_size_ = 0;
    // meta data
    size_t head_num_;
    size_t size_per_head_;
    size_t beam_width_;
    size_t vocab_size_;
    size_t vocab_size_padded_;
    int end_id_;
    float diversity_rate_;

    float temperature_;
    float len_penalty_;
    float repetition_penalty_;

    size_t topk_softmax_workspace_size_;
    void* topk_softmax_workspace_;

    virtual void allocateBuffer() = 0;
    virtual void invokeSoftMax(std::vector<fastertransformer::Tensor>* output_tensors,
                               const std::vector<fastertransformer::Tensor>* input_tensors) = 0;

public:
    BaseBeamSearchLayer(size_t max_batch_size,
                        size_t head_num,
                        size_t size_per_head,
                        size_t beam_width,
                        size_t vocab_size,
                        size_t vocab_size_padded,
                        int end_id,
                        float diversity_rate,
                        float temperature,
                        float len_penalty,
                        float repetition_penalty,
                        cudaStream_t stream,
                        cublasMMWrapper* cublas_wrapper,
                        IAllocator* allocator,
                        bool is_free_buffer_after_forward);

    BaseBeamSearchLayer(BaseBeamSearchLayer<T> const& beam_search_layer);

    ~BaseBeamSearchLayer();

    void forward(std::vector<fastertransformer::Tensor>* output_tensors,
                 const std::vector<fastertransformer::Tensor>* input_tensors) override;
};

template<typename T>
void update_KV_cache_kernelLauncher(T* key_cache_output,
                                    T* value_cache_output,
                                    const T* key_cache_input,
                                    const T* value_cache_input,
                                    const int* beam_ids,
                                    const bool* finished,
                                    const int max_batch_size,
                                    const int local_batch_size,
                                    const int beam_width,
                                    const int head_num,
                                    const int size_per_head,
                                    const int step,
                                    const int decoder_max_seq_len,
                                    const int cache_size,
                                    const int decoder_layers,
                                    const int ite,
                                    cudaStream_t stream);

}  // namespace fastertransformer
