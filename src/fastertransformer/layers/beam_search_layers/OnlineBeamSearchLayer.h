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

#include "src/fastertransformer/kernels/online_softmax_beamsearch_kernels.h"
#include "src/fastertransformer/layers/beam_search_layers/BaseBeamSearchLayer.h"

namespace fastertransformer {

template<typename T>
class OnlineBeamSearchLayer: public BaseBeamSearchLayer<T> {
private:
    // meta data
    using BaseBeamSearchLayer<T>::head_num_;
    using BaseBeamSearchLayer<T>::size_per_head_;
    using BaseBeamSearchLayer<T>::beam_width_;
    using BaseBeamSearchLayer<T>::vocab_size_;
    using BaseBeamSearchLayer<T>::vocab_size_padded_;
    using BaseBeamSearchLayer<T>::end_id_;
    using BaseBeamSearchLayer<T>::diversity_rate_;

    // buffer handling
    using BaseBeamSearchLayer<T>::max_batch_size_;

    using BaseBeamSearchLayer<T>::topk_softmax_workspace_size_;
    using BaseBeamSearchLayer<T>::topk_softmax_workspace_;

    void allocateBuffer() override;
    void invokeSoftMax(std::vector<fastertransformer::Tensor>* output_tensors,
                       const std::vector<fastertransformer::Tensor>* input_tensors) override;

    using BaseBeamSearchLayer<T>::stream_;
    using BaseBeamSearchLayer<T>::is_allocate_buffer_;
    using BaseBeamSearchLayer<T>::allocator_;

protected:
public:
    OnlineBeamSearchLayer(size_t max_batch_size,
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

    OnlineBeamSearchLayer(OnlineBeamSearchLayer<T> const& beam_search_layer);

    ~OnlineBeamSearchLayer();
};

}  // namespace fastertransformer
