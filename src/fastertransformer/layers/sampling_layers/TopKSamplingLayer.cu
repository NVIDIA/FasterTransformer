/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include <float.h>

#include "src/fastertransformer/kernels/sampling_topk_kernels.h"
#include "src/fastertransformer/kernels/sampling_topp_kernels.h"
#include "src/fastertransformer/layers/sampling_layers/TopKSamplingLayer.h"

namespace fastertransformer {

template<typename T>
void TopKSamplingLayer<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        invokeTopKSampling<T>(nullptr,
                              sampling_workspace_size_,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              top_k_,
                              vocab_size_padded_,
                              end_id_,
                              stream_,
                              max_batch_size_);
        sampling_workspace_ = allocator_->malloc(sampling_workspace_size_, false);
        curandstate_buf_ =
            reinterpret_cast<curandState_t*>(allocator_->malloc(sizeof(curandState_t) * max_batch_size_, false));
        invokeInitialize();
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void TopKSamplingLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free(sampling_workspace_);
        allocator_->free(curandstate_buf_);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
void TopKSamplingLayer<T>::invokeInitialize()
{
    invokeCurandInitialize(curandstate_buf_, max_batch_size_, random_seed_, stream_);
    sync_check_cuda_error();
}

template<typename T>
void TopKSamplingLayer<T>::runSampling(std::vector<fastertransformer::Tensor>* output_tensors,
                                       const std::vector<fastertransformer::Tensor>* input_tensors)
{
    // input_tensors:
    //      logits [local_batch_size, vocab_size_padded]
    //      embedding_bias [vocab_size_padded]
    //      step [1] on cpu
    //      max_input_length [1] on cpu
    //      input_lengths [local_batch_size]
    //      ite [1] on cpu

    // output_tensors:
    //      output_ids [max_seq_len, batch_size]
    //      finished [local_batch_size]
    //      sequence_length [local_batch_size]
    //      cum_log_probs [local_batch_size], must be float*

    FT_CHECK(input_tensors->size() == 6);
    FT_CHECK(output_tensors->size() == 4);

    const int batch_size = output_tensors->at(0).shape[1];
    const int local_batch_size = input_tensors->at(0).shape[0];
    const int step = *((int*)input_tensors->at(2).data);
    const int ite = *((int*)input_tensors->at(5).data);

    invokeAddBiasEndMask((T*)(input_tensors->at(0).data),
                         (T*)(nullptr),
                         end_id_,
                         (bool*)output_tensors->at(1).data,
                         local_batch_size,
                         vocab_size_padded_,
                         stream_);
    sync_check_cuda_error();

    if (output_tensors->at(3).data != nullptr) {
        invokeAddBiasSoftMax((T*)(input_tensors->at(0).data),
                             (T*)(nullptr),
                             end_id_,
                             (bool*)output_tensors->at(1).data,
                             local_batch_size,
                             vocab_size_padded_,
                             vocab_size_,
                             stream_);
    }

    invokeTopKSampling(sampling_workspace_,
                       sampling_workspace_size_,
                       (T*)(input_tensors->at(0).data),
                       ((int*)output_tensors->at(0).data) + step * batch_size + ite * local_batch_size,
                       (int*)output_tensors->at(2).data,
                       (bool*)output_tensors->at(1).data,
                       (float*)output_tensors->at(3).data,
                       curandstate_buf_ + ite * local_batch_size,
                       top_k_,
                       vocab_size_padded_,
                       end_id_,
                       stream_,
                       local_batch_size);
    sync_check_cuda_error();
}

template<typename T>
TopKSamplingLayer<T>::TopKSamplingLayer(size_t max_batch_size,
                                        size_t vocab_size,
                                        size_t vocab_size_padded,
                                        int end_id,
                                        size_t top_k,
                                        unsigned long long random_seed,
                                        float temperature,
                                        float len_penalty,
                                        float repetition_penalty,
                                        cudaStream_t stream,
                                        cublasMMWrapper* cublas_wrapper,
                                        IAllocator* allocator,
                                        bool is_free_buffer_after_forward):
    BaseSamplingLayer<T>(max_batch_size,
                         vocab_size,
                         vocab_size_padded,
                         end_id,
                         top_k,
                         random_seed,
                         0.0f,
                         temperature,
                         len_penalty,
                         repetition_penalty,
                         stream,
                         cublas_wrapper,
                         allocator,
                         is_free_buffer_after_forward,
                         nullptr)
{
}

template<typename T>
TopKSamplingLayer<T>::TopKSamplingLayer(TopKSamplingLayer<T> const& top_k_sampling_layer):
    BaseSamplingLayer<T>(top_k_sampling_layer)
{
}

template<typename T>
TopKSamplingLayer<T>::~TopKSamplingLayer()
{
    freeBuffer();
}

template class TopKSamplingLayer<float>;
template class TopKSamplingLayer<half>;

}  // namespace fastertransformer