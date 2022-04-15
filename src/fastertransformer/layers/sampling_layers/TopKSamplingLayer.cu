/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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
#include "src/fastertransformer/utils/logger.h"

namespace fastertransformer {

template<typename T>
void TopKSamplingLayer<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void TopKSamplingLayer<T>::allocateBuffer(size_t batch_size, size_t top_k, float top_p)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    invokeTopKSampling<T>(nullptr,
                          sampling_workspace_size_,
                          nullptr,
                          nullptr,
                          nullptr,
                          nullptr,
                          nullptr,
                          nullptr,
                          nullptr,
                          top_k,
                          vocab_size_padded_,
                          nullptr,
                          stream_,
                          batch_size);
    sampling_workspace_ = allocator_->reMalloc(sampling_workspace_, sampling_workspace_size_, false);
    curandstate_buf_ = reinterpret_cast<curandState_t*>(
        allocator_->reMalloc(curandstate_buf_, sizeof(curandState_t) * batch_size, false));
    is_allocate_buffer_ = true;
}

template<typename T>
void TopKSamplingLayer<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free(sampling_workspace_);
        allocator_->free(curandstate_buf_);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
void TopKSamplingLayer<T>::invokeInitialize(size_t batch_size,
                                            unsigned long long random_seed,
                                            curandState_t* curandstate_buf)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    invokeCurandInitialize(curandstate_buf, batch_size, random_seed, stream_);
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
    //      random_seed [1] on cpu, optional

    // output_tensors:
    //      output_ids [max_seq_len, batch_size]
    //      finished [local_batch_size]
    //      sequence_length [local_batch_size]
    //      cum_log_probs [batch_size], must be float*
    //          The cumultative log probability of generated tokens.
    //      output_log_probs [local_batch_size], must be float*
    //          The log probs at the current step.
    FT_CHECK(false);  // TODO deprecated, need to remove
    std::unordered_map<std::string, Tensor> input_tensors_map{{"logits", input_tensors->at(0)},
                                                              {"embedding_bias", input_tensors->at(1)},
                                                              {"step", input_tensors->at(2)},
                                                              {"max_input_length", input_tensors->at(3)},
                                                              {"input_lengths", input_tensors->at(4)},
                                                              {"ite", input_tensors->at(5)}};
    if (input_tensors->size() == 7) {
        input_tensors_map.insert({"random_seed", input_tensors->at(6)});
    }
    std::unordered_map<std::string, Tensor> output_tensors_map{{"output_ids", output_tensors->at(0)},
                                                               {"finished", output_tensors->at(1)},
                                                               {"sequence_length", output_tensors->at(2)},
                                                               {"cum_log_probs", output_tensors->at(3)},
                                                               {"output_log_probs", output_tensors->at(4)}};
    runSampling(&output_tensors_map, &input_tensors_map);
}

template<typename T>
void TopKSamplingLayer<T>::runSampling(std::unordered_map<std::string, Tensor>* output_tensors,
                                       const std::unordered_map<std::string, Tensor>* input_tensors)
{
    // input_tensors:
    //      logits [local_batch_size, vocab_size_padded]
    //      embedding_bias [vocab_size_padded]
    //      step [1] on cpu
    //      max_input_length [1] on cpu
    //      input_lengths [local_batch_size]
    //      ite [1] on cpu
    //      runtime_top_k [1] or [batch_size] on cpu, optional
    //      temperature [1] or [batch_size] on cpu, optional
    //      len_penalty [1] or [batch_size] on cpu, optional
    //      repetition_penalty [1] or [batch_size] on cpu, optional
    //      random_seed [1] or [batch_size] on cpu, optional

    // output_tensors:
    //      output_ids [max_seq_len, batch_size]
    //      finished [local_batch_size]
    //      sequence_length [local_batch_size]
    //      cum_log_probs [batch_size], must be float*, optional
    //          The cumultative log probability of generated tokens.
    //      output_log_probs [local_batch_size], must be float*, optional
    //          The log probs at the current step.

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() >= 6);
    FT_CHECK(output_tensors->size() >= 3);

    const int batch_size = output_tensors->at("output_ids").shape[1];
    const int local_batch_size = input_tensors->at("logits").shape[0];
    const int step = *((int*)input_tensors->at("step").data);
    const int ite = *((int*)input_tensors->at("ite").data);

    const int runtime_top_k = input_tensors->at("runtime_top_k").shape[0] == 1 ?
                                  input_tensors->at("runtime_top_k").getVal<int>(0) :
                                  input_tensors->at("runtime_top_k").getVal<int>(ite * local_batch_size);
    allocateBuffer(batch_size, runtime_top_k, 0.0f);
    if (input_tensors->count("random_seed")) {
        unsigned long long int random_seed =
            input_tensors->at("random_seed").shape[0] == 1 ?
                (unsigned long long int)input_tensors->at("random_seed").getVal<int>(0) :
                (unsigned long long int)input_tensors->at("random_seed").getVal<int>(ite * local_batch_size);
        invokeInitialize(local_batch_size, random_seed, curandstate_buf_ + ite * local_batch_size);
    }

    invokeAddBiasEndMask((T*)(input_tensors->at("logits").data),
                         (T*)(nullptr),
                         (const int*)input_tensors->at("end_id").data,
                         (bool*)output_tensors->at("finished").data,
                         local_batch_size,
                         vocab_size_padded_,
                         stream_);
    sync_check_cuda_error();

    float* cum_log_probs =
        output_tensors->count("cum_log_probs") ? output_tensors->at("cum_log_probs").getPtr<float>() : nullptr;
    float* output_log_probs =
        output_tensors->count("output_log_probs") ? output_tensors->at("output_log_probs").getPtr<float>() : nullptr;

    if (cum_log_probs != nullptr || output_log_probs != nullptr) {
        invokeAddBiasSoftMax((T*)(input_tensors->at("logits").data),
                             (T*)(nullptr),
                             (const int*)input_tensors->at("end_id").data,
                             (bool*)output_tensors->at("finished").data,
                             local_batch_size,
                             vocab_size_padded_,
                             vocab_size_,
                             stream_);
    }

    invokeTopKSampling(
        sampling_workspace_,
        sampling_workspace_size_,
        input_tensors->at("logits").getPtr<T>(),
        output_tensors->at("output_ids").getPtrWithOffset<int>(step * batch_size + ite * local_batch_size),
        output_tensors->at("sequence_length").getPtr<int>(),
        output_tensors->at("finished").getPtr<bool>(),
        cum_log_probs,
        output_log_probs,
        curandstate_buf_ + ite * local_batch_size,
        runtime_top_k,
        vocab_size_padded_,
        (const int*)input_tensors->at("end_id").data,
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
                         0.0f,
                         random_seed,
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
