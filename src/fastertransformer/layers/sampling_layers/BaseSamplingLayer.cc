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

#include "src/fastertransformer/layers/sampling_layers/BaseSamplingLayer.h"
#include "src/fastertransformer/kernels/sampling_penalty_kernels.h"
#include "src/fastertransformer/utils/cuda_utils.h"

namespace fastertransformer {

template<typename T>
BaseSamplingLayer<T>::BaseSamplingLayer(size_t max_batch_size,
                                        size_t vocab_size,
                                        size_t vocab_size_padded,
                                        int end_id,
                                        size_t top_k,
                                        float top_p,
                                        unsigned long long random_seed,
                                        float temperature_,
                                        float len_penalty_,
                                        float repetition_penalty_,
                                        cudaStream_t stream,
                                        cublasMMWrapper* cublas_wrapper,
                                        IAllocator* allocator,
                                        bool is_free_buffer_after_forward,
                                        cudaDeviceProp* cuda_device_prop):
    DynamicDecodeBaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop),
    vocab_size_(vocab_size),
    vocab_size_padded_(vocab_size_padded)
{
}

template<typename T>
BaseSamplingLayer<T>::BaseSamplingLayer(BaseSamplingLayer const& sampling_layer):
    DynamicDecodeBaseLayer(sampling_layer),
    vocab_size_(sampling_layer.vocab_size_),
    vocab_size_padded_(sampling_layer.vocab_size_padded_),
    sampling_workspace_size_(sampling_layer.sampling_workspace_size_)
{
}

template<typename T>
BaseSamplingLayer<T>::~BaseSamplingLayer()
{
}

template<typename T>
void BaseSamplingLayer<T>::forward(std::vector<Tensor>* output_tensors, const std::vector<Tensor>* input_tensors)
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
    //      cum_log_probs [local_batch_size], must be float*

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
                                                               {"cum_log_probs", output_tensors->at(3)}};
    forward(&output_tensors_map, &input_tensors_map);
}

template<typename T>
void BaseSamplingLayer<T>::forward(std::unordered_map<std::string, Tensor>* output_tensors,
                                   const std::unordered_map<std::string, Tensor>* input_tensors)
{
    // input_tensors:
    //      logits [local_batch_size, vocab_size_padded]
    //      embedding_bias [vocab_size_padded]
    //      step [1] on cpu
    //      max_input_length [1] on cpu
    //      input_lengths [local_batch_size]
    //      ite [1] on cpu
    //      runtime_top_k [1] on cpu, optional.
    //      runtime_top_p [1] on cpu, optional
    //      temperature [1] on cpu, optional
    //      len_penalty [1] on cpu, optional
    //      repetition_penalty [1] on cpu, optional
    //      random_seed [1] on cpu, unsigned long long, optional

    // output_tensors:
    //      output_ids [max_seq_len, batch_size]
    //      finished [local_batch_size]
    //      sequence_length [local_batch_size]
    //      cum_log_probs [batch_size], must be float*
    //          The cumultative log probability of generated tokens.
    //      output_log_probs [local_batch_size], must be float*
    //          The log probs at the current step.

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() >= 6);
    FT_CHECK(output_tensors->size() >= 3);
    const int batch_size = output_tensors->at("output_ids").shape[1];

    const int local_batch_size = input_tensors->at("logits").shape[0];
    const int step = *((int*)input_tensors->at("step").data);
    const int ite = *((int*)input_tensors->at("ite").data);

    float temperature = input_tensors->count("temperature") ? input_tensors->at("temperature").getVal<float>() : 1.0f;

    if ((const T*)(input_tensors->at("embedding_bias").data) != nullptr || temperature != 1.0f) {
        invokeApplyTemperaturePenalty((T*)input_tensors->at("logits").data,
                                      (const T*)(input_tensors->at("embedding_bias").data),
                                      temperature,
                                      local_batch_size,
                                      vocab_size_,
                                      vocab_size_padded_,
                                      stream_);
    }
    sync_check_cuda_error();

    if (step > 1
        && (input_tensors->count("repetition_penalty")
            && input_tensors->at("repetition_penalty").getVal<float>() != 1.0f)) {
        invokeApplyRepetitionPenalty((T*)input_tensors->at("logits").data,
                                     input_tensors->at("repetition_penalty").getVal<float>(),
                                     nullptr,
                                     (int*)output_tensors->at("output_ids").data,
                                     batch_size,
                                     local_batch_size,
                                     vocab_size_,
                                     vocab_size_padded_,
                                     (int*)input_tensors->at("input_lengths").data,
                                     *((int*)input_tensors->at("max_input_length").data),
                                     step,
                                     ite,
                                     stream_);
        sync_check_cuda_error();
    }

    runSampling(output_tensors, input_tensors);

    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template class BaseSamplingLayer<float>;
template class BaseSamplingLayer<half>;

}  // namespace fastertransformer