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

#include "src/fastertransformer/triton_backend/t5-encoder/T5EncoderTritonModelInstance.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"
#include "src/fastertransformer/triton_backend/triton_utils.hpp"
#include "src/fastertransformer/utils/Tensor.h"
#include <algorithm>
#include <vector>

namespace ft = fastertransformer;

template<typename T>
T5EncoderTritonModelInstance<T>::T5EncoderTritonModelInstance(
    std::unique_ptr<ft::T5Encoder<T>>                       t5_encoder,
    std::shared_ptr<ft::T5EncoderWeight<T>>                 t5_encoder_weight,
    std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator,
    std::unique_ptr<ft::cublasAlgoMap>                      cublas_algo_map,
    std::unique_ptr<std::mutex>                             cublas_wrapper_mutex,
    std::unique_ptr<ft::cublasMMWrapper>                    cublas_wrapper,
    std::unique_ptr<cudaDeviceProp>                         cuda_device_prop_ptr):
    t5_encoder_(std::move(t5_encoder)),
    t5_encoder_weight_(t5_encoder_weight),
    allocator_(std::move(allocator)),
    cublas_algo_map_(std::move(cublas_algo_map)),
    cublas_wrapper_mutex_(std::move(cublas_wrapper_mutex)),
    cublas_wrapper_(std::move(cublas_wrapper)),
    cuda_device_prop_ptr_(std::move(cuda_device_prop_ptr))
{
}

template<typename T>
std::unordered_map<std::string, ft::Tensor> T5EncoderTritonModelInstance<T>::convert_inputs(
    std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors)
{
    move_tensor_H2D(input_tensors->at("input_ids"), d_input_ids_, &allocator_);
    move_tensor_H2D(input_tensors->at("sequence_length"), d_input_lengths_, &allocator_);

    std::unordered_map<std::string, ft::Tensor> ft_input_tensors{
        {"input_ids", as_GPU_tensor(input_tensors->at("input_ids"), d_input_ids_)},
        {"sequence_length", as_GPU_tensor(input_tensors->at("sequence_length"), d_input_lengths_)}};

    if (input_tensors->count("ia3_tasks")) {
        move_tensor_H2D(input_tensors->at("ia3_tasks"), d_input_ia3_tasks_, &allocator_);
        ft_input_tensors.insert({"ia3_tasks", as_GPU_tensor(input_tensors->at("ia3_tasks"), d_input_ia3_tasks_)});
    }

    return ft_input_tensors;
}

template<typename T>
std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
T5EncoderTritonModelInstance<T>::convert_outputs(const std::unordered_map<std::string, ft::Tensor>& output_tensors)
{
    std::unordered_map<std::string, triton::Tensor>* outputs_mapping =
        new std::unordered_map<std::string, triton::Tensor>();

    for (auto it = output_tensors.begin(); it != output_tensors.end(); it++) {
        outputs_mapping->insert({it->first, triton::Tensor::convertFtTensorToTriton(it->second)});
    }

    return std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>(outputs_mapping);
}

template<typename T>
std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
T5EncoderTritonModelInstance<T>::forward(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors)
{
    const size_t request_batch_size = input_tensors->at("input_ids").shape[0];
    const size_t mem_max_seq_len    = input_tensors->at("input_ids").shape[1];

    bool any_return_attentions = false;
    if (input_tensors->count("is_return_attentions")) {
        const auto is_return_attentions = input_tensors->at("is_return_attentions");
        any_return_attentions =
            std::any_of(reinterpret_cast<const bool*>(is_return_attentions.data),
                        reinterpret_cast<const bool*>(is_return_attentions.data) + is_return_attentions.shape[0],
                        [](const bool& x) { return x; });
    }

    allocateBuffer(request_batch_size, mem_max_seq_len, any_return_attentions);

    std::unordered_map<std::string, ft::Tensor> encoder_input_tensors = convert_inputs(input_tensors);

    std::unordered_map<std::string, ft::Tensor> encoder_output_tensors{
        {"output_hidden_state",
         ft::Tensor{ft::MEMORY_GPU,
                    ft::getTensorType<T>(),
                    std::vector<size_t>{request_batch_size, mem_max_seq_len, t5_encoder_->getDModel()},
                    (void*)d_encoder_outputs_}}};
    if (any_return_attentions) {
        encoder_output_tensors.insert({"output_attentions",
                                       {ft::MEMORY_GPU,
                                        ft::getTensorType<T>(),
                                        {request_batch_size,
                                         t5_encoder_->getNumLayers(),
                                         t5_encoder_->getNumHeads(),
                                         mem_max_seq_len,
                                         mem_max_seq_len},
                                        d_output_attentions_}});
    }

    if (input_tensors->count("ia3_tasks")) {
        const auto num_ia3_tasks = t5_encoder_weight_->getNumIA3Tasks();
        FT_CHECK_WITH_INFO(num_ia3_tasks > 0, "Cannot request ia3_tasks, model has no IA3 adapters");

        const bool is_within_range = ft::invokeCheckRange<int>(
            d_input_ia3_tasks_, request_batch_size, 0, num_ia3_tasks - 1, d_within_range_, t5_encoder_->getStream());
        FT_CHECK_WITH_INFO(is_within_range,
                           ft::fmtstr("Requested IA3 tasks aren't in the range [0, %d).", num_ia3_tasks));
    }
    try {
        t5_encoder_->forward(&encoder_output_tensors, &encoder_input_tensors, t5_encoder_weight_.get());
        cudaStreamSynchronize(t5_encoder_->getStream());
    }
    catch (...) {
        h_exception_ = std::current_exception();
        encoder_output_tensors.insert(
            {"error_message", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_BYTES, {1}, &h_exception_}});
    }

    return convert_outputs(encoder_output_tensors);
}

template<typename T>
T5EncoderTritonModelInstance<T>::~T5EncoderTritonModelInstance()
{
    freeBuffer();
}

template<typename T>
void T5EncoderTritonModelInstance<T>::allocateBuffer(const size_t request_batch_size,
                                                     const size_t mem_max_seq_len,
                                                     bool         any_return_attentions)
{
    d_encoder_outputs_ = (T*)(allocator_->reMalloc(
        d_encoder_outputs_, sizeof(T) * request_batch_size * mem_max_seq_len * t5_encoder_->getDModel(), false));
    if (any_return_attentions) {
        d_output_attentions_ =
            (T*)(allocator_->reMalloc(d_output_attentions_,
                                      sizeof(T) * request_batch_size * t5_encoder_->getNumHeads()
                                          * t5_encoder_->getNumLayers() * mem_max_seq_len * mem_max_seq_len,
                                      false));
    }
    d_within_range_ = (bool*)(allocator_->reMalloc(d_within_range_, sizeof(bool)));
}

template<typename T>
void T5EncoderTritonModelInstance<T>::freeBuffer()
{
    allocator_->free((void**)(&d_encoder_outputs_));
    allocator_->free((void**)(&d_output_attentions_));
    allocator_->free((void**)(&d_within_range_));
}

template struct T5EncoderTritonModelInstance<float>;
template struct T5EncoderTritonModelInstance<half>;
#ifdef ENABLE_BF16
template struct T5EncoderTritonModelInstance<__nv_bfloat16>;
#endif