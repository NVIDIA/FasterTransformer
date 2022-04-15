/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/triton_backend/t5/T5TritonModelInstance.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"
#include "src/fastertransformer/triton_backend/triton_utils.hpp"
#include "src/fastertransformer/utils/Tensor.h"
#include <vector>

namespace ft = fastertransformer;

template<typename T>
T5TritonModelInstance<T>::T5TritonModelInstance(std::unique_ptr<ft::T5Encoder<T>> t5_encoder,
                                                std::unique_ptr<ft::T5Decoding<T>> t5_decoding,
                                                std::unique_ptr<ft::T5EncoderWeight<T>> t5_encoder_weight,
                                                std::unique_ptr<ft::T5DecodingWeight<T>> t5_decoding_weight,
                                                std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator,
                                                std::unique_ptr<ft::cublasAlgoMap> cublas_algo_map,
                                                std::unique_ptr<std::mutex> cublas_wrapper_mutex,
                                                std::unique_ptr<ft::cublasMMWrapper> cublas_wrapper,
                                                std::unique_ptr<cudaDeviceProp> cuda_device_prop_ptr):
    t5_encoder_(std::move(t5_encoder)),
    t5_decoding_(std::move(t5_decoding)),
    t5_encoder_weight_(std::move(t5_encoder_weight)),
    t5_decoding_weight_(std::move(t5_decoding_weight)),
    allocator_(std::move(allocator)),
    cublas_algo_map_(std::move(cublas_algo_map)),
    cublas_wrapper_mutex_(std::move(cublas_wrapper_mutex)),
    cublas_wrapper_(std::move(cublas_wrapper)),
    cuda_device_prop_ptr_(std::move(cuda_device_prop_ptr))
{
}

template<typename T>
std::unordered_map<std::string, ft::Tensor>
T5TritonModelInstance<T>::convert_inputs(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors)
{
    move_tensor_H2D(input_tensors->at("input_ids"), d_input_ids_);
    move_tensor_H2D(input_tensors->at("sequence_length"), d_input_lengths_);

    std::unordered_map<std::string, ft::Tensor> ft_input_tensors{
        {"input_ids", as_GPU_tensor(input_tensors->at("input_ids"), d_input_ids_)},
        {"sequence_length", as_GPU_tensor(input_tensors->at("sequence_length"), d_input_lengths_)}};
    return ft_input_tensors;
}

template<typename T>
std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
T5TritonModelInstance<T>::convert_outputs(const std::unordered_map<std::string, ft::Tensor>& output_tensors)
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
T5TritonModelInstance<T>::forward(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors)
{
    const size_t request_batch_size = input_tensors->at("input_ids").shape[0];
    const size_t mem_max_seq_len = input_tensors->at("input_ids").shape[1];
    const size_t max_output_len = *((uint*)input_tensors->at("max_output_len").data);
    const size_t beam_width =
        input_tensors->count("beam_width") ? (size_t)(*(uint*)input_tensors->at("beam_width").data) : 1;

    freeBuffer();  // free buffer of previous iteration
    allocateBuffer(request_batch_size, beam_width, max_output_len, mem_max_seq_len);

    std::unordered_map<std::string, ft::Tensor> encoder_input_tensors = convert_inputs(input_tensors);

    std::unordered_map<std::string, ft::Tensor> encoder_output_tensors{
        {"output_hidden_state",
         ft::Tensor{ft::MEMORY_GPU,
                    ft::getTensorType<T>(),
                    std::vector<size_t>{request_batch_size, mem_max_seq_len, t5_encoder_->getDModel()},
                    d_encoder_outputs_}}};

    std::unordered_map<std::string, ft::Tensor> decoding_input_tensors{
        {"encoder_output", encoder_output_tensors.at("output_hidden_state")},
        {"encoder_sequence_length", encoder_input_tensors.at("sequence_length")}};

    for (auto& t : *input_tensors) {
        if (t.first.compare("input_ids") != 0 && t.first.compare("sequence_length") != 0
            && t.first.compare("bad_words_list") != 0 && t.first.compare("stop_words_list") != 0) {
            decoding_input_tensors.insert({t.first, t.second.convertTritonTensorToFt()});
        }
    }

    if (input_tensors->find("bad_words_list") != input_tensors->end()) {
        move_tensor_H2D(input_tensors->at("bad_words_list"), d_input_bad_words_);
        decoding_input_tensors.insert(
            {"bad_words_list", as_GPU_tensor(input_tensors->at("bad_words_list"), d_input_bad_words_)});
    }

    if (input_tensors->find("stop_words_list") != input_tensors->end()) {
        move_tensor_H2D(input_tensors->at("stop_words_list"), d_input_stop_words_);
        decoding_input_tensors.insert(
            {"stop_words_list", as_GPU_tensor(input_tensors->at("stop_words_list"), d_input_stop_words_)});
    }

    std::unordered_map<std::string, ft::Tensor> decoding_output_tensors{
        {"output_ids",
         ft::Tensor{ft::MEMORY_GPU,
                    ft::TYPE_INT32,
                    std::vector<size_t>{request_batch_size, beam_width, max_output_len},
                    d_output_ids_}},
        {"sequence_length",
         ft::Tensor{ft::MEMORY_GPU,
                    ft::TYPE_INT32,
                    std::vector<size_t>{request_batch_size, beam_width},
                    d_sequence_lengths_}}};
    if (input_tensors->count("is_return_log_probs") > 0
        && input_tensors->at("is_return_log_probs").convertTritonTensorToFt().getVal<bool>()) {
        decoding_output_tensors.insert({"output_log_probs",
                                        ft::Tensor{ft::MEMORY_GPU,
                                                   ft::TYPE_FP32,
                                                   std::vector<size_t>{request_batch_size, beam_width, max_output_len},
                                                   d_output_log_probs_}});
        decoding_output_tensors.insert({"cum_log_probs",
                                        ft::Tensor{ft::MEMORY_GPU,
                                                   ft::TYPE_FP32,
                                                   std::vector<size_t>{request_batch_size, beam_width},
                                                   d_cum_log_probs_}});
    }

    t5_encoder_->forward(&encoder_output_tensors, &encoder_input_tensors, t5_encoder_weight_.get());
    t5_decoding_->forward(&decoding_output_tensors, &decoding_input_tensors, t5_decoding_weight_.get());

    if (d_input_ids_ != nullptr) {
        ft::check_cuda_error(cudaFree(d_input_ids_));
        d_input_ids_ = nullptr;
    }
    if (d_input_lengths_ != nullptr) {
        ft::check_cuda_error(cudaFree(d_input_lengths_));
        d_input_lengths_ = nullptr;
    }
    if (d_input_bad_words_ != nullptr) {
        ft::check_cuda_error(cudaFree(d_input_bad_words_));
        d_input_bad_words_ = nullptr;
    }
    if (d_input_stop_words_ != nullptr) {
        ft::check_cuda_error(cudaFree(d_input_stop_words_));
        d_input_stop_words_ = nullptr;
    }

    return convert_outputs(decoding_output_tensors);
}

template<typename T>
T5TritonModelInstance<T>::~T5TritonModelInstance()
{
    freeBuffer();
}

template<typename T>
void T5TritonModelInstance<T>::allocateBuffer(const size_t request_batch_size,
                                              const size_t beam_width,
                                              const size_t max_output_len,
                                              const size_t mem_max_seq_len)
{
    ft::deviceMalloc(&d_encoder_outputs_, request_batch_size * mem_max_seq_len * t5_encoder_->getDModel());
    ft::deviceMalloc(&d_output_ids_, request_batch_size * beam_width * max_output_len);
    ft::deviceMalloc(&d_sequence_lengths_, request_batch_size * beam_width);
    ft::deviceMalloc(&d_output_log_probs_, request_batch_size * beam_width * max_output_len);
    ft::deviceMalloc(&d_cum_log_probs_, request_batch_size * beam_width * max_output_len);
}

template<typename T>
void T5TritonModelInstance<T>::freeBuffer()
{
    ft::deviceFree(d_encoder_outputs_);
    ft::deviceFree(d_output_ids_);
    ft::deviceFree(d_sequence_lengths_);
    ft::deviceFree(d_output_log_probs_);
    ft::deviceFree(d_cum_log_probs_);
}

template struct T5TritonModelInstance<float>;
template struct T5TritonModelInstance<half>;
