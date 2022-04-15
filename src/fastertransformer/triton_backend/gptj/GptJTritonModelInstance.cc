/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/triton_backend/gptj/GptJTritonModelInstance.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"
#include "src/fastertransformer/triton_backend/triton_utils.hpp"
#include "src/fastertransformer/utils/Tensor.h"
#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

namespace ft = fastertransformer;

template<typename T>
GptJTritonModelInstance<T>::GptJTritonModelInstance(std::unique_ptr<ft::GptJ<T>> gpt,
                                                    std::unique_ptr<ft::GptJWeight<T>> gpt_weight,
                                                    std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator,
                                                    std::unique_ptr<ft::cublasAlgoMap> cublas_algo_map,
                                                    std::unique_ptr<std::mutex> cublas_wrapper_mutex,
                                                    std::unique_ptr<ft::cublasMMWrapper> cublas_wrapper,
                                                    std::unique_ptr<cudaDeviceProp> cuda_device_prop_ptr):
    gpt_(std::move(gpt)),
    gpt_weight_(std::move(gpt_weight)),
    allocator_(std::move(allocator)),
    cublas_algo_map_(std::move(cublas_algo_map)),
    cublas_wrapper_mutex_(std::move(cublas_wrapper_mutex)),
    cublas_wrapper_(std::move(cublas_wrapper)),
    cuda_device_prop_ptr_(std::move(cuda_device_prop_ptr))
{
}

template<typename T>
std::unordered_map<std::string, ft::Tensor> GptJTritonModelInstance<T>::convert_inputs(
    std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    move_tensor_H2D(input_tensors->at("input_ids"), d_input_ids_);
    move_tensor_H2D(input_tensors->at("input_lengths"), d_input_lengths_);

    const int input_data_len = input_tensors->at("input_ids").shape[1];
    size_t size = 1;
    for (auto t : input_tensors->at("request_output_len").shape) {
        size = size * t;
    }

    h_total_output_lengths_ = reinterpret_cast<int*>(malloc(size * sizeof(int)));
    for (int i = 0; i < size; ++i) {
        h_total_output_lengths_[i] =
            reinterpret_cast<const int*>(input_tensors->at("request_output_len").data)[i] + input_data_len;
    }

    std::unordered_map<std::string, ft::Tensor> ft_input_tensors = std::unordered_map<std::string, ft::Tensor>{
        {"input_ids", as_GPU_tensor(input_tensors->at("input_ids"), d_input_ids_)},
        {"input_lengths", as_GPU_tensor(input_tensors->at("input_lengths"), d_input_lengths_)},
        {"max_output_seq_len",
         ft::Tensor{
             ft::MEMORY_CPU, ft::TYPE_INT32, input_tensors->at("request_output_len").shape, h_total_output_lengths_}}};

    if (input_tensors->find("bad_words_list") != input_tensors->end()) {
        move_tensor_H2D(input_tensors->at("bad_words_list"), d_input_bad_words_);
        ft_input_tensors.insert(
            {"bad_words_list", as_GPU_tensor(input_tensors->at("bad_words_list"), d_input_bad_words_)});
    }

    if (input_tensors->find("stop_words_list") != input_tensors->end()) {
        move_tensor_H2D(input_tensors->at("stop_words_list"), d_input_stop_words_);
        ft_input_tensors.insert(
            {"stop_words_list", as_GPU_tensor(input_tensors->at("stop_words_list"), d_input_stop_words_)});
    }

    if (input_tensors->count("prefix_soft_prompt_embedding") && input_tensors->count("prefix_soft_prompt_lengths")) {
        triton::Tensor soft_prompt_lengths_tensor = input_tensors->at("prefix_soft_prompt_lengths");
        size_t length_size = std::accumulate(soft_prompt_lengths_tensor.shape.begin(),
                                             soft_prompt_lengths_tensor.shape.end(),
                                             1,
                                             std::multiplies<size_t>());
        ft::deviceMalloc(&d_prefix_soft_prompt_lengths_, length_size, false);
        ft::cudaH2Dcpy(
            d_prefix_soft_prompt_lengths_, reinterpret_cast<const int*>(soft_prompt_lengths_tensor.data), length_size);
        ft_input_tensors.insert(
            {"prefix_soft_prompt_lengths",
             ft::Tensor{
                 ft::MEMORY_GPU, ft::TYPE_INT32, soft_prompt_lengths_tensor.shape, d_prefix_soft_prompt_lengths_}});

        triton::Tensor soft_prompt_embedding_tensor = input_tensors->at("prefix_soft_prompt_embedding");
        size_t emb_size = std::accumulate(soft_prompt_embedding_tensor.shape.begin(),
                                          soft_prompt_embedding_tensor.shape.end(),
                                          1,
                                          std::multiplies<size_t>());
        ft::deviceMalloc(&d_prefix_soft_prompt_embedding_, emb_size, false);
        ft::cudaH2Dcpy(d_prefix_soft_prompt_embedding_,
                       reinterpret_cast<const float*>(soft_prompt_embedding_tensor.data),
                       emb_size);
        ft_input_tensors.insert(
            {"prefix_soft_prompt_embedding",
             ft::Tensor{
                 ft::MEMORY_GPU, ft::TYPE_FP32, soft_prompt_embedding_tensor.shape, d_prefix_soft_prompt_embedding_}});
    }

    for (auto t = input_tensors->begin(); t != input_tensors->end(); ++t) {
        if (t->first.find("input_ids") == std::string::npos && t->first.find("input_lengths") == std::string::npos
            && t->first.find("max_output_seq_len") == std::string::npos
            && t->first.find("prefix_soft_prompt_embedding") == std::string::npos
            && t->first.find("prefix_soft_prompt_lengths") == std::string::npos) {
            if (ft_input_tensors.count(t->first) == 0) {
                ft_input_tensors.insert({t->first, t->second.convertTritonTensorToFt()});
            }
        }
    }

    return ft_input_tensors;
}

template<typename T>
std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
GptJTritonModelInstance<T>::convert_outputs(const std::unordered_map<std::string, ft::Tensor>& output_tensors)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    std::unordered_map<std::string, triton::Tensor>* outputs_mapping =
        new std::unordered_map<std::string, triton::Tensor>();

    for (auto it = output_tensors.begin(); it != output_tensors.end(); it++) {
        outputs_mapping->insert({it->first, triton::Tensor::convertFtTensorToTriton(it->second)});
    }

    return std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>(outputs_mapping);
}

template<typename T>
std::shared_ptr<std::vector<triton::Tensor>>
GptJTritonModelInstance<T>::forward(std::shared_ptr<std::vector<triton::Tensor>> input_tensors)
{
    ft::FT_CHECK(false);
    return nullptr;
}

template<typename T>
std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
GptJTritonModelInstance<T>::forward(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    ft::FT_CHECK_WITH_INFO(input_tensors->at("input_ids").shape.size() == 2,
                           "input_tensors->at(\"input_ids\").shape.size() == 2");
    ft::FT_CHECK_WITH_INFO(input_tensors->at("input_lengths").shape.size() == 1,
                           "input_tensors->at(\"input_lengths\").shape.size() == 1");

    const size_t request_batch_size = input_tensors->at("input_ids").shape[0];
    const size_t max_request_output_len = (size_t)*std::max_element(
        (int*)input_tensors->at("request_output_len").data,
        (int*)input_tensors->at("request_output_len").data + input_tensors->at("request_output_len").shape[0]);
    const size_t total_output_len = max_request_output_len + input_tensors->at("input_ids").shape[1];
    const size_t beam_width =
        input_tensors->count("beam_width") ? (size_t)(*(uint*)input_tensors->at("beam_width").data) : 1;

    freeBuffer();  // free buffer of previous iteration
    allocateBuffer(request_batch_size, beam_width, total_output_len, max_request_output_len);

    std::unordered_map<std::string, ft::Tensor> ft_input_tensors = convert_inputs(input_tensors);

    std::unordered_map<std::string, ft::Tensor> output_tensors = std::unordered_map<std::string, ft::Tensor>{
        {"output_ids",
         ft::Tensor{ft::MEMORY_GPU,
                    ft::TYPE_UINT32,
                    std::vector<size_t>{request_batch_size, beam_width, total_output_len},
                    d_output_ids_}},
        {"sequence_length",
         ft::Tensor{ft::MEMORY_GPU,
                    ft::TYPE_INT32,
                    std::vector<size_t>{request_batch_size, beam_width},
                    d_sequence_lengths_}}};

    if (input_tensors->count("is_return_log_probs") && *((bool*)input_tensors->at("is_return_log_probs").data)) {
        output_tensors.insert({"output_log_probs",
                               ft::Tensor{ft::MEMORY_GPU,
                                          ft::TYPE_FP32,
                                          std::vector<size_t>{request_batch_size, beam_width, max_request_output_len},
                                          d_output_log_probs_}});
        output_tensors.insert({"cum_log_probs",
                               ft::Tensor{ft::MEMORY_GPU,
                                          ft::TYPE_FP32,
                                          std::vector<size_t>{request_batch_size, beam_width},
                                          d_cum_log_probs_}});
    }
    gpt_->forward(&output_tensors, &ft_input_tensors, gpt_weight_.get());

    if (d_input_ids_ != nullptr) {
        ft::check_cuda_error(cudaFree(d_input_ids_));
        d_input_ids_ = nullptr;
    }
    if (d_input_lengths_ != nullptr) {
        ft::check_cuda_error(cudaFree(d_input_lengths_));
        d_input_lengths_ = nullptr;
    }
    if (h_total_output_lengths_ != nullptr) {
        free(h_total_output_lengths_);
        h_total_output_lengths_ = nullptr;
    }
    if (d_input_bad_words_ != nullptr) {
        ft::check_cuda_error(cudaFree(d_input_bad_words_));
        d_input_bad_words_ = nullptr;
    }
    if (d_input_stop_words_ != nullptr) {
        ft::check_cuda_error(cudaFree(d_input_stop_words_));
        d_input_stop_words_ = nullptr;
    }
    if (d_prefix_soft_prompt_embedding_ != nullptr) {
        ft::check_cuda_error(cudaFree(d_prefix_soft_prompt_embedding_));
        d_prefix_soft_prompt_embedding_ = nullptr;
    }
    if (d_prefix_soft_prompt_lengths_ != nullptr) {
        ft::check_cuda_error(cudaFree(d_prefix_soft_prompt_lengths_));
        d_prefix_soft_prompt_lengths_ = nullptr;
    }
    return convert_outputs(output_tensors);
}

template<typename T>
GptJTritonModelInstance<T>::~GptJTritonModelInstance()
{
    freeBuffer();
}

template<typename T>
void GptJTritonModelInstance<T>::allocateBuffer(const size_t request_batch_size,
                                                const size_t beam_width,
                                                const size_t total_output_len,
                                                const size_t max_request_output_len)
{
    ft::deviceMalloc(&d_output_ids_, request_batch_size * beam_width * total_output_len);
    ft::deviceMalloc(&d_sequence_lengths_, request_batch_size * beam_width);
    ft::deviceMalloc(&d_output_log_probs_, max_request_output_len * request_batch_size * beam_width);
    ft::deviceMalloc(&d_cum_log_probs_, request_batch_size * beam_width);
}

template<typename T>
void GptJTritonModelInstance<T>::freeBuffer()
{
    ft::deviceFree(d_output_ids_);
    ft::deviceFree(d_sequence_lengths_);
    ft::deviceFree(d_output_log_probs_);
    ft::deviceFree(d_cum_log_probs_);
}

template struct GptJTritonModelInstance<float>;
template struct GptJTritonModelInstance<half>;
