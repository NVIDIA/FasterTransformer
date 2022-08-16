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

#include "src/fastertransformer/triton_backend/multi_gpu_gpt/ParallelGptTritonModelInstance.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"
#include "src/fastertransformer/triton_backend/triton_utils.hpp"
#include "src/fastertransformer/utils/Tensor.h"
#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

namespace ft = fastertransformer;

template<typename T>
void triton_stream_callback(std::unordered_map<std::string, ft::Tensor>* output_tensors, void* ctx)
{
    ParallelGptTritonModelInstance<T>* model  = reinterpret_cast<ParallelGptTritonModelInstance<T>*>(ctx);
    auto                               result = ParallelGptTritonModelInstance<T>::convert_outputs(*output_tensors);

    model->stream_cb_(result, model->stream_ctx_);
}

template<typename T>
ParallelGptTritonModelInstance<T>::ParallelGptTritonModelInstance(
    std::unique_ptr<ft::ParallelGpt<T>>                     gpt,
    std::shared_ptr<ft::ParallelGptWeight<T>>               gpt_weight,
    std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator,
    std::unique_ptr<ft::cublasAlgoMap>                      cublas_algo_map,
    std::unique_ptr<std::mutex>                             cublas_wrapper_mutex,
    std::unique_ptr<ft::cublasMMWrapper>                    cublas_wrapper,
    std::unique_ptr<cudaDeviceProp>                         cuda_device_prop_ptr):
    gpt_(std::move(gpt)),
    gpt_weight_(gpt_weight),
    allocator_(std::move(allocator)),
    cublas_algo_map_(std::move(cublas_algo_map)),
    cublas_wrapper_mutex_(std::move(cublas_wrapper_mutex)),
    cublas_wrapper_(std::move(cublas_wrapper)),
    cuda_device_prop_ptr_(std::move(cuda_device_prop_ptr))
{
}

template<typename T>
std::shared_ptr<std::vector<triton::Tensor>>
ParallelGptTritonModelInstance<T>::forward(std::shared_ptr<std::vector<triton::Tensor>> input_tensors)
{
    ft::FT_CHECK(false);
    return nullptr;
}

template<typename T>
std::unordered_map<std::string, ft::Tensor> ParallelGptTritonModelInstance<T>::convert_inputs(
    std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors)
{
    const size_t request_batch_size = input_tensors->at("input_ids").shape[0];
    move_tensor_H2D(input_tensors->at("input_ids"), d_input_ids_, &allocator_);
    move_tensor_H2D(input_tensors->at("input_lengths"), d_input_lengths_, &allocator_);

    const int input_data_len = input_tensors->at("input_ids").shape[1];
    h_total_output_lengths_  = reinterpret_cast<uint32_t*>(malloc(request_batch_size * sizeof(uint32_t)));
    for (int i = 0; i < request_batch_size; ++i) {
        h_total_output_lengths_[i] =
            reinterpret_cast<const uint32_t*>(input_tensors->at("request_output_len").data)[i] + input_data_len;
    }

    std::unordered_map<std::string, ft::Tensor> ft_input_tensors{
        {"input_ids", as_GPU_tensor(input_tensors->at("input_ids"), d_input_ids_)},
        {"input_lengths", as_GPU_tensor(input_tensors->at("input_lengths"), d_input_lengths_)},
        {"output_seq_len",
         ft::Tensor{ft::MEMORY_CPU,
                    ft::TYPE_UINT32,
                    {input_tensors->at("request_output_len").shape[0]},
                    h_total_output_lengths_}}};

    if (input_tensors->count("request_prompt_embedding") && input_tensors->count("request_prompt_lengths")
        && input_tensors->count("request_prompt_type")) {

        move_tensor_H2D(input_tensors->at("request_prompt_lengths"), d_request_prompt_lengths_, &allocator_);
        ft_input_tensors.insert(
            {"request_prompt_lengths",
             as_GPU_tensor(input_tensors->at("request_prompt_lengths"), d_request_prompt_lengths_)});

        move_tensor_H2D(input_tensors->at("request_prompt_embedding"), d_request_prompt_embedding_, &allocator_);
        ft_input_tensors.insert(
            {"request_prompt_embedding",
             as_GPU_tensor(input_tensors->at("request_prompt_embedding"), d_request_prompt_embedding_)});
    }

    if (input_tensors->find("bad_words_list") != input_tensors->end()) {
        move_tensor_H2D(input_tensors->at("bad_words_list"), d_input_bad_words_, &allocator_);
        ft_input_tensors.insert(
            {"bad_words_list", as_GPU_tensor(input_tensors->at("bad_words_list"), d_input_bad_words_)});
    }

    if (input_tensors->find("stop_words_list") != input_tensors->end()) {
        move_tensor_H2D(input_tensors->at("stop_words_list"), d_input_stop_words_, &allocator_);
        ft_input_tensors.insert(
            {"stop_words_list", as_GPU_tensor(input_tensors->at("stop_words_list"), d_input_stop_words_)});
    }

    for (auto t = input_tensors->begin(); t != input_tensors->end(); ++t) {
        if (t->first.find("input_ids") == std::string::npos && t->first.find("input_lengths") == std::string::npos
            && t->first.find("output_seq_len") == std::string::npos
            && t->first.find("request_prompt_embedding") == std::string::npos
            && t->first.find("request_prompt_lengths") == std::string::npos) {
            ft_input_tensors.insert({t->first, t->second.convertTritonTensorToFt()});
        }
    }

    return ft_input_tensors;
}

template<typename T>
std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
ParallelGptTritonModelInstance<T>::convert_outputs(const std::unordered_map<std::string, ft::Tensor>& output_tensors)
{
    std::unordered_map<std::string, triton::Tensor>* outputs_mapping =
        new std::unordered_map<std::string, triton::Tensor>();

    for (auto it = output_tensors.begin(); it != output_tensors.end(); it++) {
        outputs_mapping->insert({it->first, triton::Tensor::convertFtTensorToTriton(it->second)});
    }

    return std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>(outputs_mapping);
}

template<typename T>
std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> ParallelGptTritonModelInstance<T>::forward(
    std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors)
{
    ft::FT_CHECK_WITH_INFO(input_tensors->at("input_ids").shape.size() == 2,
                           "input_tensors->at(\"input_ids\").shape.size() == 2");
    ft::FT_CHECK_WITH_INFO(input_tensors->at("input_lengths").shape.size() == 1,
                           "input_tensors->at(\"input_lengths\").shape.size() == 1");
    const size_t request_batch_size     = input_tensors->at("input_ids").shape[0];
    size_t max_request_output_len = (size_t)*std::max_element(
        (int*)input_tensors->at("request_output_len").data,
        (int*)input_tensors->at("request_output_len").data + input_tensors->at("request_output_len").shape[0]);
    size_t beam_width = input_tensors->count("beam_width") ? (size_t)(*(uint*)input_tensors->at("beam_width").data) : 1;

    size_t total_length = max_request_output_len + input_tensors->at("input_ids").shape[1];

    if (beam_width != 1 && beam_width != 2 && beam_width != 3 && beam_width != 4 && beam_width != 8 && beam_width != 16
        && beam_width != 32) {
        FT_LOG_WARNING("beam_width = %ld is invalid. Set it to 1 to use sampling by default.", beam_width);
        beam_width = 1;
    }

    std::unordered_map<std::string, ft::Tensor> ft_input_tensors = convert_inputs(input_tensors);
    // If input_tensors don't contain "START" flag, then it is non-interactive generation, allocate buffer directly.
    // If input_tensors contains "START" flag, then only allocate buffer when "START == 1".
    if (ft_input_tensors.count("START") == 0
        || (ft_input_tensors.count("START") && ft_input_tensors.at("START").getVal<int32_t>() == 1)) {
        if (ft_input_tensors.count("session_len")) {
            total_length = ft_input_tensors.at("session_len").getVal<uint32_t>();
            max_request_output_len = ft_input_tensors.at("session_len").getVal<uint32_t>();
        }
        size_t max_prefix_soft_prompt_length = 0;
        if (input_tensors->count("request_prompt_lengths")) {
            max_prefix_soft_prompt_length =
                (size_t)*std::max_element((int*)input_tensors->at("request_prompt_lengths").data,
                                          (int*)input_tensors->at("request_prompt_lengths").data + request_batch_size);
        }
        total_length += max_prefix_soft_prompt_length;
        allocateBuffer(request_batch_size, beam_width, total_length, max_request_output_len);
    }

    std::unordered_map<std::string, ft::Tensor> output_tensors = std::unordered_map<std::string, ft::Tensor>{
        {"output_ids",
         ft::Tensor{ft::MEMORY_GPU,
                    ft::TYPE_UINT32,
                    std::vector<size_t>{request_batch_size, beam_width, total_length},
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

    if (stream_cb_ != nullptr) {
        gpt_->registerCallback(triton_stream_callback<T>, this);
    }

    gpt_->forward(&output_tensors, &ft_input_tensors, gpt_weight_.get());

    if (stream_cb_ != nullptr) {
        gpt_->unRegisterCallback();
    }

    return convert_outputs(output_tensors);
}

template<typename T>
ParallelGptTritonModelInstance<T>::~ParallelGptTritonModelInstance()
{
    freeBuffer();
}

template<typename T>
void ParallelGptTritonModelInstance<T>::allocateBuffer(const size_t request_batch_size,
                                                       const size_t beam_width,
                                                       const size_t total_output_len,
                                                       const size_t request_output_len)
{
    d_output_ids_ = (int*)(allocator_->reMalloc(
        d_output_ids_, sizeof(int) * request_batch_size * beam_width * total_output_len, false));
    d_sequence_lengths_ =
        (int*)(allocator_->reMalloc(d_sequence_lengths_, sizeof(int) * request_batch_size * beam_width, false));
    d_output_log_probs_ = (float*)(allocator_->reMalloc(
        d_output_log_probs_, sizeof(float) * request_batch_size * beam_width * request_output_len, false));
    d_cum_log_probs_ =
        (float*)(allocator_->reMalloc(d_cum_log_probs_, sizeof(float) * request_batch_size * beam_width, false));
}

template<typename T>
void ParallelGptTritonModelInstance<T>::freeBuffer()
{
    allocator_->free((void**)(&d_output_ids_));
    allocator_->free((void**)(&d_sequence_lengths_));
    allocator_->free((void**)(&d_output_log_probs_));
    allocator_->free((void**)(&d_cum_log_probs_));
}

template struct ParallelGptTritonModelInstance<float>;
template struct ParallelGptTritonModelInstance<half>;
#ifdef ENABLE_BF16
template struct ParallelGptTritonModelInstance<__nv_bfloat16>;
#endif
