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

#include "src/fastertransformer/triton_backend/multi_gpu_gpt/ParallelGptTritonModelInstance.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"
#include "src/fastertransformer/utils/Tensor.h"
#include <vector>

namespace ft = fastertransformer;

template<typename T>
ParallelGptTritonModelInstance<T>::ParallelGptTritonModelInstance(
    std::unique_ptr<ft::ParallelGpt<T>> gpt,
    std::unique_ptr<ft::ParallelGptWeight<T>> gpt_weight,
    std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator,
    std::unique_ptr<ft::cublasAlgoMap> cublas_algo_map,
    std::unique_ptr<std::mutex> cublas_wrapper_mutex,
    std::unique_ptr<ft::cublasMMWrapper> cublas_wrapper,
    std::unique_ptr<cudaDeviceProp> cuda_device_prop_ptr,
    const size_t max_batch_size,
    const size_t max_seq_len,
    const size_t beam_width):
    gpt_(std::move(gpt)),
    gpt_weight_(std::move(gpt_weight)),
    allocator_(std::move(allocator)),
    cublas_algo_map_(std::move(cublas_algo_map)),
    cublas_wrapper_mutex_(std::move(cublas_wrapper_mutex)),
    cublas_wrapper_(std::move(cublas_wrapper)),
    cuda_device_prop_ptr_(std::move(cuda_device_prop_ptr)),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    beam_width_(beam_width)
{
}

template<typename T>
std::vector<ft::Tensor>
ParallelGptTritonModelInstance<T>::convert_inputs(std::shared_ptr<std::vector<triton::Tensor>> input_tensors)
{
    if (input_tensors->at(0).where == triton::MEMORY_CPU) {
        size_t size = 1;
        for (auto t : input_tensors->at(0).shape) {
            size = size * t;
        }
        ft::deviceMalloc(&d_input_ids_, size, false);
        ft::cudaH2Dcpy(d_input_ids_, (int*)(input_tensors->at(0).data), size);
    }

    if (input_tensors->at(1).where == triton::MEMORY_CPU) {
        size_t size = 1;
        for (auto t : input_tensors->at(1).shape) {
            size = size * t;
        }
        ft::deviceMalloc(&d_input_lengths_, size, false);
        ft::cudaH2Dcpy(d_input_lengths_, (int*)(input_tensors->at(1).data), size);
    }

    h_total_output_len_ =
        (*((int*)input_tensors->at(2).data))
        + (input_tensors->at(0).shape.size() == 3 ? input_tensors->at(0).shape[2] : input_tensors->at(0).shape[1]);
    std::vector<ft::Tensor> ft_input_tensors = std::vector<ft::Tensor>{
        ft::Tensor{ft::MEMORY_GPU,
                   ft::TYPE_INT32,
                   input_tensors->at(0).shape.size() == 3 ?
                       std::vector<size_t>{input_tensors->at(0).shape[0] * input_tensors->at(0).shape[1],
                                           input_tensors->at(0).shape[2]} :
                       std::vector<size_t>{input_tensors->at(0).shape[0] * beam_width_,
                                           input_tensors->at(0).shape[1]},
                   input_tensors->at(0).where == triton::MEMORY_CPU ? d_input_ids_ : input_tensors->at(0).data},
        ft::Tensor{ft::MEMORY_GPU,
                   ft::TYPE_INT32,
                   input_tensors->at(1).shape.size() == 2 ?
                       std::vector<size_t>{input_tensors->at(1).shape[0] * input_tensors->at(1).shape[1]} :
                       input_tensors->at(1).shape,
                   input_tensors->at(1).where == triton::MEMORY_CPU ? d_input_lengths_ : input_tensors->at(1).data},
        ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, {1}, &h_total_output_len_}};
    return ft_input_tensors;
}

template<typename T>
std::shared_ptr<std::vector<triton::Tensor>>
ParallelGptTritonModelInstance<T>::convert_outputs(const std::vector<ft::Tensor>& output_tensors)
{
    // Remove the dimension of beam_width of output ids
    // change output.shape to [batch_size, beam_width, total_output_len]
    return std::shared_ptr<std::vector<triton::Tensor>>(new std::vector<triton::Tensor>{
        triton::Tensor{triton::MEMORY_GPU,
                       triton::TYPE_INT32,
                       output_tensors[0].shape,
                       d_output_ids_}});
}

template<typename T>
std::shared_ptr<std::vector<triton::Tensor>>
ParallelGptTritonModelInstance<T>::forward(std::shared_ptr<std::vector<triton::Tensor>> input_tensors)
{
    const size_t request_batch_size = input_tensors->at(0).shape[0];
    const size_t request_output_len = (size_t) * ((int*)(input_tensors->at(2).data));
    const size_t total_output_len =
        request_output_len
        + (input_tensors->at(0).shape.size() == 3 ? input_tensors->at(0).shape[2] : input_tensors->at(0).shape[1]);

    freeBuffer();  // free buffer of previous iteration
    allocateBuffer(request_batch_size, total_output_len);

    std::vector<ft::Tensor> ft_input_tensors = convert_inputs(input_tensors);

    std::vector<ft::Tensor> output_tensors = std::vector<ft::Tensor>{
        ft::Tensor{ft::MEMORY_GPU,
                   ft::TYPE_INT32,
                   std::vector<size_t>{request_batch_size, beam_width_,(size_t)total_output_len},
                   d_output_ids_},
        ft::Tensor{ft::MEMORY_GPU,
                   ft::TYPE_INT32,
                   std::vector<size_t>{(size_t)total_output_len, request_batch_size, beam_width_},
                   d_parent_ids_},
        ft::Tensor{
            ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{request_batch_size, beam_width_}, d_sequence_lengths_},
        ft::Tensor{ft::MEMORY_GPU,
                   ft::TYPE_FP32,
                   std::vector<size_t>{request_output_len, request_batch_size, beam_width_},
                   nullptr}};
    gpt_->forward(&output_tensors, &ft_input_tensors, gpt_weight_.get());

    if (d_input_ids_ != nullptr) {
        ft::check_cuda_error(cudaFree(d_input_ids_));
        d_input_ids_ = nullptr;
    }
    if (d_input_lengths_ != nullptr) {
        ft::check_cuda_error(cudaFree(d_input_lengths_));
        d_input_lengths_ = nullptr;
    }
    
    return convert_outputs(output_tensors);
}

template<typename T>
ParallelGptTritonModelInstance<T>::~ParallelGptTritonModelInstance()
{
    freeBuffer();
}

template<typename T>
void ParallelGptTritonModelInstance<T>::allocateBuffer(const size_t request_batch_size, const size_t total_output_len)
{
    ft::deviceMalloc(&d_output_ids_, request_batch_size * beam_width_ * total_output_len);
    ft::deviceMalloc(&d_parent_ids_, request_batch_size * beam_width_ * total_output_len);
    ft::deviceMalloc(&d_sequence_lengths_, request_batch_size * beam_width_);
}

template<typename T>
void ParallelGptTritonModelInstance<T>::freeBuffer()
{
    ft::deviceFree(d_output_ids_);
    ft::deviceFree(d_parent_ids_);
    ft::deviceFree(d_sequence_lengths_);
}

template struct ParallelGptTritonModelInstance<float>;
template struct ParallelGptTritonModelInstance<half>;
