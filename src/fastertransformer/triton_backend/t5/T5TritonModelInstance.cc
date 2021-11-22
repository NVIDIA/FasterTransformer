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
                                                std::unique_ptr<cudaDeviceProp> cuda_device_prop_ptr,
                                                const size_t max_batch_size,
                                                const size_t max_seq_len,
                                                const size_t beam_width):
    t5_encoder_(std::move(t5_encoder)),
    t5_decoding_(std::move(t5_decoding)),
    t5_encoder_weight_(std::move(t5_encoder_weight)),
    t5_decoding_weight_(std::move(t5_decoding_weight)),
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
T5TritonModelInstance<T>::convert_inputs(std::shared_ptr<std::vector<triton::Tensor>> input_tensors)
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

    ft::FT_CHECK(input_tensors->at(1).shape[1] == 1);
    std::vector<ft::Tensor> ft_input_tensors = std::vector<ft::Tensor>{
        ft::Tensor{ft::MEMORY_GPU,
                   ft::TYPE_INT32,
                   input_tensors->at(0).shape,
                   input_tensors->at(0).where == triton::MEMORY_CPU ? d_input_ids_ : input_tensors->at(0).data},
        ft::Tensor{ft::MEMORY_GPU,
                   ft::TYPE_INT32,
                   std::vector<size_t>{input_tensors->at(1).shape[0]},
                   input_tensors->at(1).where == triton::MEMORY_CPU ? d_input_lengths_ : input_tensors->at(1).data}};
    return ft_input_tensors;
}

template<typename T>
std::shared_ptr<std::vector<triton::Tensor>>
T5TritonModelInstance<T>::convert_outputs(const std::vector<ft::Tensor>& output_tensors)
{
    // Remove the dimension of beam_width of output ids
    // change output.shape to [batch_size, beam_width, total_output_len]
    return std::shared_ptr<std::vector<triton::Tensor>>(new std::vector<triton::Tensor>{
        triton::Tensor{triton::MEMORY_GPU, triton::TYPE_INT32, output_tensors[0].shape, d_output_ids_},
        triton::Tensor{triton::MEMORY_GPU, triton::TYPE_INT32, output_tensors[2].shape, d_sequence_lengths_}});
}

template<typename T>
std::shared_ptr<std::vector<triton::Tensor>>
T5TritonModelInstance<T>::forward(std::shared_ptr<std::vector<triton::Tensor>> input_tensors)
{
    const size_t request_batch_size = input_tensors->at(0).shape[0];
    const size_t mem_max_seq_len = input_tensors->at(0).shape[1];

    freeBuffer();  // free buffer of previous iteration
    allocateBuffer(request_batch_size, mem_max_seq_len);

    std::vector<ft::Tensor> encoder_input_tensors = convert_inputs(input_tensors);

    std::vector<ft::Tensor> encoder_output_tensors = std::vector<ft::Tensor>{
        ft::Tensor{ft::MEMORY_GPU,
                   ft::getTensorType<T>(),
                   std::vector<size_t>{request_batch_size, mem_max_seq_len, t5_encoder_->getDModel()},
                   d_encoder_outputs_}};

    std::vector<ft::Tensor> decoding_input_tensors =
        std::vector<ft::Tensor>{encoder_output_tensors.at(0), encoder_input_tensors.at(1)};

    std::vector<ft::Tensor> decoding_output_tensors = std::vector<ft::Tensor>{
        ft::Tensor{ft::MEMORY_GPU,
                   ft::TYPE_INT32,
                   std::vector<size_t>{request_batch_size, beam_width_, t5_decoding_->getMaxSeqLen()},
                   d_output_ids_},
        ft::Tensor{ft::MEMORY_GPU,
                   ft::TYPE_INT32,
                   std::vector<size_t>{request_batch_size, beam_width_, t5_decoding_->getMaxSeqLen()},
                   d_parent_ids_},
        ft::Tensor{
            ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{request_batch_size, beam_width_}, d_sequence_lengths_}};

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

    return convert_outputs(decoding_output_tensors);
}

template<typename T>
T5TritonModelInstance<T>::~T5TritonModelInstance()
{
    freeBuffer();
}

template<typename T>
void T5TritonModelInstance<T>::allocateBuffer(const size_t request_batch_size, const size_t mem_max_seq_len)
{
    ft::deviceMalloc(&d_encoder_outputs_, request_batch_size * mem_max_seq_len * t5_encoder_->getDModel());
    ft::deviceMalloc(&d_output_ids_, request_batch_size * beam_width_ * t5_decoding_->getMaxSeqLen());
    ft::deviceMalloc(&d_parent_ids_, request_batch_size * beam_width_ * t5_decoding_->getMaxSeqLen());
    ft::deviceMalloc(&d_sequence_lengths_, request_batch_size * beam_width_);
}

template<typename T>
void T5TritonModelInstance<T>::freeBuffer()
{
    ft::deviceFree(d_encoder_outputs_);
    ft::deviceFree(d_output_ids_);
    ft::deviceFree(d_parent_ids_);
    ft::deviceFree(d_sequence_lengths_);
}

template struct T5TritonModelInstance<float>;
template struct T5TritonModelInstance<half>;
