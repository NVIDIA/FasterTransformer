/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/bert/Bert.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"

namespace ft = fastertransformer;

template<typename T>
struct BertTritonModelInstance: AbstractTransformerModelInstance {

    BertTritonModelInstance(std::unique_ptr<ft::Bert<T>>                            gpt,
                            std::shared_ptr<ft::BertWeight<T>>                      gpt_weight,
                            std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator,
                            std::unique_ptr<ft::cublasAlgoMap>                      cublas_algo_map,
                            std::unique_ptr<std::mutex>                             cublas_wrapper_mutex,
                            std::unique_ptr<ft::cublasMMWrapper>                    cublas_wrapper,
                            std::unique_ptr<cudaDeviceProp>                         cuda_device_prop_ptr);
    ~BertTritonModelInstance();

    std::shared_ptr<std::vector<triton::Tensor>>
    forward(std::shared_ptr<std::vector<triton::Tensor>> input_tensors) override;

    std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
    forward(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors) override;

    static std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
    convert_outputs(ft::TensorMap& output_tensors);

private:
    const std::unique_ptr<ft::Bert<T>>                            bert_;
    const std::shared_ptr<ft::BertWeight<T>>                      bert_weight_;
    const std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator_;
    const std::unique_ptr<ft::cublasAlgoMap>                      cublas_algo_map_;
    const std::unique_ptr<std::mutex>                             cublas_wrapper_mutex_;
    const std::unique_ptr<ft::cublasMMWrapper>                    cublas_wrapper_;
    const std::unique_ptr<cudaDeviceProp>                         cuda_device_prop_ptr_;

    ft::TensorMap convert_inputs(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors);

    void allocateBuffer(const size_t batch_size, const size_t seq_len, const size_t hidden_units);
    void freeBuffer();

    T*   d_input_hidden_state_  = nullptr;
    int* d_sequence_lengths_    = nullptr;
    T*   d_output_hidden_state_ = nullptr;
};
