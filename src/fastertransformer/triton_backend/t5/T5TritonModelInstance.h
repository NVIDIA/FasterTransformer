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

#pragma once

#include "src/fastertransformer/models/t5/T5Decoding.h"
#include "src/fastertransformer/models/t5/T5Encoder.h"
#include "src/fastertransformer/triton_backend/t5/T5TritonModel.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"
#include <memory>

namespace ft = fastertransformer;

template<typename T>
struct T5TritonModelInstance: AbstractTransformerModelInstance {

    T5TritonModelInstance(std::unique_ptr<ft::T5Encoder<T>> t5_encoder,
                          std::unique_ptr<ft::T5Decoding<T>> t5_decoding,
                          std::unique_ptr<ft::T5EncoderWeight<T>> t5_encoder_weight,
                          std::unique_ptr<ft::T5DecodingWeight<T>> t5_decoding_weight,
                          std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator,
                          std::unique_ptr<ft::cublasAlgoMap> cublas_algo_map,
                          std::unique_ptr<std::mutex> cublas_wrapper_mutex,
                          std::unique_ptr<ft::cublasMMWrapper> cublas_wrapper,
                          std::unique_ptr<cudaDeviceProp> cuda_device_prop_ptr);
    ~T5TritonModelInstance();

    std::shared_ptr<std::vector<triton::Tensor>>
    forward(std::shared_ptr<std::vector<triton::Tensor>> input_tensors) override
    {
        ft::FT_CHECK(false);
        return nullptr;
    };

    std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
    forward(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors) override;

private:
    const std::unique_ptr<ft::T5Encoder<T>> t5_encoder_;
    const std::unique_ptr<ft::T5EncoderWeight<T>> t5_encoder_weight_;
    const std::unique_ptr<ft::T5Decoding<T>> t5_decoding_;
    const std::unique_ptr<ft::T5DecodingWeight<T>> t5_decoding_weight_;
    const std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator_;
    const std::unique_ptr<ft::cublasAlgoMap> cublas_algo_map_;
    const std::unique_ptr<std::mutex> cublas_wrapper_mutex_;
    const std::unique_ptr<ft::cublasMMWrapper> cublas_wrapper_;
    const std::unique_ptr<cudaDeviceProp> cuda_device_prop_ptr_;

    std::unordered_map<std::string, ft::Tensor>
    convert_inputs(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors);

    std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
    convert_outputs(const std::unordered_map<std::string, ft::Tensor>& output_tensors);

    void allocateBuffer(const size_t request_batch_size,
                        const size_t beam_width,
                        const size_t max_output_len,
                        const size_t mem_max_seq_len);
    void freeBuffer();

    int* d_input_ids_ = nullptr;
    int* d_input_lengths_ = nullptr;
    int* d_input_bad_words_ = nullptr;
    int* d_input_stop_words_ = nullptr;

    T* d_encoder_outputs_ = nullptr;
    int* d_output_ids_ = nullptr;
    int* d_sequence_lengths_ = nullptr;
    float* d_output_log_probs_ = nullptr;
    float* d_cum_log_probs_ = nullptr;

    int h_total_output_len_;
};
