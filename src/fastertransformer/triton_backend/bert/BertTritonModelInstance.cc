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

#include "src/fastertransformer/triton_backend/bert/BertTritonModelInstance.h"
#include "src/fastertransformer/triton_backend/triton_utils.hpp"

namespace ft = fastertransformer;

template<typename T>
void triton_stream_callback(std::unordered_map<std::string, ft::Tensor>* output_tensors, void* ctx)
{
    BertTritonModelInstance<T>* model  = reinterpret_cast<BertTritonModelInstance<T>*>(ctx);
    auto                        result = BertTritonModelInstance<T>::convert_outputs(*output_tensors);

    model->stream_cb_(result, model->stream_ctx_);
}

template<typename T>
BertTritonModelInstance<T>::BertTritonModelInstance(std::unique_ptr<ft::Bert<T>>                            bert,
                                                    std::shared_ptr<ft::BertWeight<T>>                      bert_weight,
                                                    std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator,
                                                    std::unique_ptr<ft::cublasAlgoMap>   cublas_algo_map,
                                                    std::unique_ptr<std::mutex>          cublas_wrapper_mutex,
                                                    std::unique_ptr<ft::cublasMMWrapper> cublas_wrapper,
                                                    std::unique_ptr<cudaDeviceProp>      cuda_device_prop_ptr):
    bert_(std::move(bert)),
    bert_weight_(bert_weight),
    allocator_(std::move(allocator)),
    cublas_algo_map_(std::move(cublas_algo_map)),
    cublas_wrapper_mutex_(std::move(cublas_wrapper_mutex)),
    cublas_wrapper_(std::move(cublas_wrapper)),
    cuda_device_prop_ptr_(std::move(cuda_device_prop_ptr))
{
}

template<typename T>
std::shared_ptr<std::vector<triton::Tensor>>
BertTritonModelInstance<T>::forward(std::shared_ptr<std::vector<triton::Tensor>> input_tensors)
{
    ft::FT_CHECK(false);
    return nullptr;
}

template<typename T>
ft::TensorMap BertTritonModelInstance<T>::convert_inputs(
    std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors)
{
    move_tensor_H2D(input_tensors->at("input_hidden_state"), d_input_hidden_state_, &allocator_);
    move_tensor_H2D(input_tensors->at("sequence_lengths"), d_sequence_lengths_, &allocator_);

    ft::TensorMap ft_input_tensors(
        {{"input_hidden_state", as_GPU_tensor(input_tensors->at("input_hidden_state"), d_input_hidden_state_)},
         {"sequence_lengths", as_GPU_tensor(input_tensors->at("sequence_lengths"), d_sequence_lengths_)}});

    return ft_input_tensors;
}

template<typename T>
std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
BertTritonModelInstance<T>::convert_outputs(ft::TensorMap& output_tensors)
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
BertTritonModelInstance<T>::forward(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors)
{
    const size_t batch_size   = input_tensors->at("input_hidden_state").shape[0];
    const size_t seq_len      = input_tensors->at("input_hidden_state").shape[1];
    const size_t hidden_units = input_tensors->at("input_hidden_state").shape[2];

    allocateBuffer(batch_size, seq_len, hidden_units);

    ft::TensorMap ft_input_tensors = convert_inputs(input_tensors);

    ft::TensorMap output_tensors = ft::TensorMap({{"output_hidden_state",
                                                   ft::Tensor{ft::MEMORY_GPU,
                                                              ft::getTensorType<T>(),
                                                              std::vector<size_t>{batch_size, seq_len, hidden_units},
                                                              d_output_hidden_state_}}});

    bert_->forward(&output_tensors, &ft_input_tensors, bert_weight_.get());

    if (d_input_hidden_state_ != nullptr) {
        ft::check_cuda_error(cudaFree(d_input_hidden_state_));
        d_input_hidden_state_ = nullptr;
    }
    if (d_sequence_lengths_ != nullptr) {
        ft::check_cuda_error(cudaFree(d_sequence_lengths_));
        d_sequence_lengths_ = nullptr;
    }

    return convert_outputs(output_tensors);
}

template<typename T>
BertTritonModelInstance<T>::~BertTritonModelInstance()
{
    freeBuffer();
}

template<typename T>
void BertTritonModelInstance<T>::allocateBuffer(const size_t batch_size,
                                                const size_t seq_len,
                                                const size_t hidden_units)
{
    d_output_hidden_state_ =
        (T*)(allocator_->reMalloc(d_output_hidden_state_, sizeof(T) * batch_size * seq_len * hidden_units, false));
}

template<typename T>
void BertTritonModelInstance<T>::freeBuffer()
{
    if (d_output_hidden_state_ != nullptr) {
        allocator_->free((void**)(&d_output_hidden_state_));
    }
}

template struct BertTritonModelInstance<float>;
template struct BertTritonModelInstance<half>;
#ifdef ENABLE_BF16
template struct BertTritonModelInstance<__nv_bfloat16>;
#endif