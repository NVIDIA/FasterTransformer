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

#include "src/fastertransformer/triton_backend/deberta/DebertaTritonModelInstance.h"
#include "src/fastertransformer/triton_backend/triton_utils.hpp"

namespace ft = fastertransformer;

template<typename T>
void triton_stream_callback(std::unordered_map<std::string, ft::Tensor>* output_tensors, void* ctx)
{
    DebertaTritonModelInstance<T>* model  = reinterpret_cast<DebertaTritonModelInstance<T>*>(ctx);
    auto                        result = DebertaTritonModelInstance<T>::convert_outputs(*output_tensors);

    model->stream_cb_(result, model->stream_ctx_);
}

template<typename T>
DebertaTritonModelInstance<T>::DebertaTritonModelInstance(std::unique_ptr<ft::Deberta<T>>                       deberta,
                                                    std::shared_ptr<ft::DebertaWeight<T>>                       deberta_weight,
                                                    std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>>     allocator,
                                                    std::unique_ptr<ft::cublasAlgoMap>                          cublas_algo_map,
                                                    std::unique_ptr<std::mutex>                                 cublas_wrapper_mutex,
                                                    std::unique_ptr<ft::cublasMMWrapper>                        cublas_wrapper,
                                                    std::unique_ptr<cudaDeviceProp>                             cuda_device_prop_ptr):
    deberta_(std::move(deberta)),
    deberta_weight_(deberta_weight),
    allocator_(std::move(allocator)),
    cublas_algo_map_(std::move(cublas_algo_map)),
    cublas_wrapper_mutex_(std::move(cublas_wrapper_mutex)),
    cublas_wrapper_(std::move(cublas_wrapper)),
    cuda_device_prop_ptr_(std::move(cuda_device_prop_ptr))
{
}

template<typename T>
std::shared_ptr<std::vector<triton::Tensor>>
DebertaTritonModelInstance<T>::forward(std::shared_ptr<std::vector<triton::Tensor>> input_tensors)
{
    ft::FT_CHECK(false);
    return nullptr;
}

template<typename T>
ft::TensorMap DebertaTritonModelInstance<T>::convert_inputs(
    std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors)
{
    move_tensor_H2D(input_tensors->at("input_ids"), d_input_ids_, &allocator_);
    move_tensor_H2D(input_tensors->at("sequence_lengths"), d_input_lengths_, &allocator_);

    ft::TensorMap ft_input_tensors(
        {{"input_ids", as_GPU_tensor(input_tensors->at("input_ids"), d_input_ids_)},
         {"sequence_lengths", as_GPU_tensor(input_tensors->at("sequence_lengths"), d_input_lengths_)}});

    return ft_input_tensors;
}

template<typename T>
std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
DebertaTritonModelInstance<T>::convert_outputs(ft::TensorMap& output_tensors)
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
DebertaTritonModelInstance<T>::forward(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors)
{
    const size_t batch_size     = input_tensors->at("input_ids").shape[0];
    const size_t max_seq_len    = input_tensors->at("input_ids").shape[1];
    const size_t hidden_units   = deberta_->getHiddenUnits();

    allocateBuffer(batch_size, max_seq_len, hidden_units);

    ft::TensorMap ft_input_tensors = convert_inputs(input_tensors);

    ft::TensorMap output_tensors = ft::TensorMap({{"output_hidden_state",
                                                   ft::Tensor{ft::MEMORY_GPU,
                                                              ft::getTensorType<T>(),
                                                              std::vector<size_t>{batch_size, max_seq_len, hidden_units},
                                                              d_output_hidden_state_}}});

    try {
        deberta_->forward(&output_tensors, &ft_input_tensors, deberta_weight_.get());
        cudaStreamSynchronize(deberta_->getStream());
    }
    catch (...) {
        h_exception_ = std::current_exception();
        output_tensors.insert({"error_message", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_BYTES, {1}, &h_exception_}});
    }

    return convert_outputs(output_tensors);
}

template<typename T>
DebertaTritonModelInstance<T>::~DebertaTritonModelInstance()
{
    freeBuffer();
}

template<typename T>
void DebertaTritonModelInstance<T>::allocateBuffer(const size_t batch_size, 
                                            const size_t max_seq_len,
                                            const size_t hidden_units)
{
    d_output_hidden_state_ =
        (T*)(allocator_->reMalloc(d_output_hidden_state_, sizeof(T) * batch_size * max_seq_len * hidden_units, false));
}

template<typename T>
void DebertaTritonModelInstance<T>::freeBuffer()
{
    if (d_output_hidden_state_ != nullptr) {
        allocator_->free((void**)(&d_output_hidden_state_));
    }
    if (d_input_ids_ != nullptr) {
        allocator_->free((void**)(&d_input_ids_));
    }
    if (d_input_lengths_ != nullptr) {
        allocator_->free((void**)(&d_input_lengths_));
    }
}

template struct DebertaTritonModelInstance<float>;
template struct DebertaTritonModelInstance<half>;
#ifdef ENABLE_BF16
template struct DebertaTritonModelInstance<__nv_bfloat16>;
#endif
