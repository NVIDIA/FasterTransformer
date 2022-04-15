/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "LongformerEncoderOp.h"
#include <cuda_runtime.h>

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

FasterTransformerLongformerEncoder::FasterTransformerLongformerEncoder(int64_t layer_num,
                                                                       int64_t in_dim,
                                                                       int64_t head_num,
                                                                       int64_t size_per_head,
                                                                       int64_t intermediate_size,
                                                                       int64_t local_attn_window_size,
                                                                       int64_t max_global_token_num,
                                                                       int64_t max_batch_size,
                                                                       int64_t max_seq_len,
                                                                       double attn_scaler):
    layer_num_(layer_num),
    in_dim_(in_dim),
    head_num_(head_num),
    size_per_head_(size_per_head),
    intermediate_size_(intermediate_size),
    local_attn_window_size_(local_attn_window_size),
    max_global_token_num_(max_global_token_num),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    attn_scaler_(attn_scaler),
    hidden_units_(head_num * size_per_head)
{
    ft::check_cuda_error(cublasLtCreate(&_cublasltHandle));
    cublas_algo_map_ = new ft::cublasAlgoMap("gemm_config.in");
    cublas_wrapper_mutex_ = new std::mutex();
}

FasterTransformerLongformerEncoder::~FasterTransformerLongformerEncoder()
{
    cublasLtDestroy(_cublasltHandle);
    delete cublas_algo_map_;
    delete cublas_wrapper_mutex_;
}

th::Tensor FasterTransformerLongformerEncoder::forward(
    th::Tensor input, th::Tensor local_attn_mask, th::Tensor global_attn_mask, th::Tensor th_weights, int64_t device_id)
{
    auto scalar_type = th_weights.scalar_type();
    CHECK_INPUT(input, scalar_type);
    CHECK_INPUT(local_attn_mask, scalar_type);
    CHECK_INPUT(global_attn_mask, scalar_type);

    ft::check_cuda_error(cudaSetDevice(device_id));

    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int in_dim_ = input.size(2);

    auto options = th::TensorOptions().dtype(scalar_type).device(torch::kCUDA, device_id);
    auto output = th::zeros({batch_size, seq_len, (int64_t)hidden_units_}, options);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cublasHandle_t _cublasHandle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(_cublasHandle, stream);

    ft::Allocator<ft::AllocatorType::TH>* allocator = new ft::Allocator<ft::AllocatorType::TH>();

    auto cublas_wrapper = new ft::cublasMMWrapper(
        _cublasHandle, _cublasltHandle, stream, cublas_algo_map_, cublas_wrapper_mutex_, allocator);

    ft::DataType data_type;
    if (scalar_type == at::ScalarType::Float) {
        data_type = ft::TYPE_FP32;
    }
    else if (scalar_type == at::ScalarType::Half) {
        data_type = ft::TYPE_FP16;
    }
    else {
        throw std::runtime_error("Wrong Tensor type.");
    }

    std::vector<ft::Tensor> input_tensors = std::vector<ft::Tensor>{
        ft::Tensor{ft::MEMORY_GPU,
                   data_type,
                   std::vector<size_t>{(size_t)batch_size, (size_t)seq_len, (size_t)in_dim_},
                   input.data_ptr()},
        ft::Tensor{ft::MEMORY_GPU,
                   data_type,
                   std::vector<size_t>{(size_t)batch_size, (size_t)seq_len},
                   local_attn_mask.data_ptr()},
        ft::Tensor{ft::MEMORY_GPU,
                   data_type,
                   std::vector<size_t>{(size_t)batch_size, (size_t)seq_len},
                   global_attn_mask.data_ptr()},
    };

    std::vector<ft::Tensor> output_tensors = std::vector<ft::Tensor>{
        ft::Tensor{ft::MEMORY_GPU,
                   data_type,
                   std::vector<size_t>{(size_t)batch_size, (size_t)seq_len, (size_t)hidden_units_},
                   output.data_ptr()},
    };

    if (scalar_type == at::ScalarType::Float) {
        cublas_wrapper->setFP32GemmConfig();
        auto encoder = new ft::LongformerEncoder<float>(layer_num_,
                                                        in_dim_,
                                                        head_num_,
                                                        size_per_head_,
                                                        intermediate_size_,
                                                        local_attn_window_size_,
                                                        max_global_token_num_,
                                                        max_batch_size_,
                                                        max_seq_len_,
                                                        attn_scaler_,
                                                        stream,
                                                        cublas_wrapper,
                                                        allocator,
                                                        false);
        setWeight<float>(layer_num_, in_dim_, hidden_units_, intermediate_size_, th_weights, encoder->getWeightsPtr());
        encoder->forward(&output_tensors, &input_tensors);
        ft::check_cuda_error(cudaStreamSynchronize(stream));
        delete encoder;
    }
    else if (scalar_type == at::ScalarType::Half) {
        cublas_wrapper->setFP16GemmConfig();
        auto encoder = new ft::LongformerEncoder<half>(layer_num_,
                                                       in_dim_,
                                                       head_num_,
                                                       size_per_head_,
                                                       intermediate_size_,
                                                       local_attn_window_size_,
                                                       max_global_token_num_,
                                                       max_batch_size_,
                                                       max_seq_len_,
                                                       attn_scaler_,
                                                       stream,
                                                       cublas_wrapper,
                                                       allocator,
                                                       false);
        setWeight<half>(layer_num_, in_dim_, hidden_units_, intermediate_size_, th_weights, encoder->getWeightsPtr());
        encoder->forward(&output_tensors, &input_tensors);
        ft::check_cuda_error(cudaStreamSynchronize(stream));
        delete encoder;
    }
    delete cublas_wrapper;
    delete allocator;

    return output;
}

static auto fasterTransformerLongformerEncoderTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::FasterTransformerLongformerEncoder>("FasterTransformerLongformerEncoder")
#else
    torch::jit::class_<torch_ext::FasterTransformerLongformerEncoder>("FasterTransformer", "LongformerEncoder")
#endif
        .def(
            torch::jit::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, double>())
        .def("forward", &torch_ext::FasterTransformerLongformerEncoder::forward);

}  // namespace torch_ext