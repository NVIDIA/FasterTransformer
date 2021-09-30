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
/**
 * Memory Allocator
 **/

#pragma once

#include <cuda_runtime.h>
#include <vector>

#include "cuda_utils.h"

#ifdef GOOGLE_CUDA
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#endif

#ifdef TORCH_CUDA
#include "torch/extension.h"
#include <memory>
#endif

namespace fastertransformer {

enum class AllocatorType
{
    CUDA,
    TF,
    TH
};

class IAllocator {
public:
    virtual void* malloc(size_t size, const bool is_set_zero = true) const = 0;
    virtual void free(void* ptr) const = 0;
};

template<AllocatorType AllocType_>
class Allocator;

template<>
class Allocator<AllocatorType::CUDA>: public IAllocator {
    const int device_id_;

public:
    Allocator(int device_id): device_id_(device_id) {}
    virtual ~Allocator() {}

    void* malloc(size_t size, const bool is_set_zero = true) const
    {
        void* ptr = nullptr;
        int o_device = 0;
        check_cuda_error(getSetDevice(device_id_, &o_device));
        check_cuda_error(cudaMalloc(&ptr, (size_t)(ceil(size / 32.)) * 32));
        check_cuda_error(getSetDevice(o_device));
        return ptr;
    }

    void free(void* ptr) const
    {
        int o_device = 0;
        check_cuda_error(getSetDevice(device_id_, &o_device));
        check_cuda_error(cudaFree(ptr));
        check_cuda_error(getSetDevice(o_device));
        return;
    }
};

#ifdef GOOGLE_CUDA
using namespace tensorflow;
template<>
class Allocator<AllocatorType::TF>: public IAllocator {
    OpKernelContext* context_;
    std::vector<tensorflow::Tensor>* allocated_tensor_vector;
    cudaStream_t stream_;

public:
    Allocator(OpKernelContext* context, cudaStream_t stream): context_(context), stream_(stream)
    {
        allocated_tensor_vector = new std::vector<tensorflow::Tensor>;
    }

    void* malloc(size_t size, const bool is_set_zero = true) const
    {
        tensorflow::Tensor buf;
        long long int buf_size = ((long long int)ceil(size / 32.) * 32);
        tensorflow::Status status = context_->allocate_temp(DT_UINT8, TensorShape{buf_size}, &buf);
        allocated_tensor_vector->push_back(buf);

        if (status != tensorflow::Status::OK())
            throw std::runtime_error("TF error: context->allocate_temp failed");

        auto flat = buf.flat<uint8>();
        void* ptr = (void*)flat.data();
        if (is_set_zero == true)
            cudaMemsetAsync(ptr, 0, buf_size, stream_);
        return ptr;
    }

    void free(void* ptr) const
    {
#ifndef NDEBUG
        printf("call from allocator free\n");
#endif
        return;
    }

    virtual ~Allocator()
    {
        allocated_tensor_vector->clear();
        delete allocated_tensor_vector;
    }
};
#endif

#ifdef TORCH_CUDA
template<>
class Allocator<AllocatorType::TH>: public IAllocator {
    std::shared_ptr<std::vector<torch::Tensor>> allocated_tensor_vector;

public:
    Allocator(): allocated_tensor_vector(std::make_shared<std::vector<torch::Tensor>>()) {}

    void* malloc(size_t size, const bool is_set_zero = true) const
    {
        int64_t buf_size = static_cast<int64_t>(ceil(size / 32.)) * 32;
        // TODO: test this later
        // torch::Tensor buf = is_set_zero ?
        //                     torch::zeros({buf_size}, torch::dtype(torch::kUInt8).device(torch::kCUDA)) :
        //                     torch::empty({buf_size}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
        torch::Tensor buf = torch::empty({buf_size}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
        allocated_tensor_vector->push_back(buf);
        return (*allocated_tensor_vector)[allocated_tensor_vector->size() - 1].data_ptr();
    }

    void free(void* ptr) const
    {
#ifndef NDEBUG
        printf("call from allocator free\n");
#endif
        return;
    }

    virtual ~Allocator()
    {
        allocated_tensor_vector->clear();
    }
};
#endif
}  // namespace fastertransformer
