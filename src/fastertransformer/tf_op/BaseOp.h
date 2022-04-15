/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef TENSORFLOW_COMMON_OP_H
#define TENSORFLOW_COMMON_OP_H

#define EIGEN_USE_GPU

#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include "src/fastertransformer/utils/cuda_utils.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include <cuda_fp16.h>

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
namespace tf = tensorflow;
namespace ft = fastertransformer;

template<typename T>
class BaseOp: public tf::OpKernel {
public:
    explicit BaseOp(tf::OpKernelConstruction* context): tf::OpKernel(context)
    {
        try {
            ft::check_cuda_error(cublasCreate(&cublas_handle_));
            ft::check_cuda_error(cublasLtCreate(&cublaslt_handle_));
            cublas_wrapper_mutex_ = new std::mutex();
        }
        catch (std::runtime_error& error) {
            OP_REQUIRES(context, false, tf::errors::Internal(error.what()));
        }
    };

    ~BaseOp()
    {
        ft::check_cuda_error(cublasDestroy(cublas_handle_));
        ft::check_cuda_error(cublasLtDestroy(cublaslt_handle_));
        delete cublas_wrapper_mutex_;
    }

protected:
    template<typename DataType_>
    void get_tensor(tf::OpKernelContext* context, int tensor_id, const DataType_** tensor_ptr, int off_set = 0)
    {
        *tensor_ptr = reinterpret_cast<const DataType_*>(context->input(tensor_id).flat<T>().data()) + off_set;
        OP_REQUIRES(context, *tensor_ptr != nullptr, tf::errors::InvalidArgument("tensor %d is null", tensor_id));
    }

    template<typename DataType_>
    void get_tensor(tf::OpKernelContext* context, int tensor_id, DataType_** tensor_ptr, int off_set = 0)
    {
        *tensor_ptr = reinterpret_cast<DataType_*>(context->input(tensor_id).flat<T>().data()) + off_set;
        OP_REQUIRES(context, *tensor_ptr != nullptr, tf::errors::InvalidArgument("tensor %d is null", tensor_id));
    }

    void get_tensor(tf::OpKernelContext* context, int tensor_id, const int** tensor_ptr, int off_set = 0)
    {
        *tensor_ptr = reinterpret_cast<const int*>(context->input(tensor_id).flat<int>().data()) + off_set;
        OP_REQUIRES(context, *tensor_ptr != nullptr, tf::errors::InvalidArgument("tensor %d is null", tensor_id));
    }

    // convert the shape of TensorFlow Tensor to a vector
    std::vector<size_t> convert_shape(tf::Tensor tensor)
    {
        ft::FT_CHECK(tensor.dims() != -1);
        std::vector<size_t> v_shape;
        for (int i = 0; i < tensor.dims(); i++) {
            v_shape.push_back(tensor.dim_size(i));
        }
        return v_shape;
    }

    // convert TensorFlow Tensor to FasterTransformer Tensor
    ft::Tensor convert_tensor(tf::Tensor tensor)
    {
        if (std::is_same<T, Eigen::half>::value == true) {
            return ft::Tensor{
                ft::MEMORY_GPU, ft::getTensorType<half>(), convert_shape(tensor), (half*)(tensor.flat<T>().data())};
        }
#ifdef ENABLE_BF16
        if (std::is_same<T, Eigen::bfloat16>::value == true) {
            return ft::Tensor{ft::MEMORY_GPU,
                              ft::getTensorType<__nv_bfloat16>(),
                              convert_shape(tensor),
                              (__nv_bfloat16*)(tensor.flat<T>().data())};
        }
#endif
        else if (std::is_same<T, float>::value == true) {
            return ft::Tensor{
                ft::MEMORY_GPU, ft::getTensorType<float>(), convert_shape(tensor), (float*)(tensor.flat<T>().data())};
        }
        else {
            printf("[ERROR] Unknown data type \n");
            exit(-1);
        }
    }

    // convert int type TensorFlow Tensor to FasterTransformer Tensor
    ft::Tensor convert_int_tensor(tf::Tensor tensor)
    {
        return ft::Tensor{ft::MEMORY_GPU, ft::TYPE_INT32, convert_shape(tensor), (int*)(tensor.flat<int>().data())};
    }

    cublasHandle_t get_cublas_handler()
    {
        return cublas_handle_;
    }
    cublasLtHandle_t get_cublaslt_handler()
    {
        return cublaslt_handle_;
    }

    std::mutex* get_cublas_wrapper_mutex()
    {
        return cublas_wrapper_mutex_;
    }

private:
    cublasHandle_t cublas_handle_;
    cublasLtHandle_t cublaslt_handle_;
    std::mutex* cublas_wrapper_mutex_;
};

#endif
