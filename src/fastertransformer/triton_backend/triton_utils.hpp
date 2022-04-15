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

#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"
#include "src/fastertransformer/utils/Tensor.h"

namespace ft = fastertransformer;

template<typename T>
void move_tensor_H2D(const triton::Tensor &tensor, T* &d_ptr)
{
    if (tensor.where == triton::MEMORY_GPU) {
        return;
    }

    size_t tensor_size = 1;
    for (auto t : tensor.shape) {
        tensor_size *= t;
    }
    ft::deviceMalloc(&d_ptr, tensor_size, false);
    ft::cudaH2Dcpy(d_ptr, (T*) tensor.data, tensor_size);
}

template<typename T>
ft::Tensor as_GPU_tensor(const triton::Tensor &tensor, T* d_ptr)
{
    return ft::Tensor {ft::MEMORY_GPU,
                       triton::Tensor::convertTritonTypeToFt(tensor.type),
                       tensor.shape,
                       tensor.where == triton::MEMORY_CPU ? d_ptr : tensor.data};
}
