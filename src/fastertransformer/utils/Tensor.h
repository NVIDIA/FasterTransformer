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

#pragma once

#include <cuda_fp16.h>
#include <vector>
#include "stdlib.h"

namespace fastertransformer {

typedef enum datatype_enum
{
    TYPE_INVALID,
    TYPE_BOOL,
    TYPE_UINT8,
    TYPE_UINT16,
    TYPE_UINT32,
    TYPE_UINT64,
    TYPE_INT8,
    TYPE_INT16,
    TYPE_INT32,
    TYPE_INT64,
    TYPE_FP16,
    TYPE_FP32,
    TYPE_FP64,
    TYPE_BYTES
} DataType;

typedef enum memorytype_enum
{
    MEMORY_CPU,
    MEMORY_CPU_PINNED,
    MEMORY_GPU
} MemoryType;

struct Tensor {
    const MemoryType where;
    const DataType type;
    const std::vector<size_t> shape;
    const void* data; // TODO(bhseuh) modify from const void* to void* const

    Tensor(const MemoryType _where, const DataType _type, const std::vector<size_t> _shape, const void* _data):
        where(_where), type(_type), shape(_shape), data(_data)
    {
    }
};

template<typename T>
DataType getTensorType()
{
    if (std::is_same<T, float>::value)
        return TYPE_FP32;
    else if (std::is_same<T, half>::value)
        return TYPE_FP16;
    else if (std::is_same<T, int>::value)
        return TYPE_INT32;
    else if (std::is_same<T, int8_t>::value)
        return TYPE_INT8;
    else
        return TYPE_INVALID;
}

}  // namespace fastertransformer
