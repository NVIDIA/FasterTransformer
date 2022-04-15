/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/string_utils.h"

#include "stdlib.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <string>
#include <vector>

namespace fastertransformer {

typedef enum datatype_enum {
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
    TYPE_BYTES,
    TYPE_BF16,
} DataType;

typedef enum memorytype_enum {
    MEMORY_CPU,
    MEMORY_CPU_PINNED,
    MEMORY_GPU
} MemoryType;

struct Tensor {
    const MemoryType where;
    const DataType type;
    const std::vector<size_t> shape;
    const void* data;  // TODO(bhseuh) modify from const void* to void* const

    Tensor(const MemoryType _where, const DataType _type, const std::vector<size_t> _shape, const void* _data):
        where(_where), type(_type), shape(_shape), data(_data)
    {
    }

    size_t size() const
    {
        size_t n_elements = 1;
        for (size_t s : shape) {
            n_elements *= s;
        }
        return n_elements;
    }

    std::string toString() const
    {
        std::string memtype_str;
        switch (where) {
            case MEMORY_CPU:
                memtype_str = "CPU";
                break;
            case MEMORY_CPU_PINNED:
                memtype_str = "CPU_PINNED";
                break;
            case MEMORY_GPU:
                memtype_str = "GPU";
                break;
        }

        std::string dtype_str = "";
        switch (type) {
            case TYPE_BOOL:
                dtype_str = "BOOL";
                break;
            case TYPE_UINT8:
                dtype_str = "UINT8";
                break;
            case TYPE_UINT16:
                dtype_str = "UINT16";
                break;
            case TYPE_UINT32:
                dtype_str = "UINT32";
                break;
            case TYPE_UINT64:
                dtype_str = "UINT64";
                break;
            case TYPE_INT8:
                dtype_str = "INT8";
                break;
            case TYPE_INT16:
                dtype_str = "INT16";
                break;
            case TYPE_INT32:
                dtype_str = "INT32";
                break;
            case TYPE_INT64:
                dtype_str = "INT64";
                break;
            case TYPE_BF16:
                dtype_str = "BF16";
                break;
            case TYPE_FP16:
                dtype_str = "FP16";
                break;
            case TYPE_FP32:
                dtype_str = "FP32";
                break;
            case TYPE_FP64:
                dtype_str = "FP64";
                break;
            case TYPE_BYTES:
                dtype_str = "BYTES";
                break;
            case TYPE_INVALID:
                dtype_str = "INVALID";
                break;
            default:
                break;
        }
        return fmtstr(
            "Tensor[where=%s, type=%s, shape=%s]", memtype_str.c_str(), dtype_str.c_str(), vec2str(shape).c_str());
    }

    template<typename T>
    inline T getVal(size_t index) const
    {
        FT_CHECK(where == MEMORY_CPU);
        FT_CHECK(data != nullptr);
        FT_CHECK_WITH_INFO(index < size(), "index is larger than buffer size");
        return ((T*)data)[index];
    }

    template<typename T>
    inline T getVal() const
    {
        return getVal<T>(0);
    }

    template<typename T>
    inline T* getPtr() const
    {
        return (T*)data;
    }

    inline void* getPtrWithOffset(size_t offset) const
    {
        if (data == nullptr) {
            return (void*)data;
        }
        else {
            FT_CHECK_WITH_INFO(offset < size(), "offset is larger than buffer size");
            return (void*)((char*)data + offset * getDataTypeByteNum(type));
        }
    }

    template<typename T>
    inline T* getPtrWithOffset(size_t offset) const
    {
        if (data == nullptr) {
            return (T*)data;
        }
        else {
            FT_CHECK_WITH_INFO(offset < size(), "offset is larger than buffer size");
            return ((T*)data) + offset;
        }
    }

    std::string getNumpyTypeDesc(DataType type) const
    {
        switch (type) {
            case TYPE_BOOL:
                return "?";
            case TYPE_UINT8:
                return "u1";
            case TYPE_UINT16:
                return "u2";
            case TYPE_UINT32:
                return "u4";
            case TYPE_UINT64:
                return "u8";
            case TYPE_INT8:
                return "i1";
            case TYPE_INT16:
                return "i2";
            case TYPE_INT32:
                return "i4";
            case TYPE_INT64:
                return "i8";
            case TYPE_FP16:
                return "f2";
            case TYPE_FP32:
                return "f4";
            case TYPE_FP64:
                return "f8";
            case TYPE_INVALID:
            default:;
        }
        return "";
    }

    int getDataTypeByteNum(DataType type) const
    {
        switch (type) {
            case TYPE_BOOL:
                return 1;
            case TYPE_UINT8:
                return 1;
            case TYPE_UINT16:
                return 2;
            case TYPE_UINT32:
                return 4;
            case TYPE_UINT64:
                return 8;
            case TYPE_INT8:
                return 1;
            case TYPE_INT16:
                return 2;
            case TYPE_INT32:
                return 4;
            case TYPE_INT64:
                return 8;
            case TYPE_FP16:
                return 2;
            case TYPE_BF16:
                return 2;
            case TYPE_FP32:
                return 4;
            case TYPE_FP64:
                return 8;
            case TYPE_INVALID:
                FT_CHECK(false);
            default:
                FT_CHECK(false);
        }
    }

    template<typename T>
    void save(const std::string& filename) const
    {
        // Save tensor to NPY 1.0 format (see https://numpy.org/neps/nep-0001-npy-format.html)
        void* cpu_data = (void*)data;
        bool is_data_temp = false;
        size_t tensor_size = size();
        if (where == MemoryType::MEMORY_GPU) {
            cpu_data = malloc(tensor_size * sizeof(T));
            is_data_temp = true;
            cudaDeviceSynchronize();
            cudaMemcpy(cpu_data, data, tensor_size * sizeof(T), cudaMemcpyDeviceToHost);
        }

        const char magic[] = "\x93"
                             "NUMPY";
        const uint8_t npy_major = 1;
        const uint8_t npy_minor = 0;

        std::stringstream header_stream;
        header_stream << "{'descr': '" << getNumpyTypeDesc(type) << "', 'fortran_order': False, 'shape': (";
        for (size_t i = 0; i < shape.size(); ++i) {
            header_stream << shape[i];
            if (i + 1 < shape.size() || shape.size() == 1) {
                header_stream << ", ";
            }
        }
        header_stream << ")}";
        int base_length = 6 + 4 + header_stream.str().size();
        int pad_length = 16 * ((base_length + 1 + 15) / 16);  // Take ceiling of base_length + 1 (for '\n' ending)
        for (int i = 0; i < pad_length - base_length; ++i) {
            header_stream << ((i == pad_length - base_length - 1) ? "\n" : "\x20");
        }
        std::string header = header_stream.str();
        const uint16_t header_len = header.size();

        FILE* f_ptr = fopen(filename.c_str(), "wb");
        if (f_ptr == nullptr) {
            printf("Unable to open %s for writing.\n", filename.c_str());
            exit(-1);
        }
        fwrite(magic, sizeof(char), sizeof(magic) - 1, f_ptr);
        fwrite(&npy_major, sizeof(uint8_t), 1, f_ptr);
        fwrite(&npy_minor, sizeof(uint8_t), 1, f_ptr);
        fwrite(&header_len, sizeof(uint16_t), 1, f_ptr);
        fwrite(header.c_str(), sizeof(char), header_len, f_ptr);
        fwrite(cpu_data, sizeof(T), tensor_size, f_ptr);

        fclose(f_ptr);

        if (is_data_temp) {
            free(cpu_data);
        }
    }
};

template<typename T>
DataType getTensorType()
{
    if (std::is_same<T, float>::value) {
        return TYPE_FP32;
    }
    else if (std::is_same<T, half>::value) {
        return TYPE_FP16;
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        return TYPE_BF16;
    }
#endif
    else if (std::is_same<T, int>::value) {
        return TYPE_INT32;
    }
    else if (std::is_same<T, int8_t>::value) {
        return TYPE_INT8;
    }
    else {
        return TYPE_INVALID;
    }
}

}  // namespace fastertransformer
