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

#include "src/fastertransformer/utils/cuda_utils.h"

namespace fastertransformer {

template<typename T>
void deviceMalloc(T** ptr, size_t size, bool is_random_initialize = true);

template<typename T>
void deviceMemSetZero(T* ptr, int size);

template<typename T>
void deviceFree(T*& ptr);

template<typename T>
void deviceFill(T* devptr, int size, T value, cudaStream_t stream = 0);

template<typename T>
void cudaD2Hcpy(T* tgt, const T* src, const int size);

template<typename T>
void cudaH2Dcpy(T* tgt, const T* src, const int size);

template<typename T>
void cudaD2Dcpy(T* tgt, const T* src, const int size);

template<typename T>
void cudaAutoCpy(T* tgt, const T* src, const int size, cudaStream_t stream = NULL);

template<typename T>
void cudaRandomUniform(T* buffer, const int size);

template<typename T>
int loadWeightFromBin(T*                  ptr,
                      std::vector<size_t> shape,
                      std::string         filename,
                      FtCudaDataType      model_file_type = FtCudaDataType::FP32);

void invokeCudaD2DcpyHalf2Float(float* dst, half* src, const int size, cudaStream_t stream);
void invokeCudaD2DcpyFloat2Half(half* dst, float* src, const int size, cudaStream_t stream);

template<typename T_IN, typename T_OUT>
void invokeCudaD2DcpyConvert(T_OUT* tgt, const T_IN* src, const int size, cudaStream_t stream = 0);

inline bool checkIfFileExist(const std::string& file_path)
{
    std::ifstream in(file_path, std::ios::in | std::ios::binary);
    if (in.is_open()) {
        in.close();
        return true;
    }
    return false;
}

template<typename T>
void saveToBinary(const T* ptr, const int size, std::string filename);

template<typename T_IN, typename T_fake_type>
void invokeFakeCast(T_IN* input_ptr, const size_t size, cudaStream_t stream);

}  // namespace fastertransformer
