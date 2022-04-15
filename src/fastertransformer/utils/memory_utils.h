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

#include "src/fastertransformer/utils/cuda_utils.h"

namespace fastertransformer {

template<typename T>
void deviceMalloc(T** ptr, int size, bool is_random_initialize = true);

template<typename T>
void deviceMemSetZero(T* ptr, int size);

template<typename T>
void deviceFree(T*& ptr);

template<typename T>
void deviceFill(T* devptr, int size, T value);

template<typename T>
void cudaD2Hcpy(T* tgt, const T* src, const int size);

template<typename T>
void cudaH2Dcpy(T* tgt, const T* src, const int size);

template<typename T>
void cudaD2Dcpy(T* tgt, const T* src, const int size);

template<typename T>
void cudaRandomUniform(T* buffer, const int size);

template<typename T>
int loadWeightFromBin(T* ptr,
                      std::vector<int> shape,
                      std::string filename,
                      FtCudaDataType model_file_type = FtCudaDataType::FP32);

void invokeCudaD2DcpyHalf2Float(float* dst, half* src, const int size, cudaStream_t stream);
void invokeCudaD2DcpyFloat2Half(half* dst, float* src, const int size, cudaStream_t stream);

}  // namespace fastertransformer
