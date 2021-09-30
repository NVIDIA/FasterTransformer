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

#include <curand_kernel.h>

#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
void deviceMalloc(T** ptr, int size, bool is_random_initialize)
{
    check_cuda_error(cudaMalloc((void**)(ptr), sizeof(T) * size));
    if (is_random_initialize) {
        cudaRandomUniform(*ptr, size);
    }
}

template void deviceMalloc(float** ptr, int size, bool is_random_initialize);
template void deviceMalloc(half** ptr, int size, bool is_random_initialize);
template void deviceMalloc(int** ptr, int size, bool is_random_initialize);
template void deviceMalloc(bool** ptr, int size, bool is_random_initialize);
template void deviceMalloc(char** ptr, int size, bool is_random_initialize);

template<typename T>
void deviceMemSetZero(T* ptr, int size)
{
    check_cuda_error(cudaMemset(static_cast<void*>(ptr), 0, sizeof(T) * size));
}

template void deviceMemSetZero(float* ptr, int size);
template void deviceMemSetZero(half* ptr, int size);
template void deviceMemSetZero(int* ptr, int size);
template void deviceMemSetZero(bool* ptr, int size);

template<typename T>
void deviceFree(T*& ptr)
{
    if (ptr != NULL) {
        check_cuda_error(cudaFree(ptr));
        ptr = NULL;
    }
}

template void deviceFree(float*& ptr);
template void deviceFree(half*& ptr);
template void deviceFree(int*& ptr);
template void deviceFree(bool*& ptr);
template void deviceFree(char*& ptr);

template<typename T>
void deviceFill(T* devptr, int size, T value)
{
    T* arr = new T[size];
    std::fill(arr, arr + size, value);
    check_cuda_error(cudaMemcpy(devptr, arr, sizeof(T) * size, cudaMemcpyHostToDevice));
    delete[] arr;
}

template void deviceFill(float* devptr, int size, float value);
template void deviceFill(half* devptr, int size, half value);
template void deviceFill(int* devptr, int size, int value);

template<typename T>
void cudaD2Hcpy(T* tgt, const T* src, const int size)
{
    check_cuda_error(cudaMemcpy(tgt, src, sizeof(T) * size, cudaMemcpyDeviceToHost));
}

template void cudaD2Hcpy(float* tgt, const float* src, int size);
template void cudaD2Hcpy(half* tgt, const half* src, int size);
template void cudaD2Hcpy(int* tgt, const int* src, int size);
template void cudaD2Hcpy(bool* tgt, const bool* src, int size);

template<typename T>
void cudaH2Dcpy(T* tgt, const T* src, const int size)
{
    check_cuda_error(cudaMemcpy(tgt, src, sizeof(T) * size, cudaMemcpyHostToDevice));
}

template void cudaH2Dcpy(float* tgt, const float* src, int size);
template void cudaH2Dcpy(half* tgt, const half* src, int size);
template void cudaH2Dcpy(int* tgt, const int* src, int size);
template void cudaH2Dcpy(bool* tgt, const bool* src, int size);

template<typename T>
void cudaD2Dcpy(T* tgt, const T* src, const int size)
{
    check_cuda_error(cudaMemcpy(tgt, src, sizeof(T) * size, cudaMemcpyDeviceToDevice));
}

template void cudaD2Dcpy(float* tgt, const float* src, int size);
template void cudaD2Dcpy(half* tgt, const half* src, int size);
template void cudaD2Dcpy(int* tgt, const int* src, int size);
template void cudaD2Dcpy(bool* tgt, const bool* src, int size);

template<typename T>
__global__ void cuda_random_uniform_kernel(T* buffer, const int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t local_state;
    curand_init((T)1337.f, idx, 0, &local_state);
    for (int index = idx; index < size; index += blockDim.x * gridDim.x) {
        buffer[index] = (T)(curand_uniform(&local_state) * 0.2f - 0.1f);
    }
}

template<typename T>
__global__ void cuda_random_uniform_kernel(int* buffer, const int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int index = idx; index < size; index += blockDim.x * gridDim.x) {
        buffer[index] = 0;
    }
}

template<typename T>
__global__ void cuda_random_uniform_kernel(bool* buffer, const int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int index = idx; index < size; index += blockDim.x * gridDim.x) {
        buffer[index] = false;
    }
}

template<typename T>
__global__ void cuda_random_uniform_kernel(char* buffer, const int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int index = idx; index < size; index += blockDim.x * gridDim.x) {
        buffer[index] = '\0';
    }
}

template<typename T>
void cudaRandomUniform(T* buffer, const int size)
{
    cuda_random_uniform_kernel<T><<<256, 256>>>(buffer, size);
}

template void cudaRandomUniform(float* buffer, const int size);
template void cudaRandomUniform(half* buffer, const int size);
template void cudaRandomUniform(int* buffer, const int size);
template void cudaRandomUniform(bool* buffer, const int size);
template void cudaRandomUniform(char* buffer, const int size);

template<typename T>
int loadWeightFromBin(T* ptr, std::vector<int> shape, std::string filename)
{
    if (shape.size() > 2) {
        printf("[ERROR] shape should have less than two dims \n");
        return -1;
    }
    int dim0 = shape[0], dim1 = 1;
    if (shape.size() == 2) {
        dim1 = shape[1];
    }
    size_t size = dim0 * dim1;

    std::vector<float> host_array(size);

    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        printf("[WARNING] file %s cannot be opened, loading model fails! \n", filename.c_str());
        return 0;
    }

    size_t float_data_size = sizeof(float) * size;
    in.read((char*)host_array.data(), float_data_size);

    size_t in_get_size = in.gcount();
    if (in_get_size != float_data_size) {
        printf("[WARNING] file %s only has %ld, but request %ld, loading model fails! \n",
               filename.c_str(),
               in_get_size,
               float_data_size);
        return 0;
    }

    if (std::is_same<T, float>::value == true)
        cudaH2Dcpy(ptr, (T*)host_array.data(), size);
    else {
        std::vector<T> host_array_2(size);
        for (size_t i = 0; i < size; i++) {
            host_array_2[i] = __float2half(host_array[i]);
        }
        cudaH2Dcpy(ptr, (T*)host_array_2.data(), size);
    }
    return 0;
}

template int loadWeightFromBin(float* ptr, std::vector<int> shape, std::string filename);
template int loadWeightFromBin(half* ptr, std::vector<int> shape, std::string filename);


__global__ void cudaD2DcpyHalf2Float(float* dst, half* src, const int size)
{
    for(int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x) {
        dst[tid] = __half2float(src[tid]);
    }
}

void invokeCudaD2DcpyHalf2Float(float* dst, half* src, const int size, cudaStream_t stream)
{
    cudaD2DcpyHalf2Float<<<256, 256, 0, stream>>>(dst, src, size);
}

__global__ void cudaD2DcpyFloat2Half(half* dst, float* src, const int size)
{
    for(int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x) {
        dst[tid] = __float2half(src[tid]);
    }
}

void invokeCudaD2DcpyFloat2Half(half* dst, float* src, const int size, cudaStream_t stream)
{
    cudaD2DcpyFloat2Half<<<256, 256, 0, stream>>>(dst, src, size);
}

}  // namespace fastertransformer