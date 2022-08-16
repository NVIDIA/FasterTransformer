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

#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include <curand_kernel.h>

namespace fastertransformer {

template<typename T>
void deviceMalloc(T** ptr, size_t size, bool is_random_initialize)
{
    FT_CHECK_WITH_INFO(size >= 0, "Ask deviceMalloc size " + std::to_string(size) + "< 0 is invalid.");
    check_cuda_error(cudaMalloc((void**)(ptr), sizeof(T) * size));
    if (is_random_initialize) {
        cudaRandomUniform(*ptr, size);
    }
}

template void deviceMalloc(float** ptr, size_t size, bool is_random_initialize);
template void deviceMalloc(half** ptr, size_t size, bool is_random_initialize);
#ifdef ENABLE_BF16
template void deviceMalloc(__nv_bfloat16** ptr, size_t size, bool is_random_initialize);
#endif
template void deviceMalloc(uint16_t** ptr, size_t size, bool is_random_initialize);
template void deviceMalloc(int** ptr, size_t size, bool is_random_initialize);
template void deviceMalloc(bool** ptr, size_t size, bool is_random_initialize);
template void deviceMalloc(char** ptr, size_t size, bool is_random_initialize);
template void deviceMalloc(int8_t** ptr, size_t size, bool is_random_initialize);

template<typename T>
void deviceMemSetZero(T* ptr, int size)
{
    check_cuda_error(cudaMemset(static_cast<void*>(ptr), 0, sizeof(T) * size));
}

template void deviceMemSetZero(float* ptr, int size);
template void deviceMemSetZero(half* ptr, int size);
template void deviceMemSetZero(int* ptr, int size);
template void deviceMemSetZero(uint32_t* ptr, int size);
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
#ifdef ENABLE_BF16
template void deviceFree(__nv_bfloat16*& ptr);
#endif
template void deviceFree(unsigned short*& ptr);
template void deviceFree(int*& ptr);
template void deviceFree(bool*& ptr);
template void deviceFree(char*& ptr);
template void deviceFree(int8_t*& ptr);

template<typename T>
void deviceFill(T* devptr, int size, T value, cudaStream_t stream)
{
    T* arr = new T[size];
    std::fill(arr, arr + size, value);
    check_cuda_error(cudaMemcpyAsync(devptr, arr, sizeof(T) * size, cudaMemcpyHostToDevice, stream));
    delete[] arr;
}

template void deviceFill(float* devptr, int size, float value, cudaStream_t stream);
template void deviceFill(half* devptr, int size, half value, cudaStream_t stream);
#ifdef ENABLE_BF16
template void deviceFill(__nv_bfloat16* devptr, int size, __nv_bfloat16 value, cudaStream_t stream);
#endif
template void deviceFill(int* devptr, int size, int value, cudaStream_t stream);
template void deviceFill(bool* devptr, int size, bool value, cudaStream_t stream);

template<typename T>
void cudaD2Hcpy(T* tgt, const T* src, const int size)
{
    check_cuda_error(cudaMemcpy(tgt, src, sizeof(T) * size, cudaMemcpyDeviceToHost));
}

template void cudaD2Hcpy(float* tgt, const float* src, int size);
template void cudaD2Hcpy(half* tgt, const half* src, int size);
#ifdef ENABLE_BF16
template void cudaD2Hcpy(__nv_bfloat16* tgt, const __nv_bfloat16* src, int size);
#endif
template void cudaD2Hcpy(int* tgt, const int* src, int size);
template void cudaD2Hcpy(bool* tgt, const bool* src, int size);
template void cudaD2Hcpy(unsigned long long* tgt, const unsigned long long* src, int size);
template void cudaD2Hcpy(unsigned int* tgt, const unsigned int* src, int size);

template<typename T>
void cudaH2Dcpy(T* tgt, const T* src, const int size)
{
    check_cuda_error(cudaMemcpy(tgt, src, sizeof(T) * size, cudaMemcpyHostToDevice));
}

template void cudaH2Dcpy(float* tgt, const float* src, int size);
template void cudaH2Dcpy(half* tgt, const half* src, int size);
#ifdef ENABLE_BF16
template void cudaH2Dcpy(__nv_bfloat16* tgt, const __nv_bfloat16* src, int size);
#endif
template void cudaH2Dcpy(int* tgt, const int* src, int size);
template void cudaH2Dcpy(bool* tgt, const bool* src, int size);
template void cudaH2Dcpy(unsigned long long* tgt, const unsigned long long* src, int size);
template void cudaH2Dcpy(unsigned int* tgt, const unsigned int* src, int size);

template<typename T>
void cudaD2Dcpy(T* tgt, const T* src, const int size)
{
    check_cuda_error(cudaMemcpy(tgt, src, sizeof(T) * size, cudaMemcpyDeviceToDevice));
}

template void cudaD2Dcpy(float* tgt, const float* src, int size);
template void cudaD2Dcpy(half* tgt, const half* src, int size);
#ifdef ENABLE_BF16
template void cudaD2Dcpy(__nv_bfloat16* tgt, const __nv_bfloat16* src, int size);
#endif
template void cudaD2Dcpy(int* tgt, const int* src, int size);
template void cudaD2Dcpy(bool* tgt, const bool* src, int size);
template void cudaD2Dcpy(int8_t* tgt, const int8_t* src, int size);
template void cudaD2Dcpy(unsigned long long* tgt, const unsigned long long* src, int size);

template<typename T>
void cudaAutoCpy(T* tgt, const T* src, const int size, cudaStream_t stream)
{
    if (stream != NULL) {
        check_cuda_error(cudaMemcpyAsync(tgt, src, sizeof(T) * size, cudaMemcpyDefault, stream));
    }
    else {
        check_cuda_error(cudaMemcpy(tgt, src, sizeof(T) * size, cudaMemcpyDefault));
    }
}

template void cudaAutoCpy(float* tgt, const float* src, int size, cudaStream_t stream);
template void cudaAutoCpy(half* tgt, const half* src, int size, cudaStream_t stream);
#ifdef ENABLE_BF16
template void cudaAutoCpy(__nv_bfloat16* tgt, const __nv_bfloat16* src, int size, cudaStream_t stream);
#endif
template void cudaAutoCpy(int* tgt, const int* src, int size, cudaStream_t stream);
template void cudaAutoCpy(bool* tgt, const bool* src, int size, cudaStream_t stream);
template void cudaAutoCpy(int8_t* tgt, const int8_t* src, int size, cudaStream_t stream);
template void cudaAutoCpy(uint* tgt, const uint* src, int size, cudaStream_t stream);
template void cudaAutoCpy(unsigned long long* tgt, const unsigned long long* src, int size, cudaStream_t stream);

template void cudaAutoCpy(float const** tgt, float const* const* src, int size, cudaStream_t stream);
template void cudaAutoCpy(half const** tgt, half const* const* src, int size, cudaStream_t stream);
#ifdef ENABLE_BF16
template void cudaAutoCpy(__nv_bfloat16 const** tgt, __nv_bfloat16 const* const* src, int size, cudaStream_t stream);
#endif
template void cudaAutoCpy(int const** tgt, int const* const* src, int size, cudaStream_t stream);
template void cudaAutoCpy(bool const** tgt, bool const* const* src, int size, cudaStream_t stream);
template void cudaAutoCpy(int8_t const** tgt, int8_t const* const* src, int size, cudaStream_t stream);
template void
cudaAutoCpy(unsigned long long const** tgt, unsigned long long const* const* src, int size, cudaStream_t stream);

template<typename T>
__global__ void cuda_random_uniform_kernel(T* buffer, const int size)
{
    const int     idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t local_state;
    curand_init((unsigned long long int)1337, idx, 0, &local_state);
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
#ifdef ENABLE_BF16
template void cudaRandomUniform(__nv_bfloat16* buffer, const int size);
#endif
template void cudaRandomUniform(int* buffer, const int size);
template void cudaRandomUniform(bool* buffer, const int size);
template void cudaRandomUniform(char* buffer, const int size);

template<typename T_IN, typename T_OUT>
__host__ __device__ inline T_OUT convert_to_type(T_IN val)
{
    return (T_OUT)val;
}

#ifdef ENABLE_BF16
template<>
__host__ __device__ inline __nv_bfloat16 convert_to_type<float, __nv_bfloat16>(float val)
{
    return __float2bfloat16(val);
}

template<>
__host__ __device__ inline __nv_bfloat16 convert_to_type<half, __nv_bfloat16>(half val)
{
    return __float2bfloat16(__half2float(val));
}

template<>
__host__ __device__ inline float convert_to_type<__nv_bfloat16, float>(__nv_bfloat16 val)
{
    return __bfloat162float(val);
}

template<>
__host__ __device__ inline half convert_to_type<__nv_bfloat16, half>(__nv_bfloat16 val)
{
    return __float2half(__bfloat162float(val));
}
#endif  // ENABLE_BF16

template<typename T, typename T_IN>
int loadWeightFromBinFunc(T* ptr, std::vector<size_t> shape, std::string filename)
{
    if (shape.size() > 2) {
        printf("[ERROR] shape should have less than two dims \n");
        return -1;
    }
    size_t dim0 = shape[0], dim1 = 1;
    if (shape.size() == 2) {
        dim1 = shape[1];
    }
    size_t size = dim0 * dim1;
    if (size == 0) {
        FT_LOG_WARNING("shape is zero, skip loading weight from file %s \n", filename.c_str());
        return 0;
    }
    std::vector<T_IN> host_array(size);
    std::ifstream     in(filename, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        FT_LOG_WARNING("file %s cannot be opened, loading model fails! \n", filename.c_str());
        return 0;
    }

    size_t loaded_data_size = sizeof(T_IN) * size;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);

    FT_LOG_DEBUG("Read " + std::to_string(loaded_data_size) + " bytes from " + filename);
    in.read((char*)host_array.data(), loaded_data_size);

    size_t in_get_size = in.gcount();
    if (in_get_size != loaded_data_size) {
        FT_LOG_WARNING("file %s only has %ld, but request %ld, loading model fails! \n",
                       filename.c_str(),
                       in_get_size,
                       loaded_data_size);
        return 0;
    }

    if (std::is_same<T, T_IN>::value == true) {
        cudaH2Dcpy(ptr, (T*)host_array.data(), size);
    }
    else {
        T_IN* ptr_2 = nullptr;
        deviceMalloc(&ptr_2, size, false);
        cudaH2Dcpy(ptr_2, host_array.data(), size);
        invokeCudaD2DcpyConvert(ptr, ptr_2, size);
        deviceFree(ptr_2);
    }
    in.close();
    return 0;
}

template int loadWeightFromBinFunc<float, float>(float* ptr, std::vector<size_t> shape, std::string filename);
template int loadWeightFromBinFunc<half, float>(half* ptr, std::vector<size_t> shape, std::string filename);
#ifdef ENABLE_BF16
template int
loadWeightFromBinFunc<__nv_bfloat16, float>(__nv_bfloat16* ptr, std::vector<size_t> shape, std::string filename);
#endif
template int loadWeightFromBinFunc<float, half>(float* ptr, std::vector<size_t> shape, std::string filename);
template int loadWeightFromBinFunc<half, half>(half* ptr, std::vector<size_t> shape, std::string filename);
#ifdef ENABLE_BF16
template int
loadWeightFromBinFunc<__nv_bfloat16, half>(__nv_bfloat16* ptr, std::vector<size_t> shape, std::string filename);
template int loadWeightFromBinFunc<float, __nv_bfloat16>(float* ptr, std::vector<size_t> shape, std::string filename);
template int loadWeightFromBinFunc<half, __nv_bfloat16>(half* ptr, std::vector<size_t> shape, std::string filename);
template int loadWeightFromBinFunc<__nv_bfloat16, __nv_bfloat16>(__nv_bfloat16*      ptr,
                                                                 std::vector<size_t> shape,
                                                                 std::string         filename);
#endif  // ENABLE_BF16

template<typename T>
int loadWeightFromBin(T* ptr, std::vector<size_t> shape, std::string filename, FtCudaDataType model_file_type)
{
    switch (model_file_type) {
        case FtCudaDataType::FP32:
            loadWeightFromBinFunc<T, float>(ptr, shape, filename);
            break;
        case FtCudaDataType::FP16:
            loadWeightFromBinFunc<T, half>(ptr, shape, filename);
            break;
#ifdef ENABLE_BF16
        case FtCudaDataType::BF16:
            loadWeightFromBinFunc<T, __nv_bfloat16>(ptr, shape, filename);
            break;
#endif
        default:
            FT_LOG_ERROR("Does not support FtCudaDataType=%d", model_file_type);
            FT_CHECK(false);
    }
    return 0;
}

template int
loadWeightFromBin(float* ptr, std::vector<size_t> shape, std::string filename, FtCudaDataType model_file_type);
template int
loadWeightFromBin(half* ptr, std::vector<size_t> shape, std::string filename, FtCudaDataType model_file_type);
#ifdef ENABLE_BF16
template int
loadWeightFromBin(__nv_bfloat16* ptr, std::vector<size_t> shape, std::string filename, FtCudaDataType model_file_type);
#endif

template<typename T_IN, typename T_OUT>
__global__ void cudaD2DcpyConvert(T_OUT* dst, const T_IN* src, const int size)
{
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x) {
        dst[tid] = convert_to_type<T_IN, T_OUT>(src[tid]);
    }
}

template<typename T_IN, typename T_OUT>
void invokeCudaD2DcpyConvert(T_OUT* tgt, const T_IN* src, const int size, cudaStream_t stream)
{
    cudaD2DcpyConvert<<<256, 256, 0, stream>>>(tgt, src, size);
}

template void invokeCudaD2DcpyConvert(float* tgt, const float* src, const int size, cudaStream_t stream);
template void invokeCudaD2DcpyConvert(half* tgt, const float* src, const int size, cudaStream_t stream);
template void invokeCudaD2DcpyConvert(float* tgt, const half* src, const int size, cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeCudaD2DcpyConvert(__nv_bfloat16* tgt, const float* src, const int size, cudaStream_t stream);
template void invokeCudaD2DcpyConvert(float* tgt, const __nv_bfloat16* src, const int size, cudaStream_t stream);
#endif  // ENABLE_BF16

void invokeCudaD2DcpyHalf2Float(float* dst, half* src, const int size, cudaStream_t stream)
{
    invokeCudaD2DcpyConvert(dst, src, size, stream);
}

void invokeCudaD2DcpyFloat2Half(half* dst, float* src, const int size, cudaStream_t stream)
{
    invokeCudaD2DcpyConvert(dst, src, size, stream);
}

template<typename T>
void saveToBinary(const T* ptr, const int size, std::string filename)
{

    std::vector<T> h_ptr(size);
    cudaD2Hcpy(h_ptr.data(), ptr, size);
    std::vector<float> float_ptr(size);
    for (int i = 0; i < size; i++) {
        float_ptr[i] = (float)h_ptr[i];
    }

    std::ofstream out(filename, std::ios::out | std::ios::binary);
    FT_CHECK_WITH_INFO(out.is_open(), "Fail to open file " + filename);

    out.write((char*)float_ptr.data(), size * sizeof(float));
}

template void saveToBinary(const float* ptr, const int size, std::string filename);
template void saveToBinary(const half* ptr, const int size, std::string filename);
#ifdef ENABLE_BF16
template void saveToBinary(const __nv_bfloat16* ptr, const int size, std::string filename);
#endif  // ENABLE_BF16

template<>
void saveToBinary(const int* ptr, const int size, std::string filename)
{
    std::vector<int> h_ptr(size);
    cudaD2Hcpy(h_ptr.data(), ptr, size);
    std::ofstream out(filename, std::ios::out | std::ios::binary);
    FT_CHECK_WITH_INFO(out.is_open(), "Fail to open file " + filename);
    out.write((char*)h_ptr.data(), size * sizeof(int));
}

template<typename T_IN, typename T_fake_type>
__global__ void fakeCast(T_IN* input_ptr, const size_t size)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        T_fake_type tmp_val = (T_fake_type)((float)input_ptr[i]);
        tmp_val             = tmp_val * (T_fake_type)(1.0f);
        input_ptr[i]        = (T_IN)((float)tmp_val);
    }
}

template<typename T_IN, typename T_fake_type>
void invokeFakeCast(T_IN* input_ptr, const size_t size, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((size + 255) / 256);
    fakeCast<T_IN, T_fake_type><<<grid, block, 0, stream>>>(input_ptr, size);
}

#ifdef ENABLE_BF16
template void invokeFakeCast<float, __nv_bfloat16>(float* input_ptr, const size_t size, cudaStream_t stream);
template void
invokeFakeCast<__nv_bfloat16, __nv_bfloat16>(__nv_bfloat16* input_ptr, const size_t size, cudaStream_t stream);
template void invokeFakeCast<half, __nv_bfloat16>(half* input_ptr, const size_t size, cudaStream_t stream);
#endif
template void invokeFakeCast<float, half>(float* input_ptr, const size_t size, cudaStream_t stream);
template void invokeFakeCast<float, float>(float* input_ptr, const size_t size, cudaStream_t stream);

}  // namespace fastertransformer
