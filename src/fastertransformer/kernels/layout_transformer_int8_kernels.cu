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

#include "src/fastertransformer/kernels/layout_transformer_int8_kernels.h"

namespace fastertransformer {

// transpose matrix & transform COL32 to col-major
// input matrix is (m n) COL32
// output matrix is (n m) col-major
// grid((n+31)/32, (m+31)/32)
// block(32, 32)
template<typename T>
__global__ void transposeMatrix_COL32ToColMajor_kernel(T* dst, const T* src, const int m, const int n)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    bool check = ((x < n) && (y < m));
    // COL32_col = x >> 5 ; COL32_row = (y << 5) + (x & 31);
    // COL32_idx = (COL32_col << 5) * m + COL32_row = (x & 0xffffffe0)*m + (y << 5) + (x & 31)
    if (check) {
        dst[y * n + x] = __ldg(src + ((x & 0xffffffe0) * m + (y << 5) + (x & 31)));
    }
}

// transpose matrix & transform COL32 to col-major
// input matrix is (m n) COL32
// output matrix is (n m) col-major
// grid((n+31)/32, (m+31)/32)
// block(16, 32)
template<>
__global__ void transposeMatrix_COL32ToColMajor_kernel(half2* dst, const half2* src, const int m, const int n)
{

    int x = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    bool check = ((x < n) && (y < m));
    // COL32_col = x >> 5 ; COL32_row = (y << 5) + (x & 31);
    // COL32_idx = (COL32_col << 5) * m + COL32_row = (x & 0xffffffe0)*m + (y << 5) + (x & 31)
    if (check) {
        dst[(y * n + x) >> 1] = __ldg(src + (((x & 0xffffffe0) * m + (y << 5) + (x & 31)) >> 1));
    }
}

// transpose matrix & transform COL32 to col-major
// input matrix is (m n) COL32
// output matrix is (n m) col-major
template<typename T>
void invokeTransposeMatrixCOL32ToColMajor(T* dst, const T* src, const int m, const int n, cudaStream_t stream)
{
    assert(n % 32 == 0);
    if (sizeof(T) == sizeof(half)) {
        transposeMatrix_COL32ToColMajor_kernel<<<dim3((n + 31) / 32, (m + 31) / 32), dim3(16, 32), 0, stream>>>(
            (half2*)dst, (const half2*)src, m, n);
    }
    else {
        transposeMatrix_COL32ToColMajor_kernel<T>
            <<<dim3((n + 31) / 32, (m + 31) / 32), dim3(32, 32), 0, stream>>>(dst, src, m, n);
    }
}
template void invokeTransposeMatrixCOL32ToColMajor<float>(
    float* dst, const float* src, const int m, const int n, cudaStream_t stream);

template void
invokeTransposeMatrixCOL32ToColMajor<half>(half* dst, const half* src, const int m, const int n, cudaStream_t stream);

template void invokeTransposeMatrixCOL32ToColMajor<int8_t>(
    int8_t* dst, const int8_t* src, const int m, const int n, cudaStream_t stream);

// transpose matrix & transfrom col-major to COL32
// input matrix is (m, n) col-major
// output matrix is (n, m) COL32
// m should be a mutiple of 32
// grid((m+31)/32, (n+31)/32)
// block(32, 32)
template<typename T>
__global__ void transposeMatrix_colMajorToCOL32_kernel(T* dst, const T* src, const int m, const int n)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    bool check = ((x < m) && (y < n));
    if (check) {

        // COL32_col = x >> 5 ; COL32_row = (y << 5) + (x & 31);
        // COL32_idx = (COL32_col << 5) * n + COL32_row = (x & 0xffffffe0)*n + (y << 5) + (x & 31)
        dst[(x & 0xffffffe0) * n + (y << 5) + (x & 31)] = __ldg(src + y * m + x);
    }
}

// transpose matrix & transfrom col-major to COL32
// input matrix is (m, n) col-major
// output matrix is (n, m) COL32
// m should be a mutiple of 32
// grid((m+31)/32, (n+31)/32)
// block(16, 32)
template<>
__global__ void transposeMatrix_colMajorToCOL32_kernel(half2* dst, const half2* src, const int m, const int n)
{

    int x = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    bool check = ((x < m) && (y < n));
    if (check) {

        // COL32_col = x >> 5 ; COL32_row = (y << 5) + (x & 31);
        // COL32_idx = (COL32_col << 5) * n + COL32_row = (x & 0xffffffe0)*n + (y << 5) + (x & 31)
        dst[((x & 0xffffffe0) * n + (y << 5) + (x & 31)) >> 1] = __ldg(src + ((y * m + x) >> 1));
    }
}

// transpose matrix & transfrom col-major to COL32
// input matrix is (m, n) col-major
// output matrix is (n, m) COL32, using char4 to write out
// m should be a mutiple of 32
// grid((m+31)/32, (n+31)/32)
// block(8, 32)
template<typename T>
void invokeTransposeMatrixColMajorToCOL32(T* dst, const T* src, const int m, const int n, cudaStream_t stream)
{
    assert(m % 32 == 0);
    if (sizeof(T) == sizeof(float)) {
        transposeMatrix_colMajorToCOL32_kernel<T>
            <<<dim3((m + 31) / 32, (n + 31) / 32), dim3(32, 32), 0, stream>>>(dst, src, m, n);
    }
    else if (sizeof(T) == sizeof(half)) {
        transposeMatrix_colMajorToCOL32_kernel<<<dim3((m + 31) / 32, (n + 31) / 32), dim3(16, 32), 0, stream>>>(
            (half2*)dst, (const half2*)src, m, n);
    }
}

template void invokeTransposeMatrixColMajorToCOL32<float>(
    float* dst, const float* src, const int m, const int n, cudaStream_t stream);

template void
invokeTransposeMatrixColMajorToCOL32<half>(half* dst, const half* src, const int m, const int n, cudaStream_t stream);

// transpose matrix & transfrom col-major to COL32 & quantize
// input matrix is (m, n) col-major
// output matrix is (n, m) COL32, using char4 to write out
// m should be a mutiple of 32
// grid((m+31)/32, (n+31)/32)
// block(8, 32)
template<typename T>
__global__ void transposeMatrix_colMajorToCOL32_quantize_kernel(
    char4* dst, const T* src, const int m, const int n, const float* scale_ptr)
{

    const float scale = __ldg(scale_ptr);

    int x = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    bool check = ((x < m) && (y < n));
    if (check) {
        char4 tmp4;
        tmp4.x = float_to_int8_rn(static_cast<float>(__ldg(src + y * m + x)) * scale);
        tmp4.y = float_to_int8_rn(static_cast<float>(__ldg(src + y * m + x + 1)) * scale);
        tmp4.z = float_to_int8_rn(static_cast<float>(__ldg(src + y * m + x + 2)) * scale);
        tmp4.w = float_to_int8_rn(static_cast<float>(__ldg(src + y * m + x + 3)) * scale);

        // COL32_col = x >> 5 ; COL32_row = (y << 5) + (x & 31);
        // COL32_idx = (COL32_col << 5) * n + COL32_row = (x & 0xffffffe0)*n + (y << 5) + (x & 31)

        dst[((x & 0xffffffe0) * n + (y << 5) + (x & 31)) >> 2] = tmp4;
    }
}

// transpose matrix & transfrom col-major to COL32 & quantize
// input matrix is (m, n) col-major
// output matrix is (n, m) COL32, using char4 to write out
// m should be a mutiple of 32
// grid((m+31)/32, (n+31)/32)
// block(8, 32)
template<typename T>
void invokeTransposeMatrixColMajorToCOL32Quantize(
    int8_t* dst, const T* src, const int m, const int n, const float* scale_ptr, cudaStream_t stream)
{
    assert(m % 32 == 0);
    transposeMatrix_colMajorToCOL32_quantize_kernel<T>
        <<<dim3((m + 31) / 32, (n + 31) / 32), dim3(8, 32), 0, stream>>>((char4*)dst, src, m, n, scale_ptr);
}

template void invokeTransposeMatrixColMajorToCOL32Quantize<float>(
    int8_t* dst, const float* src, const int m, const int n, const float* scale_ptr, cudaStream_t stream);

template void invokeTransposeMatrixColMajorToCOL32Quantize<half>(
    int8_t* dst, const half* src, const int m, const int n, const float* scale_ptr, cudaStream_t stream);

// transfrom row-major to COL32
// input matrix is (m, n) row-major
// output matrix is (m, n) COL32
// n should be a mutiple of 32
// grid((n+31)/32, (m+31)/32)
// block(8, 32)
__global__ void rowMajorToCOL32_kernel(char4* dst, const char4* src, const int m, const int n)
{

    int n_id = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
    int m_id = blockIdx.y * blockDim.y + threadIdx.y;

    bool check = ((m_id < m) && (n_id < n));
    if (check) {

        // COL32_col = n_id >> 5 ; COL32_row = (m_id << 5) + (n_id & 31);
        // COL32_idx = (COL32_col << 5) * m + COL32_row = (n_id & 0xffffffe0)*m + (m_id << 5) + (n_id & 31)
        dst[((n_id & 0xffffffe0) * m + (m_id << 5) + (n_id & 31)) >> 2] = __ldg(src + ((m_id * n + n_id) >> 2));
    }
}

// transfrom row-major to COL32
// input matrix is (m, n) row-major
// output matrix is (m, n) COL32
// n should be a mutiple of 32
// grid((n+31)/32, (m+31)/32)
// block(8, 32)
void invokeRowMajorToCOL32(int8_t* dst, const int8_t* src, const int m, const int n, cudaStream_t stream)
{
    assert(n % 32 == 0);
    rowMajorToCOL32_kernel<<<dim3((n + 31) / 32, (m + 31) / 32), dim3(8, 32), 0, stream>>>(
        (char4*)dst, (const char4*)src, m, n);
}

}  // namespace fastertransformer