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

#include "src/fastertransformer/kernels/matrix_transpose_kernels.h"

namespace fastertransformer {

// src is [k, n] row-major
// dst is [n, k] row-major
template<typename T>
__global__ void matrix_transpose(T* dst, const T* src, const int k, const int n)
{
    __shared__ T shm[32][33];
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    int n_idx = blockIdx.x * 32 + tidx;
    int k_idx = blockIdx.y * 32 + tidy;
    if (n_idx < n && k_idx < k) {
        shm[tidx][tidy] = src[k_idx * n + n_idx];
    }

    __syncthreads();
    n_idx = blockIdx.x * 32 + tidy;
    k_idx = blockIdx.y * 32 + tidx;
    if (n_idx < n && k_idx < k) {
        dst[n_idx * k + k_idx] = shm[tidy][tidx];
    }
}

// src is [k, n] row-major
// dst is [n, k] row-major
template<typename T>
void invokeMatrixTranspose(T* dst, const T* src, const int k, const int n, cudaStream_t stream)
{
    dim3 grid(n / 32, k / 32);
    dim3 block(32, 32);
    matrix_transpose<<<grid, block, 0, stream>>>(dst, src, k, n);
}

template void invokeMatrixTranspose(float* dst, const float* src, const int m, const int n, cudaStream_t stream);

template void invokeMatrixTranspose(half* dst, const half* src, const int m, const int n, cudaStream_t stream);

}  // namespace fastertransformer
