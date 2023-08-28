/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2022.  Authored by Yuqing Ding.
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

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"
#include "src/fastertransformer/models/wenet/WenetKernels.h"
#include "src/fastertransformer/utils/cuda_utils.h"
namespace fastertransformer {


template<typename T>
__global__ void repeat_kv(T* dst, const T* src, const int kv_head_num, const int repeat_num, const int size_per_head, const int token_num)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < kv_head_num * token_num * size_per_head; id += blockDim.x * gridDim.x) {
        int token_id = id / (size_per_head * kv_head_num);
        int head_id = id / size_per_head % kv_head_num;
        int inner_id = id % size_per_head;
        for (int r = 0; r < repeat_num; r++) {
            int q_dst = 3 * repeat_num * kv_head_num * size_per_head * token_id + repeat_num * size_per_head * head_id + size_per_head * r + inner_id;
            int q_src = (repeat_num + 2) * kv_head_num * size_per_head * token_id + repeat_num * size_per_head * head_id + size_per_head * r + inner_id;
            dst[q_dst] = src[q_src];

            int k_dst = 3 * repeat_num * kv_head_num * size_per_head * token_id + repeat_num * kv_head_num * size_per_head + repeat_num * size_per_head * head_id + size_per_head * r + inner_id;
            int k_src = (repeat_num + 2) * kv_head_num * size_per_head * token_id + repeat_num * kv_head_num * size_per_head + size_per_head * head_id + inner_id;
            dst[k_dst] = src[k_src];

            int v_dst = 3 * repeat_num * kv_head_num * size_per_head * token_id + 2 * repeat_num * kv_head_num * size_per_head + repeat_num * size_per_head * head_id + size_per_head * r + inner_id;
            int v_src = (repeat_num + 2) * kv_head_num * size_per_head * token_id + (repeat_num + 1) * kv_head_num * size_per_head + size_per_head * head_id + inner_id;
            dst[v_dst] = src[v_src];
        }
    }
}

template<typename T>
void invokeRepeatKv(T* dst, const T* src, const int head_num, const int kv_head_num, const int size_per_head, const int token_num, cudaStream_t stream)
{
    dim3      block, grid;
    const int n = kv_head_num * token_num;
    if (n <= 1024) {
        block.x = n;
        grid.x  = size_per_head;
    }
    else {
        block.x = 1024;
        grid.x  = ceil(size_per_head * n / 1024.);
    }
    repeat_kv<T><<<grid, block, 0, stream>>>(dst, src, kv_head_num, head_num/kv_head_num, size_per_head, token_num);
}

template void
invokeRepeatKv(float* dst, const float* src, const int head_num, const int kv_head_num, const int size_per_head, const int token_num, cudaStream_t stream);
template void
invokeRepeatKv(half* dst, const half* src, const int head_num, const int kv_head_num, const int size_per_head, const int token_num, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeRepeatKv(__nv_bfloat16* dst,
                             const __nv_bfloat16* src,
                             const int head_num,
                             const int kv_head_num,
                             const int size_per_head,
                             const int token_num,
                             cudaStream_t stream);
#endif

}  // namespace fastertransformer
