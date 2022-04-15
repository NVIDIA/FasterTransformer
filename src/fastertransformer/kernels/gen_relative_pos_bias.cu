/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "gen_relative_pos_bias.h"
#include "reduce_kernel_utils.cuh"

namespace fastertransformer {

/*******************  invokeGenRelativePosBias  ***********************/
// relative_position_bias_table is [(2*window_size-1)*(2*window_size-1), headNum]
// relative_position_bias is [head_num, window_size^2, window_size^2]
// grid(window_size*window_size, head_num)
// block(window_size*window_size)

// relative_position_bias_table is [(2*window_size-1)*(2*window_size-1), headNum]
// relative_position_bias_index is [window_size^2, window_size^2]
// relative_position_bias is [head_num, window_size^2, window_size^2]
// grid(window_size*window_size, head_num)
// block(window_size*window_size)
template<typename T, typename Tindex>
__global__ void gen_relative_pos_bias(T* relative_position_bias,
                                      const T* relative_position_bias_table,
                                      const Tindex* relative_position_bias_index,
                                      const int window_size,
                                      const int head_num)
{
    const int h_in_window = blockIdx.x / window_size;
    const int w_in_window = blockIdx.x % window_size;
    const int h_in_token = threadIdx.x / window_size;
    const int w_in_token = threadIdx.x % window_size;
    const int head_idx = blockIdx.y;
    const int elements_per_window = window_size * window_size;
    const size_t elements_per_window_2 = elements_per_window * elements_per_window;
    const size_t output_idx = head_idx * elements_per_window_2 + blockIdx.x * elements_per_window + threadIdx.x;
    if (output_idx < head_num * elements_per_window_2) {
        const Tindex idx_in_table =
            relative_position_bias_index[(h_in_window * window_size + w_in_window) * elements_per_window
                                         + h_in_token * window_size + w_in_token];
        relative_position_bias[output_idx] = relative_position_bias_table[idx_in_table * head_num + head_idx];
    }
}

template<typename T, typename Tindex>
void invokeGenRelativePosBias(T* relative_position_bias,
                              const T* relative_position_bias_table,
                              const Tindex* relative_position_bias_index,
                              const int window_size,
                              const int head_num,
                              cudaStream_t stream)
{
    dim3 grid(window_size * window_size, head_num);
    dim3 block(window_size * window_size);

    if (block.x > 1024) {
        printf("[ERROR][invokeGenRelativePosBias] window_size*window_size > 1024.\n");
        exit(-1);
    }

    gen_relative_pos_bias<<<grid, block, 0, stream>>>(
        relative_position_bias, relative_position_bias_table, relative_position_bias_index, window_size, head_num);
}

/*******************  instantiation  ***********************/

template void invokeGenRelativePosBias(float* relative_position_bias,
                                       const float* relative_position_bias_table,
                                       const int* relative_position_bias_index,
                                       const int window_size,
                                       const int head_num,
                                       cudaStream_t stream);

template void invokeGenRelativePosBias(half* relative_position_bias,
                                       const half* relative_position_bias_table,
                                       const int* relative_position_bias_index,
                                       const int window_size,
                                       const int head_num,
                                       cudaStream_t stream);

template void invokeGenRelativePosBias(float* relative_position_bias,
                                       const float* relative_position_bias_table,
                                       const int64_t* relative_position_bias_index,
                                       const int window_size,
                                       const int head_num,
                                       cudaStream_t stream);

template void invokeGenRelativePosBias(half* relative_position_bias,
                                       const half* relative_position_bias_table,
                                       const int64_t* relative_position_bias_index,
                                       const int window_size,
                                       const int head_num,
                                       cudaStream_t stream);

}  // namespace fastertransformer
