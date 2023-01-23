/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/kernels/image_merge_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"

namespace fastertransformer {

/*******************  invokeImageMerge  ***********************/

// input is [batch, 2*H, 2*W, n/4]
// output is [batch, H, W, n]
// grid (W, H, batch)
// block (min(n, 1024))
template<typename T>
__global__ void image_merge_kernel(T* out, const T* __restrict input, int batch, int H, int W, int n)
{
    const int    tid             = threadIdx.x;
    const int    W_idx           = blockIdx.x;
    const int    H_idx           = blockIdx.y;
    const size_t batch_offset    = blockIdx.z * H * W * n;
    const int    input_H_stride  = W * n / 2;
    const int    output_H_stride = W * n;
    const int    n_4             = n >> 2;
    const int    bdim            = blockDim.x;

#pragma unroll
    for (int col_id = tid; col_id < n; col_id += bdim) {
        int    part_id     = col_id / n_4;
        int    offset_in_W = part_id / 2;
        int    offset_in_H = part_id % 2;
        size_t input_id    = batch_offset + (2 * H_idx + offset_in_H) * input_H_stride + (2 * W_idx + offset_in_W) * n_4
                          + (col_id % n_4);
        size_t output_idx = batch_offset + H_idx * output_H_stride + W_idx * n + col_id;
        out[output_idx]   = ldg(input + input_id);
    }
}

// TODO : accelerate with half2
template<typename T>
void invokeImageMerge(T* output, const T* input, int batch, int H, int W, int n, cudaStream_t stream)
{
    if ((W % 2 != 0) || (H % 2 != 0)) {
        printf("[ERROR][invokeImageMerge] H(W) should be a multiple of 2.\n");
        return;
    }
    dim3 grid(W / 2, H / 2, batch);
    int  blockSize = (4 * n < 1024) ? (4 * n) : 1024;
    image_merge_kernel<T><<<grid, blockSize, 0, stream>>>(output, input, batch, H / 2, W / 2, n * 4);
}

template void
invokeImageMerge<float>(float* output, const float* input, int batch, int H, int W, int n, cudaStream_t stream);

template void
invokeImageMerge<half>(half* output, const half* input, int batch, int H, int W, int n, cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeImageMerge<__nv_bfloat16>(
    __nv_bfloat16* output, const __nv_bfloat16* input, int batch, int H, int W, int n, cudaStream_t stream);
#endif

/*******************  invokeMergeLayerNormCol32  ***********************/

// input is [batch, 2*H, 2*W, n/4]
// output is [batch, H, W, n]
// grid (W, H, batch)
// block (min(n, 1024))
template<typename T>
__global__ void image_merge_col32_kernel(
    int8_t* out, const T* __restrict input, int batch, int H, int W, int n, const float* merge_inFactor)
{
    // const int   ite       = 4; // not used
    const int   tid       = threadIdx.x;
    const int   W_idx     = blockIdx.x;
    const int   H_idx     = blockIdx.y;
    const float out_scale = __ldg(merge_inFactor);
    const int   n_4       = n >> 2;
    const int   m         = batch * 4 * H * W;
    const int   bdim      = blockDim.x;

#pragma unroll
    for (int col_id = tid; col_id < n; col_id += bdim) {
        int part_id     = col_id / n_4;
        int offset_in_W = part_id / 2;
        int offset_in_H = part_id % 2;

        int   col_input       = col_id % n_4;
        int   row_input       = blockIdx.z * H * W * 4 + (2 * H_idx + offset_in_H) * W * 2 + (2 * W_idx + offset_in_W);
        int   input_idx_col32 = ((col_input >> 5) << 5) * m + (row_input << 5) + (col_input & 31);
        float local_out       = (float)(__ldg(input + input_idx_col32));

        int col_output        = col_id;
        int row_output        = blockIdx.z * H * W + H_idx * W + W_idx;
        int output_idx_col32  = ((col_output >> 5) << 5) * (m >> 2) + (row_output << 5) + (col_output & 31);
        out[output_idx_col32] = float_to_int8_rn(out_scale * local_out);
    }
}

template<typename T>
void invokeImageMergeCol32(
    int8_t* output, const T* input, int batch, int H, int W, int n, const float* merge_inFactor, cudaStream_t stream)
{
    if ((W % 2 != 0) || (H % 2 != 0)) {
        printf("[ERROR][invokeImageMergeCol32] H(W) should be a multiple of 2.\n");
        return;
    }
    dim3 grid(W / 2, H / 2, batch);
    int  blockSize = 4 * n;
    blockSize      = ((blockSize / 4) + 31) / 32 * 32;
    blockSize      = (blockSize < 1024) ? blockSize : 1024;
    image_merge_col32_kernel<T>
        <<<grid, blockSize, 0, stream>>>(output, input, batch, H / 2, W / 2, n * 4, merge_inFactor);
}

template void invokeImageMergeCol32(int8_t*      output,
                                    const float* input,
                                    int          batch,
                                    int          H,
                                    int          W,
                                    int          n,
                                    const float* merge_inFactor,
                                    cudaStream_t stream);

template void invokeImageMergeCol32(int8_t*      output,
                                    const half*  input,
                                    int          batch,
                                    int          H,
                                    int          W,
                                    int          n,
                                    const float* merge_inFactor,
                                    cudaStream_t stream);

}  // namespace fastertransformer
