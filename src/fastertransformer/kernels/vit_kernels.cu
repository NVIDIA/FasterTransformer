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

#include "src/fastertransformer/kernels/vit_kernels.h"
#include "src/fastertransformer/utils/cuda_utils.h"
namespace fastertransformer {

template<typename T>
__global__ void add_bias_slice(
    const T* __restrict in, T* __restrict out, const T* __restrict bias, int m, int n, int s, bool on_top = true)
{
    // s = slice_cnt, input_slice_size = s*n,
    // n = element size after concate axis. for example: [2,3,4,5] concate_axis = 1, then n = 4*5=20
    // on_top: concate slice on top or not.
    const int offset = on_top ? 1 : 0;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        int slice_id = id / (s * n);
        out[id + (slice_id + offset) * n] = __ldg(&in[id]) + __ldg(&bias[id % n]);
    }
}

template<>
__global__ void add_bias_slice(
    const half* __restrict in, half* __restrict out, const half* __restrict bias, int m, int n, int s, bool on_top)
{
    const int offset = on_top ? 1 : 0;
    const half2* in_ptr = (half2*)in;
    const half2* bias_ptr = (half2*)bias;
    half2* out_ptr = (half2*)out;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        half2 d1 = __ldg(&in_ptr[id]);
        half2 d2 = __ldg(&bias_ptr[id % n]);
        int slice_id = id / (s * n);
        out_ptr[id + (slice_id + offset) * n] = __hadd2(d1, d2);
    }
}

template<typename T>
void invokeAddBiasSlice(T* in, T* out, const T* bias, const int m, const int n, const int s, cudaStream_t stream)
{
    const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16
    dim3 block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x = m;
    }
    else {
        block.x = 1024;
        grid.x = (m * n + 1023) / 1024;
    }
    add_bias_slice<<<grid, block, 0, stream>>>(in, out, bias, m, n / data_type_factor, s);
}

template<typename T>
__global__ void add_bias_concat_clstoken_add_posembed(const T* __restrict in,         // b*h*w*n
                                                      T* __restrict out,              // b*(h*w+1)*n
                                                      const T* __restrict bias,       // n
                                                      const T* __restrict cls_token,  // n
                                                      const T* __restrict pos_embed,  // (h*w+1)*n
                                                      const int m,                    // b * (h*w+1)
                                                      const int n,
                                                      const int s,  // h*w+1
                                                      bool on_top = true)
{
    const int concat_row_idx = on_top ? 0 : (s - 1);
    const int offset = on_top ? 1 : 0;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        int col_idx = id % n;
        int row_idx = id / n;
        int slice_row_idx = row_idx % s;
        int slice_idx = row_idx / s;
        int idx_s = slice_row_idx * n + col_idx;
        int idx_i = (slice_row_idx - offset + slice_idx * (s - 1)) * n + col_idx;

        if (slice_row_idx == concat_row_idx) {
            out[id] = __ldg(&cls_token[col_idx]) + __ldg(&pos_embed[idx_s]);
        }
        else {
            out[id] = __ldg(&in[idx_i]) + __ldg(&bias[col_idx]) + __ldg(&pos_embed[idx_s]);
        }
    }
}

template<>
__global__ void add_bias_concat_clstoken_add_posembed(const half* __restrict in,         // b*h*w*n
                                                      half* __restrict out,              // b*(h*w+1)*n
                                                      const half* __restrict bias,       // n
                                                      const half* __restrict cls_token,  // n
                                                      const half* __restrict pos_embed,  // (h*w+1)*n
                                                      const int m,                       // b * (h*w+1)
                                                      const int n,
                                                      const int s,  // h*w+1
                                                      bool on_top)
{
    const int concat_row_idx = on_top ? 0 : (s - 1);
    const int offset = on_top ? 1 : 0;
    half2* out_ptr = (half2*)out;
    const half2* in_ptr = (half2*)in;
    const half2* bias_ptr = (half2*)bias;
    const half2* token_ptr = (half2*)cls_token;
    const half2* embed_ptr = (half2*)pos_embed;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        int col_idx = id % n;
        int row_idx = id / n;
        int slice_row_idx = row_idx % s;
        int slice_idx = row_idx / s;
        int idx_s = slice_row_idx * n + col_idx;
        int idx_i = (slice_row_idx - offset + slice_idx * (s - 1)) * n + col_idx;

        if (slice_row_idx == concat_row_idx) {
            half2 d1 = __ldg(&token_ptr[col_idx]);
            half2 d2 = __ldg(&embed_ptr[idx_s]);
            out_ptr[id] = __hadd2(d1, d2);
        }
        else {
            half2 d1 = __ldg(&in_ptr[idx_i]);
            half2 d2 = __ldg(&bias_ptr[col_idx]);
            half2 d3 = __ldg(&embed_ptr[idx_s]);
            out_ptr[id] = __hadd2(d3, __hadd2(d1, d2));
        }
    }
}

template<typename T>
void invokeAddBiasConcatClsTokenAddPosEmbed(const T* in,
                                            T* out,
                                            const T* bias,
                                            const T* cls_token,
                                            const T* pos_embed,
                                            const int m,
                                            const int n,
                                            const int s,
                                            cudaStream_t stream)
{
    const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16
    dim3 block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x = m;
    }
    else {
        block.x = 1024;
        grid.x = (m * n + 1023) / 1024;
    }
    add_bias_concat_clstoken_add_posembed<<<grid, block, 0, stream>>>(
        in, out, bias, cls_token, pos_embed, m, n / data_type_factor, s);
}

template void invokeAddBiasConcatClsTokenAddPosEmbed(const float* in,
                                                     float* out,
                                                     const float* bias,
                                                     const float* cls_token,
                                                     const float* pos_embed,
                                                     const int m,
                                                     const int n,
                                                     const int s,
                                                     cudaStream_t stream);

template void invokeAddBiasConcatClsTokenAddPosEmbed(const half* in,
                                                     half* out,
                                                     const half* bias,
                                                     const half* cls_token,
                                                     const half* pos_embed,
                                                     const int m,
                                                     const int n,
                                                     const int s,
                                                     cudaStream_t stream);

template<typename T>
__global__ void
slice_copy(const T* __restrict in, T* __restrict out, const int m, const int n, const int s, const int offset_s)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx > m * n) {
        return;
    }

    int m_idx = idx / n;
    int col = idx % n;
    int in_idx = (m_idx * s + offset_s) * n + col;

    out[idx] = __ldg(&in[in_idx]);
}

template<>
__global__ void
slice_copy(const half* __restrict in, half* __restrict out, const int m, const int n, const int s, const int offset_s)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx > m * n) {
        return;
    }

    int m_idx = idx / n;
    int col = idx % n;
    int in_idx = (m_idx * s + offset_s) * n + col;

    half2* out_ptr = (half2*)out;
    const half2* in_ptr = (half2*)in;

    out_ptr[idx] = __ldg(&in_ptr[in_idx]);
}

template<typename T>
void invokeSliceCopy(
    const T* in, T* out, const int m, const int n, const int s, const int offset_s, cudaStream_t stream)
{
    const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16
    dim3 block, grid;
    if (n / data_type_factor <= 1024) {
        block.x = n / data_type_factor;
        grid.x = m;
    }
    else {
        block.x = 1024;
        grid.x = (m * n + 1023) / 1024;
    }
    slice_copy<<<grid, block, 0, stream>>>(in, out, m, n / data_type_factor, s, offset_s);
}

template void invokeSliceCopy(
    const float* in, float* out, const int m, const int n, const int s, const int offset_s, cudaStream_t stream);
template void invokeSliceCopy(
    const half* in, half* out, const int m, const int n, const int s, const int offset_s, cudaStream_t stream);

template<typename T>
__global__ void add_bias_add_posembed(T* __restrict out,              // b*(h*w)*n
                                      const T* __restrict bias,       // n
                                      const T* __restrict pos_embed,  // (h*w)*n
                                      const int m,                    // b * (h*w)
                                      const int n,
                                      const int s  // h*w*n
)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        int b_idx = id % n;
        int p_idx = id % s;

        out[id] += __ldg(&bias[b_idx]) + __ldg(&pos_embed[p_idx]);
    }
}

template<>
__global__ void add_bias_add_posembed(half* __restrict out,              // b*(h*w+1)*n
                                      const half* __restrict bias,       // n
                                      const half* __restrict pos_embed,  // (h*w)*n
                                      const int m,                       // b * (h*w)
                                      const int n,
                                      const int s  // h*w *n
)
{
    half2* out_ptr = (half2*)out;
    const half2* bias_ptr = (half2*)bias;
    const half2* embed_ptr = (half2*)pos_embed;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        int b_idx = id % n;
        int p_idx = id % s;
        half2 d1 = __ldg(&bias_ptr[b_idx]);
        half2 d2 = __ldg(&embed_ptr[p_idx]);
        out_ptr[id] = __hadd2(out_ptr[id], __hadd2(d1, d2));
    }
}

template<typename T>
void invokeAddBiasAddPosEmbed(
    T* out, const T* bias, const T* pos_embed, const int m, const int n, const int s, cudaStream_t stream)
{
    const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16
    dim3 block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x = m;
    }
    else {
        block.x = 1024;
        grid.x = (m * n + 1023) / 1024;
    }
    add_bias_add_posembed<<<grid, block, 0, stream>>>(out, bias, pos_embed, m, n / data_type_factor, s);
}

template void invokeAddBiasAddPosEmbed(
    float* out, const float* bias, const float* pos_embed, const int m, const int n, const int s, cudaStream_t stream);

template void invokeAddBiasAddPosEmbed(
    half* out, const half* bias, const half* pos_embed, const int m, const int n, const int s, cudaStream_t stream);

}  // namespace fastertransformer
