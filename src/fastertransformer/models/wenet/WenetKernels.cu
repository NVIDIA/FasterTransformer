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

__inline__ __device__ int target_index(int id1, int id2, int id3, int id4, int dim_1, int dim_2, int dim_3, int dim_4)
{
    return id1 * (dim_2 * dim_3 * dim_4) + id3 * (dim_2 * dim_4) + id2 * dim_4 + id4;
}

namespace {
constexpr auto EPS = 1e-6f;  // 1e-5;
}

template<typename T>
__inline__ __device__ T sigmoid(T x)
{
    return T(1.0f) / (T(1.0f) + exp(-x));
    // return T(1.0f) / (T(1.0f) + __expf(-x));
}

////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void add_bias_mul(T* out, const T* __restrict bias, T scale, int m, int n)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        out[id] = (out[id] + (T)ldg(&bias[id % n])) * scale;
    }
}

template<>
__global__ void add_bias_mul(half* out, const half* __restrict bias, half scale, int m, int n)
{
    int real_n = 2 * n;
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * real_n; id += blockDim.x * gridDim.x) {
        out[id] = __hmul(__hadd(out[id], bias[id % real_n]), scale);
    }
}
#ifdef ENABLE_BF16
template<>
__global__ void
add_bias_mul(__nv_bfloat16* out, const __nv_bfloat16* __restrict bias, __nv_bfloat16 scale, int m, int n)
{
}
#endif

template<typename T>
void invokeAddBiasMul(T* out, const T* bias, const T scale, const int m, const int n, cudaStream_t stream)
{
    const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16 and bf16
    dim3      block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x  = m;
    }
    else {
        block.x = 1024;
        grid.x  = ceil(m * n / 1024.);
    }
    add_bias_mul<T><<<grid, block, 0, stream>>>(out, bias, scale, m, n / data_type_factor);
}

template void
invokeAddBiasMul(float* out, const float* bias, const float scale, const int m, const int n, cudaStream_t stream);
template void
invokeAddBiasMul(half* out, const half* bias, const half scale, const int m, const int n, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeAddBiasMul(__nv_bfloat16*       out,
                               const __nv_bfloat16* bias,
                               const __nv_bfloat16  scale,
                               const int            m,
                               const int            n,
                               cudaStream_t         stream);
#endif

////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void cmvn(T* out, const T* __restrict in, const T* __restrict mean, const T* __restrict istd, int m, int n)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        out[id] = (in[id] - mean[id % n]) * istd[id % n];
    }
}

template<>
__global__ void
cmvn(half* out, const half* __restrict in, const half* __restrict mean, const half* __restrict istd, int m, int n)
{
    int real_n = 2 * n;
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * real_n; id += blockDim.x * gridDim.x) {
        out[id] = __hmul(__hsub(in[id], mean[id % real_n]), istd[id % real_n]);
    }
}

#ifdef ENABLE_BF16
template<>
__global__ void cmvn(__nv_bfloat16* out,
                     const __nv_bfloat16* __restrict in,
                     const __nv_bfloat16* __restrict mean,
                     const __nv_bfloat16* __restrict istd,
                     int m,
                     int n)
{
}
#endif

template<typename T>
void invokeCMVN(T* out, const T* in, const T* mean, const T* istd, const int m, const int n, cudaStream_t stream)
{
    const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16 and bf16
    dim3      block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x  = m;
    }
    else {
        block.x = 1024;
        grid.x  = ceil(m * n / 1024.);
    }
    cmvn<T><<<grid, block, 0, stream>>>(out, in, mean, istd, m, n / data_type_factor);
}

template void invokeCMVN(
    float* out, const float* in, const float* mean, const float* istd, const int m, const int n, cudaStream_t stream);
template void invokeCMVN(
    half* out, const half* in, const half* mean, const half* istd, const int m, const int n, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeCMVN(__nv_bfloat16*       out,
                         const __nv_bfloat16* in,
                         const __nv_bfloat16* mean,
                         const __nv_bfloat16* istd,
                         const int            m,
                         const int            n,
                         cudaStream_t         stream);
#endif

////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void slice(T* out, const T* __restrict in, int batch_size, int seq_len, int hidden_unit)
{
    // in: (1, 5000, hidden_unit)
    // out: (batch_size, seq_len, hidden_unit)
    int seq_id  = blockIdx.x % seq_len;
    int in_id   = seq_id * blockDim.x + threadIdx.x;
    int out_id  = blockIdx.x * blockDim.x + threadIdx.x;
    out[out_id] = (T)ldg(&in[in_id]);
}

template<>
__global__ void slice(half* out, const half* __restrict in, int batch_size, int seq_len, int hidden_unit)
{
    half2*       out_ptr = (half2*)out;
    const half2* in_ptr  = (half2*)in;
    int          seq_id  = blockIdx.x % seq_len;
    int          in_id   = seq_id * blockDim.x + threadIdx.x;
    int          out_id  = blockIdx.x * blockDim.x + threadIdx.x;
    out_ptr[out_id]      = __ldg(&in_ptr[in_id]);
}

#ifdef ENABLE_BF16
template<>
__global__ void
slice(__nv_bfloat16* out, const __nv_bfloat16* __restrict in, int batch_size, int seq_len, int hidden_unit)
{
}
#endif

template<typename T>
void invokeSlice(
    T* out, const T* in, const int batch_size, const int seq_len, const int hidden_unit, cudaStream_t stream)
{
    const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16 and bf16
    const int m                = batch_size * seq_len;
    dim3      block, grid;
    if (hidden_unit / data_type_factor <= 1024) {
        block.x = hidden_unit / data_type_factor;
        grid.x  = m;
    }
    else {
        block.x = 1024;
        grid.x  = ceil(m * hidden_unit / 1024.);
    }
    slice<T><<<grid, block, 0, stream>>>(out, in, batch_size, seq_len, hidden_unit / data_type_factor);
}

template void invokeSlice(
    float* out, const float* in, const int batch_size, const int seq_len, const int hidden_unit, cudaStream_t stream);
template void invokeSlice(
    half* out, const half* in, const int batch_size, const int seq_len, const int hidden_unit, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeSlice(__nv_bfloat16*       out,
                          const __nv_bfloat16* in,
                          const int            batch_size,
                          const int            seq_len,
                          const int            hidden_unit,
                          cudaStream_t         stream);
#endif

////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void
transpose0213(T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    int batch_id       = blockIdx.x / (head_num * seq_len);
    int seq_id         = blockIdx.x % seq_len;
    int head_id        = (blockIdx.x % (head_num * seq_len)) / seq_len;
    dst[batch_id * (head_num * seq_len * size_per_head) + seq_id * head_num * size_per_head + head_id * size_per_head
        + threadIdx.x] = src[blockIdx.x * size_per_head + threadIdx.x];
}

template<>
__global__ void transpose0213(
    half* src, half* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int batch_id = tid / (head_num * seq_len * size_per_head);
    int head_id  = (tid % (head_num * seq_len * size_per_head)) / (seq_len * size_per_head);
    int seq_id   = (tid % (seq_len * size_per_head)) / size_per_head;
    int id       = tid % size_per_head;

    int target_id = target_index(batch_id, head_id, seq_id, id, batch_size, head_num, seq_len, size_per_head);

    dst[target_id] = src[tid];
}

template<>
__global__ void transpose0213(
    half2* src, half2* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int batch_id = tid / (head_num * seq_len * size_per_head);
    int head_id  = (tid % (head_num * seq_len * size_per_head)) / (seq_len * size_per_head);
    int seq_id   = (tid % (seq_len * size_per_head)) / size_per_head;
    int id       = tid % size_per_head;

    int target_id = target_index(batch_id, head_id, seq_id, id, batch_size, head_num, seq_len, size_per_head);

    dst[target_id] = src[tid];
}

#ifdef ENABLE_BF16
template<>
__global__ void transpose0213(__nv_bfloat16* src,
                              __nv_bfloat16* dst,
                              const int      batch_size,
                              const int      seq_len,
                              const int      head_num,
                              const int      size_per_head)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int batch_id = tid / (head_num * seq_len * size_per_head);
    int head_id  = (tid % (head_num * seq_len * size_per_head)) / (seq_len * size_per_head);
    int seq_id   = (tid % (seq_len * size_per_head)) / size_per_head;
    int id       = tid % size_per_head;

    int target_id = target_index(batch_id, head_id, seq_id, id, batch_size, head_num, seq_len, size_per_head);

    dst[target_id] = src[tid];
}

template<>
__global__ void transpose0213(__nv_bfloat162* src,
                              __nv_bfloat162* dst,
                              const int       batch_size,
                              const int       seq_len,
                              const int       head_num,
                              const int       size_per_head)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int batch_id = tid / (head_num * seq_len * size_per_head);
    int head_id  = (tid % (head_num * seq_len * size_per_head)) / (seq_len * size_per_head);
    int seq_id   = (tid % (seq_len * size_per_head)) / size_per_head;
    int id       = tid % size_per_head;

    int target_id = target_index(batch_id, head_id, seq_id, id, batch_size, head_num, seq_len, size_per_head);

    dst[target_id] = src[tid];
}
#endif

template<typename T>
void invokeTranspose0213(T*           dst,
                         T*           src,
                         const int    batch_size,
                         const int    seq_len,
                         const int    head_num,
                         const int    size_per_head,
                         cudaStream_t stream)
{
    // src: [batch_size, head_num, seq_len, size_per_head]
    // dst: [batch_size, seq_len, head_num, size_per_head]
    dim3 grid, block;
    if (sizeof(T) == 2) {
        int seq_per_block = 1;
        grid.x            = batch_size * head_num * seq_len / seq_per_block;
        if (seq_per_block * size_per_head % 2 == 0) {
            block.x = seq_per_block * size_per_head / 2;
            if (std::is_same<T, half>::value) {
                transpose0213<half2><<<grid, block, 0, stream>>>(
                    (half2*)src, (half2*)dst, batch_size, seq_len, head_num, size_per_head / 2);
            }
#ifdef ENABLE_BF16
            else {
                transpose0213<__nv_bfloat162><<<grid, block, 0, stream>>>(
                    (__nv_bfloat162*)src, (__nv_bfloat162*)dst, batch_size, seq_len, head_num, size_per_head / 2);
            }
#endif
        }
        else {
            block.x = seq_per_block * size_per_head;
            transpose0213<T><<<grid, block, 0, stream>>>(src, dst, batch_size, seq_len, head_num, size_per_head);
        }
    }
    else {
        const int seq_per_block = 1;
        grid.x                  = batch_size * head_num * seq_len / seq_per_block;
        block.x                 = seq_per_block * size_per_head;
        transpose0213<T><<<grid, block, 0, stream>>>(src, dst, batch_size, seq_len, head_num, size_per_head);
    }
}

template void invokeTranspose0213(float*       src,
                                  float*       dst,
                                  const int    batch_size,
                                  const int    seq_len,
                                  const int    head_num,
                                  const int    size_per_head,
                                  cudaStream_t stream);

template void invokeTranspose0213(half*        src,
                                  half*        dst,
                                  const int    batch_size,
                                  const int    seq_len,
                                  const int    head_num,
                                  const int    size_per_head,
                                  cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeTranspose0213(__nv_bfloat16* src,
                                  __nv_bfloat16* dst,
                                  const int      batch_size,
                                  const int      seq_len,
                                  const int      head_num,
                                  const int      size_per_head,
                                  cudaStream_t   stream);
#endif

////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void embedDecoderInput(T* out,
                                  const int* __restrict in,
                                  const T* __restrict embed_weights,
                                  const T* __restrict encoding_weights,
                                  int vocab_size,
                                  int seq_len,
                                  int max_len,
                                  int m,
                                  int n)
{
    // blockDim: (bs * beam_size * 63), threadDim: (256)
    // in: (bs * beam_size, 64)
    // embed_weights: (4233, 256)
    // encoding_weights: (1, 5000, 256)
    // out: (bs * beam_size, 63, n = 256)
    int in_row       = blockIdx.x / seq_len;
    int in_col       = blockIdx.x % seq_len;
    int id           = blockIdx.x * blockDim.x + threadIdx.x;
    int embed_row    = (int)ldg(&in[in_row * (seq_len + 1) + in_col]);
    int embed_id     = embed_row * blockDim.x + threadIdx.x;
    int encoding_row = blockIdx.x % seq_len;
    int encoding_id  = encoding_row * blockDim.x + threadIdx.x;
    T   scale        = (T)sqrtf((float)n);
    out[id]          = (T)ldg(&embed_weights[embed_id]) * scale;
    T encoding_data  = (T)ldg(&encoding_weights[encoding_id]);
    out[id] += encoding_data;
}

template<>
__global__ void embedDecoderInput(half* out,
                                  const int* __restrict in,
                                  const half* __restrict embed_weights,
                                  const half* __restrict encoding_weights,
                                  int vocab_size,
                                  int seq_len,
                                  int max_len,
                                  int m,
                                  int n)
{
    int  in_row       = blockIdx.x / seq_len;
    int  in_col       = blockIdx.x % seq_len;
    int  id           = blockIdx.x * blockDim.x + threadIdx.x;
    int  embed_row    = (int)ldg(&in[in_row * (seq_len + 1) + in_col]);
    int  embed_id     = embed_row * blockDim.x + threadIdx.x;
    int  encoding_row = blockIdx.x % seq_len;
    int  encoding_id  = encoding_row * blockDim.x + threadIdx.x;
    half scale        = (half)sqrtf((float)n);
    out[id]           = __hadd(__hmul(embed_weights[embed_id], scale), encoding_weights[encoding_id]);
}

#ifdef ENABLE_BF16
template<>
__global__ void embedDecoderInput(__nv_bfloat16* out,
                                  const int* __restrict in,
                                  const __nv_bfloat16* __restrict embed_weights,
                                  const __nv_bfloat16* __restrict encoding_weights,
                                  int vocab_size,
                                  int seq_len,
                                  int max_len,
                                  int m,
                                  int n)
{
}
#endif

template<typename T>
void invokeEmbedDecoderInput(T*           out,
                             const int*   in,
                             const T*     embed_weights,
                             const T*     encoding_weights,
                             const int    vocab_size,
                             const int    max_len,
                             const int    batch_size,
                             const int    seq_len,
                             const int    hidden_units,
                             cudaStream_t stream)
{
    const int data_type_factor = 1;
    const int m                = batch_size * seq_len;
    const int n                = hidden_units;
    dim3      block, grid;
    if (n / data_type_factor <= 1024) {
        block.x = n / data_type_factor;
        grid.x  = m;
    }
    else {
        block.x = 1024;
        grid.x  = ceil(m * n / 1024.);
    }
    embedDecoderInput<T><<<grid, block, 0, stream>>>(
        out, in, embed_weights, encoding_weights, vocab_size, seq_len, max_len, m, n / data_type_factor);
}

template void invokeEmbedDecoderInput(float*       out,
                                      const int*   in,
                                      const float* embed_weights,
                                      const float* encoding_weights,
                                      const int    vocab_size,
                                      const int    max_len,
                                      const int    batch_size,
                                      const int    seq_len,
                                      const int    hidden_units,
                                      cudaStream_t stream);
template void invokeEmbedDecoderInput(half*        out,
                                      const int*   in,
                                      const half*  embed_weights,
                                      const half*  encoding_weights,
                                      const int    vocab_size,
                                      const int    max_len,
                                      const int    batch_size,
                                      const int    seq_len,
                                      const int    hidden_units,
                                      cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeEmbedDecoderInput(__nv_bfloat16*       out,
                                      const int*           in,
                                      const __nv_bfloat16* embed_weights,
                                      const __nv_bfloat16* encoding_weights,
                                      const int            vocab_size,
                                      const int            max_len,
                                      const int            batch_size,
                                      const int            seq_len,
                                      const int            hidden_units,
                                      cudaStream_t         stream);
#endif

////////////////////////////////////////////////////////////////////////////
template<typename T>
__global__ void maskDecoderOutput(float* score,
                                  const T* __restrict decoder_output,
                                  const int* __restrict decoder_sequence_length,
                                  const int* __restrict decoder_input,
                                  int max_seq_len,
                                  int vocab_size)
{
    // gridDim: (bs * beam_size), blockDim: (max_seq_len=63)
    // decoder_output: [batch_size, beam_size, max_seq_len, vocab_size]
    // decoder_sequence_length: [bs * beam_size]
    // decoder_input: [bs * beam_size, max_seq_len+1=64]
    // score: [bs * beam_size]
    const int len  = decoder_sequence_length[blockIdx.x];
    T         mask = (T)(0.0f);
    if (threadIdx.x < len)
        mask = (T)(1.0f);

    int in_id     = blockIdx.x * (max_seq_len + 1) + threadIdx.x + 1;
    int in_value  = decoder_input[in_id];
    int vocab_id  = in_value * (int)mask;
    int out_id    = (blockIdx.x * blockDim.x + threadIdx.x) * vocab_size + vocab_id;
    T   out_value = decoder_output[out_id] * mask;

    float sum = blockReduceSum((float)(out_value));
    if (threadIdx.x == 0)
        score[blockIdx.x] = sum;
}

template<typename T>
void invokeMaskDecoderOutput(float*       score,
                             const T*     decoder_output,
                             const int*   decoder_sequence_length,
                             const int*   decoder_input,
                             const int    batch_size,
                             const int    max_seq_len,
                             const int    vocab_size,
                             cudaStream_t stream)
{
    maskDecoderOutput<T><<<batch_size, std::min(1024, max_seq_len), 0, stream>>>(
        score, decoder_output, decoder_sequence_length, decoder_input, max_seq_len, vocab_size);
}

template void invokeMaskDecoderOutput(float*       score,
                                      const float* decoder_output,
                                      const int*   decoder_sequence_length,
                                      const int*   decoder_input,
                                      const int    batch_size,
                                      const int    max_seq_len,
                                      const int    vocab_size,
                                      cudaStream_t stream);
template void invokeMaskDecoderOutput(float*       score,
                                      const half*  decoder_output,
                                      const int*   decoder_sequence_length,
                                      const int*   decoder_input,
                                      const int    batch_size,
                                      const int    max_seq_len,
                                      const int    vocab_size,
                                      cudaStream_t stream);
// #ifdef ENABLE_BF16
// template void
// invokeMaskDecoderOutput(__nv_bfloat16* score, const __nv_bfloat16* decoder_output, const int*
// decoder_sequence_length, const int* decoder_input, const int batch_size, const int max_seq_len, cudaStream_t stream);
// #endif

////////////////////////////////////////////////////////////////////////////
template<typename T>
__global__ void buildBestIndex(
    int* best_index, const float* __restrict decoder_score, const T* __restrict ctc_score, T ctc_weight, int beam_size)
{
    // gridDim: (bs), blockDim: (beam_size=10)
    // decoder_score: [bs, 10]
    // ctc_score: [bs, 10]
    // best_index: [bs,]
    int id  = blockIdx.x * blockDim.x + threadIdx.x;
    T   val = (T)ldg(&decoder_score[id]) + (T)ldg(&ctc_score[id]) * ctc_weight;

    extern __shared__ float score[];
    score[threadIdx.x] = (float)val;
    __syncthreads();

    int   index = 0;
    float max   = score[0];
    for (int i = 1; i < blockDim.x; i++) {
        if (score[i] > max) {
            max   = score[i];
            index = i;
        }
    }
    if (threadIdx.x == 0) {
        best_index[blockIdx.x] = index;
    }
}

template<typename T>
void invokeBuildBestIndex(int*         best_index,
                          const float* decoder_score,
                          const T*     ctc_score,
                          const T      ctc_weight,
                          const int    batch_size,
                          const int    beam_size,
                          cudaStream_t stream)
{
    buildBestIndex<T><<<batch_size, std::min(1024, beam_size), beam_size * sizeof(float), stream>>>(
        best_index, decoder_score, ctc_score, ctc_weight, beam_size);
}

template void invokeBuildBestIndex(int*         best_index,
                                   const float* decoder_score,
                                   const float* ctc_score,
                                   const float  ctc_weight,
                                   const int    batch_size,
                                   const int    beam_size,
                                   cudaStream_t stream);
template void invokeBuildBestIndex(int*         best_index,
                                   const float* decoder_score,
                                   const half*  ctc_score,
                                   const half   ctc_weight,
                                   const int    batch_size,
                                   const int    beam_size,
                                   cudaStream_t stream);
// #ifdef ENABLE_BF16
// template void
// invokeBuildBestIndex(__nv_bfloat16* score, const __nv_bfloat16* decoder_output, const int* decoder_sequence_length,
// const int* decoder_input, const int batch_size, const int max_seq_len, cudaStream_t stream); #endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void mask_bias_glu(
    T* out, const T* __restrict in, const T* __restrict bias, int m, int n, const T* __restrict attr_mask, int seq_len)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        int m_id     = id / n;
        int n_id     = id % n;
        int s_id     = m_id % seq_len;
        int b_id     = m_id / seq_len;
        T   cur_mask = attr_mask[b_id * seq_len * seq_len + s_id];
        // in = in + m_id * 2 * n;
        const T* in_ptr = in + m_id * 2 * n;

        float val1 = 0.f;
        float val2 = 0.f;

        if (cur_mask != T(0.f)) {
            val1 = in_ptr[n_id];
            val2 = in_ptr[n_id + n];
        }

        if (bias != nullptr) {
            val1 = val1 + ldg(&bias[n_id]);
            val2 = val2 + ldg(&bias[n_id + n]);
        }
        out[id] = val1 * sigmoid<float>(val2);
    }
}

template<>
__global__ void mask_bias_glu(half* out,
                              const half* __restrict in,
                              const half* __restrict bias,
                              int m,
                              int n,
                              const half* __restrict attr_mask,
                              int seq_len)
{
    half2*       out_ptr  = (half2*)out;
    const half2* bias_ptr = (half2*)bias;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        int m_id = id / n;
        int n_id = id % n;
        int s_id = m_id % seq_len;
        int b_id = m_id / seq_len;
        // const half* attr_mask = attr_mask + ;
        half cur_mask = attr_mask[b_id * seq_len * seq_len + s_id];

        const half2* in_ptr = (half2*)in + m_id * 2 * n;
        // in = in + m_id * 2 * n;

        half2 val1 = cuda_cast<half2>(0.f);
        half2 val2 = cuda_cast<half2>(0.f);

        if (cur_mask != half(0.f)) {
            val1 = in_ptr[n_id];
            val2 = in_ptr[n_id + n];
        }

        if (bias != nullptr) {
            half2 bias1 = __ldg(&bias_ptr[n_id]);
            half2 bias2 = __ldg(&bias_ptr[n_id + n]);
            val1        = hadd2(val1, bias1);
            val2        = hadd2(val2, bias2);
        }
        half2 local_out;
        local_out.x = (float)val1.x * sigmoid<float>((float)val2.x);
        local_out.y = (float)val1.y * sigmoid<float>((float)val2.y);

        out_ptr[id] = local_out;
    }
}

#ifdef ENABLE_BF16
template<>
__global__ void mask_bias_glu(__nv_bfloat16* out,
                              const __nv_bfloat16* __restrict in,
                              const __nv_bfloat16* __restrict bias,
                              int m,
                              int n,
                              const __nv_bfloat16* __restrict attr_mask,
                              int seq_len)
{
    __nv_bfloat162*       out_ptr  = (__nv_bfloat162*)out;
    const __nv_bfloat162* bias_ptr = (__nv_bfloat162*)bias;
    // const __nv_bfloat162* in_ptr = (__nv_bfloat162*)in;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        int           m_id     = id / n;
        int           n_id     = id % n;
        int           s_id     = m_id % seq_len;
        int           b_id     = m_id / seq_len;
        __nv_bfloat16 cur_mask = attr_mask[b_id * seq_len * seq_len + s_id];

        // in = in + m_id * 2 * n;
        const __nv_bfloat162* in_ptr = (__nv_bfloat162*)in + m_id * 2 * n;

        __nv_bfloat162 val1 = cuda_cast<__nv_bfloat162>(0.f);
        __nv_bfloat162 val2 = cuda_cast<__nv_bfloat162>(0.f);

        if (cur_mask != __nv_bfloat16(0.f)) {
            val1 = in_ptr[n_id];
            val2 = in_ptr[n_id + n];
        }

        if (bias != nullptr) {
            __nv_bfloat162 bias1 = ldg(&bias_ptr[n_id]);
            __nv_bfloat162 bias2 = ldg(&bias_ptr[n_id + n]);
            val1                 = bf16hadd2(val1, bias1);
            val2                 = bf16hadd2(val2, bias2);
        }
        __nv_bfloat162 local_out;
        local_out.x = (float)val1.x * sigmoid<float>((float)val2.x);
        local_out.y = (float)val1.y * sigmoid<float>((float)val2.y);

        out_ptr[id] = local_out;
    }
}
#endif

template<typename T>
void invokeMaskBiasGlu(T*           out,
                       const T*     in,
                       const T*     bias,
                       const int    m,
                       const int    n,
                       const T*     attr_mask,
                       const int    seq_len,
                       cudaStream_t stream)
{
    const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16 and bf16
    dim3      block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x  = m;
    }
    else {
        block.x = 1024;
        grid.x  = ceil(m * n / 1024.);
    }
    mask_bias_glu<T><<<grid, block, 0, stream>>>(out, in, bias, m, n / data_type_factor, attr_mask, seq_len);
}

template void invokeMaskBiasGlu(float*       out,
                                const float* in,
                                const float* bias,
                                const int    m,
                                const int    n,
                                const float* attr_mask,
                                const int    seq_len,
                                cudaStream_t stream);
template void invokeMaskBiasGlu(half*        out,
                                const half*  in,
                                const half*  bias,
                                const int    m,
                                const int    n,
                                const half*  attr_mask,
                                const int    seq_len,
                                cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeMaskBiasGlu(__nv_bfloat16*       out,
                                const __nv_bfloat16* in,
                                const __nv_bfloat16* bias,
                                const int            m,
                                const int            n,
                                const __nv_bfloat16* attr_mask,
                                const int            seq_len,
                                cudaStream_t         stream);
#endif

////////////////////////////////////////////////////////////////////////////////////////////

// template<typename T>
// __global__ void bias_glu(T* out, const T* __restrict in, const T* __restrict bias, int m, int n)
// {
//     for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
//         int m_id = id / n;
//         int n_id = id % n;
//         // in = in + m_id * 2 * n;
//         const T* in_ptr = in + m_id * 2 * n;

//         float val1 = in_ptr[n_id];
//         float val2 = in_ptr[n_id + n];

//         if (bias != nullptr) {
//             val1 = val1 + ldg(&bias[n_id]);
//             val2 = val2 + ldg(&bias[n_id + n]);
//         }
//         out[id] = val1 * sigmoid<float>(val2);
//     }
// }

// template<>
// __global__ void bias_glu(half* out, const half* __restrict in, const half* __restrict bias, int m, int n)
// {
//     half2* out_ptr = (half2*)out;
//     const half2* bias_ptr = (half2*)bias;

//     for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
//         int m_id = id / n;
//         int n_id = id % n;
//         // const half* attr_mask = attr_mask + ;

//         const half2* in_ptr = (half2*)in + m_id * 2 * n;
//         // in = in + m_id * 2 * n;

//         half2 val1 = in_ptr[n_id];
//         half2 val2 = in_ptr[n_id + n];

//         if (bias != nullptr) {
//             half2 bias1 = __ldg(&bias_ptr[n_id]);
//             half2 bias2 = __ldg(&bias_ptr[n_id + n]);
//             val1 = hadd2(val1, bias1);
//             val2 = hadd2(val2, bias2);
//         }
//         half2 local_out;
//         local_out.x = (float)val1.x * sigmoid<float>((float)val2.x);
//         local_out.y = (float)val1.y * sigmoid<float>((float)val2.y);

//         out_ptr[id] = local_out;
//     }
// }

// #ifdef ENABLE_BF16
// template<>
// __global__ void
// bias_glu(__nv_bfloat16* out, const __nv_bfloat16* __restrict in, const __nv_bfloat16* __restrict bias, int m, int n)
// {
//     __nv_bfloat162* out_ptr = (__nv_bfloat162*)out;
//     const __nv_bfloat162* bias_ptr = (__nv_bfloat162*)bias;
//     // const __nv_bfloat162* in_ptr = (__nv_bfloat162*)in;

//     for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
//         int m_id = id / n;
//         int n_id = id % n;

//         // in = in + m_id * 2 * n;
//         const __nv_bfloat162* in_ptr = (__nv_bfloat162*)in + m_id * 2 * n;

//         __nv_bfloat162 val1 = in_ptr[n_id];
//         __nv_bfloat162 val2 = in_ptr[n_id + n];

//         if (bias != nullptr) {
//             __nv_bfloat162 bias1 = ldg(&bias_ptr[n_id]);
//             __nv_bfloat162 bias2 = ldg(&bias_ptr[n_id + n]);
//             val1 = bf16hadd2(val1, bias1);
//             val2 = bf16hadd2(val2, bias2);
//         }
//         __nv_bfloat162 local_out;
//         local_out.x = (float)val1.x * sigmoid<float>((float)val2.x);
//         local_out.y = (float)val1.y * sigmoid<float>((float)val2.y);

//         out_ptr[id] = local_out;
//     }
// }
// #endif

// template<typename T>
// void invokeBiasGlu(T* out, const T* in, const T* bias, const int m, const int n, cudaStream_t stream)
// {
//     const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16 and bf16
//     dim3 block, grid;
//     if (n / 4 / data_type_factor <= 1024) {
//         block.x = n / 4 / data_type_factor;
//         grid.x = m;
//     }
//     else {
//         block.x = 1024;
//         grid.x = ceil(m * n / 1024.);
//     }
//     bias_glu<T><<<grid, block, 0, stream>>>(out, in, bias, m, n / data_type_factor);
// }

// template void
// invokeBiasGlu(float* out, const float* in, const float* bias, const int m, const int n, cudaStream_t stream);
// template void invokeBiasGlu(half* out, const half* in, const half* bias, const int m, const int n, cudaStream_t
// stream); #ifdef ENABLE_BF16 template void invokeBiasGlu(__nv_bfloat16* out,
//                             const __nv_bfloat16* in,
//                             const __nv_bfloat16* bias,
//                             const int m,
//                             const int n,
//                             cudaStream_t stream);
// #endif

////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
__global__ void mask_bias(
    T* out, const T* __restrict in, const T* __restrict bias, int m, int n, const T* __restrict attr_mask, int seq_len)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        int m_id = id / n;
        int n_id = id % n;
        int s_id = m_id % seq_len;
        int b_id = m_id / seq_len;

        T cur_mask = attr_mask[b_id * seq_len * seq_len + s_id];
        // const T* in_ptr = in + m_id * n;

        float val1 = 0.f;

        if (cur_mask != T(0.f)) {
            val1 = in[id];

            if (bias != nullptr) {
                val1 = val1 + ldg(&bias[n_id]);
            }
        }
        out[id] = val1;
    }
}

template<>
__global__ void mask_bias(half* out,
                          const half* __restrict in,
                          const half* __restrict bias,
                          int m,
                          int n,
                          const half* __restrict attr_mask,
                          int seq_len)
{
    half2*       out_ptr  = (half2*)out;
    const half2* bias_ptr = (half2*)bias;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        int          m_id     = id / n;  // TODO optimize / and % with +-
        int          n_id     = id % n;
        int          s_id     = m_id % seq_len;
        int          b_id     = m_id / seq_len;
        half         cur_mask = attr_mask[b_id * seq_len * seq_len + s_id];
        const half2* in_ptr   = (const half2*)in + m_id * n;  // TODO optimize

        half2 val1 = cuda_cast<half2>(0.f);

        if (cur_mask != half(0.f)) {
            val1 = in_ptr[n_id];

            if (bias != nullptr) {
                half2 bias1 = __ldg(&bias_ptr[n_id]);
                val1        = hadd2(val1, bias1);
            }
        }
        out_ptr[id] = val1;
    }
}

#ifdef ENABLE_BF16
template<>
__global__ void mask_bias(__nv_bfloat16* out,
                          const __nv_bfloat16* __restrict in,
                          const __nv_bfloat16* __restrict bias,
                          int m,
                          int n,
                          const __nv_bfloat16* __restrict attr_mask,
                          int seq_len)
{
    __nv_bfloat162*       out_ptr  = (__nv_bfloat162*)out;
    const __nv_bfloat162* bias_ptr = (__nv_bfloat162*)bias;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        int                   m_id     = id / n;
        int                   n_id     = id % n;
        int                   s_id     = m_id % seq_len;
        int                   b_id     = m_id / seq_len;
        __nv_bfloat16         cur_mask = attr_mask[b_id * seq_len * seq_len + s_id];
        const __nv_bfloat162* in_ptr   = (const __nv_bfloat162*)in + m_id * n;

        __nv_bfloat162 val1 = cuda_cast<__nv_bfloat162>(0.f);

        if (cur_mask != __nv_bfloat16(0.f)) {
            val1 = in_ptr[n_id];

            if (bias != nullptr) {
                __nv_bfloat162 bias1 = ldg(&bias_ptr[n_id]);
                val1                 = bf16hadd2(val1, bias1);
            }
        }
        out_ptr[id] = val1;
    }
}
#endif

template<typename T>
void invokeMaskBias(T*           out,
                    const T*     in,
                    const T*     bias,
                    const int    m,
                    const int    n,
                    const T*     attr_mask,
                    const int    seq_len,
                    cudaStream_t stream)
{
    const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16 and bf16
    dim3      block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x  = m;
    }
    else {
        block.x = 1024;
        grid.x  = ceil(m * n / 1024.);
    }
    mask_bias<T><<<grid, block, 0, stream>>>(out, in, bias, m, n / data_type_factor, attr_mask, seq_len);
}

template void invokeMaskBias(float*       out,
                             const float* in,
                             const float* bias,
                             const int    m,
                             const int    n,
                             const float* attr_mask,
                             const int    seq_len,
                             cudaStream_t stream);
template void invokeMaskBias(half*        out,
                             const half*  in,
                             const half*  bias,
                             const int    m,
                             const int    n,
                             const half*  attr_mask,
                             const int    seq_len,
                             cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeMaskBias(__nv_bfloat16*       out,
                             const __nv_bfloat16* in,
                             const __nv_bfloat16* bias,
                             const int            m,
                             const int            n,
                             const __nv_bfloat16* attr_mask,
                             const int            seq_len,
                             cudaStream_t         stream);
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

// * Note that typename T is half2 or bfloat2 type
template<typename T, bool IS_OUTPUT, bool IS_BIAS, bool IS_RESIDUAL, bool IS_BETA, int UNROLL_FACTOR>
__global__ void generalScaleAddBiasResidualLayerNormOpt(T* normed_output,
                                                        T* output,
                                                        const T* __restrict bias,
                                                        const T* __restrict residual,
                                                        const T* __restrict gamma,
                                                        const T* __restrict beta,
                                                        int   m,
                                                        int   n,
                                                        float scale_input,
                                                        float scale_residual)
{
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    T local_sum            = cuda_cast<T>(0.0f);
    T local_scale_input    = cuda_cast<T>(scale_input);
    T local_scale_residual = cuda_cast<T>(scale_residual);
#pragma unroll
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = blockIdx.x * n + i;
        T         val   = cuda_cast<T>(0.0f);

        if (IS_OUTPUT) {
            val = hadd2(val, output[index]);
        }

        if (IS_BIAS) {
            val = hadd2(val, ldg(&bias[i]));
        }
        val = hmul2(val, local_scale_input);
        if (IS_RESIDUAL) {
            val = hadd2(val, hmul2(ldg(&residual[index]), local_scale_residual));
        }

        output[index] = val;

        local_sum = hadd2(local_sum, val);
    }

    mean = blockReduceSum((float)(local_sum.x + local_sum.y));

    if (threadIdx.x == 0) {
        s_mean = mean / n / 2;
    }
    __syncthreads();

    float local_var_sum = 0.0f;
#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        T     val    = output[blockIdx.x * n + i];
        float diff_1 = (float)(val.x) - s_mean;
        float diff_2 = (float)(val.y) - s_mean;
        local_var_sum += (diff_1 * diff_1 + diff_2 * diff_2);
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n / 2 + EPS);
    }
    __syncthreads();

    T mean_2 = cuda_cast<T>(s_mean);
    T var_2  = cuda_cast<T>(s_variance);
#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = blockIdx.x * n + i;
        T         val   = hmul2(hmul2(hsub2(output[index], mean_2), var_2), ldg(&gamma[i]));
        if (IS_BETA) {
            val = hadd2(val, ldg(&beta[i]));
        }
        normed_output[index] = val;
    }
}

// * Note that typename T is half2 or bfloat2 type
template<typename T, bool IS_OUTPUT, bool IS_BIAS, bool IS_RESIDUAL, bool IS_BETA, int UNROLL_FACTOR>
__global__ void generalScaleAddBiasResidualLayerNormOpt2(T* normed_output,
                                                         T* output,
                                                         const T* __restrict bias,
                                                         const T* __restrict residual,
                                                         const T* __restrict gamma,
                                                         const T* __restrict beta,
                                                         int   m,
                                                         int   n,
                                                         float scale_input,
                                                         float scale_residual)
{
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            x_sum    = 0.0f;
    float            x2_sum   = 0.0f;
    const int        b_offset = blockIdx.x * n;
    using T1                  = typename TypeConverter<T>::Type;

#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = b_offset + i;
        float     val_1 = 0.0f;
        float     val_2 = 0.0f;
        T         tmp;

        if (IS_OUTPUT) {
            tmp = ldg(&output[index]);
            val_1 += static_cast<float>(tmp.x);
            val_2 += static_cast<float>(tmp.y);
        }

        if (IS_BIAS) {
            tmp = ldg(&bias[i]);
            val_1 += static_cast<float>(tmp.x);
            val_2 += static_cast<float>(tmp.y);
        }
        val_1 *= scale_input;
        val_2 *= scale_input;

        if (IS_RESIDUAL) {
            tmp = ldg(&residual[index]);
            val_1 += static_cast<float>(tmp.x) * scale_residual;
            val_2 += static_cast<float>(tmp.y) * scale_residual;
        }

        tmp.x         = cuda_cast<T1>(val_1);
        tmp.y         = cuda_cast<T1>(val_2);
        output[index] = tmp;
        x_sum += val_1 + val_2;
        x2_sum += val_1 * val_1 + val_2 * val_2;
    }
    float sums[2];
    sums[0] = x_sum;
    sums[1] = x2_sum;
    blockReduceSumV2<float, 2>(sums);

    if (threadIdx.x == 0) {
        s_mean     = sums[0] / n / 2;
        s_variance = rsqrtf(sums[1] / n / 2 - s_mean * s_mean + EPS);
    }
    __syncthreads();

    T mean_2 = cuda_cast<T>(s_mean);
    T var_2  = cuda_cast<T>(s_variance);

#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = b_offset + i;
        T         val   = hmul2(hmul2(hsub2(output[index], mean_2), var_2), ldg(&gamma[i]));
        if (IS_BETA) {
            val = hadd2(val, ldg(&beta[i]));
        }
        normed_output[index] = val;
    }
}

#define HALF_LAYERNORM_OPT(UNROLL_FACTOR)                                                                              \
    if (bias != nullptr)                                                                                               \
        generalScaleAddBiasResidualLayerNormOpt<T2, true, true, true, true, UNROLL_FACTOR>                             \
            <<<grid, block, 0, stream>>>((T2*)norm_output,                                                             \
                                         (T2*)output,                                                                  \
                                         (const T2*)bias,                                                              \
                                         (const T2*)input,                                                             \
                                         (const T2*)gamma,                                                             \
                                         (const T2*)beta,                                                              \
                                         m,                                                                            \
                                         half_n,                                                                       \
                                         scale_input,                                                                  \
                                         scale_residual);                                                              \
    else                                                                                                               \
        generalScaleAddBiasResidualLayerNormOpt<T2, true, false, true, true, UNROLL_FACTOR>                            \
            <<<grid, block, 0, stream>>>((T2*)norm_output,                                                             \
                                         (T2*)output,                                                                  \
                                         (const T2*)bias,                                                              \
                                         (const T2*)input,                                                             \
                                         (const T2*)gamma,                                                             \
                                         (const T2*)beta,                                                              \
                                         m,                                                                            \
                                         half_n,                                                                       \
                                         scale_input,                                                                  \
                                         scale_residual);

#define HALF_LAYERNORM_OPT2(UNROLL_FACTOR)                                                                             \
    if (bias != nullptr)                                                                                               \
        generalScaleAddBiasResidualLayerNormOpt2<T2, true, true, true, true, UNROLL_FACTOR>                            \
            <<<grid, block, 0, stream>>>((T2*)norm_output,                                                             \
                                         (T2*)output,                                                                  \
                                         (const T2*)bias,                                                              \
                                         (const T2*)input,                                                             \
                                         (const T2*)gamma,                                                             \
                                         (const T2*)beta,                                                              \
                                         m,                                                                            \
                                         half_n,                                                                       \
                                         scale_input,                                                                  \
                                         scale_residual);                                                              \
    else                                                                                                               \
        generalScaleAddBiasResidualLayerNormOpt2<T2, true, false, true, true, UNROLL_FACTOR>                           \
            <<<grid, block, 0, stream>>>((T2*)norm_output,                                                             \
                                         (T2*)output,                                                                  \
                                         (const T2*)bias,                                                              \
                                         (const T2*)input,                                                             \
                                         (const T2*)gamma,                                                             \
                                         (const T2*)beta,                                                              \
                                         m,                                                                            \
                                         half_n,                                                                       \
                                         scale_input,                                                                  \
                                         scale_residual);

template<typename T>
__global__ void generalScaleAddBiasResidualLayerNorm(const T* __restrict input,
                                                     const T* __restrict gamma,
                                                     const T* __restrict beta,
                                                     const T* __restrict bias,
                                                     T*    output,
                                                     T*    norm_output,
                                                     int   m,
                                                     int   n,
                                                     float scale_input,
                                                     float scale_residual)

{
    int tid = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float local_out = (float)(output[blockIdx.x * n + i]);
        if (bias != nullptr) {
            local_out += (float)(ldg(&bias[i]));
        }
        local_out *= scale_input;
        local_out += (float)(ldg(&input[blockIdx.x * n + i])) * scale_residual;

        output[blockIdx.x * n + i] = (T)local_out;
        local_sum += local_out;
    }

    mean = blockReduceSum(local_sum);

    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float local_var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = (float)(output[blockIdx.x * n + i]) - s_mean;
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + EPS);
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        float beta_val = (beta == nullptr) ? 0.0f : (float)(ldg(&beta[i]));
        if (norm_output != nullptr)
            norm_output[blockIdx.x * n + i] =
                (T)((((float)output[blockIdx.x * n + i] - s_mean) * s_variance) * (float)(ldg(&gamma[i])) + beta_val);
    }
}

template<typename T>
void invokeGeneralScaleAddBiasResidualPreLayerNorm(T*           output,
                                                   T*           norm_output,
                                                   const T*     input,
                                                   const T*     gamma,
                                                   const T*     beta,
                                                   const T*     bias,
                                                   int          m,
                                                   int          n,
                                                   cudaStream_t stream,
                                                   int          opt_version,
                                                   float        scale_input,
                                                   float        scale_residual)

{
    if (opt_version > 0 && sizeof(T) == 2 && n % 2 == 0) {
        dim3 grid(m);
        int  half_n    = n / 2;
        int  half_n_32 = (half_n + 31) / 32 * 32;
        dim3 block(min(half_n_32, 512));
        int  rolls_per_thread = half_n / block.x;
        int  unroll_factor    = 8;
        while (unroll_factor > rolls_per_thread && unroll_factor > 1) {
            unroll_factor /= 2;
        }
        using T2 = typename TypeConverter<T>::Type;
        if (opt_version == 1) {
            if (unroll_factor == 1) {
                HALF_LAYERNORM_OPT(1);
            }
            else if (unroll_factor == 2) {
                HALF_LAYERNORM_OPT(2);
            }
            else if (unroll_factor == 3) {
                HALF_LAYERNORM_OPT(3);
            }
            else if (unroll_factor == 4) {
                HALF_LAYERNORM_OPT(4);
            }
            else if (unroll_factor == 8) {
                HALF_LAYERNORM_OPT(8);
            }
        }
        else {
            if (unroll_factor == 1) {
                HALF_LAYERNORM_OPT2(1);
            }
            else if (unroll_factor == 2) {
                HALF_LAYERNORM_OPT2(2);
            }
            else if (unroll_factor == 3) {
                HALF_LAYERNORM_OPT2(3);
            }
            else if (unroll_factor == 4) {
                HALF_LAYERNORM_OPT2(4);
            }
            else if (unroll_factor == 8) {
                HALF_LAYERNORM_OPT2(8);
            }
        }
    }
    else {

        dim3 grid(m);
        dim3 block(min(n, 1024));

        /* For general cases, n is equal to hidden_units, e.g., 512/1024.
        Since we have warp shuffle inside the code, block.x % 32 should be 0.
        */

        if (n % 32 != 0) {
            block.x = 1024;
        }

        block.x = block.x / (4 / sizeof(T));  // if using half, only need half of block.x

        /* should pay attention to the rsqrt precision*/
        generalScaleAddBiasResidualLayerNorm<T><<<grid, block, 0, stream>>>(
            input, gamma, beta, bias, output, norm_output, m, n, scale_input, scale_residual);  // For gpt-3
    }
}

#undef HALF_LAYERNORM_OPT
#undef HALF_LAYERNORM_OPT2

template void invokeGeneralScaleAddBiasResidualPreLayerNorm(float*       output,
                                                            float*       norm_output,
                                                            const float* input,
                                                            const float* gamma,
                                                            const float* beta,
                                                            const float* bias,
                                                            int          m,
                                                            int          n,
                                                            cudaStream_t stream,
                                                            int          opt_version,
                                                            float        scale_input,
                                                            float        scale_residual);

template void invokeGeneralScaleAddBiasResidualPreLayerNorm(half*        output,
                                                            half*        norm_output,
                                                            const half*  input,
                                                            const half*  gamma,
                                                            const half*  beta,
                                                            const half*  bias,
                                                            int          m,
                                                            int          n,
                                                            cudaStream_t stream,
                                                            int          opt_version,
                                                            float        scale_input,
                                                            float        scale_residual);

#ifdef ENABLE_BF16
template void invokeGeneralScaleAddBiasResidualPreLayerNorm(__nv_bfloat16*       output,
                                                            __nv_bfloat16*       norm_output,
                                                            const __nv_bfloat16* input,
                                                            const __nv_bfloat16* gamma,
                                                            const __nv_bfloat16* beta,
                                                            const __nv_bfloat16* bias,
                                                            int                  m,
                                                            int                  n,
                                                            cudaStream_t         stream,
                                                            int                  opt_version,
                                                            float                scale_input,
                                                            float                scale_residual);

#endif

//////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void scaleAddBiasResidual(
    T* output, const T* input, const T* bias, const int m, const int n, float scale_input, float scale_residual)
{
    const int col_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (col_index < n) {
        float local_val = output[blockIdx.x * n + col_index];
        float bias_val  = (bias == nullptr) ? (0.0f) : (float)bias[col_index];
        local_val += bias_val;
        local_val *= scale_input;
        local_val += (float)input[blockIdx.x * n + col_index] * scale_residual;
        output[blockIdx.x * n + col_index] = local_val;
    }
}

template<typename T>
void invokeScaleAddBiasResidual(T*           output,
                                const T*     input,
                                const T*     bias,
                                const int    m,
                                const int    n,
                                cudaStream_t stream,
                                float        scale_input,
                                float        scale_residual)
{
    int  blocks_per_row = ceil(float(n) / 1024);
    dim3 grid(m, blocks_per_row);
    dim3 block(min(n, 1024));
    scaleAddBiasResidual<<<grid, block, 0, stream>>>(output, input, bias, m, n, scale_input, scale_residual);
}

template void invokeScaleAddBiasResidual(float*       output,
                                         const float* input,
                                         const float* bias,
                                         const int    m,
                                         const int    n,
                                         cudaStream_t stream,
                                         float        scale_input,
                                         float        scale_residual);

template void invokeScaleAddBiasResidual(half*        output,
                                         const half*  input,
                                         const half*  bias,
                                         const int    m,
                                         const int    n,
                                         cudaStream_t stream,
                                         float        scale_input,
                                         float        scale_residual);

#ifdef ENABLE_BF16
template void invokeScaleAddBiasResidual(__nv_bfloat16*       output,
                                         const __nv_bfloat16* input,
                                         const __nv_bfloat16* bias,
                                         const int            m,
                                         const int            n,
                                         cudaStream_t         stream,
                                         float                scale_input,
                                         float                scale_residual);
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
__global__ void ConformerDepthwiseConvBias(T*        out,
                                           const T*  in,
                                           const T*  weight,
                                           const T*  bias,
                                           const int batch_size,
                                           const int seq_len,
                                           const int hidden_unit,
                                           const int kernel_size,
                                           const int pad_size)
{
    int c_id = threadIdx.x;
    int s_id = blockIdx.x % seq_len;
    int b_id = blockIdx.x / seq_len;

    int s_start = s_id - pad_size;
    int s_end   = min(s_start + kernel_size, seq_len);
    s_start     = max(s_start, 0);

    int k_start = max(pad_size - s_id, 0);

    in     = in + b_id * seq_len * hidden_unit + c_id;
    weight = weight + c_id;

    float val = 0.0f;
    for (int i = s_start; i < s_end; ++i) {
        val += (float)in[i * hidden_unit] * (float)weight[(k_start + i - s_start) * hidden_unit];
    }
    val = val + (float)bias[c_id];
    // val = val * sigmoid<float>(val);

    out[blockIdx.x * hidden_unit + c_id] = (T)val;
}

template<typename T>
void invokeConformerDepthwiseConvBias(T*           out,
                                      const T*     in,
                                      const T*     weight,
                                      const T*     bias,
                                      const int    batch_size,
                                      const int    seq_len,
                                      const int    hidden_unit,
                                      const int    kernel_size,
                                      const int    pad_size,
                                      cudaStream_t stream)
{
    FT_CHECK(hidden_unit <= 1024);
    ConformerDepthwiseConvBias<T><<<batch_size * seq_len, hidden_unit, 0, stream>>>(
        out, in, weight, bias, batch_size, seq_len, hidden_unit, kernel_size, pad_size);
}
template void invokeConformerDepthwiseConvBias(float*       out,
                                               const float* in,
                                               const float* weight,
                                               const float* bias,
                                               const int    batch_size,
                                               const int    seq_len,
                                               const int    hidden_unit,
                                               const int    kernel_size,
                                               const int    pad_size,
                                               cudaStream_t stream);

template void invokeConformerDepthwiseConvBias(half*        out,
                                               const half*  in,
                                               const half*  weight,
                                               const half*  bias,
                                               const int    batch_size,
                                               const int    seq_len,
                                               const int    hidden_unit,
                                               const int    kernel_size,
                                               const int    pad_size,
                                               cudaStream_t stream);

#ifdef ENABLE_BF16

template void invokeConformerDepthwiseConvBias(__nv_bfloat16*       out,
                                               const __nv_bfloat16* in,
                                               const __nv_bfloat16* weight,
                                               const __nv_bfloat16* bias,
                                               const int            batch_size,
                                               const int            seq_len,
                                               const int            hidden_unit,
                                               const int            kernel_size,
                                               const int            pad_size,
                                               cudaStream_t         stream);
#endif

template<typename T>
__global__ void ConformerDepthwiseConvBiasSilu(T*        out,
                                               const T*  in,
                                               const T*  weight,
                                               const T*  bias,
                                               const int batch_size,
                                               const int seq_len,
                                               const int hidden_unit,
                                               const int kernel_size,
                                               const int pad_size)
{
    int c_id = threadIdx.x;
    int s_id = blockIdx.x % seq_len;
    int b_id = blockIdx.x / seq_len;

    int s_start = s_id - pad_size;
    int s_end   = min(s_start + kernel_size, seq_len);
    s_start     = max(s_start, 0);

    int k_start = max(pad_size - s_id, 0);

    in     = in + b_id * seq_len * hidden_unit;
    weight = weight + c_id;

    float val = 0.0f;
    for (int i = s_start; i < s_end; ++i) {
        val += (float)in[i * hidden_unit + c_id] * (float)weight[(k_start + i - s_start) * hidden_unit];
    }
    val = val + (float)bias[c_id];
    val = val * sigmoid<float>(val);

    out[blockIdx.x * hidden_unit + c_id] = val;
}

template<typename T>
void invokeConformerDepthwiseConvBiasSilu(T*           out,
                                          const T*     in,
                                          const T*     weight,
                                          const T*     bias,
                                          const int    batch_size,
                                          const int    seq_len,
                                          const int    hidden_unit,
                                          const int    kernel_size,
                                          const int    pad_size,
                                          cudaStream_t stream)
{
    FT_CHECK(hidden_unit <= 1024);
    ConformerDepthwiseConvBiasSilu<T><<<batch_size * seq_len, hidden_unit, 0, stream>>>(
        out, in, weight, bias, batch_size, seq_len, hidden_unit, kernel_size, pad_size);
}
template void invokeConformerDepthwiseConvBiasSilu(float*       out,
                                                   const float* in,
                                                   const float* weight,
                                                   const float* bias,
                                                   const int    batch_size,
                                                   const int    seq_len,
                                                   const int    hidden_unit,
                                                   const int    kernel_size,
                                                   const int    pad_size,
                                                   cudaStream_t stream);

template void invokeConformerDepthwiseConvBiasSilu(half*        out,
                                                   const half*  in,
                                                   const half*  weight,
                                                   const half*  bias,
                                                   const int    batch_size,
                                                   const int    seq_len,
                                                   const int    hidden_unit,
                                                   const int    kernel_size,
                                                   const int    pad_size,
                                                   cudaStream_t stream);

#ifdef ENABLE_BF16

template void invokeConformerDepthwiseConvBiasSilu(__nv_bfloat16*       out,
                                                   const __nv_bfloat16* in,
                                                   const __nv_bfloat16* weight,
                                                   const __nv_bfloat16* bias,
                                                   const int            batch_size,
                                                   const int            seq_len,
                                                   const int            hidden_unit,
                                                   const int            kernel_size,
                                                   const int            pad_size,
                                                   cudaStream_t         stream);
#endif

/////////////////////////////////////////////////////////////////////////////

// template<typename T>
// __global__ void VarLenConformerDepthwiseConvBiasSilu(T* out,
//                                                      const T* in,
//                                                      const T* weight,
//                                                      const T* bias,
//                                                      const int* bid_start_end,
//                                                      const T* bias_before_glu,
//                                                      const int m,
//                                                      const int batch_size,
//                                                      const int seq_len,
//                                                      const int hidden_unit,
//                                                      const int kernel_size,
//                                                      const int pad_size)
// {
//     __shared__ int b_start, b_end;
//     if (threadIdx.x == 0) {
//         // b_id = bid_start_end[blockIdx.x * 3];
//         b_start = bid_start_end[blockIdx.x * 3 + 1];
//         b_end = bid_start_end[blockIdx.x * 3 + 2];
//     }
//     __syncthreads();

//     int cur_len = b_end - b_start;

//     int c_id = threadIdx.x;
//     int s_id = blockIdx.x - b_start;

//     int s_start = s_id - pad_size;
//     int s_end = min(s_start + kernel_size, seq_len);
//     s_start = max(s_start, 0);

//     int k_start = max(pad_size - s_id, 0);

//     in = in + b_start * hidden_unit;
//     weight = weight + c_id;

//     float val = 0.0f;
//     for (int i = s_start; i < s_end; ++i) {
//         float cur_in = 0.f;
//         if (i < cur_len)
//             cur_in = (float)in[i * hidden_unit + c_id];
//         else {
//             // glu(0+bias)
//             cur_in =
//                 (float)ldg(&bias_before_glu[c_id]) * sigmoid<float>((float)ldg(&bias_before_glu[c_id +
//                 hidden_unit]));
//         }
//         val += cur_in * (float)weight[(k_start + i - s_start) * hidden_unit];
//     }
//     val = val + (float)bias[c_id];
//     val = val * sigmoid<float>(val);

//     out[blockIdx.x * hidden_unit + c_id] = val;
// }

// template<typename T>
// void invokeVarLenConformerDepthwiseConvBiasSilu(T* out,
//                                                 const T* in,
//                                                 const T* weight,
//                                                 const T* bias,
//                                                 const int* bid_start_end,
//                                                 const T* bias_before_glu,
//                                                 const int m,
//                                                 const int batch_size,
//                                                 const int seq_len,
//                                                 const int hidden_unit,
//                                                 const int kernel_size,
//                                                 const int pad_size,
//                                                 cudaStream_t stream)
// {
//     FT_CHECK(hidden_unit <= 1024);
//     VarLenConformerDepthwiseConvBiasSilu<T><<<m, hidden_unit, 0, stream>>>(out,
//                                                                            in,
//                                                                            weight,
//                                                                            bias,
//                                                                            bid_start_end,
//                                                                            bias_before_glu,
//                                                                            m,
//                                                                            batch_size,
//                                                                            seq_len,
//                                                                            hidden_unit,
//                                                                            kernel_size,
//                                                                            pad_size);
// }
// template void invokeVarLenConformerDepthwiseConvBiasSilu(float* out,
//                                                          const float* in,
//                                                          const float* weight,
//                                                          const float* bias,
//                                                          const int* bid_start_end,
//                                                          const float* bias_before_glu,
//                                                          const int m,
//                                                          const int batch_size,
//                                                          const int seq_len,
//                                                          const int hidden_unit,
//                                                          const int kernel_size,
//                                                          const int pad_size,
//                                                          cudaStream_t stream);

// template void invokeVarLenConformerDepthwiseConvBiasSilu(half* out,
//                                                          const half* in,
//                                                          const half* weight,
//                                                          const half* bias,
//                                                          const int* bid_start_end,
//                                                          const half* bias_before_glu,
//                                                          const int m,
//                                                          const int batch_size,
//                                                          const int seq_len,
//                                                          const int hidden_unit,
//                                                          const int kernel_size,
//                                                          const int pad_size,
//                                                          cudaStream_t stream);

// #ifdef ENABLE_BF16

// template void invokeVarLenConformerDepthwiseConvBiasSilu(__nv_bfloat16* out,
//                                                          const __nv_bfloat16* in,
//                                                          const __nv_bfloat16* weight,
//                                                          const __nv_bfloat16* bias,
//                                                          const int* bid_start_end,
//                                                          const __nv_bfloat16* bias_before_glu,
//                                                          const int m,
//                                                          const int batch_size,
//                                                          const int seq_len,
//                                                          const int hidden_unit,
//                                                          const int kernel_size,
//                                                          const int pad_size,
//                                                          cudaStream_t stream);
// #endif

////////////////////////////////////////////////////////////////////////////

// __global__ void
// getBatchIDStartEndKernel(int* bid_start_end, const int* sequence_length, const int batch_size, const int seq_len)
// {
//     int index = 0;
//     int total_seq_len = 0;
//     for (int i = 0; i < batch_size; i++) {
//         const int seq_len = sequence_length[i];
//         for (int j = 0; j < seq_len; j++) {
//             bid_start_end[index * 3] = i;
//             bid_start_end[index * 3 + 1] = total_seq_len;
//             bid_start_end[index * 3 + 2] = total_seq_len + seq_len;
//             index++;
//         }
//         total_seq_len += seq_len;
//     }
// }

// void invokeGetBatchIDStartEnd(
//     int* bid_start_end, const int* sequence_length, const int batch_size, const int seq_len, cudaStream_t stream)
// {
//     getBatchIDStartEndKernel<<<1, 1, 0, stream>>>(bid_start_end, sequence_length, batch_size, seq_len);
// }

////////////////////////////////////////////////////////////////////////////

// template<typename T>
// __global__ void bias_rebuild_padding(const T* src, T* dst, const T* bias, const int* padding_offset, const int n)
// {
//     const int tid = threadIdx.x;
//     const int bid = blockIdx.x;
//     const int dst_seq_id = bid + padding_offset[bid];
//     const int src_seq_id = bid;

//     for (int i = tid; i < n; i += blockDim.x) {
//         if (bias != nullptr)
//             dst[dst_seq_id * n + i] = src[src_seq_id * n + i] + bias[i];
//         else
//             dst[dst_seq_id * n + i] = src[src_seq_id * n + i];
//     }
// }

// template<typename T>
// void invokeBiasRebuildPadding(
//     T* dst, const T* src, const T* bias, const int* padding_offset, const int m, const int n, cudaStream_t stream)
// {
//     // src: [token_num, hidden_dim]
//     // dst: [batch_size*max_seq_len, hidden_dim]
//     bias_rebuild_padding<<<m, 256, 0, stream>>>(src, dst, bias, padding_offset, n);
// }

// template void invokeBiasRebuildPadding(float* dst,
//                                        const float* src,
//                                        const float* bias,
//                                        const int* padding_offset,
//                                        const int token_num,
//                                        const int hidden_dim,
//                                        cudaStream_t stream);

// template void invokeBiasRebuildPadding(half* dst,
//                                        const half* src,
//                                        const half* bias,
//                                        const int* padding_offset,
//                                        const int token_num,
//                                        const int hidden_dim,
//                                        cudaStream_t stream);

// #ifdef ENABLE_BF16
// template void invokeBiasRebuildPadding(__nv_bfloat16* dst,
//                                        const __nv_bfloat16* src,
//                                        const __nv_bfloat16* bias,
//                                        const int* padding_offset,
//                                        const int token_num,
//                                        const int hidden_dim,
//                                        cudaStream_t stream);
// #endif  // ENABLE_BF16

// Encoder
template<typename T>
__global__ void addQKVPBiasTranspose(T* q_out,
                                     T* k_out,
                                     T* v_out,
                                     const T* __restrict q_in,
                                     const T* __restrict bias_q,
                                     const T* __restrict k_in,
                                     const T* __restrict bias_k,
                                     const T* __restrict v_in,
                                     const T* __restrict bias_v,
                                     T*        p_buf,
                                     const T*  P,
                                     T*        q_buf_bias_v,
                                     const T*  pos_bias_u,
                                     const T*  pos_bias_v,
                                     const int batch_size,
                                     const int seq_len,
                                     const int head_num,
                                     const int size_per_head)
{
    const int n        = head_num * size_per_head;
    const int batch_id = blockIdx.x;
    const int word_id  = blockIdx.y;
    const int row_id   = batch_id * seq_len + word_id;
    for (int col_id = threadIdx.x; col_id < n; col_id += blockDim.x) {
        const int head_id   = col_id / size_per_head;
        const int size_id   = col_id % size_per_head;
        const int target_id = batch_id * (head_num * seq_len * size_per_head) + head_id * seq_len * size_per_head
                              + word_id * size_per_head + size_id;
        const int src_id = row_id * n + col_id;
        /*
                q_out[target_id] = __ldg(&q_in[src_id]);
                q_out[target_id] = q_out[target_id] + __ldg(&bias_q[col_id]);

                k_out[target_id] = __ldg(&k_in[src_id]);
                k_out[target_id] = k_out[target_id] + __ldg(&bias_k[col_id]);

                v_out[target_id] = __ldg(&v_in[src_id]);
                v_out[target_id] = v_out[target_id] + __ldg(&bias_v[col_id]);
        */
        T q_val          = __ldg(&q_in[src_id]) + __ldg(&bias_q[col_id]);
        q_out[target_id] = q_val + __ldg(&pos_bias_u[col_id]);

        k_out[target_id] = __ldg(&k_in[src_id]) + __ldg(&bias_k[col_id]);

        v_out[target_id] = __ldg(&v_in[src_id]) + __ldg(&bias_v[col_id]);

        p_buf[target_id] = __ldg(&P[src_id]);

        q_buf_bias_v[target_id] = q_val + __ldg(&pos_bias_v[col_id]);
    }
}

template<typename T>
__global__ void QKVPTranspose(T* q_out,
                              T* k_out,
                              T* v_out,
                              const T* __restrict q_in,
                              const T* __restrict k_in,
                              const T* __restrict v_in,
                              T*        p_buf,
                              const T*  P,
                              T*        q_buf_bias_v,
                              const T*  pos_bias_u,
                              const T*  pos_bias_v,
                              const int batch_size,
                              const int seq_len,
                              const int head_num,
                              const int size_per_head)
{
    const int n        = head_num * size_per_head;
    const int batch_id = blockIdx.x;
    const int word_id  = blockIdx.y;
    const int row_id   = batch_id * seq_len + word_id;
    for (int col_id = threadIdx.x; col_id < n; col_id += blockDim.x) {
        const int head_id   = col_id / size_per_head;
        const int size_id   = col_id % size_per_head;
        const int target_id = batch_id * (head_num * seq_len * size_per_head) + head_id * seq_len * size_per_head
                              + word_id * size_per_head + size_id;
        const int src_id = row_id * n + col_id;

        T q_val          = __ldg(&q_in[src_id]);
        q_out[target_id] = q_val + __ldg(&pos_bias_u[col_id]);

        k_out[target_id] = __ldg(&k_in[src_id]);
        v_out[target_id] = __ldg(&v_in[src_id]);

        p_buf[target_id]        = __ldg(&P[src_id]);
        q_buf_bias_v[target_id] = q_val + __ldg(&pos_bias_v[col_id]);
    }
}

template<typename T>
void invokeAddQKVPBiasTranspose(T*           q_buf,
                                T*           k_buf,
                                T*           v_buf,
                                T*           Q,
                                const T*     bias_Q,
                                T*           K,
                                const T*     bias_K,
                                T*           V,
                                const T*     bias_V,
                                T*           p_buf,
                                T*           P,
                                T*           q_buf_bias_v,
                                const T*     pos_bias_u,
                                const T*     pos_bias_v,
                                const int    batch_size,
                                const int    seq_len,
                                const int    head_num,
                                const int    size_per_head,
                                cudaStream_t stream)
{
    const int k = head_num * size_per_head;
    dim3      grid(batch_size, seq_len);
    bool      is_add_bias = bias_Q != nullptr;
    if (sizeof(T) == 4 || k % 2 != 0) {
        dim3 block(min(k, 512));
        if (is_add_bias) {
            addQKVPBiasTranspose<T><<<grid, block, 0, stream>>>(q_buf,
                                                                k_buf,
                                                                v_buf,
                                                                Q,
                                                                bias_Q,
                                                                K,
                                                                bias_K,
                                                                V,
                                                                bias_V,
                                                                p_buf,
                                                                P,
                                                                q_buf_bias_v,
                                                                pos_bias_u,
                                                                pos_bias_v,
                                                                batch_size,
                                                                seq_len,
                                                                head_num,
                                                                size_per_head);
        }
        else {
            QKVPTranspose<T><<<grid, block, 0, stream>>>(q_buf,
                                                         k_buf,
                                                         v_buf,
                                                         Q,
                                                         K,
                                                         V,
                                                         p_buf,
                                                         P,
                                                         q_buf_bias_v,
                                                         pos_bias_u,
                                                         pos_bias_v,
                                                         batch_size,
                                                         seq_len,
                                                         head_num,
                                                         size_per_head);
        }
        sync_check_cuda_error();
    }
    else {
        dim3 block(min(k / 2, 512));
        if (is_add_bias) {
            addQKVPBiasTranspose<half2><<<grid, block, 0, stream>>>((half2*)q_buf,
                                                                    (half2*)k_buf,
                                                                    (half2*)v_buf,
                                                                    (const half2*)Q,
                                                                    (const half2*)bias_Q,
                                                                    (const half2*)K,
                                                                    (const half2*)bias_K,
                                                                    (const half2*)V,
                                                                    (const half2*)bias_V,
                                                                    (half2*)p_buf,
                                                                    (const half2*)P,
                                                                    (half2*)q_buf_bias_v,
                                                                    (const half2*)pos_bias_u,
                                                                    (const half2*)pos_bias_v,
                                                                    batch_size,
                                                                    seq_len,
                                                                    head_num,
                                                                    size_per_head / 2);
        }
        else {
            QKVPTranspose<half2><<<grid, block, 0, stream>>>((half2*)q_buf,
                                                             (half2*)k_buf,
                                                             (half2*)v_buf,
                                                             (const half2*)Q,
                                                             (const half2*)K,
                                                             (const half2*)V,
                                                             (half2*)p_buf,
                                                             (const half2*)P,
                                                             (half2*)q_buf_bias_v,
                                                             (const half2*)pos_bias_u,
                                                             (const half2*)pos_bias_v,
                                                             batch_size,
                                                             seq_len,
                                                             head_num,
                                                             size_per_head / 2);
        }
        sync_check_cuda_error();
    }
}

template void invokeAddQKVPBiasTranspose(float*       q_buf,
                                         float*       k_buf,
                                         float*       v_buf,
                                         float*       Q,
                                         const float* bias_Q,
                                         float*       K,
                                         const float* bias_K,
                                         float*       V,
                                         const float* bias_V,
                                         float*       p_buf,
                                         float*       P,
                                         float*       q_buf_bias_v,
                                         const float* pos_bias_u,
                                         const float* pos_bias_v,
                                         const int    batch_size,
                                         const int    seq_len,
                                         const int    head_num,
                                         const int    size_per_head,
                                         cudaStream_t stream);

template void invokeAddQKVPBiasTranspose(half*        q_buf,
                                         half*        k_buf,
                                         half*        v_buf,
                                         half*        Q,
                                         const half*  bias_Q,
                                         half*        K,
                                         const half*  bias_K,
                                         half*        V,
                                         const half*  bias_V,
                                         half*        p_buf,
                                         half*        P,
                                         half*        q_buf_bias_v,
                                         const half*  pos_bias_u,
                                         const half*  pos_bias_v,
                                         const int    batch_size,
                                         const int    seq_len,
                                         const int    head_num,
                                         const int    size_per_head,
                                         cudaStream_t stream);

// TODO(bhsueh) Rename the softmax_kernel_v4 to softmax_kernel
template<int ITEMS_PER_THREAD, typename T, typename T_IN>
__global__ void add_softmax_kernel_v4(T*          qk_buf_,
                                      const T_IN* qk_buf_src,
                                      const T_IN* qp_buf_src,
                                      const T*    attr_mask,
                                      const int   batch_size,
                                      const int   head_num,
                                      const int   seq_len,
                                      const T     scalar)
{
    for (int seq_id = blockIdx.x; seq_id < seq_len; seq_id += gridDim.x) {
        float            data[ITEMS_PER_THREAD];
        int              qk_offset;
        __shared__ float s_mean, s_max;
        float            local_max = -1e20f;
        for (int i = 0; blockDim.x * i + threadIdx.x < seq_len; i++) {
            qk_offset =
                ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id) * seq_len + blockDim.x * i + threadIdx.x;
            int mask_offset = (blockIdx.y * seq_len + seq_id) * seq_len + blockDim.x * i + threadIdx.x;

            float qk = static_cast<float>(qk_buf_src[qk_offset]);
            qk += static_cast<float>(qp_buf_src[qk_offset]);

            float mask_val = static_cast<float>(ldg(&attr_mask[mask_offset]));

            mask_val = (1.0f - mask_val) * -10000.0f;

            data[i]   = qk * static_cast<float>(scalar) + mask_val;
            local_max = fmax(local_max, data[i]);
        }

        float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float local_sum = 0;
        for (int i = 0; blockDim.x * i + threadIdx.x < seq_len; i++) {
            data[i] = __expf(data[i] - s_max);
            local_sum += data[i];
        }
        float sum_val = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);
        if (threadIdx.x == 0) {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < seq_len; i++) {
            qk_offset =
                ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id) * seq_len + blockDim.x * i + threadIdx.x;
            qk_buf_[qk_offset] = (T)(data[i] * s_mean);
        }
    }
}

template<typename T, int ITEMS_PER_THREAD>
__global__ void add_softmax_kernel_v4_half2(T*        qk_buf_,
                                            const T*  qp_buf_,
                                            const T*  attr_mask,
                                            const int batch_size,
                                            const int head_num,
                                            const int seq_len,
                                            const T   scalar)
{
    using T2                  = typename TypeConverter<T>::Type;
    T2*       qk_buf_half2    = (T2*)qk_buf_;
    T2*       qp_buf_half2    = (T2*)qp_buf_;
    const T2* attr_mask_half2 = (const T2*)attr_mask;

    for (int seq_id = blockIdx.x; seq_id < seq_len; seq_id += gridDim.x) {
        T2               data[ITEMS_PER_THREAD];
        int              qk_offset;
        __shared__ float s_mean, s_max;
        float            local_max = -1e20f;
        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
            qk_offset = ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id) * (seq_len / 2) + blockDim.x * i
                        + threadIdx.x;
            int mask_offset = (blockIdx.y * seq_len + seq_id) * (seq_len / 2) + blockDim.x * i + threadIdx.x;

            T2 qk       = qk_buf_half2[qk_offset];
            qk          = hadd2<T2>(qk, qp_buf_half2[qk_offset]);
            T2 mask_val = ldg(&attr_mask_half2[mask_offset]);
            mask_val    = hmul2<T2>(hsub2<T2>(cuda_cast<T2>(1.0f), mask_val), cuda_cast<T2>(-10000.0f));

            data[i] = hadd2<T2>(hmul2<T2>(qk, cuda_cast<T2>(scalar)), mask_val);

            local_max = fmax(local_max, fmax((float)data[i].x, (float)data[i].y));
        }

        float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float local_sum = 0;
        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
            data[i] = hexp2<T2>(hsub2<T2>(data[i], cuda_cast<T2>(s_max)));
            local_sum += (float)(data[i].x + data[i].y);
        }

        float sum_val = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);

        if (threadIdx.x == 0) {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
            qk_offset = ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id) * (seq_len / 2) + blockDim.x * i
                        + threadIdx.x;
            qk_buf_half2[qk_offset] = hmul2<T2>(data[i], cuda_cast<T2>(s_mean));
        }
    }
}

template<typename T, int ITEMS_PER_THREAD, int NUM>
__global__ void add_softmax_kernel_v5_half2(T*        qk_buf_,
                                            const T*  qp_buf_,
                                            const T*  attr_mask,
                                            const int batch_size,
                                            const int head_num,
                                            const int seq_len,
                                            const T   scalar)
{
    using T2         = typename TypeConverter<T>::Type;
    T2* qk_buf_half2 = (T2*)qk_buf_;
    T2* qp_buf_half2 = (T2*)qp_buf_;

    const T2* attr_mask_half2 = (const T2*)attr_mask;

    for (int seq_id = blockIdx.x; seq_id < seq_len; seq_id += gridDim.x * NUM) {
        T2 data[NUM][ITEMS_PER_THREAD];

        int qk_offset[NUM];

        __shared__ float s_sum[NUM], s_max[NUM];
        float            local_max[NUM];
#pragma unroll
        for (int j = 0; j < NUM; j++) {
            local_max[j] = -1e20f;
        }

        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
            int mask_offset[NUM];
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                qk_offset[j] = ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id + j * gridDim.x) * (seq_len / 2)
                               + blockDim.x * i + threadIdx.x;
                mask_offset[j] =
                    (blockIdx.y * seq_len + seq_id + j * gridDim.x) * (seq_len / 2) + blockDim.x * i + threadIdx.x;
            }

            T2 mask_val[NUM];
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                mask_val[j] = ldg(&attr_mask_half2[mask_offset[j]]);
            }

            T2 qk[NUM];
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                qk[j] = qk_buf_half2[qk_offset[j]];
                qk[j] = hadd2<T2>(qk[j], qp_buf_half2[qk_offset[j]]);
            }

#pragma unroll
            for (int j = 0; j < NUM; j++) {
                mask_val[j] = hmul2<T2>(hsub2<T2>(cuda_cast<T2>(1.0f), mask_val[j]), cuda_cast<T2>(-10000.0f));
            }

#pragma unroll
            for (int j = 0; j < NUM; j++) {
                data[j][i]   = hadd2<T2>(hmul2<T2>(qk[j], cuda_cast<T2>(scalar)), mask_val[j]);
                local_max[j] = fmax(local_max[j], fmax((float)data[j][i].x, (float)data[j][i].y));
            }
        }

        if (blockDim.x <= 32) {
            warpReduceMaxV2<float, NUM>(local_max);
        }
        else {
            blockReduceMaxV2<float, NUM>(local_max);
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                s_max[j] = local_max[j];
            }
        }
        __syncthreads();

        float local_sum[NUM];
#pragma unroll
        for (int j = 0; j < NUM; j++) {
            local_sum[j] = {0.f};
        }

        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                data[j][i] = hexp2<T2>(hsub2<T2>(data[j][i], cuda_cast<T2>(s_max[j])));
            }

#pragma unroll
            for (int j = 0; j < NUM; j++) {
                local_sum[j] += (float)(data[j][i].x + data[j][i].y);
            }
        }

        if (blockDim.x <= 32) {
            warpReduceSumV2<float, NUM>(local_sum);
        }
        else {
            blockReduceSumV2<float, NUM>(local_sum);
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                s_sum[j] = __fdividef(1.0f, local_sum[j] + 1e-6f);
            }
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                qk_offset[j] = ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id + j * gridDim.x) * (seq_len / 2)
                               + blockDim.x * i + threadIdx.x;
            }

#pragma unroll
            for (int j = 0; j < NUM; j++) {
                qk_buf_half2[qk_offset[j]] = hmul2<T2>(data[j][i], cuda_cast<T2>(s_sum[j]));
            }
        }
    }
}

#define SOFTMAX_KERNEL(ITEMS_PER_THREAD)                                                                               \
    block.x /= ITEMS_PER_THREAD;                                                                                       \
    assert(block.x <= 1024);                                                                                           \
    if (is_half2) {                                                                                                    \
        if (grid.x % 4 == 0) {                                                                                         \
            grid.x /= 4;                                                                                               \
            add_softmax_kernel_v5_half2<half, ITEMS_PER_THREAD, 4><<<grid, block, 0, stream>>>((half*)buffer,          \
                                                                                               (const half*)qp_buf,    \
                                                                                               (const half*)attr_mask, \
                                                                                               batch_size,             \
                                                                                               head_num,               \
                                                                                               seq_len,                \
                                                                                               (const half)scalar);    \
        }                                                                                                              \
        else {                                                                                                         \
            add_softmax_kernel_v4_half2<half, ITEMS_PER_THREAD><<<grid, block, 0, stream>>>((half*)buffer,             \
                                                                                            (const half*)qp_buf,       \
                                                                                            (const half*)attr_mask,    \
                                                                                            batch_size,                \
                                                                                            head_num,                  \
                                                                                            seq_len,                   \
                                                                                            (const half)scalar);       \
        }                                                                                                              \
    }                                                                                                                  \
    else {                                                                                                             \
        add_softmax_kernel_v4<ITEMS_PER_THREAD, T, T_IN><<<grid, block, 0, stream>>>(                                  \
            buffer, buffer_src, qp_buf, attr_mask, batch_size, head_num, seq_len, scalar);                             \
    }

#ifdef ENABLE_BF16
#define SOFTMAX_KERNEL_BF16(ITEMS_PER_THREAD)                                                                          \
    block.x /= ITEMS_PER_THREAD;                                                                                       \
    assert(block.x <= 1024);                                                                                           \
    if (is_half2) {                                                                                                    \
        if (grid.x % 4 == 0) {                                                                                         \
            grid.x /= 4;                                                                                               \
            add_softmax_kernel_v5_half2<__nv_bfloat16, ITEMS_PER_THREAD, 4>                                            \
                <<<grid, block, 0, stream>>>((__nv_bfloat16*)buffer,                                                   \
                                             (const __nv_bfloat16*)qp_buf,                                             \
                                             (const __nv_bfloat16*)attr_mask,                                          \
                                             batch_size,                                                               \
                                             head_num,                                                                 \
                                             seq_len,                                                                  \
                                             (const __nv_bfloat16)scalar);                                             \
        }                                                                                                              \
        else {                                                                                                         \
            add_softmax_kernel_v4_half2<__nv_bfloat16, ITEMS_PER_THREAD>                                               \
                <<<grid, block, 0, stream>>>((__nv_bfloat16*)buffer,                                                   \
                                             (const __nv_bfloat16*)qp_buf,                                             \
                                             (const __nv_bfloat16*)attr_mask,                                          \
                                             batch_size,                                                               \
                                             head_num,                                                                 \
                                             seq_len,                                                                  \
                                             (const __nv_bfloat16)scalar);                                             \
        }                                                                                                              \
    }                                                                                                                  \
    else {                                                                                                             \
        add_softmax_kernel_v4<ITEMS_PER_THREAD, __nv_bfloat16, T_IN><<<grid, block, 0, stream>>>(                      \
            buffer, buffer_src, qp_buf, attr_mask, batch_size, head_num, seq_len, scalar);                             \
    }
#endif  // ENABLE_BF16

template<typename T, typename T_IN>
void invokeAddMaskedSoftMax(T*           buffer,
                            const T_IN*  buffer_src,
                            const T_IN*  qp_buf,
                            const T*     attr_mask,
                            const int    batch_size,
                            const int    seq_len,
                            const int    head_num,
                            const T      scalar,
                            cudaStream_t stream)
{

    dim3 grid(seq_len, batch_size, head_num);
    if (batch_size * head_num > 360) {
        grid.x = ceil(float(seq_len) / 32.0f);
    }

    bool is_half2 = sizeof(T) == 2 && sizeof(T_IN) == 2 && seq_len % 2 == 0;
    dim3 block((seq_len / (is_half2 ? 2 : 1) + 31) / 32 * 32);

    if (block.x > 3072 && block.x <= 4096) {
        SOFTMAX_KERNEL(4)
    }
    if (block.x > 2048) {
        SOFTMAX_KERNEL(3)
    }
    else if (block.x > 1024) {
        SOFTMAX_KERNEL(2)
    }
    else if (block.x > 0) {
        SOFTMAX_KERNEL(1)
    }
    else {
        FT_CHECK(seq_len <= 4096);
    }
}

#ifdef ENABLE_BF16
template<>
void invokeAddMaskedSoftMax(__nv_bfloat16*       buffer,
                            const __nv_bfloat16* buffer_src,
                            const __nv_bfloat16* qp_buf,
                            const __nv_bfloat16* attr_mask,
                            const int            batch_size,
                            const int            seq_len,
                            const int            head_num,
                            const __nv_bfloat16  scalar,
                            cudaStream_t         stream)
{

    using T_IN = __nv_bfloat16;
    dim3 grid(seq_len, batch_size, head_num);
    if (batch_size * head_num > 360) {
        grid.x = ceil(float(seq_len) / 32.0f);
    }

    bool is_half2 = seq_len % 2 == 0;
    dim3 block((seq_len / (is_half2 ? 2 : 1) + 31) / 32 * 32);

    if (block.x > 3072 && block.x <= 4096) {
        SOFTMAX_KERNEL_BF16(4)
    }
    if (block.x > 2048) {
        SOFTMAX_KERNEL_BF16(3)
    }
    else if (block.x > 1024) {
        SOFTMAX_KERNEL_BF16(2)
    }
    else if (block.x > 0) {
        SOFTMAX_KERNEL_BF16(1)
    }
    else {
        FT_CHECK(seq_len <= 4096);
    }
}

template<>
void invokeAddMaskedSoftMax(__nv_bfloat16*       buffer,
                            const float*         buffer_src,
                            const float*         qp_buf,
                            const __nv_bfloat16* attr_mask,
                            const int            batch_size,
                            const int            seq_len,
                            const int            head_num,
                            const __nv_bfloat16  scalar,
                            cudaStream_t         stream)
{
    using T_IN = float;
    dim3 grid(seq_len, batch_size, head_num);
    if (batch_size * head_num > 360) {
        grid.x = ceil(float(seq_len) / 32.0f);
    }

    bool is_half2 = false;
    dim3 block((seq_len / (is_half2 ? 2 : 1) + 31) / 32 * 32);

    if (block.x > 3072 && block.x <= 4096) {
        SOFTMAX_KERNEL_BF16(4)
    }
    if (block.x > 2048) {
        SOFTMAX_KERNEL_BF16(3)
    }
    else if (block.x > 1024) {
        SOFTMAX_KERNEL_BF16(2)
    }
    else if (block.x > 0) {
        SOFTMAX_KERNEL_BF16(1)
    }
    else {
        FT_CHECK(seq_len <= 4096);
    }
}
#endif  // ENABLE_BF16

template void invokeAddMaskedSoftMax(float*       buffer,
                                     const float* buffer_src,
                                     const float* qp_buf,
                                     const float* attr_mask,
                                     const int    batch_size,
                                     const int    seq_len,
                                     const int    head_num,
                                     const float  scalar,
                                     cudaStream_t stream);

template void invokeAddMaskedSoftMax(half*        buffer,
                                     const float* buffer_src,
                                     const float* qp_buf,
                                     const half*  attr_mask,
                                     const int    batch_size,
                                     const int    seq_len,
                                     const int    head_num,
                                     const half   scalar,
                                     cudaStream_t stream);

template void invokeAddMaskedSoftMax(half*        buffer,
                                     const half*  buffer_src,
                                     const half*  qp_buf,
                                     const half*  attr_mask,
                                     const int    batch_size,
                                     const int    seq_len,
                                     const int    head_num,
                                     const half   scalar,
                                     cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeAddMaskedSoftMax(__nv_bfloat16*       buffer,
                                     const __nv_bfloat16* buffer_src,
                                     const __nv_bfloat16* qp_buf,
                                     const __nv_bfloat16* attr_mask,
                                     const int            batch_size,
                                     const int            seq_len,
                                     const int            head_num,
                                     const __nv_bfloat16  scalar,
                                     cudaStream_t         stream);

template void invokeAddMaskedSoftMax(__nv_bfloat16*       buffer,
                                     const float*         buffer_src,
                                     const float*         qp_buf,
                                     const __nv_bfloat16* attr_mask,
                                     const int            batch_size,
                                     const int            seq_len,
                                     const int            head_num,
                                     const __nv_bfloat16  scalar,
                                     cudaStream_t         stream);
#endif  // ENABLE_BF16

// Decoder invokeAddQKVBiasTranspose
template<typename T>
__global__ void addQKVBiasTranspose(T* q_out,
                                    T* k_out,
                                    T* v_out,
                                    const T* __restrict q_in,
                                    const T* __restrict bias_q,
                                    const T* __restrict k_in,
                                    const T* __restrict bias_k,
                                    const T* __restrict v_in,
                                    const T* __restrict bias_v,
                                    const int batch_size,
                                    const int seq_len1,
                                    const int seq_len2,
                                    const int head_num,
                                    const int size_per_head)
{
    const int n        = head_num * size_per_head;
    const int batch_id = blockIdx.x;
    const int word_id  = blockIdx.y;
    const int row_id1  = batch_id * seq_len1 + word_id;
    const int row_id2  = batch_id * seq_len2 + word_id;

    for (int col_id = threadIdx.x; col_id < n; col_id += blockDim.x) {
        const int head_id    = col_id / size_per_head;
        const int size_id    = col_id % size_per_head;
        const int target_id1 = batch_id * (head_num * seq_len1 * size_per_head) + head_id * seq_len1 * size_per_head
                               + word_id * size_per_head + size_id;
        const int target_id2 = batch_id * (head_num * seq_len2 * size_per_head) + head_id * seq_len2 * size_per_head
                               + word_id * size_per_head + size_id;

        const int src_id1 = row_id1 * n + col_id;
        const int src_id2 = row_id2 * n + col_id;

        if (word_id < seq_len1)
            q_out[target_id1] = __ldg(&q_in[src_id1]) + __ldg(&bias_q[col_id]);

        if (word_id < seq_len2) {
            k_out[target_id2] = __ldg(&k_in[src_id2]) + __ldg(&bias_k[col_id]);

            v_out[target_id2] = __ldg(&v_in[src_id2]) + __ldg(&bias_v[col_id]);
        }
    }
}

template<typename T>
__global__ void QKVTranspose(T* q_out,
                             T* k_out,
                             T* v_out,
                             const T* __restrict q_in,
                             const T* __restrict k_in,
                             const T* __restrict v_in,
                             const int batch_size,
                             const int seq_len1,
                             const int seq_len2,
                             const int head_num,
                             const int size_per_head)
{
    const int n        = head_num * size_per_head;
    const int batch_id = blockIdx.x;
    const int word_id  = blockIdx.y;
    const int row_id1  = batch_id * seq_len1 + word_id;
    const int row_id2  = batch_id * seq_len2 + word_id;

    for (int col_id = threadIdx.x; col_id < n; col_id += blockDim.x) {
        const int head_id    = col_id / size_per_head;
        const int size_id    = col_id % size_per_head;
        const int target_id1 = batch_id * (head_num * seq_len1 * size_per_head) + head_id * seq_len1 * size_per_head
                               + word_id * size_per_head + size_id;
        const int target_id2 = batch_id * (head_num * seq_len2 * size_per_head) + head_id * seq_len2 * size_per_head
                               + word_id * size_per_head + size_id;

        const int src_id1 = row_id1 * n + col_id;
        const int src_id2 = row_id2 * n + col_id;

        if (word_id < seq_len1)
            q_out[target_id1] = __ldg(&q_in[src_id1]);

        if (word_id < seq_len2) {
            k_out[target_id2] = __ldg(&k_in[src_id2]);
            v_out[target_id2] = __ldg(&v_in[src_id2]);
        }
    }
}

template<typename T>
void invokeAddQKVBiasTranspose(T*           q_buf,
                               T*           k_buf,
                               T*           v_buf,
                               T*           Q,
                               const T*     bias_Q,
                               T*           K,
                               const T*     bias_K,
                               T*           V,
                               const T*     bias_V,
                               const int    batch_size,
                               const int    seq_len1,
                               const int    seq_len2,
                               const int    head_num,
                               const int    size_per_head,
                               cudaStream_t stream)
{
    const int k         = head_num * size_per_head;
    int       seq_len12 = max(seq_len1, seq_len2);
    dim3      grid(batch_size, seq_len12);
    bool      is_add_bias = bias_Q != nullptr;
    if (sizeof(T) == 4 || k % 2 != 0) {
        dim3 block(min(k, 512));
        if (is_add_bias) {
            addQKVBiasTranspose<T><<<grid, block, 0, stream>>>(q_buf,
                                                               k_buf,
                                                               v_buf,
                                                               Q,
                                                               bias_Q,
                                                               K,
                                                               bias_K,
                                                               V,
                                                               bias_V,
                                                               batch_size,
                                                               seq_len1,
                                                               seq_len2,
                                                               head_num,
                                                               size_per_head);
        }
        else {
            QKVTranspose<T><<<grid, block, 0, stream>>>(
                q_buf, k_buf, v_buf, Q, K, V, batch_size, seq_len1, seq_len2, head_num, size_per_head);
        }
        sync_check_cuda_error();
    }
    else {
        dim3 block(min(k / 2, 512));
        if (is_add_bias) {
            addQKVBiasTranspose<half2><<<grid, block, 0, stream>>>((half2*)q_buf,
                                                                   (half2*)k_buf,
                                                                   (half2*)v_buf,
                                                                   (const half2*)Q,
                                                                   (const half2*)bias_Q,
                                                                   (const half2*)K,
                                                                   (const half2*)bias_K,
                                                                   (const half2*)V,
                                                                   (const half2*)bias_V,
                                                                   batch_size,
                                                                   seq_len1,
                                                                   seq_len2,
                                                                   head_num,
                                                                   size_per_head / 2);
        }
        else {
            QKVTranspose<half2><<<grid, block, 0, stream>>>((half2*)q_buf,
                                                            (half2*)k_buf,
                                                            (half2*)v_buf,
                                                            (const half2*)Q,
                                                            (const half2*)K,
                                                            (const half2*)V,
                                                            batch_size,
                                                            seq_len1,
                                                            seq_len2,
                                                            head_num,
                                                            size_per_head / 2);
        }
        sync_check_cuda_error();
    }
}

template void invokeAddQKVBiasTranspose(float*       q_buf,
                                        float*       k_buf,
                                        float*       v_buf,
                                        float*       Q,
                                        const float* bias_Q,
                                        float*       K,
                                        const float* bias_K,
                                        float*       V,
                                        const float* bias_V,
                                        const int    batch_size,
                                        const int    seq_len1,
                                        const int    seq_len2,
                                        const int    head_num,
                                        const int    size_per_head,
                                        cudaStream_t stream);

template void invokeAddQKVBiasTranspose(half*        q_buf,
                                        half*        k_buf,
                                        half*        v_buf,
                                        half*        Q,
                                        const half*  bias_Q,
                                        half*        K,
                                        const half*  bias_K,
                                        half*        V,
                                        const half*  bias_V,
                                        const int    batch_size,
                                        const int    seq_len1,
                                        const int    seq_len2,
                                        const int    head_num,
                                        const int    size_per_head,
                                        cudaStream_t stream);

// Tile
template<typename T>
__global__ void repeat_beam_size(T* out, const T* __restrict in, int beam_size, int n)
{
    int bid = blockIdx.x / beam_size;
    out     = out + blockIdx.x * n;
    in      = in + bid * n;
    for (int id = threadIdx.x; id < n; id += blockDim.x) {
        out[id] = in[id];
    }
}

template<typename T>
void invokeRepeatBeamSize(T* out, const T* in, const int m, const int n, const int beam_size, cudaStream_t stream)
{
    dim3 block, grid;
    block.x = std::min(1024, n);
    grid.x  = m * beam_size;
    repeat_beam_size<T><<<grid, block, 0, stream>>>(out, in, beam_size, n);
}

template void
invokeRepeatBeamSize(float* out, const float* in, const int m, const int n, const int beam_size, cudaStream_t stream);
template void
invokeRepeatBeamSize(half* out, const half* in, const int m, const int n, const int beam_size, cudaStream_t stream);
template void
invokeRepeatBeamSize(int* out, const int* in, const int m, const int n, const int beam_size, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeRepeatBeamSize(
    __nv_bfloat16* out, const __nv_bfloat16* in, const int m, const int n, const int beam_size, cudaStream_t stream);
#endif

////////////////////////////////////////////////////////////////////////////
__global__ void getWenetOutLensKernel(int* out, const int* in, const int batch_size, const int max_seq_len)
{
    if (threadIdx.x < batch_size) {
        int cur      = in[threadIdx.x];
        int next_max = max_seq_len;

        next_max = (next_max - 1) / 2;
        cur      = (cur + 1) / 2;
        cur      = min(cur, next_max);

        next_max = (next_max - 1) / 2;
        cur      = (cur + 1) / 2;
        cur      = min(cur, next_max);

        out[threadIdx.x] = cur;
    }
}

void invokeGetWenetOutLens(int* out, const int* in, const int batch_size, const int max_seq_len, cudaStream_t stream)
{
    FT_CHECK(batch_size <= 1024);
    getWenetOutLensKernel<<<1, batch_size, 0, stream>>>(out, in, batch_size, max_seq_len);
    sync_check_cuda_error();
}
////////////////////////////////////////////////////////////////////////////
// __global__ void getPaddingOffsetKernelV2(size_t* valid_word_num,
//                                          int* tmp_mask_offset,
//                                          const int* sequence_length,
//                                          const int batch_size,
//                                          const int max_seq_len)
// {
//     // do cumulated sum
//     int total_seq_len = 0;
//     int cum_offset = 0;
//     int index = 0;
//     for (int i = 0; i < batch_size; i++) {
//         const int seq_len = sequence_length[i];
//         for (int j = 0; j < seq_len; j++) {
//             tmp_mask_offset[index] = cum_offset;
//             index++;
//         }
//         cum_offset += max_seq_len - seq_len;
//         total_seq_len += seq_len;
//     }
//     valid_word_num[0] = (size_t)total_seq_len;
// }

// void invokeGetPaddingOffset(size_t* d_token_num,
//                             int* tmp_mask_offset,
//                             const int* sequence_lengths,
//                             const int batch_size,
//                             const int max_seq_len,
//                             cudaStream_t stream)
// {
//     getPaddingOffsetKernelV2<<<1, 1, 0, stream>>>(
//         d_token_num, tmp_mask_offset, sequence_lengths, batch_size, max_seq_len);
//     sync_check_cuda_error();
// }

////////////////////////////////////////////////////////////////////////////
template<typename T, bool IS_CROSS>
__global__ void buildDecoderAttentionMaskKernel(T*         attention_mask,
                                                const int* sequence_lengths1,
                                                const int  max_seq_len1,
                                                const int* sequence_lengths2,
                                                const int  max_seq_len2)
{
    // sequence_lengths1: [batch_size]
    // sequence_lengths2: [batch_size]
    // attention_mask: [batch_size, 1, max_seq_len1, max_seq_len2]
    const int s1_id = blockIdx.x % max_seq_len1;
    const int b_id  = blockIdx.x / max_seq_len1;
    attention_mask += (b_id * max_seq_len1 + s1_id) * max_seq_len2;

    const int len1 = sequence_lengths1[b_id];
    int       len2 = 0;
    if (IS_CROSS)
        len2 = sequence_lengths2[b_id];

    for (int i = threadIdx.x; i < max_seq_len2; i += blockDim.x) {
        T val = (T)(0.0f);
        if (IS_CROSS) {
            if (i < len2)
                val = (T)(1.0f);
        }
        else {
            if (i < len1 && i <= s1_id)
                val = (T)(1.0f);
        }
        attention_mask[i] = val;
    }
}

template<typename T, bool IS_CROSS>
void invokeBuildDecoderAttentionMask(T*           attention_mask,
                                     const int*   sequence_lengths1,
                                     const int*   sequence_lengths2,
                                     const int    batch_size,
                                     const int    max_seq_len1,
                                     const int    max_seq_len2,
                                     cudaStream_t stream)
{
    buildDecoderAttentionMaskKernel<T, IS_CROSS>
        <<<batch_size * max_seq_len1, std::min(1024, max_seq_len2), 0, stream>>>(
            attention_mask, sequence_lengths1, max_seq_len1, sequence_lengths2, max_seq_len2);
}

template void invokeBuildDecoderAttentionMask<float, false>(float*       attention_mask,
                                                            const int*   sequence_lengths1,
                                                            const int*   sequence_lengths2,
                                                            const int    batch_size,
                                                            const int    max_seq_len1,
                                                            const int    max_seq_len2,
                                                            cudaStream_t stream);
template void invokeBuildDecoderAttentionMask<float, true>(float*       attention_mask,
                                                           const int*   sequence_lengths1,
                                                           const int*   sequence_lengths2,
                                                           const int    batch_size,
                                                           const int    max_seq_len1,
                                                           const int    max_seq_len2,
                                                           cudaStream_t stream);
template void invokeBuildDecoderAttentionMask<half, false>(half*        attention_mask,
                                                           const int*   sequence_lengths1,
                                                           const int*   sequence_lengths2,
                                                           const int    batch_size,
                                                           const int    max_seq_len1,
                                                           const int    max_seq_len2,
                                                           cudaStream_t stream);
template void invokeBuildDecoderAttentionMask<half, true>(half*        attention_mask,
                                                          const int*   sequence_lengths1,
                                                          const int*   sequence_lengths2,
                                                          const int    batch_size,
                                                          const int    max_seq_len1,
                                                          const int    max_seq_len2,
                                                          cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeBuildDecoderAttentionMask<__nv_bfloat16, false>(__nv_bfloat16* attention_mask,
                                                                    const int*     sequence_lengths1,
                                                                    const int*     sequence_lengths2,
                                                                    const int      batch_size,
                                                                    const int      max_seq_len1,
                                                                    const int      max_seq_len2,
                                                                    cudaStream_t   stream);
template void invokeBuildDecoderAttentionMask<__nv_bfloat16, true>(__nv_bfloat16* attention_mask,
                                                                   const int*     sequence_lengths1,
                                                                   const int*     sequence_lengths2,
                                                                   const int      batch_size,
                                                                   const int      max_seq_len1,
                                                                   const int      max_seq_len2,
                                                                   cudaStream_t   stream);

#endif
//////////////////////////////////////////////////////////////////////////////

template<typename T, int EPT>
__global__ void biasLogSoftmaxKernel(float*       log_probs,
                                     const T*     logits,
                                     const T*     bias,
                                     const int*   lengths,
                                     const size_t max_input_length,
                                     const size_t batch_size,
                                     const size_t vocab_size,
                                     const size_t vocab_size_padded)
{
    constexpr bool IS_FP16   = std::is_same<T, half>::value;
    const T        MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    int tidx = threadIdx.x;  // vocab dim
    int bidx = blockIdx.x;   // batch dim
    int step = blockIdx.y;   // step dim

    __shared__ float s_max_logit, s_sum_logit;

    bool is_valid = (bidx < batch_size) && (step < max_input_length);
    if (lengths != nullptr)
        is_valid = is_valid && (step < lengths[bidx]);
    if (is_valid) {
        // reposition logits to data for the current batch.
        logits += bidx * max_input_length * vocab_size_padded + step * vocab_size_padded;
        log_probs += bidx * max_input_length * vocab_size + step * vocab_size;

        // load and add bias
        T local_logit[EPT];
#pragma unroll
        for (int i = 0; i < EPT; ++i) {
            size_t cur_idx = tidx + i * blockDim.x;
            if (cur_idx < vocab_size) {
                local_logit[i] = logits[cur_idx];
                if (bias != nullptr)
                    local_logit[i] = (float)local_logit[i] + (float)bias[cur_idx];
            }
            // else
            //     local_logit[i] = -MAX_T_VAL;
        }
        // Find max(logits).
        float local_max = -MAX_T_VAL;
        float val       = -MAX_T_VAL;
#pragma unroll
        for (int i = 0; i < EPT; ++i) {
            size_t cur_idx = tidx + i * blockDim.x;
            if (cur_idx < vocab_size) {
                val       = static_cast<float>(local_logit[i]);
                local_max = fmax(local_max, val);
            }
        }

        float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
        if (tidx == 0) {
            s_max_logit = max_val;
        }
        __syncthreads();

        // Calculate the denominator: sum_i exp(logits[i])
        float local_sum_exp = 0.0f;
// val = 0.0f;
#pragma unroll
        for (int i = 0; i < EPT; ++i) {
            size_t cur_idx = tidx + i * blockDim.x;
            if (cur_idx < vocab_size) {
                val = __expf(static_cast<float>(local_logit[i]) - s_max_logit);
                local_sum_exp += val;
            }
        }

        float sum_exp = blockDim.x <= 32 ? warpReduceSum(local_sum_exp) : blockReduceSum<float>(local_sum_exp);
        if (tidx == 0) {
            s_sum_logit = sum_exp;
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < EPT; ++i) {
            size_t cur_idx = tidx + i * blockDim.x;
            if (cur_idx < vocab_size)
                log_probs[cur_idx] = static_cast<float>(local_logit[i]) - s_max_logit - __logf(s_sum_logit + 1e-9f);
        }
    }
}

template<typename T>
__global__ void biasLogSoftmaxKernel(float*       log_probs,
                                     const T*     logits,
                                     const T*     bias,
                                     const size_t max_input_length,
                                     const size_t batch_size,
                                     const size_t vocab_size,
                                     const size_t vocab_size_padded)
{
    constexpr bool IS_FP16   = std::is_same<T, half>::value;
    const T        MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    int bid    = blockIdx.x;
    int offset = bid * vocab_size_padded;

    float local_max = -MAX_T_VAL;
    float val       = -MAX_T_VAL;

    __shared__ float s_max_logit;
    __shared__ float s_sum_logit;

    for (int tid = threadIdx.x; tid < vocab_size_padded; tid += blockDim.x) {
        if (tid < vocab_size) {
            float bias_val          = (bias != nullptr) ? bias[tid] : (T)0.0f;
            log_probs[offset + tid] = (float)logits[offset + tid] + bias_val;
        }
        // else {
        //     log_probs[offset + tid] = -MAX_T_VAL;
        // }
        local_max = fmax(local_max, (float)log_probs[offset + tid]);
    }

    float max_val = blockReduceMax<float>(local_max);
    if (threadIdx.x == 0) {
        s_max_logit = max_val;
    }
    __syncthreads();

    float local_sum_exp = 0.0f;
    for (int tid = threadIdx.x; tid < vocab_size_padded; tid += blockDim.x) {
        val = __expf((float)log_probs[offset + tid] - s_max_logit);
        local_sum_exp += val;
    }

    float sum_exp = blockReduceSum<float>(local_sum_exp);
    if (threadIdx.x == 0) {
        s_sum_logit = sum_exp;
    }
    __syncthreads();

    for (int tid = threadIdx.x; tid < vocab_size_padded; tid += blockDim.x) {
        log_probs[offset + tid] = (float)log_probs[offset + tid] - s_max_logit - __logf(s_sum_logit + 1e-9f);
    }
}

template<typename T>
void invokeBiasLogSoftmax(float*       log_probs,
                          const T*     logits,
                          const T*     bias,
                          const int*   lengths,
                          const size_t max_input_length,
                          const size_t batch_size,
                          const size_t vocab_size,
                          const size_t vocab_size_padded,
                          bool         batch_first,
                          cudaStream_t stream)
{
    // FT_CHECK(vocab_size <= 768 * 6);
    dim3 block, grid;

    // Better perf at the cost of register pressure
    if (vocab_size > 768 * 5 && vocab_size <= 768 * 6) {
        block.x = 768;
        grid.x  = batch_size;
        grid.y  = max_input_length;
        biasLogSoftmaxKernel<T, 6><<<grid, block, 0, stream>>>(
            log_probs, logits, bias, lengths, max_input_length, batch_size, vocab_size, vocab_size_padded);
    }
    else {
        block.x = std::min((int)vocab_size, 1024);
        grid.x  = batch_size * max_input_length;
        /*n is the vocab_size, e.g., 30000, 7000.... vocab_size is usually very big. */
        biasLogSoftmaxKernel<T><<<grid, block, 0, stream>>>(
            log_probs, logits, bias, max_input_length, batch_size, vocab_size, vocab_size_padded);
    }
}

template void invokeBiasLogSoftmax<float>(float*       log_probs,
                                          const float* logits,
                                          const float* bias,
                                          const int*   lengths,
                                          const size_t max_input_length,
                                          const size_t batch_size,
                                          const size_t vocab_size,
                                          const size_t vocab_size_padded,
                                          bool         batch_first,
                                          cudaStream_t stream);
template void invokeBiasLogSoftmax<half>(float*       log_probs,
                                         const half*  logits,
                                         const half*  bias,
                                         const int*   lengths,
                                         const size_t max_input_length,
                                         const size_t batch_size,
                                         const size_t vocab_size,
                                         const size_t vocab_size_padded,
                                         bool         batch_first,
                                         cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeBiasLogSoftmax<__nv_bfloat16>(float*               log_probs,
                                                  const __nv_bfloat16* logits,
                                                  const __nv_bfloat16* bias,
                                                  const int*           lengths,
                                                  const size_t         max_input_length,
                                                  const size_t         batch_size,
                                                  const size_t         vocab_size,
                                                  const size_t         vocab_size_padded,
                                                  bool                 batch_first,
                                                  cudaStream_t         stream);
#endif

}  // namespace fastertransformer
