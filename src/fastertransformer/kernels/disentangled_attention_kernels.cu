/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/kernels/disentangled_attention_kernels.h"
#include "src/fastertransformer/utils/cuda_type_utils.cuh"
#include "src/fastertransformer/utils/cuda_utils.h"
#include <assert.h>

namespace fastertransformer {

#define kDISENTANGLED_VERSION 2
// Version 1: regular relative position index
// Version 2: log bucket relative position index
constexpr int32_t kDISENTANGLED_TILESIZE_V1  = 32;
constexpr int32_t kDISENTANGLED_BLOCKDIMY_V1 = 8;
constexpr int32_t kDISENTANGLED_TILESIZE_V2  = 64;
constexpr int32_t kDISENTANGLED_BLOCKDIMY_V2 = 4;

template<typename T>
__global__ void addQKBiasTransposeRepeat(T* q_out,
                                         T* k_out,
                                         const T* __restrict q_in,
                                         const T* __restrict bias_q,
                                         const T* __restrict k_in,
                                         const T* __restrict bias_k,
                                         const int batch_size,
                                         const int attention_span,
                                         const int head_num,
                                         const int size_per_head)
{
    const int n           = head_num * size_per_head;
    const int batch_id    = blockIdx.x;
    const int position_id = blockIdx.y;

    for (int col_id = threadIdx.x; col_id < n; col_id += blockDim.x) {
        const int head_id   = col_id / size_per_head;
        const int size_id   = col_id % size_per_head;
        const int target_id = batch_id * (head_num * attention_span * size_per_head)
                              + head_id * (attention_span * size_per_head) + position_id * size_per_head + size_id;
        const int src_id = position_id * n + col_id;

        T q              = add(ldg(&q_in[src_id]), ldg(&bias_q[col_id]));
        q_out[target_id] = q;

        T k              = add(ldg(&k_in[src_id]), ldg(&bias_k[col_id]));
        k_out[target_id] = k;
    }
}

template<typename T>
__global__ void QKTransposeRepeat(T* q_out,
                                  T* k_out,
                                  const T* __restrict q_in,
                                  const T* __restrict k_in,
                                  const int batch_size,
                                  const int attention_span,
                                  const int head_num,
                                  const int size_per_head)
{
    const int n           = head_num * size_per_head;
    const int batch_id    = blockIdx.x;
    const int position_id = blockIdx.y;

    for (int col_id = threadIdx.x; col_id < n; col_id += blockDim.x) {
        const int head_id   = col_id / size_per_head;
        const int size_id   = col_id % size_per_head;
        const int target_id = batch_id * (head_num * attention_span * size_per_head)
                              + head_id * attention_span * size_per_head + position_id * size_per_head + size_id;
        const int src_id = position_id * n + col_id;

        q_out[target_id] = ldg(&q_in[src_id]);
        k_out[target_id] = ldg(&k_in[src_id]);
    }
}

template<typename T>
void invokeAddQKBiasTransposeRepeat(T*           q_buf,
                                    T*           k_buf,
                                    T*           Q,
                                    const T*     bias_Q,
                                    T*           K,
                                    const T*     bias_K,
                                    const int    batch_size,
                                    const int    attention_span,
                                    const int    head_num,
                                    const int    size_per_head,
                                    cudaStream_t stream)
{
    const int k = head_num * size_per_head;
    dim3      grid(batch_size, attention_span);
    bool      is_add_bias = bias_Q != nullptr;
    if (sizeof(T) == 4 || k % 2 != 0) {
        dim3 block(min(k, 512));
        if (is_add_bias) {
            addQKBiasTransposeRepeat<T><<<grid, block, 0, stream>>>(
                q_buf, k_buf, Q, bias_Q, K, bias_K, batch_size, attention_span, head_num, size_per_head);
        }
        else {
            QKTransposeRepeat<T>
                <<<grid, block, 0, stream>>>(q_buf, k_buf, Q, K, batch_size, attention_span, head_num, size_per_head);
        }
        sync_check_cuda_error();
    }
    else {
        using T2 = typename TypeConverter<T>::Type;  // fp16 to half2, bf16 to bf162
        dim3 block(min(k / 2, 512));
        if (is_add_bias) {
            addQKBiasTransposeRepeat<T2><<<grid, block, 0, stream>>>((T2*)q_buf,
                                                                     (T2*)k_buf,
                                                                     (const T2*)Q,
                                                                     (const T2*)bias_Q,
                                                                     (const T2*)K,
                                                                     (const T2*)bias_K,
                                                                     batch_size,
                                                                     attention_span,
                                                                     head_num,
                                                                     size_per_head / 2);
        }
        else {
            QKTransposeRepeat<T2><<<grid, block, 0, stream>>>((T2*)q_buf,
                                                              (T2*)k_buf,
                                                              (const T2*)Q,
                                                              (const T2*)K,
                                                              batch_size,
                                                              attention_span,
                                                              head_num,
                                                              size_per_head / 2);
        }
        sync_check_cuda_error();
    }
}

#define INSTANTIATEADDQKBIASTRANSPOSEREPEAT(T)                                                                         \
    template void invokeAddQKBiasTransposeRepeat(T*           q_buf,                                                   \
                                                 T*           k_buf,                                                   \
                                                 T*           Q,                                                       \
                                                 const T*     bias_Q,                                                  \
                                                 T*           K,                                                       \
                                                 const T*     bias_K,                                                  \
                                                 const int    batch_size,                                              \
                                                 const int    attention_span,                                          \
                                                 const int    head_num,                                                \
                                                 const int    size_per_head,                                           \
                                                 cudaStream_t stream)
INSTANTIATEADDQKBIASTRANSPOSEREPEAT(float);
INSTANTIATEADDQKBIASTRANSPOSEREPEAT(half);
#ifdef ENABLE_BF16
INSTANTIATEADDQKBIASTRANSPOSEREPEAT(__nv_bfloat16);
#endif
#undef INSTANTIATEADDQKBIASTRANSPOSEREPEAT

// template specialization for double/float
template<typename TDataType,
         typename std::enable_if<std::is_same<std::decay_t<TDataType>, double>::value
                                     || std::is_same<std::decay_t<TDataType>, float>::value,
                                 TDataType>::type* dummy = nullptr>
__forceinline__ __device__ void
compute_attention(TDataType& res, const TDataType& res0, const TDataType& res1, const TDataType& res2)
{
    res = res0 + res1 + res2;
}

// template specialization for half
template<typename TDataType,
         typename std::enable_if<std::is_same<std::decay_t<TDataType>, __half>::value
                                     || std::is_same<std::decay_t<TDataType>, half>::value,
                                 TDataType>::type* dummy = nullptr>
__forceinline__ __device__ void
compute_attention(TDataType& res, const TDataType& res0, const TDataType& res1, const TDataType& res2)
{
#if __CUDA_ARCH__ >= 530
    // __hmul only supported >= sm_53
    res = __hadd(res0, __hadd(res1, res2));
#else
    // for < sm_53, workaround/fallback is convert to float and downconvert
    res = __float2half(__half2float(res0) + __half2float(res1) + __half2float(res2));
#endif
}

#ifdef ENABLE_BF16
template<typename TDataType,
         typename std::enable_if<std::is_same<std::decay_t<TDataType>, nv_bfloat16>::value, TDataType>::type* dummy =
             nullptr>
__forceinline__ __device__ void
compute_attention(TDataType& res, const TDataType& res0, const TDataType& res1, const TDataType& res2)
{
    res = res0 + res1 + res2;
}
#endif

// template specialization for int8
template<typename TDataType,
         typename std::enable_if<std::is_same<std::decay_t<TDataType>, int8_t>::value
                                     || std::is_same<std::decay_t<TDataType>, uint8_t>::value,
                                 TDataType>::type* dummy = nullptr>
__forceinline__ __device__ void
compute_attention(TDataType& res, const TDataType& res0, const TDataType& res1, const TDataType& res2)
{
    res = res0 + res1 + res2;
}

/**
 * Fused kernel for Disentangled Attention design (first proposed in Microsoft DeBERTa). Implementation refactored from
 * previous TensorRT plugin GatherAddGatherTransposeAddMul_fused kernel implementation by (1) removing the scaling
 * factor because each attention matrix has already been applied scaling in cuBLAS (2) add BF16 support.
 *
 * @tparam TDataType type of the input data
 * @tparam tTileSize dimension of the shared memory tile (square) and also the BlockDimX. Need for compile-time shared
 * memory size
 * @tparam tBlockDimY 2D thread block is (tTileSize, tBlockDimY)
 * @param result attention result
 * @param data0 content-to-content ("c2c") attention QcKc^T
 * @param data1 content-to-position ("c2p") attention QcKr^T
 * @param data2 position-to-content ("p2c") attention KcQr^T
 * @param batch_dim flattened batch dimension is (batch_size * num_heads)
 * @param seq_dim sequence dimension [seq_len]
 * @param span relative distance hyper-parameter, k, in Disentangled attention
 */
template<typename TDataType, int32_t tTileSize, int32_t tBlockDimY>
__global__ void disentangled_attention_kernel(TDataType*       result,
                                              TDataType*       data0,
                                              TDataType const* data1,
                                              TDataType const* data2,
                                              int32_t          batch_dim,
                                              int32_t          seq_dim,
                                              int32_t          span)
{
    // Tile size should be a multiple of number of block rows
    assert(blockDim.y * (blockDim.x / blockDim.y) == blockDim.x);

    // map block to the output (result)
    int32_t   i;
    int32_t   j;
    int32_t   k;
    int32_t   ty;
    int32_t   c;
    int32_t   pos_dim = span * 2;
    TDataType res0;
    TDataType res1;
    TDataType res2;
    TDataType res;

#if kDISENTANGLED_VERSION == 2
    int32_t bucket;
    int32_t mid = span / 2;
    int32_t index;

    // tmp values are precomputed for re-use; must be at least float to ensure accuracy
    float tmp1 = logf(mid);

    // Multiply by (1 - epsilon) to ensure that taking the ceil of approximately an integer
    // results in that integer when computing the bucket later on.
    // This corrects for the mathematical imprecision from using float.
    constexpr float kEPSILON = 1e-7;
    float           tmp      = (mid - 1) / (logf(pos_dim - 1) - tmp1) * (1 - kEPSILON);
#endif

    __shared__ TDataType T[tTileSize][tTileSize + 1];  // +1 to avoid bank conflict

    // (i,j,k) location of data2 (transposed)
    i = blockIdx.z;
    j = blockIdx.x * tTileSize + threadIdx.y;
    k = blockIdx.y * tTileSize + threadIdx.x;

    // (j+ty, k) is the location in the index matrix (implicit), where the element values of the index matrix are
    // determinisitic based on its location. Index matrix dimension is [batch_size*num_heads, seq_len, seq_len], add
    // boundary check accordingly c2p & p2c matrix dimension is [batch_size*num_heads, seq_len, 2*k], within-boundary is
    // guaranteed by the index function design, but can still add check due to floating point precision

// gather data2
#pragma unroll
    for (c = 0, ty = 0; c < tTileSize / tBlockDimY && (j + ty) < seq_dim && k < seq_dim; c++, ty += tBlockDimY) {
#if kDISENTANGLED_VERSION == 1
        // relative position -- version 1
        if (k - (j + ty) >= span) {
            res2 = data2[i * seq_dim * pos_dim + (j + ty) * pos_dim + pos_dim - 1];
        }
        else if (k - (j + ty) <= -span) {
            res2 = data2[i * seq_dim * pos_dim + (j + ty) * pos_dim + 0];
        }
        else {
            res2 = data2[i * seq_dim * pos_dim + (j + ty) * pos_dim + k - (j + ty) + span];  // compute index on the fly
        }
        T[ty + threadIdx.y][threadIdx.x] = res2;
#elif kDISENTANGLED_VERSION == 2
        // relative position w/ log bucket -- version 2
        if (k - (j + ty) >= -mid && k - (j + ty) <= mid) {
            // preserved region, (i - j) + span
            bucket = k - (j + ty);
        }
        else {
            // log bucket region, bucket(i,j) + span
            bucket = ceilf((logf(fabsf(k - (j + ty))) - tmp1) * tmp) + mid;
            bucket = k - (j + ty) < 0 ? -bucket : bucket;
        }
        // clamp [0,2k]. Although this is guaranteed by equation, but numerically the floating precision can still break
        // boundary
        index                            = bucket + span;
        index                            = min(max(0, index), pos_dim - 1);
        res2                             = data2[i * seq_dim * pos_dim + (j + ty) * pos_dim + index];
        T[ty + threadIdx.y][threadIdx.x] = res2;
#endif
    }

    __syncthreads();

    // (i,j,k) location of data1 (non-transposed) and output. i unchanged
    j = blockIdx.y * tTileSize + threadIdx.y;
    k = blockIdx.x * tTileSize + threadIdx.x;

// read data0 + gather data1 + add all + write
#pragma unroll
    for (c = 0, ty = 0; c < tTileSize / tBlockDimY && (j + ty) < seq_dim && k < seq_dim; c++, ty += tBlockDimY) {
#if kDISENTANGLED_VERSION == 1
        // relative position -- version 1
        // for non-transposed matrix 1, just fetch element at the transposed location & add to the result)
        if (j + ty - k <= -span) {
            res1 = data1[i * seq_dim * pos_dim + (j + ty) * pos_dim + 0];
        }
        else if (j + ty - k >= span) {
            res1 = data1[i * seq_dim * pos_dim + (j + ty) * pos_dim + pos_dim - 1];
        }
        else {
            res1 = data1[i * seq_dim * pos_dim + (j + ty) * pos_dim + j + ty - k + span];  // compute index on the fly
        }
#elif kDISENTANGLED_VERSION == 2
        // relative position w/ log bucket -- version 2
        if (j + ty - k >= -mid && j + ty - k <= mid) {
            // preserved region, (i - j) + span
            bucket = j + ty - k;
        }
        else {
            // log bucket region, bucket(i,j) + span
            bucket = ceilf((logf(fabsf((j + ty) - k)) - tmp1) * tmp) + mid;
            bucket = (j + ty) - k < 0 ? -bucket : bucket;
        }
        // clamp [0,2k]. Although this is guaranteed by equation, but numerically the floating precision can still break
        // boundary
        index = bucket + span;
        index = min(max(0, index), pos_dim - 1);
        res1  = data1[i * seq_dim * pos_dim + (j + ty) * pos_dim + index];
#endif

        // for non-tranposed matrix 0, same as matrix 1
        res0 = data0[i * seq_dim * seq_dim + (j + ty) * seq_dim + k];

        // (res0 + res1 + res2)
#if __cplusplus >= 201703L
        // C++ 17 has more convenient `if constexpr` for conditional implementation at compile time; before C++ 17,
        // switch to template specialization
        if constexpr (std::is_same<TDataType, double>::value || std::is_same<TDataType, float>::value) {
            // double, float32
            res = res0 + res1 + T[threadIdx.x][ty + threadIdx.y];
        }
        else if constexpr (std::is_same<TDataType, __half>::value || std::is_same<TDataType, half>::value) {
            // fp16
#if __CUDA_ARCH__ >= 530
            // half ops only supported >= sm_53
            res = __hadd(res0, __hadd(res1, T[threadIdx.x][ty + threadIdx.y]));
#else
            // for < sm_53, workaround/fallback is convert to float and downconvert
            res =
                __float2half(__half2float(res0) + __half2float(res1) + __half2float(T[threadIdx.x][ty + threadIdx.y]));
#endif
        }
#ifdef ENABLE_BF16
        else if constexpr (std::is_same<TDataType, __nv_bfloat16>::value) {
            // bf16
            res = __hadd(res0, __hadd(res1, T[threadIdx.x][ty + threadIdx.y]));
        }
#endif
        else if constexpr (std::is_same<TDataType, int8_t>::value || std::is_same<TDataType, uint8_t>::value) {
            // int8_t
            res = res0 + res1 + T[threadIdx.x][ty + threadIdx.y];
        }
#else
        // before C++ 17, use template specialization
        compute_attention<TDataType>(res, res0, res1, T[threadIdx.x][ty + threadIdx.y]);
#endif
        // write
        result[i * seq_dim * seq_dim + (j + ty) * seq_dim + k] = res;
    }
}

#define INSTANTIATEDISENTANGLEDKERNEL(T)                                                                               \
    template __global__ void disentangled_attention_kernel<T, kDISENTANGLED_TILESIZE_V1, kDISENTANGLED_BLOCKDIMY_V1>(  \
        T*, T*, T const*, T const*, int32_t, int32_t, int32_t);                                                        \
    template __global__ void disentangled_attention_kernel<T, kDISENTANGLED_TILESIZE_V2, kDISENTANGLED_BLOCKDIMY_V2>(  \
        T*, T*, T const*, T const*, int32_t, int32_t, int32_t);
INSTANTIATEDISENTANGLEDKERNEL(float)
INSTANTIATEDISENTANGLEDKERNEL(half)
INSTANTIATEDISENTANGLEDKERNEL(int8_t)
#ifdef ENABLE_BF16
INSTANTIATEDISENTANGLEDKERNEL(__nv_bfloat16)
#endif
#undef INSTANTIATEDISENTANGLEDKERNEL

template<typename T>
void invokeDisentangledAttention(
    T* result, T* c2c, T* c2p, T* p2c, const int batch_dim, const int seq_dim, const int span, cudaStream_t stream)
{
#if kDISENTANGLED_VERSION == 1
    dim3 block_optimized(kDISENTANGLED_TILESIZE_V1, kDISENTANGLED_BLOCKDIMY_V1);
    dim3 grid_optimized(
        (seq_dim - 1) / kDISENTANGLED_TILESIZE_V1 + 1, (seq_dim - 1) / kDISENTANGLED_TILESIZE_V1 + 1, batch_dim);
    disentangled_attention_kernel<T, kDISENTANGLED_TILESIZE_V1, kDISENTANGLED_BLOCKDIMY_V1>
        <<<grid_optimized, block_optimized, 0, stream>>>(result, c2c, c2p, p2c, batch_dim, seq_dim, span);

#elif kDISENTANGLED_VERSION == 2
    dim3 block_optimized(kDISENTANGLED_TILESIZE_V2, kDISENTANGLED_BLOCKDIMY_V2);
    dim3 grid_optimized(
        (seq_dim - 1) / kDISENTANGLED_TILESIZE_V2 + 1, (seq_dim - 1) / kDISENTANGLED_TILESIZE_V2 + 1, batch_dim);
    disentangled_attention_kernel<T, kDISENTANGLED_TILESIZE_V2, kDISENTANGLED_BLOCKDIMY_V2>
        <<<grid_optimized, block_optimized, 0, stream>>>(result, c2c, c2p, p2c, batch_dim, seq_dim, span);

#endif
}

#define INSTANTIATEDISENTANGLEDATTENTION(T)                                                                            \
    template void invokeDisentangledAttention(T*           result,                                                     \
                                              T*           c2c,                                                        \
                                              T*           c2p,                                                        \
                                              T*           p2c,                                                        \
                                              const int    batch_dim,                                                  \
                                              const int    seq_dim,                                                    \
                                              const int    span,                                                       \
                                              cudaStream_t stream);
INSTANTIATEDISENTANGLEDATTENTION(float)
INSTANTIATEDISENTANGLEDATTENTION(half)
INSTANTIATEDISENTANGLEDATTENTION(int8_t)
#ifdef ENABLE_BF16
INSTANTIATEDISENTANGLEDATTENTION(__nv_bfloat16)
#endif
#undef INSTANTIATEDISENTANGLEDATTENTION

}  // namespace fastertransformer
