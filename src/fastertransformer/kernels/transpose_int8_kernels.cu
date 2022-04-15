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

#include "src/fastertransformer/kernels/transpose_int8_kernels.h"

namespace fastertransformer {

// src is the result of batch MM, whose size is batch_size*head_num*(seq_len, size_per_head), CUBLASLT_ORDER_COL32
// dst is of m = batch_size*seq_len, k(n) = head_num*size_per_head, CUBLASLT_ORDER_COL32
// grid(seq_len, batch_size)
// block(size_per_head/4, head_num)
// assume size_per_head is multiples of 32
__global__ void transpose_COL32_kernel(char4* dst,
                                       const int4* src,
                                       const int batch_size,
                                       const int seq_len,
                                       const int head_num,
                                       const int size_per_head,
                                       const float* v_buf_addBias_deQFactor,
                                       const float* qk_afterSM_deQFactor,
                                       const float* out_scale_ptr,
                                       const int batch_size_x_seq_len,
                                       const int seq_len_x_size_per_head)
{
    const float scale = __ldg(v_buf_addBias_deQFactor) * __ldg(qk_afterSM_deQFactor) * __ldg(out_scale_ptr);
    int threadIdx4 = threadIdx.x << 2;
    int batch_id = blockIdx.y;
    int seq_id = blockIdx.x;
    int head_id = threadIdx.y;
    // get the (row, col) output layout of m*k
    // m = batch_size*seq_len
    // k = head_num*size_per_head
    int mk_row = batch_id * seq_len + seq_id;
    int mk_col = head_id * size_per_head + threadIdx4;
    // get the (row, col) layout of COL32; leading dimension = 32*m = 32*batch_size*seq_len
    int COL32_row = (mk_row << 5) + (mk_col & 31);
    // int COL32_col = mk_col >> 5;
    int outIdx = ((mk_col & 0xffffffe0) * batch_size_x_seq_len + COL32_row) >> 2;

    // get the (row, col) input layout of m'*k'
    // m' = seq_len
    // k' = size_per_head
    mk_row = seq_id;
    mk_col = threadIdx4;
    // get the (row, col) layout of COL32; leading dimension = 32*m' = 32*seq_len
    COL32_row = (mk_row << 5) + (mk_col & 31);
    // COL32_col = mk_col >> 5;

    int inIdx =
        ((batch_id * head_num + head_id) * seq_len_x_size_per_head + (mk_col & 0xffffffe0) * seq_len + COL32_row) >> 2;
    char4 tmp;

    int4 srcTmp4 = __ldg(src + inIdx);
    tmp.x = float_to_int8_rn(srcTmp4.x * scale);
    tmp.y = float_to_int8_rn(srcTmp4.y * scale);
    tmp.z = float_to_int8_rn(srcTmp4.z * scale);
    tmp.w = float_to_int8_rn(srcTmp4.w * scale);
    dst[outIdx] = tmp;
}

void invokeTransposeCOL32(int8_t* dst,
                          const int* src,
                          const int batch_size,
                          const int seq_len,
                          const int head_num,
                          const int size_per_head,
                          const float* v_buf_addBias_deQFactor,
                          const float* qk_afterSM_deQFactor,
                          const float* out_scale_ptr,
                          cudaStream_t stream)
{
    assert(size_per_head % 32 == 0);
    transpose_COL32_kernel<<<dim3(seq_len, batch_size), dim3(size_per_head / 4, head_num), 0, stream>>>(
        (char4*)dst,
        (const int4*)src,
        batch_size,
        seq_len,
        head_num,
        size_per_head,
        v_buf_addBias_deQFactor,
        qk_afterSM_deQFactor,
        out_scale_ptr,
        batch_size * seq_len,
        seq_len * size_per_head);
}

// src is the result of batch MM, whose size is batch_size*head_num*(seq_len, size_per_head), CUBLASLT_ORDER_COL32
// dst is of m = batch_size*seq_len, k(n) = head_num*size_per_head, CUBLASLT_ORDER_COL32
// grid(seq_len, batch_size)
// block(size_per_head/4, head_num)
// assume size_per_head is multiples of 32
__global__ void transpose_COL32_kernel(int8_t* dst,
                                       const int8_t* src,
                                       const int batch_size,
                                       const int seq_len,
                                       const int head_num,
                                       const int size_per_head,
                                       const float* bmm2_deQFactor,
                                       const float* out_scale_ptr,
                                       const int batch_size_x_seq_len,
                                       const int seq_len_x_size_per_head)
{
    int threadIdx4 = threadIdx.x << 2;
    int batch_id = blockIdx.y;
    int seq_id = blockIdx.x;
    int head_id = threadIdx.y;
    // get the (row, col) output layout of m*k
    // m = batch_size*seq_len
    // k = head_num*size_per_head
    int mk_row = batch_id * seq_len + seq_id;
    int mk_col = head_id * size_per_head + threadIdx4;
    // get the (row, col) layout of COL32; leading dimension = 32*m = 32*batch_size*seq_len
    int COL32_row = (mk_row << 5) + (mk_col & 31);
    int COL32_col = mk_col >> 5;
    int outIdx = ((COL32_col << 5) * batch_size_x_seq_len + COL32_row) >> 2;

    // get the (row, col) input layout of m'*k'
    // m' = seq_len
    // k' = size_per_head
    mk_row = seq_id;
    mk_col = threadIdx4;
    // get the (row, col) layout of COL32; leading dimension = 32*m' = 32*seq_len
    COL32_row = (mk_row << 5) + (mk_col & 31);
    COL32_col = mk_col >> 5;

    int inIdx =
        ((batch_id * head_num + head_id) * seq_len_x_size_per_head + (COL32_col << 5) * seq_len + COL32_row) >> 2;
    const char4* src_ptr4 = (const char4*)src;
    char4* dst_ptr4 = (char4*)dst;
    dst_ptr4[outIdx] = __ldg(src_ptr4 + inIdx);
}

void invokeTransposeCOL32(int8_t* dst,
                          const int8_t* src,
                          const int batch_size,
                          const int seq_len,
                          const int head_num,
                          const int size_per_head,
                          const float* bmm2_deQFactor,
                          const float* out_scale_ptr,
                          cudaStream_t stream)
{
    assert(size_per_head % 32 == 0);
    transpose_COL32_kernel<<<dim3(seq_len, batch_size), dim3(size_per_head / 4, head_num), 0, stream>>>(
        dst,
        src,
        batch_size,
        seq_len,
        head_num,
        size_per_head,
        bmm2_deQFactor,
        out_scale_ptr,
        batch_size * seq_len,
        seq_len * size_per_head);
}

// src is the result of batch MM, whose size is batch_size*head_num*(seq_len, size_per_head), CUBLASLT_ORDER_COL32
// dst is of m = valid_word_num, k(n) = head_num*size_per_head, CUBLASLT_ORDER_COL32
// grid(seq_len, batch_size)
// block(size_per_head/4, head_num)
// assume size_per_head is multiples of 32
__global__ void transpose_COL32_rebuild_padding_kernel(int8_t* dst,
                                                       const int32_t* src,
                                                       const int* sequence_id_map,
                                                       const int valid_word_num,
                                                       const int batch_size,
                                                       const int seq_len,
                                                       const int head_num,
                                                       const int size_per_head,
                                                       const float* v_buf_addBias_deQFactor,
                                                       const float* qk_afterSM_deQFactor,
                                                       const float* out_scale_ptr,
                                                       const int seq_len_x_size_per_head)
{
    const float scale = __ldg(v_buf_addBias_deQFactor) * __ldg(qk_afterSM_deQFactor) * __ldg(out_scale_ptr);
    int threadIdx4 = threadIdx.x << 2;
    int batch_id = blockIdx.y;
    int seq_id = blockIdx.x;
    int head_id = threadIdx.y;
    // get the (row, col) output layout of m*k
    // m = valid_word_num
    // k = head_num*size_per_head
    int mk_row = __ldg(sequence_id_map + batch_id * seq_len + seq_id);
    if (mk_row >= 0) {
        int mk_col = head_id * size_per_head + threadIdx4;
        // get the (row, col) layout of COL32; leading dimension = 32*m = 32*valid_word_num
        int COL32_row = (mk_row << 5) + (mk_col & 31);
        int COL32_col = mk_col >> 5;
        int outIdx = ((COL32_col << 5) * valid_word_num + COL32_row) >> 2;

        // get the (row, col) input layout of m'*k'
        // m' = seq_len
        // k' = size_per_head
        mk_row = seq_id;
        mk_col = threadIdx4;
        // get the (row, col) layout of COL32; leading dimension = 32*m' = 32*seq_len
        COL32_row = (mk_row << 5) + (mk_col & 31);
        COL32_col = mk_col >> 5;

        int inIdx = (batch_id * head_num + head_id) * seq_len_x_size_per_head + (COL32_col << 5) * seq_len + COL32_row;
        char4 tmp;
        tmp.x = float_to_int8_rn(__ldg(src + inIdx) * scale);
        tmp.y = float_to_int8_rn(__ldg(src + inIdx + 1) * scale);
        tmp.z = float_to_int8_rn(__ldg(src + inIdx + 2) * scale);
        tmp.w = float_to_int8_rn(__ldg(src + inIdx + 3) * scale);
        char4* dst_ptr4 = (char4*)dst;
        dst_ptr4[outIdx] = tmp;
    }
}

void invokeTransposeCOL32RebuildPadding(int8_t* dst,
                                        const int* src,
                                        const int* sequence_id_map,
                                        const int valid_word_num,
                                        const int batch_size,
                                        const int seq_len,
                                        const int head_num,
                                        const int size_per_head,
                                        const float* v_buf_addBias_deQFactor,
                                        const float* qk_afterSM_deQFactor,
                                        const float* out_scale_ptr,
                                        cudaStream_t stream)
{
    assert(size_per_head % 32 == 0);
    transpose_COL32_rebuild_padding_kernel<<<dim3(seq_len, batch_size), dim3(size_per_head / 4, head_num), 0, stream>>>(
        dst,
        src,
        sequence_id_map,
        valid_word_num,
        batch_size,
        seq_len,
        head_num,
        size_per_head,
        v_buf_addBias_deQFactor,
        qk_afterSM_deQFactor,
        out_scale_ptr,
        seq_len * size_per_head);
}

// src is the result of batch MM, whose size is batch_size*head_num*(seq_len, size_per_head), CUBLASLT_ORDER_COL32
// dst is of m = valid_word_num, k(n) = head_num*size_per_head, CUBLASLT_ORDER_COL32
// grid(seq_len, batch_size)
// block(size_per_head/4, head_num)
// assume size_per_head is multiples of 32
__global__ void transpose_COL32_rebuild_padding_kernel(int8_t* dst,
                                                       const int8_t* src,
                                                       const int* sequence_id_map,
                                                       const int valid_word_num,
                                                       const int batch_size,
                                                       const int seq_len,
                                                       const int head_num,
                                                       const int size_per_head,
                                                       const float* bmm2_deQFactor,
                                                       const float* out_scale_ptr,
                                                       const int seq_len_x_size_per_head)
{
    int threadIdx4 = threadIdx.x << 2;
    int batch_id = blockIdx.y;
    int seq_id = blockIdx.x;
    int head_id = threadIdx.y;
    // get the (row, col) output layout of m*k
    // m = valid_word_num
    // k = head_num*size_per_head
    int mk_row = __ldg(sequence_id_map + batch_id * seq_len + seq_id);
    if (mk_row >= 0) {
        int mk_col = head_id * size_per_head + threadIdx4;
        // get the (row, col) layout of COL32; leading dimension = 32*m = 32*valid_word_num
        int COL32_row = (mk_row << 5) + (mk_col & 31);
        int COL32_col = mk_col >> 5;
        int outIdx = ((COL32_col << 5) * valid_word_num + COL32_row) >> 2;

        // get the (row, col) input layout of m'*k'
        // m' = seq_len
        // k' = size_per_head
        mk_row = seq_id;
        mk_col = threadIdx4;
        // get the (row, col) layout of COL32; leading dimension = 32*m' = 32*seq_len
        COL32_row = (mk_row << 5) + (mk_col & 31);
        COL32_col = mk_col >> 5;

        int inIdx =
            ((batch_id * head_num + head_id) * seq_len_x_size_per_head + (COL32_col << 5) * seq_len + COL32_row) >> 2;

        const char4* src_ptr4 = (const char4*)src;

        char4* dst_ptr4 = (char4*)dst;
        dst_ptr4[outIdx] = __ldg(src_ptr4 + inIdx);
    }
}

void invokeTransposeCOL32RebuildPadding(int8_t* dst,
                                        const int8_t* src,
                                        const int* sequence_id_map,
                                        const int valid_word_num,
                                        const int batch_size,
                                        const int seq_len,
                                        const int head_num,
                                        const int size_per_head,
                                        const float* bmm2_deQFactor,
                                        const float* out_scale_ptr,
                                        cudaStream_t stream)
{
    assert(size_per_head % 32 == 0);
    transpose_COL32_rebuild_padding_kernel<<<dim3(seq_len, batch_size), dim3(size_per_head / 4, head_num), 0, stream>>>(
        dst,
        src,
        sequence_id_map,
        valid_word_num,
        batch_size,
        seq_len,
        head_num,
        size_per_head,
        bmm2_deQFactor,
        out_scale_ptr,
        seq_len * size_per_head);
}

// src is the result of batch MM, whose size is batch_size*head_num*(seq_len, size_per_head), CUBLASLT_ORDER_COL32
// dst is of m = batch_size*seq_len, k(n) = head_num*size_per_head, row major
// grid(seq_len, batch_size)
// block(size_per_head/4, head_num)
// assume size_per_head is multiples of 32
__global__ void transpose_COL32_ROW_kernel(int8_t* dst,
                                           const int8_t* src,
                                           const int batch_size,
                                           const int seq_len,
                                           const int head_num,
                                           const int size_per_head,
                                           const float* bmm2_deQFactor,
                                           const float* out_scale_ptr,
                                           const int head_num_x_size_per_head,
                                           const int seq_len_x_size_per_head)
{
    int threadIdx4 = threadIdx.x << 2;
    int batch_id = blockIdx.y;
    int seq_id = blockIdx.x;
    int head_id = threadIdx.y;
    // get the (row, col) output layout of m*k
    // m = batch_size*seq_len
    // k = head_num*size_per_head
    int mk_row = batch_id * seq_len + seq_id;
    int mk_col = head_id * size_per_head + threadIdx4;
    int outIdx = (mk_row * head_num_x_size_per_head + mk_col) >> 2;

    // get the (row, col) input layout of m'*k'
    // m' = seq_len
    // k' = size_per_head
    mk_row = seq_id;
    mk_col = threadIdx4;
    // get the (row, col) layout of COL32; leading dimension = 32*m' = 32*seq_len
    int COL32_row = (mk_row << 5) + (mk_col & 31);
    int COL32_col = mk_col >> 5;

    int inIdx =
        ((batch_id * head_num + head_id) * seq_len_x_size_per_head + (COL32_col << 5) * seq_len + COL32_row) >> 2;
    const char4* src_ptr4 = (const char4*)src;
    char4* dst_ptr4 = (char4*)dst;
    dst_ptr4[outIdx] = __ldg(src_ptr4 + inIdx);
}

void invokeTransposeCOL32ToRow(int8_t* dst,
                               const int8_t* src,
                               const int batch_size,
                               const int seq_len,
                               const int head_num,
                               const int size_per_head,
                               const float* bmm2_deQFactor,
                               const float* out_scale_ptr,
                               cudaStream_t stream)
{
    assert(size_per_head % 32 == 0);
    transpose_COL32_ROW_kernel<<<dim3(seq_len, batch_size), dim3(size_per_head / 4, head_num), 0, stream>>>(
        dst,
        src,
        batch_size,
        seq_len,
        head_num,
        size_per_head,
        bmm2_deQFactor,
        out_scale_ptr,
        head_num * size_per_head,
        seq_len * size_per_head);
}

// src is the result of batch MM, whose size is batch_size*head_num*(seq_len, size_per_head), CUBLASLT_ORDER_COL32
// dst is of m = valid_word_num, k(n) = head_num*size_per_head, row major
// grid(seq_len, batch_size)
// block(size_per_head/4, head_num)
// assume size_per_head is multiples of 32
__global__ void transpose_COL32_ROW_rebuild_padding_kernel(int8_t* dst,
                                                           const int8_t* src,
                                                           const int* sequence_id_map,
                                                           const int valid_word_num,
                                                           const int batch_size,
                                                           const int seq_len,
                                                           const int head_num,
                                                           const int size_per_head,
                                                           const float* bmm2_deQFactor,
                                                           const float* out_scale_ptr,
                                                           const int seq_len_x_size_per_head,
                                                           const int head_num_x_size_per_head)
{
    int threadIdx4 = threadIdx.x << 2;
    int batch_id = blockIdx.y;
    int seq_id = blockIdx.x;
    int head_id = threadIdx.y;
    // get the (row, col) output layout of m*k
    // m = valid_word_num
    // k = head_num*size_per_head
    int mk_row = __ldg(sequence_id_map + batch_id * seq_len + seq_id);
    if (mk_row >= 0) {
        int mk_col = head_id * size_per_head + threadIdx4;
        int outIdx = (mk_row * head_num_x_size_per_head + mk_col) >> 2;

        // get the (row, col) input layout of m'*k'
        // m' = seq_len
        // k' = size_per_head
        mk_row = seq_id;
        mk_col = threadIdx4;
        // get the (row, col) layout of COL32; leading dimension = 32*m' = 32*seq_len
        int COL32_row = (mk_row << 5) + (mk_col & 31);
        int COL32_col = mk_col >> 5;

        int inIdx =
            ((batch_id * head_num + head_id) * seq_len_x_size_per_head + (COL32_col << 5) * seq_len + COL32_row) >> 2;

        const char4* src_ptr4 = (const char4*)src;

        char4* dst_ptr4 = (char4*)dst;
        dst_ptr4[outIdx] = __ldg(src_ptr4 + inIdx);
    }
}

void invokeTransposeCOL32ToRowRebuildPadding(int8_t* dst,
                                             const int8_t* src,
                                             const int* sequence_id_map,
                                             const int valid_word_num,
                                             const int batch_size,
                                             const int seq_len,
                                             const int head_num,
                                             const int size_per_head,
                                             const float* bmm2_deQFactor,
                                             const float* out_scale_ptr,
                                             cudaStream_t stream)
{
    assert(size_per_head % 32 == 0);
    transpose_COL32_ROW_rebuild_padding_kernel<<<dim3(seq_len, batch_size),
                                                 dim3(size_per_head / 4, head_num),
                                                 0,
                                                 stream>>>(dst,
                                                           src,
                                                           sequence_id_map,
                                                           valid_word_num,
                                                           batch_size,
                                                           seq_len,
                                                           head_num,
                                                           size_per_head,
                                                           bmm2_deQFactor,
                                                           out_scale_ptr,
                                                           seq_len * size_per_head,
                                                           head_num * size_per_head);
}

}  // namespace fastertransformer
