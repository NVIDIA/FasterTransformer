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

#include "src/fastertransformer/kernels/int8_utils.cuh"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"
#include "src/fastertransformer/kernels/unfused_attention_int8_kernels.h"

namespace fastertransformer {

__inline__ __device__ int target_index(int id1, int id2, int id3, int id4, int dim_1, int dim_2, int dim_3, int dim_4)
{
    return id1 * (dim_2 * dim_3 * dim_4) + id3 * (dim_2 * dim_4) + id2 * dim_4 + id4;
}

// build a mapping for fullData to removePaddingData
// grid((valid_word_num+63)/64)
// block(64)
__global__ void mappingRemovePaddingData(int* mapping, const int* sequence_id_offset, const int valid_word_num)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < valid_word_num) {
        mapping[idx + __ldg(sequence_id_offset + idx)] = idx;
    }
}

void invokeMappingRemovePaddingData(const int batch_size,
                                    const int seq_len,
                                    const int valid_word_num,
                                    int* mapping,
                                    const int* sequence_id_offset,
                                    cudaStream_t stream)
{
    cudaMemsetAsync(mapping, -1, batch_size * seq_len * sizeof(int), stream);
    mappingRemovePaddingData<<<dim3((valid_word_num + 63) / 64), dim3(64), 0, stream>>>(
        mapping, sequence_id_offset, valid_word_num);
}

// add_QK_bias_transform for batch int8 cublasLtMatmul & per axis quantization for weight
// 1.add QK bias
// 2.transform each Q K CUBLASLT_ORDER_COL32 matrixes into a series of sub-matrix (with
// CUBLASLT_ORDER_COL32/CUBLASLT_ORDER_COL4_4R2_8C layout)
//  Q, K are CUBLASLT_ORDER_COL32 matrixes of m = batch_size * seq_len, n = head_num * size_per_head
//  q_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL32
//  k_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL4_4R2_8C
// only for int32 input & int8 output
// seq_len, size_per_head must be a multiple of 32
// grid.x = batch_size * seq_len * 2;
// block.x = head_num * size_per_head / 4;
// using char4
template<typename T>
__global__ void add_QK_bias_transform(int8_t* q_buf_,
                                      int8_t* k_buf_,
                                      const int32_t* Q,
                                      const T* bias_Q,
                                      const int32_t* K,
                                      const T* bias_K,
                                      const int m,
                                      const int batch_size,
                                      const int seq_len,
                                      const int head_num,
                                      const int size_per_head,
                                      int stride,
                                      const float* q_weight_amax,
                                      const float* q_input_deQFactor_div127_ptr,
                                      const float* k_weight_amax,
                                      const float* k_input_deQFactor_div127_ptr,
                                      const float* q_output_scale_ptr,
                                      const float* k_output_scale_ptr,
                                      bool use_ORDER_COL32_2R_4R4)
{
    const int32_t* data_ptr;
    char4* buf_ptr4;
    const T* bias_ptr;
    const float* weight_amax;
    int qk_id = blockIdx.x / m;

    data_ptr = qk_id == 0 ? Q : K;
    buf_ptr4 = qk_id == 0 ? (char4*)q_buf_ : (char4*)k_buf_;
    bias_ptr = qk_id == 0 ? bias_Q : bias_K;
    const float input_deQFactor_div127 =
        qk_id == 0 ? __ldg(q_input_deQFactor_div127_ptr) : __ldg(k_input_deQFactor_div127_ptr);
    weight_amax = qk_id == 0 ? q_weight_amax : k_weight_amax;
    const float output_scale = qk_id == 0 ? __ldg(q_output_scale_ptr) : __ldg(k_output_scale_ptr);

    int threadIdx4 = threadIdx.x << 2;
    int batch_id = (blockIdx.x % m) / seq_len;
    int head_id = threadIdx4 / size_per_head;
    int id_in_head = threadIdx4 % size_per_head;
    int word_id = blockIdx.x % seq_len;

    int data_id = (((threadIdx4 >> 5) << 5) * m + ((blockIdx.x % m) << 5) + (threadIdx4 & 31));

    float scale;
    float tmp;
    char4 tmp4;
    scale = static_cast<float>(__ldg(data_ptr + data_id)) * __ldg(weight_amax + threadIdx4) * input_deQFactor_div127;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.x = float_to_int8_rn(tmp * output_scale);

    data_id = data_id + 1;
    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(__ldg(data_ptr + data_id)) * __ldg(weight_amax + threadIdx4) * input_deQFactor_div127;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.y = float_to_int8_rn(tmp * output_scale);

    data_id = data_id + 1;
    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(__ldg(data_ptr + data_id)) * __ldg(weight_amax + threadIdx4) * input_deQFactor_div127;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.z = float_to_int8_rn(tmp * output_scale);

    data_id = data_id + 1;
    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(__ldg(data_ptr + data_id)) * __ldg(weight_amax + threadIdx4) * input_deQFactor_div127;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.w = float_to_int8_rn(tmp * output_scale);

    // row_id, col_id of sub-matrix (m = seq_len, n = size_per_head), column-major

    int row_id = word_id;
    int col_id = id_in_head;
    // new (row, rol) of LtTrans COL32/COL4 sub-matrix, leading dim = (COL32_ * seq_len)
    int new_col = col_id >> 5;
    int new_row;
    if (use_ORDER_COL32_2R_4R4) {
        int row_in_tile = row_id & 31;
        int col_in_tile = col_id & 31;
        new_row = (qk_id != 1) ?
                      // COL32
                      ((row_id << 5) + (col_id & 31)) :
                      // COL32_2R_4R4
                      (((row_id >> 5) << 10) +
                       //(((row%8)/2*4+row/8)*2+row%2)*32+col
                       (((((((row_in_tile % 8) >> 1) << 2) + (row_in_tile >> 3)) << 1) + (row_in_tile & 1)) << 5)
                       + col_in_tile);
    }
    else {
        new_row = (qk_id != 1) ?
                      // COL32
                      ((row_id << 5) + (col_id & 31)) :
                      // COL4
                      ////row_id/8 is the number of tile of (8 rows 32 columns) -- column-major
                      ////row_id%2 is even row, otherwise odd row
                      ////col_id%COL32_/8 is the number tile of (8 rows 8 columns)
                      (((((row_id >> 3) << 3) + ((row_id & 1) << 2) + ((col_id % 32) >> 3)) << 5) +
                       ////col_id%8 >= 4 is the right half of (8 rows 8 columns) tile
                       ////(row_id%8/2) is (the row id of alternating 4 rows) - 1
                       (((((col_id & 7) >= 4) ? 4 : 0) + ((row_id % 8) >> 1)) << 2) +
                       ////col_id%4 is the id of 4 cols
                       (col_id & 3));
    }

    buf_ptr4[(((batch_id * head_num + head_id) * stride + (new_col << 5) * seq_len + new_row) >> 2)] = tmp4;
}

// add_QK_bias_transform_varlen for batch int8 cublasLtMatmul & per axis quantization for weight
// 1.add QK bias
// 2.transform each Q K CUBLASLT_ORDER_COL32 matrixes into a series of sub-matrix (with
// CUBLASLT_ORDER_COL32/CUBLASLT_ORDER_COL4_4R2_8C layout)
//  Q, K are CUBLASLT_ORDER_COL32 matrixes of m = batch_size * seq_len, n = head_num * size_per_head
//  q_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL32
//  k_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL4_4R2_8C
// only for int32 input & int8 output
// seq_len, size_per_head must be a multiple of 32
// grid.x = batch_size * seq_len * 2;
// block.x = head_num * size_per_head / 4;
// using char4
template<typename T>
__global__ void add_QK_bias_transform_varlen(int8_t* q_buf_,
                                             int8_t* k_buf_,
                                             const int32_t* Q,
                                             const T* bias_Q,
                                             const int32_t* K,
                                             const T* bias_K,
                                             const int m,
                                             const int batch_size,
                                             const int seq_len,
                                             const int head_num,
                                             const int size_per_head,
                                             const int seq_len_padded,
                                             int stride_q,
                                             int stride_k,
                                             const float* q_weight_amax,
                                             const float* q_input_deQFactor_div127_ptr,
                                             const float* k_weight_amax,
                                             const float* k_input_deQFactor_div127_ptr,
                                             const float* q_output_scale_ptr,
                                             const float* k_output_scale_ptr,
                                             bool use_ORDER_COL32_2R_4R4)
{
    const int32_t* data_ptr;
    char4* buf_ptr4;
    const T* bias_ptr;
    const float* weight_amax;
    int qk_id = blockIdx.x / m;

    data_ptr = qk_id == 0 ? Q : K;
    buf_ptr4 = qk_id == 0 ? (char4*)q_buf_ : (char4*)k_buf_;
    bias_ptr = qk_id == 0 ? bias_Q : bias_K;
    const float input_deQFactor_div127 =
        qk_id == 0 ? __ldg(q_input_deQFactor_div127_ptr) : __ldg(k_input_deQFactor_div127_ptr);
    weight_amax = qk_id == 0 ? q_weight_amax : k_weight_amax;
    const float output_scale = qk_id == 0 ? __ldg(q_output_scale_ptr) : __ldg(k_output_scale_ptr);

    int threadIdx4 = threadIdx.x << 2;
    int batch_id = (blockIdx.x % m) / seq_len;
    int head_id = threadIdx4 / size_per_head;
    int id_in_head = threadIdx4 % size_per_head;
    int word_id = blockIdx.x % seq_len;

    int data_id = (((threadIdx4 >> 5) << 5) * m + ((blockIdx.x % m) << 5) + (threadIdx4 & 31));

    float scale;
    float tmp;
    char4 tmp4;
    scale = static_cast<float>(__ldg(data_ptr + data_id)) * __ldg(weight_amax + threadIdx4) * input_deQFactor_div127;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.x = float_to_int8_rn(tmp * output_scale);

    data_id = data_id + 1;
    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(__ldg(data_ptr + data_id)) * __ldg(weight_amax + threadIdx4) * input_deQFactor_div127;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.y = float_to_int8_rn(tmp * output_scale);

    data_id = data_id + 1;
    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(__ldg(data_ptr + data_id)) * __ldg(weight_amax + threadIdx4) * input_deQFactor_div127;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.z = float_to_int8_rn(tmp * output_scale);

    data_id = data_id + 1;
    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(__ldg(data_ptr + data_id)) * __ldg(weight_amax + threadIdx4) * input_deQFactor_div127;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.w = float_to_int8_rn(tmp * output_scale);

    // row_id, col_id of sub-matrix (m = seq_len, n = size_per_head), column-major

    int row_id = word_id;
    int col_id = id_in_head;
    // new (row, rol) of LtTrans COL32/COL4 sub-matrix, leading dim = (COL32_ * seq_len)
    int new_col = col_id >> 5;
    int new_row;
    if (use_ORDER_COL32_2R_4R4) {
        int row_in_tile = row_id & 31;
        int col_in_tile = col_id & 31;
        new_row = (qk_id != 1) ?
                      // COL32
                      ((row_id << 5) + (col_id & 31)) :
                      // COL32_2R_4R4
                      (((row_id >> 5) << 10) +
                       //(((row%8)/2*4+row/8)*2+row%2)*32+col
                       (((((((row_in_tile % 8) >> 1) << 2) + (row_in_tile >> 3)) << 1) + (row_in_tile & 1)) << 5)
                       + col_in_tile);
    }
    else {
        new_row = (qk_id != 1) ?
                      // COL32
                      ((row_id << 5) + (col_id & 31)) :
                      // COL4
                      ////row_id/8 is the number of tile of (8 rows 32 columns) -- column-major
                      ////row_id%2 is even row, otherwise odd row
                      ////col_id%COL32_/8 is the number tile of (8 rows 8 columns)
                      (((((row_id >> 3) << 3) + ((row_id & 1) << 2) + ((col_id % 32) >> 3)) << 5) +
                       ////col_id%8 >= 4 is the right half of (8 rows 8 columns) tile
                       ////(row_id%8/2) is (the row id of alternating 4 rows) - 1
                       (((((col_id & 7) >= 4) ? 4 : 0) + ((row_id % 8) >> 1)) << 2) +
                       ////col_id%4 is the id of 4 cols
                       (col_id & 3));
    }

    const int act_seq_len = (qk_id == 0) ? seq_len : seq_len_padded;
    const int stride = (qk_id == 0) ? stride_q : stride_k;
    buf_ptr4[(((batch_id * head_num + head_id) * stride + (new_col << 5) * act_seq_len + new_row) >> 2)] = tmp4;
}

template<typename T>
void invokeAddQKBiasTransform(int8_t* q_buf,
                              int8_t* k_buf,
                              const int32_t* Q,
                              const T* bias_Q,
                              const int32_t* K,
                              const T* bias_K,
                              const int batch_size,
                              const int seq_len,
                              const int head_num,
                              const int size_per_head,
                              const float* q_weight_amax,
                              const float* q_input_deQFactor_div127_ptr,
                              const float* k_weight_amax,
                              const float* k_input_deQFactor_div127_ptr,
                              const float* q_output_scale_ptr,
                              const float* k_output_scale_ptr,
                              bool use_ORDER_COL32_2R_4R4,
                              cudaStream_t stream)
{
    if (seq_len % 32 == 0) {
        add_QK_bias_transform<<<dim3(batch_size * seq_len * 2), dim3((head_num * size_per_head) / 4), 0, stream>>>(
            q_buf,
            k_buf,
            Q,
            bias_Q,
            K,
            bias_K,
            batch_size * seq_len,
            batch_size,
            seq_len,
            head_num,
            size_per_head,
            seq_len * size_per_head,
            q_weight_amax,
            q_input_deQFactor_div127_ptr,
            k_weight_amax,
            k_input_deQFactor_div127_ptr,
            q_output_scale_ptr,
            k_output_scale_ptr,
            use_ORDER_COL32_2R_4R4);
    }
    else {
        int seq_len_padded = (seq_len + 31) / 32 * 32;
        add_QK_bias_transform_varlen<<<dim3(batch_size * seq_len * 2),
                                       dim3((head_num * size_per_head) / 4),
                                       0,
                                       stream>>>(q_buf,
                                                 k_buf,
                                                 Q,
                                                 bias_Q,
                                                 K,
                                                 bias_K,
                                                 batch_size * seq_len,
                                                 batch_size,
                                                 seq_len,
                                                 head_num,
                                                 size_per_head,
                                                 seq_len_padded,
                                                 seq_len * size_per_head,
                                                 seq_len_padded * size_per_head,
                                                 q_weight_amax,
                                                 q_input_deQFactor_div127_ptr,
                                                 k_weight_amax,
                                                 k_input_deQFactor_div127_ptr,
                                                 q_output_scale_ptr,
                                                 k_output_scale_ptr,
                                                 use_ORDER_COL32_2R_4R4);
    }
}

template void invokeAddQKBiasTransform(int8_t* q_buf,
                                       int8_t* k_buf,
                                       const int32_t* Q,
                                       const float* bias_Q,
                                       const int32_t* K,
                                       const float* bias_K,
                                       const int batch_size,
                                       const int seq_len,
                                       const int head_num,
                                       const int size_per_head,
                                       const float* q_weight_amax,
                                       const float* q_input_deQFactor_div127_ptr,
                                       const float* k_weight_amax,
                                       const float* k_input_deQFactor_div127_ptr,
                                       const float* q_output_scale_ptr,
                                       const float* k_output_scale_ptr,
                                       bool use_ORDER_COL32_2R_4R4,
                                       cudaStream_t stream);

template void invokeAddQKBiasTransform(int8_t* q_buf,
                                       int8_t* k_buf,
                                       const int32_t* Q,
                                       const half* bias_Q,
                                       const int32_t* K,
                                       const half* bias_K,
                                       const int batch_size,
                                       const int seq_len,
                                       const int head_num,
                                       const int size_per_head,
                                       const float* q_weight_amax,
                                       const float* q_input_deQFactor_div127_ptr,
                                       const float* k_weight_amax,
                                       const float* k_input_deQFactor_div127_ptr,
                                       const float* q_output_scale_ptr,
                                       const float* k_output_scale_ptr,
                                       bool use_ORDER_COL32_2R_4R4,
                                       cudaStream_t stream);

// add_QK_bias_padding_transform for batch int8 cublasLtMatmul & per tensor quantization for weight
// 1.add QK bias
// 2.padding seq_len in k_buf_ to a multiple of 32 named seq_len_padded
// 3.transform each Q K CUBLASLT_ORDER_COL32 matrixes into a series of sub-matrix (with
// CUBLASLT_ORDER_COL32/CUBLASLT_ORDER_COL4_4R2_8C layout)
//  Q, K are CUBLASLT_ORDER_COL32 matrixes of m = batch_size * seq_len, n = head_num * size_per_head
//  q_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL32
//  k_buf_ is of batchCount = batch_size * head_num, m = seq_len_padded, n = size_per_head, CUBLASLT_ORDER_COL4_4R2_8C
// only for int8 IO
// size_per_head must be a multiple of 32
// grid.x = batch_size * seq_len * 2;
// block.x = head_num * size_per_head / 4;
// using char4
template<typename T>
__global__ void add_QK_bias_transform_varlen(int8_t* q_buf_,
                                             int8_t* k_buf_,
                                             const int8_t* Q,
                                             const T* bias_Q,
                                             const int8_t* K,
                                             const T* bias_K,
                                             const int m,
                                             const int batch_size,
                                             const int seq_len,
                                             const int head_num,
                                             const int size_per_head,
                                             const int seq_len_padded,
                                             const int stride_q,
                                             const int stride_k,
                                             const float* q_input_deQFactor_ptr,
                                             const float* k_input_deQFactor_ptr,
                                             const float* q_output_scale_ptr,
                                             const float* k_output_scale_ptr,
                                             bool use_ORDER_COL32_2R_4R4)
{
    const char4* data_ptr;
    char4* buf_ptr4;
    const T* bias_ptr;
    int qk_id = blockIdx.x / m;

    data_ptr = qk_id == 0 ? (const char4*)Q : (const char4*)K;
    buf_ptr4 = qk_id == 0 ? (char4*)q_buf_ : (char4*)k_buf_;
    bias_ptr = qk_id == 0 ? bias_Q : bias_K;
    const float input_deQFactor = qk_id == 0 ? __ldg(q_input_deQFactor_ptr) : __ldg(k_input_deQFactor_ptr);
    const float output_scale = qk_id == 0 ? __ldg(q_output_scale_ptr) : __ldg(k_output_scale_ptr);

    int threadIdx4 = threadIdx.x << 2;
    int batch_id = (blockIdx.x % m) / seq_len;
    int head_id = threadIdx4 / size_per_head;
    int id_in_head = threadIdx4 % size_per_head;
    int word_id = blockIdx.x % seq_len;

    int data_id = (((threadIdx4 >> 5) << 5) * m + ((blockIdx.x % m) << 5) + (threadIdx4 & 31)) >> 2;

    float scale;
    float tmp;
    char4 tmp4 = __ldg(data_ptr + data_id);
    scale = static_cast<float>(tmp4.x) * input_deQFactor;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.x = float_to_int8_rn(tmp * output_scale);

    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(tmp4.y) * input_deQFactor;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.y = float_to_int8_rn(tmp * output_scale);

    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(tmp4.z) * input_deQFactor;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.z = float_to_int8_rn(tmp * output_scale);

    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(tmp4.w) * input_deQFactor;
    ;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.w = float_to_int8_rn(tmp * output_scale);

    // row_id, col_id of sub-matrix (m = seq_len/seq_len_padded, n = size_per_head), column-major

    int row_id = word_id;
    int col_id = id_in_head;
    // new (row, rol) of LtTrans COL32/COL4 sub-matrix, leading dim = (COL32_ * seq_len / COL32_ * seq_len_padded)
    int new_col = col_id >> 5;
    int new_row;
    if (use_ORDER_COL32_2R_4R4) {
        int row_in_tile = row_id & 31;
        int col_in_tile = col_id & 31;
        new_row = (qk_id != 1) ?
                      // COL32
                      ((row_id << 5) + (col_id & 31)) :
                      // COL32_2R_4R4
                      (((row_id >> 5) << 10) +
                       //(((row%8)/2*4+row/8)*2+row%2)*32+col
                       (((((((row_in_tile % 8) >> 1) << 2) + (row_in_tile >> 3)) << 1) + (row_in_tile & 1)) << 5)
                       + col_in_tile);
    }
    else {
        new_row = (qk_id != 1) ?
                      // COL32
                      ((row_id << 5) + (col_id & 31)) :
                      // COL4
                      ////row_id/8 is the number of tile of (8 rows 32 columns) -- column-major
                      ////row_id%2 is even row, otherwise odd row
                      ////col_id%COL32_/8 is the number tile of (8 rows 8 columns)
                      (((((row_id >> 3) << 3) + ((row_id & 1) << 2) + ((col_id % 32) >> 3)) << 5) +
                       ////col_id%8 >= 4 is the right half of (8 rows 8 columns) tile
                       ////(row_id%8/2) is (the row id of alternating 4 rows) - 1
                       (((((col_id & 7) >= 4) ? 4 : 0) + ((row_id % 8) >> 1)) << 2) +
                       ////col_id%4 is the id of 4 cols
                       (col_id & 3));
    }

    const int act_seq_len = (qk_id == 0) ? seq_len : seq_len_padded;
    const int stride = (qk_id == 0) ? stride_q : stride_k;
    buf_ptr4[(((batch_id * head_num + head_id) * stride + (new_col << 5) * act_seq_len + new_row) >> 2)] = tmp4;
}

// add_QK_bias_transform for batch int8 cublasLtMatmul & per axis quantization for weight
// 1.add QK bias
// 2.transform each Q K CUBLASLT_ORDER_COL32 matrixes into a series of sub-matrix (with
// CUBLASLT_ORDER_COL32/CUBLASLT_ORDER_COL4_4R2_8C layout)
//  Q, K are CUBLASLT_ORDER_COL32 matrixes of m = batch_size * seq_len, n = head_num * size_per_head
//  q_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL32
//  k_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL4_4R2_8C
// only for int8 IO
// seq_len, size_per_head must be a multiple of 32
// grid.x = batch_size * seq_len * 2;
// block.x = head_num * size_per_head / 4;
// using char4
template<typename T>
__global__ void add_QK_bias_transform(int8_t* q_buf_,
                                      int8_t* k_buf_,
                                      const int8_t* Q,
                                      const T* bias_Q,
                                      const int8_t* K,
                                      const T* bias_K,
                                      const int m,
                                      const int batch_size,
                                      const int seq_len,
                                      const int head_num,
                                      const int size_per_head,
                                      int stride,
                                      const float* q_input_deQFactor_ptr,
                                      const float* k_input_deQFactor_ptr,
                                      const float* q_output_scale_ptr,
                                      const float* k_output_scale_ptr,
                                      bool use_ORDER_COL32_2R_4R4)
{
    const char4* data_ptr;
    char4* buf_ptr4;
    const T* bias_ptr;
    int qk_id = blockIdx.x / m;

    data_ptr = qk_id == 0 ? (const char4*)Q : (const char4*)K;
    buf_ptr4 = qk_id == 0 ? (char4*)q_buf_ : (char4*)k_buf_;
    bias_ptr = qk_id == 0 ? bias_Q : bias_K;
    const float input_deQFactor = qk_id == 0 ? __ldg(q_input_deQFactor_ptr) : __ldg(k_input_deQFactor_ptr);
    const float output_scale = qk_id == 0 ? __ldg(q_output_scale_ptr) : __ldg(k_output_scale_ptr);

    int threadIdx4 = threadIdx.x << 2;
    int batch_id = (blockIdx.x % m) / seq_len;
    int head_id = threadIdx4 / size_per_head;
    int id_in_head = threadIdx4 % size_per_head;
    int word_id = blockIdx.x % seq_len;

    int data_id = (((threadIdx4 >> 5) << 5) * m + ((blockIdx.x % m) << 5) + (threadIdx4 & 31)) >> 2;

    float scale;
    float tmp;
    char4 tmp4 = __ldg(data_ptr + data_id);
    scale = static_cast<float>(tmp4.x) * input_deQFactor;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.x = float_to_int8_rn(tmp * output_scale);

    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(tmp4.y) * input_deQFactor;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.y = float_to_int8_rn(tmp * output_scale);

    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(tmp4.z) * input_deQFactor;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.z = float_to_int8_rn(tmp * output_scale);

    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(tmp4.w) * input_deQFactor;
    ;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.w = float_to_int8_rn(tmp * output_scale);

    // row_id, col_id of sub-matrix (m = seq_len, n = size_per_head), column-major

    int row_id = word_id;
    int col_id = id_in_head;
    // new (row, rol) of LtTrans COL32/COL4 sub-matrix, leading dim = (COL32_ * seq_len)
    int new_col = col_id >> 5;
    int new_row;
    if (use_ORDER_COL32_2R_4R4) {
        int row_in_tile = row_id & 31;
        int col_in_tile = col_id & 31;
        new_row = (qk_id != 1) ?
                      // COL32
                      ((row_id << 5) + (col_id & 31)) :
                      // COL32_2R_4R4
                      (((row_id >> 5) << 10) +
                       //(((row%8)/2*4+row/8)*2+row%2)*32+col
                       (((((((row_in_tile % 8) >> 1) << 2) + (row_in_tile >> 3)) << 1) + (row_in_tile & 1)) << 5)
                       + col_in_tile);
    }
    else {
        new_row = (qk_id != 1) ?
                      // COL32
                      ((row_id << 5) + (col_id & 31)) :
                      // COL4
                      ////row_id/8 is the number of tile of (8 rows 32 columns) -- column-major
                      ////row_id%2 is even row, otherwise odd row
                      ////col_id%COL32_/8 is the number tile of (8 rows 8 columns)
                      (((((row_id >> 3) << 3) + ((row_id & 1) << 2) + ((col_id % 32) >> 3)) << 5) +
                       ////col_id%8 >= 4 is the right half of (8 rows 8 columns) tile
                       ////(row_id%8/2) is (the row id of alternating 4 rows) - 1
                       (((((col_id & 7) >= 4) ? 4 : 0) + ((row_id % 8) >> 1)) << 2) +
                       ////col_id%4 is the id of 4 cols
                       (col_id & 3));
    }

    buf_ptr4[(((batch_id * head_num + head_id) * stride + (new_col << 5) * seq_len + new_row) >> 2)] = tmp4;
}

template<typename T>
void invokeAddQKBiasTransform(int8_t* q_buf,
                              int8_t* k_buf,
                              const int8_t* Q,
                              const T* bias_Q,
                              const int8_t* K,
                              const T* bias_K,
                              const int batch_size,
                              const int seq_len,
                              const int head_num,
                              const int size_per_head,
                              const float* q_input_deQFactor_ptr,
                              const float* k_input_deQFactor_ptr,
                              const float* q_output_scale_ptr,
                              const float* k_output_scale_ptr,
                              bool use_ORDER_COL32_2R_4R4,
                              cudaStream_t stream)
{
    assert(size_per_head % 32 == 0);
    if (seq_len % 32 == 0) {
        add_QK_bias_transform_varlen<<<dim3(batch_size * seq_len * 2),
                                       dim3((head_num * size_per_head) / 4),
                                       0,
                                       stream>>>(q_buf,
                                                 k_buf,
                                                 Q,
                                                 bias_Q,
                                                 K,
                                                 bias_K,
                                                 batch_size * seq_len,
                                                 batch_size,
                                                 seq_len,
                                                 head_num,
                                                 size_per_head,
                                                 seq_len,
                                                 seq_len * size_per_head,
                                                 seq_len * size_per_head,
                                                 q_input_deQFactor_ptr,
                                                 k_input_deQFactor_ptr,
                                                 q_output_scale_ptr,
                                                 k_output_scale_ptr,
                                                 use_ORDER_COL32_2R_4R4);
    }
    else {
        int seq_len_padded = (seq_len + 31) / 32 * 32;
        // The padding words will not be considered in softmax, so we don't need memset for k_buf_
        // cudaMemsetAsync(k_buf, 0, batch_size * head_num * seq_len_padded * size_per_head * sizeof(int8_t), stream);
        add_QK_bias_transform_varlen<<<dim3(batch_size * seq_len * 2),
                                       dim3((head_num * size_per_head) / 4),
                                       0,
                                       stream>>>(q_buf,
                                                 k_buf,
                                                 Q,
                                                 bias_Q,
                                                 K,
                                                 bias_K,
                                                 batch_size * seq_len,
                                                 batch_size,
                                                 seq_len,
                                                 head_num,
                                                 size_per_head,
                                                 seq_len_padded,
                                                 seq_len * size_per_head,
                                                 seq_len_padded * size_per_head,
                                                 q_input_deQFactor_ptr,
                                                 k_input_deQFactor_ptr,
                                                 q_output_scale_ptr,
                                                 k_output_scale_ptr,
                                                 use_ORDER_COL32_2R_4R4);
    }
}

template void invokeAddQKBiasTransform(int8_t* q_buf,
                                       int8_t* k_buf,
                                       const int8_t* Q,
                                       const float* bias_Q,
                                       const int8_t* K,
                                       const float* bias_K,
                                       const int batch_size,
                                       const int seq_len,
                                       const int head_num,
                                       const int size_per_head,
                                       const float* q_input_deQFactor_ptr,
                                       const float* k_input_deQFactor_ptr,
                                       const float* q_output_scale_ptr,
                                       const float* k_output_scale_ptr,
                                       bool use_ORDER_COL32_2R_4R4,
                                       cudaStream_t stream);

template void invokeAddQKBiasTransform(int8_t* q_buf,
                                       int8_t* k_buf,
                                       const int8_t* Q,
                                       const half* bias_Q,
                                       const int8_t* K,
                                       const half* bias_K,
                                       const int batch_size,
                                       const int seq_len,
                                       const int head_num,
                                       const int size_per_head,
                                       const float* q_input_deQFactor_ptr,
                                       const float* k_input_deQFactor_ptr,
                                       const float* q_output_scale_ptr,
                                       const float* k_output_scale_ptr,
                                       bool use_ORDER_COL32_2R_4R4,
                                       cudaStream_t stream);

// add_QK_bias_padding_transform for batch int8 cublasLtMatmul & per tensor quantization for weight
// 1.add QK bias
// 2.padding seq_len in k_buf_ to a multiple of 32 named seq_len_padded
// 3.transform each Q K row-major matrixes into a series of sub-matrix (with
// CUBLASLT_ORDER_COL32/CUBLASLT_ORDER_COL4_4R2_8C layout)
//  Q, K are row-major matrixes of m = batch_size * seq_len, n = head_num * size_per_head
//  q_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL32
//  k_buf_ is of batchCount = batch_size * head_num, m = seq_len_padded, n = size_per_head, CUBLASLT_ORDER_COL4_4R2_8C
// only for int8 IO
// size_per_head must be a multiple of 32
// grid.x = batch_size * seq_len * 2;
// block.x = head_num * size_per_head / 4;
// using char4
template<typename T>
__global__ void add_QK_bias_transform_varlen_row(int8_t* q_buf_,
                                                 int8_t* k_buf_,
                                                 const int8_t* Q,
                                                 const T* bias_Q,
                                                 const int8_t* K,
                                                 const T* bias_K,
                                                 const int m,
                                                 const int batch_size,
                                                 const int seq_len,
                                                 const int head_num,
                                                 const int size_per_head,
                                                 const int seq_len_padded,
                                                 const int stride_q,
                                                 const int stride_k,
                                                 const float* q_input_deQFactor_ptr,
                                                 const float* k_input_deQFactor_ptr,
                                                 const float* q_output_scale_ptr,
                                                 const float* k_output_scale_ptr,
                                                 bool use_ORDER_COL32_2R_4R4,
                                                 const int head_num_x_size_per_head)
{
    const char4* data_ptr;
    char4* buf_ptr4;
    const T* bias_ptr;
    int qk_id = blockIdx.x / m;

    data_ptr = qk_id == 0 ? (const char4*)Q : (const char4*)K;
    buf_ptr4 = qk_id == 0 ? (char4*)q_buf_ : (char4*)k_buf_;
    bias_ptr = qk_id == 0 ? bias_Q : bias_K;
    const float input_deQFactor = qk_id == 0 ? __ldg(q_input_deQFactor_ptr) : __ldg(k_input_deQFactor_ptr);
    const float output_scale = qk_id == 0 ? __ldg(q_output_scale_ptr) : __ldg(k_output_scale_ptr);

    int threadIdx4 = threadIdx.x << 2;
    int batch_seq_id = blockIdx.x % m;
    int batch_id = (batch_seq_id) / seq_len;
    int head_id = threadIdx4 / size_per_head;
    int id_in_head = threadIdx4 % size_per_head;
    int word_id = blockIdx.x % seq_len;

    int data_id = (batch_seq_id * head_num_x_size_per_head + threadIdx4) >> 2;

    float scale;
    float tmp;
    char4 tmp4 = __ldg(data_ptr + data_id);
    scale = static_cast<float>(tmp4.x) * input_deQFactor;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.x = float_to_int8_rn(tmp * output_scale);

    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(tmp4.y) * input_deQFactor;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.y = float_to_int8_rn(tmp * output_scale);

    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(tmp4.z) * input_deQFactor;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.z = float_to_int8_rn(tmp * output_scale);

    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(tmp4.w) * input_deQFactor;
    ;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.w = float_to_int8_rn(tmp * output_scale);

    // row_id, col_id of sub-matrix (m = seq_len/seq_len_padded, n = size_per_head), column-major

    int row_id = word_id;
    int col_id = id_in_head;
    // new (row, rol) of LtTrans COL32/COL4 sub-matrix, leading dim = (COL32_ * seq_len / COL32_ * seq_len_padded)
    int new_col = col_id >> 5;
    int new_row;
    if (use_ORDER_COL32_2R_4R4) {
        int row_in_tile = row_id & 31;
        int col_in_tile = col_id & 31;
        new_row = (qk_id != 1) ?
                      // COL32
                      ((row_id << 5) + (col_id & 31)) :
                      // COL32_2R_4R4
                      (((row_id >> 5) << 10) +
                       //(((row%8)/2*4+row/8)*2+row%2)*32+col
                       (((((((row_in_tile % 8) >> 1) << 2) + (row_in_tile >> 3)) << 1) + (row_in_tile & 1)) << 5)
                       + col_in_tile);
    }
    else {
        new_row = (qk_id != 1) ?
                      // COL32
                      ((row_id << 5) + (col_id & 31)) :
                      // COL4
                      ////row_id/8 is the number of tile of (8 rows 32 columns) -- column-major
                      ////row_id%2 is even row, otherwise odd row
                      ////col_id%COL32_/8 is the number tile of (8 rows 8 columns)
                      (((((row_id >> 3) << 3) + ((row_id & 1) << 2) + ((col_id % 32) >> 3)) << 5) +
                       ////col_id%8 >= 4 is the right half of (8 rows 8 columns) tile
                       ////(row_id%8/2) is (the row id of alternating 4 rows) - 1
                       (((((col_id & 7) >= 4) ? 4 : 0) + ((row_id % 8) >> 1)) << 2) +
                       ////col_id%4 is the id of 4 cols
                       (col_id & 3));
    }

    const int act_seq_len = (qk_id == 0) ? seq_len : seq_len_padded;
    const int stride = (qk_id == 0) ? stride_q : stride_k;
    buf_ptr4[(((batch_id * head_num + head_id) * stride + (new_col << 5) * act_seq_len + new_row) >> 2)] = tmp4;
}

template<typename T>
void invokeAddQKBiasTransformRow(int8_t* q_buf,
                                 int8_t* k_buf,
                                 const int8_t* Q,
                                 const T* bias_Q,
                                 const int8_t* K,
                                 const T* bias_K,
                                 const int batch_size,
                                 const int seq_len,
                                 const int head_num,
                                 const int size_per_head,
                                 const float* q_input_deQFactor_ptr,
                                 const float* k_input_deQFactor_ptr,
                                 const float* q_output_scale_ptr,
                                 const float* k_output_scale_ptr,
                                 bool use_ORDER_COL32_2R_4R4,
                                 cudaStream_t stream)
{
    assert(size_per_head % 32 == 0);
    if (seq_len % 32 == 0) {
        add_QK_bias_transform_varlen_row<<<dim3(batch_size * seq_len * 2),
                                           dim3((head_num * size_per_head) / 4),
                                           0,
                                           stream>>>(q_buf,
                                                     k_buf,
                                                     Q,
                                                     bias_Q,
                                                     K,
                                                     bias_K,
                                                     batch_size * seq_len,
                                                     batch_size,
                                                     seq_len,
                                                     head_num,
                                                     size_per_head,
                                                     seq_len,
                                                     seq_len * size_per_head,
                                                     seq_len * size_per_head,
                                                     q_input_deQFactor_ptr,
                                                     k_input_deQFactor_ptr,
                                                     q_output_scale_ptr,
                                                     k_output_scale_ptr,
                                                     use_ORDER_COL32_2R_4R4,
                                                     head_num * size_per_head);
    }
    else {
        int seq_len_padded = (seq_len + 31) / 32 * 32;
        // The padding words will not be considered in softmax, so we don't need memset for k_buf_
        // cudaMemsetAsync(k_buf, 0, batch_size * head_num * seq_len_padded * size_per_head * sizeof(int8_t), stream);
        add_QK_bias_transform_varlen_row<<<dim3(batch_size * seq_len * 2),
                                           dim3((head_num * size_per_head) / 4),
                                           0,
                                           stream>>>(q_buf,
                                                     k_buf,
                                                     Q,
                                                     bias_Q,
                                                     K,
                                                     bias_K,
                                                     batch_size * seq_len,
                                                     batch_size,
                                                     seq_len,
                                                     head_num,
                                                     size_per_head,
                                                     seq_len_padded,
                                                     seq_len * size_per_head,
                                                     seq_len_padded * size_per_head,
                                                     q_input_deQFactor_ptr,
                                                     k_input_deQFactor_ptr,
                                                     q_output_scale_ptr,
                                                     k_output_scale_ptr,
                                                     use_ORDER_COL32_2R_4R4,
                                                     head_num * size_per_head);
    }
}

template void invokeAddQKBiasTransformRow(int8_t* q_buf,
                                          int8_t* k_buf,
                                          const int8_t* Q,
                                          const float* bias_Q,
                                          const int8_t* K,
                                          const float* bias_K,
                                          const int batch_size,
                                          const int seq_len,
                                          const int head_num,
                                          const int size_per_head,
                                          const float* q_input_deQFactor_ptr,
                                          const float* k_input_deQFactor_ptr,
                                          const float* q_output_scale_ptr,
                                          const float* k_output_scale_ptr,
                                          bool use_ORDER_COL32_2R_4R4,
                                          cudaStream_t stream);

template void invokeAddQKBiasTransformRow(int8_t* q_buf,
                                          int8_t* k_buf,
                                          const int8_t* Q,
                                          const half* bias_Q,
                                          const int8_t* K,
                                          const half* bias_K,
                                          const int batch_size,
                                          const int seq_len,
                                          const int head_num,
                                          const int size_per_head,
                                          const float* q_input_deQFactor_ptr,
                                          const float* k_input_deQFactor_ptr,
                                          const float* q_output_scale_ptr,
                                          const float* k_output_scale_ptr,
                                          bool use_ORDER_COL32_2R_4R4,
                                          cudaStream_t stream);

// add_QK_bias_transform & rebuild padding for batch int8 cublasLtMatmul & per axis quantization for weight
// 1.add QK bias
// 2.transform each Q K CUBLASLT_ORDER_COL32 matrixes into a series of sub-matrix (with
// CUBLASLT_ORDER_COL32/CUBLASLT_ORDER_COL4_4R2_8C layout)
//  Q, K are CUBLASLT_ORDER_COL32 matrixes of m = valid_word_num, n = head_num * size_per_head
//  q_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL32
//  k_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL4_4R2_8C or
//  CUBLASLT_ORDER_COL32_2R_4R4
// only for int32 input & int8 output
// seq_len, size_per_head must be a multiple of 32
// grid.x = valid_word_num * 2;
// block.x = head_num * size_per_head / 4;
// using char4
template<typename T>
__global__ void add_QK_bias_transform_rebuild_padding(int8_t* q_buf_,
                                                      int8_t* k_buf_,
                                                      const int32_t* Q,
                                                      const T* bias_Q,
                                                      const int32_t* K,
                                                      const T* bias_K,
                                                      const int* sequence_id_offset,
                                                      const int valid_word_num,
                                                      const int m,
                                                      const int batch_size,
                                                      const int seq_len,
                                                      const int head_num,
                                                      const int size_per_head,
                                                      int stride,
                                                      const float* q_weight_amax,
                                                      const float* q_input_deQFactor_div127_ptr,
                                                      const float* k_weight_amax,
                                                      const float* k_input_deQFactor_div127_ptr,
                                                      const float* q_output_scale_ptr,
                                                      const float* k_output_scale_ptr,
                                                      bool use_ORDER_COL32_2R_4R4)
{
    const int32_t* data_ptr;
    char4* buf_ptr4;
    const T* bias_ptr;
    const float* weight_amax;
    int qk_id = blockIdx.x / valid_word_num;

    data_ptr = qk_id == 0 ? Q : K;
    buf_ptr4 = qk_id == 0 ? (char4*)q_buf_ : (char4*)k_buf_;
    bias_ptr = qk_id == 0 ? bias_Q : bias_K;

    int threadIdx4 = threadIdx.x << 2;
    int m_full_idx = blockIdx.x % valid_word_num;
    m_full_idx = (valid_word_num != m) ? (m_full_idx + __ldg(sequence_id_offset + m_full_idx)) : m_full_idx;
    int batch_id = m_full_idx / seq_len;
    int head_id = threadIdx4 / size_per_head;
    int id_in_head = threadIdx4 % size_per_head;
    int word_id = m_full_idx % seq_len;

    const float input_deQFactor_div127 =
        qk_id == 0 ? __ldg(q_input_deQFactor_div127_ptr) : __ldg(k_input_deQFactor_div127_ptr);
    weight_amax = qk_id == 0 ? q_weight_amax : k_weight_amax;
    const float output_scale = qk_id == 0 ? __ldg(q_output_scale_ptr) : __ldg(k_output_scale_ptr);

    int data_id =
        (((threadIdx4 >> 5) << 5) * valid_word_num + ((blockIdx.x % valid_word_num) << 5) + (threadIdx4 & 31));

    float scale;
    float tmp;
    char4 tmp4;
    scale = static_cast<float>(__ldg(data_ptr + data_id)) * __ldg(weight_amax + threadIdx4) * input_deQFactor_div127;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.x = float_to_int8_rn(tmp * output_scale);

    data_id = data_id + 1;
    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(__ldg(data_ptr + data_id)) * __ldg(weight_amax + threadIdx4) * input_deQFactor_div127;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.y = float_to_int8_rn(tmp * output_scale);

    data_id = data_id + 1;
    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(__ldg(data_ptr + data_id)) * __ldg(weight_amax + threadIdx4) * input_deQFactor_div127;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.z = float_to_int8_rn(tmp * output_scale);

    data_id = data_id + 1;
    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(__ldg(data_ptr + data_id)) * __ldg(weight_amax + threadIdx4) * input_deQFactor_div127;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.w = float_to_int8_rn(tmp * output_scale);

    // row_id, col_id of sub-matrix (m = seq_len, n = size_per_head), column-major
    int row_id = word_id;
    int col_id = id_in_head;
    // new (row, rol) of LtTrans COL32/COL4 sub-matrix, leading dim = (COL32_ * seq_len)
    int new_col = col_id >> 5;
    int new_row;
    if (use_ORDER_COL32_2R_4R4) {
        int row_in_tile = row_id & 31;
        int col_in_tile = col_id & 31;
        new_row = (qk_id != 1) ?
                      // COL32
                      ((row_id << 5) + (col_id & 31)) :
                      // COL32_2R_4R4
                      (((row_id >> 5) << 10) +
                       //(((row%8)/2*4+row/8)*2+row%2)*32+col
                       (((((((row_in_tile % 8) >> 1) << 2) + (row_in_tile >> 3)) << 1) + (row_in_tile & 1)) << 5)
                       + col_in_tile);
    }
    else {
        new_row = (qk_id != 1) ?
                      // COL32
                      ((row_id << 5) + (col_id & 31)) :
                      // COL4
                      ////row_id/8 is the number of tile of (8 rows 32 columns) -- column-major
                      ////row_id%2 is even row, otherwise odd row
                      ////col_id%COL32_/8 is the number tile of (8 rows 8 columns)
                      (((((row_id >> 3) << 3) + ((row_id & 1) << 2) + ((col_id % 32) >> 3)) << 5) +
                       ////col_id%8 >= 4 is the right half of (8 rows 8 columns) tile
                       ////(row_id%8/2) is (the row id of alternating 4 rows) - 1
                       (((((col_id & 7) >= 4) ? 4 : 0) + ((row_id % 8) >> 1)) << 2) +
                       ////col_id%4 is the id of 4 cols
                       (col_id & 3));
    }

    buf_ptr4[(((batch_id * head_num + head_id) * stride + (new_col << 5) * seq_len + new_row) >> 2)] = tmp4;
}

template<typename T>
void invokeAddQKBiasTransformRebuildPadding(int8_t* q_buf,
                                            int8_t* k_buf,
                                            const int32_t* Q,
                                            const T* bias_Q,
                                            const int32_t* K,
                                            const T* bias_K,
                                            const int* sequence_id_offset,
                                            const int valid_word_num,
                                            const int batch_size,
                                            const int seq_len,
                                            const int head_num,
                                            const int size_per_head,
                                            const float* q_weight_amax,
                                            const float* q_input_deQFactor_div127_ptr,
                                            const float* k_weight_amax,
                                            const float* k_input_deQFactor_div127_ptr,
                                            const float* q_output_scale_ptr,
                                            const float* k_output_scale_ptr,
                                            bool use_ORDER_COL32_2R_4R4,
                                            cudaStream_t stream)
{
    add_QK_bias_transform_rebuild_padding<<<dim3(valid_word_num * 2),
                                            dim3((head_num * size_per_head) / 4),
                                            0,
                                            stream>>>(q_buf,
                                                      k_buf,
                                                      Q,
                                                      bias_Q,
                                                      K,
                                                      bias_K,
                                                      sequence_id_offset,
                                                      valid_word_num,
                                                      batch_size * seq_len,
                                                      batch_size,
                                                      seq_len,
                                                      head_num,
                                                      size_per_head,
                                                      seq_len * size_per_head,
                                                      q_weight_amax,
                                                      q_input_deQFactor_div127_ptr,
                                                      k_weight_amax,
                                                      k_input_deQFactor_div127_ptr,
                                                      q_output_scale_ptr,
                                                      k_output_scale_ptr,
                                                      use_ORDER_COL32_2R_4R4);
}

template void invokeAddQKBiasTransformRebuildPadding(int8_t* q_buf,
                                                     int8_t* k_buf,
                                                     const int32_t* Q,
                                                     const float* bias_Q,
                                                     const int32_t* K,
                                                     const float* bias_K,
                                                     const int* sequence_id_offset,
                                                     const int valid_word_num,
                                                     const int batch_size,
                                                     const int seq_len,
                                                     const int head_num,
                                                     const int size_per_head,
                                                     const float* q_weight_amax,
                                                     const float* q_input_deQFactor_div127_ptr,
                                                     const float* k_weight_amax,
                                                     const float* k_input_deQFactor_div127_ptr,
                                                     const float* q_output_scale_ptr,
                                                     const float* k_output_scale_ptr,
                                                     bool use_ORDER_COL32_2R_4R4,
                                                     cudaStream_t stream);

template void invokeAddQKBiasTransformRebuildPadding(int8_t* q_buf,
                                                     int8_t* k_buf,
                                                     const int32_t* Q,
                                                     const half* bias_Q,
                                                     const int32_t* K,
                                                     const half* bias_K,
                                                     const int* sequence_id_offset,
                                                     const int valid_word_num,
                                                     const int batch_size,
                                                     const int seq_len,
                                                     const int head_num,
                                                     const int size_per_head,
                                                     const float* q_weight_amax,
                                                     const float* q_input_deQFactor_div127_ptr,
                                                     const float* k_weight_amax,
                                                     const float* k_input_deQFactor_div127_ptr,
                                                     const float* q_output_scale_ptr,
                                                     const float* k_output_scale_ptr,
                                                     bool use_ORDER_COL32_2R_4R4,
                                                     cudaStream_t stream);

// add_QK_bias_transform & rebuild padding for batch int8 cublasLtMatmul & per tensor quantization for weight
// 1.add QK bias
// 2.transform each Q K CUBLASLT_ORDER_COL32 matrixes into a series of sub-matrix (with
// CUBLASLT_ORDER_COL32/CUBLASLT_ORDER_COL4_4R2_8C layout)
//  Q, K are CUBLASLT_ORDER_COL32 matrixes of m = valid_word_num, n = head_num * size_per_head
//  q_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL32
//  k_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL4_4R2_8C or
//  CUBLASLT_ORDER_COL32_2R_4R4
// only for int8 IO
// seq_len, size_per_head must be a multiple of 32
// grid.x = valid_word_num * 2;
// block.x = head_num * size_per_head / 4;
// using char4
template<typename T>
__global__ void add_QK_bias_transform_rebuild_padding(int8_t* q_buf_,
                                                      int8_t* k_buf_,
                                                      const int8_t* Q,
                                                      const T* bias_Q,
                                                      const int8_t* K,
                                                      const T* bias_K,
                                                      const int* sequence_id_offset,
                                                      const int valid_word_num,
                                                      const int m,
                                                      const int batch_size,
                                                      const int seq_len,
                                                      const int head_num,
                                                      const int size_per_head,
                                                      int stride,
                                                      const float* q_deQFactor_ptr,
                                                      const float* k_deQFactor_ptr,
                                                      const float* q_output_scale_ptr,
                                                      const float* k_output_scale_ptr,
                                                      bool use_ORDER_COL32_2R_4R4)
{
    const char4* data_ptr;
    char4* buf_ptr4;
    const T* bias_ptr;
    int qk_id = blockIdx.x / valid_word_num;

    data_ptr = qk_id == 0 ? (const char4*)Q : (const char4*)K;
    buf_ptr4 = qk_id == 0 ? (char4*)q_buf_ : (char4*)k_buf_;
    bias_ptr = qk_id == 0 ? bias_Q : bias_K;

    int threadIdx4 = threadIdx.x << 2;
    int m_full_idx = blockIdx.x % valid_word_num;
    m_full_idx = (valid_word_num != m) ? (m_full_idx + __ldg(sequence_id_offset + m_full_idx)) : m_full_idx;
    int batch_id = m_full_idx / seq_len;
    int head_id = threadIdx4 / size_per_head;
    int id_in_head = threadIdx4 % size_per_head;
    int word_id = m_full_idx % seq_len;

    const float deQFactor = qk_id == 0 ? __ldg(q_deQFactor_ptr) : __ldg(k_deQFactor_ptr);
    const float output_scale = qk_id == 0 ? __ldg(q_output_scale_ptr) : __ldg(k_output_scale_ptr);

    int data_id =
        (((threadIdx4 >> 5) << 5) * valid_word_num + ((blockIdx.x % valid_word_num) << 5) + (threadIdx4 & 31)) >> 2;

    float scale;
    float tmp;
    char4 tmp4;

    tmp4 = __ldg(data_ptr + data_id);

    scale = static_cast<float>(tmp4.x) * deQFactor;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.x = float_to_int8_rn(tmp * output_scale);

    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(tmp4.y) * deQFactor;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.y = float_to_int8_rn(tmp * output_scale);

    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(tmp4.z) * deQFactor;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.z = float_to_int8_rn(tmp * output_scale);

    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(tmp4.w) * deQFactor;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.w = float_to_int8_rn(tmp * output_scale);

    // row_id, col_id of sub-matrix (m = seq_len, n = size_per_head), column-major
    int row_id = word_id;
    int col_id = id_in_head;
    // new (row, rol) of LtTrans COL32/COL4 sub-matrix, leading dim = (COL32_ * seq_len)
    int new_col = col_id >> 5;
    int new_row;
    if (use_ORDER_COL32_2R_4R4) {
        int row_in_tile = row_id & 31;
        int col_in_tile = col_id & 31;
        new_row = (qk_id != 1) ?
                      // COL32
                      ((row_id << 5) + (col_id & 31)) :
                      // COL32_2R_4R4
                      (((row_id >> 5) << 10) +
                       //(((row%8)/2*4+row/8)*2+row%2)*32+col
                       (((((((row_in_tile % 8) >> 1) << 2) + (row_in_tile >> 3)) << 1) + (row_in_tile & 1)) << 5)
                       + col_in_tile);
    }
    else {
        new_row = (qk_id != 1) ?
                      // COL32
                      ((row_id << 5) + (col_id & 31)) :
                      // COL4
                      ////row_id/8 is the number of tile of (8 rows 32 columns) -- column-major
                      ////row_id%2 is even row, otherwise odd row
                      ////col_id%COL32_/8 is the number tile of (8 rows 8 columns)
                      (((((row_id >> 3) << 3) + ((row_id & 1) << 2) + ((col_id % 32) >> 3)) << 5) +
                       ////col_id%8 >= 4 is the right half of (8 rows 8 columns) tile
                       ////(row_id%8/2) is (the row id of alternating 4 rows) - 1
                       (((((col_id & 7) >= 4) ? 4 : 0) + ((row_id % 8) >> 1)) << 2) +
                       ////col_id%4 is the id of 4 cols
                       (col_id & 3));
    }

    buf_ptr4[(((batch_id * head_num + head_id) * stride + (new_col << 5) * seq_len + new_row) >> 2)] = tmp4;
}

// add_QK_bias_transform & rebuild padding for batch int8 cublasLtMatmul & per tensor quantization for weight
// 1.add QK bias
// 2.transform each Q K CUBLASLT_ORDER_COL32 matrixes into a series of sub-matrix (with
// CUBLASLT_ORDER_COL32/CUBLASLT_ORDER_COL4_4R2_8C layout)
//  Q, K are CUBLASLT_ORDER_COL32 matrixes of m = valid_word_num, n = head_num * size_per_head
//  q_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL32
//  seq_len_padded = (seq_len + 31)/32*32;
//  k_buf_ is of batchCount = batch_size * head_num, m = seq_len_padded, n = size_per_head, CUBLASLT_ORDER_COL4_4R2_8C
//  or CUBLASLT_ORDER_COL32_2R_4R4
// only for int8 IO
// seq_len, size_per_head must be a multiple of 32
// grid.x = valid_word_num * 2;
// block.x = head_num * size_per_head / 4;
// using char4
template<typename T>
__global__ void add_QK_bias_transform_rebuild_padding_varlen(int8_t* q_buf_,
                                                             int8_t* k_buf_,
                                                             const int8_t* Q,
                                                             const T* bias_Q,
                                                             const int8_t* K,
                                                             const T* bias_K,
                                                             const int* sequence_id_offset,
                                                             const int valid_word_num,
                                                             const int m,
                                                             const int batch_size,
                                                             const int seq_len,
                                                             const int seq_len_padded,
                                                             const int head_num,
                                                             const int size_per_head,
                                                             int stride_q,
                                                             int stride_k,
                                                             const float* q_deQFactor_ptr,
                                                             const float* k_deQFactor_ptr,
                                                             const float* q_output_scale_ptr,
                                                             const float* k_output_scale_ptr,
                                                             bool use_ORDER_COL32_2R_4R4)
{
    const char4* data_ptr;
    char4* buf_ptr4;
    const T* bias_ptr;
    int qk_id = blockIdx.x / valid_word_num;

    data_ptr = qk_id == 0 ? (const char4*)Q : (const char4*)K;
    buf_ptr4 = qk_id == 0 ? (char4*)q_buf_ : (char4*)k_buf_;
    bias_ptr = qk_id == 0 ? bias_Q : bias_K;

    int threadIdx4 = threadIdx.x << 2;
    int m_full_idx = blockIdx.x % valid_word_num;
    m_full_idx = (valid_word_num != m) ? (m_full_idx + __ldg(sequence_id_offset + m_full_idx)) : m_full_idx;
    int batch_id = m_full_idx / seq_len;
    int head_id = threadIdx4 / size_per_head;
    int id_in_head = threadIdx4 % size_per_head;
    int word_id = m_full_idx % seq_len;

    const float deQFactor = qk_id == 0 ? __ldg(q_deQFactor_ptr) : __ldg(k_deQFactor_ptr);
    const float output_scale = qk_id == 0 ? __ldg(q_output_scale_ptr) : __ldg(k_output_scale_ptr);

    int data_id =
        (((threadIdx4 >> 5) << 5) * valid_word_num + ((blockIdx.x % valid_word_num) << 5) + (threadIdx4 & 31)) >> 2;

    float scale;
    float tmp;
    char4 tmp4;

    tmp4 = __ldg(data_ptr + data_id);

    scale = static_cast<float>(tmp4.x) * deQFactor;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.x = float_to_int8_rn(tmp * output_scale);

    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(tmp4.y) * deQFactor;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.y = float_to_int8_rn(tmp * output_scale);

    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(tmp4.z) * deQFactor;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.z = float_to_int8_rn(tmp * output_scale);

    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(tmp4.w) * deQFactor;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.w = float_to_int8_rn(tmp * output_scale);

    // row_id, col_id of sub-matrix (m = seq_len or seq_len_padded, n = size_per_head), column-major
    int row_id = word_id;
    int col_id = id_in_head;
    // new (row, rol) of LtTrans COL32/COL4 sub-matrix, leading dim = (COL32_ * seq_len) or (COL32_ * seq_len_padded)
    int new_col = col_id >> 5;
    int new_row;
    if (use_ORDER_COL32_2R_4R4) {
        int row_in_tile = row_id & 31;
        int col_in_tile = col_id & 31;
        new_row = (qk_id != 1) ?
                      // COL32
                      ((row_id << 5) + (col_id & 31)) :
                      // COL32_2R_4R4
                      (((row_id >> 5) << 10) +
                       //(((row%8)/2*4+row/8)*2+row%2)*32+col
                       (((((((row_in_tile % 8) >> 1) << 2) + (row_in_tile >> 3)) << 1) + (row_in_tile & 1)) << 5)
                       + col_in_tile);
    }
    else {
        new_row = (qk_id != 1) ?
                      // COL32
                      ((row_id << 5) + (col_id & 31)) :
                      // COL4
                      ////row_id/8 is the number of tile of (8 rows 32 columns) -- column-major
                      ////row_id%2 is even row, otherwise odd row
                      ////col_id%COL32_/8 is the number tile of (8 rows 8 columns)
                      (((((row_id >> 3) << 3) + ((row_id & 1) << 2) + ((col_id % 32) >> 3)) << 5) +
                       ////col_id%8 >= 4 is the right half of (8 rows 8 columns) tile
                       ////(row_id%8/2) is (the row id of alternating 4 rows) - 1
                       (((((col_id & 7) >= 4) ? 4 : 0) + ((row_id % 8) >> 1)) << 2) +
                       ////col_id%4 is the id of 4 cols
                       (col_id & 3));
    }

    const int stride = (qk_id != 1) ? stride_q : stride_k;
    const int len = (qk_id != 1) ? seq_len : seq_len_padded;
    buf_ptr4[(((batch_id * head_num + head_id) * stride + (new_col << 5) * len + new_row) >> 2)] = tmp4;
}

template<typename T>
void invokeAddQKBiasTransformRebuildPadding(int8_t* q_buf,
                                            int8_t* k_buf,
                                            const int8_t* Q,
                                            const T* bias_Q,
                                            const int8_t* K,
                                            const T* bias_K,
                                            const int* sequence_id_offset,
                                            const int valid_word_num,
                                            const int batch_size,
                                            const int seq_len,
                                            const int head_num,
                                            const int size_per_head,
                                            const float* q_deQFactor_ptr,
                                            const float* k_deQFactor_ptr,
                                            const float* q_output_scale_ptr,
                                            const float* k_output_scale_ptr,
                                            bool use_ORDER_COL32_2R_4R4,
                                            cudaStream_t stream)
{
    int seq_len_padded = (seq_len + 31) / 32 * 32;
    add_QK_bias_transform_rebuild_padding_varlen<<<dim3(valid_word_num * 2),
                                                   dim3((head_num * size_per_head) / 4),
                                                   0,
                                                   stream>>>(q_buf,
                                                             k_buf,
                                                             Q,
                                                             bias_Q,
                                                             K,
                                                             bias_K,
                                                             sequence_id_offset,
                                                             valid_word_num,
                                                             batch_size * seq_len,
                                                             batch_size,
                                                             seq_len,
                                                             seq_len_padded,
                                                             head_num,
                                                             size_per_head,
                                                             seq_len * size_per_head,
                                                             seq_len_padded * size_per_head,
                                                             q_deQFactor_ptr,
                                                             k_deQFactor_ptr,
                                                             q_output_scale_ptr,
                                                             k_output_scale_ptr,
                                                             use_ORDER_COL32_2R_4R4);
}

template void invokeAddQKBiasTransformRebuildPadding(int8_t* q_buf,
                                                     int8_t* k_buf,
                                                     const int8_t* Q,
                                                     const float* bias_Q,
                                                     const int8_t* K,
                                                     const float* bias_K,
                                                     const int* sequence_id_offset,
                                                     const int valid_word_num,
                                                     const int batch_size,
                                                     const int seq_len,
                                                     const int head_num,
                                                     const int size_per_head,
                                                     const float* q_deQFactor_ptr,
                                                     const float* k_deQFactor_ptr,
                                                     const float* q_output_scale_ptr,
                                                     const float* k_output_scale_ptr,
                                                     bool use_ORDER_COL32_2R_4R4,
                                                     cudaStream_t stream);

template void invokeAddQKBiasTransformRebuildPadding(int8_t* q_buf,
                                                     int8_t* k_buf,
                                                     const int8_t* Q,
                                                     const half* bias_Q,
                                                     const int8_t* K,
                                                     const half* bias_K,
                                                     const int* sequence_id_offset,
                                                     const int valid_word_num,
                                                     const int batch_size,
                                                     const int seq_len,
                                                     const int head_num,
                                                     const int size_per_head,
                                                     const float* q_deQFactor_ptr,
                                                     const float* k_deQFactor_ptr,
                                                     const float* q_output_scale_ptr,
                                                     const float* k_output_scale_ptr,
                                                     bool use_ORDER_COL32_2R_4R4,
                                                     cudaStream_t stream);

// add_QK_bias_transform & rebuild padding for batch int8 cublasLtMatmul & per tensor quantization for weight
// 1.add QK bias
// 2.transform each Q K row-major matrixes into a series of sub-matrix (with
// CUBLASLT_ORDER_COL32/CUBLASLT_ORDER_COL4_4R2_8C layout)
//  Q, K are CUBLASLT_ORDER_COL32 matrixes of m = valid_word_num, n = head_num * size_per_head
//  q_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL32
//  seq_len_padded = (seq_len + 31)/32*32;
//  k_buf_ is of batchCount = batch_size * head_num, m = seq_len_padded, n = size_per_head, CUBLASLT_ORDER_COL4_4R2_8C
//  or CUBLASLT_ORDER_COL32_2R_4R4
// only for int8 IO
// seq_len, size_per_head must be a multiple of 32
// grid.x = valid_word_num * 2;
// block.x = head_num * size_per_head / 4;
// using char4
template<typename T>
__global__ void add_QK_bias_transform_rebuild_padding_varlen_row(int8_t* q_buf_,
                                                                 int8_t* k_buf_,
                                                                 const int8_t* Q,
                                                                 const T* bias_Q,
                                                                 const int8_t* K,
                                                                 const T* bias_K,
                                                                 const int* sequence_id_offset,
                                                                 const int valid_word_num,
                                                                 const int m,
                                                                 const int batch_size,
                                                                 const int seq_len,
                                                                 const int seq_len_padded,
                                                                 const int head_num,
                                                                 const int size_per_head,
                                                                 int stride_q,
                                                                 int stride_k,
                                                                 const float* q_deQFactor_ptr,
                                                                 const float* k_deQFactor_ptr,
                                                                 const float* q_output_scale_ptr,
                                                                 const float* k_output_scale_ptr,
                                                                 bool use_ORDER_COL32_2R_4R4,
                                                                 const int head_num_x_size_per_head)
{
    const char4* data_ptr;
    char4* buf_ptr4;
    const T* bias_ptr;
    int qk_id = blockIdx.x / valid_word_num;

    data_ptr = qk_id == 0 ? (const char4*)Q : (const char4*)K;
    buf_ptr4 = qk_id == 0 ? (char4*)q_buf_ : (char4*)k_buf_;
    bias_ptr = qk_id == 0 ? bias_Q : bias_K;

    int threadIdx4 = threadIdx.x << 2;
    int batch_seq_id = blockIdx.x % valid_word_num;
    int m_full_idx = (valid_word_num != m) ? (batch_seq_id + __ldg(sequence_id_offset + batch_seq_id)) : batch_seq_id;
    int batch_id = m_full_idx / seq_len;
    int head_id = threadIdx4 / size_per_head;
    int id_in_head = threadIdx4 % size_per_head;
    int word_id = m_full_idx % seq_len;

    const float deQFactor = qk_id == 0 ? __ldg(q_deQFactor_ptr) : __ldg(k_deQFactor_ptr);
    const float output_scale = qk_id == 0 ? __ldg(q_output_scale_ptr) : __ldg(k_output_scale_ptr);

    int data_id = (batch_seq_id * head_num_x_size_per_head + threadIdx4) >> 2;

    float scale;
    float tmp;
    char4 tmp4;

    tmp4 = __ldg(data_ptr + data_id);

    scale = static_cast<float>(tmp4.x) * deQFactor;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.x = float_to_int8_rn(tmp * output_scale);

    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(tmp4.y) * deQFactor;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.y = float_to_int8_rn(tmp * output_scale);

    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(tmp4.z) * deQFactor;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.z = float_to_int8_rn(tmp * output_scale);

    threadIdx4 = threadIdx4 + 1;
    scale = static_cast<float>(tmp4.w) * deQFactor;
    tmp = static_cast<float>(__ldg(bias_ptr + threadIdx4)) + scale;
    tmp4.w = float_to_int8_rn(tmp * output_scale);

    // row_id, col_id of sub-matrix (m = seq_len or seq_len_padded, n = size_per_head), column-major
    int row_id = word_id;
    int col_id = id_in_head;
    // new (row, rol) of LtTrans COL32/COL4 sub-matrix, leading dim = (COL32_ * seq_len) or (COL32_ * seq_len_padded)
    int new_col = col_id >> 5;
    int new_row;
    if (use_ORDER_COL32_2R_4R4) {
        int row_in_tile = row_id & 31;
        int col_in_tile = col_id & 31;
        new_row = (qk_id != 1) ?
                      // COL32
                      ((row_id << 5) + (col_id & 31)) :
                      // COL32_2R_4R4
                      (((row_id >> 5) << 10) +
                       //(((row%8)/2*4+row/8)*2+row%2)*32+col
                       (((((((row_in_tile % 8) >> 1) << 2) + (row_in_tile >> 3)) << 1) + (row_in_tile & 1)) << 5)
                       + col_in_tile);
    }
    else {
        new_row = (qk_id != 1) ?
                      // COL32
                      ((row_id << 5) + (col_id & 31)) :
                      // COL4
                      ////row_id/8 is the number of tile of (8 rows 32 columns) -- column-major
                      ////row_id%2 is even row, otherwise odd row
                      ////col_id%COL32_/8 is the number tile of (8 rows 8 columns)
                      (((((row_id >> 3) << 3) + ((row_id & 1) << 2) + ((col_id % 32) >> 3)) << 5) +
                       ////col_id%8 >= 4 is the right half of (8 rows 8 columns) tile
                       ////(row_id%8/2) is (the row id of alternating 4 rows) - 1
                       (((((col_id & 7) >= 4) ? 4 : 0) + ((row_id % 8) >> 1)) << 2) +
                       ////col_id%4 is the id of 4 cols
                       (col_id & 3));
    }

    const int stride = (qk_id != 1) ? stride_q : stride_k;
    const int len = (qk_id != 1) ? seq_len : seq_len_padded;
    buf_ptr4[(((batch_id * head_num + head_id) * stride + (new_col << 5) * len + new_row) >> 2)] = tmp4;
}

template<typename T>
void invokeAddQKBiasTransformRebuildPaddingRow(int8_t* q_buf,
                                               int8_t* k_buf,
                                               const int8_t* Q,
                                               const T* bias_Q,
                                               const int8_t* K,
                                               const T* bias_K,
                                               const int* sequence_id_offset,
                                               const int valid_word_num,
                                               const int batch_size,
                                               const int seq_len,
                                               const int head_num,
                                               const int size_per_head,
                                               const float* q_deQFactor_ptr,
                                               const float* k_deQFactor_ptr,
                                               const float* q_output_scale_ptr,
                                               const float* k_output_scale_ptr,
                                               bool use_ORDER_COL32_2R_4R4,
                                               cudaStream_t stream)
{
    int seq_len_padded = (seq_len + 31) / 32 * 32;
    add_QK_bias_transform_rebuild_padding_varlen_row<<<dim3(valid_word_num * 2),
                                                       dim3((head_num * size_per_head) / 4),
                                                       0,
                                                       stream>>>(q_buf,
                                                                 k_buf,
                                                                 Q,
                                                                 bias_Q,
                                                                 K,
                                                                 bias_K,
                                                                 sequence_id_offset,
                                                                 valid_word_num,
                                                                 batch_size * seq_len,
                                                                 batch_size,
                                                                 seq_len,
                                                                 seq_len_padded,
                                                                 head_num,
                                                                 size_per_head,
                                                                 seq_len * size_per_head,
                                                                 seq_len_padded * size_per_head,
                                                                 q_deQFactor_ptr,
                                                                 k_deQFactor_ptr,
                                                                 q_output_scale_ptr,
                                                                 k_output_scale_ptr,
                                                                 use_ORDER_COL32_2R_4R4,
                                                                 head_num * size_per_head);
}

template void invokeAddQKBiasTransformRebuildPaddingRow(int8_t* q_buf,
                                                        int8_t* k_buf,
                                                        const int8_t* Q,
                                                        const float* bias_Q,
                                                        const int8_t* K,
                                                        const float* bias_K,
                                                        const int* sequence_id_offset,
                                                        const int valid_word_num,
                                                        const int batch_size,
                                                        const int seq_len,
                                                        const int head_num,
                                                        const int size_per_head,
                                                        const float* q_deQFactor_ptr,
                                                        const float* k_deQFactor_ptr,
                                                        const float* q_output_scale_ptr,
                                                        const float* k_output_scale_ptr,
                                                        bool use_ORDER_COL32_2R_4R4,
                                                        cudaStream_t stream);

template void invokeAddQKBiasTransformRebuildPaddingRow(int8_t* q_buf,
                                                        int8_t* k_buf,
                                                        const int8_t* Q,
                                                        const half* bias_Q,
                                                        const int8_t* K,
                                                        const half* bias_K,
                                                        const int* sequence_id_offset,
                                                        const int valid_word_num,
                                                        const int batch_size,
                                                        const int seq_len,
                                                        const int head_num,
                                                        const int size_per_head,
                                                        const float* q_deQFactor_ptr,
                                                        const float* k_deQFactor_ptr,
                                                        const float* q_output_scale_ptr,
                                                        const float* k_output_scale_ptr,
                                                        bool use_ORDER_COL32_2R_4R4,
                                                        cudaStream_t stream);

// input matrix a matrix of m = batch_size*seq_len , n = head_num*size_per_head, CUBLASLT_ORDER_COL32
// output matrixes are a series of sub-matrixes with size of m = size_per_head, n = seq_len , CUBLASLT_ORDER_COL4_4R2_8C
// or CUBLASLT_ORDER_COL32_2R_4R4 only for int32_t Input int8_t Output seq_len, size_per_head must be a multiple of 32
// grid = (size_per_head/32, seq_len/32, batch_size*head_num)
// block = (8, 32);
// using char4
// per axis quantization for weight
template<typename T>
__global__ void add_V_bias_transform(int8_t* v_buf_,
                                     const int32_t* V,
                                     const T* V_bias,
                                     const int batch_size,
                                     const int seq_len,
                                     const int head_num,
                                     const int size_per_head,
                                     int stride,
                                     const float* weight_amax,
                                     const float* input_deQFactor_div127_ptr,
                                     const float* out_scale_ptr,
                                     bool use_ORDER_COL32_2R_4R4)
{
    const float input_deQFactor_div127 = __ldg(input_deQFactor_div127_ptr);
    const float out_scale = __ldg(out_scale_ptr);
    __shared__ int8_t shm[32][33];
    const int32_t* data_ptr = V;
    char4* buf_ptr4 = (char4*)v_buf_;
    const T* bias_ptr = V_bias;

    int threadIdx4 = threadIdx.x << 2;

    // for src of (seq_len, size_per_head)
    int batch_id = blockIdx.z / head_num;
    int head_id = blockIdx.z % head_num;
    int word_id = (blockIdx.y << 5) + threadIdx.y;
    int id_in_size = (blockIdx.x << 5) + threadIdx4;

    // for V layout (batch_size*seq_len, head_num*size_per_head)
    int col = head_id * size_per_head + id_in_size;
    int row = batch_id * seq_len + word_id;
    int inIdx = (((col >> 5) << 5) * batch_size * seq_len + ((row << 5) + (col & 31)));
    // for shm row-major
    int sh_col = threadIdx4;
    int sh_row = threadIdx.y;

    float tmp;
    float scale;

    // const half2* bias_ptr2 = (const half2*)bias_ptr;
    // half2 tmp2;

    // tmp2 = __ldg(&bias_ptr2[col >> 1]);

    scale = __ldg(data_ptr + inIdx) * __ldg(weight_amax + col) * input_deQFactor_div127;
    tmp = scale + static_cast<float>(__ldg(bias_ptr + col));  //(tmp2.x);
    shm[sh_row][sh_col] = float_to_int8_rn(tmp * out_scale);

    scale = __ldg(data_ptr + inIdx + 1) * __ldg(weight_amax + col + 1) * input_deQFactor_div127;
    tmp = scale + static_cast<float>(__ldg(bias_ptr + col + 1));  //(tmp2.y);
    shm[sh_row][sh_col + 1] = float_to_int8_rn(tmp * out_scale);

    // tmp2 = __ldg(&bias_ptr2[(col >> 1) + 1]);

    scale = __ldg(data_ptr + inIdx + 2) * __ldg(weight_amax + col + 2) * input_deQFactor_div127;
    tmp = scale + static_cast<float>(__ldg(bias_ptr + col + 2));  //(tmp2.x);
    shm[sh_row][sh_col + 2] = float_to_int8_rn(tmp * out_scale);

    scale = __ldg(data_ptr + inIdx + 3) * __ldg(weight_amax + col + 3) * input_deQFactor_div127;
    tmp = scale + static_cast<float>(__ldg(bias_ptr + col + 3));  //(tmp2.y);
    shm[sh_row][sh_col + 3] = float_to_int8_rn(tmp * out_scale);

    __syncthreads();

    // for dst of (size_per_head, seq_len)
    word_id = (blockIdx.y << 5) + threadIdx4;
    id_in_size = (blockIdx.x << 5) + threadIdx.y;
    col = (word_id >> 5);

    if (use_ORDER_COL32_2R_4R4) {
        int row_in_tile = id_in_size & 31;
        int col_in_tile = word_id & 31;
        row = (
            // COL32_2R_4R4
            ((id_in_size >> 5) << 10) +
            //(((row%8)/2*4+row/8)*2+row%2)*32+col
            (((((((row_in_tile % 8) >> 1) << 2) + (row_in_tile >> 3)) << 1) + (row_in_tile & 1)) << 5) + col_in_tile);
    }
    else {
        row = (
            // COL4
            ////id_in_size/8 is the number of tile of (8 rows 32 columns) -- column-major
            ////id_in_size%2 is even row, otherwise odd row
            ////word_id%COL32_/8 is the number tile of (8 rows 8 columns)
            ((((id_in_size >> 3) << 3) + ((id_in_size & 1) << 2) + ((word_id % 32) >> 3)) << 5) +
            ////word_id%8 >= 4 is the right half of (8 rows 8 columns) tile
            ////(id_in_size%8/2) is (the row id of alternating 4 rows) - 1
            (((((word_id & 7) >= 4) ? 4 : 0) + ((id_in_size % 8) >> 1)) << 2) +
            ////word_id%4 is the id of 4 cols
            (word_id & 3));
    }

    char4 dataTmp;
    dataTmp.x = shm[sh_col][sh_row];
    dataTmp.y = shm[sh_col + 1][sh_row];
    dataTmp.z = shm[sh_col + 2][sh_row];
    dataTmp.w = shm[sh_col + 3][sh_row];
    buf_ptr4[(blockIdx.z * stride + (col << 5) * size_per_head + row) >> 2] = dataTmp;
}

template<>
__global__ void add_V_bias_transform(int8_t* v_buf_,
                                     const int32_t* V,
                                     const half* V_bias,
                                     const int batch_size,
                                     const int seq_len,
                                     const int head_num,
                                     const int size_per_head,
                                     int stride,
                                     const float* weight_amax,
                                     const float* input_deQFactor_div127_ptr,
                                     const float* out_scale_ptr,
                                     bool use_ORDER_COL32_2R_4R4)
{
    const float input_deQFactor_div127 = __ldg(input_deQFactor_div127_ptr);
    const float out_scale = __ldg(out_scale_ptr);
    __shared__ int8_t shm[32][33];
    const int32_t* data_ptr = V;
    char4* buf_ptr4 = (char4*)v_buf_;

    int threadIdx4 = threadIdx.x << 2;

    // for src of (seq_len, size_per_head)
    int batch_id = blockIdx.z / head_num;
    int head_id = blockIdx.z % head_num;

    int blockIdy32 = (blockIdx.y << 5);
    int blockIdx32 = (blockIdx.x << 5);
    int word_id = blockIdy32 + threadIdx.y;
    int id_in_size = blockIdx32 + threadIdx4;

    // for V layout (batch_size*seq_len, head_num*size_per_head)
    int col = head_id * size_per_head + id_in_size;
    int row = batch_id * seq_len + word_id;
    int inIdx = ((col & 0xffffffe0) * batch_size * seq_len + ((row << 5) + (col & 31)));
    // for shm row-major
    int sh_col = threadIdx4;
    int sh_row = threadIdx.y;

    int col_2 = col >> 1;
    float scale;

    const half2* bias_ptr2 = (const half2*)V_bias;
    half2 tmp2;

    tmp2 = __ldg(bias_ptr2 + col_2);

    scale = __ldg(data_ptr + inIdx) * __ldg(weight_amax + col) * input_deQFactor_div127;
    scale = scale + static_cast<float>(tmp2.x);
    shm[sh_row][sh_col] = float_to_int8_rn(scale * out_scale);

    scale = __ldg(data_ptr + inIdx + 1) * __ldg(weight_amax + col + 1) * input_deQFactor_div127;
    scale = scale + static_cast<float>(tmp2.y);
    shm[sh_row][sh_col + 1] = float_to_int8_rn(scale * out_scale);

    tmp2 = __ldg(bias_ptr2 + col_2 + 1);

    scale = __ldg(data_ptr + inIdx + 2) * __ldg(weight_amax + col + 2) * input_deQFactor_div127;
    scale = scale + static_cast<float>(tmp2.x);
    shm[sh_row][sh_col + 2] = float_to_int8_rn(scale * out_scale);

    scale = __ldg(data_ptr + inIdx + 3) * __ldg(weight_amax + col + 3) * input_deQFactor_div127;
    scale = scale + static_cast<float>(tmp2.y);
    shm[sh_row][sh_col + 3] = float_to_int8_rn(scale * out_scale);

    __syncthreads();

    // for dst of (size_per_head, seq_len)
    word_id = blockIdy32 + threadIdx4;
    id_in_size = blockIdx32 + threadIdx.y;
    col = (word_id >> 5);

    if (use_ORDER_COL32_2R_4R4) {
        int row_in_tile = id_in_size & 31;
        int col_in_tile = word_id & 31;
        row = (
            // COL32_2R_4R4
            ((id_in_size >> 5) << 10) +
            //(((row%8)/2*4+row/8)*2+row%2)*32+col
            (((((((row_in_tile % 8) >> 1) << 2) + (row_in_tile >> 3)) << 1) + (row_in_tile & 1)) << 5) + col_in_tile);
    }
    else {
        row = (
            // COL4
            ////id_in_size/8 is the number of tile of (8 rows 32 columns) -- column-major
            ////id_in_size%2 is even row, otherwise odd row
            ////word_id%COL32_/8 is the number tile of (8 rows 8 columns)
            (((id_in_size & 0xfffffff8) + ((id_in_size & 1) << 2) + ((word_id % 32) >> 3)) << 5) +
            ////word_id%8 >= 4 is the right half of (8 rows 8 columns) tile
            ////(id_in_size%8/2) is (the row id of alternating 4 rows) - 1
            (((((word_id & 7) >= 4) ? 4 : 0) + ((id_in_size % 8) >> 1)) << 2) +
            ////word_id%4 is the id of 4 cols
            (word_id & 3));
    }

    char4 dataTmp;
    dataTmp.x = shm[sh_col][sh_row];
    dataTmp.y = shm[sh_col + 1][sh_row];
    dataTmp.z = shm[sh_col + 2][sh_row];
    dataTmp.w = shm[sh_col + 3][sh_row];
    buf_ptr4[(blockIdx.z * stride + (col << 5) * size_per_head + row) >> 2] = dataTmp;
}

template<typename T>
__global__ void add_V_bias_transform_varlen(int8_t* v_buf_,
                                            const int32_t* V,
                                            const T* V_bias,
                                            const int batch_size,
                                            const int seq_len,
                                            const int head_num,
                                            const int size_per_head,
                                            int stride,
                                            const float* weight_amax,
                                            const float* input_deQFactor_div127_ptr,
                                            const float* out_scale_ptr,
                                            bool use_ORDER_COL32_2R_4R4)
{
    const float input_deQFactor_div127 = __ldg(input_deQFactor_div127_ptr);
    const float out_scale = __ldg(out_scale_ptr);
    __shared__ int8_t shm[32][33];
    const int32_t* data_ptr = V;
    char4* buf_ptr4 = (char4*)v_buf_;

    int threadIdx4 = threadIdx.x << 2;

    // for src of (seq_len, size_per_head)
    int batch_id = blockIdx.z / head_num;
    int head_id = blockIdx.z % head_num;

    int blockIdy32 = (blockIdx.y << 5);
    int blockIdx32 = (blockIdx.x << 5);
    int word_id = blockIdy32 + threadIdx.y;
    int id_in_size = blockIdx32 + threadIdx4;

    // for V layout (batch_size*seq_len, head_num*size_per_head)
    int col = head_id * size_per_head + id_in_size;
    int row = batch_id * seq_len + word_id;
    int inIdx = ((col & 0xffffffe0) * batch_size * seq_len + ((row << 5) + (col & 31)));
    // for shm row-major
    int sh_col = threadIdx4;
    int sh_row = threadIdx.y;

    float scale;

    if (word_id < seq_len) {
        const T* bias_ptr = V_bias;

        scale = __ldg(data_ptr + inIdx) * __ldg(weight_amax + col) * input_deQFactor_div127;
        scale = scale + static_cast<float>(__ldg(bias_ptr + col));
        shm[sh_row][sh_col] = float_to_int8_rn(scale * out_scale);

        scale = __ldg(data_ptr + inIdx + 1) * __ldg(weight_amax + col + 1) * input_deQFactor_div127;
        scale = scale + static_cast<float>(__ldg(bias_ptr + col + 1));
        shm[sh_row][sh_col + 1] = float_to_int8_rn(scale * out_scale);

        scale = __ldg(data_ptr + inIdx + 2) * __ldg(weight_amax + col + 2) * input_deQFactor_div127;
        scale = scale + static_cast<float>(__ldg(bias_ptr + col + 2));
        shm[sh_row][sh_col + 2] = float_to_int8_rn(scale * out_scale);

        scale = __ldg(data_ptr + inIdx + 3) * __ldg(weight_amax + col + 3) * input_deQFactor_div127;
        scale = scale + static_cast<float>(__ldg(bias_ptr + col + 3));
        shm[sh_row][sh_col + 3] = float_to_int8_rn(scale * out_scale);
    }
    else {
        shm[sh_row][sh_col] = shm[sh_row][sh_col + 1] = shm[sh_row][sh_col + 2] = shm[sh_row][sh_col + 3] = 0;
    }

    __syncthreads();

    // for dst of (size_per_head, seq_len)
    word_id = blockIdy32 + threadIdx4;
    id_in_size = blockIdx32 + threadIdx.y;
    col = (word_id >> 5);

    if (use_ORDER_COL32_2R_4R4) {
        int row_in_tile = id_in_size & 31;
        int col_in_tile = word_id & 31;
        row = (
            // COL32_2R_4R4
            ((id_in_size >> 5) << 10) +
            //(((row%8)/2*4+row/8)*2+row%2)*32+col
            (((((((row_in_tile % 8) >> 1) << 2) + (row_in_tile >> 3)) << 1) + (row_in_tile & 1)) << 5) + col_in_tile);
    }
    else {
        row = (
            // COL4
            ////id_in_size/8 is the number of tile of (8 rows 32 columns) -- column-major
            ////id_in_size%2 is even row, otherwise odd row
            ////word_id%COL32_/8 is the number tile of (8 rows 8 columns)
            (((id_in_size & 0xfffffff8) + ((id_in_size & 1) << 2) + ((word_id % 32) >> 3)) << 5) +
            ////word_id%8 >= 4 is the right half of (8 rows 8 columns) tile
            ////(id_in_size%8/2) is (the row id of alternating 4 rows) - 1
            (((((word_id & 7) >= 4) ? 4 : 0) + ((id_in_size % 8) >> 1)) << 2) +
            ////word_id%4 is the id of 4 cols
            (word_id & 3));
    }

    char4 dataTmp;
    dataTmp.x = shm[sh_col][sh_row];
    dataTmp.y = shm[sh_col + 1][sh_row];
    dataTmp.z = shm[sh_col + 2][sh_row];
    dataTmp.w = shm[sh_col + 3][sh_row];
    buf_ptr4[(blockIdx.z * stride + (col << 5) * size_per_head + row) >> 2] = dataTmp;
}

template<typename T>
void invokeAddVBiasTransform(int8_t* v_buf,
                             const int32_t* V,
                             const T* V_bias,
                             const int batch_size,
                             const int seq_len,
                             const int head_num,
                             const int size_per_head,
                             const float* weight_amax,
                             const float* input_deQFactor_div127_ptr,
                             const float* out_scale_ptr,
                             bool use_ORDER_COL32_2R_4R4,
                             cudaStream_t stream)
{
    if (seq_len % 32 == 0) {
        add_V_bias_transform<<<dim3(size_per_head / 32, seq_len / 32, batch_size * head_num), dim3(8, 32), 0, stream>>>(
            v_buf,
            V,
            V_bias,
            batch_size,
            seq_len,
            head_num,
            size_per_head,
            seq_len * size_per_head,
            weight_amax,
            input_deQFactor_div127_ptr,
            out_scale_ptr,
            use_ORDER_COL32_2R_4R4);
    }
    else {
        const int seq_len_padded = (seq_len + 31) / 32 * 32;
        add_V_bias_transform_varlen<<<dim3(size_per_head / 32, seq_len_padded / 32, batch_size * head_num),
                                      dim3(8, 32),
                                      0,
                                      stream>>>(v_buf,
                                                V,
                                                V_bias,
                                                batch_size,
                                                seq_len,
                                                head_num,
                                                size_per_head,
                                                seq_len_padded * size_per_head,
                                                weight_amax,
                                                input_deQFactor_div127_ptr,
                                                out_scale_ptr,
                                                use_ORDER_COL32_2R_4R4);
    }
}

template void invokeAddVBiasTransform(int8_t* v_buf,
                                      const int32_t* V,
                                      const float* V_bias,
                                      const int batch_size,
                                      const int seq_len,
                                      const int head_num,
                                      const int size_per_head,
                                      const float* weight_amax,
                                      const float* input_deQFactor_div127_ptr,
                                      const float* out_scale_ptr,
                                      bool use_ORDER_COL32_2R_4R4,
                                      cudaStream_t stream);

template void invokeAddVBiasTransform(int8_t* v_buf,
                                      const int32_t* V,
                                      const half* V_bias,
                                      const int batch_size,
                                      const int seq_len,
                                      const int head_num,
                                      const int size_per_head,
                                      const float* weight_amax,
                                      const float* input_deQFactor_div127_ptr,
                                      const float* out_scale_ptr,
                                      bool use_ORDER_COL32_2R_4R4,
                                      cudaStream_t stream);

// input matrix a matrix of m = batch_size*seq_len , n = head_num*size_per_head, CUBLASLT_ORDER_COL32
// seq_len_padded = (seq_len+31)/32*32
// output matrixes are a series of sub-matrixes with size of m = size_per_head, n = seq_len_padded ,
// CUBLASLT_ORDER_COL4_4R2_8C or CUBLASLT_ORDER_COL32_2R_4R4 only for int8_t IO size_per_head must be a multiple of 32
// grid = (size_per_head/32, seq_len_padded/32, batch_size*head_num)
// block = (8, 32);
// using char4
// per tensor quantization for weight
template<typename T>
__global__ void add_V_bias_transform_varlen(int8_t* v_buf_,
                                            const int8_t* V,
                                            const T* V_bias,
                                            const int batch_size,
                                            const int seq_len,
                                            const int head_num,
                                            const int size_per_head,
                                            const int seq_len_padded,
                                            int stride,
                                            const float* input_deQFactor_ptr,
                                            const float* out_scale_ptr,
                                            bool use_ORDER_COL32_2R_4R4)
{
    const float input_deQFactor = __ldg(input_deQFactor_ptr);
    const float out_scale = __ldg(out_scale_ptr);
    __shared__ int8_t shm[32][33];
    const char4* data_ptr = (const char4*)V;
    char4* buf_ptr4 = (char4*)v_buf_;
    const T* bias_ptr = V_bias;

    int threadIdx4 = threadIdx.x << 2;

    // for src of (seq_len, size_per_head)
    int batch_id = blockIdx.z / head_num;
    int head_id = blockIdx.z % head_num;
    int word_id = (blockIdx.y << 5) + threadIdx.y;
    int id_in_size = (blockIdx.x << 5) + threadIdx4;

    int col, row;
    // for shm row-major
    int sh_col = threadIdx4;
    int sh_row = threadIdx.y;
    char4 dataTmp;
    if (word_id < seq_len) {
        // for V layout (batch_size*seq_len, head_num*size_per_head)
        col = head_id * size_per_head + id_in_size;
        row = batch_id * seq_len + word_id;
        int inIdx = (((col >> 5) << 5) * batch_size * seq_len + ((row << 5) + (col & 31))) >> 2;

        float tmp;
        float scale;

        dataTmp = __ldg(data_ptr + inIdx);

        scale = dataTmp.x * input_deQFactor;
        tmp = scale + static_cast<float>(__ldg(bias_ptr + col));  //(tmp2.x);
        shm[sh_row][sh_col] = float_to_int8_rn(tmp * out_scale);

        scale = dataTmp.y * input_deQFactor;
        tmp = scale + static_cast<float>(__ldg(bias_ptr + col + 1));  //(tmp2.y);
        shm[sh_row][sh_col + 1] = float_to_int8_rn(tmp * out_scale);

        scale = dataTmp.z * input_deQFactor;
        tmp = scale + static_cast<float>(__ldg(bias_ptr + col + 2));  //(tmp2.x);
        shm[sh_row][sh_col + 2] = float_to_int8_rn(tmp * out_scale);

        scale = dataTmp.w * input_deQFactor;
        tmp = scale + static_cast<float>(__ldg(bias_ptr + col + 3));  //(tmp2.y);
        shm[sh_row][sh_col + 3] = float_to_int8_rn(tmp * out_scale);
    }
    else {
        shm[sh_row][sh_col] = shm[sh_row][sh_col + 1] = shm[sh_row][sh_col + 2] = shm[sh_row][sh_col + 3] = 0;
    }

    __syncthreads();

    // for dst of (size_per_head, seq_len_padded)
    word_id = (blockIdx.y << 5) + threadIdx4;
    id_in_size = (blockIdx.x << 5) + threadIdx.y;
    col = (word_id >> 5);

    if (use_ORDER_COL32_2R_4R4) {
        int row_in_tile = id_in_size & 31;
        int col_in_tile = word_id & 31;
        row = (
            // COL32_2R_4R4
            ((id_in_size >> 5) << 10) +
            //(((row%8)/2*4+row/8)*2+row%2)*32+col
            (((((((row_in_tile % 8) >> 1) << 2) + (row_in_tile >> 3)) << 1) + (row_in_tile & 1)) << 5) + col_in_tile);
    }
    else {
        row = (
            // COL4
            ////id_in_size/8 is the number of tile of (8 rows 32 columns) -- column-major
            ////id_in_size%2 is even row, otherwise odd row
            ////word_id%COL32_/8 is the number tile of (8 rows 8 columns)
            ((((id_in_size >> 3) << 3) + ((id_in_size & 1) << 2) + ((word_id % 32) >> 3)) << 5) +
            ////word_id%8 >= 4 is the right half of (8 rows 8 columns) tile
            ////(id_in_size%8/2) is (the row id of alternating 4 rows) - 1
            (((((word_id & 7) >= 4) ? 4 : 0) + ((id_in_size % 8) >> 1)) << 2) +
            ////word_id%4 is the id of 4 cols
            (word_id & 3));
    }

    dataTmp.x = shm[sh_col][sh_row];
    dataTmp.y = shm[sh_col + 1][sh_row];
    dataTmp.z = shm[sh_col + 2][sh_row];
    dataTmp.w = shm[sh_col + 3][sh_row];
    buf_ptr4[(blockIdx.z * stride + (col << 5) * size_per_head + row) >> 2] = dataTmp;
}

template<typename T>
void invokeAddVBiasTransform(int8_t* v_buf,
                             const int8_t* V,
                             const T* V_bias,
                             const int batch_size,
                             const int seq_len,
                             const int head_num,
                             const int size_per_head,
                             const float* input_deQFactor_ptr,
                             const float* out_scale_ptr,
                             bool use_ORDER_COL32_2R_4R4,
                             cudaStream_t stream)
{
    assert(size_per_head % 32 == 0);
    if (seq_len % 32 == 0) {
        add_V_bias_transform_varlen<<<dim3(size_per_head / 32, seq_len / 32, batch_size * head_num),
                                      dim3(8, 32),
                                      0,
                                      stream>>>(v_buf,
                                                V,
                                                V_bias,
                                                batch_size,
                                                seq_len,
                                                head_num,
                                                size_per_head,
                                                seq_len,
                                                seq_len * size_per_head,
                                                input_deQFactor_ptr,
                                                out_scale_ptr,
                                                use_ORDER_COL32_2R_4R4);
    }
    else {
        const int seq_len_padded = (seq_len + 31) / 32 * 32;
        add_V_bias_transform_varlen<<<dim3(size_per_head / 32, seq_len_padded / 32, batch_size * head_num),
                                      dim3(8, 32),
                                      0,
                                      stream>>>(v_buf,
                                                V,
                                                V_bias,
                                                batch_size,
                                                seq_len,
                                                head_num,
                                                size_per_head,
                                                seq_len_padded,
                                                seq_len_padded * size_per_head,
                                                input_deQFactor_ptr,
                                                out_scale_ptr,
                                                use_ORDER_COL32_2R_4R4);
    }
}

template void invokeAddVBiasTransform(int8_t* v_buf,
                                      const int8_t* V,
                                      const float* V_bias,
                                      const int batch_size,
                                      const int seq_len,
                                      const int head_num,
                                      const int size_per_head,
                                      const float* input_deQFactor_ptr,
                                      const float* out_scale_ptr,
                                      bool use_ORDER_COL32_2R_4R4,
                                      cudaStream_t stream);

template void invokeAddVBiasTransform(int8_t* v_buf,
                                      const int8_t* V,
                                      const half* V_bias,
                                      const int batch_size,
                                      const int seq_len,
                                      const int head_num,
                                      const int size_per_head,
                                      const float* input_deQFactor_ptr,
                                      const float* out_scale_ptr,
                                      bool use_ORDER_COL32_2R_4R4,
                                      cudaStream_t stream);
// input matrix a matrix of m = batch_size*seq_len , n = head_num*size_per_head, row major
// seq_len_padded = (seq_len+31)/32*32
// output matrixes are a series of sub-matrixes with size of m = size_per_head, n = seq_len_padded ,
// CUBLASLT_ORDER_COL4_4R2_8C or CUBLASLT_ORDER_COL32_2R_4R4 only for int8_t IO size_per_head must be a multiple of 32
// grid = (size_per_head/32, seq_len_padded/32, batch_size*head_num)
// block = (8, 32);
// using char4
// per tensor quantization for weight
template<typename T>
__global__ void add_V_bias_transform_varlen_row(int8_t* v_buf_,
                                                const int8_t* V,
                                                const T* V_bias,
                                                const int batch_size,
                                                const int seq_len,
                                                const int head_num,
                                                const int size_per_head,
                                                const int seq_len_padded,
                                                int stride,
                                                const float* input_deQFactor_ptr,
                                                const float* out_scale_ptr,
                                                bool use_ORDER_COL32_2R_4R4,
                                                const int head_num_x_size_per_head)
{
    const float input_deQFactor = __ldg(input_deQFactor_ptr);
    const float out_scale = __ldg(out_scale_ptr);
    __shared__ int8_t shm[32][33];
    const char4* data_ptr = (const char4*)V;
    char4* buf_ptr4 = (char4*)v_buf_;
    const T* bias_ptr = V_bias;

    int threadIdx4 = threadIdx.x << 2;

    // for src of (seq_len, size_per_head)
    int batch_id = blockIdx.z / head_num;
    int head_id = blockIdx.z % head_num;
    int word_id = (blockIdx.y << 5) + threadIdx.y;
    int id_in_size = (blockIdx.x << 5) + threadIdx4;

    int col, row;
    // for shm row-major
    int sh_col = threadIdx4;
    int sh_row = threadIdx.y;
    char4 dataTmp;
    if (word_id < seq_len) {
        // for V layout (batch_size*seq_len, head_num*size_per_head)
        col = head_id * size_per_head + id_in_size;
        row = batch_id * seq_len + word_id;
        int inIdx = (row * head_num_x_size_per_head + col) >> 2;

        float tmp;
        float scale;

        dataTmp = __ldg(data_ptr + inIdx);

        scale = dataTmp.x * input_deQFactor;
        tmp = scale + static_cast<float>(__ldg(bias_ptr + col));  //(tmp2.x);
        shm[sh_row][sh_col] = float_to_int8_rn(tmp * out_scale);

        scale = dataTmp.y * input_deQFactor;
        tmp = scale + static_cast<float>(__ldg(bias_ptr + col + 1));  //(tmp2.y);
        shm[sh_row][sh_col + 1] = float_to_int8_rn(tmp * out_scale);

        scale = dataTmp.z * input_deQFactor;
        tmp = scale + static_cast<float>(__ldg(bias_ptr + col + 2));  //(tmp2.x);
        shm[sh_row][sh_col + 2] = float_to_int8_rn(tmp * out_scale);

        scale = dataTmp.w * input_deQFactor;
        tmp = scale + static_cast<float>(__ldg(bias_ptr + col + 3));  //(tmp2.y);
        shm[sh_row][sh_col + 3] = float_to_int8_rn(tmp * out_scale);
    }
    else {
        shm[sh_row][sh_col] = shm[sh_row][sh_col + 1] = shm[sh_row][sh_col + 2] = shm[sh_row][sh_col + 3] = 0;
    }

    __syncthreads();

    // for dst of (size_per_head, seq_len_padded)
    word_id = (blockIdx.y << 5) + threadIdx4;
    id_in_size = (blockIdx.x << 5) + threadIdx.y;
    col = (word_id >> 5);

    if (use_ORDER_COL32_2R_4R4) {
        int row_in_tile = id_in_size & 31;
        int col_in_tile = word_id & 31;
        row = (
            // COL32_2R_4R4
            ((id_in_size >> 5) << 10)
            + (((((((row_in_tile % 8) / 2) * 4) + (row_in_tile / 8)) * 2) + (row_in_tile % 2)) * 32) + col_in_tile
            // (((((((row_in_tile%8)>>1)<<2)+(row_in_tile>>3))<<1)+(row_in_tile&1))<<5)+col_in_tile
        );
    }
    else {
        row = (
            // COL4
            ////id_in_size/8 is the number of tile of (8 rows 32 columns) -- column-major
            ////id_in_size%2 is even row, otherwise odd row
            ////word_id%COL32_/8 is the number tile of (8 rows 8 columns)
            ((((id_in_size >> 3) << 3) + ((id_in_size & 1) << 2) + ((word_id % 32) >> 3)) << 5) +
            ////word_id%8 >= 4 is the right half of (8 rows 8 columns) tile
            ////(id_in_size%8/2) is (the row id of alternating 4 rows) - 1
            (((((word_id & 7) >= 4) ? 4 : 0) + ((id_in_size % 8) >> 1)) << 2) +
            ////word_id%4 is the id of 4 cols
            (word_id & 3));
    }

    dataTmp.x = shm[sh_col][sh_row];
    dataTmp.y = shm[sh_col + 1][sh_row];
    dataTmp.z = shm[sh_col + 2][sh_row];
    dataTmp.w = shm[sh_col + 3][sh_row];
    buf_ptr4[(blockIdx.z * stride + (col << 5) * size_per_head + row) >> 2] = dataTmp;
}

template<typename T>
void invokeAddVBiasTransformRow(int8_t* v_buf,
                                const int8_t* V,
                                const T* V_bias,
                                const int batch_size,
                                const int seq_len,
                                const int head_num,
                                const int size_per_head,
                                const float* input_deQFactor_ptr,
                                const float* out_scale_ptr,
                                bool use_ORDER_COL32_2R_4R4,
                                cudaStream_t stream)
{
    assert(size_per_head % 32 == 0);
    if (seq_len % 32 == 0) {
        add_V_bias_transform_varlen_row<<<dim3(size_per_head / 32, seq_len / 32, batch_size * head_num),
                                          dim3(8, 32),
                                          0,
                                          stream>>>(v_buf,
                                                    V,
                                                    V_bias,
                                                    batch_size,
                                                    seq_len,
                                                    head_num,
                                                    size_per_head,
                                                    seq_len,
                                                    seq_len * size_per_head,
                                                    input_deQFactor_ptr,
                                                    out_scale_ptr,
                                                    use_ORDER_COL32_2R_4R4,
                                                    head_num * size_per_head);
    }
    else {
        const int seq_len_padded = (seq_len + 31) / 32 * 32;
        add_V_bias_transform_varlen_row<<<dim3(size_per_head / 32, seq_len_padded / 32, batch_size * head_num),
                                          dim3(8, 32),
                                          0,
                                          stream>>>(v_buf,
                                                    V,
                                                    V_bias,
                                                    batch_size,
                                                    seq_len,
                                                    head_num,
                                                    size_per_head,
                                                    seq_len_padded,
                                                    seq_len_padded * size_per_head,
                                                    input_deQFactor_ptr,
                                                    out_scale_ptr,
                                                    use_ORDER_COL32_2R_4R4,
                                                    head_num * size_per_head);
    }
}

template void invokeAddVBiasTransformRow(int8_t* v_buf,
                                         const int8_t* V,
                                         const float* V_bias,
                                         const int batch_size,
                                         const int seq_len,
                                         const int head_num,
                                         const int size_per_head,
                                         const float* input_deQFactor_ptr,
                                         const float* out_scale_ptr,
                                         bool use_ORDER_COL32_2R_4R4,
                                         cudaStream_t stream);

template void invokeAddVBiasTransformRow(int8_t* v_buf,
                                         const int8_t* V,
                                         const half* V_bias,
                                         const int batch_size,
                                         const int seq_len,
                                         const int head_num,
                                         const int size_per_head,
                                         const float* input_deQFactor_ptr,
                                         const float* out_scale_ptr,
                                         bool use_ORDER_COL32_2R_4R4,
                                         cudaStream_t stream);

// add bias into V & rebuild padding
// input matrix a matrix of m = valid_word_num, n = head_num*size_per_head, CUBLASLT_ORDER_COL32
// output matrixes are a series of sub-matrixes with size of m = size_per_head, n = seq_len , CUBLASLT_ORDER_COL4_4R2_8C
// or CUBLASLT_ORDER_COL32_2R_4R4 only for int32_t Input int8_t Output seq_len, size_per_head must be a multiple of 32
// grid = (size_per_head/32, seq_len/32, batch_size*head_num)
// block = (8, 32);
// using char4
// per axis quantization for weight
template<typename T>
__global__ void add_V_bias_transform_rebuild_padding(int8_t* v_buf_,
                                                     const int32_t* V,
                                                     const T* V_bias,
                                                     const int* sequence_id_map,
                                                     const int valid_word_num,
                                                     const int batch_size,
                                                     const int seq_len,
                                                     const int head_num,
                                                     const int size_per_head,
                                                     int stride,
                                                     const float* weight_amax,
                                                     const float* input_deQFactor_div127_ptr,
                                                     const float* out_scale_ptr,
                                                     bool use_ORDER_COL32_2R_4R4)
{
    __shared__ int8_t shm[32][33];
    const int32_t* data_ptr = V;
    char4* buf_ptr4 = (char4*)v_buf_;
    const T* bias_ptr = V_bias;

    int threadIdx4 = threadIdx.x << 2;

    // for src of (seq_len, size_per_head)
    int batch_id = blockIdx.z / head_num;
    int head_id = blockIdx.z % head_num;
    int word_id = (blockIdx.y << 5) + threadIdx.y;
    int id_in_size = (blockIdx.x << 5) + threadIdx4;

    // for shm row-major
    int sh_col = threadIdx4;
    int sh_row = threadIdx.y;

    // for V layout (batch_size*seq_len, head_num*size_per_head)
    int col;
    int row = __ldg(sequence_id_map + batch_id * seq_len + word_id);

    if (row != -1) {
        col = head_id * size_per_head + id_in_size;
        int inIdx = ((col & 0xffffffe0) * valid_word_num + ((row << 5) + (col & 31)));

        float tmp;
        float scale;

        const float input_deQFactor_div127 = __ldg(input_deQFactor_div127_ptr);
        const float out_scale = __ldg(out_scale_ptr);

        scale = __ldg(data_ptr + inIdx) * __ldg(weight_amax + col) * input_deQFactor_div127;
        tmp = scale + static_cast<float>(__ldg(bias_ptr + col));
        shm[sh_row][sh_col] = float_to_int8_rn(tmp * out_scale);

        scale = __ldg(data_ptr + inIdx + 1) * __ldg(weight_amax + col + 1) * input_deQFactor_div127;
        tmp = scale + static_cast<float>(__ldg(bias_ptr + col + 1));
        shm[sh_row][sh_col + 1] = float_to_int8_rn(tmp * out_scale);

        scale = __ldg(data_ptr + inIdx + 2) * __ldg(weight_amax + col + 2) * input_deQFactor_div127;
        tmp = scale + static_cast<float>(__ldg(bias_ptr + col + 2));
        shm[sh_row][sh_col + 2] = float_to_int8_rn(tmp * out_scale);

        scale = __ldg(data_ptr + inIdx + 3) * __ldg(weight_amax + col + 3) * input_deQFactor_div127;
        tmp = scale + static_cast<float>(__ldg(bias_ptr + col + 3));
        shm[sh_row][sh_col + 3] = float_to_int8_rn(tmp * out_scale);
    }
    else {
        shm[sh_row][sh_col] = shm[sh_row][sh_col + 1] = shm[sh_row][sh_col + 2] = shm[sh_row][sh_col + 3] = 0;
    }
    __syncthreads();

    char4 dataTmp;
    dataTmp.x = shm[sh_col][sh_row];
    dataTmp.y = shm[sh_col + 1][sh_row];
    dataTmp.z = shm[sh_col + 2][sh_row];
    dataTmp.w = shm[sh_col + 3][sh_row];

    // for dst of (size_per_head, seq_len)
    word_id = (blockIdx.y << 5) + threadIdx4;
    id_in_size = (blockIdx.x << 5) + threadIdx.y;
    col = (word_id >> 5);

    if (use_ORDER_COL32_2R_4R4) {
        int row_in_tile = id_in_size & 31;
        int col_in_tile = word_id & 31;
        row = (
            // COL32_2R_4R4
            ((id_in_size >> 5) << 10) +
            //(((row%8)/2*4+row/8)*2+row%2)*32+col
            (((((((row_in_tile % 8) >> 1) << 2) + (row_in_tile >> 3)) << 1) + (row_in_tile & 1)) << 5) + col_in_tile);
    }
    else {
        row = (
            // COL4
            ////id_in_size/8 is the number of tile of (8 rows 32 columns) -- column-major
            ////id_in_size%2 is even row, otherwise odd row
            ////word_id%COL32_/8 is the number tile of (8 rows 8 columns)
            (((id_in_size & 0xfffffff8) + ((id_in_size & 1) << 2) + ((word_id % 32) >> 3)) << 5) +
            ////word_id%8 >= 4 is the right half of (8 rows 8 columns) tile
            ////(id_in_size%8/2) is (the row id of alternating 4 rows) - 1
            (((((word_id & 7) >= 4) ? 4 : 0) + ((id_in_size % 8) >> 1)) << 2) +
            ////word_id%4 is the id of 4 cols
            (word_id & 3));
    }

    buf_ptr4[(blockIdx.z * stride + (col << 5) * size_per_head + row) >> 2] = dataTmp;
}

template<>
__global__ void add_V_bias_transform_rebuild_padding(int8_t* v_buf_,
                                                     const int32_t* V,
                                                     const half* V_bias,
                                                     const int* sequence_id_map,
                                                     const int valid_word_num,
                                                     const int batch_size,
                                                     const int seq_len,
                                                     const int head_num,
                                                     const int size_per_head,
                                                     int stride,
                                                     const float* weight_amax,
                                                     const float* input_deQFactor_div127_ptr,
                                                     const float* out_scale_ptr,
                                                     bool use_ORDER_COL32_2R_4R4)
{
    __shared__ int8_t shm[32][33];
    const int32_t* data_ptr = V;
    char4* buf_ptr4 = (char4*)v_buf_;

    int threadIdx4 = threadIdx.x << 2;

    // for src of (seq_len, size_per_head)
    int batch_id = blockIdx.z / head_num;
    int head_id = blockIdx.z % head_num;

    int blockIdy32 = (blockIdx.y << 5);
    int blockIdx32 = (blockIdx.x << 5);
    int word_id = blockIdy32 + threadIdx.y;
    int id_in_size = blockIdx32 + threadIdx4;

    // for shm row-major
    int sh_col = threadIdx4;
    int sh_row = threadIdx.y;

    // for V layout (batch_size*seq_len, head_num*size_per_head)
    int col;
    int row = __ldg(sequence_id_map + batch_id * seq_len + word_id);

    if (row >= 0) {
        const float input_deQFactor_div127 = __ldg(input_deQFactor_div127_ptr);
        const float out_scale = __ldg(out_scale_ptr);
        col = head_id * size_per_head + id_in_size;
        int inIdx = ((col & 0xffffffe0) * valid_word_num + ((row << 5) + (col & 31)));
        int col_2 = col >> 1;
        float scale;

        const half2* bias_ptr2 = (const half2*)V_bias;
        half2 tmp2;

        tmp2 = __ldg(bias_ptr2 + col_2);

        scale = __ldg(data_ptr + inIdx) * __ldg(weight_amax + col) * input_deQFactor_div127;
        scale = scale + static_cast<float>(tmp2.x);
        shm[sh_row][sh_col] = float_to_int8_rn(scale * out_scale);

        scale = __ldg(data_ptr + inIdx + 1) * __ldg(weight_amax + col + 1) * input_deQFactor_div127;
        scale = scale + static_cast<float>(tmp2.y);
        shm[sh_row][sh_col + 1] = float_to_int8_rn(scale * out_scale);

        tmp2 = __ldg(bias_ptr2 + col_2 + 1);

        scale = __ldg(data_ptr + inIdx + 2) * __ldg(weight_amax + col + 2) * input_deQFactor_div127;
        scale = scale + static_cast<float>(tmp2.x);
        shm[sh_row][sh_col + 2] = float_to_int8_rn(scale * out_scale);

        scale = __ldg(data_ptr + inIdx + 3) * __ldg(weight_amax + col + 3) * input_deQFactor_div127;
        scale = scale + static_cast<float>(tmp2.y);
        shm[sh_row][sh_col + 3] = float_to_int8_rn(scale * out_scale);
    }
    else {
        shm[sh_row][sh_col] = shm[sh_row][sh_col + 1] = shm[sh_row][sh_col + 2] = shm[sh_row][sh_col + 3] = 0;
    }
    __syncthreads();

    char4 dataTmp;
    dataTmp.x = shm[sh_col][sh_row];
    dataTmp.y = shm[sh_col + 1][sh_row];
    dataTmp.z = shm[sh_col + 2][sh_row];
    dataTmp.w = shm[sh_col + 3][sh_row];

    // for dst of (size_per_head, seq_len)
    word_id = blockIdy32 + threadIdx4;
    id_in_size = blockIdx32 + threadIdx.y;
    col = (word_id >> 5);

    if (use_ORDER_COL32_2R_4R4) {
        int row_in_tile = id_in_size & 31;
        int col_in_tile = word_id & 31;
        row = (
            // COL32_2R_4R4
            ((id_in_size >> 5) << 10) +
            //(((row%8)/2*4+row/8)*2+row%2)*32+col
            (((((((row_in_tile % 8) >> 1) << 2) + (row_in_tile >> 3)) << 1) + (row_in_tile & 1)) << 5) + col_in_tile);
    }
    else {
        row = (
            // COL4
            ////id_in_size/8 is the number of tile of (8 rows 32 columns) -- column-major
            ////id_in_size%2 is even row, otherwise odd row
            ////word_id%COL32_/8 is the number tile of (8 rows 8 columns)
            (((id_in_size & 0xfffffff8) + ((id_in_size & 1) << 2) + ((word_id % 32) >> 3)) << 5) +
            ////word_id%8 >= 4 is the right half of (8 rows 8 columns) tile
            ////(id_in_size%8/2) is (the row id of alternating 4 rows) - 1
            (((((word_id & 7) >= 4) ? 4 : 0) + ((id_in_size % 8) >> 1)) << 2) +
            ////word_id%4 is the id of 4 cols
            (word_id & 3));
    }

    buf_ptr4[(blockIdx.z * stride + (col << 5) * size_per_head + row) >> 2] = dataTmp;
}

template<typename T>
void invokeAddVBiasTransformRebuildPadding(int8_t* v_buf,
                                           const int32_t* V,
                                           const T* V_bias,
                                           const int* sequence_id_map,
                                           const int valid_word_num,
                                           const int batch_size,
                                           const int seq_len,
                                           const int head_num,
                                           const int size_per_head,
                                           const float* weight_amax,
                                           const float* input_deQFactor_div127_ptr,
                                           const float* out_scale_ptr,
                                           bool use_ORDER_COL32_2R_4R4,
                                           cudaStream_t stream)
{
    add_V_bias_transform_rebuild_padding<<<dim3(size_per_head / 32, seq_len / 32, batch_size * head_num),
                                           dim3(8, 32),
                                           0,
                                           stream>>>(v_buf,
                                                     V,
                                                     V_bias,
                                                     sequence_id_map,
                                                     valid_word_num,
                                                     batch_size,
                                                     seq_len,
                                                     head_num,
                                                     size_per_head,
                                                     seq_len * size_per_head,
                                                     weight_amax,
                                                     input_deQFactor_div127_ptr,
                                                     out_scale_ptr,
                                                     use_ORDER_COL32_2R_4R4);
}

template void invokeAddVBiasTransformRebuildPadding(int8_t* v_buf,
                                                    const int32_t* V,
                                                    const float* V_bias,
                                                    const int* sequence_id_map,
                                                    const int valid_word_num,
                                                    const int batch_size,
                                                    const int seq_len,
                                                    const int head_num,
                                                    const int size_per_head,
                                                    const float* weight_amax,
                                                    const float* input_deQFactor_div127_ptr,
                                                    const float* out_scale_ptr,
                                                    bool use_ORDER_COL32_2R_4R4,
                                                    cudaStream_t stream);

template void invokeAddVBiasTransformRebuildPadding(int8_t* v_buf,
                                                    const int32_t* V,
                                                    const half* V_bias,
                                                    const int* sequence_id_map,
                                                    const int valid_word_num,
                                                    const int batch_size,
                                                    const int seq_len,
                                                    const int head_num,
                                                    const int size_per_head,
                                                    const float* weight_amax,
                                                    const float* input_deQFactor_div127_ptr,
                                                    const float* out_scale_ptr,
                                                    bool use_ORDER_COL32_2R_4R4,
                                                    cudaStream_t stream);

// add bias into V & rebuild padding
// input matrix a matrix of m = valid_word_num, n = head_num*size_per_head, CUBLASLT_ORDER_COL32
// output matrixes are a series of sub-matrixes with size of m = size_per_head, n = seq_len , CUBLASLT_ORDER_COL4_4R2_8C
// or CUBLASLT_ORDER_COL32_2R_4R4 only for int8_t IO seq_len, size_per_head must be a multiple of 32 grid =
// (size_per_head/32, seq_len/32, batch_size*head_num) block = (8, 32); using char4 per tensor quantization for weight
template<typename T>
__global__ void add_V_bias_transform_rebuild_padding(int8_t* v_buf_,
                                                     const int8_t* V,
                                                     const T* V_bias,
                                                     const int* sequence_id_map,
                                                     const int valid_word_num,
                                                     const int batch_size,
                                                     const int seq_len,
                                                     const int head_num,
                                                     const int size_per_head,
                                                     int stride,
                                                     const float* deQFactor_ptr,
                                                     const float* out_scale_ptr,
                                                     bool use_ORDER_COL32_2R_4R4)
{
    __shared__ int8_t shm[32][33];
    const char4* data_ptr = (const char4*)V;
    char4* buf_ptr4 = (char4*)v_buf_;
    const T* bias_ptr = V_bias;

    int threadIdx4 = threadIdx.x << 2;

    // for src of (seq_len, size_per_head)
    int batch_id = blockIdx.z / head_num;
    int head_id = blockIdx.z % head_num;
    int word_id = (blockIdx.y << 5) + threadIdx.y;
    int id_in_size = (blockIdx.x << 5) + threadIdx4;

    // for shm row-major
    int sh_col = threadIdx4;
    int sh_row = threadIdx.y;

    // for V layout (batch_size*seq_len, head_num*size_per_head)
    int col;
    int row = __ldg(sequence_id_map + batch_id * seq_len + word_id);

    if (row != -1) {
        col = head_id * size_per_head + id_in_size;
        int inIdx = ((col & 0xffffffe0) * valid_word_num + ((row << 5) + (col & 31))) >> 2;

        float tmp;
        float scale;

        const float deQFactor = __ldg(deQFactor_ptr);
        const float out_scale = __ldg(out_scale_ptr);

        char4 dataTmp = __ldg(data_ptr + inIdx);

        scale = dataTmp.x * deQFactor;
        tmp = scale + static_cast<float>(__ldg(bias_ptr + col));
        shm[sh_row][sh_col] = float_to_int8_rn(tmp * out_scale);

        scale = dataTmp.y * deQFactor;
        tmp = scale + static_cast<float>(__ldg(bias_ptr + col + 1));
        shm[sh_row][sh_col + 1] = float_to_int8_rn(tmp * out_scale);

        scale = dataTmp.z * deQFactor;
        tmp = scale + static_cast<float>(__ldg(bias_ptr + col + 2));
        shm[sh_row][sh_col + 2] = float_to_int8_rn(tmp * out_scale);

        scale = dataTmp.w * deQFactor;
        tmp = scale + static_cast<float>(__ldg(bias_ptr + col + 3));
        shm[sh_row][sh_col + 3] = float_to_int8_rn(tmp * out_scale);
    }
    else {
        shm[sh_row][sh_col] = shm[sh_row][sh_col + 1] = shm[sh_row][sh_col + 2] = shm[sh_row][sh_col + 3] = 0;
    }
    __syncthreads();

    char4 dataTmp;
    dataTmp.x = shm[sh_col][sh_row];
    dataTmp.y = shm[sh_col + 1][sh_row];
    dataTmp.z = shm[sh_col + 2][sh_row];
    dataTmp.w = shm[sh_col + 3][sh_row];

    // for dst of (size_per_head, seq_len)
    word_id = (blockIdx.y << 5) + threadIdx4;
    id_in_size = (blockIdx.x << 5) + threadIdx.y;
    col = (word_id >> 5);

    if (use_ORDER_COL32_2R_4R4) {
        int row_in_tile = id_in_size & 31;
        int col_in_tile = word_id & 31;
        row = (
            // COL32_2R_4R4
            ((id_in_size >> 5) << 10) +
            //(((row%8)/2*4+row/8)*2+row%2)*32+col
            (((((((row_in_tile % 8) >> 1) << 2) + (row_in_tile >> 3)) << 1) + (row_in_tile & 1)) << 5) + col_in_tile);
    }
    else {
        row = (
            // COL4
            ////id_in_size/8 is the number of tile of (8 rows 32 columns) -- column-major
            ////id_in_size%2 is even row, otherwise odd row
            ////word_id%COL32_/8 is the number tile of (8 rows 8 columns)
            (((id_in_size & 0xfffffff8) + ((id_in_size & 1) << 2) + ((word_id % 32) >> 3)) << 5) +
            ////word_id%8 >= 4 is the right half of (8 rows 8 columns) tile
            ////(id_in_size%8/2) is (the row id of alternating 4 rows) - 1
            (((((word_id & 7) >= 4) ? 4 : 0) + ((id_in_size % 8) >> 1)) << 2) +
            ////word_id%4 is the id of 4 cols
            (word_id & 3));
    }

    buf_ptr4[(blockIdx.z * stride + (col << 5) * size_per_head + row) >> 2] = dataTmp;
}

// add bias into V & rebuild padding
// input matrix a matrix of m = valid_word_num, n = head_num*size_per_head, CUBLASLT_ORDER_COL32
// output matrixes are a series of sub-matrixes with size of m = size_per_head, n = seq_len_padded ,
// CUBLASLT_ORDER_COL4_4R2_8C or CUBLASLT_ORDER_COL32_2R_4R4 only for int8_t IO seq_len, size_per_head must be a
// multiple of 32 grid = (size_per_head/32, seq_len_padded/32, batch_size*head_num) block = (8, 32); using char4 per
// tensor quantization for weight
template<typename T>
__global__ void add_V_bias_transform_rebuild_padding_varlen(int8_t* v_buf_,
                                                            const int8_t* V,
                                                            const T* V_bias,
                                                            const int* sequence_id_map,
                                                            const int valid_word_num,
                                                            const int batch_size,
                                                            const int seq_len,
                                                            const int seq_len_padded,
                                                            const int head_num,
                                                            const int size_per_head,
                                                            int stride,
                                                            const float* deQFactor_ptr,
                                                            const float* out_scale_ptr,
                                                            bool use_ORDER_COL32_2R_4R4)
{
    __shared__ int8_t shm[32][33];
    const char4* data_ptr = (const char4*)V;
    char4* buf_ptr4 = (char4*)v_buf_;
    const T* bias_ptr = V_bias;

    int threadIdx4 = threadIdx.x << 2;

    // for src of (seq_len, size_per_head)
    int batch_id = blockIdx.z / head_num;
    int head_id = blockIdx.z % head_num;
    int word_id = (blockIdx.y << 5) + threadIdx.y;
    int id_in_size = (blockIdx.x << 5) + threadIdx4;

    // for shm row-major
    int sh_col = threadIdx4;
    int sh_row = threadIdx.y;

    // for V layout (batch_size*seq_len, head_num*size_per_head)
    int col;
    int row = word_id < seq_len ? __ldg(sequence_id_map + batch_id * seq_len + word_id) : -1;

    if (row != -1) {
        col = head_id * size_per_head + id_in_size;
        int inIdx = ((col & 0xffffffe0) * valid_word_num + ((row << 5) + (col & 31))) >> 2;

        float tmp;
        float scale;

        const float deQFactor = __ldg(deQFactor_ptr);
        const float out_scale = __ldg(out_scale_ptr);

        char4 dataTmp = __ldg(data_ptr + inIdx);

        scale = dataTmp.x * deQFactor;
        tmp = scale + static_cast<float>(__ldg(bias_ptr + col));
        shm[sh_row][sh_col] = float_to_int8_rn(tmp * out_scale);

        scale = dataTmp.y * deQFactor;
        tmp = scale + static_cast<float>(__ldg(bias_ptr + col + 1));
        shm[sh_row][sh_col + 1] = float_to_int8_rn(tmp * out_scale);

        scale = dataTmp.z * deQFactor;
        tmp = scale + static_cast<float>(__ldg(bias_ptr + col + 2));
        shm[sh_row][sh_col + 2] = float_to_int8_rn(tmp * out_scale);

        scale = dataTmp.w * deQFactor;
        tmp = scale + static_cast<float>(__ldg(bias_ptr + col + 3));
        shm[sh_row][sh_col + 3] = float_to_int8_rn(tmp * out_scale);
    }
    else {
        shm[sh_row][sh_col] = shm[sh_row][sh_col + 1] = shm[sh_row][sh_col + 2] = shm[sh_row][sh_col + 3] = 0;
    }
    __syncthreads();

    char4 dataTmp;
    dataTmp.x = shm[sh_col][sh_row];
    dataTmp.y = shm[sh_col + 1][sh_row];
    dataTmp.z = shm[sh_col + 2][sh_row];
    dataTmp.w = shm[sh_col + 3][sh_row];

    // for dst of (size_per_head, seq_len_padded)
    word_id = (blockIdx.y << 5) + threadIdx4;
    id_in_size = (blockIdx.x << 5) + threadIdx.y;
    col = (word_id >> 5);

    if (use_ORDER_COL32_2R_4R4) {
        int row_in_tile = id_in_size & 31;
        int col_in_tile = word_id & 31;
        row = (
            // COL32_2R_4R4
            ((id_in_size >> 5) << 10) +
            //(((row%8)/2*4+row/8)*2+row%2)*32+col
            (((((((row_in_tile % 8) >> 1) << 2) + (row_in_tile >> 3)) << 1) + (row_in_tile & 1)) << 5) + col_in_tile);
    }
    else {
        row = (
            // COL4
            ////id_in_size/8 is the number of tile of (8 rows 32 columns) -- column-major
            ////id_in_size%2 is even row, otherwise odd row
            ////word_id%COL32_/8 is the number tile of (8 rows 8 columns)
            (((id_in_size & 0xfffffff8) + ((id_in_size & 1) << 2) + ((word_id % 32) >> 3)) << 5) +
            ////word_id%8 >= 4 is the right half of (8 rows 8 columns) tile
            ////(id_in_size%8/2) is (the row id of alternating 4 rows) - 1
            (((((word_id & 7) >= 4) ? 4 : 0) + ((id_in_size % 8) >> 1)) << 2) +
            ////word_id%4 is the id of 4 cols
            (word_id & 3));
    }

    buf_ptr4[(blockIdx.z * stride + (col << 5) * size_per_head + row) >> 2] = dataTmp;
}

template<typename T>
void invokeAddVBiasTransformRebuildPadding(int8_t* v_buf,
                                           const int8_t* V,
                                           const T* V_bias,
                                           const int* sequence_id_map,
                                           const int valid_word_num,
                                           const int batch_size,
                                           const int seq_len,
                                           const int head_num,
                                           const int size_per_head,
                                           const float* deQFactor_ptr,
                                           const float* out_scale_ptr,
                                           bool use_ORDER_COL32_2R_4R4,
                                           cudaStream_t stream)
{
    int seq_len_padded = (seq_len + 31) / 32 * 32;
    add_V_bias_transform_rebuild_padding_varlen<<<dim3(size_per_head / 32, seq_len_padded / 32, batch_size * head_num),
                                                  dim3(8, 32),
                                                  0,
                                                  stream>>>(v_buf,
                                                            V,
                                                            V_bias,
                                                            sequence_id_map,
                                                            valid_word_num,
                                                            batch_size,
                                                            seq_len,
                                                            seq_len_padded,
                                                            head_num,
                                                            size_per_head,
                                                            seq_len_padded * size_per_head,
                                                            deQFactor_ptr,
                                                            out_scale_ptr,
                                                            use_ORDER_COL32_2R_4R4);
}

template void invokeAddVBiasTransformRebuildPadding(int8_t* v_buf,
                                                    const int8_t* V,
                                                    const float* V_bias,
                                                    const int* sequence_id_map,
                                                    const int valid_word_num,
                                                    const int batch_size,
                                                    const int seq_len,
                                                    const int head_num,
                                                    const int size_per_head,
                                                    const float* deQFactor_ptr,
                                                    const float* out_scale_ptr,
                                                    bool use_ORDER_COL32_2R_4R4,
                                                    cudaStream_t stream);

template void invokeAddVBiasTransformRebuildPadding(int8_t* v_buf,
                                                    const int8_t* V,
                                                    const half* V_bias,
                                                    const int* sequence_id_map,
                                                    const int valid_word_num,
                                                    const int batch_size,
                                                    const int seq_len,
                                                    const int head_num,
                                                    const int size_per_head,
                                                    const float* deQFactor_ptr,
                                                    const float* out_scale_ptr,
                                                    bool use_ORDER_COL32_2R_4R4,
                                                    cudaStream_t stream);

// add bias into V & rebuild padding
// input matrix a matrix of m = valid_word_num, n = head_num*size_per_head, row major
// output matrixes are a series of sub-matrixes with size of m = size_per_head, n = seq_len_padded ,
// CUBLASLT_ORDER_COL4_4R2_8C or CUBLASLT_ORDER_COL32_2R_4R4 only for int8_t IO seq_len, size_per_head must be a
// multiple of 32 grid = (size_per_head/32, seq_len_padded/32, batch_size*head_num) block = (8, 32); using char4 per
// tensor quantization for weight
template<typename T>
__global__ void add_V_bias_transform_rebuild_padding_varlen_row(int8_t* v_buf_,
                                                                const int8_t* V,
                                                                const T* V_bias,
                                                                const int* sequence_id_map,
                                                                const int valid_word_num,
                                                                const int batch_size,
                                                                const int seq_len,
                                                                const int seq_len_padded,
                                                                const int head_num,
                                                                const int size_per_head,
                                                                int stride,
                                                                const float* deQFactor_ptr,
                                                                const float* out_scale_ptr,
                                                                bool use_ORDER_COL32_2R_4R4,
                                                                const int head_num_x_size_per_head)
{
    __shared__ int8_t shm[32][33];
    const char4* data_ptr = (const char4*)V;
    char4* buf_ptr4 = (char4*)v_buf_;
    const T* bias_ptr = V_bias;

    int threadIdx4 = threadIdx.x << 2;

    // for src of (seq_len, size_per_head)
    int batch_id = blockIdx.z / head_num;
    int head_id = blockIdx.z % head_num;
    int word_id = (blockIdx.y << 5) + threadIdx.y;
    int id_in_size = (blockIdx.x << 5) + threadIdx4;

    // for shm row-major
    int sh_col = threadIdx4;
    int sh_row = threadIdx.y;

    // for V layout (batch_size*seq_len, head_num*size_per_head)
    int col;
    int row = word_id < seq_len ? __ldg(sequence_id_map + batch_id * seq_len + word_id) : -1;

    if (row != -1) {
        col = head_id * size_per_head + id_in_size;
        int inIdx = (row * head_num_x_size_per_head + col) >> 2;

        float tmp;
        float scale;

        const float deQFactor = __ldg(deQFactor_ptr);
        const float out_scale = __ldg(out_scale_ptr);

        char4 dataTmp = __ldg(data_ptr + inIdx);

        scale = dataTmp.x * deQFactor;
        tmp = scale + static_cast<float>(__ldg(bias_ptr + col));
        shm[sh_row][sh_col] = float_to_int8_rn(tmp * out_scale);

        scale = dataTmp.y * deQFactor;
        tmp = scale + static_cast<float>(__ldg(bias_ptr + col + 1));
        shm[sh_row][sh_col + 1] = float_to_int8_rn(tmp * out_scale);

        scale = dataTmp.z * deQFactor;
        tmp = scale + static_cast<float>(__ldg(bias_ptr + col + 2));
        shm[sh_row][sh_col + 2] = float_to_int8_rn(tmp * out_scale);

        scale = dataTmp.w * deQFactor;
        tmp = scale + static_cast<float>(__ldg(bias_ptr + col + 3));
        shm[sh_row][sh_col + 3] = float_to_int8_rn(tmp * out_scale);
    }
    else {
        shm[sh_row][sh_col] = shm[sh_row][sh_col + 1] = shm[sh_row][sh_col + 2] = shm[sh_row][sh_col + 3] = 0;
    }
    __syncthreads();

    char4 dataTmp;
    dataTmp.x = shm[sh_col][sh_row];
    dataTmp.y = shm[sh_col + 1][sh_row];
    dataTmp.z = shm[sh_col + 2][sh_row];
    dataTmp.w = shm[sh_col + 3][sh_row];

    // for dst of (size_per_head, seq_len_padded)
    word_id = (blockIdx.y << 5) + threadIdx4;
    id_in_size = (blockIdx.x << 5) + threadIdx.y;
    col = (word_id >> 5);

    if (use_ORDER_COL32_2R_4R4) {
        int row_in_tile = id_in_size & 31;
        int col_in_tile = word_id & 31;
        row = (
            // COL32_2R_4R4
            ((id_in_size >> 5) << 10)
            + (((((((row_in_tile % 8) / 2) * 4) + (row_in_tile / 8)) * 2) + (row_in_tile % 2)) * 32) + col_in_tile
            // (((((((row_in_tile%8)>>1)<<2)+(row_in_tile>>3))<<1)+(row_in_tile&1))<<5)+col_in_tile
        );
    }
    else {
        row = (
            // COL4
            ////id_in_size/8 is the number of tile of (8 rows 32 columns) -- column-major
            ////id_in_size%2 is even row, otherwise odd row
            ////word_id%COL32_/8 is the number tile of (8 rows 8 columns)
            (((id_in_size & 0xfffffff8) + ((id_in_size & 1) << 2) + ((word_id % 32) >> 3)) << 5) +
            ////word_id%8 >= 4 is the right half of (8 rows 8 columns) tile
            ////(id_in_size%8/2) is (the row id of alternating 4 rows) - 1
            (((((word_id & 7) >= 4) ? 4 : 0) + ((id_in_size % 8) >> 1)) << 2) +
            ////word_id%4 is the id of 4 cols
            (word_id & 3));
    }

    buf_ptr4[(blockIdx.z * stride + (col << 5) * size_per_head + row) >> 2] = dataTmp;
}

template<typename T>
void invokeAddVBiasTransformRebuildPaddingRow(int8_t* v_buf,
                                              const int8_t* V,
                                              const T* V_bias,
                                              const int* sequence_id_map,
                                              const int valid_word_num,
                                              const int batch_size,
                                              const int seq_len,
                                              const int head_num,
                                              const int size_per_head,
                                              const float* deQFactor_ptr,
                                              const float* out_scale_ptr,
                                              bool use_ORDER_COL32_2R_4R4,
                                              cudaStream_t stream)
{
    int seq_len_padded = (seq_len + 31) / 32 * 32;
    add_V_bias_transform_rebuild_padding_varlen_row<<<
        dim3(size_per_head / 32, seq_len_padded / 32, batch_size * head_num),
        dim3(8, 32),
        0,
        stream>>>(v_buf,
                  V,
                  V_bias,
                  sequence_id_map,
                  valid_word_num,
                  batch_size,
                  seq_len,
                  seq_len_padded,
                  head_num,
                  size_per_head,
                  seq_len_padded * size_per_head,
                  deQFactor_ptr,
                  out_scale_ptr,
                  use_ORDER_COL32_2R_4R4,
                  head_num * size_per_head);
}

template void invokeAddVBiasTransformRebuildPaddingRow(int8_t* v_buf,
                                                       const int8_t* V,
                                                       const float* V_bias,
                                                       const int* sequence_id_map,
                                                       const int valid_word_num,
                                                       const int batch_size,
                                                       const int seq_len,
                                                       const int head_num,
                                                       const int size_per_head,
                                                       const float* deQFactor_ptr,
                                                       const float* out_scale_ptr,
                                                       bool use_ORDER_COL32_2R_4R4,
                                                       cudaStream_t stream);

template void invokeAddVBiasTransformRebuildPaddingRow(int8_t* v_buf,
                                                       const int8_t* V,
                                                       const half* V_bias,
                                                       const int* sequence_id_map,
                                                       const int valid_word_num,
                                                       const int batch_size,
                                                       const int seq_len,
                                                       const int head_num,
                                                       const int size_per_head,
                                                       const float* deQFactor_ptr,
                                                       const float* out_scale_ptr,
                                                       bool use_ORDER_COL32_2R_4R4,
                                                       cudaStream_t stream);

}  // namespace fastertransformer
