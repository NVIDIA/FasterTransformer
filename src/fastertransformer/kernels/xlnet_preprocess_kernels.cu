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

#include "xlnet_preprocess_kernels.h"

namespace fastertransformer {
/*************Device Function**************/
template<typename T>
T __device__ cast(float v)
{
    return (T)v;
}

template<>
__half __device__ cast(float v)
{
    return __float2half(v);
}

template<typename T>
int numPerThread()
{
    return sizeof(float) / sizeof(T);
}
/********************** Kernels ************************/
template<typename T>
void __global__ getWordEmdK(T* word_emb_k, T* params_word_emb_k, int* inp_k, int seq_len, int hidden_dim)
{

    int col = threadIdx.x;   // the index of column
    int row = blockIdx.y;    // the index of row
    int batch = blockIdx.x;  // the index of batch

    int index = inp_k[batch * seq_len + row];
    T data = params_word_emb_k[index * hidden_dim + col];

    word_emb_k[batch * seq_len * hidden_dim + row * hidden_dim + col] = data;
}

//    dim3 grid(batch_size, seq_len);
//    getWordEmdK<<<grid, hidden_dim/2,0, stream>>>(word_emb_k, params_word_emb_k, inp_k, seq_len, hidden_dim);
template<>
void __global__ getWordEmdK(__half* word_emb_k, __half* params_word_emb_k, int* inp_k, int seq_len, int hidden_dim)
{
    int col = threadIdx.x;   // the index of column
    int row = blockIdx.y;    // the index of row
    int batch = blockIdx.x;  // the index of batch

    int index = __ldg(inp_k + batch * seq_len + row);
    half2 data = ((half2*)params_word_emb_k)[(index * hidden_dim + col * 2) >> 1];

    ((half2*)word_emb_k)[(batch * seq_len * hidden_dim + row * hidden_dim + col * 2) >> 1] = data;
}

template<typename T>
void __global__ getAttnMask(T* attn_mask, float* input_mask, int seq_len)
{
    int col = threadIdx.x;
    int row = blockIdx.y;
    int batch = blockIdx.x;

    float data = 1;
    if (col == row) {
        data = 0;
    }
    float mask = input_mask[batch * seq_len + col];
    attn_mask[batch * seq_len * seq_len + row * seq_len + col] = cast<T>(data * mask);
}

template<>
void __global__ getAttnMask(__half* attn_mask, float* input_mask, int seq_len)
{
    int in_index = blockIdx.y * blockDim.x + threadIdx.x;
    int col = in_index % (seq_len / 2) * 2;
    int row = in_index / (seq_len / 2);
    int batch = blockIdx.x;

    float2 tmp;
    if (row < seq_len && col < seq_len - 1) {
        float data = 1;
        if (col == row) {
            data = 0;
        }
        tmp.x = input_mask[batch * seq_len + col] * data;

        col += 1;
        data = 1;
        if (col == row) {
            data = 0;
        }
        tmp.y = input_mask[batch * seq_len + col] * data;

        int out_index = (batch * seq_len * seq_len + row * seq_len + col) >> 1;
        ((half2*)attn_mask)[out_index] = __float22half2_rn(tmp);
    }
}

template<typename T>
void __global__ getSegMat(T* seg_mat, int* seg_id, int seq_len)
{
    int col = threadIdx.x;
    int row = blockIdx.y;
    int batch = blockIdx.x;

    int w[4] = {0, 1, 1, 0};
    int d1 = seg_id[batch * seq_len + col];
    int d2 = seg_id[batch * seq_len + row];
    int d = 0;

    d = int(floor(exp(-1 * abs(double(d1 - d2)))));

    int index = batch * seq_len * seq_len + row * seq_len + col;
    seg_mat[index * 2] = w[d * 2 + 0];
    seg_mat[index * 2 + 1] = w[d * 2 + 1];
}

template<>
void __global__ getSegMat(__half* seg_mat, int* seg_id, int seq_len)
{
    int col = threadIdx.x;
    int row = blockIdx.y;
    int batch = blockIdx.x;

    int w[4] = {0, 1, 1, 0};
    int d1 = seg_id[batch * seq_len + col];
    int d2 = seg_id[batch * seq_len + row];
    int d = 0;

    d = int(floor(exp(-1 * abs(double(d1 - d2)))));

    int index = batch * seq_len * seq_len + row * seq_len + col;
    float2 tmp_w;
    tmp_w.x = w[d * 2 + 0];
    tmp_w.y = w[d * 2 + 1];

    ((half2*)seg_mat)[index] = __float22half2_rn(tmp_w);
    // seg_mat[index*2]=__int2half_rn(w[d*2+0]);
    // seg_mat[index*2+1]=__int2half_rn(w[d*2+1]);
}

template<typename T>
void __global__ relativePosition(T* attr_k_head_r, int hidden_dim, int seq_len)
{
    int row = blockIdx.x;   //(0,256)
    int col = threadIdx.x;  //(0,384)

    float freq_seq = col * 2;
    float inv_freq = 1 / (pow(10000, freq_seq / (hidden_dim)));

    float fwd_pos_seq = seq_len - row;

    float pos_emd = inv_freq * fwd_pos_seq;
    float s = sinf(pos_emd);
    float c = cosf(pos_emd);

    attr_k_head_r[row * hidden_dim + col] = cast<T>(s);
    attr_k_head_r[row * hidden_dim + hidden_dim / 2 + col] = cast<T>(c);
}

/***********************Pre-Process************************/
template<>
void blockAttnMask<float>(dim3& grid, dim3& block, int batch_size, int seq_len)
{
    grid.x = batch_size;
    grid.y = seq_len;
    block.x = seq_len;
}

template<>
void blockAttnMask<half>(dim3& grid, dim3& block, int batch_size, int seq_len)
{
    int numThreads = 512;
    int numBlocky = (seq_len * seq_len / 2 - 1) / numThreads + 1;
    grid.x = batch_size;
    grid.y = numBlocky;
    block.x = numThreads;
}

template<typename T>
void genWordEmdK(
    int batch_size, int seq_len, int hidden_dim, T* word_emb_k, T* params_word_emb_k, int* inp_k, cudaStream_t stream)
{
    dim3 grid_word_emd_k(batch_size, seq_len);
    dim3 block_word_emd_k(hidden_dim / numPerThread<T>());

    getWordEmdK<<<grid_word_emd_k, block_word_emd_k, 0, stream>>>(
        word_emb_k, params_word_emb_k, inp_k, seq_len, hidden_dim);
}

template<typename T>
void preProcess(int batch_size,
                int seq_len,
                int hidden_dim,
                T* attn_mask,
                float* input_mask,
                T* seg_mat,
                int* seg_id,
                T* attr_k_head_r,
                cudaStream_t stream)
{
    dim3 grid_attn_mask;
    dim3 block_attn_mask;
    blockAttnMask<T>(grid_attn_mask, block_attn_mask, batch_size, seq_len);
    getAttnMask<<<grid_attn_mask, block_attn_mask, 0, stream>>>(attn_mask, input_mask, seq_len);

    dim3 grid_seg_mat(batch_size, seq_len);
    dim3 block_seg_mat(seq_len);
    getSegMat<<<grid_seg_mat, block_seg_mat, 0, stream>>>(seg_mat, seg_id, seq_len);

    // relative_positional_encoding
    dim3 grid_rel_position(seq_len * 2);
    dim3 block_rel_position(hidden_dim / 2);
    relativePosition<<<grid_rel_position, block_rel_position, 0, stream>>>(attr_k_head_r, hidden_dim, seq_len);
}

template void preProcess<float>(int batch_size,
                                int seq_len,
                                int hidden_dim,
                                float* attn_mask,
                                float* input_mask,
                                float* seg_mat,
                                int* seg_id,
                                float* attr_k_head_r,
                                cudaStream_t stream);

template void preProcess<half>(int batch_size,
                               int seq_len,
                               int hidden_dim,
                               half* attn_mask,
                               float* input_mask,
                               half* seg_mat,
                               int* seg_id,
                               half* attr_k_head_r,
                               cudaStream_t stream);

template void genWordEmdK<float>(int batch_size,
                                 int seq_len,
                                 int hidden_dim,
                                 float* word_emb_k,
                                 float* params_word_emb_k,
                                 int* inp_k,
                                 cudaStream_t stream);
template void genWordEmdK<half>(int batch_size,
                                int seq_len,
                                int hidden_dim,
                                half* word_emb_k,
                                half* params_word_emb_k,
                                int* inp_k,
                                cudaStream_t stream);
}  // namespace fastertransformer
