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
#include "src/fastertransformer/kernels/xlnet_attention_kernels.h"
namespace fastertransformer {
template<typename T>
int numPerThread()
{
    return sizeof(float) / sizeof(T);
}

int next_pow2(int a)
{
    int rval = 32;
    if (a > 32) {
        while (rval < a) {
            rval <<= 1;
        }
    }
    return rval;
}

__device__ half2 cH2(const half* ptr, int offset)
{
    return __ldg((half2*)(ptr + offset));
}
__forceinline__ __device__ unsigned lane_id()
{
    unsigned ret;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

__forceinline__ __device__ unsigned warp_id()
{
    // this is not equal to threadIdx.x / 32
    unsigned ret;
    asm volatile("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}

template<typename T>
__inline__ __device__ T warpReduceSum(T val)
{
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
    }
    return val;
}

template<typename T>
__inline__ __device__ T warpReduceMax(T val)
{
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    }
    return val;
}

template<typename T>
__inline__ __device__ T blockReduceSum(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = warpReduceSum<T>(val);

    if (lane == 0) {
        shared[wid] = val;
    }

    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)(0.0f);
    val = warpReduceSum<T>(val);

    return val;
}

template<typename T>
__inline__ __device__ T blockReduceMax(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;  // in-warp idx
    int wid = threadIdx.x >> 5;     // warp idx

    val = warpReduceMax(val);  // get maxx in each warp

    if (lane == 0) {  // record in-warp maxx by warp Idx
        shared[wid] = val;
    }

    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : -1e30f;
    val = warpReduceMax(val);

    return val;
}

/********************** Kernels ************************/
template<typename T>
void __global__ prepareMatrixes(T* q_buf,
                                T* q_buf_bd,
                                T* q_buf_ef,
                                T* k_buf,
                                T* k_buf_bd,
                                T* k_buf_ef,
                                const T* query_buf,
                                const T* key_buf,
                                const T* k_head_r,
                                const T* attr_seg_embed,
                                const T* attr_bias_Q_w,
                                const T* attr_bias_Q_r,
                                const T* attr_bias_Q_s,
                                const int off0,
                                const int i_off1,
                                const int o_off1,
                                int off2)
{
    int batch = blockIdx.y;
    int seq = blockIdx.x;
    int head_loc = threadIdx.x;

    T tmp;
    if (head_loc < i_off1) {
        int head = head_loc / off2;
        int loc = head_loc % off2;

        int index = batch * off0 + seq * i_off1 + head_loc;
        tmp = query_buf[index];
        int index_out = batch * off0 + head * o_off1 + seq * off2 + loc;
        // left matrix
        q_buf[index_out] = tmp + __ldg(attr_bias_Q_w + head_loc);     // tex2D(t_attr_bias_Q_w, loc, head);
        q_buf_bd[index_out] = tmp + __ldg(attr_bias_Q_r + head_loc);  // tex2D(t_attr_bias_Q_r, loc, head);
        q_buf_ef[index_out] = tmp + __ldg(attr_bias_Q_s + head_loc);  // tex2D(t_attr_bias_Q_s, loc, head);

        // right matrix
        k_buf[index_out] = key_buf[index];  // ac

        // bd
        index = seq * i_off1 + head_loc;  //(seq, head_loc)
        tmp = k_head_r[index];
        index_out = index_out + batch * off0 + head * o_off1;  //(batch, head,seq,loc)
        k_buf_bd[index_out] = tmp;

        index = index + off0;  //(seq+seq_len, head_loc)
        tmp = k_head_r[index];
        index_out = index_out + o_off1;  //(batch, head,seq+seq_len,loc)
        k_buf_bd[index_out] = tmp;

        // ef
        if (seq <= 1) {
            index = seq * i_off1 + head_loc;  //(seq, head, loc)
            tmp = attr_seg_embed[index];
            index_out = batch * 2 * i_off1 + (head * 2 + seq) * off2 + loc;  //(head,seq,loc)
            k_buf_ef[index_out] = tmp;
        }
    }
}
template<>
void __global__ prepareMatrixes(__half* q_buf,
                                __half* q_buf_bd,
                                __half* q_buf_ef,
                                __half* k_buf,
                                __half* k_buf_bd,
                                __half* k_buf_ef,
                                const __half* query_buf,
                                const __half* key_buf,
                                const __half* k_head_r,
                                const __half* attr_seg_embed,
                                const __half* attr_bias_Q_w,
                                const __half* attr_bias_Q_r,
                                const __half* attr_bias_Q_s,
                                const int off0,
                                const int i_off1,
                                const int o_off1,
                                int off2)
{
    int batch = blockIdx.y;
    int seq = blockIdx.x;
    int head_loc = threadIdx.x * 2;

    half2 tmp;
    if (head_loc < i_off1) {
        int head = head_loc / off2;
        int loc = head_loc % off2;
        int h2_index = (batch * off0 + seq * i_off1 + head_loc) >> 1;

        tmp = ((half2*)query_buf)[h2_index];
        int h2_index_out = (batch * off0 + head * o_off1 + seq * off2 + loc) >> 1;
        // left matrix
        ((half2*)q_buf)[h2_index_out] = __hadd2(tmp, cH2(attr_bias_Q_w, head_loc));
        ((half2*)q_buf_bd)[h2_index_out] = __hadd2(tmp, cH2(attr_bias_Q_r, head_loc));
        ((half2*)q_buf_ef)[h2_index_out] = __hadd2(tmp, cH2(attr_bias_Q_s, head_loc));

        // right matrix
        ((half2*)k_buf)[h2_index_out] = ((half2*)key_buf)[h2_index];  // ac

        // bd
        h2_index = (seq * i_off1 + head_loc) >> 1;  //(seq, head_loc)
        tmp = ((half2*)k_head_r)[h2_index];
        h2_index_out = (batch * off0 * 2 + head * o_off1 * 2 + seq * off2 + loc) >> 1;  //(batch, head,seq,loc)
        ((half2*)k_buf_bd)[h2_index_out] = tmp;

        h2_index = (seq * i_off1 + head_loc + off0) >> 1;  //(seq+seq_len, head_loc)
        tmp = ((half2*)k_head_r)[h2_index];
        h2_index_out =
            (batch * off0 * 2 + (head * 2 + 1) * o_off1 + seq * off2 + loc) >> 1;  //(batch, head,seq+seq_len,loc)
        ((half2*)k_buf_bd)[h2_index_out] = tmp;

        // ef
        if (seq <= 1) {
            h2_index = (seq * i_off1 + head_loc) >> 1;  //(seq, head, loc)
            tmp = ((half2*)attr_seg_embed)[h2_index];
            h2_index_out = (batch * 2 * i_off1 + (head * 2 + seq) * off2 + loc) >> 1;  //(head,seq,loc)
            ((half2*)k_buf_ef)[h2_index_out] = tmp;
        }
    }
}

template<typename T>
void __global__ transpose102(T* dst, const T* src, const int off0, const int i_off1, const int o_off1, const int off2)
{
    int x[4] = {0};
    x[0] = blockIdx.x;   //[0,7]
    x[1] = blockIdx.y;   //[0,11]
    x[2] = threadIdx.x;  //[0,127]
    x[3] = threadIdx.y;  //[0,1]

    int input_index = x[0] * off0 + x[1] * i_off1 + x[2] * off2 + x[3];  // [batch, 0, 1, 2]=[d0,d1,d2,d3]

    int out_index = x[0] * off0 + x[2] * o_off1 + x[1] * off2 + x[3];  // [batch, 1, 0, 2]=[d0,d2,d1,d3]

    dst[out_index] = src[input_index];
}

template<>
void __global__
transpose102(__half* dst, const __half* src, const int off0, const int i_off1, const int o_off1, const int off2)
{
    int x[4] = {0};
    x[0] = blockIdx.x;   //[0,7]
    x[1] = blockIdx.y;   //[0,11]
    x[2] = threadIdx.x;  //[0,127]

    int input_index = (x[0] * off0 + x[1] * i_off1 + x[2] * off2) >> 1;  // [batch, 0, 1, 2]=[d0,d1,d2,d3]

    int out_index = (x[0] * off0 + x[2] * o_off1 + x[1] * off2) >> 1;  // [batch, 1, 0, 2]=[d0,d2,d1,d3]

    ((half2*)dst)[out_index] = ((half2*)src)[input_index];
}

void __global__ transpose201(float* dst,
                             const float* src,
                             const int off0,
                             const int i_off1,
                             const int head_num,
                             const int o_off1,
                             const int seq_len)
{
    int batch = blockIdx.x;
    int d0 = blockIdx.y;
    int d1 = threadIdx.x;

    extern __shared__ float sdata[];
    int i = 0;

    // Read data into shared memory
    int index = batch * off0 + d0 * i_off1;  // d1*i_off2+d2
    int offset = d1;
    src = src + index;
    int row = offset / head_num;
    int col = offset % head_num;
    for (i = 0; i < head_num; i++) {
        sdata[row * (head_num + 1) + col] = src[offset];
        offset += seq_len;
        row = offset / head_num;
        col = offset % head_num;
    }

    __syncthreads();

    index = batch * off0 + d0 * seq_len + d1;
    offset = 0;
    dst = dst + index;
    for (i = 0; i < head_num; i++) {
        dst[offset] = sdata[d1 * (head_num + 1) + i];
        offset += o_off1;
    }
}

void __global__ transpose201(__half* dst,
                             const __half* src,
                             const int off0,
                             const int i_off1,
                             const int head_num,
                             const int o_off1,
                             const int seq_len)
{
    int batch = blockIdx.x;
    int d0 = blockIdx.y;
    int d1 = threadIdx.x;
    extern __shared__ float sdata[];
    int i = 0;

    // Read data into shared memory
    int index = batch * off0 + d0 * i_off1;  // d1*i_off2+d2
    int offset = d1;
    src = src + index;
    int row = offset / head_num;
    int col = offset % head_num;
    for (i = 0; i < head_num; i++) {
        sdata[row * (head_num + 1) + col] = __half2float(src[offset]);
        offset += seq_len;
        row = offset / head_num;
        col = offset % head_num;
    }

    __syncthreads();

    index = batch * off0 + d0 * seq_len + d1;
    offset = 0;
    dst = dst + index;
    for (i = 0; i < head_num; i++) {
        dst[offset] = __float2half(sdata[d1 * (head_num + 1) + i]);
        offset += o_off1;
    }
}
/*dim3 grid_shift(batch_size, head_num, seq_len);
  dim3 block_shift(seq_len*2);
  int off0=head_num*seq_len*seq_len;
  int off1=seq_len*seq_len; */
template<typename T>
void __global__ relShiftBd(T* outMatrix, const T* inputMatrix, const int off0, const int off1, const int seq_len)
{
    int batch = blockIdx.x;  //[0,7]
    int head = blockIdx.y;   //[0,11]
    int row = blockIdx.z;    //[0,127]
    int col = threadIdx.x;   //[0,255]

    int input_index = (batch * off0 + head * off1 + row * seq_len) * 2 + col;
    if (col >= seq_len || row != 0) {
        T idata = inputMatrix[input_index];
        // int tmp_index=row*(2*seq_len-1)+row+col-seq_len;
        int tmp_index = row * 2 * seq_len + col - seq_len;
        int out_row = tmp_index / (seq_len * 2 - 1);
        int out_col = tmp_index % (seq_len * 2 - 1);
        if (out_col < seq_len) {
            int out_index = batch * off0 + head * off1 + out_row * seq_len + out_col;
            outMatrix[out_index] = idata;
        }
    }
}
/*int threads=512;
  seq_dim1=threads/seq_len
  seq_dim2=seq_len/dimx
  dim3 grid_rel(batch_size, head_num, seq_dim2);
  dim3 block_rel(seq_dim1, seq_len);*/
template<>
void __global__
relShiftBd(__half* outMatrix, const __half* inputMatrix, const int off0, const int off1, const int seq_len)
{
    int batch = blockIdx.x;                           //[0,7]
    int head = blockIdx.y;                            //[0,11]
    int row = blockIdx.z * blockDim.x + threadIdx.x;  //[0,127]
    int col = threadIdx.y * 2;                        //[0,255]

    int input_index = (batch * off0 + head * off1 + row * seq_len) * 2 + col;
    int out_index;
    int out_row;
    int out_col;
    int tmp_index;
    half2 idata;
    if (col >= seq_len || row != 0) {
        idata = ((half2*)inputMatrix)[input_index >> 1];
        // int tmp_index=row*(2*seq_len-1)+row+col-seq_len;
        tmp_index = row * 2 * seq_len + col - seq_len;
        out_row = tmp_index / (seq_len * 2 - 1);
        out_col = tmp_index % (seq_len * 2 - 1);
        if (out_col < seq_len) {
            out_index = (batch * off0 + head * off1 + out_row * seq_len + out_col);
            outMatrix[out_index] = __low2half(idata);
        }
        tmp_index += 1;
        out_row = tmp_index / (seq_len * 2 - 1);
        out_col = tmp_index % (seq_len * 2 - 1);
        if (out_col < seq_len) {
            out_index = (batch * off0 + head * off1 + out_row * seq_len + out_col);
            outMatrix[out_index] = __high2half(idata);
        }
    }
}

/*dim3 grid_score(batch_size,head_num,seq_len);
  dim3 block_score(next_pow2(seq_len));
  int off0=head_num*seq_len*seq_len;
  int off1=seq_len*seq_len;
  float p=(1/(pow(size_per_head,0.5)));

  int voff0=head_num*seq_len*size_per_head;
  int v_o_off1=seq_len*size_per_head;
  int voff2=size_per_head;
  int v_i_off1=head_num*size_per_head;*/
template<typename T>
__global__ void calAttnScore_valueBuf(T* attn_score,
                                      const T* ac,
                                      const T* bd,
                                      const T* ef,
                                      const T* attn_mask,
                                      const int off0,
                                      const int off1,
                                      const int seq_len,
                                      const float p,
                                      T* value_buf_trans,
                                      const T* value_buf,
                                      const int voff0,
                                      const int v_i_off1,
                                      const int v_o_off1,
                                      const int voff2)
{
    int batch = blockIdx.x;
    int head = blockIdx.y;
    int seq1 = blockIdx.z;
    int seq2 = threadIdx.x;

    int offset = batch * off0 + head * off1 + seq1 * seq_len;
    int index = offset + seq2;
    int out_index;
    T score;
    T mask;
    T large_value = -1e4;
    if (sizeof(T) == 4) {
        large_value = -1e30f;
    }
    if (seq2 < seq_len) {
        score = ac[index] + bd[index] + ef[index];
        score = score * p;

        out_index = batch * off1 + seq1 * seq_len + seq2;
        mask = attn_mask[out_index] * (large_value);
        score = score + mask;
    }
    // softmax(attn_score+offset,seq_len, seq2);
    __shared__ float s_sum, s_max;
    float tmp = seq2 < seq_len ? score : large_value;
    float max_val = blockReduceMax<float>(tmp);
    if (seq2 == 0) {
        s_max = max_val;
    }
    __syncthreads();
    float qk_tmp = seq2 < seq_len ? __expf((float)(tmp - s_max)) : 0.0f;
    float sum_val = blockReduceSum<float>(qk_tmp);
    __syncthreads();
    if (seq2 == 0) {
        s_sum = sum_val;
    }
    __syncthreads();
    if (seq2 < seq_len) {
        attn_score[index] = (T)(qk_tmp / s_sum);
    }
    // end softmax

    offset = seq2;
    while (offset < voff2) {
        out_index = batch * voff0 + head * v_o_off1 + seq1 * voff2 + offset;
        index = batch * voff0 + seq1 * v_i_off1 + head * voff2 + offset;
        value_buf_trans[out_index] = value_buf[index];
        offset += seq_len;
    }
}

void __global__ calAttnScore_valueBuf_small(__half* attn_score,
                                            const __half* ac,
                                            const __half* bd,
                                            const __half* ef,
                                            const __half* attn_mask,
                                            const int off0,
                                            const int off1,
                                            const int seq_len,
                                            int n_seq1,
                                            const float p,
                                            __half* value_buf_trans,
                                            const __half* value_buf,
                                            const int voff0,
                                            const int v_i_off1,
                                            const int v_o_off1,
                                            const int voff2)
{
    int lid = lane_id();
    int tid = threadIdx.x;
    int wid = tid / 32;
    int seq2 = lid << 1;

    int batch = blockIdx.x;
    int head = blockIdx.y;
    int seq1 = blockIdx.z * n_seq1 + wid;

    int offset = batch * off0 + head * off1 + seq1 * seq_len;
    int index = (offset + seq2) >> 1;
    int out_index;
    float2 tmp1, tmp2;

    // Data prepare section
    if (seq2 < seq_len) {
        tmp1 = __half22float2(((half2*)ac)[index]);
        tmp2 = __half22float2(((half2*)bd)[index]);
        tmp1.x += tmp2.x;
        tmp1.y += tmp2.y;
        // tmp1=__hadd2(tmp1,tmp2);
        tmp2 = __half22float2(((half2*)ef)[index]);
        tmp1.x += tmp2.x;
        tmp1.y += tmp2.y;

        // half2 score=__hadd2(tmp1, tmp2);
        tmp1.x = tmp1.x * p;
        tmp1.y = tmp1.y * p;

        out_index = (batch * off1 + seq1 * seq_len + seq2) >> 1;
        tmp2 = __half22float2(((half2*)attn_mask)[out_index]);

        tmp1.x = tmp1.x + -1e4 * tmp2.x;
        tmp1.y = tmp1.y + -1e4 * tmp2.y;
    }
    else {
        tmp1.x = tmp1.y = -1e4f;
    }

    // Softmax section
    float tmp = tmp1.x > tmp1.y ? tmp1.x : tmp1.y;
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp = max(tmp, __shfl_xor_sync(FINAL_MASK, tmp, mask, 32));
    }
    tmp = __shfl_sync(FINAL_MASK, tmp, 0);

    /// normalize the input
    tmp1.x = seq2 < seq_len ? __expf((float)(tmp1.x - tmp)) : 0.0f;
    tmp1.y = seq2 < seq_len ? __expf((float)(tmp1.y - tmp)) : 0.0f;
    tmp = tmp1.x + tmp1.y;
    /// get sum of the normalized value
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp = tmp + __shfl_xor_sync(FINAL_MASK, tmp, mask, 32);
    }
    if (seq2 == 0) {
        tmp = tmp;
    }
    tmp = __shfl_sync(FINAL_MASK, tmp, 0);

    /// set the value
    if (seq2 < seq_len) {
        tmp1.x = tmp1.x / tmp;
        tmp1.y = tmp1.y / tmp;
        ((half2*)attn_score)[index] = __float22half2_rn(tmp1);
    }

    // value_buf section
    offset = seq2;
    while (offset < voff2) {
        index = (batch * voff0 + seq1 * v_i_off1 + head * voff2 + offset) >> 1;
        half2 v = ((half2*)value_buf)[index];

        out_index = (batch * voff0 + head * v_o_off1 + seq1 * voff2 + offset) >> 1;
        ((half2*)value_buf_trans)[out_index] = v;
        offset += seq_len;
    }
}
void __global__ calAttnScore_valueBuf_large(__half* attn_score,
                                            const __half* ac,
                                            const __half* bd,
                                            const __half* ef,
                                            const __half* attn_mask,
                                            const int off0,
                                            const int off1,
                                            const int seq_len,
                                            const float p,
                                            __half* value_buf_trans,
                                            const __half* value_buf,
                                            const int voff0,
                                            const int v_i_off1,
                                            const int v_o_off1,
                                            const int voff2)
{
    int batch = blockIdx.x;
    int head = blockIdx.y;
    int seq1 = blockIdx.z;

    int lid = lane_id();
    int tid = threadIdx.x;
    int wid = tid / 32;
    int seq2 = tid << 1;

    int offset = batch * off0 + head * off1 + seq1 * seq_len;
    int index = (offset + seq2) >> 1;
    int out_index;
    float2 tmp1, tmp2;
    __shared__ float sdata[32];
    __shared__ float s_max;
    __shared__ float s_sum;
    // Data prepare section
    if (seq2 < seq_len) {
        tmp1 = __half22float2(((half2*)ac)[index]);
        tmp2 = __half22float2(((half2*)bd)[index]);
        tmp1.x += tmp2.x;
        tmp1.y += tmp2.y;
        // tmp1=__hadd2(tmp1,tmp2);
        tmp2 = __half22float2(((half2*)ef)[index]);
        tmp1.x += tmp2.x;
        tmp1.y += tmp2.y;

        // half2 score=__hadd2(tmp1, tmp2);
        tmp1.x = tmp1.x * p;
        tmp1.y = tmp1.y * p;

        out_index = (batch * off1 + seq1 * seq_len + seq2) >> 1;
        tmp2 = __half22float2(((half2*)attn_mask)[out_index]);

        tmp1.x = tmp1.x + -1e4 * tmp2.x;
        tmp1.y = tmp1.y + -1e4 * tmp2.y;
    }
    else {
        tmp1.x = tmp1.y = -1e4f;
    }
    // Softmax section
    float tmp = tmp1.x > tmp1.y ? tmp1.x : tmp1.y;
    for (int mask = 16; mask > 0; mask /= 2) {
        tmp = max(tmp, __shfl_xor_sync(FINAL_MASK, tmp, mask, 32));
    }
    if (wid == 0) {
        sdata[lid] = -1e4f;
    }
    __syncthreads();
    if (lid == 0) {
        sdata[wid] = tmp;
    }
    __syncthreads();

    if (wid == 0) {
        tmp = sdata[lid];
        for (int mask = 16; mask > 0; mask /= 2) {
            tmp = max(tmp, __shfl_xor_sync(FINAL_MASK, tmp, mask, 32));
        }
    }
    if (tid == 0) {
        s_max = tmp;
    }
    __syncthreads();

    /// normalize the input

    tmp1.x = seq2 < seq_len ? __expf((float)(tmp1.x - s_max)) : 0.0f;
    tmp1.y = seq2 < seq_len ? __expf((float)(tmp1.y - s_max)) : 0.0f;
    tmp = tmp1.x + tmp1.y;

    /// get sum of the normalized value
    for (int mask = 16; mask > 0; mask /= 2) {
        tmp = tmp + __shfl_xor_sync(FINAL_MASK, tmp, mask, 32);
    }
    if (wid == 0) {
        sdata[lid] = 0;
    }
    __syncthreads();
    if (lid == 0) {
        sdata[wid] = tmp;
    }
    __syncthreads();

    if (wid == 0) {
        tmp = sdata[tid];
        for (int mask = 16; mask > 0; mask /= 2) {
            tmp = tmp + __shfl_xor_sync(FINAL_MASK, tmp, mask, 32);
        }
    }
    if (tid == 0) {
        s_sum = tmp;
    }
    __syncthreads();

    /// set the value
    if (seq2 < seq_len) {
        tmp1.x = tmp1.x / s_sum;
        tmp1.y = tmp1.y / s_sum;
        ((half2*)attn_score)[index] = __float22half2_rn(tmp1);
    }
    // value_buf section
    offset = seq2;
    while (offset < voff2) {
        index = (batch * voff0 + seq1 * v_i_off1 + head * voff2 + offset) >> 1;
        half2 v = ((half2*)value_buf)[index];

        out_index = (batch * voff0 + head * v_o_off1 + seq1 * voff2 + offset) >> 1;
        ((half2*)value_buf_trans)[out_index] = v;
        offset += seq_len;
    }
}

// dim3 grid_trans_v(batch_size,seq_len, head_num);
// dim3 block_trans_v(size_per_head);
template<typename T>
__global__ void
transpose102_v2(T* dst, const T* src, const int off0, const int i_off1, const int o_off1, const int off2)
{
    int x[4] = {0};
    x[0] = blockIdx.x;
    x[1] = threadIdx.x / off2;
    x[2] = blockIdx.y;          //[0,128] seq_len
    x[3] = threadIdx.x % off2;  //[0,31] size_per_head

    T tmp;
    if (x[3] < off2) {
        int input_index = x[0] * off0 + x[1] * i_off1 + x[2] * off2 + x[3];  // [batch, 0, 1, 2]=[d0,d1,d2,d3]
        tmp = src[input_index];
        int out_index = x[0] * off0 + x[2] * o_off1 + x[1] * off2 + x[3];  // [batch, 1, 0, 2]=[d0,d2,d1,d3]

        dst[out_index] = tmp;
    }
}

template<>
__global__ void
transpose102_v2(__half* dst, const __half* src, const int off0, const int i_off1, const int o_off1, const int off2)
{
    int x[4] = {0};
    x[0] = blockIdx.x;              //[0,7] batch_size
    x[1] = threadIdx.x * 2 / off2;  // head_num
    x[2] = blockIdx.y;              // seq_len
    x[3] = threadIdx.x * 2 % off2;  //[0,63] size_per_head

    if (x[3] < off2) {
        half2 tmp;
        int in_index = (x[0] * off0 + x[1] * i_off1 + x[2] * off2 + x[3]) >> 1;  // [batch, 0, 1, 2]=[d0,d1,d2,d3]
        tmp = ((half2*)src)[in_index];
        int out_index = (x[0] * off0 + x[2] * o_off1 + x[1] * off2 + x[3]) >> 1;  // [batch, 1, 0, 2]=[d0,d2,d1,d3]
        ((half2*)dst)[out_index] = tmp;
    }
}

template<typename T>
__global__ void
addBias_layerNorm(T* out, const T* input, const T* bias, const T* gamma, const T* beta, int m, int n, float epsilon)
{
    int tid = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    float local_out = 0.0f;
    int id = blockIdx.x * n + tid;
    local_out += (float)(input[id]);
    local_out += (float)(__ldg(&bias[id]));

    mean = blockReduceSum<float>(local_out);
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();
    variance = blockReduceSum<float>((local_out - s_mean) * (local_out - s_mean));
    if (threadIdx.x == 0) {
        s_variance = variance / n + epsilon;
    }
    __syncthreads();

    out[id] =
        (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg(&gamma[tid])) + (float)(__ldg(&beta[tid])));
}

template<>
__global__ void addBias_layerNorm(__half* out,
                                  const __half* input,
                                  const __half* bias,
                                  const __half* gamma,
                                  const __half* beta,
                                  int m,
                                  int n,
                                  float epsilon)
{

    int tid = threadIdx.x;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;
    float2 local_out_fp2;

    half2* out_ptr = (half2*)out;
    const half2* input_ptr = (const half2*)input;
    const half2* bias_ptr = (const half2*)bias;  // blockIdx.x * n + i
    const half2* gamma_ptr = (const half2*)gamma;
    const half2* beta_ptr = (const half2*)beta;

    float local_out = 0.0f;
    int id = (blockIdx.x * n + tid * 2) >> 1;
    local_out_fp2 = __half22float2(__hadd2(input_ptr[id], __ldg(&bias_ptr[id])));

    local_out += local_out_fp2.x;
    local_out += local_out_fp2.y;

    mean = blockReduceSum<float>(local_out);
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    variance = (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
    variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + epsilon);
    }
    __syncthreads();

    float2 gamma_val = __half22float2(__ldg(&gamma_ptr[tid]));
    float2 beta_val = __half22float2(__ldg(&beta_ptr[tid]));
    local_out_fp2.x = (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
    local_out_fp2.y = (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
    out_ptr[id] = __float22half2_rn(local_out_fp2);
}

/*width=hidden_dim_ff;
  height=seq_len;
  dim3 block(1024);
  dim3 grid(batch_size, seq_len);
  gelu_bias_loop<<<grid, block, 0, stream>>>(output_fc1, output_fc1, attr_fc1_bias, hidden_dim_ff,seq_len); */
template<typename T>
__global__ void gelu_bias_loop(T* src, const T* bias, int width, int height)
{
    int batch = blockIdx.x;
    int x = blockIdx.y;
    int y = threadIdx.x;

    if (x < height) {
        int index = batch * width * height + x * width;
        float v_src;
        float v_bias;
        float v;
        for (; y < width; y = y + blockDim.x) {
            v_bias = bias[y];
            v_src = src[index + y];
            v = v_src + v_bias;

            src[index + y] = (T)(0.5f * v * (1.0f + tanhf(0.79788456f * (v + 0.044715f * v * v * v))));
        }
    }
}
template<>
__global__ void gelu_bias_loop(__half* src, const __half* bias, int width, int height)
{
    int batch = blockIdx.x;
    int x = blockIdx.y;
    int y = threadIdx.x * 2;

    if (x < height) {
        int index = batch * width * height + x * width;
        half2 v_src;
        half2 v_bias;
        half2 v;
        float2 t;
        for (; y < width; y = y + blockDim.x * 2) {
            v_bias = ((half2*)bias)[y >> 1];
            v_src = ((half2*)src)[(index + y) >> 1];
            v = __hadd2(v_src, v_bias);
            t = __half22float2(v);
            t.x = (0.5f * t.x * (1.0f + tanhf(0.79788456f * (t.x + 0.044715f * t.x * t.x * t.x))));
            t.y = (0.5f * t.y * (1.0f + tanhf(0.79788456f * (t.y + 0.044715f * t.y * t.y * t.y))));

            ((half2*)src)[(index + y) >> 1] = __float22half2_rn(t);
        }
    }
}

/*********************Invoke Functions***********************/

template<typename T>
void invokePrepareMatrixes(int batch_size,
                           int seq_len,
                           int hidden_dim,
                           int size_per_head,
                           T* q_buf,
                           T* q_buf_bd,
                           T* q_buf_ef,
                           T* k_buf,
                           T* k_buf_bd,
                           T* k_buf_ef,
                           T* query_buf,
                           T* key_buf,
                           T* k_head_r,
                           T* attr_seg_embed,
                           T* attr_bias_Q_w,
                           T* attr_bias_Q_r,
                           T* attr_bias_Q_s,
                           cudaStream_t stream)
{
    int off0 = seq_len * hidden_dim;  // seq_len*head_num*size_per_head
    int i_off1 = hidden_dim;          // head_num*size_per_head
    int o_off1 = seq_len * size_per_head;
    int off2 = size_per_head;

    dim3 grid(seq_len, batch_size);
    dim3 block(next_pow2(hidden_dim) / numPerThread<T>());
    prepareMatrixes<<<grid, block, 0, stream>>>(q_buf,
                                                q_buf_bd,
                                                q_buf_ef,
                                                k_buf,
                                                k_buf_bd,
                                                k_buf_ef,
                                                query_buf,
                                                key_buf,
                                                k_head_r,
                                                attr_seg_embed,
                                                attr_bias_Q_w,
                                                attr_bias_Q_r,
                                                attr_bias_Q_s,
                                                off0,
                                                i_off1,
                                                o_off1,
                                                off2);
}

template<typename T>
void invokeTranspose102(
    int batch_size, int seq_len, int head_num, T* qk_buf_ef_trans, T* qk_buf_ef, cudaStream_t stream)
{
    dim3 grid_trans_v(batch_size, head_num);

    // dim3 block_trans_v(seq_len,2);//float
    // dim3 block_trans_v(seq_len);//__half
    dim3 block_trans_v(seq_len, 2 / (numPerThread<T>()));

    int toff0 = head_num * seq_len * 2;
    int ti_off1 = seq_len * 2;
    int to_off1 = head_num * 2;
    int toff2 = 2;

    transpose102<<<grid_trans_v, block_trans_v, 0, stream>>>(
        qk_buf_ef_trans, qk_buf_ef, toff0, ti_off1, to_off1, toff2);
}

template<typename T>
void invokeTranspose201(
    int batch_size, int seq_len, int head_num, T* qk_buf_ef_seg_trans, T* qk_buf_ef_seg, cudaStream_t stream)
{
    dim3 grid_trans2(batch_size, seq_len);
    dim3 block_trans2(seq_len);
    int t2_off0 = seq_len * seq_len * head_num;
    int t2_i_off1 = seq_len * head_num;
    int t2_o_off1 = seq_len * seq_len;
    int t2_i_off2 = head_num;
    int t2_o_off2 = seq_len;

    transpose201<<<grid_trans2, block_trans2, seq_len*(head_num + 1) * sizeof(float), stream>>>(
        qk_buf_ef_seg_trans, qk_buf_ef_seg, t2_off0, t2_i_off1, t2_i_off2, t2_o_off1, t2_o_off2);
}

template<typename T>
void blockRelShiftBd(dim3& grid, dim3& block, int batch_size, int head_num, int seq_len)
{
    grid.x = batch_size;
    grid.y = head_num;
    grid.z = seq_len;

    block.x = seq_len * 2;
}
template<>
void blockRelShiftBd<half>(dim3& grid, dim3& block, int batch_size, int head_num, int seq_len)
{
    int threads = 512;
    int seq_dim1 = threads / seq_len;
    int seq_dim2 = seq_len / seq_dim1;

    grid.x = batch_size;
    grid.y = head_num;
    grid.z = seq_dim2;

    block.x = seq_dim1;
    block.y = seq_len;
}

template<typename T>
void invokeRelShiftBd(int batch_size, int head_num, int seq_len, T* qk_buf_bd_shift, T* qk_buf_bd, cudaStream_t stream)
{
    dim3 grid_shift;
    dim3 block_shift;
    blockRelShiftBd<T>(grid_shift, block_shift, batch_size, head_num, seq_len);

    int off0 = head_num * seq_len * seq_len;
    int off1 = seq_len * seq_len;

    relShiftBd<<<grid_shift, block_shift, 0, stream>>>(qk_buf_bd_shift, qk_buf_bd, off0, off1, seq_len);
}

template<typename T>
void invokeCalAttnScore(int batch_size,
                        int head_num,
                        int seq_len,
                        int size_per_head,
                        float q_scaling,
                        T* attn_score,
                        T* qk_buf,
                        T* qk_buf_bd_shift,
                        T* qk_buf_ef_seg_trans,
                        T* attn_mask,
                        T* value_buf_trans,
                        T* value_buf,
                        cudaStream_t stream)
{
    int off0 = head_num * seq_len * seq_len;
    int off1 = seq_len * seq_len;
    float p = 1 / ((pow(size_per_head, 0.5)) * q_scaling);

    int voff0 = head_num * seq_len * size_per_head;
    int v_o_off1 = seq_len * size_per_head;
    int voff2 = size_per_head;
    int v_i_off1 = head_num * size_per_head;

    dim3 grid_score(batch_size, head_num, seq_len);
    dim3 block_score(next_pow2(seq_len));
    calAttnScore_valueBuf<<<grid_score, block_score, 0, stream>>>(attn_score,
                                                                  qk_buf,
                                                                  qk_buf_bd_shift,
                                                                  qk_buf_ef_seg_trans,
                                                                  attn_mask,
                                                                  off0,
                                                                  off1,
                                                                  seq_len,
                                                                  p,
                                                                  value_buf_trans,
                                                                  value_buf,
                                                                  voff0,
                                                                  v_i_off1,
                                                                  v_o_off1,
                                                                  voff2);
}

template<>
void invokeCalAttnScore<half>(int batch_size,
                              int head_num,
                              int seq_len,
                              int size_per_head,
                              float q_scaling,
                              half* attn_score,
                              half* qk_buf,
                              half* qk_buf_bd_shift,
                              half* qk_buf_ef_seg_trans,
                              half* attn_mask,
                              half* value_buf_trans,
                              half* value_buf,
                              cudaStream_t stream)
{
    int off0 = head_num * seq_len * seq_len;
    int off1 = seq_len * seq_len;
    float p = 1 / ((pow(size_per_head, 0.5)) * q_scaling);

    int voff0 = head_num * seq_len * size_per_head;
    int v_o_off1 = seq_len * size_per_head;
    int voff2 = size_per_head;
    int v_i_off1 = head_num * size_per_head;
    if (seq_len <= 32) {
        dim3 grid_score(batch_size, head_num, 2);
        dim3 block_score(seq_len / 2 * next_pow2(seq_len / 2));

        calAttnScore_valueBuf_small<<<grid_score, block_score, 0, stream>>>(attn_score,
                                                                            qk_buf,
                                                                            qk_buf_bd_shift,
                                                                            qk_buf_ef_seg_trans,
                                                                            attn_mask,
                                                                            off0,
                                                                            off1,
                                                                            seq_len,
                                                                            seq_len / 2,
                                                                            p,
                                                                            value_buf_trans,
                                                                            value_buf,
                                                                            voff0,
                                                                            v_i_off1,
                                                                            v_o_off1,
                                                                            voff2);
    }
    else if (seq_len <= 64) {
        dim3 grid_score(batch_size, head_num, seq_len / 2);
        dim3 block_score(2 * next_pow2(seq_len / 2));

        calAttnScore_valueBuf_small<<<grid_score, block_score, 0, stream>>>(attn_score,
                                                                            qk_buf,
                                                                            qk_buf_bd_shift,
                                                                            qk_buf_ef_seg_trans,
                                                                            attn_mask,
                                                                            off0,
                                                                            off1,
                                                                            seq_len,
                                                                            2,
                                                                            p,
                                                                            value_buf_trans,
                                                                            value_buf,
                                                                            voff0,
                                                                            v_i_off1,
                                                                            v_o_off1,
                                                                            voff2);
    }
    else {
        dim3 grid_score(batch_size, head_num, seq_len);
        dim3 block_score(next_pow2(seq_len / 2));

        calAttnScore_valueBuf_large<<<grid_score, block_score, 0, stream>>>(attn_score,
                                                                            qk_buf,
                                                                            qk_buf_bd_shift,
                                                                            qk_buf_ef_seg_trans,
                                                                            attn_mask,
                                                                            off0,
                                                                            off1,
                                                                            seq_len,
                                                                            p,
                                                                            value_buf_trans,
                                                                            value_buf,
                                                                            voff0,
                                                                            v_i_off1,
                                                                            v_o_off1,
                                                                            voff2);
    }
}

template<typename T>
void invokeTranspose102v2(
    int batch_size, int seq_len, int head_num, int size_per_head, T* attn_vec_trans, T* attn_vec, cudaStream_t stream)
{
    dim3 grid_trans_v(batch_size, seq_len);
    dim3 block_trans_v(head_num * size_per_head / numPerThread<T>());

    int off0 = head_num * seq_len * size_per_head;
    int i_off1 = seq_len * size_per_head;
    int o_off1 = head_num * size_per_head;
    int off2 = size_per_head;

    transpose102_v2<<<grid_trans_v, block_trans_v, 0, stream>>>(attn_vec_trans, attn_vec, off0, i_off1, o_off1, off2);
}

template<typename T>
void invokeAddResidualLayerNorm(int batch_size,
                                int seq_len,
                                int hidden_dim,
                                T* attn_layernorm,
                                T* attn_out,
                                const T* in_tensor,
                                const T* attr_layernorm_gamma,
                                const T* attr_layernorm_beta,
                                cudaStream_t stream)
{
    dim3 grid(batch_size * seq_len);
    dim3 block(hidden_dim / numPerThread<T>());
    addBias_layerNorm<T><<<grid, block, 0, stream>>>(attn_layernorm,
                                                     attn_out,
                                                     in_tensor,
                                                     attr_layernorm_gamma,
                                                     attr_layernorm_beta,
                                                     batch_size * seq_len,
                                                     hidden_dim,
                                                     epsilon);
}

template<typename T>
void invokeGelu(
    int batch_size, int seq_len, int hidden_dim_ff, T* output_fc1, const T* attr_fc1_bias, cudaStream_t stream)
{
    dim3 block(1024 / numPerThread<T>());
    dim3 grid(batch_size, seq_len);
    gelu_bias_loop<<<grid, block, 0, stream>>>(output_fc1, attr_fc1_bias, hidden_dim_ff, seq_len);
}

/*********************The explicit instantiation part***********************/
template void invokePrepareMatrixes<float>(int batch_size,
                                           int seq_len,
                                           int hidden_dim,
                                           int size_per_head,
                                           float* q_buf,
                                           float* q_buf_bd,
                                           float* q_buf_ef,
                                           float* k_buf,
                                           float* k_buf_bd,
                                           float* k_buf_ef,
                                           float* query_buf,
                                           float* key_buf,
                                           float* k_head_r,
                                           float* attr_seg_embed,
                                           float* attr_bias_Q_w,
                                           float* attr_bias_Q_r,
                                           float* attr_bias_Q_s,
                                           cudaStream_t stream);

template void invokePrepareMatrixes<half>(int batch_size,
                                          int seq_len,
                                          int hidden_dim,
                                          int size_per_head,
                                          half* q_buf,
                                          half* q_buf_bd,
                                          half* q_buf_ef,
                                          half* k_buf,
                                          half* k_buf_bd,
                                          half* k_buf_ef,
                                          half* query_buf,
                                          half* key_buf,
                                          half* k_head_r,
                                          half* attr_seg_embed,
                                          half* attr_bias_Q_w,
                                          half* attr_bias_Q_r,
                                          half* attr_bias_Q_s,
                                          cudaStream_t stream);

template void invokeTranspose102<float>(
    int batch_size, int seq_len, int head_num, float* qk_buf_ef_trans, float* qk_buf_ef, cudaStream_t stream);

template void invokeTranspose102<half>(
    int batch_size, int seq_len, int head_num, half* qk_buf_ef_trans, half* qk_buf_ef, cudaStream_t stream);

template void invokeTranspose201<float>(
    int batch_size, int seq_len, int head_num, float* qk_buf_ef_seg_trans, float* qk_buf_ef_seg, cudaStream_t stream);

template void invokeTranspose201<half>(
    int batch_size, int seq_len, int head_num, half* qk_buf_ef_seg_trans, half* qk_buf_ef_seg, cudaStream_t stream);

template void invokeRelShiftBd<float>(
    int batch_size, int head_num, int seq_len, float* qk_buf_bd_shift, float* qk_buf_bd, cudaStream_t stream);

template void invokeRelShiftBd<half>(
    int batch_size, int head_num, int seq_len, half* qk_buf_bd_shift, half* qk_buf_bd, cudaStream_t stream);

template void invokeCalAttnScore<float>(int batch_size,
                                        int head_num,
                                        int seq_len,
                                        int size_per_head,
                                        float q_scaling,
                                        float* attn_score,
                                        float* qk_buf,
                                        float* qk_buf_bd_shift,
                                        float* qk_buf_ef_seg_trans,
                                        float* attn_mask,
                                        float* value_buf_trans,
                                        float* value_buf,
                                        cudaStream_t stream);

template void invokeCalAttnScore<half>(int batch_size,
                                       int head_num,
                                       int seq_len,
                                       int size_per_head,
                                       float q_scaling,
                                       half* attn_score,
                                       half* qk_buf,
                                       half* qk_buf_bd_shift,
                                       half* qk_buf_ef_seg_trans,
                                       half* attn_mask,
                                       half* value_buf_trans,
                                       half* value_buf,
                                       cudaStream_t stream);

template void invokeTranspose102v2<float>(int batch_size,
                                          int seq_len,
                                          int head_num,
                                          int size_per_head,
                                          float* attn_vec_trans,
                                          float* attn_vec,
                                          cudaStream_t stream);
template void invokeTranspose102v2<half>(int batch_size,
                                         int seq_len,
                                         int head_num,
                                         int size_per_head,
                                         half* attn_vec_trans,
                                         half* attn_vec,
                                         cudaStream_t stream);

template void invokeAddResidualLayerNorm<float>(int batch_size,
                                                int seq_len,
                                                int hidden_dim,
                                                float* attn_layernorm,
                                                float* attn_out,
                                                const float* in_tensor,
                                                const float* attr_layernorm_gamma,
                                                const float* attr_layernorm_beta,
                                                cudaStream_t stream);

template void invokeAddResidualLayerNorm<half>(int batch_size,
                                               int seq_len,
                                               int hidden_dim,
                                               half* attn_layernorm,
                                               half* attn_out,
                                               const half* in_tensor,
                                               const half* attr_layernorm_gamma,
                                               const half* attr_layernorm_beta,
                                               cudaStream_t stream);

template void invokeGelu<float>(
    int batch_size, int seq_len, int hidden_dim_ff, float* output_fc1, const float* attr_fc1_bias, cudaStream_t stream);

template void invokeGelu<half>(
    int batch_size, int seq_len, int hidden_dim_ff, half* output_fc1, const half* attr_fc1_bias, cudaStream_t stream);

}  // namespace fastertransformer
