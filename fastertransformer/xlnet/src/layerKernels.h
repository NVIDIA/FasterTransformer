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




#pragma once

#include "utils.h"

template<typename T> void __global__ prepareMatrixes(
        T* q_buf, T* q_buf_bd, T* q_buf_ef,
        T* k_buf, T* k_buf_bd, T* k_buf_ef,
        const T* query_buf, 
        const T* key_buf, const T* k_head_r, const T* attr_seg_embed, 
        const T* attr_bias_Q_w, const T* attr_bias_Q_r, const T* attr_bias_Q_s,
        const int off0, const int i_off1,const int o_off1, int off2);
template<typename T> void __global__ transpose102(T* dst, const T* src, const int off0, 
        const int i_off1, const int o_off1, const int off2);

void __global__ transpose201(float* dst, const float* src, 
        const int off0, const int i_off1, const int head_num, const int o_off1, const int seq_len );
void __global__ transpose201(__half* dst, const __half* src,
        const int off0, const int i_off1, const int head_num, const int o_off1, const int seq_len );


template<typename T> void __global__ relShiftBd(T* outMatrix, const T* inputMatrix, 
        const int off0, const int off1, const int seq_len);

template<typename T> __global__ void calAttnScore_valueBuf(T* attn_score, const T* ac, 
        const T* bd, const T* ef, const T* attn_mask,
        const int off0, const int off1,const int seq_len, const float p,
        T* value_buf_trans, const T* value_buf, 
        const int voff0, const int v_i_off1, 
        const int v_o_off1, const int voff2);

void __global__ calAttnScore_valueBuf_small(__half* attn_score, const __half* ac, 
        const __half* bd, const __half* ef, const __half* attn_mask,
        const int off0, const int off1,const int seq_len, int n_seq1,const float p,
        __half* value_buf_trans, const __half* value_buf,
        const int voff0, const int v_i_off1, const int v_o_off1, const int voff2);
void __global__ calAttnScore_valueBuf_large(__half* attn_score, const __half* ac,
        const __half* bd, const __half* ef, const __half* attn_mask,
        const int off0, const int off1,const int seq_len, const float p,
        __half* value_buf_trans, const __half* value_buf,
        const int voff0, const int v_i_off1, const int v_o_off1, const int voff2);


template<typename T> __global__ void transpose102_v2(T* dst, const T* src, const int off0,
        const int i_off1, const int o_off1, const int off2);

template <typename T>  __global__  void addBias_layerNorm(T* out, const T* input, const T* bias, 
        const T* gamma, const T* beta, int m, int n, float epsilon);

template<typename T> __global__ void gelu_bias_loop(T *src,T* bias, int width, int height);

template <typename T> __global__  void addBias_layerNorm2(T* out, const T* input, const T* add, 
        const T* bias,const T* gamma, const T* beta, int m, int n, float epsilon);






