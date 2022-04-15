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
#include "src/fastertransformer/kernels/layout_transformer_int8_kernels.h"
#include "src/fastertransformer/layers/attention_layers_int8/FusedAttentionLayerINT8.h"

namespace fastertransformer {

// add bias and then transform from
// 3 * ([valid_word_num, head*size] + CUBLASLT_ORDER_COL32) -> [valid_word_num, head, 3, size] + row-major
// input is INT32 && per axis quantization for weight
// output is INT8 && per tensor quantization
// grid((head*size + 31)/32, (valid_word_num + 31)/32, 3)
// block(8, 32)
// size should be a multiple of 4
// using char4 as output, int4 as input
template<typename T>
__global__ void trt_add_QKV_bias_COL32_int32IInt8O(char4* output,
                                                   const int4* QKV,
                                                   const T* bias_Q,
                                                   const T* bias_K,
                                                   const T* bias_V,
                                                   const float* input_deQFactor_div127_ptr,
                                                   const float* q_weight_amax,
                                                   const float* k_weight_amax,
                                                   const float* v_weight_amax,
                                                   const float qkv_output_scale,
                                                   const int valid_word_num,
                                                   const int head_num,
                                                   const int size_per_head,
                                                   const int head_num_x_size_per_head)
{
    const int qkv_id = blockIdx.z;
    const int seq_id = (blockIdx.y << 5) + threadIdx.y;
    const int threadIdx4 = threadIdx.x << 2;
    int hidden_id = (blockIdx.x << 5) + threadIdx4;
    const int size_id = hidden_id % size_per_head;
    const int head_id = hidden_id / size_per_head;

    const bool qual = (seq_id < valid_word_num) && (hidden_id < head_num_x_size_per_head);
    if (qual) {
        const float* weight_amax = qkv_id == 0 ? q_weight_amax : (qkv_id == 1 ? k_weight_amax : v_weight_amax);
        const float input_deQFactor_div127 = __ldg(input_deQFactor_div127_ptr);

        const T* bias_ptr = (qkv_id == 0) ? bias_Q : ((qkv_id == 1) ? bias_K : bias_V);

        const int input_id = (qkv_id * valid_word_num * head_num_x_size_per_head
                              + ((hidden_id & 0xffffffe0) * valid_word_num + (seq_id << 5) + (hidden_id & 31)))
                             >> 2;

        char4 tmp;
        const int4 tmp_int4 = __ldg(QKV + input_id);

        tmp.x =
            float_to_int8_rn((static_cast<float>(tmp_int4.x) * __ldg(weight_amax + hidden_id) * input_deQFactor_div127
                              + static_cast<float>(__ldg(bias_ptr + hidden_id)))
                             * qkv_output_scale);

        hidden_id += 1;
        tmp.y =
            float_to_int8_rn((static_cast<float>(tmp_int4.y) * __ldg(weight_amax + hidden_id) * input_deQFactor_div127
                              + static_cast<float>(__ldg(bias_ptr + hidden_id)))
                             * qkv_output_scale);

        hidden_id += 1;
        tmp.z =
            float_to_int8_rn((static_cast<float>(tmp_int4.z) * __ldg(weight_amax + hidden_id) * input_deQFactor_div127
                              + static_cast<float>(__ldg(bias_ptr + hidden_id)))
                             * qkv_output_scale);

        hidden_id += 1;
        tmp.w =
            float_to_int8_rn((static_cast<float>(tmp_int4.w) * __ldg(weight_amax + hidden_id) * input_deQFactor_div127
                              + static_cast<float>(__ldg(bias_ptr + hidden_id)))
                             * qkv_output_scale);

        // const int output_id = (seq_id * 3 * head_num_x_size_per_head + head_id * 3 * size_per_head + qkv_id *
        // size_per_head + size_id) >> 2;
        const int output_id =
            ((seq_id * head_num_x_size_per_head + head_id * size_per_head) * 3 + qkv_id * size_per_head + size_id) >> 2;

        output[output_id] = tmp;
    }
}

template<typename T>
void invokeTrtAddQkvBiasInt32Iint8O(int8_t* output,
                                    const int32_t* Q,
                                    const T* bias_Q,
                                    const T* bias_K,
                                    const T* bias_V,
                                    const size_t token_num,
                                    const size_t head_num,
                                    const size_t size_per_head,
                                    const float* input_deQFactor_div127_ptr,
                                    const float* q_weight_amax,
                                    const float* k_weight_amax,
                                    const float* v_weight_amax,
                                    const float mScaleQkv,
                                    const cudaStream_t stream)
{

    int head_num_x_size_per_head = head_num * size_per_head;
    dim3 grid((head_num_x_size_per_head + 31) / 32, (token_num + 31) / 32, 3);
    dim3 block(8, 32);

    assert(size_per_head % 4 == 0);

    trt_add_QKV_bias_COL32_int32IInt8O<<<grid, block, 0, stream>>>((char4*)output,
                                                                   (const int4*)Q,
                                                                   bias_Q,
                                                                   bias_K,
                                                                   bias_V,
                                                                   input_deQFactor_div127_ptr,
                                                                   q_weight_amax,
                                                                   k_weight_amax,
                                                                   v_weight_amax,
                                                                   1.f / mScaleQkv,
                                                                   token_num,
                                                                   head_num,
                                                                   size_per_head,
                                                                   head_num_x_size_per_head);
}

// Add bias, and then transform from
// 3 * ([valid_word_num, head*size] + CUBLASLT_ORDER_COL32) -> [valid_word_num, head, 3, size] + row-major
// grid((head*size + 31)/32, (valid_word_num + 31)/32, 3)
// block(8, 32)
// size should be a multiple of 4
template<typename T>
__global__ void trt_add_QKV_bias_COL32_int8IO(char4* output,
                                              const char4* QKV,
                                              const T* bias_Q,
                                              const T* bias_K,
                                              const T* bias_V,
                                              const float* q_input_deQFactor_ptr,
                                              const float* k_input_deQFactor_ptr,
                                              const float* v_input_deQFactor_ptr,
                                              const float qkv_output_scale,
                                              const int valid_word_num,
                                              const int head_num,
                                              const int size_per_head,
                                              const int head_num_x_size_per_head)
{
    const int qkv_id = blockIdx.z;
    const int seq_id = (blockIdx.y << 5) + threadIdx.y;
    const int threadIdx4 = threadIdx.x << 2;
    const int hidden_id = (blockIdx.x << 5) + threadIdx4;
    const int size_id = hidden_id % size_per_head;
    const int head_id = hidden_id / size_per_head;

    const bool qual = (seq_id < valid_word_num) && (hidden_id < head_num_x_size_per_head);
    if (qual) {
        const float* input_deQFactor_ptr =
            (qkv_id == 0) ? q_input_deQFactor_ptr : ((qkv_id == 1) ? k_input_deQFactor_ptr : v_input_deQFactor_ptr);
        const float input_deQFactor = __ldg(input_deQFactor_ptr);

        const T* bias_ptr = (qkv_id == 0) ? bias_Q : ((qkv_id == 1) ? bias_K : bias_V);

        const int input_id = (qkv_id * valid_word_num * head_num_x_size_per_head
                              + ((hidden_id & 0xffffffe0) * valid_word_num + (seq_id << 5) + (hidden_id & 31)))
                             >> 2;

        char4 tmp = __ldg(QKV + input_id);

        tmp.x = float_to_int8_rn(
            (static_cast<float>(tmp.x) * input_deQFactor + static_cast<float>(__ldg(bias_ptr + hidden_id)))
            * qkv_output_scale);

        tmp.y = float_to_int8_rn(
            (static_cast<float>(tmp.y) * input_deQFactor + static_cast<float>(__ldg(bias_ptr + hidden_id + 1)))
            * qkv_output_scale);

        tmp.z = float_to_int8_rn(
            (static_cast<float>(tmp.z) * input_deQFactor + static_cast<float>(__ldg(bias_ptr + hidden_id + 2)))
            * qkv_output_scale);

        tmp.w = float_to_int8_rn(
            (static_cast<float>(tmp.w) * input_deQFactor + static_cast<float>(__ldg(bias_ptr + hidden_id + 3)))
            * qkv_output_scale);

        // const int output_id = (seq_id * 3 * head_num_x_size_per_head + head_id * 3 * size_per_head + qkv_id *
        // size_per_head + size_id) >> 2;
        const int output_id =
            ((seq_id * head_num_x_size_per_head + head_id * size_per_head) * 3 + qkv_id * size_per_head + size_id) >> 2;

        output[output_id] = tmp;
    }
}

template<typename T>
void invokeTrtAddQkvBiasInt8IO(int8_t* output,
                               const int8_t* Q,
                               const T* bias_Q,
                               const T* bias_K,
                               const T* bias_V,
                               const size_t token_num,
                               const size_t head_num,
                               const size_t size_per_head,
                               const float* q_input_deQFactor_ptr,
                               const float* k_input_deQFactor_ptr,
                               const float* v_input_deQFactor_ptr,
                               const float mScaleQkv,
                               const cudaStream_t stream)
{

    int head_num_x_size_per_head = head_num * size_per_head;
    dim3 grid((head_num_x_size_per_head + 31) / 32, (token_num + 31) / 32, 3);
    dim3 block(8, 32);

    assert(size_per_head % 4 == 0);

    trt_add_QKV_bias_COL32_int8IO<<<grid, block, 0, stream>>>((char4*)output,
                                                              (const char4*)Q,
                                                              bias_Q,
                                                              bias_K,
                                                              bias_V,
                                                              q_input_deQFactor_ptr,
                                                              k_input_deQFactor_ptr,
                                                              v_input_deQFactor_ptr,
                                                              1.0f / mScaleQkv,
                                                              token_num,
                                                              head_num,
                                                              size_per_head,
                                                              head_num_x_size_per_head);
}

// Add bias, and then transform from
// 3 * ([valid_word_num, head*size] + row-major) -> [valid_word_num, head, 3, size] + row-major
// grid((head*size + 31)/32, (valid_word_num + 31)/32, 3)
// block(8, 32)
// size should be a multiple of 4
template<typename T>
__global__ void trt_add_QKV_bias_ROW_int8IO(char4* output,
                                            const char4* QKV,
                                            const T* bias_Q,
                                            const T* bias_K,
                                            const T* bias_V,
                                            const float* q_input_deQFactor_ptr,
                                            const float* k_input_deQFactor_ptr,
                                            const float* v_input_deQFactor_ptr,
                                            const float qkv_output_scale,
                                            const int valid_word_num,
                                            const int head_num,
                                            const int size_per_head,
                                            const int head_num_x_size_per_head)
{
    const int qkv_id = blockIdx.z;
    const int seq_id = (blockIdx.y << 5) + threadIdx.y;
    const int threadIdx4 = threadIdx.x << 2;
    const int hidden_id = (blockIdx.x << 5) + threadIdx4;
    const int size_id = hidden_id % size_per_head;
    const int head_id = hidden_id / size_per_head;

    const bool qual = (seq_id < valid_word_num) && (hidden_id < head_num_x_size_per_head);
    if (qual) {
        const float* input_deQFactor_ptr =
            (qkv_id == 0) ? q_input_deQFactor_ptr : ((qkv_id == 1) ? k_input_deQFactor_ptr : v_input_deQFactor_ptr);
        const float input_deQFactor = __ldg(input_deQFactor_ptr);

        const T* bias_ptr = (qkv_id == 0) ? bias_Q : ((qkv_id == 1) ? bias_K : bias_V);

        const int input_id =
            (qkv_id * valid_word_num * head_num_x_size_per_head + seq_id * head_num_x_size_per_head + hidden_id) >> 2;

        char4 tmp = __ldg(QKV + input_id);

        tmp.x = float_to_int8_rn(
            (static_cast<float>(tmp.x) * input_deQFactor + static_cast<float>(__ldg(bias_ptr + hidden_id)))
            * qkv_output_scale);

        tmp.y = float_to_int8_rn(
            (static_cast<float>(tmp.y) * input_deQFactor + static_cast<float>(__ldg(bias_ptr + hidden_id + 1)))
            * qkv_output_scale);

        tmp.z = float_to_int8_rn(
            (static_cast<float>(tmp.z) * input_deQFactor + static_cast<float>(__ldg(bias_ptr + hidden_id + 2)))
            * qkv_output_scale);

        tmp.w = float_to_int8_rn(
            (static_cast<float>(tmp.w) * input_deQFactor + static_cast<float>(__ldg(bias_ptr + hidden_id + 3)))
            * qkv_output_scale);

        const int output_id =
            ((seq_id * head_num_x_size_per_head + head_id * size_per_head) * 3 + qkv_id * size_per_head + size_id) >> 2;

        output[output_id] = tmp;
    }
}

template<typename T>
void invokeTrtAddQkvBiasInt8IORow(int8_t* output,
                                  const int8_t* Q,
                                  const T* bias_Q,
                                  const T* bias_K,
                                  const T* bias_V,
                                  const size_t token_num,
                                  const size_t head_num,
                                  const size_t size_per_head,
                                  const float* q_input_deQFactor_ptr,
                                  const float* k_input_deQFactor_ptr,
                                  const float* v_input_deQFactor_ptr,
                                  const float mScaleQkv,
                                  const cudaStream_t stream)
{

    int head_num_x_size_per_head = head_num * size_per_head;
    dim3 grid((head_num_x_size_per_head + 31) / 32, (token_num + 31) / 32, 3);
    dim3 block(8, 32);

    assert(size_per_head % 4 == 0);

    trt_add_QKV_bias_ROW_int8IO<<<grid, block, 0, stream>>>((char4*)output,
                                                            (const char4*)Q,
                                                            bias_Q,
                                                            bias_K,
                                                            bias_V,
                                                            q_input_deQFactor_ptr,
                                                            k_input_deQFactor_ptr,
                                                            v_input_deQFactor_ptr,
                                                            1.0f / mScaleQkv,
                                                            token_num,
                                                            head_num,
                                                            size_per_head,
                                                            head_num_x_size_per_head);
}

template<typename T>
void FusedAttentionLayerINT8<T>::forward(std::vector<fastertransformer::Tensor>* output_tensors,
                                         const std::vector<fastertransformer::Tensor>* input_tensors,
                                         const AttentionWeight<T>* attention_weights)
{
    // input_tensors: [input (token_num, hidden_dimension),
    //                 attention_mask (batch, 1, seqlen, seqlen),
    //                 padding_offset (token_num)]
    // output_tensors: [output (token_num, hidden_dimension)]
    // If padding_offset.data is nullptr, then not remove padding

    const ScaleList* scale_list = ((const AttentionINT8Weight<T>*)attention_weights)->scale_list_ptr;
    cublasINT8MMWrapper* cublas_wrapper = (cublasINT8MMWrapper*)cublas_wrapper_;

    // input_tensors: [input_query (token_num, hidden_dimension),
    //                 attention_mask (batch, 1, seqlen, seqlen),
    //                 padding_offset (batch + 1 or batch * 2 + 1))]
    // If padding_offset.data is nullptr, then not remove padding

    FT_CHECK(isValidBatchSize(input_tensors->at(1).shape[0]));
    FT_CHECK(isValidSeqLen(input_tensors->at(1).shape[2]));
    allocateBuffer();

    int32_t* attention_out = (int32_t*)output_tensors->at(0).data;
    const int8_t* from_tensor = (const int8_t*)input_tensors->at(0).data;
    const T* attention_mask = (const T*)input_tensors->at(1).data;
    const int* padding_offset = (const int*)input_tensors->at(2).data;

    const int request_batch_size = input_tensors->at(1).shape[0];
    const int request_seq_len = input_tensors->at(1).shape[2];
    const int m = input_tensors->at(0).shape[0];
    const int k = hidden_units_;
    const int n = hidden_units_;
#ifdef SPARSITY_ENABLED
    int m_tmp = m;
    if (m_tmp % 16 != 0) {
        m_tmp = (m_tmp / 16 + 1) * 16;
    }
    const int m_padded = m_tmp;
#endif

    const int fusedINT8QKV_type = cublas_wrapper->getFusedINT8QKVType(k, n, attention_weights);
    if (int8_mode_ == 1) {
        // K_int_buf_ V_int_buf_ should point to correct buffer according to m
        K_int_buf_ = (int*)Q_int_buf_ + m * head_num_ * size_per_head_;
        V_int_buf_ = (int*)K_int_buf_ + m * head_num_ * size_per_head_;
        if (fusedINT8QKV_type == 0) {
            cublas_wrapper->Gemm(
                Q_int_buf_, 1, m, n, k, 0, 0, 0, from_tensor, (int8_t*)(attention_weights->query_weight.kernel));
            cublas_wrapper->Gemm(
                K_int_buf_, 1, m, n, k, 0, 0, 0, from_tensor, (int8_t*)(attention_weights->key_weight.kernel));
            cublas_wrapper->Gemm(
                V_int_buf_, 1, m, n, k, 0, 0, 0, from_tensor, (int8_t*)(attention_weights->value_weight.kernel));
        }
        else {
            int strideFactor = (fusedINT8QKV_type == 1) ? (sizeof(T) / sizeof(int8_t)) : 1;
            cublas_wrapper->Gemm(Q_int_buf_,
                                 3,
                                 m,
                                 n,
                                 k,
                                 0,
                                 n * k * strideFactor,
                                 n * m,
                                 from_tensor,
                                 (int8_t*)(attention_weights->query_weight.kernel));
        }
        invokeTrtAddQkvBiasInt32Iint8O(qkv_buf_,
                                       Q_int_buf_,
                                       attention_weights->query_weight.bias,
                                       attention_weights->key_weight.bias,
                                       attention_weights->value_weight.bias,
                                       m,
                                       head_num_,
                                       size_per_head_,
                                       &(scale_list->d_scale_list_[2]),
                                       &(scale_list->d_scale_list_[scale_list->p2_offset_]),
                                       &(scale_list->d_scale_list_[scale_list->p2_offset_ + hidden_units_]),
                                       &(scale_list->d_scale_list_[scale_list->p2_offset_ + 2 * hidden_units_]),
                                       scale_list->h_scale_list_[scale_list->p4_offset_] / 127.0f,
                                       stream_);
    }
    else if (int8_mode_ == 2 || int8_mode_ == 3) {
        // K_int_buf_ V_int_buf_ should point to correct buffer according to m
        K_int_buf_ = (int*)((int8_t*)Q_int_buf_ + m * head_num_ * size_per_head_);
        V_int_buf_ = (int*)((int8_t*)K_int_buf_ + m * head_num_ * size_per_head_);

#ifdef SPARSITY_ENABLED
        if (sparse_) {
            cublas_wrapper->SpGemm(n,
                                   m_padded,
                                   k,
                                   scale_list->h_scale_list_[scale_list->p3_offset_ + 0],
                                   (int8_t*)(attention_weights->query_weight.sp_kernel),
                                   from_tensor,
                                   (int8_t*)Q_int_buf_);
            cublas_wrapper->SpGemm(n,
                                   m_padded,
                                   k,
                                   scale_list->h_scale_list_[scale_list->p3_offset_ + 1],
                                   (int8_t*)(attention_weights->key_weight.sp_kernel),
                                   from_tensor,
                                   (int8_t*)K_int_buf_);
            cublas_wrapper->SpGemm(n,
                                   m_padded,
                                   k,
                                   scale_list->h_scale_list_[scale_list->p3_offset_ + 2],
                                   (int8_t*)(attention_weights->value_weight.sp_kernel),
                                   from_tensor,
                                   (int8_t*)V_int_buf_);
        }
        else {
#endif
            if (fusedINT8QKV_type == 0) {
                cublas_wrapper->Gemm((int8_t*)Q_int_buf_,
                                     1,
                                     m,
                                     n,
                                     k,
                                     0,
                                     0,
                                     0,
                                     scale_list->h_scale_list_[scale_list->p3_offset_ + 0],
                                     from_tensor,
                                     (int8_t*)(attention_weights->query_weight.kernel));
                cublas_wrapper->Gemm((int8_t*)K_int_buf_,
                                     1,
                                     m,
                                     n,
                                     k,
                                     0,
                                     0,
                                     0,
                                     scale_list->h_scale_list_[scale_list->p3_offset_ + 1],
                                     from_tensor,
                                     (int8_t*)(attention_weights->key_weight.kernel));
                cublas_wrapper->Gemm((int8_t*)V_int_buf_,
                                     1,
                                     m,
                                     n,
                                     k,
                                     0,
                                     0,
                                     0,
                                     scale_list->h_scale_list_[scale_list->p3_offset_ + 2],
                                     from_tensor,
                                     (int8_t*)(attention_weights->value_weight.kernel));
            }
            else {
                int strideFactor = (fusedINT8QKV_type == 1) ? (sizeof(T) / sizeof(int8_t)) : 1;
                cublas_wrapper->Gemm((int8_t*)Q_int_buf_,
                                     3,
                                     m,
                                     n,
                                     k,
                                     0,
                                     n * k * strideFactor,
                                     n * m,
                                     scale_list->h_scale_list_[scale_list->p3_offset_ + 0],
                                     from_tensor,
                                     (int8_t*)(attention_weights->query_weight.kernel));
            }
#ifdef SPARSITY_ENABLED
        }
        if (sparse_) {
            invokeTrtAddQkvBiasInt8IORow(qkv_buf_,
                                         (int8_t*)Q_int_buf_,
                                         attention_weights->query_weight.bias,
                                         attention_weights->key_weight.bias,
                                         attention_weights->value_weight.bias,
                                         m,
                                         head_num_,
                                         size_per_head_,
                                         &(scale_list->d_scale_list_[4 + 1]),
                                         &(scale_list->d_scale_list_[12 + 1]),
                                         &(scale_list->d_scale_list_[20 + 1]),
                                         scale_list->h_scale_list_[scale_list->p4_offset_] / 127.0f,
                                         stream_);
        }
        else {
#endif
            invokeTrtAddQkvBiasInt8IO(qkv_buf_,
                                      (int8_t*)Q_int_buf_,
                                      attention_weights->query_weight.bias,
                                      attention_weights->key_weight.bias,
                                      attention_weights->value_weight.bias,
                                      m,
                                      head_num_,
                                      size_per_head_,
                                      &(scale_list->d_scale_list_[4 + 1]),
                                      &(scale_list->d_scale_list_[12 + 1]),
                                      &(scale_list->d_scale_list_[20 + 1]),
                                      scale_list->h_scale_list_[scale_list->p4_offset_] / 127.0f,
                                      stream_);
#ifdef SPARSITY_ENABLED
        }
#endif
    }

    int S = dispatcher_int8_->getSFromMaxSeqLen(request_seq_len);
    FT_CHECK(dispatcher_int8_->isValid(S));
    const int B = input_tensors->at(2).shape[0] - 1;
    // setScaleList should be executed before setup
    dispatcher_int8_->setScaleList(scale_list->h_scale_list_[scale_list->p4_offset_] / 127.0f,
                                   scale_list->h_scale_list_[scale_list->p4_offset_ + 1] / 127.0f,
                                   scale_list->h_scale_list_[scale_list->p4_offset_ + 2] / 127.0f);
    dispatcher_int8_->setup(S, B);
    dispatcher_int8_->run(qkv_buf_, nullptr, (int*)input_tensors->at(2).data, attn_workspace_, qkv_buf_2_, stream_);
    sync_check_cuda_error();

#ifdef SPARSITY_ENABLED
    if (!sparse_) {
#endif
        // qkv_buf_2_ is [batch*seqlen, hidden_dim] row-major
        invokeRowMajorToCOL32(qkv_buf_, qkv_buf_2_, m, hidden_units_, stream_);
#ifdef SPARSITY_ENABLED
    }
#endif

    if (int8_mode_ == 1) {
        cublas_wrapper->Gemm(
            attention_out, 1, m, n, k, 0, 0, 0, qkv_buf_, (int8_t*)(attention_weights->attention_output_weight.kernel));
    }
    else if (int8_mode_ == 2 || int8_mode_ == 3) {
#ifdef SPARSITY_ENABLED
        if (sparse_) {
            cublas_wrapper->SpGemm(n,
                                   m_padded,
                                   k,
                                   scale_list->h_scale_list_[scale_list->p3_offset_ + 5],
                                   (int8_t*)(attention_weights->attention_output_weight.sp_kernel),
                                   qkv_buf_2_,
                                   (int8_t*)attention_out);
        }
        else {
#endif
            cublas_wrapper->Gemm((int8_t*)attention_out,
                                 1,
                                 m,
                                 n,
                                 k,
                                 0,
                                 0,
                                 0,
                                 scale_list->h_scale_list_[scale_list->p3_offset_ + 5],
                                 qkv_buf_,
                                 (int8_t*)(attention_weights->attention_output_weight.kernel));
#ifdef SPARSITY_ENABLED
        }
#endif
    }
    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
FusedAttentionLayerINT8<T>::FusedAttentionLayerINT8(size_t max_batch_size,
                                                    size_t max_seq_len,
                                                    size_t head_num,
                                                    size_t size_per_head,
                                                    int sm,
                                                    float q_scaling,
                                                    int int8_mode,
                                                    cudaStream_t stream,
                                                    cublasMMWrapper* cublas_wrapper,
                                                    IAllocator* allocator,
                                                    bool is_free_buffer_after_forward,
                                                    bool sparse):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    sm_(sm),
    q_scaling_(q_scaling),
    int8_mode_(int8_mode),
    sparse_(sparse)
{
    if ((sm_ == kSM_86 || sm_ == kSM_80 || sm_ == kSM_75 || sm_ == kSM_72) && size_per_head_ == 64) {
        dispatcher_int8_.reset(new FusedMHARunnerInt8v2(head_num_, size_per_head_, sm_, q_scaling_));
    }
    else {
        throw std::runtime_error(std::string("[FT][ERROR] FusedAttentionLayerINT8 not support \n"));
    }
    hidden_units_ = head_num_ * size_per_head_;
}

template<typename T>
FusedAttentionLayerINT8<T>::FusedAttentionLayerINT8(FusedAttentionLayerINT8<T> const& attention_layer):
    BaseAttentionLayer<T>(attention_layer.stream_,
                          attention_layer.cublas_wrapper_,
                          attention_layer.allocator_,
                          attention_layer.is_free_buffer_after_forward_),
    max_batch_size_(attention_layer.max_batch_size_),
    max_seq_len_(attention_layer.max_seq_len_),
    head_num_(attention_layer.head_num_),
    size_per_head_(attention_layer.size_per_head_),
    hidden_units_(attention_layer.hidden_units_),
    sm_(attention_layer.sm_),
    q_scaling_(attention_layer.q_scaling_),
    int8_mode_(attention_layer.int8_mode_),
    sparse_(attention_layer.sparse_)
{
    if ((sm_ == kSM_86 || sm_ == kSM_80 || sm_ == kSM_75 || sm_ == kSM_72) && size_per_head_ == 64) {
        dispatcher_int8_.reset(new FusedMHARunnerInt8v2(head_num_, size_per_head_, sm_, q_scaling_));
    }
    else {
        throw std::runtime_error(std::string("[FT][ERROR] FusedAttentionLayerINT8 not support \n"));
    }
}

template<typename T>
FusedAttentionLayerINT8<T>::~FusedAttentionLayerINT8()
{
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void FusedAttentionLayerINT8<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        Q_int_buf_ =
            (int32_t*)allocator_->malloc(sizeof(int32_t) * max_batch_size_ * max_seq_len_ * hidden_units_ * 3, false);
        K_int_buf_ = Q_int_buf_ + max_batch_size_ * max_seq_len_ * hidden_units_;
        V_int_buf_ = K_int_buf_ + max_batch_size_ * max_seq_len_ * hidden_units_;
        qkv_buf_ = (int8_t*)allocator_->malloc(
            (sizeof(int8_t) * 3 * max_batch_size_ * max_seq_len_ * hidden_units_ + 3) / 4 * 4, false);
        qkv_buf_2_ = (int8_t*)allocator_->malloc(
            (sizeof(int8_t) * max_batch_size_ * max_seq_len_ * hidden_units_ + 3) / 4 * 4, false);
        attn_workspace_ = (T*)allocator_->malloc(dispatcher_int8_->getWorkspaceSize(), false);

        is_allocate_buffer_ = true;
    }
}

template<typename T>
void FusedAttentionLayerINT8<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free(Q_int_buf_);
        allocator_->free(qkv_buf_);
        allocator_->free(qkv_buf_2_);
        allocator_->free(attn_workspace_);

        is_allocate_buffer_ = false;
        sync_check_cuda_error();
    }
}

template<typename T>
bool FusedAttentionLayerINT8<T>::isValidBatchSize(size_t batch_size)
{
    if (max_batch_size_ == 0) {
        max_batch_size_ = batch_size;
        return true;
    }
    else {
        return batch_size <= max_batch_size_;
    }
}

template<typename T>
bool FusedAttentionLayerINT8<T>::isValidSeqLen(size_t seq_len)
{
    if (max_seq_len_ == 0) {
        max_seq_len_ = seq_len;
    }
    return seq_len <= max_seq_len_ && seq_len <= 384;
}

template class FusedAttentionLayerINT8<float>;
template class FusedAttentionLayerINT8<half>;

}  // namespace fastertransformer
