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

#include "src/fastertransformer/layers/attention_layers_int8/UnfusedAttentionLayerINT8.h"
#include "src/fastertransformer/kernels/softmax_int8_kernels.h"
#include "src/fastertransformer/kernels/transpose_int8_kernels.h"
#include "src/fastertransformer/kernels/unfused_attention_int8_kernels.h"

namespace fastertransformer {

template<typename T>
void UnfusedAttentionLayerINT8<T>::forward(std::vector<fastertransformer::Tensor>* output_tensors,
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
    int m_tmp = m;
    if (m_tmp % 16 != 0) {
        m_tmp = (m_tmp / 16 + 1) * 16;
    }
#ifdef SPARSITY_ENABLED
    const int m_padded = m_tmp;
#endif

    if (size_per_head_ % 32 != 0) {
        printf(
            "[ERROR][FT][UnfusedAttentionLayerINT8::forward] int8 unfused mha kernel only works when size_per_head is a multiple of 32.\n");
        exit(-1);
    }

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
#endif
    }

    const int seq_len_padded = (request_seq_len + 31) / 32 * 32;
    if (padding_offset == nullptr) {
        if (int8_mode_ == 1) {
            invokeAddQKBiasTransform(q_buf_,
                                     k_buf_,
                                     Q_int_buf_,
                                     attention_weights->query_weight.bias,
                                     K_int_buf_,
                                     attention_weights->key_weight.bias,
                                     request_batch_size,
                                     request_seq_len,
                                     head_num_,
                                     size_per_head_,
                                     &(scale_list->d_scale_list_[scale_list->p2_offset_]),
                                     &(scale_list->d_scale_list_[2]),
                                     &(scale_list->d_scale_list_[scale_list->p2_offset_ + hidden_units_]),
                                     &(scale_list->d_scale_list_[2]),
                                     &(scale_list->d_scale_list_[8 + 3]),
                                     &(scale_list->d_scale_list_[16 + 3]),
                                     cublas_wrapper->getUseOrderCol322R4R4(),
                                     stream_);
            invokeAddVBiasTransform(v_buf_,
                                    V_int_buf_,
                                    attention_weights->value_weight.bias,
                                    request_batch_size,
                                    request_seq_len,
                                    head_num_,
                                    size_per_head_,
                                    &(scale_list->d_scale_list_[scale_list->p2_offset_ + 2 * hidden_units_]),
                                    &(scale_list->d_scale_list_[2]),
                                    &(scale_list->d_scale_list_[24 + 3]),
                                    cublas_wrapper->getUseOrderCol322R4R4(),
                                    stream_);
        }
        else if (int8_mode_ == 2 || int8_mode_ == 3) {
#ifdef SPARSITY_ENABLED
            if (sparse_) {
                invokeAddQKBiasTransformRow(q_buf_,
                                            k_buf_,
                                            (const int8_t*)Q_int_buf_,
                                            attention_weights->query_weight.bias,
                                            (const int8_t*)K_int_buf_,
                                            attention_weights->key_weight.bias,
                                            request_batch_size,
                                            request_seq_len,
                                            head_num_,
                                            size_per_head_,
                                            &(scale_list->d_scale_list_[4 + 1]),
                                            &(scale_list->d_scale_list_[12 + 1]),
                                            &(scale_list->d_scale_list_[8 + 3]),
                                            &(scale_list->d_scale_list_[16 + 3]),
                                            cublas_wrapper->getUseOrderCol322R4R4(),
                                            stream_);
                invokeAddVBiasTransformRow(v_buf_,
                                           (const int8_t*)V_int_buf_,
                                           attention_weights->value_weight.bias,
                                           request_batch_size,
                                           request_seq_len,
                                           head_num_,
                                           size_per_head_,
                                           &(scale_list->d_scale_list_[20 + 1]),
                                           &(scale_list->d_scale_list_[24 + 3]),
                                           cublas_wrapper->getUseOrderCol322R4R4(),
                                           stream_);
            }
            else {
#endif
                invokeAddQKBiasTransform(q_buf_,
                                         k_buf_,
                                         (const int8_t*)Q_int_buf_,
                                         attention_weights->query_weight.bias,
                                         (const int8_t*)K_int_buf_,
                                         attention_weights->key_weight.bias,
                                         request_batch_size,
                                         request_seq_len,
                                         head_num_,
                                         size_per_head_,
                                         &(scale_list->d_scale_list_[4 + 1]),
                                         &(scale_list->d_scale_list_[12 + 1]),
                                         &(scale_list->d_scale_list_[8 + 3]),
                                         &(scale_list->d_scale_list_[16 + 3]),
                                         cublas_wrapper->getUseOrderCol322R4R4(),
                                         stream_);
                invokeAddVBiasTransform(v_buf_,
                                        (const int8_t*)V_int_buf_,
                                        attention_weights->value_weight.bias,
                                        request_batch_size,
                                        request_seq_len,
                                        head_num_,
                                        size_per_head_,
                                        &(scale_list->d_scale_list_[20 + 1]),
                                        &(scale_list->d_scale_list_[24 + 3]),
                                        cublas_wrapper->getUseOrderCol322R4R4(),
                                        stream_);
#ifdef SPARSITY_ENABLED
            }
#endif
        }
        sync_check_cuda_error();
    }
    else {
        invokeMappingRemovePaddingData(
            request_batch_size, request_seq_len, m, sequence_id_map_, padding_offset, stream_);
        // if we use remove padding, then initialize the q_buf_, k_buf_ and v_buf_ to prevent bugs. v_buf_ will be
        // properly initiaized in invokeAddVBiasTransformRebuildPadding()
        cudaMemsetAsync(
            q_buf_, 0, 2 * request_batch_size * seq_len_padded * head_num_ * size_per_head_ * sizeof(int8_t), stream_);
        if (int8_mode_ == 1) {

            invokeAddQKBiasTransformRebuildPadding(q_buf_,
                                                   k_buf_,
                                                   Q_int_buf_,
                                                   attention_weights->query_weight.bias,
                                                   K_int_buf_,
                                                   attention_weights->key_weight.bias,
                                                   padding_offset,
                                                   m,
                                                   request_batch_size,
                                                   request_seq_len,
                                                   head_num_,
                                                   size_per_head_,
                                                   &(scale_list->d_scale_list_[scale_list->p2_offset_]),
                                                   &(scale_list->d_scale_list_[2]),
                                                   &(scale_list->d_scale_list_[scale_list->p2_offset_ + hidden_units_]),
                                                   &(scale_list->d_scale_list_[2]),
                                                   &(scale_list->d_scale_list_[8 + 3]),
                                                   &(scale_list->d_scale_list_[16 + 3]),
                                                   cublas_wrapper->getUseOrderCol322R4R4(),
                                                   stream_);

            invokeAddVBiasTransformRebuildPadding(
                v_buf_,
                V_int_buf_,
                attention_weights->value_weight.bias,
                sequence_id_map_,
                m,
                request_batch_size,
                request_seq_len,
                head_num_,
                size_per_head_,
                &(scale_list->d_scale_list_[scale_list->p2_offset_ + 2 * hidden_units_]),
                &(scale_list->d_scale_list_[2]),
                &(scale_list->d_scale_list_[24 + 3]),
                cublas_wrapper->getUseOrderCol322R4R4(),
                stream_);
        }
        else if (int8_mode_ == 2 || int8_mode_ == 3) {
#ifdef SPARSITY_ENABLED
            if (sparse_) {
                invokeAddQKBiasTransformRebuildPaddingRow(q_buf_,
                                                          k_buf_,
                                                          (const int8_t*)Q_int_buf_,
                                                          attention_weights->query_weight.bias,
                                                          (const int8_t*)K_int_buf_,
                                                          attention_weights->key_weight.bias,
                                                          padding_offset,
                                                          m,
                                                          request_batch_size,
                                                          request_seq_len,
                                                          head_num_,
                                                          size_per_head_,
                                                          &(scale_list->d_scale_list_[4 + 1]),
                                                          &(scale_list->d_scale_list_[12 + 1]),
                                                          &(scale_list->d_scale_list_[8 + 3]),
                                                          &(scale_list->d_scale_list_[16 + 3]),
                                                          cublas_wrapper->getUseOrderCol322R4R4(),
                                                          stream_);
                invokeAddVBiasTransformRebuildPaddingRow(v_buf_,
                                                         (const int8_t*)V_int_buf_,
                                                         attention_weights->value_weight.bias,
                                                         sequence_id_map_,
                                                         m,
                                                         request_batch_size,
                                                         request_seq_len,
                                                         head_num_,
                                                         size_per_head_,
                                                         &(scale_list->d_scale_list_[20 + 1]),
                                                         &(scale_list->d_scale_list_[24 + 3]),
                                                         cublas_wrapper->getUseOrderCol322R4R4(),
                                                         stream_);
            }
            else {
#endif
                invokeAddQKBiasTransformRebuildPadding(q_buf_,
                                                       k_buf_,
                                                       (const int8_t*)Q_int_buf_,
                                                       attention_weights->query_weight.bias,
                                                       (const int8_t*)K_int_buf_,
                                                       attention_weights->key_weight.bias,
                                                       padding_offset,
                                                       m,
                                                       request_batch_size,
                                                       request_seq_len,
                                                       head_num_,
                                                       size_per_head_,
                                                       &(scale_list->d_scale_list_[4 + 1]),
                                                       &(scale_list->d_scale_list_[12 + 1]),
                                                       &(scale_list->d_scale_list_[8 + 3]),
                                                       &(scale_list->d_scale_list_[16 + 3]),
                                                       cublas_wrapper->getUseOrderCol322R4R4(),
                                                       stream_);

                invokeAddVBiasTransformRebuildPadding(v_buf_,
                                                      (const int8_t*)V_int_buf_,
                                                      attention_weights->value_weight.bias,
                                                      sequence_id_map_,
                                                      m,
                                                      request_batch_size,
                                                      request_seq_len,
                                                      head_num_,
                                                      size_per_head_,
                                                      &(scale_list->d_scale_list_[20 + 1]),
                                                      &(scale_list->d_scale_list_[24 + 3]),
                                                      cublas_wrapper->getUseOrderCol322R4R4(),
                                                      stream_);
#ifdef SPARSITY_ENABLED
            }
#endif
        }
        sync_check_cuda_error();
    }

    int batchCount = request_batch_size * head_num_;
    float scalar = 1.0f / (sqrtf(size_per_head_ * 1.0f) * q_scaling_);
    if (int8_mode_ == 1) {
        cublas_wrapper->Gemm(qk_int_buf_,
                             batchCount,
                             request_seq_len,
                             seq_len_padded,
                             size_per_head_,
                             size_per_head_ * request_seq_len,
                             size_per_head_ * seq_len_padded,
                             request_seq_len * seq_len_padded,
                             q_buf_,
                             k_buf_);

        invokeSoftmaxCOL32(qk_buf_,
                           qk_int_buf_,
                           attention_mask,
                           request_batch_size,
                           head_num_,
                           request_seq_len,
                           scalar,
                           &(scale_list->d_scale_list_[8 + 1]),
                           &(scale_list->d_scale_list_[16 + 1]),
                           &(scale_list->d_scale_list_[32]),
                           stream_);

        cublas_wrapper->Gemm(transpose_dst_int_buf_,
                             batchCount,
                             request_seq_len,
                             size_per_head_,
                             seq_len_padded,
                             request_seq_len * seq_len_padded,
                             size_per_head_ * seq_len_padded,
                             size_per_head_ * request_seq_len,
                             qk_buf_,
                             v_buf_);

        if (padding_offset == nullptr) {
            invokeTransposeCOL32(dst_,
                                 transpose_dst_int_buf_,
                                 request_batch_size,
                                 request_seq_len,
                                 head_num_,
                                 size_per_head_,
                                 &(scale_list->d_scale_list_[24 + 1]),
                                 &(scale_list->d_scale_list_[32 + 1]),
                                 &(scale_list->d_scale_list_[36 + 3]),
                                 stream_);
        }
        else {
            invokeTransposeCOL32RebuildPadding(dst_,
                                               transpose_dst_int_buf_,
                                               sequence_id_map_,
                                               m,
                                               request_batch_size,
                                               request_seq_len,
                                               head_num_,
                                               size_per_head_,
                                               &(scale_list->d_scale_list_[24 + 1]),
                                               &(scale_list->d_scale_list_[32 + 1]),
                                               &(scale_list->d_scale_list_[36 + 3]),
                                               stream_);
        }
    }
    else if (int8_mode_ == 2 || int8_mode_ == 3) {

        cublas_wrapper->Gemm((int8_t*)qk_int_buf_,
                             batchCount,
                             request_seq_len,
                             seq_len_padded,
                             size_per_head_,
                             size_per_head_ * request_seq_len,
                             size_per_head_ * seq_len_padded,
                             request_seq_len * seq_len_padded,
                             scale_list->h_scale_list_[scale_list->p3_offset_ + 3],
                             q_buf_,
                             k_buf_);

        invokeSoftmaxCOL32(qk_buf_,
                           (int8_t*)qk_int_buf_,
                           attention_mask,
                           request_batch_size,
                           head_num_,
                           request_seq_len,
                           scalar,
                           &(scale_list->d_scale_list_[28 + 1]),
                           &(scale_list->d_scale_list_[32]),
                           stream_);

        cublas_wrapper->Gemm((int8_t*)transpose_dst_int_buf_,
                             batchCount,
                             request_seq_len,
                             size_per_head_,
                             seq_len_padded,
                             request_seq_len * seq_len_padded,
                             size_per_head_ * seq_len_padded,
                             size_per_head_ * request_seq_len,
                             scale_list->h_scale_list_[scale_list->p3_offset_ + 4],
                             qk_buf_,
                             v_buf_);
#ifdef SPARSITY_ENABLED
        if (sparse_) {
            if (padding_offset == nullptr) {
                invokeTransposeCOL32ToRow(dst_,
                                          (const int8_t*)transpose_dst_int_buf_,
                                          request_batch_size,
                                          request_seq_len,
                                          head_num_,
                                          size_per_head_,
                                          &(scale_list->d_scale_list_[36 + 1]),
                                          &(scale_list->d_scale_list_[36 + 3]),
                                          stream_);
            }
            else {
                invokeTransposeCOL32ToRowRebuildPadding(dst_,
                                                        (const int8_t*)transpose_dst_int_buf_,
                                                        sequence_id_map_,
                                                        m,
                                                        request_batch_size,
                                                        request_seq_len,
                                                        head_num_,
                                                        size_per_head_,
                                                        &(scale_list->d_scale_list_[36 + 1]),
                                                        &(scale_list->d_scale_list_[36 + 3]),
                                                        stream_);
            }
        }
        else {
#endif
            if (padding_offset == nullptr) {
                invokeTransposeCOL32(dst_,
                                     (const int8_t*)transpose_dst_int_buf_,
                                     request_batch_size,
                                     request_seq_len,
                                     head_num_,
                                     size_per_head_,
                                     &(scale_list->d_scale_list_[36 + 1]),
                                     &(scale_list->d_scale_list_[36 + 3]),
                                     stream_);
            }
            else {
                invokeTransposeCOL32RebuildPadding(dst_,
                                                   (const int8_t*)transpose_dst_int_buf_,
                                                   sequence_id_map_,
                                                   m,
                                                   request_batch_size,
                                                   request_seq_len,
                                                   head_num_,
                                                   size_per_head_,
                                                   &(scale_list->d_scale_list_[36 + 1]),
                                                   &(scale_list->d_scale_list_[36 + 3]),
                                                   stream_);
            }
#ifdef SPARSITY_ENABLED
        }
#endif
    }

    if (int8_mode_ == 1) {
        cublas_wrapper->Gemm(
            attention_out, 1, m, n, k, 0, 0, 0, dst_, (int8_t*)(attention_weights->attention_output_weight.kernel));
    }
    else if (int8_mode_ == 2 || int8_mode_ == 3) {
#ifdef SPARSITY_ENABLED
        if (sparse_) {
            cublas_wrapper->SpGemm(n,
                                   m_padded,
                                   k,
                                   scale_list->h_scale_list_[scale_list->p3_offset_ + 5],
                                   (int8_t*)(attention_weights->attention_output_weight.sp_kernel),
                                   dst_,
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
                                 dst_,
                                 (int8_t*)(attention_weights->attention_output_weight.kernel));
#ifdef SPARSITY_ENABLED
        }
#endif
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template<typename T>
UnfusedAttentionLayerINT8<T>::UnfusedAttentionLayerINT8(size_t max_batch_size,
                                                        size_t max_seq_len,
                                                        size_t head_num,
                                                        size_t size_per_head,
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
    hidden_units_(head_num * size_per_head),
    q_scaling_(q_scaling),
    int8_mode_(int8_mode),
    sparse_(sparse)
{
}

template<typename T>
UnfusedAttentionLayerINT8<T>::UnfusedAttentionLayerINT8(UnfusedAttentionLayerINT8<T> const& attention_layer):
    BaseAttentionLayer<T>(attention_layer.stream_,
                          attention_layer.cublas_wrapper_,
                          attention_layer.allocator_,
                          attention_layer.is_free_buffer_after_forward_),
    max_batch_size_(attention_layer.max_batch_size_),
    max_seq_len_(attention_layer.max_seq_len_),
    head_num_(attention_layer.head_num_),
    size_per_head_(attention_layer.size_per_head_),
    hidden_units_(attention_layer.hidden_units_),
    q_scaling_(attention_layer.q_scaling_),
    int8_mode_(attention_layer.int8_mode_),
    sparse_(attention_layer.sparse_)
{
}

template<typename T>
UnfusedAttentionLayerINT8<T>::~UnfusedAttentionLayerINT8()
{
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void UnfusedAttentionLayerINT8<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        int padded_max_seq_len = (max_seq_len_ + 31) / 32 * 32;
        q_buf_ = (int8_t*)allocator_->malloc(sizeof(int8_t) * max_batch_size_ * padded_max_seq_len * hidden_units_ * 3,
                                             false);
        k_buf_ = q_buf_ + max_batch_size_ * padded_max_seq_len * hidden_units_;
        v_buf_ = k_buf_ + max_batch_size_ * padded_max_seq_len * hidden_units_;
        qk_buf_ = (int8_t*)allocator_->malloc(
            sizeof(int8_t) * max_batch_size_ * head_num_ * padded_max_seq_len * padded_max_seq_len, false);
        dst_ = (int8_t*)allocator_->malloc(sizeof(int8_t) * max_batch_size_ * max_seq_len_ * hidden_units_, false);

        Q_int_buf_ =
            (int32_t*)allocator_->malloc(sizeof(int32_t) * max_batch_size_ * max_seq_len_ * hidden_units_ * 3, false);
        V_int_buf_ = Q_int_buf_ + max_batch_size_ * max_seq_len_ * hidden_units_;
        K_int_buf_ = V_int_buf_ + max_batch_size_ * max_seq_len_ * hidden_units_;
        qk_int_buf_ = (int32_t*)allocator_->malloc(
            sizeof(int32_t) * max_batch_size_ * head_num_ * padded_max_seq_len * padded_max_seq_len, false);
        transpose_dst_int_buf_ =
            (int32_t*)allocator_->malloc(sizeof(int32_t) * max_batch_size_ * max_seq_len_ * hidden_units_, false);

        sequence_id_map_ = (int32_t*)allocator_->malloc(sizeof(int32_t) * max_batch_size_ * max_seq_len_, false);

        is_allocate_buffer_ = true;
    }
}

template<typename T>
void UnfusedAttentionLayerINT8<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free(q_buf_);
        allocator_->free(Q_int_buf_);
        allocator_->free(qk_buf_);
        allocator_->free(qk_int_buf_);
        allocator_->free(dst_);
        allocator_->free(transpose_dst_int_buf_);
        allocator_->free(sequence_id_map_);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool UnfusedAttentionLayerINT8<T>::isValidBatchSize(size_t batch_size)
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
bool UnfusedAttentionLayerINT8<T>::isValidSeqLen(size_t seq_len)
{
    if (max_seq_len_ == 0) {
        max_seq_len_ = seq_len;
        return true;
    }
    else {
        return seq_len <= max_seq_len_;
    }
}

template class UnfusedAttentionLayerINT8<float>;
template class UnfusedAttentionLayerINT8<half>;

}  // namespace fastertransformer
