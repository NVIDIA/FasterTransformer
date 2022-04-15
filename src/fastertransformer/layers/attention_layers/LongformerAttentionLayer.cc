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

#include "LongformerAttentionLayer.h"
#include "src/fastertransformer/kernels/longformer_kernels.h"

#include <cuda_runtime.h>
#include <functional>
#include <numeric>
#include <stdio.h>
#include <string>
#include <vector>

namespace fastertransformer {

template<typename T>
LongformerAttentionLayer<T>::LongformerAttentionLayer(size_t head_num,
                                                      size_t size_per_head,
                                                      size_t local_attn_window_size,
                                                      size_t max_global_token_num,
                                                      size_t max_batch_size,
                                                      size_t max_seq_len,
                                                      float attn_scaler,
                                                      cudaStream_t stream,
                                                      cublasMMWrapper* cublas_wrapper,
                                                      IAllocator* allocator,
                                                      bool is_free_buffer_after_forward):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    head_num_(head_num),
    size_per_head_(size_per_head),
    local_attn_window_size_(local_attn_window_size),
    max_global_token_num_(max_global_token_num),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    attn_scaler_(attn_scaler)
{
    FT_CHECK(max_global_token_num_ <= local_attn_window_size_);
    check_cuda_error(cudaStreamCreate(&memcpy_stream_));
}

template<typename T>
LongformerAttentionLayer<T>::~LongformerAttentionLayer()
{
    check_cuda_error(cudaStreamDestroy(memcpy_stream_));
    freeBuffer();
}

template<typename T>
void LongformerAttentionLayer<T>::forward(std::vector<Tensor>* output_tensors, const std::vector<Tensor>* input_tensors)
{
    allocateBuffer();

    const int batch_size = input_tensors->at(0).shape[0];
    const int seq_len = input_tensors->at(0).shape[2];
    FT_CHECK(seq_len % local_attn_window_size_ == 0);
    FT_CHECK(size_per_head_ == 64);

    const int batch_stride = head_num_ * seq_len * size_per_head_;
    const int global_batch_stride = head_num_ * max_global_token_num_ * size_per_head_;
    const int attn_head_stride = seq_len * size_per_head_;
    const int attn_window_stride = local_attn_window_size_ * size_per_head_;
    const int local_attn_head_tail_gemm_strides_count = batch_size * head_num_;
    const int local_attn_middle_gemm_strides_count = (seq_len / local_attn_window_size_) - 2;

    const void* q = input_tensors->at(0).data;
    const void* k = input_tensors->at(1).data;
    const void* v = input_tensors->at(2).data;
    const void* qg = input_tensors->at(3).data;
    const void* kg = input_tensors->at(4).data;
    const void* vg = input_tensors->at(5).data;
    const void* local_attn_mask = (const T*)input_tensors->at(6).data;
    const void* global_attn_mask = (const T*)input_tensors->at(7).data;
    const int* global_idx = (const int*)input_tensors->at(8).data;
    const int* global_token_nums = (const int*)input_tensors->at(9).data;

    void* output = (void*)output_tensors->at(0).data;

    int global_token_nums_cpu[batch_size];

    cudaEvent_t memcpy_finished;
    check_cuda_error(cudaEventCreateWithFlags(&memcpy_finished, cudaEventDisableTiming));
    check_cuda_error(cudaMemcpyAsync(
        global_token_nums_cpu, global_token_nums, batch_size * sizeof(int), cudaMemcpyDeviceToHost, memcpy_stream_));

    /*  internal_var sections:
      0 - 4: store buf_ptrs which point to buffer 0, 1, 2, 3, 4, respectively. size: 5 * size_t
      5 - 9: store buf_sizes for each buffer. size: 5 * size_t
      10 - 14: store buf_strides per head for each buffer. size: 5 * size_t
    */
    size_t* internal_var[15];
    void** buf_ptrs = (void**)&internal_var[0];
    size_t* buf_sizes = (size_t*)(&internal_var[5]);
    size_t* buf_strides = (size_t*)(&internal_var[10]);

    buf_sizes[0] = (size_t)local_attn_window_size_ * 2 * local_attn_window_size_;
    buf_sizes[1] = (size_t)local_attn_window_size_ * 3 * local_attn_window_size_ * local_attn_middle_gemm_strides_count;
    buf_sizes[2] = (size_t)local_attn_window_size_ * 2 * local_attn_window_size_;
    buf_sizes[3] = (size_t)local_attn_window_size_ * (seq_len - local_attn_window_size_);
    buf_sizes[4] = (size_t)local_attn_window_size_ * seq_len;

    buf_strides[0] = (size_t)local_attn_window_size_ * 2;
    buf_strides[1] = (size_t)local_attn_window_size_ * 3;
    buf_strides[2] = (size_t)local_attn_window_size_ * 2;
    buf_strides[3] = (size_t)local_attn_window_size_;
    buf_strides[4] = (size_t)seq_len;

    /* buffer partition:
      0: local attn output per head - head, size:
        local_attn_window_size_ * 2 * local_attn_window_size_ * local_attn_head_tail_gemm_strides_count
      1: local attn output per head - middle, size:
        local_attn_window_size_ * 3 * local_attn_window_size_ * local_attn_middle_gemm_strides_count * batch_size *
      head_num_; 2: local attn output per head - tail, size: local_attn_window_size_ * 2 * local_attn_window_size_ *
      local_attn_head_tail_gemm_strides_count 3: global part of local k attending q per head, size:
        local_attn_window_size_ * (seq_len - local_attn_window_size_)
      4: global q attending global k per head, size:
        local_attn_window_size_ * seq_len
    */

    buf_ptrs[0] = attn_buffer_;
    for (int i = 1; i < 5; ++i) {
        buf_ptrs[i] = (void*)((char*)buf_ptrs[i - 1] + buf_sizes[i - 1] * head_num_ * batch_size * sizeof(T));
    }
    check_cuda_error(cudaMemcpyAsync(
        internal_vars_device_, (void*)internal_var, 15 * sizeof(size_t), cudaMemcpyHostToDevice, memcpy_stream_));
    check_cuda_error(cudaEventRecord(memcpy_finished, memcpy_stream_));

    // Local attention
    // local attn per head - head
    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        2 * local_attn_window_size_,
                                        local_attn_window_size_,
                                        size_per_head_,
                                        k,
                                        size_per_head_,
                                        attn_head_stride,
                                        q,
                                        size_per_head_,
                                        attn_head_stride,
                                        buf_ptrs[0],
                                        2 * local_attn_window_size_,
                                        buf_sizes[0],
                                        local_attn_head_tail_gemm_strides_count);

    // local attn per head - middle
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < (int)head_num_; ++j) {
            void* q_middle =
                (char*)q
                + (i * batch_stride + j * size_per_head_ * seq_len + local_attn_window_size_ * size_per_head_)
                      * sizeof(T);
            void* k_middle = (char*)k + (i * batch_stride + j * size_per_head_ * seq_len) * sizeof(T);
            void* qk_middle = (char*)buf_ptrs[1] + (i * head_num_ + j) * buf_sizes[1] * sizeof(T);

            cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                                CUBLAS_OP_N,
                                                3 * local_attn_window_size_,
                                                local_attn_window_size_,
                                                size_per_head_,
                                                k_middle,
                                                size_per_head_,
                                                attn_window_stride,
                                                q_middle,
                                                size_per_head_,
                                                attn_window_stride,
                                                qk_middle,
                                                3 * local_attn_window_size_,
                                                3 * local_attn_window_size_ * local_attn_window_size_,
                                                local_attn_middle_gemm_strides_count);
        }
    }

    // local attn per head - tail
    int tail_blk_q = (seq_len / local_attn_window_size_) - 1;
    int tail_blk_k = (seq_len / local_attn_window_size_) - 2;
    void* q_tail = (char*)q + tail_blk_q * local_attn_window_size_ * size_per_head_ * sizeof(T);
    void* k_tail = (char*)k + tail_blk_k * local_attn_window_size_ * size_per_head_ * sizeof(T);

    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        2 * local_attn_window_size_,
                                        local_attn_window_size_,
                                        size_per_head_,
                                        k_tail,
                                        size_per_head_,
                                        attn_head_stride,
                                        q_tail,
                                        size_per_head_,
                                        attn_head_stride,
                                        buf_ptrs[2],
                                        2 * local_attn_window_size_,
                                        buf_sizes[2],
                                        local_attn_head_tail_gemm_strides_count);

    check_cuda_error(cudaEventSynchronize(memcpy_finished));
    // global attention
    for (int i = 0; i < batch_size; ++i) {
        if (global_token_nums_cpu[i] > (int)local_attn_window_size_) {
            std::cerr << "ERROR! the number of global tokens are larger than the attention window";
            std::cerr << ", the code will not work correctly!!";
        }
        if (global_token_nums_cpu[i] > 0) {
            // local tokens attending global tokens
            void* q_local = (char*)q + (i * batch_stride + local_attn_window_size_ * size_per_head_) * sizeof(T);
            void* k_local = (char*)k + i * batch_stride * sizeof(T);
            void* qk_local = (char*)buf_ptrs[3] + i * buf_sizes[3] * head_num_ * sizeof(T);

            cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                                CUBLAS_OP_N,
                                                global_token_nums_cpu[i],
                                                seq_len - local_attn_window_size_,
                                                size_per_head_,
                                                k_local,
                                                size_per_head_,
                                                attn_head_stride,
                                                q_local,
                                                size_per_head_,
                                                attn_head_stride,
                                                qk_local,
                                                local_attn_window_size_,
                                                buf_sizes[3],
                                                head_num_);

            // global token attending everything
            void* q_global = (char*)qg + (i * global_batch_stride) * sizeof(T);
            void* k_global = (char*)kg + (i * batch_stride) * sizeof(T);
            void* qk_global = (char*)buf_ptrs[4] + (i * buf_sizes[4] * head_num_) * sizeof(T);
            cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                                CUBLAS_OP_N,
                                                seq_len,
                                                global_token_nums_cpu[i],
                                                size_per_head_,
                                                k_global,
                                                size_per_head_,
                                                attn_head_stride,
                                                q_global,
                                                size_per_head_,
                                                max_global_token_num_ * size_per_head_,
                                                qk_global,
                                                seq_len,
                                                buf_sizes[4],
                                                head_num_);
        }
    }

    invokeLongformerMHASoftmax((const T*)global_attn_mask,
                               global_idx,
                               global_token_nums,
                               internal_vars_device_,
                               (const T*)local_attn_mask,
                               attn_scaler_,
                               seq_len,
                               head_num_,
                               batch_size,
                               local_attn_window_size_,
                               stream_);
    sync_check_cuda_error();

    // local values attending the softmax score.
    // local attn per head - head
    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        size_per_head_,
                                        local_attn_window_size_,
                                        2 * local_attn_window_size_,
                                        v,
                                        size_per_head_,
                                        seq_len * size_per_head_,
                                        buf_ptrs[0],
                                        buf_strides[0],
                                        buf_sizes[0],
                                        output,
                                        size_per_head_,
                                        seq_len * size_per_head_,
                                        batch_size * head_num_);

    // local attn per head - middle
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < (int)head_num_; ++j) {
            void* v_local = (char*)v + (i * batch_stride + j * size_per_head_ * seq_len) * sizeof(T);
            void* prob = (char*)buf_ptrs[1] + (i * head_num_ + j) * buf_sizes[1] * sizeof(T);
            void* out = (char*)output
                        + (i * batch_stride + j * size_per_head_ * seq_len + local_attn_window_size_ * size_per_head_)
                              * sizeof(T);
            cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
                                                CUBLAS_OP_N,
                                                size_per_head_,
                                                local_attn_window_size_,
                                                3 * local_attn_window_size_,
                                                v_local,
                                                size_per_head_,
                                                local_attn_window_size_ * size_per_head_,
                                                prob,
                                                buf_strides[1],
                                                3 * local_attn_window_size_ * local_attn_window_size_,
                                                out,
                                                size_per_head_,
                                                local_attn_window_size_ * size_per_head_,
                                                local_attn_middle_gemm_strides_count);
        }
    }

    // local attn per head - tail
    int tail_blk_v = (seq_len / local_attn_window_size_) - 2;
    int tail_blk_out = (seq_len / local_attn_window_size_) - 1;
    void* tail_v = (char*)v + tail_blk_v * local_attn_window_size_ * size_per_head_ * sizeof(T);
    void* tail_out = (char*)output + tail_blk_out * local_attn_window_size_ * size_per_head_ * sizeof(T);
    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        size_per_head_,
                                        local_attn_window_size_,
                                        2 * local_attn_window_size_,
                                        tail_v,
                                        size_per_head_,
                                        seq_len * size_per_head_,
                                        buf_ptrs[2],
                                        buf_strides[2],
                                        buf_sizes[2],
                                        tail_out,
                                        size_per_head_,
                                        seq_len * size_per_head_,
                                        batch_size * head_num_);

    // global attention part
    for (int i = 0; i < batch_size; ++i) {
        if (global_token_nums_cpu[i] > 0) {
            int glob_longdim_mm = seq_len - 2 * local_attn_window_size_;

            void* v_local = (char*)v + (i * batch_stride) * sizeof(T);
            void* prob = (char*)buf_ptrs[3]
                         + (i * buf_sizes[3] * head_num_ + local_attn_window_size_ * buf_strides[3]) * sizeof(T);
            void* out = (char*)output + (i * batch_stride + 2 * local_attn_window_size_ * size_per_head_) * sizeof(T);

            cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
                                                CUBLAS_OP_N,
                                                size_per_head_,
                                                glob_longdim_mm,
                                                global_token_nums_cpu[i],
                                                v_local,
                                                size_per_head_,
                                                seq_len * size_per_head_,
                                                prob,
                                                buf_strides[3],
                                                buf_sizes[3],
                                                out,
                                                size_per_head_,
                                                seq_len * size_per_head_,
                                                head_num_,
                                                1.0f,
                                                1.0f);

            // global tokens
            void* v_global = (char*)vg + (i * batch_stride) * sizeof(T);
            prob = (char*)buf_ptrs[4] + (i * buf_sizes[4] * head_num_) * sizeof(T);
            out = (char*)output + (i * batch_stride) * sizeof(T);

            cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
                                                CUBLAS_OP_N,
                                                size_per_head_,
                                                global_token_nums_cpu[i],
                                                seq_len,
                                                v_global,
                                                size_per_head_,
                                                seq_len * size_per_head_,
                                                prob,
                                                buf_strides[4],
                                                buf_sizes[4],
                                                out,
                                                size_per_head_,
                                                seq_len * size_per_head_,
                                                head_num_);
        }
    }

    check_cuda_error(cudaEventDestroy(memcpy_finished));
    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
void LongformerAttentionLayer<T>::allocateBuffer()
{
    if (!is_allocate_buffer_) {

        internal_vars_device_ = (void*)allocator_->malloc(sizeof(size_t) * 15);

        attn_buffer_ =
            (T*)allocator_->malloc(sizeof(T) * head_num_ * max_batch_size_
                                       * (2 * local_attn_window_size_ * local_attn_window_size_
                                          + 3 * local_attn_window_size_ * local_attn_window_size_
                                                * (max_seq_len_ / local_attn_window_size_ - 2)
                                          + 2 * local_attn_window_size_ * local_attn_window_size_
                                          + local_attn_window_size_ * (max_seq_len_ - local_attn_window_size_)
                                          + local_attn_window_size_ * max_seq_len_),
                                   false);

        is_allocate_buffer_ = true;
    }
}

template<typename T>
void LongformerAttentionLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        allocator_->free(internal_vars_device_);
        allocator_->free(attn_buffer_);

        is_allocate_buffer_ = false;
    }
}

template class LongformerAttentionLayer<float>;
template class LongformerAttentionLayer<half>;

}  // namespace fastertransformer