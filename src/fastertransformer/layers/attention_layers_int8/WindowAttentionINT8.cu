/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/layers/attention_layers_int8/WindowAttentionINT8.h"

namespace fastertransformer {

// Add bias, and then transform from
// 3 * ([valid_word_num, head*size] + CUBLASLT_ORDER_COL32) -> [valid_word_num, head, 3, size] + row-major
// grid((head*size + 31)/32, (valid_word_num + 31)/32, 3)
// block(8, 32)
// size should be a multiple of 4
template<typename T>
__global__ void swin_trt_add_QKV_bias_COL32_int8IO(char4* output,
                                                   const char4* QKV,
                                                   const T* bias_Q,
                                                   const T* bias_K,
                                                   const T* bias_V,
                                                   const float* q_bias_QFactor_ptr,
                                                   const float* k_bias_QFactor_ptr,
                                                   const float* v_bias_QFactor_ptr,
                                                   const float qkv_deQFactor,
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
    const int col_id = qkv_id * head_num_x_size_per_head + hidden_id;

    const bool qual = (seq_id < valid_word_num) && (hidden_id < head_num_x_size_per_head);
    if (qual) {
        const float* bias_QFactor_ptr =
            (qkv_id == 0) ? q_bias_QFactor_ptr : ((qkv_id == 1) ? k_bias_QFactor_ptr : v_bias_QFactor_ptr);
        const float qkv_output_scale = __ldg(bias_QFactor_ptr);

        const T* bias_ptr = (qkv_id == 0) ? bias_Q : ((qkv_id == 1) ? bias_K : bias_V);

        const int input_id = ((col_id >> 5) << 5) * valid_word_num + (seq_id << 5) + (col_id & 31);

        char4 tmp = __ldg(QKV + (input_id >> 2));

        tmp.x = float_to_int8_rn(
            (static_cast<float>(tmp.x) * qkv_deQFactor + static_cast<float>(__ldg(bias_ptr + hidden_id)))
            * qkv_output_scale);

        tmp.y = float_to_int8_rn(
            (static_cast<float>(tmp.y) * qkv_deQFactor + static_cast<float>(__ldg(bias_ptr + hidden_id + 1)))
            * qkv_output_scale);

        tmp.z = float_to_int8_rn(
            (static_cast<float>(tmp.z) * qkv_deQFactor + static_cast<float>(__ldg(bias_ptr + hidden_id + 2)))
            * qkv_output_scale);

        tmp.w = float_to_int8_rn(
            (static_cast<float>(tmp.w) * qkv_deQFactor + static_cast<float>(__ldg(bias_ptr + hidden_id + 3)))
            * qkv_output_scale);

        const int output_id =
            ((seq_id * head_num_x_size_per_head + head_id * size_per_head) * 3 + qkv_id * size_per_head + size_id) >> 2;

        output[output_id] = tmp;
    }
}

template<typename T>
void invokeSwinTrtAddQkvBiasInt8IO(int8_t* output,
                                   const int8_t* Q,
                                   const T* bias_Q,
                                   const T* bias_K,
                                   const T* bias_V,
                                   const size_t token_num,
                                   const size_t head_num,
                                   const size_t size_per_head,
                                   const float* q_bias_QFactor_ptr,
                                   const float* k_bias_QFactor_ptr,
                                   const float* v_bias_QFactor_ptr,
                                   const float qkv_deQFactor,
                                   const cudaStream_t stream)
{

    int head_num_x_size_per_head = head_num * size_per_head;
    dim3 grid((head_num_x_size_per_head + 31) / 32, (token_num + 31) / 32, 3);
    dim3 block(8, 32);

    assert(size_per_head % 4 == 0);

    swin_trt_add_QKV_bias_COL32_int8IO<<<grid, block, 0, stream>>>((char4*)output,
                                                                   (const char4*)Q,
                                                                   bias_Q,
                                                                   bias_K,
                                                                   bias_V,
                                                                   q_bias_QFactor_ptr,
                                                                   k_bias_QFactor_ptr,
                                                                   v_bias_QFactor_ptr,
                                                                   qkv_deQFactor,
                                                                   token_num,
                                                                   head_num,
                                                                   size_per_head,
                                                                   head_num_x_size_per_head);
}

template<typename T>
int WindowAttentionINT8<T>::trt_getS(const int actual_seqlen)
{
    int S = 384;
    if (actual_seqlen <= 64) {
        S = 64;
    }
    else if (actual_seqlen <= 128) {
        S = 128;
    }
    else if (actual_seqlen <= 256) {
        S = 256;
    }
    else if (actual_seqlen <= 384) {
        S = 384;
    }
    return S;
}

template<typename T>
size_t WindowAttentionINT8<T>::roundByteSize(const size_t size, const int factor)
{
    return (size + factor - 1) / factor * factor;
}

template<typename T>
void WindowAttentionINT8<T>::allocateBuffer()
{
    int num_head = num_head_;
    if (!is_free_buffer_after_forward_) {
        num_head = num_head * 8;
    }
    if (is_allocate_buffer_ == false) {
        if (use_trt_) {
            int S = trt_getS(window_len_);
            trt_attention_mask_ = (half*)allocator_->malloc(roundByteSize(window_num_ * S * S * sizeof(T), 4), false);
            trt_relative_position_bias_ =
                (half*)allocator_->malloc(roundByteSize(num_head * S * S * sizeof(T), 4), false);
            Q_buf_ = (int8_t*)allocator_->malloc(
                3 * max_batch_ * patches_resolution_ * patches_resolution_ * embed_dim_ * sizeof(int8_t), false);
            K_buf_ = Q_buf_ + max_batch_ * patches_resolution_ * patches_resolution_ * embed_dim_;
            V_buf_ = K_buf_ + max_batch_ * patches_resolution_ * patches_resolution_ * embed_dim_;
            q_buf_ = (int8_t*)allocator_->malloc(
                3 * max_batch_ * patches_resolution_ * patches_resolution_ * embed_dim_ * sizeof(int8_t), false);
            k_buf_ = q_buf_ + max_batch_ * patches_resolution_ * patches_resolution_ * embed_dim_;
            v_buf_ = k_buf_ + max_batch_ * patches_resolution_ * patches_resolution_ * embed_dim_;
            dst_buf_ = (int8_t*)allocator_->malloc(
                max_batch_ * patches_resolution_ * patches_resolution_ * embed_dim_ * sizeof(int8_t), false);
        }
        else {
            int padded_winlen = (window_len_ + 31) / 32 * 32;
            Q_buf_ = (int8_t*)allocator_->malloc(
                3 * max_batch_ * patches_resolution_ * patches_resolution_ * embed_dim_ * sizeof(int8_t), false);
            K_buf_ = Q_buf_ + max_batch_ * patches_resolution_ * patches_resolution_ * embed_dim_;
            V_buf_ = K_buf_ + max_batch_ * patches_resolution_ * patches_resolution_ * embed_dim_;
            q_buf_ = (int8_t*)allocator_->malloc(max_batch_ * window_num_ * window_len_ * embed_dim_ * sizeof(int8_t),
                                                 false);
            k_buf_ = (int8_t*)allocator_->malloc(max_batch_ * window_num_ * padded_winlen * embed_dim_ * sizeof(int8_t),
                                                 false);
            v_buf_ = (int8_t*)allocator_->malloc(max_batch_ * window_num_ * padded_winlen * embed_dim_ * sizeof(int8_t),
                                                 false);
            qk_buf_ =
                (int8_t*)allocator_->malloc(max_batch_ * window_num_ * num_head * window_len_ * padded_winlen, false);
            dst_buf_ = (int8_t*)allocator_->malloc(max_batch_ * window_num_ * window_len_ * embed_dim_ * sizeof(int8_t),
                                                   false);
        }
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void WindowAttentionINT8<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        if (use_trt_) {
            allocator_->free(trt_attention_mask_);
            allocator_->free(trt_relative_position_bias_);
            allocator_->free(Q_buf_);
            allocator_->free(q_buf_);
            allocator_->free(dst_buf_);
        }
        else {
            allocator_->free(Q_buf_);
            allocator_->free(q_buf_);
            allocator_->free(k_buf_);
            allocator_->free(v_buf_);
            allocator_->free(dst_buf_);
        }
        is_allocate_buffer_ = false;
    }
}

template<typename T>
WindowAttentionINT8<T>::WindowAttentionINT8(int max_batch,
                                            int window_size,
                                            int patches_resolution,
                                            int embed_dim,
                                            cudaStream_t stream,
                                            cublasMMWrapper* cublas_wrapper,
                                            IAllocator* allocator,
                                            bool is_free_buffer_after_forward,
                                            bool qkv_bias,
                                            float qk_scale):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    patches_resolution_(patches_resolution),
    embed_dim_(embed_dim),
    max_batch_(max_batch),
    window_size_(window_size),
    qkv_bias_(qkv_bias),
    qk_scale_(qk_scale)
{
    window_len_ = window_size_ * window_size_;
}

template<typename T>
WindowAttentionINT8<T>::~WindowAttentionINT8()
{
}

template<typename T>
void WindowAttentionINT8<T>::forward(std::vector<fastertransformer::Tensor>* output_tensors,
                                     const std::vector<fastertransformer::Tensor>* input_tensors,
                                     const AttentionWeight<T>* attention_weights)
{
    // input_tensors:
    //      input [batch * window_num * window_len, dim]
    //      attention_mask [window_num, window_len, window_len]
    //      attention_relative_position_bias [num_head, window_len, window_len]
    //      input_parameters [6] {batch, dim, input_resolution, num_head, shift_size, sm}
    // output_tensors:
    //      output [batch * window_num * window_len, dim]

    cublasINT8MMWrapper* cublas_wrapper = (cublasINT8MMWrapper*)cublas_wrapper_;
    int8_t* attention_out = (int8_t*)output_tensors->at(0).data;
    const int8_t* from_tensor = (const int8_t*)input_tensors->at(0).data;
    const T* attention_mask = (const T*)input_tensors->at(1).data;
    const T* attention_relative_pos_bias = (const T*)input_tensors->at(2).data;
    const int* input_parameters = (const int*)input_tensors->at(3).data;
    const int batch = input_parameters[0];
    const int dim = input_parameters[1];
    const int input_resolution = input_parameters[2];
    const int num_head = input_parameters[3];
    const int shift_size = input_parameters[4];
    const int sm = input_parameters[5];

    int use_ORDER_COL32_2R_4R4 = (sm >= 80 ? 1 : 0);

    int size_per_head = dim / num_head;
    int trt_S = 1024;
    if ((sm == 75 || sm == 80 || sm == 86) && size_per_head == 32 && window_len_ <= TRT_MAX_LEN
        && std::is_same<T, half>::value) {
        trt_S = trt_getS(window_len_);
        use_trt_ = true;
    }
    num_head_ = num_head;
    patches_resolution_ = input_resolution;
    window_num_ = (input_resolution / window_size_) * (input_resolution / window_size_);
    embed_dim_ = dim;
    allocateBuffer();

    float scale = 1.0f / sqrt(size_per_head);
    if (fabs(qk_scale_ - 1.0f) > 0.0001) {
        scale = qk_scale_;
    }

    if (use_trt_) {
        if (dispatcher_int8_.get() && num_head == dispatcher_int8_num_head_) {}
        else {
            dispatcher_int8_.reset(new FusedMHARunnerInt8v2(num_head, size_per_head, sm, scale));
            dispatcher_int8_num_head_ = num_head;
        }
    }

    const ScaleList* scale_list = ((const AttentionINT8Weight<T>*)attention_weights)->scale_list_ptr;

    cublas_wrapper->Gemm(Q_buf_,
                         1,
                         batch * window_num_ * window_len_,
                         3 * dim,
                         dim,
                         0,
                         0,
                         0,
                         scale_list->h_scale_list_[scale_list->p3_offset_ + 0],
                         from_tensor,
                         (int8_t*)attention_weights->query_weight.kernel);

    int S;
    if (use_trt_ && dispatcher_int8_.get()) {
        S = trt_S;
    }
    if (use_trt_ && dispatcher_int8_.get() && dispatcher_int8_->isValid(S)) {
        const T* bias_Q = attention_weights->query_weight.bias;
        const T* bias_K = attention_weights->query_weight.bias + dim;
        const T* bias_V = attention_weights->query_weight.bias + (dim << 1);
        invokeSwinTrtAddQkvBiasInt8IO(q_buf_,
                                      Q_buf_,
                                      bias_Q,
                                      bias_K,
                                      bias_V,
                                      batch * window_num_ * window_len_,
                                      num_head,
                                      size_per_head,
                                      &(scale_list->d_scale_list_[16 + 3]),
                                      &(scale_list->d_scale_list_[20 + 3]),
                                      &(scale_list->d_scale_list_[24 + 3]),
                                      scale_list->h_scale_list_[4 + 1],
                                      stream_);

        half* trt_attention_mask = nullptr;
        if (shift_size != 0) {
            invokeTransformMask(trt_attention_mask_, (const half*)attention_mask, window_num_, window_len_, stream_);
            trt_attention_mask = trt_attention_mask_;
        }

        invokeTransformMask(
            trt_relative_position_bias_, (const half*)attention_relative_pos_bias, num_head, window_len_, stream_);

        const int B = batch * window_num_;
        // dispatcher_int8_->setScaleList(scale_list->h_scale_list_[scale_list->p4_offset_]/127.0f,
        //                                scale_list->h_scale_list_[scale_list->p4_offset_ + 1]/127.0f,
        //                                scale_list->h_scale_list_[scale_list->p4_offset_ + 2]/127.0f);
        dispatcher_int8_->setScaleList(scale_list->h_scale_list_[scale_list->p4_offset_],
                                       scale_list->h_scale_list_[scale_list->p4_offset_ + 1],
                                       scale_list->h_scale_list_[scale_list->p4_offset_ + 2]);
        dispatcher_int8_->setup(S, B, window_num_);
        dispatcher_int8_->run(
            q_buf_, trt_attention_mask, trt_relative_position_bias_, window_len_, nullptr, dst_buf_, stream_);
        invokeRowMajorToCOL32(v_buf_, dst_buf_, batch * window_num_ * window_len_, dim, stream_);

        cublas_wrapper->Gemm(q_buf_,
                             1,
                             batch * window_num_ * window_len_,
                             dim,
                             dim,
                             0,
                             0,
                             0,
                             scale_list->h_scale_list_[scale_list->p3_offset_ + 1],
                             v_buf_,
                             (int8_t*)attention_weights->attention_output_weight.kernel);
    }
    else {
        const T* bias_Q = attention_weights->query_weight.bias;
        const T* bias_K = attention_weights->query_weight.bias + dim;
        const T* bias_V = attention_weights->query_weight.bias + 2 * dim;
        invokeAddQKBiasTransform(q_buf_,
                                 k_buf_,
                                 Q_buf_,
                                 bias_Q,
                                 K_buf_,
                                 bias_K,
                                 batch * window_num_,
                                 window_len_,
                                 num_head,
                                 size_per_head,
                                 &(scale_list->d_scale_list_[4 + 1]),
                                 &(scale_list->d_scale_list_[4 + 1]),
                                 &(scale_list->d_scale_list_[16 + 3]),
                                 &(scale_list->d_scale_list_[20 + 3]),
                                 use_ORDER_COL32_2R_4R4,
                                 stream_);

        invokeAddVBiasTransform(v_buf_,
                                V_buf_,
                                bias_V,
                                batch * window_num_,
                                window_len_,
                                num_head,
                                size_per_head,
                                &(scale_list->d_scale_list_[4 + 1]),
                                &(scale_list->d_scale_list_[24 + 3]),
                                use_ORDER_COL32_2R_4R4,
                                stream_);

        int padded_winlen = (window_len_ + 31) / 32 * 32;
        cublas_wrapper->Gemm(qk_buf_,
                             batch * window_num_ * num_head,
                             window_len_,
                             padded_winlen,
                             size_per_head,
                             window_len_ * size_per_head,
                             padded_winlen * size_per_head,
                             window_len_ * padded_winlen,
                             scale_list->h_scale_list_[scale_list->p3_offset_ + 4],
                             q_buf_,
                             k_buf_);

        if (shift_size != 0) {
            invokeSoftmaxWithRelPosBiasCOL32(qk_buf_,
                                             qk_buf_,
                                             attention_mask,
                                             attention_relative_pos_bias,
                                             batch,
                                             num_head,
                                             window_num_,
                                             window_len_,
                                             scale,
                                             &(scale_list->d_scale_list_[32 + 1]),
                                             &(scale_list->d_scale_list_[28 + 3]),
                                             stream_);
        }
        else {
            const T* attn_mask_tmp = nullptr;
            invokeSoftmaxWithRelPosBiasCOL32(qk_buf_,
                                             qk_buf_,
                                             attn_mask_tmp,
                                             attention_relative_pos_bias,
                                             batch,
                                             num_head,
                                             window_num_,
                                             window_len_,
                                             scale,
                                             &(scale_list->d_scale_list_[32 + 1]),
                                             &(scale_list->d_scale_list_[28 + 3]),
                                             stream_);
        }

        cublas_wrapper->Gemm(dst_buf_,
                             batch * window_num_ * num_head,
                             window_len_,
                             size_per_head,
                             padded_winlen,
                             window_len_ * padded_winlen,
                             size_per_head * padded_winlen,
                             window_len_ * size_per_head,
                             scale_list->h_scale_list_[scale_list->p3_offset_ + 5],
                             qk_buf_,
                             v_buf_);

        invokeTransposeCOL32(v_buf_,
                             dst_buf_,
                             batch * window_num_,
                             window_len_,
                             num_head,
                             size_per_head,
                             &(scale_list->d_scale_list_[36 + 1]),
                             &(scale_list->d_scale_list_[36 + 3]),
                             stream_);

        cublas_wrapper->Gemm(q_buf_,
                             1,
                             batch * window_num_ * window_len_,
                             dim,
                             dim,
                             0,
                             0,
                             0,
                             scale_list->h_scale_list_[scale_list->p3_offset_ + 1],
                             v_buf_,
                             (int8_t*)attention_weights->attention_output_weight.kernel);
    }

    invokeReverseRollCol32(attention_out,
                           q_buf_,
                           batch,
                           window_num_,
                           window_len_,
                           window_size_,
                           input_resolution,
                           input_resolution,
                           dim,
                           shift_size,
                           stream_);

    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
}

template class WindowAttentionINT8<float>;
template class WindowAttentionINT8<half>;
}  // namespace fastertransformer