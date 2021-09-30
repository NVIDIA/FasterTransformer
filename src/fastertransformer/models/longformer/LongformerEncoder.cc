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

#include "LongformerEncoder.h"

#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/kernels/add_bias_transpose_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/longformer_kernels.h"
#include "src/fastertransformer/layers/FfnLayer.h"
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/layers/attention_layers/LongformerAttentionLayer.h"

namespace fastertransformer {

template<typename T>
LongformerEncoder<T>::LongformerEncoder(size_t layers_num,
                                        size_t in_dim,
                                        size_t head_num,
                                        size_t size_per_head,
                                        size_t intermediate_size,
                                        size_t local_attn_window_size,
                                        size_t max_global_token_num,
                                        size_t max_batch_size,
                                        size_t max_seq_len,
                                        float attn_scaler,
                                        cudaStream_t stream,
                                        cublasMMWrapper* cublas_wrapper,
                                        IAllocator* allocator,
                                        bool is_free_buffer_after_forward):
    layers_num_(layers_num),
    in_dim_(in_dim),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    intermediate_size_(intermediate_size),
    max_batch_size_(max_batch_size),
    max_global_token_num_(max_global_token_num),
    max_seq_len_(max_seq_len),
    cublas_wrapper_(cublas_wrapper),
    allocator_(allocator),
    stream_(stream),
    is_free_buffer_after_forward_(is_free_buffer_after_forward)

{
    longformer_attn_layer_ = new LongformerAttentionLayer<T>(head_num,
                                                             size_per_head,
                                                             local_attn_window_size,
                                                             max_global_token_num,
                                                             max_batch_size,
                                                             max_seq_len,
                                                             attn_scaler,
                                                             stream,
                                                             cublas_wrapper,
                                                             allocator,
                                                             is_free_buffer_after_forward);
    inter_gelu_out_ffn_ = new GeluFfnLayer<T>(max_batch_size,
                                              max_seq_len,
                                              head_num,
                                              size_per_head,
                                              intermediate_size_,
                                              stream_,
                                              cublas_wrapper_,
                                              allocator_,
                                              is_free_buffer_after_forward_,
                                              false);
}

template<typename T>
LongformerEncoder<T>::~LongformerEncoder()
{
    delete longformer_attn_layer_;
    delete inter_gelu_out_ffn_;
    freeBuffer();
}

template<typename T>
void LongformerEncoder<T>::allocateBuffer()
{
    if (!is_allocate_buffer_) {
        cub_storage_ = (void*)allocator_->malloc(getInitLongformerCubStorage<T>(max_seq_len_), false);
        global_idx_ = (int*)allocator_->malloc(sizeof(int) * max_seq_len_ * max_batch_size_, false);
        global_token_nums_ = (int*)allocator_->malloc(sizeof(int) * max_batch_size_, false);
        seq_idx_ = (int*)allocator_->malloc(sizeof(int) * max_seq_len_, false);

        local_attn_mask_shifted_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_, false);
        size_t qkv_buffer_size = sizeof(T) * max_batch_size_ * hidden_units_ * 6 * max_seq_len_;

        size_t input_output_buffer_size = sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_;

        qkv_buffer_ = (T*)allocator_->malloc(qkv_buffer_size, false);
        mha_qkv_buffer_ = (T*)allocator_->malloc(qkv_buffer_size, false);
        mha_out_buffer_ = (T*)allocator_->malloc(input_output_buffer_size, false);
        attn_out_buffer_ = (T*)allocator_->malloc(input_output_buffer_size, false);
        attn_output_buffer_ = (T*)allocator_->malloc(input_output_buffer_size, false);
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void LongformerEncoder<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        allocator_->free(cub_storage_);
        allocator_->free(global_idx_);
        allocator_->free(global_token_nums_);
        allocator_->free(seq_idx_);
        allocator_->free(local_attn_mask_shifted_);
        allocator_->free(qkv_buffer_);
        allocator_->free(mha_qkv_buffer_);
        allocator_->free(mha_out_buffer_);
        allocator_->free(attn_out_buffer_);
        allocator_->free(attn_output_buffer_);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
std::vector<LongformerLayerWeight<T>>* LongformerEncoder<T>::getWeightsPtr()
{
    return &weights_;
}

template<typename T>
void LongformerEncoder<T>::forward(std::vector<Tensor>* output_tensors, std::vector<Tensor>* input_tensors)
{
    /*
    input_tensors: 0: input (batch_size x seq_len x in_dim)
                   1: local_attn_mask (batch_size x seq_len), 0.0 for non local attn, 1.0 for local attn.
                   2: global_attn_mask (batch_size x seq_len), -10000.0 for non global attn, 0.0 for global attn.

    output_tensors: 0: output (batch_size x seq_len x hidden_units_)
    */

    allocateBuffer();
    const size_t batch_size = input_tensors->at(0).shape[0];
    const size_t seq_len = input_tensors->at(0).shape[1];

    invokeInitLongformerIdx((T*)input_tensors->at(2).data,
                            seq_idx_,
                            global_idx_,
                            global_token_nums_,
                            seq_len,
                            batch_size,
                            cub_storage_,
                            stream_);
    invokeLocalAttnMaskShift((T*)input_tensors->at(1).data, local_attn_mask_shifted_, batch_size, seq_len, stream_);
    sync_check_cuda_error();

    for (size_t i = 0; i < layers_num_; i++) {
        forwardLayer(i == 0 ? (T*)input_tensors->at(0).data : (T*)output_tensors->at(0).data,
                     (T*)output_tensors->at(0).data,
                     local_attn_mask_shifted_,
                     (T*)input_tensors->at(2).data,
                     global_idx_,
                     global_token_nums_,
                     &(weights_[i]),
                     batch_size,
                     seq_len,
                     i == 0 ? in_dim_ : hidden_units_);
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
void LongformerEncoder<T>::forwardLayer(T* input,
                                        T* output,
                                        const T* local_attn_mask,
                                        const T* global_attn_mask,
                                        const int* global_idx,
                                        const int* global_token_nums,
                                        const LongformerLayerWeight<T>* weight,
                                        const size_t batch_size,
                                        const size_t seq_len,
                                        const size_t in_dim_)
{
    T* const q = qkv_buffer_;
    T* const k = qkv_buffer_ + batch_size * hidden_units_ * seq_len;
    T* const v = qkv_buffer_ + batch_size * hidden_units_ * 2 * seq_len;
    T* const kg = qkv_buffer_ + batch_size * hidden_units_ * 3 * seq_len;
    T* const vg = qkv_buffer_ + batch_size * hidden_units_ * 4 * seq_len;
    T* const qg = qkv_buffer_ + batch_size * hidden_units_ * 5 * seq_len;

    // q
    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          hidden_units_,
                          batch_size * seq_len,
                          in_dim_,
                          weight->query_weights.kernel,
                          hidden_units_,
                          input,
                          in_dim_,
                          q,
                          hidden_units_);

    // k
    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          hidden_units_,
                          batch_size * seq_len,
                          in_dim_,
                          weight->key_weights.kernel,
                          hidden_units_,
                          input,
                          in_dim_,
                          k,
                          hidden_units_);
    // v
    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          hidden_units_,
                          batch_size * seq_len,
                          in_dim_,
                          weight->value_weights.kernel,
                          hidden_units_,
                          input,
                          in_dim_,
                          v,
                          hidden_units_);
    // global k
    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          hidden_units_,
                          batch_size * seq_len,
                          in_dim_,
                          weight->global_key_weights.kernel,
                          hidden_units_,
                          input,
                          in_dim_,
                          kg,
                          hidden_units_);
    // global v
    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          hidden_units_,
                          batch_size * seq_len,
                          in_dim_,
                          weight->global_value_weights.kernel,
                          hidden_units_,
                          input,
                          in_dim_,
                          vg,
                          hidden_units_);

    // global key
    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        hidden_units_,
                                        max_global_token_num_,
                                        in_dim_,
                                        weight->global_query_weights.kernel,
                                        hidden_units_,
                                        0,
                                        input,
                                        in_dim_,
                                        seq_len * in_dim_,
                                        qg,
                                        hidden_units_,
                                        max_global_token_num_ * hidden_units_,
                                        batch_size);
    // reset all qkv pointer to transposed
    T* const q_mha = mha_qkv_buffer_;
    T* const k_mha = mha_qkv_buffer_ + batch_size * head_num_ * size_per_head_ * seq_len;
    T* const v_mha = mha_qkv_buffer_ + batch_size * head_num_ * size_per_head_ * 2 * seq_len;
    T* const kg_mha = mha_qkv_buffer_ + batch_size * head_num_ * size_per_head_ * 3 * seq_len;
    T* const vg_mha = mha_qkv_buffer_ + batch_size * head_num_ * size_per_head_ * 4 * seq_len;
    T* const qg_mha = mha_qkv_buffer_ + batch_size * head_num_ * size_per_head_ * 5 * seq_len;

    // q k v qk qv bias must stored continuously for this layer
    const T* qkv_5_bias = weight->query_weights.bias;
    invokeAddBiasTransposeToMultiHead(
        qkv_buffer_, qkv_5_bias, mha_qkv_buffer_, batch_size, head_num_, size_per_head_, seq_len, 5, stream_);
    sync_check_cuda_error();

    // calculate qg seperately cause the dimension is not the same with others.
    const T* qg_bias = weight->global_query_weights.bias;
    invokeAddBiasTransposeToMultiHead(
        qg, qg_bias, qg_mha, batch_size, head_num_, size_per_head_, max_global_token_num_, 1, stream_);
    sync_check_cuda_error();

    DataType data_type = getTensorType<T>();
    std::vector<Tensor> attn_inputs{
        Tensor{MEMORY_GPU, data_type, std::vector<size_t>{batch_size, head_num_, seq_len, size_per_head_}, q_mha},
        Tensor{MEMORY_GPU, data_type, std::vector<size_t>{batch_size, head_num_, seq_len, size_per_head_}, k_mha},
        Tensor{MEMORY_GPU, data_type, std::vector<size_t>{batch_size, head_num_, seq_len, size_per_head_}, v_mha},
        Tensor{MEMORY_GPU,
               data_type,
               std::vector<size_t>{batch_size, head_num_, max_global_token_num_, size_per_head_},
               qg_mha},
        Tensor{MEMORY_GPU, data_type, std::vector<size_t>{batch_size, head_num_, seq_len, size_per_head_}, kg_mha},
        Tensor{MEMORY_GPU, data_type, std::vector<size_t>{batch_size, head_num_, seq_len, size_per_head_}, vg_mha},
        Tensor{MEMORY_GPU, data_type, std::vector<size_t>{batch_size, seq_len}, local_attn_mask},
        Tensor{MEMORY_GPU, data_type, std::vector<size_t>{batch_size, seq_len}, global_attn_mask},
        Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size, seq_len}, global_idx},
        Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size}, global_token_nums}};
    std::vector<Tensor> longformer_outputs{Tensor{
        MEMORY_GPU, data_type, std::vector<size_t>{batch_size, head_num_, seq_len, size_per_head_}, mha_out_buffer_}};

    longformer_attn_layer_->forward(&longformer_outputs, &attn_inputs);

    invokeTransposeMultiHeadToSingle(
        attn_out_buffer_, mha_out_buffer_, batch_size, seq_len, head_num_, size_per_head_, stream_);
    sync_check_cuda_error();

    // attn output
    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          hidden_units_,
                          batch_size * seq_len,
                          hidden_units_,
                          weight->attention_output_weights.kernel,
                          hidden_units_,
                          attn_out_buffer_,
                          hidden_units_,
                          attn_output_buffer_,
                          hidden_units_);

    invokeAddBiasResidualLayerNorm(attn_output_buffer_,
                                   input,
                                   weight->attention_output_weights.bias,
                                   weight->attention_output_layernorm_weights.gamma,
                                   weight->attention_output_layernorm_weights.beta,
                                   batch_size * seq_len,
                                   hidden_units_,
                                   stream_);
    sync_check_cuda_error();

    // intermediate gemm + bias + gelu + output gemm
    std::vector<Tensor> attn_out_tensors = {
        Tensor{MEMORY_GPU, data_type, std::vector<size_t>{batch_size * seq_len, hidden_units_}, attn_output_buffer_}};
    std::vector<Tensor> output_tensors = {
        Tensor{MEMORY_GPU, data_type, std::vector<size_t>{batch_size * seq_len, hidden_units_}, output}};

    inter_gelu_out_ffn_->forward(&output_tensors, &attn_out_tensors, &(weight->ffn_weights));
    sync_check_cuda_error();

    // + bias and residual
    invokeAddBiasResidualLayerNorm(output,
                                   attn_output_buffer_,
                                   weight->ffn_weights.output_weight.bias,
                                   weight->output_layernorm_weights.gamma,
                                   weight->output_layernorm_weights.beta,
                                   batch_size * seq_len,
                                   hidden_units_,
                                   stream_);
    sync_check_cuda_error();
}

template class LongformerEncoder<float>;
template class LongformerEncoder<half>;

}  // namespace fastertransformer