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

#include "src/fastertransformer/layers/attention_layers/TensorParallelDecoderSelfAttentionLayer.h"

namespace fastertransformer {

template<typename T>
TensorParallelDecoderSelfAttentionLayer<T>::TensorParallelDecoderSelfAttentionLayer(
    size_t max_batch_size,
    size_t head_num,
    size_t size_per_head,
    size_t tensor_para_size,
    ncclComm_t tensor_para_comm,
    cudaStream_t stream,
    cublasMMWrapper* cublas_wrapper,
    IAllocator* allocator,
    bool is_free_buffer_after_forward
):
    DecoderSelfAttentionLayer<T>(max_batch_size,
                                 head_num,
                                 size_per_head,
                                 head_num / tensor_para_size,
                                 stream,
                                 cublas_wrapper,
                                 allocator,
                                 is_free_buffer_after_forward),
    tensor_para_size_(tensor_para_size),
    tensor_para_comm_(tensor_para_comm)
{
}

template<typename T>
TensorParallelDecoderSelfAttentionLayer<T>::TensorParallelDecoderSelfAttentionLayer(
    size_t max_batch_size,
    size_t head_num,
    size_t size_per_head,
    size_t rotary_embedding_dim,
    size_t tensor_para_size,
    ncclComm_t tensor_para_comm,
    cudaStream_t stream,
    cublasMMWrapper* cublas_wrapper,
    IAllocator* allocator,
    bool is_free_buffer_after_forward
):
    DecoderSelfAttentionLayer<T>(max_batch_size,
                                 head_num,
                                 size_per_head,
                                 head_num / tensor_para_size,
                                 rotary_embedding_dim,
                                 stream,
                                 cublas_wrapper,
                                 allocator,
                                 is_free_buffer_after_forward),
    tensor_para_size_(tensor_para_size),
    tensor_para_comm_(tensor_para_comm)
{
}

template<typename T>
TensorParallelDecoderSelfAttentionLayer<T>::TensorParallelDecoderSelfAttentionLayer(
    TensorParallelDecoderSelfAttentionLayer<T> const& attention_layer):
    DecoderSelfAttentionLayer<T>(attention_layer),
    tensor_para_size_(attention_layer.tensor_para_size_),
    tensor_para_comm_(attention_layer.tensor_para_comm_)
{
}

template<typename T>
void TensorParallelDecoderSelfAttentionLayer<T>::forward(std::vector<fastertransformer::Tensor>* output_tensors,
                                                         const std::vector<fastertransformer::Tensor>* input_tensors,
                                                         const AttentionWeight<T>* attention_weights)
{
    // input tensors:
    //      attention_input [batch_size, hidden_dimension],
    //      finished [batch_size],
    //      sequence_lengths [batch_size]
    //      input_lengths [batch_size]
    //      max_input_length [1] on cpu
    //      step [1] on cpu

    // output tensors:
    //      attention_output [batch_size, hidden_dimension],
    //      key_cache [batch, head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [batch, head_num, max_seq_len, size_per_head]

    DecoderSelfAttentionLayer<T>::forward(output_tensors, input_tensors, attention_weights);

    const size_t batch_size = output_tensors->at(0).shape[0];
    const size_t hidden_units = output_tensors->at(0).shape[1];

    T* attention_out = (T*)(output_tensors->at(0).data);
    if (tensor_para_size_ > 1) {
        ftNcclAllReduceSum(attention_out,
                           attention_out,
                           batch_size * hidden_units,
                           tensor_para_comm_,
                           DecoderSelfAttentionLayer<T>::stream_);
        sync_check_cuda_error();
    }
}

template class TensorParallelDecoderSelfAttentionLayer<float>;
template class TensorParallelDecoderSelfAttentionLayer<half>;

}  // namespace fastertransformer