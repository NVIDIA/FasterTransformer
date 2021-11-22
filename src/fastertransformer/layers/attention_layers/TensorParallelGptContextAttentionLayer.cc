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

#include "src/fastertransformer/layers/attention_layers/TensorParallelGptContextAttentionLayer.h"

namespace fastertransformer {

template<typename T>
void TensorParallelGptContextAttentionLayer<T>::forward(std::vector<fastertransformer::Tensor>* output_tensors,
                                                        const std::vector<fastertransformer::Tensor>* input_tensors,
                                                        const AttentionWeight<T>* attention_weights)
{
    // input_tensors:
    //      input_query [batch_size * seq_len, hidden_dimension]
    //      attention_mask [batch_size, 1, seq_len, seq_len]
    //      is_final_layer [1], bool on cpu

    // output_tensors:
    //      attention_out [batch_size * seq_len, hidden_dimension]
    //      key_cache [batch, local_head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [batch, local_head_num, max_seq_len, size_per_head]

    GptContextAttentionLayer<T>::forward(output_tensors, input_tensors, attention_weights);

    const size_t m = output_tensors->at(0).shape[0];
    const size_t hidden_units = output_tensors->at(0).shape[1];

    T* attention_out = (T*)(output_tensors->at(0).data);
    if (tensor_para_size_ > 1) {
        ftNcclAllReduceSum(
            attention_out, attention_out, m * hidden_units, tensor_para_comm_, GptContextAttentionLayer<T>::stream_);
        sync_check_cuda_error();
    }
}

template<typename T>
TensorParallelGptContextAttentionLayer<T>::TensorParallelGptContextAttentionLayer(
    size_t max_batch_size,
    size_t max_seq_len,
    size_t head_num,
    size_t size_per_head,
    size_t tensor_para_size,
    ncclComm_t tensor_para_comm,
    cudaStream_t stream,
    cublasMMWrapper* cublas_wrapper,
    IAllocator* allocator,
    bool is_free_buffer_after_forward,
    bool is_qk_buf_float,
    bool sparse
):
    GptContextAttentionLayer<T>(max_batch_size,
                                max_seq_len,
                                head_num,
                                size_per_head,
                                head_num / tensor_para_size,
                                stream,
                                cublas_wrapper,
                                allocator,
                                is_free_buffer_after_forward,
                                is_qk_buf_float,
                                sparse),
    tensor_para_size_(tensor_para_size),
    tensor_para_comm_(tensor_para_comm)
{
    FT_CHECK(head_num % tensor_para_size == 0);
}

template<typename T>
TensorParallelGptContextAttentionLayer<T>::TensorParallelGptContextAttentionLayer(
    size_t max_batch_size,
    size_t max_seq_len,
    size_t head_num,
    size_t size_per_head,
    size_t rotary_embedding_dim,
    size_t tensor_para_size,
    ncclComm_t tensor_para_comm,
    cudaStream_t stream,
    cublasMMWrapper* cublas_wrapper,
    IAllocator* allocator,
    bool is_free_buffer_after_forward,
    bool is_qk_buf_float,
    bool sparse
):
    GptContextAttentionLayer<T>(max_batch_size,
                                max_seq_len,
                                head_num,
                                size_per_head,
                                head_num / tensor_para_size,
                                rotary_embedding_dim,
                                stream,
                                cublas_wrapper,
                                allocator,
                                is_free_buffer_after_forward,
                                is_qk_buf_float,
                                sparse),
    tensor_para_size_(tensor_para_size),
    tensor_para_comm_(tensor_para_comm)
{
}

template<typename T>
TensorParallelGptContextAttentionLayer<T>::TensorParallelGptContextAttentionLayer(
    TensorParallelGptContextAttentionLayer<T> const& attention_layer):
    GptContextAttentionLayer<T>(attention_layer),
    tensor_para_size_(attention_layer.tensor_para_size_),
    tensor_para_comm_(attention_layer.tensor_para_comm_)
{
}

template class TensorParallelGptContextAttentionLayer<float>;
template class TensorParallelGptContextAttentionLayer<half>;

}  // namespace fastertransformer