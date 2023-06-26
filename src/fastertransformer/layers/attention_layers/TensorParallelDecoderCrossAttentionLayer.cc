/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/layers/attention_layers/TensorParallelDecoderCrossAttentionLayer.h"

namespace fastertransformer {

template<typename T>
TensorParallelDecoderCrossAttentionLayer<T>::TensorParallelDecoderCrossAttentionLayer(
    size_t                              max_batch_size,
    size_t                              head_num,
    size_t                              size_per_head,
    size_t                              d_model,
    float                               q_scaling,
    NcclParam                           tensor_para,
    cudaStream_t                        stream,
    cublasMMWrapper*                    cublas_wrapper,
    IAllocator*                         allocator,
    bool                                is_free_buffer_after_forward,
    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
    int                                 enable_custom_all_reduce):
    DecoderCrossAttentionLayer<T>(max_batch_size,
                                  head_num / tensor_para.world_size_,
                                  size_per_head,
                                  d_model,
                                  q_scaling,
                                  stream,
                                  cublas_wrapper,
                                  allocator,
                                  is_free_buffer_after_forward),
    tensor_para_(tensor_para),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce)
{
    FT_CHECK(head_num % tensor_para_.world_size_ == 0);
}

template<typename T>
TensorParallelDecoderCrossAttentionLayer<T>::TensorParallelDecoderCrossAttentionLayer(
    size_t                              max_batch_size,
    size_t                              head_num,
    size_t                              size_per_head,
    NcclParam                           tensor_para,
    cudaStream_t                        stream,
    cublasMMWrapper*                    cublas_wrapper,
    IAllocator*                         allocator,
    bool                                is_free_buffer_after_forward,
    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
    int                                 enable_custom_all_reduce):
    TensorParallelDecoderCrossAttentionLayer(max_batch_size,
                                             head_num,
                                             size_per_head,
                                             head_num * size_per_head,
                                             1.0f,
                                             tensor_para,
                                             stream,
                                             cublas_wrapper,
                                             allocator,
                                             is_free_buffer_after_forward,
                                             custom_all_reduce_comm,
                                             enable_custom_all_reduce)
{
}

template<typename T>
TensorParallelDecoderCrossAttentionLayer<T>::TensorParallelDecoderCrossAttentionLayer(
    TensorParallelDecoderCrossAttentionLayer<T> const& attention_layer):
    DecoderCrossAttentionLayer<T>(attention_layer),
    tensor_para_(attention_layer.tensor_para_),
    custom_all_reduce_comm_(attention_layer.custom_all_reduce_comm_),
    enable_custom_all_reduce_(attention_layer.enable_custom_all_reduce_)
{
}

template<typename T>
void TensorParallelDecoderCrossAttentionLayer<T>::forward(TensorMap*                output_tensors,
                                                          TensorMap*                input_tensors,
                                                          const AttentionWeight<T>* attention_weights)
{
    // input tensors:
    //      input_query [batch_size, hidden_dimension],
    //      finished [batch_size],
    //      sequence_lengths [batch_size]
    //      input_lengths [batch_size]
    //      max_input_length [1] on cpu
    //      step [1] on cpu

    // output tensors:
    //      hidden_features [batch_size, hidden_dimension],
    //      key_cache [batch, head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [batch, head_num, max_seq_len, size_per_head]

    const size_t size = output_tensors->at("hidden_features").size();
    std::vector<Tensor> reduce_tensor{output_tensors->at("hidden_features")};

    bool use_custom_all_reduce_kernel = false;
    if (enable_custom_all_reduce_ && custom_all_reduce_comm_ != nullptr) {
        use_custom_all_reduce_kernel = custom_all_reduce_comm_->swapInternalBuffer(&reduce_tensor, size);
        output_tensors->at("hidden_features").data = reduce_tensor[0].data;
    }

    DecoderCrossAttentionLayer<T>::forward(output_tensors, input_tensors, attention_weights);

    T* attention_out = output_tensors->getPtr<T>("hidden_features");
    if (tensor_para_.world_size_ > 1) {
        if (!use_custom_all_reduce_kernel) {
            ftNcclAllReduceSum(
                attention_out, attention_out, size, tensor_para_, DecoderCrossAttentionLayer<T>::stream_);
        }
        else {
            custom_all_reduce_comm_->customAllReduce(size, DecoderCrossAttentionLayer<T>::stream_);
            output_tensors->at("hidden_features").data = reduce_tensor[0].data;
        }
        sync_check_cuda_error();
    }
}

template class TensorParallelDecoderCrossAttentionLayer<float>;
template class TensorParallelDecoderCrossAttentionLayer<half>;
#ifdef ENABLE_BF16
template class TensorParallelDecoderCrossAttentionLayer<__nv_bfloat16>;
#endif

}  // namespace fastertransformer