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

#include "src/fastertransformer/layers/TensorParallelGeluFfnLayer.h"

namespace fastertransformer {

template<typename T>
void TensorParallelGeluFfnLayer<T>::forward(std::vector<fastertransformer::Tensor>* output_tensors,
                                            const std::vector<fastertransformer::Tensor>* input_tensors,
                                            const FfnWeight<T>* ffn_weights)
{
    GeluFfnLayer<T>::forward(output_tensors, input_tensors, ffn_weights);

    const size_t token_num = output_tensors->at(0).shape[0];
    const size_t hidden_units = output_tensors->at(0).shape[1];

    T* ffn_out = (T*)(output_tensors->at(0).data);
    if (tensor_para_size_ > 1) {
        ftNcclAllReduceSum(ffn_out, ffn_out, token_num * hidden_units, tensor_para_comm_, GeluFfnLayer<T>::stream_);
        sync_check_cuda_error();
    }
}

template<typename T>
TensorParallelGeluFfnLayer<T>::TensorParallelGeluFfnLayer(size_t max_batch_size,
                                                          size_t max_seq_len,
                                                          size_t head_num,
                                                          size_t size_per_head,
                                                          size_t inter_size,
                                                          size_t tensor_para_size,
                                                          ncclComm_t tensor_para_comm,
                                                          cudaStream_t stream,
                                                          cublasMMWrapper* cublas_wrapper,
                                                          IAllocator* allocator,
                                                          bool is_free_buffer_after_forward,
                                                          bool is_sparse,
                                                          int int8_mode):
    GeluFfnLayer<T>(max_batch_size,
                    max_seq_len,
                    head_num,
                    size_per_head,
                    inter_size / tensor_para_size,
                    stream,
                    cublas_wrapper,
                    allocator,
                    is_free_buffer_after_forward,
                    is_sparse,
                    int8_mode),
    tensor_para_size_(tensor_para_size),
    tensor_para_comm_(tensor_para_comm)
{
    FT_CHECK(inter_size % tensor_para_size == 0);
}

template<typename T>
TensorParallelGeluFfnLayer<T>::TensorParallelGeluFfnLayer(TensorParallelGeluFfnLayer<T> const& ffn_layer):
    GeluFfnLayer<T>(ffn_layer),
    tensor_para_size_(ffn_layer.tensor_para_size_),
    tensor_para_comm_(ffn_layer.tensor_para_comm_)
{
}

template class TensorParallelGeluFfnLayer<float>;
template class TensorParallelGeluFfnLayer<half>;

}  // namespace fastertransformer
