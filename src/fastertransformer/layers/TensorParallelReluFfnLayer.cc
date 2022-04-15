/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/layers/TensorParallelReluFfnLayer.h"

namespace fastertransformer {

template<typename T>
void TensorParallelReluFfnLayer<T>::forward(std::vector<fastertransformer::Tensor>* output_tensors,
                                            const std::vector<fastertransformer::Tensor>* input_tensors,
                                            const FfnWeight<T>* ffn_weights)
{
    const size_t token_num = output_tensors->at(0).shape[0];
    const size_t hidden_units = output_tensors->at(0).shape[1];

    bool use_custom_all_reduce_kernel = false;
    if (enable_custom_all_reduce_ && custom_all_reduce_comm_ != nullptr) {
        use_custom_all_reduce_kernel =
            custom_all_reduce_comm_->swapInternalBuffer(output_tensors, token_num * hidden_units);
    }

    ReluFfnLayer<T>::forward(output_tensors, input_tensors, ffn_weights);

    T* ffn_out = (T*)(output_tensors->at(0).data);
    if (tensor_para_.world_size_ > 1) {
        if (!use_custom_all_reduce_kernel) {
            ftNcclAllReduceSum(ffn_out, ffn_out, token_num * hidden_units, tensor_para_, ReluFfnLayer<T>::stream_);
        }
        else {
            custom_all_reduce_comm_->customAllReduce(token_num * hidden_units, ReluFfnLayer<T>::stream_);
        }
        sync_check_cuda_error();
    }
}

template<typename T>
TensorParallelReluFfnLayer<T>::TensorParallelReluFfnLayer(size_t max_batch_size,
                                                          size_t max_seq_len,
                                                          size_t head_num,
                                                          size_t size_per_head,
                                                          size_t inter_size,
                                                          NcclParam tensor_para,
                                                          cudaStream_t stream,
                                                          cublasMMWrapper* cublas_wrapper,
                                                          IAllocator* allocator,
                                                          bool is_free_buffer_after_forward,
                                                          bool is_sparse,
                                                          std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                                                          int enable_custom_all_reduce):
    ReluFfnLayer<T>(max_batch_size,
                    max_seq_len,
                    head_num,
                    size_per_head,
                    inter_size / tensor_para.world_size_,
                    stream,
                    cublas_wrapper,
                    allocator,
                    is_free_buffer_after_forward,
                    is_sparse),
    tensor_para_(tensor_para),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce)
{
    FT_CHECK(inter_size % tensor_para_.world_size_ == 0);
}

template<typename T>
TensorParallelReluFfnLayer<T>::TensorParallelReluFfnLayer(TensorParallelReluFfnLayer<T> const& ffn_layer):
    ReluFfnLayer<T>(ffn_layer), tensor_para_(ffn_layer.tensor_para_)
{
}

template class TensorParallelReluFfnLayer<float>;
template class TensorParallelReluFfnLayer<half>;

}  // namespace fastertransformer