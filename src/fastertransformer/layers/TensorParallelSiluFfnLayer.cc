/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/layers/TensorParallelSiluFfnLayer.h"

namespace fastertransformer {

template<typename T>
void TensorParallelSiluFfnLayer<T>::forward(std::vector<fastertransformer::Tensor>*       output_tensors,
                                            const std::vector<fastertransformer::Tensor>* input_tensors,
                                            const FfnWeight<T>*                           ffn_weights)
{
    TensorMap input_tensor({{"ffn_input", input_tensors->at(0)}});
    TensorMap output_tensor({{"ffn_output", output_tensors->at(0)}});
    forward(&output_tensor, &input_tensor, ffn_weights);
}

template<typename T>
void TensorParallelSiluFfnLayer<T>::forward(TensorMap*          output_tensors,
                                            TensorMap*          input_tensors,
                                            const FfnWeight<T>* ffn_weights)
{
    FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    Tensor       out_tensor   = output_tensors->at("ffn_output");
    const size_t token_num    = out_tensor.shape[0];
    const size_t hidden_units = out_tensor.shape[1];

    std::vector<Tensor> swap_tensors = {out_tensor};

    bool use_custom_all_reduce_kernel = false;
    if (enable_custom_all_reduce_ && custom_all_reduce_comm_ != nullptr) {
        use_custom_all_reduce_kernel =
            custom_all_reduce_comm_->swapInternalBuffer(&swap_tensors, token_num * hidden_units);
    }

    SiluFfnLayer<T>::forward(output_tensors, input_tensors, ffn_weights);

    T* ffn_out = out_tensor.getPtr<T>();
    if (do_all_reduce_ && tensor_para_.world_size_ > 1) {
        if (!use_custom_all_reduce_kernel) {
            ftNcclAllReduceSum(ffn_out, ffn_out, token_num * hidden_units, tensor_para_, SiluFfnLayer<T>::stream_);
        }
        else {
            custom_all_reduce_comm_->customAllReduce(token_num * hidden_units, SiluFfnLayer<T>::stream_);
        }
        sync_check_cuda_error();
    }
}

template<typename T>
TensorParallelSiluFfnLayer<T>::TensorParallelSiluFfnLayer(size_t           max_batch_size,
                                                          size_t           max_seq_len,
                                                          size_t           head_num,
                                                          size_t           size_per_head,
                                                          size_t           expert_num,
                                                          size_t           inter_size,
                                                          NcclParam        tensor_para,
                                                          cudaStream_t     stream,
                                                          cublasMMWrapper* cublas_wrapper,
                                                          IAllocator*      allocator,
                                                          bool             do_all_reduce,
                                                          bool             is_free_buffer_after_forward,
                                                          bool             is_sparse,
                                                          bool             use_gated_activation,
                                                          std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                                                          int                                 enable_custom_all_reduce):
    SiluFfnLayer<T>(max_batch_size,
                    max_seq_len,
                    head_num,
                    size_per_head,
                    expert_num,
                    inter_size / tensor_para.world_size_,
                    stream,
                    cublas_wrapper,
                    allocator,
                    is_free_buffer_after_forward,
                    is_sparse,
                    use_gated_activation),
    tensor_para_(tensor_para),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce),
    do_all_reduce_(do_all_reduce)
{
    FT_CHECK(inter_size % tensor_para_.world_size_ == 0);
}

template<typename T>
TensorParallelSiluFfnLayer<T>::TensorParallelSiluFfnLayer(TensorParallelSiluFfnLayer<T> const& ffn_layer):
    SiluFfnLayer<T>(ffn_layer), tensor_para_(ffn_layer.tensor_para_), do_all_reduce_(ffn_layer.do_all_reduce_)
{
}

template class TensorParallelSiluFfnLayer<float>;
template class TensorParallelSiluFfnLayer<half>;
#ifdef ENABLE_BF16
template class TensorParallelSiluFfnLayer<__nv_bfloat16>;
#endif

}  // namespace fastertransformer