/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "src/fastertransformer/utils/nvtx_utils.h"

namespace fastertransformer {

template<typename T>
void TensorParallelGptContextAttentionLayer<T>::forward(TensorMap*                output_tensors,
                                                        TensorMap*                input_tensors,
                                                        const AttentionWeight<T>* attention_weights)
{
    // input_tensors:
    //      input_query [batch_size * seq_len, hidden_dimension]
    //      attention_mask [batch_size, 1, seq_len, seq_len]
    //      is_final_layer [1], bool on cpu

    // output_tensors:
    //      hidden_features [batch_size * seq_len, hidden_dimension]
    //      key_cache [batch, local_head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [batch, local_head_num, max_seq_len, size_per_head]

    const size_t size = output_tensors->at("hidden_features").size();

    bool use_custom_all_reduce_kernel = false;
    if (do_all_reduce_ && enable_custom_all_reduce_ && custom_all_reduce_comm_ != nullptr) {
        std::vector<Tensor> reduce_tensor{output_tensors->at("hidden_features")};
        use_custom_all_reduce_kernel = custom_all_reduce_comm_->swapInternalBuffer(&reduce_tensor, size);
    }

    GptContextAttentionLayer<T>::forward(output_tensors, input_tensors, attention_weights);

    PUSH_RANGE("all reduce sum");
    T* attention_out = output_tensors->getPtr<T>("hidden_features");
    if (do_all_reduce_ && tensor_para_.world_size_ > 1) {
        if (!use_custom_all_reduce_kernel) {
            ftNcclAllReduceSum(attention_out, attention_out, size, tensor_para_, GptContextAttentionLayer<T>::stream_);
        }
        else {
            custom_all_reduce_comm_->customAllReduce(size, GptContextAttentionLayer<T>::stream_);
        }
        sync_check_cuda_error();
    }
    POP_RANGE;
}

template<typename T>
TensorParallelGptContextAttentionLayer<T>::TensorParallelGptContextAttentionLayer(
    size_t                              max_batch_size,
    size_t                              max_seq_len,
    size_t                              head_num,
    size_t                              size_per_head,
    NcclParam                           tensor_para,
    cudaStream_t                        stream,
    cublasMMWrapper*                    cublas_wrapper,
    IAllocator*                         allocator,
    bool                                do_all_reduce,
    bool                                is_free_buffer_after_forward,
    bool                                is_qk_buf_float,
    bool                                sparse,
    int                                 int8_mode,
    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
    int                                 enable_custom_all_reduce):
    GptContextAttentionLayer<T>(max_batch_size,
                                max_seq_len,
                                head_num,
                                size_per_head,
                                head_num / tensor_para.world_size_,
                                stream,
                                cublas_wrapper,
                                allocator,
                                is_free_buffer_after_forward,
                                is_qk_buf_float,
                                sparse,
                                int8_mode),
    tensor_para_(tensor_para),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce),
    do_all_reduce_(do_all_reduce)
{
    FT_CHECK(head_num % tensor_para_.world_size_ == 0);
}

template<typename T>
TensorParallelGptContextAttentionLayer<T>::TensorParallelGptContextAttentionLayer(
    size_t                              max_batch_size,
    size_t                              max_seq_len,
    size_t                              head_num,
    size_t                              size_per_head,
    size_t                              rotary_embedding_dim,
    bool                                neox_rotary_style,
    NcclParam                           tensor_para,
    cudaStream_t                        stream,
    cublasMMWrapper*                    cublas_wrapper,
    IAllocator*                         allocator,
    bool                                do_all_reduce,
    bool                                is_free_buffer_after_forward,
    bool                                is_qk_buf_float,
    bool                                sparse,
    int                                 int8_mode,
    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
    int                                 enable_custom_all_reduce):
    GptContextAttentionLayer<T>(max_batch_size,
                                max_seq_len,
                                head_num,
                                size_per_head,
                                head_num / tensor_para.world_size_,
                                rotary_embedding_dim,
                                neox_rotary_style,
                                stream,
                                cublas_wrapper,
                                allocator,
                                is_free_buffer_after_forward,
                                is_qk_buf_float,
                                sparse,
                                int8_mode),
    tensor_para_(tensor_para),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce),
    do_all_reduce_(do_all_reduce)
{
    FT_CHECK(head_num % tensor_para_.world_size_ == 0);
}

template<typename T>
TensorParallelGptContextAttentionLayer<T>::TensorParallelGptContextAttentionLayer(
    TensorParallelGptContextAttentionLayer<T> const& attention_layer):
    GptContextAttentionLayer<T>(attention_layer),
    tensor_para_(attention_layer.tensor_para_),
    custom_all_reduce_comm_(attention_layer.custom_all_reduce_comm_),
    enable_custom_all_reduce_(attention_layer.enable_custom_all_reduce_),
    do_all_reduce_(attention_layer.do_all_reduce_)
{
}

template class TensorParallelGptContextAttentionLayer<float>;
template class TensorParallelGptContextAttentionLayer<half>;
#ifdef ENABLE_BF16
template class TensorParallelGptContextAttentionLayer<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
