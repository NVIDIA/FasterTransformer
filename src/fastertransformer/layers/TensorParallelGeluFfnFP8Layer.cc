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

#include "src/fastertransformer/layers/TensorParallelGeluFfnFP8Layer.h"

namespace fastertransformer {

template<typename T1, typename T2>
void TensorParallelGeluFfnFP8Layer<T1, T2>::forward(TensorMap*                  output_tensors,
                                                    TensorMap*                  input_tensors,
                                                    const FfnFP8Weight<T1, T2>* ffn_weights)
{
    GeluFfnFP8Layer<T1, T2>::forward(output_tensors, input_tensors, ffn_weights);

    const size_t token_num    = output_tensors->at("output_hidden_state").shape[0];
    const size_t hidden_units = output_tensors->at("output_hidden_state").shape[1];

    T2* ffn_out = output_tensors->at("output_hidden_state").getPtr<T2>();
    if (tensor_para_.world_size_ > 1) {
        ftNcclAllReduceSum(ffn_out, ffn_out, token_num * hidden_units, tensor_para_, GeluFfnFP8Layer<T1, T2>::stream_);
        sync_check_cuda_error();
    }
}

template<typename T1, typename T2>
TensorParallelGeluFfnFP8Layer<T1, T2>::TensorParallelGeluFfnFP8Layer(size_t           inter_size,
                                                                     NcclParam        tensor_para,
                                                                     int              fp8_mode,
                                                                     cudaStream_t     stream,
                                                                     cublasMMWrapper* cublas_wrapper,
                                                                     IAllocator*      allocator,
                                                                     bool             is_free_buffer_after_forward,
                                                                     bool             is_sparse):
    GeluFfnFP8Layer<T1, T2>(inter_size / tensor_para.world_size_,
                            fp8_mode,
                            stream,
                            cublas_wrapper,
                            allocator,
                            is_free_buffer_after_forward,
                            is_sparse),
    tensor_para_(tensor_para)
{
    FT_CHECK(inter_size % tensor_para_.world_size_ == 0);
}

template<typename T1, typename T2>
TensorParallelGeluFfnFP8Layer<T1, T2>::TensorParallelGeluFfnFP8Layer(
    TensorParallelGeluFfnFP8Layer<T1, T2> const& ffn_layer):
    GeluFfnFP8Layer<T1, T2>(ffn_layer), tensor_para_(ffn_layer.tensor_para_)
{
}

template class TensorParallelGeluFfnFP8Layer<__nv_fp8_e4m3, __nv_bfloat16>;

}  // namespace fastertransformer
