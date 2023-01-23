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

#pragma once

#include "src/fastertransformer/layers/FfnFP8Layer.h"
#include "src/fastertransformer/utils/cuda_fp8_utils.h"
#include "src/fastertransformer/utils/nccl_utils.h"

namespace fastertransformer {

template<typename T1, typename T2>
class TensorParallelGeluFfnFP8Layer: public GeluFfnFP8Layer<T1, T2> {
private:
    NcclParam tensor_para_;

protected:
public:
    TensorParallelGeluFfnFP8Layer(size_t           inter_size,
                                  NcclParam        tensor_para,
                                  int              fp8_mode,
                                  cudaStream_t     stream,
                                  cublasMMWrapper* cublas_wrapper,
                                  IAllocator*      allocator,
                                  bool             is_free_buffer_after_forward,
                                  bool             is_sparse = false);

    TensorParallelGeluFfnFP8Layer(TensorParallelGeluFfnFP8Layer<T1, T2> const& ffn_layer);

    virtual ~TensorParallelGeluFfnFP8Layer() = default;

    void forward(TensorMap* output_tensors, TensorMap* input_tensors, const FfnFP8Weight<T1, T2>* ffn_weights) override;
};

}  // namespace fastertransformer
