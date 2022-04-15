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

#pragma once

#include <string>

#include "src/fastertransformer/kernels/calibrate_quantize_weight_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"

namespace fastertransformer {

template<typename T>
struct ParallelGptDecoderLayerWeight {
public:
    ParallelGptDecoderLayerWeight() = default;
    ParallelGptDecoderLayerWeight(const int int8_mode);
    ParallelGptDecoderLayerWeight(const int hidden_units,
                                  const int inter_size,
                                  const int tensor_para_size,
                                  const int tensor_para_rank,
                                  const int int8_mode = 0);
    ~ParallelGptDecoderLayerWeight();
    ParallelGptDecoderLayerWeight(const ParallelGptDecoderLayerWeight& other);
    ParallelGptDecoderLayerWeight& operator=(const ParallelGptDecoderLayerWeight& other);
    void loadModel(std::string dir_path, FtCudaDataType model_file_type);
#ifdef SPARSITY_ENABLED
    void compress_weights(cublasMMWrapper& cublas_wrapper, int hidden_dim);
#endif
    void transposeCalibrateQuantizeWeight();

    LayerNormWeight<T> pre_layernorm_weights;
    AttentionWeight<T> self_attention_weights;
    LayerNormWeight<T> self_attn_layernorm_weights;
    FfnWeight<T> ffn_weights;

private:
    void setWeightPtr();
    void mallocWeights();

protected:
    size_t hidden_units_;
    size_t inter_size_;
    size_t tensor_para_size_ = 1;
    size_t tensor_para_rank_ = 0;
    bool is_maintain_buffer = false;
    int int8_mode_ = 0;
    T* weights_ptr[12];

    int8_t* int8_weights_ptr[4];
    float* scale_ptr[4];
    cudaStream_t stream_ = 0;

#ifdef SPARSITY_ENABLED
    T* sp_weights_ptr[4];
    bool is_maintain_sp_buffer = false;
#endif
};

}  // namespace fastertransformer
