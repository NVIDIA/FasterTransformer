/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/kernels/decoder_masked_multihead_attention.h"
#include "src/fastertransformer/layers/attention_layers_fp8/AttentionFP8Weight.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include "src/fastertransformer/utils/cuda_fp8_utils.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

template<typename T>
struct GroupedQuery_attention_params: public Multihead_attention_params_base<T> {
    // allows to exist attention eary
    bool* finished          = nullptr;
    int   num_kv_heads      = 0;
    // required in case of masked attention with different length
    const int* length_per_sample = nullptr;

    float rope_theta;
};

template<class T>
using Masked_groupedquery_attention_params = GroupedQuery_attention_params<T>;

////////////////////////////////////////////////////////////////////////////////////////////////////

void masked_groupedquery_attention(const Masked_groupedquery_attention_params<float>& params, const cudaStream_t& stream);
void masked_groupedquery_attention(const Masked_groupedquery_attention_params<uint16_t>& params, const cudaStream_t& stream);
#ifdef ENABLE_BF16
void masked_groupedquery_attention(const Masked_groupedquery_attention_params<__nv_bfloat16>& params,
                                const cudaStream_t&                                     stream);
#endif
#ifdef ENABLE_FP8
void masked_groupedquery_attention(const Masked_groupedquery_attention_params<__nv_fp8_e4m3>& params,
                                const cudaStream_t&                                     stream);
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
