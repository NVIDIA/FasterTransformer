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

#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/utils/cuda_fp8_utils.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace fastertransformer {

template<typename T, int QUANTIZE_MODE>
void invokeQuatizeVectorE4M3(__nv_fp8_e4m3* output,
                             float const*   input_qua_amax_ptr,
                             T const*       input,
                             uint32_t       size,
                             uint32_t       n,
                             cudaStream_t   stream);

template<typename T, int QUANTIZE_MODE>
void invokeDequatizeVectorE4M3(
    T* output, float const* qua_amax_ptr, __nv_fp8_e4m3 const* input, uint32_t size, uint32_t n, cudaStream_t stream);

template<typename T1, typename T2>
struct FP8LayerNormParam {
    T1*       normed_output;
    const T2* input;
    const T2* gamma = nullptr;
    const T2* beta  = nullptr;

    // Can we use half here?
    const float* input_deq_ptr;
    const float* output_qua_ptr;

    int          m;
    int          n;
    cudaStream_t stream;
    bool         use_fp8x2;
};

template<typename T1, typename T2, int QUANTIZE_MODE = 1>
void invokeFP8LayerNorm(FP8LayerNormParam<T1, T2> param);

template<typename T1, typename T2>
struct GeneralFP8IOPostLayerNormParam {
    T1*       normed_output;
    const T1* input;
    const T2* gamma = nullptr;
    const T2* beta  = nullptr;

    // Can we use half here?
    const float* input_deq_ptr;
    const float* output_qua_ptr;

    int          m;
    int          n;
    cudaStream_t stream;
    bool         use_fp8x2;
};

template<typename T1, typename T2, int QUANTIZE_MODE = 1>
void invokeGeneralFP8IOPostLayerNorm(GeneralFP8IOPostLayerNormParam<T1, T2> param);

// template<typename T1, typename T2>
// void cudaCast(T1* tgt, T2 const* src, const uint64_t size);

template<typename T1, typename T2>
struct GeneralFP8AddBiasResidualPreLayerNormParam {
    T1*       normed_output;
    T2*       output;
    const T2* residual;
    const T2* bias  = nullptr;
    const T2* gamma = nullptr;
    const T2* beta  = nullptr;

    const float* input_deq_ptr;
    const float* output_qua_ptr;

    int          m;
    int          n;
    cudaStream_t stream;
    bool         use_fp8x2;
};

template<typename T1, typename T2, int QUANTIZE_MODE = 1>
void invokeGeneralFP8AddBiasResidualPreLayerNorm(GeneralFP8AddBiasResidualPreLayerNormParam<T1, T2> param);

template<typename T1, typename T2>
struct GeneralFP8IOAddBiasResidualPostLayerNormParam {
    T1*       normed_output;
    T1*       input;
    const T1* residual;
    const T2* gamma = nullptr;
    const T2* beta  = nullptr;
    const T2* bias  = nullptr;

    // Can we use half here?
    const float* input_scale;    // for per tensor weight scale
    const float* input_scale_2;  // for per channel weight scale
    const float* input_scale_2_min;
    const float* output_scale;
    const float* residual_scale;

    int          m;
    int          n;
    cudaStream_t stream;
    bool         use_fp8x2;
};

template<typename T1, typename T2, int QUANTIZE_MODE = 1>
void invokeGeneralFP8IOAddBiasResidualPostLayerNorm(GeneralFP8IOAddBiasResidualPostLayerNormParam<T1, T2> param);

template<typename T1, typename T2>
struct RemovePaddingEmbLookupLayerNormFP8OutParam {
    T1*              normed_output;
    int const* const input_ids;
    int const* const position_ids;
    int const* const token_type_ids;
    int const* const padding_offset;
    T2 const* const  word_embeddings;
    T2 const* const  position_embeddings;
    T2 const* const  token_type_embeddings;

    T2 const* const gamma = nullptr;
    T2 const* const beta  = nullptr;

    // Can we use half here?
    const float*     output_scale;
    int const* const lengths;

    int const    m;
    int const    n;
    int const    batch_size;
    int const    max_seq_len;
    int const    pad_token_id;
    cudaStream_t stream;
    bool         use_fp8x2;
};

template<typename T1, typename T2>
void invokeRemovePaddingEmbLookupLayerNormFP8Out(RemovePaddingEmbLookupLayerNormFP8OutParam<T1, T2> param);

}  // namespace fastertransformer
