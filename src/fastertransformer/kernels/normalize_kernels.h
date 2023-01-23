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

#pragma once

#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fastertransformer {

// input should be the qkv for trt fmha kernels with shape of [batch, seqlen, num_head, 3, size_per_head]
// do normlization on size_per_head for q & k [-, -, -, 0, *] && [-, -, -, 1, *] && *(logit_scale/sqrt(size_per_head))
// for q (since fmha will divide sqrt(size_per_head), we multiple sqrt(size_per_head) first)
template<typename T>
void invokeNormalizeForFMHA(
    T* data, const T* logit_scales, int batch, int seqlen, int num_head, int size_per_head, cudaStream_t stream);

// input should be the qkv for trt fmha kernels with shape of [batch, seqlen, num_head, 3, size_per_head]
// do normlization on size_per_head for q & k [-, -, -, 0, *] && [-, -, -, 1, *] && *(logit_scale/sqrt(size_per_head))
// for q (since fmha will divide sqrt(size_per_head), we multiple sqrt(size_per_head) first)
template<typename T>
void invokeNormalizeForFMHA(int8_t*      data,
                            const T*     logit_scales,
                            int          batch,
                            int          seqlen,
                            int          num_head,
                            int          size_per_head,
                            cudaStream_t stream,
                            const float  query_deQ_scale,
                            const float  key_deQ_scale,
                            const float  query_Q_scale,
                            const float  key_Q_scale);

// input should be [batch, seqlen, num_head, size_per_head]
// do normlization on size_per_head && *(logit_scale/sqrt(size_per_head))
template<typename T>
void invokeNormalize(
    T* data, const T* logit_scales, int batch, int seqlen, int num_head, int size_per_head, cudaStream_t stream);

// input should be [batch, seqlen, num_head, size_per_head]
// do normlization on size_per_head && *(logit_scale)
template<typename T>
void invokeNormalize(int8_t*      data,
                     const T*     logit_scales,
                     int          batch,
                     int          seqlen,
                     int          num_head,
                     int          size_per_head,
                     cudaStream_t stream,
                     const float  deQ_scale,
                     const float  Q_scale);

}  // namespace fastertransformer
