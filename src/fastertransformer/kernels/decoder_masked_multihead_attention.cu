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

#include "src/fastertransformer/kernels/decoder_masked_multihead_attention.h"
#include "src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp"
#include "src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include <assert.h>
#include <float.h>
#include <type_traits>

template<typename T, typename KERNEL_PARAMS_TYPE>
void multihead_attention_(const KERNEL_PARAMS_TYPE& params, const cudaStream_t& stream)
{
    switch (params.hidden_size_per_head) {
        case 32:
            mmha_launch_kernel<T, 32, 32, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 48:
            mmha_launch_kernel<T, 48, 64, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 64:
            mmha_launch_kernel<T, 64, 64, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 80:
            mmha_launch_kernel<T, 80, 128, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 96:
            mmha_launch_kernel<T, 96, 128, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 112:
            mmha_launch_kernel<T, 112, 128, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 128:
            mmha_launch_kernel<T, 128, 128, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 144:
            mmha_launch_kernel<T, 144, 256, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 160:
            mmha_launch_kernel<T, 160, 256, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 192:
            mmha_launch_kernel<T, 192, 256, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 224:
            mmha_launch_kernel<T, 224, 256, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 256:
            mmha_launch_kernel<T, 256, 256, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        default:
            assert(false);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void masked_multihead_attention(const Masked_multihead_attention_params<float>& params, const cudaStream_t& stream)
{
    multihead_attention_<float, Masked_multihead_attention_params<float>>(params, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void masked_multihead_attention(const Masked_multihead_attention_params<uint16_t>& params, const cudaStream_t& stream)
{
    multihead_attention_<uint16_t, Masked_multihead_attention_params<uint16_t>>(params, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_BF16
void masked_multihead_attention(const Masked_multihead_attention_params<__nv_bfloat16>& params,
                                const cudaStream_t&                                     stream)
{
    multihead_attention_<__nv_bfloat16, Masked_multihead_attention_params<__nv_bfloat16>>(params, stream);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_FP8
void masked_multihead_attention(const Masked_multihead_attention_params<__nv_fp8_e4m3>& params,
                                const cudaStream_t&                                     stream)
{
    multihead_attention_<__nv_fp8_e4m3, Masked_multihead_attention_params<__nv_fp8_e4m3>>(params, stream);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

void cross_multihead_attention(const Cross_multihead_attention_params<float>& params, const cudaStream_t& stream)
{
    multihead_attention_<float, Cross_multihead_attention_params<float>>(params, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void cross_multihead_attention(const Cross_multihead_attention_params<uint16_t>& params, const cudaStream_t& stream)
{
    multihead_attention_<uint16_t, Cross_multihead_attention_params<uint16_t>>(params, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_BF16
void cross_multihead_attention(const Cross_multihead_attention_params<__nv_bfloat16>& params,
                               const cudaStream_t&                                    stream)
{
    multihead_attention_<__nv_bfloat16, Cross_multihead_attention_params<__nv_bfloat16>>(params, stream);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_FP8
void cross_multihead_attention(const Cross_multihead_attention_params<__nv_fp8_e4m3>& params,
                               const cudaStream_t&                                    stream)
{
    multihead_attention_<__nv_fp8_e4m3, Cross_multihead_attention_params<__nv_fp8_e4m3>>(params, stream);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
