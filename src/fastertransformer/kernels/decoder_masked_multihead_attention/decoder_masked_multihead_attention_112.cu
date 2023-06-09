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

#include "decoder_masked_multihead_attention_template.hpp"
#include "src/fastertransformer/kernels/decoder_masked_multihead_attention.h"
#include "src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include <assert.h>
#include <float.h>
#include <type_traits>

////////////////////////////////////////////////////////////////////////////////////////////////////

#define MMHA_LAUNCH_KERNEL(                                                                                            \
    T, Dh, Dh_MAX, THDS_PER_KEY, THDS_PER_VALUE, THDS_PER_BLOCK, DO_CROSS_ATTENTION, HAS_BEAMS, stream)                \
    size_t smem_sz = mmha::smem_size_in_bytes<T, DO_CROSS_ATTENTION>(params, THDS_PER_VALUE, THDS_PER_BLOCK);          \
    dim3   grid(params.num_heads, params.batch_size);                                                                  \
    mmha::masked_multihead_attention_kernel<T,                                                                         \
                                            Dh,                                                                        \
                                            Dh_MAX,                                                                    \
                                            THDS_PER_KEY,                                                              \
                                            THDS_PER_VALUE,                                                            \
                                            THDS_PER_BLOCK,                                                            \
                                            DO_CROSS_ATTENTION,                                                        \
                                            HAS_BEAMS><<<grid, THDS_PER_BLOCK, smem_sz, stream>>>(params)

////////////////////////////////////////////////////////////////////////////////////////////////////

// !!! Specialize the launcher for Cross attention
template<typename T, int Dh, int Dh_MAX, typename KERNEL_PARAMS_TYPE>
void mmha_launch_kernel(const KERNEL_PARAMS_TYPE& params, const cudaStream_t& stream)
{
    constexpr int  THREADS_PER_VALUE  = threads_per_value_t<T, Dh_MAX>::value;
    constexpr bool DO_CROSS_ATTENTION = std::is_same<KERNEL_PARAMS_TYPE, Cross_multihead_attention_params<T>>::value;
    int            tlength            = (DO_CROSS_ATTENTION) ? params.memory_max_len : params.timestep;
    if (params.cache_indir == nullptr) {
        if (tlength < 32) {
            MMHA_LAUNCH_KERNEL(T, Dh, Dh_MAX, 4, THREADS_PER_VALUE, 64, DO_CROSS_ATTENTION, false, stream);
        }
        else if (tlength < 2048) {
            MMHA_LAUNCH_KERNEL(T, Dh, Dh_MAX, 2, THREADS_PER_VALUE, 128, DO_CROSS_ATTENTION, false, stream);
        }
        else {
            MMHA_LAUNCH_KERNEL(T, Dh, Dh_MAX, 1, THREADS_PER_VALUE, 256, DO_CROSS_ATTENTION, false, stream);
        }
    }
    else {
        if (tlength < 32) {
            MMHA_LAUNCH_KERNEL(T, Dh, Dh_MAX, 4, THREADS_PER_VALUE, 64, DO_CROSS_ATTENTION, true, stream);
        }
        else if (tlength < 2048) {
            MMHA_LAUNCH_KERNEL(T, Dh, Dh_MAX, 2, THREADS_PER_VALUE, 128, DO_CROSS_ATTENTION, true, stream);
        }
        else {
            MMHA_LAUNCH_KERNEL(T, Dh, Dh_MAX, 1, THREADS_PER_VALUE, 256, DO_CROSS_ATTENTION, true, stream);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template void mmha_launch_kernel<float, 112, 128, Masked_multihead_attention_params<float>>(
    const Masked_multihead_attention_params<float>& params, const cudaStream_t& stream);
template void mmha_launch_kernel<uint16_t, 112, 128, Masked_multihead_attention_params<uint16_t>>(
    const Masked_multihead_attention_params<uint16_t>& params, const cudaStream_t& stream);
#ifdef ENABLE_BF16
template void mmha_launch_kernel<__nv_bfloat16, 112, 128, Masked_multihead_attention_params<__nv_bfloat16>>(
    const Masked_multihead_attention_params<__nv_bfloat16>& params, const cudaStream_t& stream);
#endif
#ifdef ENABLE_FP8
template void mmha_launch_kernel<__nv_fp8_e4m3, 112, 128, Masked_multihead_attention_params<__nv_fp8_e4m3>>(
    const Masked_multihead_attention_params<__nv_fp8_e4m3>& params, const cudaStream_t& stream);
#endif

template void mmha_launch_kernel<float, 112, 128, Cross_multihead_attention_params<float>>(
    const Cross_multihead_attention_params<float>& params, const cudaStream_t& stream);
template void mmha_launch_kernel<uint16_t, 112, 128, Cross_multihead_attention_params<uint16_t>>(
    const Cross_multihead_attention_params<uint16_t>& params, const cudaStream_t& stream);
#ifdef ENABLE_BF16
template void mmha_launch_kernel<__nv_bfloat16, 112, 128, Cross_multihead_attention_params<__nv_bfloat16>>(
    const Cross_multihead_attention_params<__nv_bfloat16>& params, const cudaStream_t& stream);
#endif
#ifdef ENABLE_FP8
template void mmha_launch_kernel<__nv_fp8_e4m3, 112, 128, Cross_multihead_attention_params<__nv_fp8_e4m3>>(
    const Cross_multihead_attention_params<__nv_fp8_e4m3>& params, const cudaStream_t& stream);
#endif

#undef MMHA_LAUNCH_KERNEL
