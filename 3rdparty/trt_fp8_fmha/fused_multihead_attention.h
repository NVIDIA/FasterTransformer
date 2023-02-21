/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
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
#include "../trt_fused_multihead_attention/fused_multihead_attention.h"
#include "../../src/fastertransformer/utils/cuda_utils.h"
#include "../trt_fused_multihead_attention/fused_multihead_attention_common.h"
#include <assert.h>
#include <stdint.h>

namespace fastertransformer{
struct Fused_multihead_attention_params_base {
    // The QKV matrices.
    void *qkv_ptr;
    // The O matrix (output).
    void *o_ptr;

    // The stride between rows of the Q, K and V matrices.
    int64_t qkv_stride_in_bytes;
    // The stride between rows of O.
    int64_t o_stride_in_bytes;

#if defined(STORE_P)
    // The pointer to the P matrix (for debugging).
    void *p_ptr;
    // The stride between rows of the P matrix (for debugging).
    int64_t p_stride_in_bytes;
#endif  // defined(STORE_P)

#if defined(STORE_S)
    // The pointer to the S matrix (for debugging).
    void *s_ptr;
    // The stride between rows of the S matrix (for debugging).
    int64_t s_stride_in_bytes;
#endif  // defined(STORE_S)


#if defined(DEBUG_HAS_PRINT_BUFFER)
    void *print_ptr;
#endif

    // The dimensions.
    int b, h, s, d;
    // The scaling factors for the kernel.
    uint32_t scale_bmm1, scale_softmax, scale_bmm2;

    // Do we use Niall's trick to avoid I2F/F2I in the INT8 kernel.
    // See https://confluence.nvidia.com/pages/viewpage.action?pageId=302779721 for details.
    bool enable_i2f_trick;

    // true: for int8, instead of doing max reduce, use max value encoded in scale factor
    bool use_int8_scale_max = false;

    // The number of heads computed by one iteration of the wave.
    int heads_per_wave;
    // Buffers to perform a global sync and a critical section.
    int *counters, *max_barriers, *sum_barriers, *locks;
    // Scratch buffers to finalize softmax.
    float *max_scratch_ptr, *sum_scratch_ptr;
    // Scratch buffer to finalize the output (not needed for FP16).
    int *o_scratch_ptr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Fused_multihead_attention_params_v1 : Fused_multihead_attention_params_base {
    // The mask to implement drop-out.
    void *packed_mask_ptr;

    // The stride between matrices of packed mask.
    int64_t packed_mask_stride_in_bytes;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Fused_multihead_attention_params_FP8_v2 : Fused_multihead_attention_params_base {
    // array of length b+1 holding prefix sum of actual sequence lenghts.
    int *cu_seqlens;
    // use C/32 Format.
    bool interleaved = false;

    // flags to control small batch kernel choice
    // true: never unroll
    bool ignore_b1opt = false;
    // true: always unroll
    bool force_unroll = false;
    // by default TMA is not used.
    bool use_tma = false;
    // when we use TMA we need to know the actual seqlens to set the TMA desc. 
    uint32_t *seqlens;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
extern unsigned char cubin_fmha_v2_e4m3_128_64_ldgsts_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_e4m3_192_64_ldgsts_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_e4m3_256_64_ldgsts_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_e4m3_384_64_ldgsts_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_e4m3_512_64_ldgsts_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_e4m3_fp32_128_64_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_e4m3_fp32_192_64_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_e4m3_fp32_256_64_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_e4m3_fp32_384_64_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_e4m3_fp32_512_64_sm89_cu_cubin[];

extern unsigned int cubin_fmha_v2_e4m3_128_64_ldgsts_sm90_cu_cubin_len;
extern unsigned int cubin_fmha_v2_e4m3_192_64_ldgsts_sm90_cu_cubin_len;
extern unsigned int cubin_fmha_v2_e4m3_256_64_ldgsts_sm90_cu_cubin_len;
extern unsigned int cubin_fmha_v2_e4m3_384_64_ldgsts_sm90_cu_cubin_len;
extern unsigned int cubin_fmha_v2_e4m3_512_64_ldgsts_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_e4m3_fp32_128_64_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_e4m3_fp32_192_64_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_e4m3_fp32_256_64_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_e4m3_fp32_384_64_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_e4m3_fp32_512_64_sm89_cu_cubin_len;

static const struct FusedMultiHeadAttentionKernelMetaInfoFP8V2
{
    Data_type mDataType;
    unsigned int mS;
    unsigned int mD;
    unsigned int mSM;
    const unsigned char* mCubin;
    unsigned int mCubinSize;
    const char* mFuncName;
    unsigned int mSharedMemBytes;
    unsigned int mThreadsPerCTA;
    unsigned int mUnrollStep;
    bool mInterleaved;
} sMhaKernelMetaInfosFP8V2[] = {
#if CUDA_VERSION >= 11080
    // Ada
    {DATA_TYPE_E4M3, 128, 64, kSM_89, cubin_fmha_v2_e4m3_fp32_128_64_sm89_cu_cubin,
     cubin_fmha_v2_e4m3_fp32_128_64_sm89_cu_cubin_len,
     "fmha_v2_e4m3_fp32_128_64_sm89_kernel", 32768, 128, 0, false},
    {DATA_TYPE_E4M3, 128, 64, kSM_89, cubin_fmha_v2_e4m3_fp32_128_64_sm89_cu_cubin,
     cubin_fmha_v2_e4m3_fp32_128_64_sm89_cu_cubin_len,
     "fmha_v2_e4m3_fp32_128_64_sm89_kernel_nl", 20480, 128, 16, false},
    {DATA_TYPE_E4M3, 192, 64, kSM_89, cubin_fmha_v2_e4m3_fp32_192_64_sm89_cu_cubin,
     cubin_fmha_v2_e4m3_fp32_192_64_sm89_cu_cubin_len,
     "fmha_v2_e4m3_fp32_192_64_sm89_kernel", 36864, 128, 0, false},
    {DATA_TYPE_E4M3, 192, 64, kSM_89, cubin_fmha_v2_e4m3_fp32_192_64_sm89_cu_cubin,
     cubin_fmha_v2_e4m3_fp32_192_64_sm89_cu_cubin_len,
     "fmha_v2_e4m3_fp32_192_64_sm89_kernel_nl", 36864, 128, 32, false},
    {DATA_TYPE_E4M3, 256, 64, kSM_89, cubin_fmha_v2_e4m3_fp32_256_64_sm89_cu_cubin,
     cubin_fmha_v2_e4m3_fp32_256_64_sm89_cu_cubin_len,
     "fmha_v2_e4m3_fp32_256_64_sm89_kernel", 36864, 128, 0, false},
    {DATA_TYPE_E4M3, 256, 64, kSM_89, cubin_fmha_v2_e4m3_fp32_256_64_sm89_cu_cubin,
     cubin_fmha_v2_e4m3_fp32_256_64_sm89_cu_cubin_len,
     "fmha_v2_e4m3_fp32_256_64_sm89_kernel_nl", 36864, 128, 32, false},
    {DATA_TYPE_E4M3, 384, 64, kSM_89, cubin_fmha_v2_e4m3_fp32_384_64_sm89_cu_cubin,
     cubin_fmha_v2_e4m3_fp32_384_64_sm89_cu_cubin_len,
     "fmha_v2_e4m3_fp32_384_64_sm89_kernel", 53248, 128, 0, false},
    {DATA_TYPE_E4M3, 384, 64, kSM_89, cubin_fmha_v2_e4m3_fp32_384_64_sm89_cu_cubin,
     cubin_fmha_v2_e4m3_fp32_384_64_sm89_cu_cubin_len,
     "fmha_v2_e4m3_fp32_384_64_sm89_kernel_nl", 53248, 128, 32, false},
    {DATA_TYPE_E4M3, 512, 64, kSM_89, cubin_fmha_v2_e4m3_fp32_512_64_sm89_cu_cubin,
     cubin_fmha_v2_e4m3_fp32_512_64_sm89_cu_cubin_len,
     "fmha_v2_e4m3_fp32_512_64_sm89_kernel", 73728, 256, 0, false},
    {DATA_TYPE_E4M3, 512, 64, kSM_89, cubin_fmha_v2_e4m3_fp32_512_64_sm89_cu_cubin,
     cubin_fmha_v2_e4m3_fp32_512_64_sm89_cu_cubin_len,
     "fmha_v2_e4m3_fp32_512_64_sm89_kernel_nl", 73728, 256, 32, false},
    // Hopper
    {DATA_TYPE_E4M3, 128, 64, kSM_90, cubin_fmha_v2_e4m3_128_64_ldgsts_sm90_cu_cubin,
     cubin_fmha_v2_e4m3_128_64_ldgsts_sm90_cu_cubin_len,
     "fmha_v2_e4m3_128_64_ldgsts_sm90_kernel_nl", 25600, 128, 64, false},
    {DATA_TYPE_E4M3, 192, 64, kSM_90, cubin_fmha_v2_e4m3_192_64_ldgsts_sm90_cu_cubin,
     cubin_fmha_v2_e4m3_192_64_ldgsts_sm90_cu_cubin_len,
     "fmha_v2_e4m3_192_64_ldgsts_sm90_kernel_nl", 33792, 128, 64, false},
    {DATA_TYPE_E4M3, 256, 64, kSM_90, cubin_fmha_v2_e4m3_256_64_ldgsts_sm90_cu_cubin,
     cubin_fmha_v2_e4m3_256_64_ldgsts_sm90_cu_cubin_len,
     "fmha_v2_e4m3_256_64_ldgsts_sm90_kernel_nl", 41984, 128, 64, false},
    {DATA_TYPE_E4M3, 384, 64, kSM_90, cubin_fmha_v2_e4m3_384_64_ldgsts_sm90_cu_cubin,
     cubin_fmha_v2_e4m3_384_64_ldgsts_sm90_cu_cubin_len,
     "fmha_v2_e4m3_384_64_ldgsts_sm90_kernel_nl", 58368, 128, 64, false},
    {DATA_TYPE_E4M3, 512, 64, kSM_90, cubin_fmha_v2_e4m3_512_64_ldgsts_sm90_cu_cubin,
     cubin_fmha_v2_e4m3_512_64_ldgsts_sm90_cu_cubin_len,
     "fmha_v2_e4m3_512_64_ldgsts_sm90_kernel_nl", 108032, 256, 64, false},
    {DATA_TYPE_E4M3, 128, 64, kSM_90, cubin_fmha_v2_e4m3_128_64_ldgsts_sm90_cu_cubin,
     cubin_fmha_v2_e4m3_128_64_ldgsts_sm90_cu_cubin_len,
     "fmha_v2_e4m3_128_64_ldgsts_sm90_kernel", 25600, 128, 0, false},
    {DATA_TYPE_E4M3, 192, 64, kSM_90, cubin_fmha_v2_e4m3_192_64_ldgsts_sm90_cu_cubin,
     cubin_fmha_v2_e4m3_192_64_ldgsts_sm90_cu_cubin_len,
     "fmha_v2_e4m3_192_64_ldgsts_sm90_kernel", 33792, 128, 0, false},
    {DATA_TYPE_E4M3, 256, 64, kSM_90, cubin_fmha_v2_e4m3_256_64_ldgsts_sm90_cu_cubin,
     cubin_fmha_v2_e4m3_256_64_ldgsts_sm90_cu_cubin_len,
     "fmha_v2_e4m3_256_64_ldgsts_sm90_kernel", 41984, 128, 0, false},
    {DATA_TYPE_E4M3, 384, 64, kSM_90, cubin_fmha_v2_e4m3_384_64_ldgsts_sm90_cu_cubin,
     cubin_fmha_v2_e4m3_384_64_ldgsts_sm90_cu_cubin_len,
     "fmha_v2_e4m3_384_64_ldgsts_sm90_kernel", 58368, 128, 0, false},
    {DATA_TYPE_E4M3, 512, 64, kSM_90, cubin_fmha_v2_e4m3_512_64_ldgsts_sm90_cu_cubin,
     cubin_fmha_v2_e4m3_512_64_ldgsts_sm90_cu_cubin_len,
     "fmha_v2_e4m3_512_64_ldgsts_sm90_kernel", 108032, 256, 0, false},
#endif
};

class FusedMultiHeadAttentionXMMAKernelFP8V2
    : public TFusedMultiHeadAttentionXMMAKernel<FusedMultiHeadAttentionKernelMetaInfoFP8V2,
          Fused_multihead_attention_params_FP8_v2>
{
public:
    FusedMultiHeadAttentionXMMAKernelFP8V2(const FusedMultiHeadAttentionKernelMetaInfoFP8V2* pMetaStart,
        unsigned int nMetaCount, Data_type type, unsigned int sm)
        : TFusedMultiHeadAttentionXMMAKernel<FusedMultiHeadAttentionKernelMetaInfoFP8V2,
              Fused_multihead_attention_params_FP8_v2>(pMetaStart, nMetaCount, type, sm)
    {
    }

    inline uint64_t hashID(unsigned int s, unsigned int d, bool interleaved, bool unroll) const
    {
        return (uint64_t) s << 32 | d |(interleaved ? 2ull : 0ull) | (unroll ? 1ull : 0ull);
    }

    virtual uint64_t hashID(const KernelMeta& kernelMeta) const
    {
        assert(kernelMeta.mD == 64 || kernelMeta.mD == 32);
        return hashID(kernelMeta.mS, kernelMeta.mD, kernelMeta.mInterleaved, kernelMeta.mUnrollStep);
    }

    virtual void run(Fused_multihead_attention_params_FP8_v2& params, cudaStream_t ss) const
    {
        assert(params.d == 64);
        assert(!params.interleaved && !params.use_tma);

        bool forceUnroll = true;

        if (!1 || (!params.force_unroll && (params.ignore_b1opt || params.b > 1)))
        {
            forceUnroll = false;
        }

        const auto findIter = mFunctions.find(hashID(params.s, params.d, params.interleaved, forceUnroll));
        assert(findIter != mFunctions.end());

        const auto& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
        const CUfunction func = findIter->second.mDeviceFunction;

        void* kernelParams[] = {(void*)&params};
        if (!forceUnroll)
        {
            cuErrCheck(mDriver.cuLaunchKernel(func, params.h, params.b, 1, kernelMeta.mThreadsPerCTA, 1, 1,
                           kernelMeta.mSharedMemBytes, ss, kernelParams, nullptr),
                mDriver);
        }
        else
        {
            int unroll = kernelMeta.mS / kernelMeta.mUnrollStep;
            assert(kernelMeta.mS == kernelMeta.mUnrollStep * unroll);
            cuErrCheck(mDriver.cuLaunchKernel(func, params.h, params.b, unroll, kernelMeta.mThreadsPerCTA, 1, 1,
                           kernelMeta.mSharedMemBytes, ss, kernelParams, nullptr),
                mDriver);
        }
    }
};

using FusedMHAKernelFactoryFP8V2 = TFusedMHAKernelFactory<FusedMultiHeadAttentionXMMAKernelFP8V2>;

inline const FusedMultiHeadAttentionXMMAKernelFP8V2* getXMMAKernelsFP8V2(Data_type type, unsigned int sm)
{
    return FusedMHAKernelFactoryFP8V2::Get().getXMMAKernels(
        sMhaKernelMetaInfosFP8V2, sizeof(sMhaKernelMetaInfosFP8V2) / sizeof(sMhaKernelMetaInfosFP8V2[0]), type, sm);
}

} // namespace fastertransformer
