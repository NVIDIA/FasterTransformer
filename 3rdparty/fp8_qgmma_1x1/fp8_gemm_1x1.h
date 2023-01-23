/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef FP8_GEMM_1X1_H
#define FP8_GEMM_1X1_H

#include "sharedCubinLoader.h"
#include "tile_profile.cuh"
#include "utils.h"

namespace fp8_gemm_1x1 {

enum DataType
{
    DATA_TYPE_BOOL,
    DATA_TYPE_E8M10,
    DATA_TYPE_E8M7,
    DATA_TYPE_FP16,
    DATA_TYPE_FP32,
    DATA_TYPE_INT4,
    DATA_TYPE_INT8,
    DATA_TYPE_INT32
};

} // namespace fp8_gemm_1x1

constexpr int32_t kSM_90 = 90;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Do not modify this, it is integrated from original fp8_gemm_1x1 codes.
////////////////////////////////////////////////////////////////////////////////////////////////////
struct SchedulerParams {
    int tiles_m;
    int tiles_n;
    int num_tiles;
    int* tile_counter;
    int* cta_completion_counter;
};

struct ComputeParams {
    uint8_t* D;
    int N;
    int NPQ;
    int PQ;
    int P;
    int Q;
    int C;
    int K;
    float ab_scale;
    float d_scale;
};

struct DMAParams {
    cudaTmaDesc* a_desc;
    cudaTmaDesc* b_desc;
    cudaTmaDesc* bias_desc;
    int C;
    int HW;
    int H;
    int W;

    // fast_divmod stuff
    uint32_t mul_hw, shr_hw;
    uint32_t mul_w, shr_w;
};

struct KernelParams {
    SchedulerParams scheduler_params;
    DMAParams dma_params;
    ComputeParams compute_params;
    ProfileParams profile_params;
    bool gelu;
    bool relu;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Do not modify this, it is integrated from original fp8_gemm_1x1 codes.
////////////////////////////////////////////////////////////////////////////////////////////////////
extern unsigned char cubin_fp8_gemm_1x1_gelu_cu_cubin[];
extern unsigned char cubin_fp8_gemm_1x1_no_act_cu_cubin[];
extern unsigned char cubin_fp8_gemm_1x1_relu_cu_cubin[];

extern uint32_t cubin_fp8_gemm_1x1_gelu_cu_cubin_len;
extern uint32_t cubin_fp8_gemm_1x1_no_act_cu_cubin_len;
extern uint32_t cubin_fp8_gemm_1x1_relu_cu_cubin_len;

static const struct Fp8Gemm1x1MetaInfo
{
    bool mGelu;
    bool mRelu;
    int32_t mSM;
    unsigned char const* mCubin;
    uint32_t mCubinSize;
    char const* mFuncName;
    int32_t mSharedMemBytes;
    int32_t mThreadsPerCTA;
} sFp8Gemm1x1KernelMetaInfos[] = {
    { true,  false, kSM_90, cubin_fp8_gemm_1x1_gelu_cu_cubin,   cubin_fp8_gemm_1x1_gelu_cu_cubin_len,   "kernel_GELU",   197920, 384 },
    { false, false, kSM_90, cubin_fp8_gemm_1x1_no_act_cu_cubin, cubin_fp8_gemm_1x1_no_act_cu_cubin_len, "kernel_no_act", 197920, 384 },
    { false, true,  kSM_90, cubin_fp8_gemm_1x1_relu_cu_cubin,   cubin_fp8_gemm_1x1_relu_cu_cubin_len,   "kernel_RELU",   197920, 384 },
};

////////////////////////////////////////////////////////////////////////////////////////////////////
class Fp8Gemm1x1Kernel
    : public TSharedCubinKernel<Fp8Gemm1x1MetaInfo, KernelParams>
{
public:
    Fp8Gemm1x1Kernel(Fp8Gemm1x1MetaInfo const* pMetaStart,
        int32_t nMetaCount, int32_t sm)
        : TSharedCubinKernel<Fp8Gemm1x1MetaInfo, KernelParams>(
            pMetaStart, nMetaCount, sm)
    {
    }

    uint64_t hashID(bool gelu, bool relu) const
    {
        return (gelu ? 2U : 0U) | (relu ? 1U : 0U);
    }

    uint64_t hashID(KernelParams const& param) const
    {
        return hashID(param.gelu, param.relu);
    }

    uint64_t hashID(KernelMeta const& kernelMeta) const
    {
        return hashID(kernelMeta.mGelu, kernelMeta.mRelu);
    }
};

using Fp8Gemm1x1KernelFactory = TSharedCubinKernelFactory<Fp8Gemm1x1Kernel>;

inline Fp8Gemm1x1Kernel const* getFp8Gemm1x1Kernels(int32_t sm)
{
    return Fp8Gemm1x1KernelFactory::Get().getCubinKernels(
        sFp8Gemm1x1KernelMetaInfos, sizeof(sFp8Gemm1x1KernelMetaInfos) / sizeof(sFp8Gemm1x1KernelMetaInfos[0]), sm);
}

#endif // FP8_GEMM_1X1_H
