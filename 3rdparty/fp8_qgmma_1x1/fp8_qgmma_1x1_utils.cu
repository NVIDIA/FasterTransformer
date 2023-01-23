/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

#define gpuErrChk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#include <vector>

#include "conv1x1.cuh"
#include "fp8_qgmma_1x1_utils.h"


namespace fastertransformer
{
    void invokeSwizzleQgmmaWeights(int C, int K, uint8_t *h_B, uint8_t *d_B, uint16_t *h_bias, uint16_t *d_bias)
    {
        swizzleQgmmaWeights(C, K, h_B, d_B, h_bias, d_bias);
    }

    template <bool RELU, bool GELU>
    void qgmma1x1Launcher::invokeQgmma1x1(__nv_fp8_e4m3 *res,
                                          int m,
                                          int n,
                                          int k,
                                          const __nv_fp8_e4m3 *input,
                                          const __nv_fp8_e4m3 *kernel,
                                          const __nv_bfloat16 *bias,
                                          const float input_scale,
                                          const float kernel_scale,
                                          const float output_scale,
                                          void *workspace,
                                          cudaStream_t stream)
    {
        qgmmaDimsKey key(RELU, GELU);
        if (launcher_map_.count(key) == 0)
        {
            launcher_map_[key] = std::make_unique<
                Conv1x1<RELU,
                        GELU>>(reinterpret_cast<uint8_t*>(workspace), stream);
        }
    
        launcher_map_[key]->run(reinterpret_cast<uint8_t *>(res),
                                reinterpret_cast<uint8_t *>(const_cast<__nv_fp8_e4m3 *>(input)),
                                reinterpret_cast<uint8_t *>(const_cast<__nv_fp8_e4m3 *>(kernel)),
                                reinterpret_cast<uint16_t *>(const_cast<__nv_bfloat16 *>(bias)),
                                input_scale * kernel_scale,
                                output_scale,
                                m,
                                1,
                                1,
                                k,
                                n);
    }

    template <bool RELU, bool GELU>
    void qgmma1x1Launcher::getWorkSpaceSize(int n,
                                            size_t &workspace_size)
    {
        using Conv1x1Launcher = Conv1x1<RELU,
                                        GELU>;
        workspace_size = (size_t)(Conv1x1Launcher::getWorkSpaceSize(n));
    }

    template void qgmma1x1Launcher::invokeQgmma1x1<true, false>(__nv_fp8_e4m3 *res,
                                                                int m,
                                                                int n,
                                                                int k,
                                                                const __nv_fp8_e4m3 *input,
                                                                const __nv_fp8_e4m3 *kernel,
                                                                const __nv_bfloat16 *bias,
                                                                const float input_scale,
                                                                const float kernel_scale,
                                                                const float output_scale,
                                                                void *workspace,
                                                                cudaStream_t stream);

    template void qgmma1x1Launcher::invokeQgmma1x1<true, true>(__nv_fp8_e4m3 *res,
                                                               int m,
                                                               int n,
                                                               int k,
                                                               const __nv_fp8_e4m3 *input,
                                                               const __nv_fp8_e4m3 *kernel,
                                                               const __nv_bfloat16 *bias,
                                                               const float input_scale,
                                                               const float kernel_scale,
                                                               const float output_scale,
                                                               void *workspace,
                                                               cudaStream_t stream);

    template void qgmma1x1Launcher::invokeQgmma1x1<false, false>(__nv_fp8_e4m3 *res,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const __nv_fp8_e4m3 *input,
                                                                 const __nv_fp8_e4m3 *kernel,
                                                                 const __nv_bfloat16 *bias,
                                                                 const float input_scale,
                                                                 const float kernel_scale,
                                                                 const float output_scale,
                                                                 void *workspace,
                                                                 cudaStream_t stream);

    template void qgmma1x1Launcher::invokeQgmma1x1<false, true>(__nv_fp8_e4m3 *res,
                                                                int m,
                                                                int n,
                                                                int k,
                                                                const __nv_fp8_e4m3 *input,
                                                                const __nv_fp8_e4m3 *kernel,
                                                                const __nv_bfloat16 *bias,
                                                                const float input_scale,
                                                                const float kernel_scale,
                                                                const float output_scale,
                                                                void *workspace,
                                                                cudaStream_t stream);

    template void qgmma1x1Launcher::getWorkSpaceSize<false, false>(int n,
                                                                   size_t &workspace_size);

    template void qgmma1x1Launcher::getWorkSpaceSize<false, true>(int n,
                                                                  size_t &workspace_size);

    template void qgmma1x1Launcher::getWorkSpaceSize<true, false>(int n,
                                                                  size_t &workspace_size);

    template void qgmma1x1Launcher::getWorkSpaceSize<true, true>(int n,
                                                                 size_t &workspace_size);
} // namespace fastertransformer