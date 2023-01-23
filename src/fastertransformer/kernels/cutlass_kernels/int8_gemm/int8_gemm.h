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

#include "src/fastertransformer/cutlass_extensions/include/cutlass_extensions/epilogue/epilogue_quant_helper.h"
#include "src/fastertransformer/cutlass_extensions/include/cutlass_extensions/ft_gemm_configs.h"
#include "src/fastertransformer/utils/activation_types.h"
#include "src/fastertransformer/utils/allocator.h"
#include <cuda_runtime_api.h>

using cutlass::epilogue::QuantMode;
namespace fastertransformer {

/*
  This runner supports:
  int8_t inputs (A and B)
  float alpha scalings (either per-col, or per-col x per-row)
  T output (D) where T = {float, half, __nv_bfloat16} // TODO(mseznec)

  Activations, biases, scales and outputs are all assumed to be row-major.
  Weights are assumed to be column-major.
*/

template<typename T>
class CutlassInt8GemmRunner {
public:
    CutlassInt8GemmRunner();
    ~CutlassInt8GemmRunner();

    void gemm(const int8_t* A,
              const int8_t* B,
              QuantMode     quant_mode,
              const float*  alpha_col,
              const float*  alpha_row,
              T*            C,
              int           m,
              int           n,
              int           k,
              char*         workspace_ptr,
              const size_t  workspace_bytes,
              cudaStream_t  stream);

    // Returns desired workspace size in bytes.
    int getWorkspaceSize(const int m, const int n, const int k);

private:
    void dispatch_to_arch(const int8_t*     A,
                          const int8_t*     B,
                          QuantMode         quant_mode,
                          const float*      alpha_col,
                          const float*      alpha_row,
                          T*                C,
                          int               m,
                          int               n,
                          int               k,
                          CutlassGemmConfig gemm_config,
                          char*             workspace_ptr,
                          const size_t      workspace_bytes,
                          cudaStream_t      stream,
                          int*              occupancy = nullptr);

    void run_gemm(const int8_t* A,
                  const int8_t* B,
                  QuantMode     quant_mode,
                  const float*  alpha_col,
                  const float*  alpha_row,
                  T*            C,
                  int           m,
                  int           n,
                  int           k,
                  char*         workspace_ptr,
                  const size_t  workspace_bytes,
                  cudaStream_t  stream);

private:
    static constexpr int split_k_limit = 7;

    int sm_;
    int multi_processor_count_;
};

}  // namespace fastertransformer
