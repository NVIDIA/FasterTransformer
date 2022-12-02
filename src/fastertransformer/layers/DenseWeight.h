/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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
#include "stdlib.h"
namespace fastertransformer {

// Note that the int8 mode of BERT and GPT are different.
// For int8 mode = 2 on GPT:
// scale (gemm input scale): quantize input of GEMM (float/half) in the int8 range. Namely, int8_x = scale * x
// scale_inter: (gemm output scale) / (gemm input scale * gemm weight scale)
// scale_out: 1 / (gemm output scale), dequantize activation from int8 range to float/half.

template<typename T>
struct DenseWeight {
    const T* kernel    = nullptr;
    const T* bias      = nullptr;
    const T* sp_kernel = nullptr;
    // for int8 kernel
    const int8_t* int8_kernel             = nullptr;
    const float*  scale                   = nullptr;
    const T*      weight_only_quant_scale = nullptr;
    const T*      moe_scale               = nullptr;
    const float*  scale_inter             = nullptr;
    const float*  scale_out               = nullptr;
};

}  // namespace fastertransformer