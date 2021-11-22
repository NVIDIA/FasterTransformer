/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/kernels/calibrate_quantize_weight_kernels.h"
#include "src/fastertransformer/th_op/th_traits.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "torch/csrc/cuda/Stream.h"
#include <cuda_fp16.h>
#include <torch/custom_class.h>
#include <torch/script.h>

namespace torch_ext {
using torch::Tensor;

std::vector<Tensor> weight_transpose_calibrate_quantize(Tensor weight);

}  // namespace torch_ext
