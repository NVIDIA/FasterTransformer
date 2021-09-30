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

#include <cuda_fp16.h>
#include <iostream>
#include <nvToolsExt.h>
#include <vector>

#include "torch/csrc/cuda/Stream.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/custom_class.h>
#include <torch/script.h>

#include "src/fastertransformer/kernels/decoding_kernels.h"
#include "src/fastertransformer/th_op/th_traits.h"
#include "src/fastertransformer/th_op/th_utils.h"

namespace th = torch;

namespace torch_ext {

th::Tensor gather_tree(th::Tensor step_ids, th::Tensor parent_ids, th::Tensor max_sequence_lengths, int64_t end_token);

}  // namespace torch_ext