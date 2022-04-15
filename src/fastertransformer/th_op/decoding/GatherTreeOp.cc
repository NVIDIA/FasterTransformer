/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/th_op/decoding/GatherTreeOp.h"

namespace th = torch;

namespace torch_ext {
th::Tensor
gather_tree(th::Tensor step_ids, th::Tensor parent_ids, th::Tensor max_sequence_lengths, th::Tensor end_tokens)
{
    CHECK_TH_CUDA(step_ids);
    CHECK_CONTIGUOUS(step_ids);
    TORCH_CHECK(step_ids.dtype() == th::kInt32, "step_ids dtype should be int32");
    CHECK_TH_CUDA(parent_ids);
    CHECK_CONTIGUOUS(parent_ids);
    TORCH_CHECK(parent_ids.dtype() == th::kInt32, "parent_ids dtype should be int32");
    CHECK_TH_CUDA(max_sequence_lengths);
    CHECK_CONTIGUOUS(max_sequence_lengths);
    TORCH_CHECK(max_sequence_lengths.dtype() == th::kInt32, "max_sequence_lengths dtype should be int32");
    int max_step = step_ids.size(0);
    int batch_size = step_ids.size(1);
    int beam_width = step_ids.size(2);
    auto beams = th::empty_like(step_ids);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    fastertransformer::invokeGatherTree(torch_ext::get_ptr<int>(beams),
                                        torch_ext::get_ptr<int>(max_sequence_lengths),
                                        max_step,
                                        batch_size,
                                        beam_width,
                                        torch_ext::get_ptr<int>(step_ids),
                                        torch_ext::get_ptr<int>(parent_ids),
                                        torch_ext::get_ptr<int>(end_tokens),
                                        stream);

    return beams;
}

}  // namespace torch_ext

static auto gather_tree = th::RegisterOperators("fastertransformer::gather_tree", &torch_ext::gather_tree);