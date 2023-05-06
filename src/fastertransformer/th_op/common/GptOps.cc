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

#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace torch_ext {
namespace ft = fastertransformer;
using torch::Tensor;

// Results a tensor of {batch_to_compact_idx, compact_to_batch_idx}
std::vector<Tensor> find_context_duplications(Tensor input_ids)
{
    CHECK_INPUT(input_ids, torch::kInt32);
    TORCH_CHECK(input_ids.dim() == 2, "Invalid dim. Input ids must be a matrix [batch, seq_len]");

    const auto stream = at::cuda::getCurrentCUDAStream().stream();

    const int batch_size = input_ids.size(0);
    const int seq_len    = input_ids.size(1);

    Tensor shared_contexts =
        torch::empty({batch_size}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    Tensor batch_to_compact     = torch::empty_like(shared_contexts);
    Tensor compact_to_batch_tmp = torch::empty_like(shared_contexts);

    Tensor compact_size_tensor =
        torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    ft::invokeFindContextDups(get_ptr<int>(shared_contexts),
                              get_ptr<int>(batch_to_compact),
                              get_ptr<int>(compact_to_batch_tmp),
                              get_ptr<int>(compact_size_tensor),
                              get_ptr<const int>(input_ids),
                              batch_size,
                              1,
                              seq_len,
                              stream);

    Tensor    compact_size_cpu_tensor = compact_size_tensor.to(torch::kCPU);
    const int compact_size            = compact_size_cpu_tensor.item<int>();

    Tensor compact_to_batch =
        torch::empty({compact_size}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    ft::cudaD2Dcpy(get_ptr<int>(compact_to_batch), get_ptr<const int>(compact_to_batch_tmp), compact_size);
    return {batch_to_compact, compact_to_batch};
}

}  // namespace torch_ext

// Utility methods that may be useful for preprocessing weights in torch.
static auto find_context_duplications =
    torch::RegisterOperators("fastertransformer::find_context_duplications", &torch_ext::find_context_duplications);
