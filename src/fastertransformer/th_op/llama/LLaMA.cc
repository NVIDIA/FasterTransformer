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

#include "src/fastertransformer/th_op/llama/LLaMA.h"

namespace th = torch;
namespace torch_ext {

LLaMA::LLaMA(const int64_t            num_heads,
             const int64_t            size_per_head,
             const int64_t            inter_size,
             const int64_t            num_layers,
             const int64_t            vocab_size,
             const int64_t            rotary_embedding_dim,
             const int64_t            random_seed,
             const int64_t            max_seq_len,
             const int64_t            tensor_para_size,
             const int64_t            pipeline_para_size,
             const vector<th::Tensor> weights):
    vocab_size_(vocab_size),
    st_(weights[0].scalar_type())
{
    for (auto t : weights) {
        CHECK_INPUT(t, st_);
    }

    switch (st_) {
        case at::ScalarType::Float:
            ftllama = new FTLLaMA<float>((size_t)num_heads,
                                         (size_t)size_per_head,
                                         (size_t)inter_size,
                                         (size_t)num_layers,
                                         (size_t)vocab_size,
                                         (size_t)rotary_embedding_dim,
                                         (size_t)random_seed,
                                         (size_t)max_seq_len,
                                         tensor_para_size,
                                         pipeline_para_size,
                                         weights);
            break;
        case at::ScalarType::Half:
            ftllama = new FTLLaMA<half>((size_t)num_heads,
                                        (size_t)size_per_head,
                                        (size_t)inter_size,
                                        (size_t)num_layers,
                                        (size_t)vocab_size,
                                        (size_t)rotary_embedding_dim,
                                        (size_t)random_seed,
                                        (size_t)max_seq_len,
                                        tensor_para_size,
                                        pipeline_para_size,
                                        weights);
            break;
        default:
            throw std::runtime_error("Wrong Tensor type.");
    }
}

LLaMA::~LLaMA()
{
    delete ftllama;
}

th::Tensor
LLaMA::forward(th::Tensor& input_ids, th::Tensor& input_lengths, const int64_t start_pos)
{
    CHECK_TH_CUDA(input_ids);
    CHECK_CONTIGUOUS(input_ids);
    TORCH_CHECK(input_ids.dtype() == torch::kInt32, "input_ids dtype should be int32");
    CHECK_TH_CUDA(input_lengths);
    CHECK_CONTIGUOUS(input_lengths);
    TORCH_CHECK(input_lengths.dtype() == torch::kInt32, "input_lengths dtype should be int32");

    const int  batch_size    = input_ids.size(0);
    const int  seq_len       = input_ids.size(1);
    th::Tensor output_logits = torch::empty({batch_size, seq_len, (long)vocab_size_},
                                            torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));
    ftllama->forward(output_logits, input_ids, input_lengths, start_pos);
    return output_logits;
}

}  // namespace torch_ext

static auto fasterTransformerGptTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::LLaMA>("FasterTransformerLLaMA")
#else
    torch::jit::class_<torch_ext::LLaMA>("FasterTransformer", "LLaMA")
#endif
        .def(torch::jit::init<int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              std::vector<th::Tensor>>())
        .def("forward", &torch_ext::LLaMA::forward);
