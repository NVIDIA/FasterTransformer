/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/th_op/multi_gpu_gpt/ParallelGptOp.h"
#include "src/fastertransformer/th_op/multi_gpu_gpt/WeightTransposeCalibrateQuantizeOp.h"

namespace th = torch;
namespace torch_ext {

ParallelGptOp::ParallelGptOp(const int64_t max_batch_size,
                             const int64_t max_seq_len,
                             const int64_t beam_width,
                             const int64_t head_num,
                             const int64_t size_per_head,
                             const int64_t inter_size,
                             const int64_t layer_num,
                             const int64_t vocab_size,
                             const int64_t start_id,
                             const int64_t end_id,
                             const double beam_search_diversity_rate,
                             const int64_t top_k,
                             const double top_p,
                             const unsigned long long random_seed,
                             const double temperature,
                             const double len_penalty,
                             const double repetition_penalty,
                             const int64_t tensor_para_size,
                             const int64_t pipeline_para_size,
                             const int64_t int8_mode,
                             const std::vector<th::Tensor> weights,
                             const std::vector<th::Tensor> int8_weights,
                             const std::vector<th::Tensor> scale):
    st_(weights[0].scalar_type())
{
    for (auto t : weights) {
        CHECK_INPUT(t, st_);
    }

    switch (st_) {
        case at::ScalarType::Float:
            ftgpt = new FTGpt<float>((size_t)max_batch_size,
                                     (size_t)max_seq_len,
                                     (size_t)beam_width,
                                     (size_t)head_num,
                                     (size_t)size_per_head,
                                     (size_t)inter_size,
                                     (size_t)layer_num,
                                     (size_t)vocab_size,
                                     start_id,
                                     end_id,
                                     beam_search_diversity_rate,
                                     top_k,
                                     top_p,
                                     random_seed,
                                     temperature,
                                     len_penalty,
                                     repetition_penalty,
                                     tensor_para_size,
                                     pipeline_para_size,
                                     int8_mode,
                                     weights,
                                     int8_weights,
                                     scale);
            break;
        case at::ScalarType::Half:
            ftgpt = new FTGpt<half>((size_t)max_batch_size,
                                    (size_t)max_seq_len,
                                    (size_t)beam_width,
                                    (size_t)head_num,
                                    (size_t)size_per_head,
                                    (size_t)inter_size,
                                    (size_t)layer_num,
                                    (size_t)vocab_size,
                                    start_id,
                                    end_id,
                                    beam_search_diversity_rate,
                                    top_k,
                                    top_p,
                                    random_seed,
                                    temperature,
                                    len_penalty,
                                    repetition_penalty,
                                    tensor_para_size,
                                    pipeline_para_size,
                                    int8_mode,
                                    weights,
                                    int8_weights,
                                    scale);
            break;
        default:
            throw std::runtime_error("Wrong Tensor type.");
    }
}

ParallelGptOp::~ParallelGptOp()
{
    delete ftgpt;
}

std::vector<th::Tensor> ParallelGptOp::forward(th::Tensor input_ids, th::Tensor input_lengths, const int64_t output_len)
{
    CHECK_TH_CUDA(input_ids);
    CHECK_CONTIGUOUS(input_ids);
    TORCH_CHECK(input_ids.dtype() == torch::kInt32, "input_ids dtype should be int32");
    CHECK_TH_CUDA(input_lengths);
    CHECK_CONTIGUOUS(input_lengths);
    TORCH_CHECK(input_lengths.dtype() == torch::kInt32, "input_lengths dtype should be int32");

    const int batchbeam = input_ids.size(0);
    const int max_input_length = input_ids.size(1);
    const int total_request_output_len = max_input_length + output_len;
    th::Tensor output_ids = torch::empty({total_request_output_len, batchbeam},
                                         torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    th::Tensor parent_ids = torch::empty({total_request_output_len, batchbeam},
                                         torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    th::Tensor sequence_lengths =
        torch::empty({batchbeam}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    ftgpt->forward(input_ids, input_lengths, output_ids, parent_ids, sequence_lengths, (size_t)output_len);
    return std::vector<th::Tensor>{output_ids, sequence_lengths};
}

}  // namespace torch_ext

static auto fasterTransformerGptTHS = torch::jit::class_<torch_ext::ParallelGptOp>("FasterTransformer", "ParallelGptOp")
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
                                                                double,
                                                                int64_t,
                                                                double,
                                                                int64_t,
                                                                double,
                                                                double,
                                                                double,
                                                                int64_t,
                                                                int64_t,
                                                                int64_t,
                                                                std::vector<th::Tensor>,
                                                                std::vector<th::Tensor>,
                                                                std::vector<th::Tensor>>())
                                          .def("forward", &torch_ext::ParallelGptOp::forward);

static auto weight_transpose_calibrate_quantize = torch::RegisterOperators(
    "fastertransformer::weight_transpose_calibrate_quantize", &torch_ext::weight_transpose_calibrate_quantize);
