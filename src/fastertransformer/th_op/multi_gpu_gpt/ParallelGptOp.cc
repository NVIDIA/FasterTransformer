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

#include "src/fastertransformer/th_op/multi_gpu_gpt/ParallelGptOp.h"
#include "src/fastertransformer/th_op/multi_gpu_gpt/WeightTransposeCalibrateQuantizeOp.h"

namespace th = torch;
namespace torch_ext {

ParallelGptOp::ParallelGptOp(const int64_t head_num,
                             const int64_t size_per_head,
                             const int64_t inter_size,
                             const int64_t layer_num,
                             const int64_t vocab_size,
                             const int64_t start_id,
                             const int64_t end_id,
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
            ftgpt = new FTGpt<float>((size_t)head_num,
                                     (size_t)size_per_head,
                                     (size_t)inter_size,
                                     (size_t)layer_num,
                                     (size_t)vocab_size,
                                     start_id,
                                     end_id,
                                     tensor_para_size,
                                     pipeline_para_size,
                                     int8_mode,
                                     weights,
                                     int8_weights,
                                     scale);
            break;
        case at::ScalarType::Half:
            ftgpt = new FTGpt<half>((size_t)head_num,
                                    (size_t)size_per_head,
                                    (size_t)inter_size,
                                    (size_t)layer_num,
                                    (size_t)vocab_size,
                                    start_id,
                                    end_id,
                                    tensor_para_size,
                                    pipeline_para_size,
                                    int8_mode,
                                    weights,
                                    int8_weights,
                                    scale);
            break;
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16:
            ftgpt = new FTGpt<__nv_bfloat16>((size_t)head_num,
                                             (size_t)size_per_head,
                                             (size_t)inter_size,
                                             (size_t)layer_num,
                                             (size_t)vocab_size,
                                             start_id,
                                             end_id,
                                             tensor_para_size,
                                             pipeline_para_size,
                                             int8_mode,
                                             weights,
                                             int8_weights,
                                             scale);
            break;
#endif
        default:
            throw std::runtime_error("Wrong Tensor type.");
    }
}

ParallelGptOp::~ParallelGptOp()
{
    delete ftgpt;
}

std::vector<th::Tensor> ParallelGptOp::forward(th::Tensor input_ids,
                                               th::Tensor input_lengths,
                                               const int64_t output_len,
                                               const int64_t beam_width,
                                               const int64_t top_k,
                                               const double top_p,
                                               const double beam_search_diversity_rate,
                                               const double temperature,
                                               const double len_penalty,
                                               const double repetition_penalty,
                                               const int64_t random_seed,
                                               const int64_t return_cum_log_probs)
{
    CHECK_TH_CUDA(input_ids);
    CHECK_CONTIGUOUS(input_ids);
    TORCH_CHECK(input_ids.dtype() == torch::kInt32, "input_ids dtype should be int32");
    CHECK_TH_CUDA(input_lengths);
    CHECK_CONTIGUOUS(input_lengths);
    TORCH_CHECK(input_lengths.dtype() == torch::kInt32, "input_lengths dtype should be int32");
    TORCH_CHECK(return_cum_log_probs == 0 || return_cum_log_probs == 1 || return_cum_log_probs == 2,
                "return_cum_log_probs should be"
                " 0 (no return cum_log_probs), "
                " 1 (the cumulative log probs of generated sequences), or"
                " 2 (the cumulative log probs of sequences).")

    const int batch_size = input_ids.size(0) / beam_width;
    const int max_input_length = input_ids.size(1);
    const int total_request_output_len = max_input_length + output_len;
    th::Tensor output_ids = torch::empty({batch_size, beam_width, total_request_output_len},
                                         torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    th::Tensor parent_ids = torch::empty({total_request_output_len, batch_size, beam_width},
                                         torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    th::Tensor sequence_lengths =
        torch::empty({batch_size, beam_width}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    th::Tensor cum_log_probs =
        torch::empty({batch_size, beam_width}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));

    ftgpt->forward(input_ids,
                   input_lengths,
                   output_ids,
                   parent_ids,
                   sequence_lengths,
                   cum_log_probs,
                   (const size_t)output_len,
                   (const size_t)beam_width,
                   (const size_t)top_k,
                   (const float)top_p,
                   (const float)beam_search_diversity_rate,
                   (const float)temperature,
                   (const float)len_penalty,
                   (const float)repetition_penalty,
                   (const unsigned long long int)random_seed,
                   return_cum_log_probs);
    if (return_cum_log_probs > 0) {
        return std::vector<th::Tensor>{output_ids, sequence_lengths, cum_log_probs};
    }
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
                                                                std::vector<th::Tensor>,
                                                                std::vector<th::Tensor>,
                                                                std::vector<th::Tensor>>())
                                          .def("forward", &torch_ext::ParallelGptOp::forward);

static auto weight_transpose_calibrate_quantize = torch::RegisterOperators(
    "fastertransformer::weight_transpose_calibrate_quantize", &torch_ext::weight_transpose_calibrate_quantize);
