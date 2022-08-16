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

#include "src/fastertransformer/th_op/gpt/GptOp.h"

namespace th = torch;
namespace ft = fastertransformer;
namespace torch_ext {

GptOp::GptOp(const int64_t                 head_num,
             const int64_t                 size_per_head,
             const int64_t                 inter_size,
             const int64_t                 layer_num,
             const int64_t                 vocab_size,
             const int64_t                 start_id,
             const int64_t                 end_id,
             const bool                    sparse,
             const double                  layernorm_eps,
             const std::string             layernorm_type,
             const std::string             activation_type,
             const bool                    has_post_decoder_layernorm,
             const bool                    has_adapters,
             const int64_t                 adapter_inter_size,
             const std::vector<th::Tensor> weights):
    st_(weights[0].scalar_type())
{
    for (auto t : weights) {
        CHECK_INPUT(t, st_);
    }

    ft::gptVariantParams gpt_variant_params{(float)layernorm_eps,
                                            ft::getLayerNormType(layernorm_type),
                                            ft::getActivationType(activation_type),
                                            has_post_decoder_layernorm,
                                            has_adapters,
                                            (size_t)adapter_inter_size};

    switch (st_) {
        case at::ScalarType::Float:
            ftgpt = new FTGpt<float>((size_t)head_num,
                                     (size_t)size_per_head,
                                     (size_t)inter_size,
                                     (size_t)layer_num,
                                     (size_t)vocab_size,
                                     gpt_variant_params,
                                     start_id,
                                     end_id,
                                     sparse,
                                     weights);
            break;
        case at::ScalarType::Half:
            ftgpt = new FTGpt<half>((size_t)head_num,
                                    (size_t)size_per_head,
                                    (size_t)inter_size,
                                    (size_t)layer_num,
                                    (size_t)vocab_size,
                                    gpt_variant_params,
                                    start_id,
                                    end_id,
                                    sparse,
                                    weights);
            break;
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16:
            ftgpt = new FTGpt<__nv_bfloat16>((size_t)head_num,
                                             (size_t)size_per_head,
                                             (size_t)inter_size,
                                             (size_t)layer_num,
                                             (size_t)vocab_size,
                                             gpt_variant_params,
                                             start_id,
                                             end_id,
                                             sparse,
                                             weights);
            break;
#endif
        default:
            throw std::runtime_error("Wrong Tensor type.");
    }
}

GptOp::~GptOp()
{
    delete ftgpt;
}

std::vector<th::Tensor> GptOp::forward(th::Tensor               input_ids,
                                       th::Tensor               input_lengths,
                                       const int64_t            output_len,
                                       th::optional<int64_t>    beam_width_opt,
                                       th::optional<th::Tensor> top_k_opt,
                                       th::optional<th::Tensor> top_p_opt,
                                       th::optional<th::Tensor> beam_search_diversity_rate_opt,
                                       th::optional<th::Tensor> temperature_opt,
                                       th::optional<th::Tensor> len_penalty_opt,
                                       th::optional<th::Tensor> repetition_penalty_opt,
                                       th::optional<th::Tensor> random_seed_opt,
                                       th::optional<int64_t>    return_cum_log_probs_opt)
{
    CHECK_TH_CUDA(input_ids);
    CHECK_CONTIGUOUS(input_ids);
    TORCH_CHECK(input_ids.dtype() == torch::kInt32, "input_ids dtype should be int32");
    CHECK_TH_CUDA(input_lengths);
    CHECK_CONTIGUOUS(input_lengths);
    TORCH_CHECK(input_lengths.dtype() == torch::kInt32, "input_lengths dtype should be int32");
    int64_t return_cum_log_probs = return_cum_log_probs_opt.has_value() ? (int64_t)return_cum_log_probs_opt.value() : 0;
    if (return_cum_log_probs_opt.has_value()) {
        TORCH_CHECK(return_cum_log_probs == 0 || return_cum_log_probs == 1 || return_cum_log_probs == 2,
                    "return_cum_log_probs should be"
                    " 0 (no return cum_log_probs), "
                    " 1 (the cumulative log probs of generated sequences), or"
                    " 2 (the cumulative log probs of sequences).")
    }

    const int beam_width = beam_width_opt.has_value() ? (int)beam_width_opt.value() : 1;

    const int  batch_size               = input_ids.size(0);
    const int  max_input_length         = input_ids.size(1);
    const int  total_request_output_len = max_input_length + output_len;
    th::Tensor output_ids               = torch::empty({batch_size, beam_width, total_request_output_len},
                                         torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    th::Tensor sequence_lengths =
        torch::empty({batch_size, beam_width}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    th::Tensor cum_log_probs =
        torch::empty({batch_size, beam_width}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));

    ftgpt->forward(input_ids,
                   input_lengths,
                   output_ids,
                   sequence_lengths,
                   cum_log_probs,
                   (const size_t)output_len,
                   (const size_t)beam_width,
                   top_k_opt,
                   top_p_opt,
                   beam_search_diversity_rate_opt,
                   temperature_opt,
                   len_penalty_opt,
                   repetition_penalty_opt,
                   random_seed_opt,
                   return_cum_log_probs_opt);
    if (return_cum_log_probs > 0) {
        return std::vector<th::Tensor>{output_ids, sequence_lengths, cum_log_probs};
    }
    return std::vector<th::Tensor>{output_ids, sequence_lengths};
}

}  // namespace torch_ext

static auto fasterTransformerGptTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::GptOp>("FasterTransformerGptOp")
#else
    torch::jit::class_<torch_ext::GptOp>("FasterTransformer", "GptOp")
#endif
        .def(torch::jit::init<int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              bool,
                              double,
                              std::string,
                              std::string,
                              bool,
                              bool,
                              int64_t,
                              std::vector<th::Tensor>>())
        .def("forward", &torch_ext::GptOp::forward);
