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

#include "src/fastertransformer/th_op/decoding/DecodingOp.h"

namespace th = torch;

namespace torch_ext {
using torch::Tensor;

FasterTransformerDecoding::FasterTransformerDecoding(int64_t head_num,
                                                     int64_t size_per_head,
                                                     int64_t inter_size,
                                                     int64_t mem_hidden_dim,
                                                     int64_t layer_num,
                                                     int64_t vocab_size,
                                                     int64_t start_id,
                                                     int64_t end_id,
                                                     double beam_search_diversity_rate,
                                                     int64_t top_k,
                                                     double top_p,
                                                     double temperature,
                                                     double len_penalty,
                                                     double repetition_penalty,
                                                     th::Tensor self_layernorm_gamma,
                                                     th::Tensor self_layernorm_beta,
                                                     th::Tensor self_kernel_q,
                                                     th::Tensor self_bias_q,
                                                     th::Tensor self_output_kernel,
                                                     th::Tensor self_output_bias,
                                                     th::Tensor cross_layernorm_gamma,
                                                     th::Tensor cross_layernorm_beta,
                                                     th::Tensor cross_kernel_q,
                                                     th::Tensor cross_kernel_k,
                                                     th::Tensor cross_kernel_v,
                                                     th::Tensor cross_bias_q,
                                                     th::Tensor cross_bias_k,
                                                     th::Tensor cross_bias_v,
                                                     th::Tensor cross_output_kernel,
                                                     th::Tensor cross_output_bias,
                                                     th::Tensor ffn_layernorm_gamma,
                                                     th::Tensor ffn_layernorm_beta,
                                                     th::Tensor inter_kernel,
                                                     th::Tensor inter_bias,
                                                     th::Tensor output_kernel,
                                                     th::Tensor output_bias,
                                                     th::Tensor decoding_gamma,
                                                     th::Tensor decoding_beta,
                                                     th::Tensor embedding_table,
                                                     th::Tensor position_encoding_table,
                                                     th::Tensor embedding_kernel,
                                                     th::Tensor embedding_bias):
    _st(self_layernorm_gamma.scalar_type()),
    weights{self_layernorm_gamma,    self_layernorm_beta,   self_kernel_q,        self_bias_q,    self_output_kernel,
            self_output_bias,        cross_layernorm_gamma, cross_layernorm_beta, cross_kernel_q, cross_kernel_k,
            cross_kernel_v,          cross_bias_q,          cross_bias_k,         cross_bias_v,   cross_output_kernel,
            cross_output_bias,       ffn_layernorm_gamma,   ffn_layernorm_beta,   inter_kernel,   inter_bias,
            output_kernel,           output_bias,           decoding_gamma,       decoding_beta,  embedding_table,
            position_encoding_table, embedding_kernel,      embedding_bias}
{
    CHECK_INPUT(self_layernorm_gamma, _st);     // layer_num, hidden_dim
    CHECK_INPUT(self_layernorm_beta, _st);      // layer_num, hidden_dim
    CHECK_INPUT(self_kernel_q, _st);            // layer_num, hidden_dim, 3 * hidden_dim
    CHECK_INPUT(self_bias_q, _st);              // layer_num, 3 * hidden_dim
    CHECK_INPUT(self_output_kernel, _st);       // layer_num, hidden_dim, hidden_dim
    CHECK_INPUT(self_output_bias, _st);         // layer_num, hidden_dim
    CHECK_INPUT(cross_layernorm_gamma, _st);    // layer_num, hidden_dim
    CHECK_INPUT(cross_layernorm_beta, _st);     // layer_num, hidden_dim
    CHECK_INPUT(cross_kernel_q, _st);           // layer_num, hidden_dim, hidden_dim
    CHECK_INPUT(cross_kernel_k, _st);           // layer_num, mem_hidden_dim, hidden_dim
    CHECK_INPUT(cross_kernel_v, _st);           // layer_num, mem_hidden_dim, hidden_dim
    CHECK_INPUT(cross_bias_q, _st);             // layer_num, hidden_dim
    CHECK_INPUT(cross_bias_k, _st);             // layer_num, hidden_dim
    CHECK_INPUT(cross_bias_v, _st);             // layer_num, hidden_dim
    CHECK_INPUT(cross_output_kernel, _st);      // layer_num, hidden_dim, hidden_dim
    CHECK_INPUT(cross_output_bias, _st);        // layer_num, hidden_dim
    CHECK_INPUT(ffn_layernorm_gamma, _st);      // layer_num, hidden_dim
    CHECK_INPUT(ffn_layernorm_beta, _st);       // layer_num, hidden_dim
    CHECK_INPUT(inter_kernel, _st);             // layer_num, hidden_dim, 4 * hidden_dim
    CHECK_INPUT(inter_bias, _st);               // layer_num, 4 * hidden_dim
    CHECK_INPUT(output_kernel, _st);            // layer_num, 4 * hidden_dim, hidden_dim
    CHECK_INPUT(output_bias, _st);              // layer_num, hidden_dim
    CHECK_INPUT(decoding_gamma, _st);           // hidden_dim
    CHECK_INPUT(decoding_beta, _st);            // hidden_dim
    CHECK_INPUT(embedding_table, _st);          // vocab_size, hidden_dim
    CHECK_INPUT(position_encoding_table, _st);  // max_step, hidden_dim
    CHECK_INPUT(embedding_kernel, _st);         // hidden_dim, vocab_size
    CHECK_INPUT(embedding_bias, _st);           // vocab_size
    switch (_st) {
        case at::ScalarType::Float:
            ftdecoding = new torch_ext::FTDecoding<float>(head_num,
                                                          size_per_head,
                                                          inter_size,
                                                          mem_hidden_dim,
                                                          layer_num,
                                                          vocab_size,
                                                          start_id,
                                                          end_id,
                                                          (float)beam_search_diversity_rate,
                                                          top_k,
                                                          (float)top_p,
                                                          (float)temperature,
                                                          (float)len_penalty,
                                                          (float)repetition_penalty,
                                                          weights);
            break;
        case at::ScalarType::Half:
            ftdecoding = new torch_ext::FTDecoding<half>(head_num,
                                                         size_per_head,
                                                         inter_size,
                                                         mem_hidden_dim,
                                                         layer_num,
                                                         vocab_size,
                                                         start_id,
                                                         end_id,
                                                         (float)beam_search_diversity_rate,
                                                         top_k,
                                                         (float)top_p,
                                                         (float)temperature,
                                                         (float)len_penalty,
                                                         (float)repetition_penalty,
                                                         weights);
            break;
        default:
            throw std::runtime_error("Wrong Tensor type.");
    }
    int_info_ = torch::empty({11}, torch::dtype(torch::kInt64));
    float_info_ = torch::empty({5}, torch::dtype(torch::kFloat64));
    int_info_[0] = head_num;
    int_info_[1] = size_per_head;
    int_info_[2] = inter_size;
    int_info_[3] = mem_hidden_dim;
    int_info_[4] = layer_num;
    int_info_[5] = vocab_size;
    int_info_[6] = start_id;
    int_info_[7] = end_id;
    int_info_[8] = top_k;

    float_info_[0] = beam_search_diversity_rate;
    float_info_[1] = top_p;
    float_info_[2] = temperature;
    float_info_[3] = len_penalty;
    float_info_[4] = repetition_penalty;
}

FasterTransformerDecoding::~FasterTransformerDecoding()
{
    delete ftdecoding;
}

std::vector<th::Tensor> FasterTransformerDecoding::forward(int64_t beam_width,
                                                           int64_t max_seq_len,
                                                           th::Tensor memory,
                                                           th::Tensor memory_seq_lens)
{
    CHECK_INPUT(memory, _st);
    CHECK_TH_CUDA(memory_seq_lens);
    CHECK_CONTIGUOUS(memory_seq_lens);
    TORCH_CHECK(memory_seq_lens.dtype() == torch::kInt32, "mem_seq_lens dtype should be int32");

    int batch_size = (int)(memory.size(0) / beam_width);
    auto output_ids = torch::empty({batch_size * beam_width * max_seq_len},
                                   torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    auto parent_ids = torch::empty({batch_size * beam_width * max_seq_len},
                                   torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    auto out_seq_lens =
        torch::empty({batch_size * beam_width}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    ftdecoding->forward(
        (size_t)beam_width, (size_t)max_seq_len, memory, memory_seq_lens, output_ids, parent_ids, out_seq_lens);
    return std::vector<th::Tensor>{output_ids, parent_ids, out_seq_lens};
}

std::vector<th::Tensor> FasterTransformerDecoding::get_pickle_info() const
{
    std::vector<th::Tensor> tmp(weights);
    tmp.push_back(int_info_);
    tmp.push_back(float_info_);
    return tmp;
}

}  // namespace torch_ext

static auto fasterTransformerDecodingTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::FasterTransformerDecoding>("FasterTransformerDecoding")
#else
    torch::jit::class_<torch_ext::FasterTransformerDecoding>("FasterTransformer", "Decoding")
#endif
        .def(torch::jit::init<int64_t,
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
                              double,
                              double,
                              double,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor>())
        .def("forward", &torch_ext::FasterTransformerDecoding::forward)
        .def_pickle(
            [](const c10::intrusive_ptr<torch_ext::FasterTransformerDecoding>& self) -> std::vector<th::Tensor> {
                return self->get_pickle_info();
            },
            [](std::vector<th::Tensor> state) -> c10::intrusive_ptr<torch_ext::FasterTransformerDecoding> {
                int head_num = state[28][0].item().to<int>();
                int size_per_head = state[28][1].item().to<int>();
                int inter_size = state[28][2].item().to<int>();
                int mem_hidden_dim = state[28][3].item().to<int>();
                int layer_num = state[28][4].item().to<int>();
                int vocab_size = state[28][5].item().to<int>();
                int start_id = state[28][6].item().to<int>();
                int end_id = state[28][7].item().to<int>();
                int top_k = state[28][8].item().to<int>();

                // TODO(bhsueh) Here may have bugs
                double beam_search_diversity_rate = state[33][0].item().to<double>();
                double top_p = state[33][1].item().to<double>();
                double temperature = state[33][2].item().to<double>();
                double len_penalty = state[33][3].item().to<double>();
                double repetition_penalty = state[33][4].item().to<double>();

                return c10::make_intrusive<torch_ext::FasterTransformerDecoding>(head_num,
                                                                                 size_per_head,
                                                                                 inter_size,
                                                                                 mem_hidden_dim,
                                                                                 layer_num,
                                                                                 vocab_size,
                                                                                 start_id,
                                                                                 end_id,
                                                                                 beam_search_diversity_rate,
                                                                                 top_k,
                                                                                 top_p,
                                                                                 temperature,
                                                                                 len_penalty,
                                                                                 repetition_penalty,
                                                                                 state[0],
                                                                                 state[1],
                                                                                 state[2],
                                                                                 state[3],
                                                                                 state[4],
                                                                                 state[5],
                                                                                 state[6],
                                                                                 state[7],
                                                                                 state[8],
                                                                                 state[9],
                                                                                 state[10],
                                                                                 state[11],
                                                                                 state[12],
                                                                                 state[13],
                                                                                 state[14],
                                                                                 state[15],
                                                                                 state[16],
                                                                                 state[17],
                                                                                 state[18],
                                                                                 state[19],
                                                                                 state[20],
                                                                                 state[21],
                                                                                 state[22],
                                                                                 state[23],
                                                                                 state[24],
                                                                                 state[25],
                                                                                 state[26],
                                                                                 state[27]);
            });