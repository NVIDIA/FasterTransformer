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

#include "src/fastertransformer/th_op/t5/T5DecodingOp.h"

namespace th = torch;

namespace torch_ext {

FasterTransformerT5Decoding::FasterTransformerT5Decoding(int64_t head_num,
                                                         int64_t size_per_head,
                                                         int64_t inter_size,
                                                         int64_t mem_d_model,
                                                         int64_t d_model,
                                                         int64_t layer_num,
                                                         int64_t vocab_size,
                                                         int64_t num_bucket,
                                                         int64_t max_distance,
                                                         int64_t start_id,
                                                         int64_t end_id,
                                                         double beam_search_diversity_rate,
                                                         int64_t top_k,
                                                         double top_p,
                                                         double temperature,
                                                         double len_penalty,
                                                         double repetition_penalty,
                                                         int64_t tensor_para_size,
                                                         int64_t pipeline_para_size,
                                                         th::Tensor self_layernorm_gamma,
                                                         th::Tensor self_kernel_q,
                                                         th::Tensor self_output_kernel,
                                                         th::Tensor cross_layernorm_gamma,
                                                         th::Tensor cross_kernel_q,
                                                         th::Tensor cross_kernel_k,
                                                         th::Tensor cross_kernel_v,
                                                         th::Tensor cross_output_kernel,
                                                         th::Tensor ffn_layernorm_gamma,
                                                         th::Tensor inter_kernel,
                                                         th::Tensor output_kernel,
                                                         th::Tensor decoding_gamma,
                                                         th::Tensor embedding_table,
                                                         th::Tensor relative_attention_bias):
    _st(self_layernorm_gamma.scalar_type()),
    weights{self_layernorm_gamma,
            self_kernel_q,
            self_output_kernel,
            cross_layernorm_gamma,
            cross_kernel_q,
            cross_kernel_k,
            cross_kernel_v,
            cross_output_kernel,
            ffn_layernorm_gamma,
            inter_kernel,
            output_kernel,
            decoding_gamma,
            embedding_table,
            relative_attention_bias}
{
    CHECK_INPUT(self_layernorm_gamma, _st);     // layer_num, d_model
    CHECK_INPUT(self_kernel_q, _st);            // layer_num, d_model, 3 * hidden_dim
    CHECK_INPUT(self_output_kernel, _st);       // layer_num, hidden_dim, d_model
    CHECK_INPUT(cross_layernorm_gamma, _st);    // layer_num, d_model
    CHECK_INPUT(cross_kernel_q, _st);           // layer_num, d_model, hidden_dim
    CHECK_INPUT(cross_kernel_k, _st);           // layer_num, mem_d_model, hidden_dim
    CHECK_INPUT(cross_kernel_v, _st);           // layer_num, mem_d_model, hidden_dim
    CHECK_INPUT(cross_output_kernel, _st);      // layer_num, hidden_dim, d_model
    CHECK_INPUT(ffn_layernorm_gamma, _st);      // layer_num, d_model
    CHECK_INPUT(inter_kernel, _st);             // layer_num, d_model, inter_size
    CHECK_INPUT(output_kernel, _st);            // layer_num, inter_size, d_model
    CHECK_INPUT(decoding_gamma, _st);           // d_model
    CHECK_INPUT(embedding_table, _st);          // vocab_size, d_model
    CHECK_INPUT(relative_attention_bias, _st);  // head_num, num_bucket
    switch (_st) {
        case at::ScalarType::Float:
            ftdecoding = new torch_ext::FTT5Decoding<float>(head_num,
                                                            size_per_head,
                                                            inter_size,
                                                            mem_d_model,
                                                            d_model,
                                                            layer_num,
                                                            vocab_size,
                                                            num_bucket,
                                                            max_distance,
                                                            start_id,
                                                            end_id,
                                                            (float)beam_search_diversity_rate,
                                                            top_k,
                                                            (float)top_p,
                                                            (float)temperature,
                                                            (float)len_penalty,
                                                            (float)repetition_penalty,
                                                            tensor_para_size,
                                                            pipeline_para_size,
                                                            weights);
            break;
        case at::ScalarType::Half:
            ftdecoding = new torch_ext::FTT5Decoding<half>(head_num,
                                                           size_per_head,
                                                           inter_size,
                                                           mem_d_model,
                                                           d_model,
                                                           layer_num,
                                                           vocab_size,
                                                           num_bucket,
                                                           max_distance,
                                                           start_id,
                                                           end_id,
                                                           (float)beam_search_diversity_rate,
                                                           top_k,
                                                           (float)top_p,
                                                           (float)temperature,
                                                           (float)len_penalty,
                                                           (float)repetition_penalty,
                                                           tensor_para_size,
                                                           pipeline_para_size,
                                                           weights);
            break;
        default:
            throw std::runtime_error("Wrong Tensor type.");
    }
    int_info_ = torch::empty({14}, torch::dtype(torch::kInt64));
    float_info_ = torch::empty({5}, torch::dtype(torch::kFloat64));
    int_info_[0] = head_num;
    int_info_[1] = size_per_head;
    int_info_[2] = inter_size;
    int_info_[3] = mem_d_model;
    int_info_[4] = d_model;
    int_info_[5] = layer_num;
    int_info_[6] = vocab_size;
    int_info_[7] = num_bucket;
    int_info_[8] = max_distance;
    int_info_[9] = start_id;
    int_info_[10] = end_id;
    int_info_[11] = top_k;
    int_info_[12] = tensor_para_size;
    int_info_[13] = pipeline_para_size;

    float_info_[0] = beam_search_diversity_rate;
    float_info_[1] = top_p;
    float_info_[2] = temperature;
    float_info_[3] = len_penalty;
    float_info_[4] = repetition_penalty;
}

FasterTransformerT5Decoding::~FasterTransformerT5Decoding()
{
    delete ftdecoding;
}

std::vector<th::Tensor> FasterTransformerT5Decoding::forward(int64_t beam_width,
                                                             int64_t max_seq_len,
                                                             th::Tensor memory,
                                                             th::Tensor memory_seq_lens)
{
    CHECK_INPUT(memory, _st);
    CHECK_TH_CUDA(memory_seq_lens);
    CHECK_CONTIGUOUS(memory_seq_lens);
    TORCH_CHECK(memory_seq_lens.dtype() == torch::kInt32, "mem_seq_lens dtype should be int32");

    int batch_size = (int)(memory.size(0));
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

std::vector<th::Tensor> FasterTransformerT5Decoding::get_pickle_info() const
{
    std::vector<th::Tensor> tmp(weights);
    tmp.push_back(int_info_);
    tmp.push_back(float_info_);
    return tmp;
}

}  // namespace torch_ext

static auto fasterTransformerT5DecodingTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::FasterTransformerT5Decoding>("FasterTransformerT5Decoding")
#else
    torch::jit::class_<torch_ext::FasterTransformerT5Decoding>("FasterTransformer", "T5Decoding")
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
                              int64_t,
                              double,
                              int64_t,
                              double,
                              double,
                              double,
                              double,
                              int64_t,
                              int64_t,
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
        .def("forward", &torch_ext::FasterTransformerT5Decoding::forward)
        .def_pickle(
            [](const c10::intrusive_ptr<torch_ext::FasterTransformerT5Decoding>& self) -> std::vector<th::Tensor> {
                return self->get_pickle_info();
            },
            [](std::vector<th::Tensor> state) -> c10::intrusive_ptr<torch_ext::FasterTransformerT5Decoding> {
                int head_num = state[15][0].item().to<int>();
                int size_per_head = state[15][1].item().to<int>();
                int inter_size = state[15][2].item().to<int>();
                int mem_d_model = state[15][3].item().to<int>();
                int d_model = state[15][4].item().to<int>();
                int layer_num = state[15][5].item().to<int>();
                int vocab_size = state[15][6].item().to<int>();
                int num_bucket = state[15][7].item().to<int>();
                int max_distance = state[15][8].item().to<int>();
                int start_id = state[15][9].item().to<int>();
                int end_id = state[15][10].item().to<int>();
                int top_k = state[15][11].item().to<int>();
                int tensor_para_size = state[15][12].item().to<int>();
                int pipeline_para_size = state[15][13].item().to<int>();

                double beam_search_diversity_rate = state[16][0].item().to<double>();
                double top_p = state[16][1].item().to<double>();
                double temperature = state[16][2].item().to<double>();
                double len_penalty = state[16][3].item().to<double>();
                double repetition_penalty = state[16][4].item().to<double>();

                return c10::make_intrusive<torch_ext::FasterTransformerT5Decoding>(head_num,
                                                                                   size_per_head,
                                                                                   inter_size,
                                                                                   mem_d_model,
                                                                                   d_model,
                                                                                   layer_num,
                                                                                   vocab_size,
                                                                                   num_bucket,
                                                                                   max_distance,
                                                                                   start_id,
                                                                                   end_id,
                                                                                   beam_search_diversity_rate,
                                                                                   top_k,
                                                                                   top_p,
                                                                                   temperature,
                                                                                   len_penalty,
                                                                                   repetition_penalty,
                                                                                   tensor_para_size,
                                                                                   pipeline_para_size,
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
                                                                                   state[13]);
            });