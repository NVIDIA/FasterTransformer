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

#include "src/fastertransformer/th_op/encoder/EncoderOp.h"

namespace th = torch;
namespace torch_ext {

FasterTransformerEncoder::FasterTransformerEncoder(th::Tensor pre_attr_layernorm_gamma,
                                                   th::Tensor pre_attr_layernorm_beta,
                                                   th::Tensor q_kernel,
                                                   th::Tensor q_bias,
                                                   th::Tensor k_kernel,
                                                   th::Tensor k_bias,
                                                   th::Tensor v_kernel,
                                                   th::Tensor v_bias,
                                                   th::Tensor attr_output_kernel,
                                                   th::Tensor attr_output_bias,
                                                   th::Tensor pre_ffn_layernorm_gamma,
                                                   th::Tensor pre_ffn_layernorm_beta,
                                                   th::Tensor inter_kernel,
                                                   th::Tensor inter_bias,
                                                   th::Tensor output_kernel,
                                                   th::Tensor output_bias,
                                                   th::Tensor post_transformer_layernorm_gamma,
                                                   th::Tensor post_transformer_layernorm_beta,
                                                   int64_t head_num,
                                                   int64_t head_size,
                                                   int64_t inter_size,
                                                   bool remove_padding,
                                                   int64_t layer_num,
                                                   bool allow_gemm_test,
                                                   bool sparse,
                                                   double q_scaling):
    _st(q_kernel.scalar_type()),
    _remove_padding(remove_padding),
    weights{pre_attr_layernorm_gamma,
            pre_attr_layernorm_beta,
            q_kernel,
            q_bias,
            k_kernel,
            k_bias,
            v_kernel,
            v_bias,
            attr_output_kernel,
            attr_output_bias,
            pre_ffn_layernorm_gamma,
            pre_ffn_layernorm_beta,
            inter_kernel,
            inter_bias,
            output_kernel,
            output_bias,
            post_transformer_layernorm_gamma,
            post_transformer_layernorm_beta}
{
    CHECK_INPUT(pre_attr_layernorm_gamma, _st);          // hidden_dim
    CHECK_INPUT(pre_attr_layernorm_beta, _st);           // hidden_dim
    CHECK_INPUT(q_kernel, _st);                          // hidden_dim, hidden_dim
    CHECK_INPUT(q_bias, _st);                            // hidden_dim
    CHECK_INPUT(k_kernel, _st);                          // hidden_dim, hidden_dim
    CHECK_INPUT(k_bias, _st);                            // hidden_dim
    CHECK_INPUT(v_kernel, _st);                          // hidden_dim, hidden_dim
    CHECK_INPUT(v_bias, _st);                            // hidden_dim
    CHECK_INPUT(attr_output_kernel, _st);                // hidden_dim, hidden_dim
    CHECK_INPUT(attr_output_bias, _st);                  // hidden_dim
    CHECK_INPUT(pre_ffn_layernorm_gamma, _st);           // hidden_dim
    CHECK_INPUT(pre_ffn_layernorm_beta, _st);            // hidden_dim
    CHECK_INPUT(inter_kernel, _st);                      // 4 * hidden_dim, hidden_dim
    CHECK_INPUT(inter_bias, _st);                        // 4 * hidden_dim
    CHECK_INPUT(output_kernel, _st);                     // hidden_dim, 4 * hidden_dim
    CHECK_INPUT(output_bias, _st);                       // hidden_dim
    CHECK_INPUT(post_transformer_layernorm_gamma, _st);  // hidden_dim
    CHECK_INPUT(post_transformer_layernorm_beta, _st);   // hidden_dim

    switch (_st) {
        case at::ScalarType::Float:
            ftencoder = new FTEncoder<float>(
                head_num, head_size, inter_size, layer_num, allow_gemm_test, sparse, q_scaling, weights);
            break;
        case at::ScalarType::Half:
            ftencoder = new FTEncoder<half>(
                head_num, head_size, inter_size, layer_num, allow_gemm_test, sparse, q_scaling, weights);
            break;
        default:
            throw std::runtime_error("Wrong Tensor type.");
    }
    head_info = torch::empty({7}, torch::dtype(torch::kInt64));
    head_info[0] = head_num;
    head_info[1] = head_size;
    head_info[2] = (int64_t)remove_padding;
    head_info[3] = layer_num;
    head_info[4] = (int64_t)allow_gemm_test;
    head_info[5] = (int64_t)sparse;
    head_info[6] = inter_size;
    scaling_info = torch::empty({1}, torch::dtype(torch::kFloat64));
    scaling_info[0] = (double)q_scaling;
}

FasterTransformerEncoder::~FasterTransformerEncoder()
{
    delete ftencoder;
}

th::Tensor FasterTransformerEncoder::forward(th::Tensor input, th::Tensor sequence_lengths)
{
    CHECK_INPUT(input, _st);
    CHECK_TH_CUDA(sequence_lengths);
    CHECK_CONTIGUOUS(sequence_lengths);
    TORCH_CHECK(sequence_lengths.dtype() == torch::kInt32, "sequence_lengths dtype should be int32");
    size_t batch_size = (size_t)input.size(0);
    size_t seq_len = (size_t)input.size(1);

    auto output = torch::empty_like(input);
    ftencoder->forward(batch_size, seq_len, input, sequence_lengths, output, _remove_padding);
    return output;
}

std::vector<th::Tensor> FasterTransformerEncoder::get_pickle_info() const
{
    std::vector<th::Tensor> tmp(weights);
    tmp.push_back(head_info);
    tmp.push_back(scaling_info);
    return tmp;
}

}  // namespace torch_ext

static auto fasterTransformerEncoderTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::FasterTransformerEncoder>("FasterTransformerEncoder")
#else
    torch::jit::class_<torch_ext::FasterTransformerEncoder>("FasterTransformer", "Encoder")
#endif
        .def(torch::jit::init<th::Tensor,
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
                              int64_t,
                              int64_t,
                              int64_t,
                              bool,
                              int64_t,
                              bool,
                              bool,
                              double>())
        .def("forward", &torch_ext::FasterTransformerEncoder::forward)
        .def_pickle(
            [](const c10::intrusive_ptr<torch_ext::FasterTransformerEncoder>& self) -> std::vector<th::Tensor> {
                return self->get_pickle_info();
            },
            [](std::vector<th::Tensor> state) -> c10::intrusive_ptr<torch_ext::FasterTransformerEncoder> {
                int64_t head_num = state[18][0].item().to<int>();
                int64_t head_size = state[18][1].item().to<int>();
                bool remove_padding = (bool)(state[18][2].item().to<int>());
                int64_t layer_num = state[18][3].item().to<int>();
                bool allow_gemm_test = (bool)(state[18][4].item().to<int>());
                bool sparse = (bool)(state[18][5].item().to<int>());
                int64_t inter_size = state[18][6].item().to<int>();
                double q_scaling = state[19][0].item().to<double>();
                return c10::make_intrusive<torch_ext::FasterTransformerEncoder>(state[0],
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
                                                                                head_num,
                                                                                head_size,
                                                                                inter_size,
                                                                                remove_padding,
                                                                                layer_num,
                                                                                allow_gemm_test,
                                                                                sparse,
                                                                                q_scaling);
            });