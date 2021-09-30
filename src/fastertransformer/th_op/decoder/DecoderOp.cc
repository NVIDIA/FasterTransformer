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

#include "src/fastertransformer/th_op/decoder/DecoderOp.h"

namespace th = torch;
namespace torch_ext {

FasterTransformerDecoder::FasterTransformerDecoder(th::Tensor self_layernorm_gamma,
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
                                                   int64_t head_num,
                                                   int64_t head_size,
                                                   int64_t inter_size,
                                                   int64_t layer_num,
                                                   int64_t mem_hidden_dim):
    _st(self_kernel_q.scalar_type()),
    weights{self_layernorm_gamma, self_layernorm_beta, self_kernel_q,         self_bias_q,
            self_output_kernel,   self_output_bias,    cross_layernorm_gamma, cross_layernorm_beta,
            cross_kernel_q,       cross_kernel_k,      cross_kernel_v,        cross_bias_q,
            cross_bias_k,         cross_bias_v,        cross_output_kernel,   cross_output_bias,
            ffn_layernorm_gamma,  ffn_layernorm_beta,  inter_kernel,          inter_bias,
            output_kernel,        output_bias}
{
    for (auto t : weights) {
        CHECK_INPUT(t, _st);
    }

    switch (_st) {
        case at::ScalarType::Float:
            ftdecoder = new FTDecoder<float>(head_num, head_size, inter_size, layer_num, mem_hidden_dim, weights);
            break;
        case at::ScalarType::Half:
            ftdecoder = new FTDecoder<half>(head_num, head_size, inter_size, layer_num, mem_hidden_dim, weights);
            break;
        default:
            throw std::runtime_error("Wrong Tensor type.");
    }
    head_info = torch::empty({4}, torch::dtype(torch::kInt64));
    head_info[0] = head_num;
    head_info[1] = head_size;
    head_info[2] = layer_num;
    head_info[3] = inter_size;
    head_info[3] = mem_hidden_dim;
}

FasterTransformerDecoder::~FasterTransformerDecoder()
{
    delete ftdecoder;
}

std::vector<th::Tensor> FasterTransformerDecoder::forward(int64_t step,
                                                          th::Tensor from_tensor,
                                                          th::Tensor memory_tensor,
                                                          th::Tensor memory_sequence_length,
                                                          th::Tensor sequence_length,
                                                          th::Tensor self_cache_keys_tensor,
                                                          th::Tensor self_cache_values_tensor,
                                                          th::Tensor memory_cache_keys_tensor,
                                                          th::Tensor memory_cache_values_tensor)
{
    CHECK_INPUT(from_tensor, _st);
    CHECK_INPUT(memory_tensor, _st);

    CHECK_TH_CUDA(memory_sequence_length);
    CHECK_CONTIGUOUS(memory_sequence_length);
    TORCH_CHECK(memory_sequence_length.dtype() == torch::kInt32, "memory_sequence_length dtype should be int32");

    CHECK_TH_CUDA(sequence_length);
    CHECK_CONTIGUOUS(sequence_length);
    TORCH_CHECK(sequence_length.dtype() == torch::kInt32, "sequence_length dtype should be int32");

    CHECK_INPUT(self_cache_keys_tensor, _st);
    CHECK_INPUT(self_cache_values_tensor, _st);
    CHECK_INPUT(memory_cache_keys_tensor, _st);
    CHECK_INPUT(memory_cache_values_tensor, _st);

    size_t batch_size = (size_t)from_tensor.size(0);

    auto output_tensor = torch::empty_like(from_tensor);
    ftdecoder->forward(batch_size,
                       (size_t)step,
                       from_tensor,
                       memory_tensor,
                       memory_sequence_length,
                       sequence_length,
                       output_tensor,
                       self_cache_keys_tensor,
                       self_cache_values_tensor,
                       memory_cache_keys_tensor,
                       memory_cache_values_tensor);
    return {output_tensor,
            self_cache_keys_tensor,
            self_cache_values_tensor,
            memory_cache_keys_tensor,
            memory_cache_values_tensor};
}

std::vector<th::Tensor> FasterTransformerDecoder::get_pickle_info() const
{
    std::vector<th::Tensor> tmp(weights);
    tmp.push_back(head_info);
    return tmp;
}

}  // namespace torch_ext

static auto fasterTransformerDecoderTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::FasterTransformerDecoder>("FasterTransformerDecoder")
#else
    torch::jit::class_<torch_ext::FasterTransformerDecoder>("FasterTransformer", "Decoder")
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
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t>())
        .def("forward", &torch_ext::FasterTransformerDecoder::forward)
        .def_pickle(
            [](const c10::intrusive_ptr<torch_ext::FasterTransformerDecoder>& self) -> std::vector<th::Tensor> {
                return self->get_pickle_info();
            },
            [](std::vector<th::Tensor> state) -> c10::intrusive_ptr<torch_ext::FasterTransformerDecoder> {
                int64_t head_num = state[22][0].item().to<int>();
                int64_t head_size = state[22][1].item().to<int>();
                int64_t layer_num = state[22][2].item().to<int>();
                int64_t inter_size = state[22][3].item().to<int>();
                int64_t mem_hidden_dim = state[22][4].item().to<int>();
                return c10::make_intrusive<torch_ext::FasterTransformerDecoder>(state[0],
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
                                                                                head_num,
                                                                                head_size,
                                                                                inter_size,
                                                                                layer_num,
                                                                                mem_hidden_dim);
            });