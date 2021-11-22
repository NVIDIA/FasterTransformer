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

#include "src/fastertransformer/th_op/t5/T5DecoderOp.h"

namespace th = torch;
namespace torch_ext {

FasterTransformerT5Decoder::FasterTransformerT5Decoder(th::Tensor self_layernorm_gamma,
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
                                                       int64_t head_num,
                                                       int64_t head_size,
                                                       int64_t inter_size,
                                                       int64_t d_model,
                                                       int64_t layer_num,
                                                       int64_t mem_d_model,
                                                       int64_t tensor_para_size,
                                                       int64_t pipeline_para_size):
    _st(self_kernel_q.scalar_type()),
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
            output_kernel}
{
    for (auto t : weights) {
        CHECK_INPUT(t, _st);
    }

    switch (_st) {
        case at::ScalarType::Float:
            ftdecoder = new FTT5Decoder<float>(head_num,
                                               head_size,
                                               inter_size,
                                               d_model,
                                               layer_num,
                                               mem_d_model,
                                               tensor_para_size,
                                               pipeline_para_size,
                                               weights);
            break;
        case at::ScalarType::Half:
            ftdecoder = new FTT5Decoder<half>(head_num,
                                              head_size,
                                              inter_size,
                                              d_model,
                                              layer_num,
                                              mem_d_model,
                                              tensor_para_size,
                                              pipeline_para_size,
                                              weights);
            break;
        default:
            throw std::runtime_error("Wrong Tensor type.");
    }
    head_info = torch::empty({8}, torch::dtype(torch::kInt64));
    head_info[0] = head_num;
    head_info[1] = head_size;
    head_info[2] = layer_num;
    head_info[3] = inter_size;
    head_info[4] = d_model;
    head_info[5] = mem_d_model;
    head_info[6] = tensor_para_size;
    head_info[7] = pipeline_para_size;
}

FasterTransformerT5Decoder::~FasterTransformerT5Decoder()
{
    delete ftdecoder;
}

std::vector<th::Tensor> FasterTransformerT5Decoder::forward(int64_t step,
                                                            th::Tensor from_tensor,
                                                            th::Tensor memory_tensor,
                                                            th::Tensor memory_sequence_length,
                                                            th::Tensor sequence_length,
                                                            th::Tensor self_cache_keys_tensor,
                                                            th::Tensor self_cache_values_tensor,
                                                            th::Tensor memory_cache_keys_tensor,
                                                            th::Tensor memory_cache_values_tensor,
                                                            th::Tensor relative_attention_bias_tensor)
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
                       memory_cache_values_tensor,
                       relative_attention_bias_tensor);
    return {output_tensor,
            self_cache_keys_tensor,
            self_cache_values_tensor,
            memory_cache_keys_tensor,
            memory_cache_values_tensor};
}

std::vector<th::Tensor> FasterTransformerT5Decoder::get_pickle_info() const
{
    std::vector<th::Tensor> tmp(weights);
    tmp.push_back(head_info);
    return tmp;
}

}  // namespace torch_ext

static auto fasterTransformerDecoderTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::FasterTransformerT5Decoder>("FasterTransformerT5Decoder")
#else
    torch::jit::class_<torch_ext::FasterTransformerT5Decoder>("FasterTransformer", "T5Decoder")
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
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t>())
        .def("forward", &torch_ext::FasterTransformerT5Decoder::forward)
        .def_pickle(
            [](const c10::intrusive_ptr<torch_ext::FasterTransformerT5Decoder>& self) -> std::vector<th::Tensor> {
                return self->get_pickle_info();
            },
            [](std::vector<th::Tensor> state) -> c10::intrusive_ptr<torch_ext::FasterTransformerT5Decoder> {
                int64_t head_num = state[11][0].item().to<int>();
                int64_t head_size = state[11][1].item().to<int>();
                int64_t layer_num = state[11][2].item().to<int>();
                int64_t inter_size = state[11][3].item().to<int>();
                int64_t d_model = state[11][4].item().to<int>();
                int64_t mem_d_model = state[11][5].item().to<int>();
                int64_t tensor_para_size = state[11][6].item().to<int>();
                int64_t pipeline_para_size = state[11][7].item().to<int>();
                return c10::make_intrusive<torch_ext::FasterTransformerT5Decoder>(state[0],
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
                                                                                  head_num,
                                                                                  head_size,
                                                                                  inter_size,
                                                                                  d_model,
                                                                                  layer_num,
                                                                                  mem_d_model,
                                                                                  tensor_para_size,
                                                                                  pipeline_para_size);
            });