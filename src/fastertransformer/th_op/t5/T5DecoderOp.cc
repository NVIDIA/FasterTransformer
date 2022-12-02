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

#include "src/fastertransformer/th_op/t5/T5DecoderOp.h"

namespace th = torch;
namespace torch_ext {

FasterTransformerT5Decoder::FasterTransformerT5Decoder(th::Tensor           self_layernorm_gamma,
                                                       th::Tensor           self_kernel_q,
                                                       th::Tensor           self_output_kernel,
                                                       th::Tensor           cross_layernorm_gamma,
                                                       th::Tensor           cross_kernel_q,
                                                       th::Tensor           cross_kernel_k,
                                                       th::Tensor           cross_kernel_v,
                                                       th::Tensor           cross_output_kernel,
                                                       th::Tensor           ffn_layernorm_gamma,
                                                       th::Tensor           inter_kernel,
                                                       th::Tensor           inter_kernel2,
                                                       th::Tensor           output_kernel,
                                                       th::Tensor           self_layernorm_beta,
                                                       th::Tensor           self_bias_qkv,
                                                       th::Tensor           self_output_bias,
                                                       th::Tensor           cross_layernorm_beta,
                                                       th::Tensor           cross_bias_q,
                                                       th::Tensor           cross_bias_k,
                                                       th::Tensor           cross_bias_v,
                                                       th::Tensor           cross_output_bias,
                                                       th::Tensor           ffn_layernorm_beta,
                                                       th::Tensor           inter_bias,
                                                       th::Tensor           inter_bias2,
                                                       th::Tensor           output_bias,
                                                       int64_t              head_num,
                                                       int64_t              head_size,
                                                       int64_t              inter_size,
                                                       int64_t              d_model,
                                                       int64_t              layer_num,
                                                       int64_t              expert_num,
                                                       int64_t              moe_k,
                                                       int64_t              mem_d_model,
                                                       int64_t              tensor_para_size,
                                                       int64_t              pipeline_para_size,
                                                       bool                 t5_with_bias,
                                                       bool                 use_gated_activation,
                                                       int64_t              position_embedding_type,
                                                       std::vector<int64_t> moe_layer_index):
    _st(self_kernel_q.scalar_type()),
    weights{self_layernorm_gamma, self_kernel_q,  self_output_kernel, cross_layernorm_gamma,
            cross_kernel_q,       cross_kernel_k, cross_kernel_v,     cross_output_kernel,
            ffn_layernorm_gamma,  inter_kernel,   inter_kernel2,      output_kernel,
            self_layernorm_beta,  self_bias_qkv,  self_output_bias,   cross_layernorm_beta,
            cross_bias_q,         cross_bias_k,   cross_bias_v,       cross_output_bias,
            ffn_layernorm_beta,   inter_bias,     inter_bias2,        output_bias}
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
                                               expert_num,
                                               moe_k,
                                               mem_d_model,
                                               tensor_para_size,
                                               pipeline_para_size,
                                               t5_with_bias,
                                               use_gated_activation,
                                               position_embedding_type,
                                               moe_layer_index,
                                               weights);
            break;
        case at::ScalarType::Half:
            ftdecoder = new FTT5Decoder<half>(head_num,
                                              head_size,
                                              inter_size,
                                              d_model,
                                              layer_num,
                                              expert_num,
                                              moe_k,
                                              mem_d_model,
                                              tensor_para_size,
                                              pipeline_para_size,
                                              t5_with_bias,
                                              use_gated_activation,
                                              position_embedding_type,
                                              moe_layer_index,
                                              weights);
            break;
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16:
            ftdecoder = new FTT5Decoder<__nv_bfloat16>(head_num,
                                                       head_size,
                                                       inter_size,
                                                       d_model,
                                                       layer_num,
                                                       expert_num,
                                                       moe_k,
                                                       mem_d_model,
                                                       tensor_para_size,
                                                       pipeline_para_size,
                                                       t5_with_bias,
                                                       use_gated_activation,
                                                       position_embedding_type,
                                                       moe_layer_index,
                                                       weights);
            break;
#endif
        default:
            throw std::runtime_error("Wrong Tensor type.");
    }
}

FasterTransformerT5Decoder::~FasterTransformerT5Decoder()
{
    delete ftdecoder;
}

std::vector<th::Tensor> FasterTransformerT5Decoder::forward(int64_t    step,
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
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              bool,
                              bool,
                              int64_t,
                              std::vector<int64_t>>())
        .def("forward", &torch_ext::FasterTransformerT5Decoder::forward);
