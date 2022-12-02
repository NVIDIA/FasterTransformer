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

#include "src/fastertransformer/th_op/bart/BartEncoderOp.h"

namespace th = torch;
namespace torch_ext {

FasterTransformerBartEncoder::FasterTransformerBartEncoder(th::Tensor  attr_output_layernorm_gamma,
                                                           th::Tensor  q_kernel,
                                                           th::Tensor  k_kernel,
                                                           th::Tensor  v_kernel,
                                                           th::Tensor  attr_output_kernel,
                                                           th::Tensor  output_layernorm_gamma,
                                                           th::Tensor  inter_kernel,
                                                           th::Tensor  inter_kernel2,
                                                           th::Tensor  output_kernel,
                                                           th::Tensor  absolute_or_relative_position_embedding,
                                                           th::Tensor  embedding_table,
                                                           th::Tensor  pre_transformer_layernorm_gamma,
                                                           th::Tensor  post_transformer_layernorm_gamma,
                                                           th::Tensor  attr_output_layernorm_beta,
                                                           th::Tensor  q_bias,
                                                           th::Tensor  k_bias,
                                                           th::Tensor  v_bias,
                                                           th::Tensor  attr_output_bias,
                                                           th::Tensor  output_layernorm_beta,
                                                           th::Tensor  inter_bias,
                                                           th::Tensor  inter_bias2,
                                                           th::Tensor  output_bias,
                                                           th::Tensor  pre_transformer_layernorm_beta,
                                                           th::Tensor  post_transformer_layernorm_beta,
                                                           int64_t     head_num,
                                                           int64_t     head_size,
                                                           int64_t     inter_size,
                                                           int64_t     d_model,
                                                           bool        remove_padding,
                                                           int64_t     layer_num,
                                                           int64_t     num_bucket,
                                                           int64_t     max_distance,
                                                           bool        sparse,
                                                           double      q_scaling,
                                                           int64_t     tensor_para_size,
                                                           int64_t     pipeline_para_size,
                                                           bool        bart_with_bias,
                                                           bool        mbart,
                                                           int64_t     position_embedding_type,
                                                           std::string activation_type,
                                                           std::string layernorm_type):
    d_model_(d_model),
    _st(q_kernel.scalar_type()),
    _remove_padding(remove_padding),
    weights{attr_output_layernorm_gamma,
            q_kernel,
            k_kernel,
            v_kernel,
            attr_output_kernel,
            output_layernorm_gamma,
            inter_kernel,
            inter_kernel2,
            output_kernel,
            absolute_or_relative_position_embedding,
            embedding_table,
            pre_transformer_layernorm_gamma,
            post_transformer_layernorm_gamma,
            attr_output_layernorm_beta,
            q_bias,
            k_bias,
            v_bias,
            attr_output_bias,
            output_layernorm_beta,
            inter_bias,
            inter_bias2,
            output_bias,
            pre_transformer_layernorm_beta,
            post_transformer_layernorm_beta}
{
    CHECK_INPUT(q_kernel, _st);                         // d_model, hidden_dim
    CHECK_INPUT(k_kernel, _st);                         // d_model, hidden_dim
    CHECK_INPUT(v_kernel, _st);                         // d_model, hidden_dim
    CHECK_INPUT(attr_output_kernel, _st);               // hidden_dim, d_model
    CHECK_INPUT(attr_output_layernorm_gamma, _st);      // d_model
    CHECK_INPUT(inter_kernel, _st);                     // d_model, inter_size
    CHECK_INPUT(inter_kernel2, _st);                    // d_model, inter_size
    CHECK_INPUT(output_kernel, _st);                    // inter_size, d_model
    CHECK_INPUT(output_layernorm_gamma, _st);           // d_model
    CHECK_INPUT(pre_transformer_layernorm_gamma, _st);  // d_model
    if (mbart) {
        CHECK_INPUT(post_transformer_layernorm_gamma, _st);  // d_model
    }
    CHECK_INPUT(absolute_or_relative_position_embedding, _st);  // head_num, num_bucket or max_seq_len, d_model
    CHECK_INPUT(embedding_table, _st);                          // vocab_size, d_model
    if (bart_with_bias) {
        CHECK_INPUT(q_bias, _st);                          // hidden_dim
        CHECK_INPUT(k_bias, _st);                          // hidden_dim
        CHECK_INPUT(v_bias, _st);                          // hidden_dim
        CHECK_INPUT(attr_output_bias, _st);                // d_model
        CHECK_INPUT(attr_output_layernorm_beta, _st);      // d_model
        CHECK_INPUT(inter_bias, _st);                      // inter_size
        CHECK_INPUT(inter_bias2, _st);                     // inter_size
        CHECK_INPUT(output_bias, _st);                     // d_model
        CHECK_INPUT(output_layernorm_beta, _st);           // d_model
        CHECK_INPUT(pre_transformer_layernorm_beta, _st);  // d_model
        if (mbart) {
            CHECK_INPUT(post_transformer_layernorm_beta, _st);  // d_model
        }
    }

    switch (_st) {
        case at::ScalarType::Float:
            ft_bart_encoder = new FTBartEncoder<float>(head_num,
                                                       head_size,
                                                       inter_size,
                                                       d_model,
                                                       layer_num,
                                                       num_bucket,
                                                       max_distance,
                                                       sparse,
                                                       q_scaling,
                                                       tensor_para_size,
                                                       pipeline_para_size,
                                                       bart_with_bias,
                                                       mbart,
                                                       ft::PositionEmbeddingType(position_embedding_type),
                                                       ft::getActivationType(activation_type),
                                                       ft::getLayerNormType(layernorm_type),
                                                       weights);
            break;
        case at::ScalarType::Half:
            ft_bart_encoder = new FTBartEncoder<half>(head_num,
                                                      head_size,
                                                      inter_size,
                                                      d_model,
                                                      layer_num,
                                                      num_bucket,
                                                      max_distance,
                                                      sparse,
                                                      q_scaling,
                                                      tensor_para_size,
                                                      pipeline_para_size,
                                                      bart_with_bias,
                                                      mbart,
                                                      ft::PositionEmbeddingType(position_embedding_type),
                                                      ft::getActivationType(activation_type),
                                                      ft::getLayerNormType(layernorm_type),
                                                      weights);
            break;
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16:
            ft_bart_encoder = new FTBartEncoder<__nv_bfloat16>(head_num,
                                                               head_size,
                                                               inter_size,
                                                               d_model,
                                                               layer_num,
                                                               num_bucket,
                                                               max_distance,
                                                               sparse,
                                                               q_scaling,
                                                               tensor_para_size,
                                                               pipeline_para_size,
                                                               bart_with_bias,
                                                               mbart,
                                                               ft::PositionEmbeddingType(position_embedding_type),
                                                               ft::getActivationType(activation_type),
                                                               ft::getLayerNormType(layernorm_type),
                                                               weights);
            break;
#endif
        default:
            throw std::runtime_error("Wrong Tensor type.");
    }
}

FasterTransformerBartEncoder::~FasterTransformerBartEncoder()
{
    delete ft_bart_encoder;
}

th::Tensor FasterTransformerBartEncoder::forward(th::optional<th::Tensor> input_ids,
                                                 th::Tensor               sequence_lengths,
                                                 th::optional<th::Tensor> inputs_embeds)
{
    if (input_ids.has_value()) {
        CHECK_CONTIGUOUS(input_ids.value());
        TORCH_CHECK(input_ids.value().dtype() == torch::kInt32, "input_ids dtype should be int32");
    }

    CHECK_CONTIGUOUS(sequence_lengths);
    TORCH_CHECK(sequence_lengths.dtype() == torch::kInt32, "sequence_lengths dtype should be int32");

    if (inputs_embeds.has_value()) {
        CHECK_CONTIGUOUS(inputs_embeds.value());
        TORCH_CHECK(inputs_embeds.value().dtype() == torch::kFloat32
                        || inputs_embeds.value().dtype() == torch::kFloat16,
                    "inputs_embeds dtype should be float32 or float16");
    }

    TORCH_CHECK(input_ids.has_value() || inputs_embeds.has_value(),
                "input_ids and inputs_embeds should not be empty at the same time.");

    size_t  batch_size = inputs_embeds.has_value() ? inputs_embeds.value().size(0) : input_ids.value().size(0);
    size_t  seq_len    = inputs_embeds.has_value() ? inputs_embeds.value().size(1) : input_ids.value().size(1);
    int64_t d_model    = d_model_;

    auto output = torch::empty({(long int)batch_size, (long int)seq_len, (long int)d_model},
                               torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    ft_bart_encoder->forward(batch_size, seq_len, input_ids, sequence_lengths, inputs_embeds, output, _remove_padding);
    return output;
}

std::vector<th::Tensor> FasterTransformerBartEncoder::get_pickle_info() const
{
    std::vector<th::Tensor> tmp(weights);
    return tmp;
}

}  // namespace torch_ext

static auto fasterTransformerBartEncoderTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::FasterTransformerBartEncoder>("FasterTransformerBartEncoder")
#else
    torch::jit::class_<torch_ext::FasterTransformerBartEncoder>("FasterTransformer", "BartEncoder")
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
                              bool,
                              int64_t,
                              int64_t,
                              int64_t,
                              bool,
                              double,
                              int64_t,
                              int64_t,
                              bool,
                              bool,
                              int64_t,
                              std::string,
                              std::string>())
        .def("forward", &torch_ext::FasterTransformerBartEncoder::forward);
