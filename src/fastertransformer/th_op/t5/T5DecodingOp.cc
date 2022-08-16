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

FasterTransformerT5Decoding::FasterTransformerT5Decoding(int64_t     head_num,
                                                         int64_t     size_per_head,
                                                         int64_t     inter_size,
                                                         int64_t     mem_d_model,
                                                         int64_t     d_model,
                                                         int64_t     layer_num,
                                                         int64_t     vocab_size,
                                                         int64_t     num_bucket,
                                                         int64_t     max_distance,
                                                         double      q_scaling,
                                                         int64_t     start_id,
                                                         int64_t     end_id,
                                                         int64_t     tensor_para_size,
                                                         int64_t     pipeline_para_size,
                                                         bool        t5_with_bias,
                                                         int64_t     position_embedding_type,
                                                         std::string activation_type,
                                                         bool        tie_word_embeddings,
                                                         th::Tensor  self_layernorm_gamma,
                                                         th::Tensor  self_kernel_q,
                                                         th::Tensor  self_output_kernel,
                                                         th::Tensor  cross_layernorm_gamma,
                                                         th::Tensor  cross_kernel_q,
                                                         th::Tensor  cross_kernel_k,
                                                         th::Tensor  cross_kernel_v,
                                                         th::Tensor  cross_output_kernel,
                                                         th::Tensor  ffn_layernorm_gamma,
                                                         th::Tensor  inter_kernel,
                                                         th::Tensor  inter_kernel2,
                                                         th::Tensor  output_kernel,
                                                         th::Tensor  decoding_gamma,
                                                         th::Tensor  embedding_table,
                                                         th::Tensor  lm_head,
                                                         th::Tensor  absolute_or_relative_position_embedding,
                                                         th::Tensor  self_layernorm_beta,
                                                         th::Tensor  self_bias_qkv,
                                                         th::Tensor  self_output_bias,
                                                         th::Tensor  cross_layernorm_beta,
                                                         th::Tensor  cross_bias_q,
                                                         th::Tensor  cross_bias_k,
                                                         th::Tensor  cross_bias_v,
                                                         th::Tensor  cross_output_bias,
                                                         th::Tensor  ffn_layernorm_beta,
                                                         th::Tensor  inter_bias,
                                                         th::Tensor  inter_bias2,
                                                         th::Tensor  output_bias,
                                                         th::Tensor  decoding_beta,
                                                         th::Tensor  embedding_bias):
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
            inter_kernel2,
            output_kernel,
            decoding_gamma,
            embedding_table,
            lm_head,
            absolute_or_relative_position_embedding,
            self_layernorm_beta,
            self_bias_qkv,
            self_output_bias,
            cross_layernorm_beta,
            cross_bias_q,
            cross_bias_k,
            cross_bias_v,
            cross_output_bias,
            ffn_layernorm_beta,
            inter_bias,
            inter_bias2,
            output_bias,
            decoding_beta,
            embedding_bias}
{
    CHECK_INPUT(self_layernorm_gamma, _st);                     // layer_num, d_model
    CHECK_INPUT(self_kernel_q, _st);                            // layer_num, d_model, 3 * hidden_dim
    CHECK_INPUT(self_output_kernel, _st);                       // layer_num, hidden_dim, d_model
    CHECK_INPUT(cross_layernorm_gamma, _st);                    // layer_num, d_model
    CHECK_INPUT(cross_kernel_q, _st);                           // layer_num, d_model, hidden_dim
    CHECK_INPUT(cross_kernel_k, _st);                           // layer_num, mem_d_model, hidden_dim
    CHECK_INPUT(cross_kernel_v, _st);                           // layer_num, mem_d_model, hidden_dim
    CHECK_INPUT(cross_output_kernel, _st);                      // layer_num, hidden_dim, d_model
    CHECK_INPUT(ffn_layernorm_gamma, _st);                      // layer_num, d_model
    CHECK_INPUT(inter_kernel, _st);                             // layer_num, d_model, inter_size
    CHECK_INPUT(inter_kernel2, _st);                            // layer_num, d_model, inter_size
    CHECK_INPUT(output_kernel, _st);                            // layer_num, inter_size, d_model
    CHECK_INPUT(decoding_gamma, _st);                           // d_model
    CHECK_INPUT(embedding_table, _st);                          // vocab_size, d_model
    CHECK_INPUT(lm_head, _st);                                  // d_model, vocab_size
    CHECK_INPUT(absolute_or_relative_position_embedding, _st);  // head_num, num_bucket or max_seq_len, d_model
    if (t5_with_bias) {
        CHECK_INPUT(self_layernorm_beta, _st);   // layer_num, d_model
        CHECK_INPUT(self_bias_qkv, _st);         // layer_num,3 * hidden_dim
        CHECK_INPUT(self_output_bias, _st);      // layer_num, d_model
        CHECK_INPUT(cross_layernorm_beta, _st);  // layer_num, d_model
        CHECK_INPUT(cross_bias_q, _st);          // layer_num, hidden_dim
        CHECK_INPUT(cross_bias_k, _st);          // layer_num, hidden_dim
        CHECK_INPUT(cross_bias_v, _st);          // layer_num, hidden_dim
        CHECK_INPUT(cross_output_bias, _st);     // layer_num, d_model
        CHECK_INPUT(ffn_layernorm_beta, _st);    // layer_num, d_model
        CHECK_INPUT(inter_bias, _st);            // layer_num, inter_size
        CHECK_INPUT(inter_bias2, _st);           // layer_num, inter_size
        CHECK_INPUT(output_bias, _st);           // layer_num, d_model
        CHECK_INPUT(decoding_beta, _st);         // d_model
        CHECK_INPUT(embedding_bias, _st);        // vocab_size
    }
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
                                                            q_scaling,
                                                            start_id,
                                                            end_id,
                                                            tensor_para_size,
                                                            pipeline_para_size,
                                                            t5_with_bias,
                                                            ft::PositionEmbeddingType(position_embedding_type),
                                                            ft::getActivationType(activation_type),
                                                            tie_word_embeddings,
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
                                                           q_scaling,
                                                           start_id,
                                                           end_id,
                                                           tensor_para_size,
                                                           pipeline_para_size,
                                                           t5_with_bias,
                                                           ft::PositionEmbeddingType(position_embedding_type),
                                                           ft::getActivationType(activation_type),
                                                           tie_word_embeddings,
                                                           weights);
            break;
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16:
            ftdecoding = new torch_ext::FTT5Decoding<__nv_bfloat16>(head_num,
                                                                    size_per_head,
                                                                    inter_size,
                                                                    mem_d_model,
                                                                    d_model,
                                                                    layer_num,
                                                                    vocab_size,
                                                                    num_bucket,
                                                                    max_distance,
                                                                    q_scaling,
                                                                    start_id,
                                                                    end_id,
                                                                    tensor_para_size,
                                                                    pipeline_para_size,
                                                                    t5_with_bias,
                                                                    ft::PositionEmbeddingType(position_embedding_type),
                                                                    ft::getActivationType(activation_type),
                                                                    tie_word_embeddings,
                                                                    weights);
            break;
#endif
        default:
            throw std::runtime_error("Wrong Tensor type.");
    }
}

FasterTransformerT5Decoding::~FasterTransformerT5Decoding()
{
    delete ftdecoding;
}

std::vector<th::Tensor> FasterTransformerT5Decoding::forward(th::optional<int64_t> beam_width,
                                                             int64_t               max_seq_len,
                                                             th::optional<int64_t> top_k,
                                                             th::optional<double>  top_p,
                                                             th::optional<double>  beam_search_diversity_rate,
                                                             th::optional<double>  temperature,
                                                             th::optional<double>  len_penalty,
                                                             th::optional<double>  repetition_penalty,
                                                             th::optional<int64_t> random_seed,
                                                             th::optional<bool>    is_return_output_log_probs,
                                                             th::optional<bool>    is_return_cum_log_probs,
                                                             th::optional<bool>    is_return_cross_attentions,
                                                             th::Tensor            memory,
                                                             th::Tensor            memory_seq_lens)
{
    CHECK_INPUT(memory, _st);
    CHECK_TH_CUDA(memory_seq_lens);
    CHECK_CONTIGUOUS(memory_seq_lens);
    TORCH_CHECK(memory_seq_lens.dtype() == torch::kInt32, "mem_seq_lens dtype should be int32");

    auto results = ftdecoding->forward(beam_width,
                                       (size_t)max_seq_len,
                                       top_k,
                                       top_p,
                                       beam_search_diversity_rate,
                                       temperature,
                                       len_penalty,
                                       repetition_penalty,
                                       random_seed,
                                       is_return_output_log_probs,
                                       is_return_cum_log_probs,
                                       is_return_cross_attentions,
                                       memory,
                                       memory_seq_lens);
    return results;
}

std::vector<th::Tensor> FasterTransformerT5Decoding::get_pickle_info() const
{
    std::vector<th::Tensor> tmp(weights);
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
                              double,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              bool,
                              int64_t,
                              std::string,
                              bool,
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
                              th::Tensor,
                              th::Tensor,
                              th::Tensor>())
        .def("forward", &torch_ext::FasterTransformerT5Decoding::forward);