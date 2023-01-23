/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/t5/T5Decoding.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/nccl_utils.h"

namespace ft = fastertransformer;
namespace th = torch;

namespace torch_ext {

class IFTT5Decoding {
public:
    virtual ~IFTT5Decoding()                                                          = default;
    virtual std::vector<th::Tensor> forward(th::optional<int64_t>    beam_width_opt,
                                            size_t                   max_seq_len,
                                            th::optional<int64_t>    top_k_opt,
                                            th::optional<double>     top_p_opt,
                                            th::optional<double>     beam_search_diversity_rate_opt,
                                            th::optional<double>     temperature_opt,
                                            th::optional<double>     len_penalty_opt,
                                            th::optional<double>     repetition_penalty_opt,
                                            th::optional<double>     presence_penalty_opt,
                                            th::optional<int64_t>    min_length_opt,
                                            th::optional<int64_t>    random_seed_opt,
                                            th::Tensor               memory,
                                            th::Tensor               memory_seq_lens,
                                            th::optional<bool>       is_return_output_log_probs_opt,
                                            th::optional<bool>       is_return_cum_log_probs_opt,
                                            th::optional<bool>       is_return_cross_attentions_opt,
                                            th::optional<th::Tensor> bad_words_list,
                                            th::optional<th::Tensor> stop_words_list) = 0;
};

template<typename T>
class FTT5Decoding: public IFTT5Decoding {
public:
    FTT5Decoding(int64_t                        head_num,
                 int64_t                        size_per_head,
                 int64_t                        inter_size,
                 int64_t                        mem_d_model,
                 int64_t                        d_model,
                 int64_t                        layer_num,
                 int64_t                        vocab_size,
                 int64_t                        num_bucket,
                 int64_t                        expert_num,
                 int64_t                        max_distance,
                 double                         q_scaling,
                 int64_t                        start_id,
                 int64_t                        end_id,
                 int64_t                        tensor_para_size,
                 int64_t                        pipeline_para_size,
                 bool                           t5_with_bias,
                 int64_t                        moe_k,
                 ft::PositionEmbeddingType      position_embedding_type,
                 ft::ActivationType             activation_type,
                 bool                           tie_word_embeddings,
                 int64_t                        adapter_inter_size,
                 ft::LayerNormType              adapter_layer_norm_type,
                 std::vector<int64_t>           moe_layer_index,
                 const std::vector<th::Tensor>& w);

    ~FTT5Decoding() override
    {
        ft::ftNcclParamDestroy(tensor_para_);
        ft::ftNcclParamDestroy(pipeline_para_);
        cublasLtDestroy(cublasltHandle_);
        delete cublas_algo_map_;
        delete cublas_wrapper_mutex_;
    }

    std::vector<th::Tensor> forward(th::optional<int64_t>    beam_width_opt,
                                    size_t                   max_seq_len,
                                    th::optional<int64_t>    top_k_opt,
                                    th::optional<double>     top_p_opt,
                                    th::optional<double>     beam_search_diversity_rate_opt,
                                    th::optional<double>     temperature_opt,
                                    th::optional<double>     len_penalty_opt,
                                    th::optional<double>     repetition_penalty_opt,
                                    th::optional<double>     presence_penalty_opt,
                                    th::optional<int64_t>    min_length_opt,
                                    th::optional<int64_t>    random_seed_opt,
                                    th::Tensor               memory,
                                    th::Tensor               memory_seq_lens,
                                    th::optional<bool>       is_return_output_log_probs_opt,
                                    th::optional<bool>       is_return_cum_log_probs_opt,
                                    th::optional<bool>       is_return_cross_attentions_opt,
                                    th::optional<th::Tensor> bad_words_list_opt,
                                    th::optional<th::Tensor> stop_words_list_opt) override;

private:
    const int64_t                   head_num_;
    const int64_t                   size_per_head_;
    const int64_t                   inter_size_;
    const int64_t                   mem_d_model_;
    const int64_t                   d_model_;
    const int64_t                   layer_num_;
    const int64_t                   vocab_size_;
    const int64_t                   num_bucket_;
    const int64_t                   expert_num_;
    const int64_t                   max_distance_;
    double                          q_scaling_;
    const int64_t                   start_id_;
    const int64_t                   end_id_;
    const int64_t                   moe_k_;
    const bool                      t5_with_bias_;
    const ft::PositionEmbeddingType position_embedding_type_;
    const ft::ActivationType        activation_type_;
    const bool                      tie_word_embeddings_;
    int64_t                         adapter_inter_size_;
    ft::LayerNormType               adapter_layer_norm_type_;
    std::vector<int64_t>            moe_layer_index_;

    std::vector<th::Tensor> _weights;
    cublasLtHandle_t        cublasltHandle_;
    std::mutex*             cublas_wrapper_mutex_;
    ft::cublasAlgoMap*      cublas_algo_map_;
    struct cudaDeviceProp   prop_;
    ft::T5DecodingWeight<T> decoding_weights;

    ft::NcclParam tensor_para_;
    ft::NcclParam pipeline_para_;
};

class FasterTransformerT5Decoding: public torch::jit::CustomClassHolder {
public:
    FasterTransformerT5Decoding(int64_t              head_num,
                                int64_t              size_per_head,
                                int64_t              inter_size,
                                int64_t              mem_d_model,
                                int64_t              d_model,
                                int64_t              layer_num,
                                int64_t              vocab_size,
                                int64_t              num_bucket,
                                int64_t              expert_num,
                                int64_t              max_distance,
                                double               q_scaling,
                                int64_t              start_id,
                                int64_t              end_id,
                                int64_t              tensor_para_size,
                                int64_t              pipeline_para_size,
                                bool                 t5_with_bias,
                                int64_t              position_embedding_type,
                                int64_t              moe_k,
                                std::string          activaiton_type,
                                bool                 tie_word_embeddings,
                                int64_t              adapter_inter_size,
                                std::string          adapter_norm_position,
                                std::vector<int64_t> moe_layer_index,
                                th::Tensor           self_layernorm_gamma,                     // 0
                                th::Tensor           self_kernel_qkv,                          // 1
                                th::Tensor           self_output_kernel,                       // 2
                                th::Tensor           cross_layernorm_gamma,                    // 3
                                th::Tensor           cross_kernel_q,                           // 4
                                th::Tensor           cross_kernel_k,                           // 5
                                th::Tensor           cross_kernel_v,                           // 6
                                th::Tensor           cross_output_kernel,                      // 7
                                th::Tensor           ffn_layernorm_gamma,                      // 8
                                th::Tensor           inter_kernel,                             // 9
                                th::Tensor           inter_kernel2,                            // 10
                                th::Tensor           output_kernel,                            // 11
                                th::Tensor           decoding_gamma,                           // 12
                                th::Tensor           embedding_table,                          // 13
                                th::Tensor           lm_head,                                  // 14
                                th::Tensor           absolute_or_relative_position_embedding,  // 15
                                th::Tensor           self_layernorm_beta,                      // 16
                                th::Tensor           self_bias_qkv,                            // 17
                                th::Tensor           self_output_bias,                         // 18
                                th::Tensor           cross_layernorm_beta,                     // 19
                                th::Tensor           cross_bias_q,                             // 20
                                th::Tensor           cross_bias_k,                             // 21
                                th::Tensor           cross_bias_v,                             // 22
                                th::Tensor           cross_output_bias,                        // 23
                                th::Tensor           ffn_layernorm_beta,                       // 24
                                th::Tensor           inter_bias,                               // 25
                                th::Tensor           inter_bias2,                              // 26
                                th::Tensor           output_bias,                              // 27
                                th::Tensor           decoding_beta,                            // 28
                                th::Tensor           embedding_bias,                           // 29
                                th::Tensor           moe_gate,                                 // 30
                                th::Tensor           after_attn_adapter_weight_in,             // 31
                                th::Tensor           after_attn_adapter_weight_out,            // 32
                                th::Tensor           after_attn_adapter_layernorm_gamma,       // 33
                                th::Tensor           after_attn_adapter_layernorm_beta,        // 34
                                th::Tensor           after_ffn_adapter_weight_in,              // 35
                                th::Tensor           after_ffn_adapter_weight_out,             // 36
                                th::Tensor           after_ffn_adapter_layernorm_gamma,        // 37
                                th::Tensor           after_ffn_adapter_layernorm_beta          // 38
    );

    ~FasterTransformerT5Decoding();

    std::vector<th::Tensor> forward(th::optional<int64_t>    beam_width,
                                    int64_t                  max_seq_len,
                                    th::optional<int64_t>    top_k,
                                    th::optional<double>     top_p,
                                    th::optional<double>     beam_search_diversity_rate,
                                    th::optional<double>     temperature,
                                    th::optional<double>     len_penalty,
                                    th::optional<double>     repetition_penalty,
                                    th::optional<double>     presence_penalty,
                                    th::optional<int64_t>    min_length,
                                    th::optional<int64_t>    random_seed,
                                    th::Tensor               memory,
                                    th::Tensor               memory_seq_lens,
                                    th::optional<bool>       is_return_output_log_probs,
                                    th::optional<bool>       is_return_cum_log_probs,
                                    th::optional<bool>       is_return_cross_attentions,
                                    th::optional<th::Tensor> bad_words_list,
                                    th::optional<th::Tensor> stop_words_list);

    std::vector<th::Tensor> get_pickle_info() const;

private:
    const at::ScalarType      _st;
    torch_ext::IFTT5Decoding* ftdecoding;
    std::vector<th::Tensor>   weights;
};

}  // namespace torch_ext
