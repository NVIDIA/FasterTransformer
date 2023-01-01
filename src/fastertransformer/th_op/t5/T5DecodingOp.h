/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
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
    virtual ~IFTT5Decoding() {}
    virtual std::vector<th::Tensor> forward(th::optional<int64_t>    beam_width_opt,
                                            size_t                   max_seq_len,
                                            th::optional<int64_t>    top_k_opt,
                                            th::optional<double>     top_p_opt,
                                            th::optional<double>     beam_search_diversity_rate_opt,
                                            th::optional<double>     temperature_opt,
                                            th::optional<double>     len_penalty_opt,
                                            th::optional<double>     repetition_penalty_opt,
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
                 const std::vector<th::Tensor>& w,
                 std::vector<int64_t>           moe_layer_index):
        head_num_(head_num),
        size_per_head_(size_per_head),
        inter_size_(inter_size),
        mem_d_model_(mem_d_model),
        d_model_(d_model),
        layer_num_(layer_num),
        vocab_size_(vocab_size),
        num_bucket_(num_bucket),
        expert_num_(expert_num),
        max_distance_(max_distance),
        q_scaling_(q_scaling),
        start_id_(start_id),
        end_id_(end_id),
        t5_with_bias_(t5_with_bias),
        moe_k_(moe_k),
        position_embedding_type_(position_embedding_type),
        moe_layer_index_(moe_layer_index),
        activation_type_(activation_type),
        tie_word_embeddings_(tie_word_embeddings),
        _weights(w)
    {
        bool use_gated_activation = isGatedActivation(activation_type_);
        ft::ftNcclInitialize(tensor_para_, pipeline_para_, tensor_para_size, pipeline_para_size);

        ft::check_cuda_error(cublasLtCreate(&cublasltHandle_));
        cublas_algo_map_      = new ft::cublasAlgoMap("gemm_config.in");
        cublas_wrapper_mutex_ = new std::mutex();

        decoding_weights.resizeLayer(layer_num_);
        decoding_weights.setT5StructureDiff(t5_with_bias, use_gated_activation, position_embedding_type);
        const int hidden_dim = head_num_ * size_per_head_;

        int dense_weight_index     = 0;  // the inter and out kernel has the same index
        int moe_dense_weight_index = 0;  // the moe inter and out kernel has the same index

        for (int i = 0; i < layer_num_; ++i) {
            int local_num_layer = (int)(ceil(layer_num_ * 1.0f / pipeline_para_.world_size_));
            if (!(i < layer_num_ && (i >= local_num_layer * pipeline_para_.rank_)
                  && (i < local_num_layer * (pipeline_para_.rank_ + 1)))) {
                continue;
            }
            const int first_layer_index = local_num_layer * pipeline_para_.rank_;

            decoding_weights.decoder_layer_weights[i]->pre_layernorm_weights.gamma =
                get_ptr<T>(_weights[0]) + (i - first_layer_index) * d_model;
            decoding_weights.decoder_layer_weights[i]->self_attention_weights.query_weight.kernel =
                get_ptr<T>(_weights[1]) + (i - first_layer_index) * d_model * 3 * hidden_dim / tensor_para_.world_size_;
            decoding_weights.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.kernel =
                get_ptr<T>(_weights[2]) + (i - first_layer_index) * hidden_dim / tensor_para_.world_size_ * d_model;
            decoding_weights.decoder_layer_weights[i]->self_attn_layernorm_weights.gamma =
                get_ptr<T>(_weights[3]) + (i - first_layer_index) * d_model;
            decoding_weights.decoder_layer_weights[i]->cross_attention_weights.query_weight.kernel =
                get_ptr<T>(_weights[4]) + (i - first_layer_index) * d_model * hidden_dim / tensor_para_.world_size_;
            decoding_weights.decoder_layer_weights[i]->cross_attention_weights.key_weight.kernel =
                get_ptr<T>(_weights[5]) + (i - first_layer_index) * mem_d_model * hidden_dim / tensor_para_.world_size_;
            decoding_weights.decoder_layer_weights[i]->cross_attention_weights.value_weight.kernel =
                get_ptr<T>(_weights[6]) + (i - first_layer_index) * mem_d_model * hidden_dim / tensor_para_.world_size_;
            decoding_weights.decoder_layer_weights[i]->cross_attention_weights.attention_output_weight.kernel =
                get_ptr<T>(_weights[7]) + (i - first_layer_index) * hidden_dim / tensor_para_.world_size_ * d_model;
            decoding_weights.decoder_layer_weights[i]->cross_attn_layernorm_weights.gamma =
                get_ptr<T>(_weights[8]) + (i - first_layer_index) * d_model;
            decoding_weights.decoder_layer_weights[i]->ffn_weights.intermediate_weight.kernel =
                get_ptr<T>(_weights[9]) + dense_weight_index * d_model * inter_size_ / tensor_para_.world_size_
                + moe_dense_weight_index * expert_num_ * d_model * inter_size_ / tensor_para_.world_size_;

            if (use_gated_activation) {
                decoding_weights.decoder_layer_weights[i]->ffn_weights.intermediate_weight2.kernel =
                    get_ptr<T>(_weights[10]) + dense_weight_index * d_model * inter_size_ / tensor_para_.world_size_;
            }

            decoding_weights.decoder_layer_weights[i]->ffn_weights.output_weight.kernel =
                get_ptr<T>(_weights[11]) + dense_weight_index * inter_size_ / tensor_para_.world_size_ * d_model
                + moe_dense_weight_index * expert_num_ * inter_size_ / tensor_para_.world_size_ * d_model;

            if (t5_with_bias_) {
                decoding_weights.decoder_layer_weights[i]->pre_layernorm_weights.beta =
                    get_ptr<T>(_weights[16]) + (i - first_layer_index) * d_model;
                decoding_weights.decoder_layer_weights[i]->self_attention_weights.query_weight.bias =
                    get_ptr<T>(_weights[17]) + (i - first_layer_index) * 3 * hidden_dim / tensor_para_.world_size_;
                decoding_weights.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.bias =
                    get_ptr<T>(_weights[18]) + (i - first_layer_index) * d_model;
                decoding_weights.decoder_layer_weights[i]->self_attn_layernorm_weights.beta =
                    get_ptr<T>(_weights[19]) + (i - first_layer_index) * d_model;
                decoding_weights.decoder_layer_weights[i]->cross_attention_weights.query_weight.bias =
                    get_ptr<T>(_weights[20]) + (i - first_layer_index) * hidden_dim / tensor_para_.world_size_;
                decoding_weights.decoder_layer_weights[i]->cross_attention_weights.key_weight.bias =
                    get_ptr<T>(_weights[21]) + (i - first_layer_index) * hidden_dim / tensor_para_.world_size_;
                decoding_weights.decoder_layer_weights[i]->cross_attention_weights.value_weight.bias =
                    get_ptr<T>(_weights[22]) + (i - first_layer_index) * hidden_dim / tensor_para_.world_size_;
                decoding_weights.decoder_layer_weights[i]->cross_attention_weights.attention_output_weight.bias =
                    get_ptr<T>(_weights[23]) + (i - first_layer_index) * d_model;
                decoding_weights.decoder_layer_weights[i]->cross_attn_layernorm_weights.beta =
                    get_ptr<T>(_weights[24]) + (i - first_layer_index) * d_model;
                decoding_weights.decoder_layer_weights[i]->ffn_weights.intermediate_weight.bias =
                    get_ptr<T>(_weights[25]) + dense_weight_index * inter_size_ / tensor_para_.world_size_
                    + moe_dense_weight_index * expert_num_ * inter_size_ / tensor_para_.world_size_;

                if (use_gated_activation) {
                    decoding_weights.decoder_layer_weights[i]->ffn_weights.intermediate_weight2.bias =
                        get_ptr<T>(_weights[26]) + (i - first_layer_index) * inter_size_ / tensor_para_.world_size_;
                }

                decoding_weights.decoder_layer_weights[i]->ffn_weights.output_weight.bias =
                    get_ptr<T>(_weights[27]) + dense_weight_index * d_model
                    + moe_dense_weight_index * expert_num_ * d_model;
            }
            if (std::find(moe_layer_index.begin(), moe_layer_index.end(), i) == moe_layer_index.end()) {
                dense_weight_index += 1;
            }

            if (std::find(moe_layer_index.begin(), moe_layer_index.end(), i) != moe_layer_index.end()) {
                decoding_weights.decoder_layer_weights[i]->ffn_weights.gating_weight.kernel =
                    get_ptr<T>(_weights[30]) + moe_dense_weight_index * d_model * expert_num_;
                moe_dense_weight_index += 1;
            }
        }
        decoding_weights.post_decoder_layernorm.gamma            = get_ptr<T>(_weights[12]);
        decoding_weights.pre_decoder_embedding_table             = get_ptr<T>(_weights[13]);
        decoding_weights.post_decoder_embedding.kernel           = get_ptr<T>(_weights[14]);
        decoding_weights.absolute_or_relative_position_embedding = get_ptr<T>(_weights[15]);
        if (t5_with_bias_) {
            decoding_weights.post_decoder_layernorm.beta = get_ptr<T>(_weights[28]);
            decoding_weights.post_decoder_embedding.bias = get_ptr<T>(_weights[29]);
        }
        int device_id = 0;
        ft::check_cuda_error(cudaGetDevice(&device_id));
        ft::check_cuda_error(cudaGetDeviceProperties(&prop_, device_id));
    }

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
                                    th::optional<int64_t>    random_seed_opt,
                                    th::Tensor               memory,
                                    th::Tensor               memory_seq_lens,
                                    th::optional<bool>       is_return_output_log_probs_opt,
                                    th::optional<bool>       is_return_cum_log_probs_opt,
                                    th::optional<bool>       is_return_cross_attentions_opt,
                                    th::optional<th::Tensor> bad_words_list_opt,
                                    th::optional<th::Tensor> stop_words_list_opt) override
    {
        // input validation
        size_t beam_width = beam_width_opt.has_value() ? (size_t)beam_width_opt.value() : 1;
        uint   top_k      = top_k_opt.has_value() ? (uint)top_k_opt.value() : 1;
        float  top_p      = top_p_opt.has_value() ? (float)top_p_opt.value() : 0.0f;
        float  beam_search_diversity_rate =
            beam_search_diversity_rate_opt.has_value() ? (float)beam_search_diversity_rate_opt.value() : 0.0f;
        float temperature        = temperature_opt.has_value() ? (float)temperature_opt.value() : 1.0f;
        float len_penalty        = len_penalty_opt.has_value() ? (float)len_penalty_opt.value() : 0.0f;
        float repetition_penalty = repetition_penalty_opt.has_value() ? (float)repetition_penalty_opt.value() : 1.0f;
        unsigned long long random_seed = random_seed_opt.has_value() ? (unsigned long long)random_seed_opt.value() : 0;
        bool               is_return_output_log_probs =
            is_return_output_log_probs_opt.has_value() ? (bool)is_return_output_log_probs_opt.value() : false;
        bool is_return_cum_log_probs =
            is_return_cum_log_probs_opt.has_value() ? (bool)is_return_cum_log_probs_opt.value() : false;
        bool is_return_cross_attentions =
            is_return_cross_attentions_opt.has_value() ? (bool)is_return_cross_attentions_opt.value() : false;

        auto           stream       = at::cuda::getCurrentCUDAStream().stream();
        cublasHandle_t cublasHandle = at::cuda::getCurrentCUDABlasHandle();
        cublasSetStream(cublasHandle, stream);
        ft::Allocator<ft::AllocatorType::TH> allocator      = ft::Allocator<ft::AllocatorType::TH>();
        ft::cublasMMWrapper                  cublas_wrapper = ft::cublasMMWrapper(
            cublasHandle, cublasltHandle_, stream, cublas_algo_map_, cublas_wrapper_mutex_, &allocator);

        if (std::is_same<T, half>::value) {
            cublas_wrapper.setFP16GemmConfig();
        }
#ifdef ENABLE_BF16
        else if (std::is_same<T, __nv_bfloat16>::value) {
            cublas_wrapper.setBF16GemmConfig();
        }
#endif
        else if (std::is_same<T, float>::value) {
            cublas_wrapper.setFP32GemmConfig();
        }

        const size_t batch_size      = (size_t)memory.size(0);
        const size_t mem_max_seq_len = (size_t)memory.size(1);

        ft::T5Decoding<T> decoding  = ft::T5Decoding<T>(batch_size,
                                                       max_seq_len,
                                                       mem_max_seq_len,
                                                       beam_width,
                                                       head_num_,
                                                       size_per_head_,
                                                       inter_size_,
                                                       d_model_,
                                                       layer_num_,
                                                       vocab_size_,
                                                       num_bucket_,
                                                       expert_num_,
                                                       max_distance_,
                                                       moe_k_,
                                                       q_scaling_,
                                                       start_id_,
                                                       end_id_,
                                                       beam_search_diversity_rate,
                                                       top_k,
                                                       top_p,
                                                       temperature,
                                                       len_penalty,
                                                       repetition_penalty,
                                                       moe_layer_index_,
                                                       stream,
                                                       &cublas_wrapper,
                                                       &allocator,
                                                       false,
                                                       &prop_,
                                                       tensor_para_,
                                                       pipeline_para_,
                                                       activation_type_,
                                                       tie_word_embeddings_);
        ft::DataType      data_type = ft::getTensorType<T>();

        ft::TensorMap input_tensors(
            {{"encoder_output",
              ft::Tensor{ft::MEMORY_GPU,
                         data_type,
                         std::vector<size_t>{(size_t)memory.size(0), (size_t)memory.size(1), (size_t)memory.size(2)},
                         get_ptr<T>(memory)}},
             {"encoder_sequence_length",
              ft::Tensor{ft::MEMORY_GPU,
                         ft::TYPE_INT32,
                         std::vector<size_t>{(size_t)memory_seq_lens.size(0)},
                         get_ptr<T>(memory_seq_lens)}}});

        if (beam_width > 1) {
            input_tensors.insert(
                {"beam_search_diversity_rate",
                 ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &beam_search_diversity_rate}});
        }
        if (top_p_opt.has_value()) {
            input_tensors.insert(
                {"runtime_top_p", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &top_p}});
        }
        if (top_k_opt.has_value()) {
            input_tensors.insert(
                {"runtime_top_k", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_UINT32, std::vector<size_t>{1}, &top_k}});
        }
        if (temperature_opt.has_value()) {
            input_tensors.insert(
                {"temperature", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &temperature}});
        }
        if (len_penalty_opt.has_value()) {
            input_tensors.insert(
                {"len_penalty", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &len_penalty}});
        }
        if (repetition_penalty_opt.has_value()) {
            input_tensors.insert(
                {"repetition_penalty",
                 ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &repetition_penalty}});
        }
        if (random_seed_opt.has_value()) {
            input_tensors.insert(
                {"random_seed", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_UINT64, std::vector<size_t>{1}, &random_seed}});
        }
        if (stop_words_list_opt.has_value()) {
            input_tensors.insert({"stop_words_list", convert_tensor<int>(stop_words_list_opt.value())});
        }
        if (bad_words_list_opt.has_value()) {
            input_tensors.insert({"bad_words_list", convert_tensor<int>(bad_words_list_opt.value())});
        }

        auto output_ids      = torch::empty({(long int)(batch_size * beam_width * max_seq_len)},
                                       torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
        auto sequence_length = torch::empty({(long int)(batch_size * beam_width)},
                                            torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

        std::vector<th::Tensor> th_output_tensors = {output_ids, sequence_length};

        ft::TensorMap output_tensors({{"output_ids",
                                       ft::Tensor{ft::MEMORY_GPU,
                                                  ft::TYPE_INT32,
                                                  std::vector<size_t>{batch_size, beam_width, max_seq_len},
                                                  get_ptr<int>(output_ids)}},
                                      {"sequence_length",
                                       ft::Tensor{ft::MEMORY_GPU,
                                                  ft::TYPE_INT32,
                                                  std::vector<size_t>{batch_size, beam_width},
                                                  get_ptr<int>(sequence_length)}}});

        if (is_return_output_log_probs) {
             auto output_log_probs = torch::empty({batch_size, beam_width, max_seq_len},
                                                torch::dtype(torch::kFloat).device(torch::kCUDA).requires_grad(false));
            output_tensors.insert({"output_log_probs",
                                   ft::Tensor{ft::MEMORY_GPU,
                                              ft::TYPE_FP32,
                                              {batch_size, beam_width, max_seq_len},
                                              get_ptr<float>(output_log_probs)}});
            th_output_tensors.push_back(output_log_probs);
        }
        if (is_return_cum_log_probs) {
            auto cum_log_probs = torch::empty({(long int)(batch_size * beam_width)},
                                              torch::dtype(torch::kFloat).device(torch::kCUDA).requires_grad(false));
            output_tensors.insert(
                {"cum_log_probs",
                 ft::Tensor{ft::MEMORY_GPU, ft::TYPE_FP32, {batch_size, beam_width}, get_ptr<float>(cum_log_probs)}});
            th_output_tensors.push_back(cum_log_probs);
        }
        if (is_return_cross_attentions) {
            auto cross_attentions =
                torch::empty({(long int)(ceil(layer_num_ * 1.0 / pipeline_para_.world_size_) * batch_size * beam_width
                                         * (head_num_ / tensor_para_.world_size_) * max_seq_len * mem_max_seq_len)},
                             torch::dtype(torch::kFloat).device(torch::kCUDA).requires_grad(false));
            output_tensors.insert({"cross_attentions",
                                   ft::Tensor{ft::MEMORY_GPU,
                                              ft::TYPE_FP32,
                                              {(size_t)(layer_num_ / pipeline_para_.world_size_),
                                               (size_t)batch_size,
                                               (size_t)beam_width,
                                               (size_t)(head_num_ / tensor_para_.world_size_),
                                               (size_t)max_seq_len,
                                               (size_t)mem_max_seq_len},
                                              get_ptr<float>(cross_attentions)}});
            th_output_tensors.push_back(cross_attentions);
        }

        decoding.forward(&output_tensors, &input_tensors, &decoding_weights);
        return th_output_tensors;
    }

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
                                std::vector<int64_t> moe_layer_index,
                                th::Tensor           self_layernorm_gamma,
                                th::Tensor           self_kernel_qkv,
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
                                th::Tensor           decoding_gamma,
                                th::Tensor           embedding_table,
                                th::Tensor           lm_head,
                                th::Tensor           absolute_or_relative_position_embedding,
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
                                th::Tensor           decoding_beta,
                                th::Tensor           embedding_bias,
                                th::Tensor           moe_gate);

    ~FasterTransformerT5Decoding();

    std::vector<th::Tensor> forward(th::optional<int64_t>    beam_width,
                                    int64_t                  max_seq_len,
                                    th::optional<int64_t>    top_k,
                                    th::optional<double>     top_p,
                                    th::optional<double>     beam_search_diversity_rate,
                                    th::optional<double>     temperature,
                                    th::optional<double>     len_penalty,
                                    th::optional<double>     repetition_penalty,
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
