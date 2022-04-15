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

#include "src/fastertransformer/models/decoding/Decoding.h"
#include "src/fastertransformer/th_op/th_utils.h"

namespace ft = fastertransformer;
namespace th = torch;

namespace torch_ext {

class IFTDecoding {
public:
    virtual ~IFTDecoding() {}
    virtual void forward(size_t beam_width,
                         size_t max_seq_len,
                         th::Tensor memory,
                         th::Tensor memory_seq_lens,
                         th::Tensor output_ids,
                         th::Tensor parent_ids,
                         th::Tensor out_seq_lens) = 0;
};

template<typename T>
class FTDecoding: public IFTDecoding {
public:
    FTDecoding(int head_num,
               int size_per_head,
               int inter_size,
               int mem_hidden_dim,
               int layer_num,
               int vocab_size,
               int start_id,
               int end_id,
               float beam_search_diversity_rate,
               int top_k,
               float top_p,
               float temperature,
               float len_penalty,
               float repetition_penalty,
               const std::vector<th::Tensor>& w):
        head_num_(head_num),
        size_per_head_(size_per_head),
        inter_size_(inter_size),
        mem_hidden_dim_(mem_hidden_dim),
        layer_num_(layer_num),
        vocab_size_(vocab_size),
        start_id_(start_id),
        end_id_(end_id),
        beam_search_diversity_rate_(beam_search_diversity_rate),
        top_k_(top_k),
        top_p_(top_p),
        temperature_(temperature),
        len_penalty_(len_penalty),
        repetition_penalty_(repetition_penalty),
        _weights(w)
    {
        ft::check_cuda_error(cublasLtCreate(&cublasltHandle_));
        cublas_algo_map_ = new ft::cublasAlgoMap("gemm_config.in");
        cublas_wrapper_mutex_ = new std::mutex();

        decoding_weights.decoder_layer_weights.resize(layer_num_);
        const int hidden_dim = head_num_ * size_per_head_;

        for (int i = 0; i < layer_num_; ++i) {
            decoding_weights.decoder_layer_weights[i].pre_layernorm_weights.gamma =
                get_ptr<T>(_weights[0]) + i * hidden_dim;
            decoding_weights.decoder_layer_weights[i].pre_layernorm_weights.beta =
                get_ptr<T>(_weights[1]) + i * hidden_dim;
            decoding_weights.decoder_layer_weights[i].self_attention_weights.query_weight.kernel =
                get_ptr<T>(_weights[2]) + i * hidden_dim * 3 * hidden_dim;
            decoding_weights.decoder_layer_weights[i].self_attention_weights.query_weight.bias =
                get_ptr<T>(_weights[3]) + i * 3 * hidden_dim;
            decoding_weights.decoder_layer_weights[i].self_attention_weights.attention_output_weight.kernel =
                get_ptr<T>(_weights[4]) + i * hidden_dim * hidden_dim;
            decoding_weights.decoder_layer_weights[i].self_attention_weights.attention_output_weight.bias =
                get_ptr<T>(_weights[5]) + i * hidden_dim;
            decoding_weights.decoder_layer_weights[i].self_attn_layernorm_weights.gamma =
                get_ptr<T>(_weights[6]) + i * hidden_dim;
            decoding_weights.decoder_layer_weights[i].self_attn_layernorm_weights.beta =
                get_ptr<T>(_weights[7]) + i * hidden_dim;
            decoding_weights.decoder_layer_weights[i].cross_attention_weights.query_weight.kernel =
                get_ptr<T>(_weights[8]) + i * hidden_dim * hidden_dim;
            decoding_weights.decoder_layer_weights[i].cross_attention_weights.key_weight.kernel =
                get_ptr<T>(_weights[9]) + i * mem_hidden_dim * hidden_dim;
            decoding_weights.decoder_layer_weights[i].cross_attention_weights.value_weight.kernel =
                get_ptr<T>(_weights[10]) + i * mem_hidden_dim * hidden_dim;
            decoding_weights.decoder_layer_weights[i].cross_attention_weights.query_weight.bias =
                get_ptr<T>(_weights[11]) + i * hidden_dim;
            decoding_weights.decoder_layer_weights[i].cross_attention_weights.key_weight.bias =
                get_ptr<T>(_weights[12]) + i * hidden_dim;
            decoding_weights.decoder_layer_weights[i].cross_attention_weights.value_weight.bias =
                get_ptr<T>(_weights[13]) + i * hidden_dim;
            decoding_weights.decoder_layer_weights[i].cross_attention_weights.attention_output_weight.kernel =
                get_ptr<T>(_weights[14]) + i * hidden_dim * hidden_dim;
            decoding_weights.decoder_layer_weights[i].cross_attention_weights.attention_output_weight.bias =
                get_ptr<T>(_weights[15]) + i * hidden_dim;
            decoding_weights.decoder_layer_weights[i].cross_attn_layernorm_weights.gamma =
                get_ptr<T>(_weights[16]) + i * hidden_dim;
            decoding_weights.decoder_layer_weights[i].cross_attn_layernorm_weights.beta =
                get_ptr<T>(_weights[17]) + i * hidden_dim;
            decoding_weights.decoder_layer_weights[i].ffn_weights.intermediate_weight.kernel =
                get_ptr<T>(_weights[18]) + i * hidden_dim * inter_size_;
            decoding_weights.decoder_layer_weights[i].ffn_weights.intermediate_weight.bias =
                get_ptr<T>(_weights[19]) + i * inter_size_;
            decoding_weights.decoder_layer_weights[i].ffn_weights.output_weight.kernel =
                get_ptr<T>(_weights[20]) + i * hidden_dim * inter_size_;
            decoding_weights.decoder_layer_weights[i].ffn_weights.output_weight.bias =
                get_ptr<T>(_weights[21]) + i * hidden_dim;
        }
        decoding_weights.post_decoder_layernorm.gamma = get_ptr<T>(_weights[22]);
        decoding_weights.post_decoder_layernorm.beta = get_ptr<T>(_weights[23]);
        decoding_weights.pre_decoder_embedding_table = get_ptr<T>(_weights[24]);
        decoding_weights.position_encoding_table = get_ptr<T>(_weights[25]);
        decoding_weights.post_decoder_embedding.kernel = get_ptr<T>(_weights[26]);
        decoding_weights.post_decoder_embedding.bias = get_ptr<T>(_weights[27]);

        ft::check_cuda_error(cudaGetDeviceProperties(&prop_, 0));
    }

    ~FTDecoding() override
    {
        cublasLtDestroy(cublasltHandle_);
        delete cublas_algo_map_;
        delete cublas_wrapper_mutex_;
    }

    void forward(size_t beam_width,
                 size_t max_seq_len,
                 th::Tensor memory,
                 th::Tensor memory_seq_lens,
                 th::Tensor output_ids,
                 th::Tensor parent_ids,
                 th::Tensor sequence_lengths) override
    {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        cublasHandle_t cublasHandle = at::cuda::getCurrentCUDABlasHandle();
        cublasSetStream(cublasHandle, stream);
        ft::Allocator<ft::AllocatorType::TH> allocator = ft::Allocator<ft::AllocatorType::TH>();
        ft::cublasMMWrapper cublas_wrapper = ft::cublasMMWrapper(
            cublasHandle, cublasltHandle_, stream, cublas_algo_map_, cublas_wrapper_mutex_, &allocator);

        if (std::is_same<T, half>::value) {
            cublas_wrapper.setFP16GemmConfig();
        }
        else if (std::is_same<T, float>::value) {
            cublas_wrapper.setFP32GemmConfig();
        }

        const size_t batch_size = (size_t)memory.size(0) / beam_width;
        const size_t mem_max_seq_len = (size_t)memory.size(1);

        ft::Decoding<T> decoding = ft::Decoding<T>(batch_size,
                                                   max_seq_len,
                                                   mem_max_seq_len,
                                                   beam_width,
                                                   head_num_,
                                                   size_per_head_,
                                                   inter_size_,
                                                   layer_num_,
                                                   vocab_size_,
                                                   start_id_,
                                                   end_id_,
                                                   beam_search_diversity_rate_,
                                                   top_k_,
                                                   top_p_,
                                                   temperature_,
                                                   len_penalty_,
                                                   repetition_penalty_,
                                                   stream,
                                                   &cublas_wrapper,
                                                   &allocator,
                                                   false,
                                                   &prop_);
        ft::DataType data_type = ft::getTensorType<T>();
        std::vector<ft::Tensor> input_tensors = std::vector<ft::Tensor>{
            ft::Tensor{ft::MEMORY_GPU,
                       data_type,
                       std::vector<size_t>{(size_t)memory.size(0), (size_t)memory.size(1), (size_t)memory.size(2)},
                       get_ptr<T>(memory)},
            ft::Tensor{ft::MEMORY_GPU,
                       ft::TYPE_INT32,
                       std::vector<size_t>{(size_t)memory_seq_lens.size(0)},
                       get_ptr<T>(memory_seq_lens)}};

        std::vector<ft::Tensor> output_tensors =
            std::vector<ft::Tensor>{ft::Tensor{ft::MEMORY_GPU,
                                               ft::TYPE_INT32,
                                               std::vector<size_t>{max_seq_len, batch_size, beam_width},
                                               get_ptr<int>(output_ids)},
                                    ft::Tensor{ft::MEMORY_GPU,
                                               ft::TYPE_INT32,
                                               std::vector<size_t>{max_seq_len, batch_size, beam_width},
                                               get_ptr<int>(parent_ids)},
                                    ft::Tensor{ft::MEMORY_GPU,
                                               ft::TYPE_INT32,
                                               std::vector<size_t>{batch_size, beam_width},
                                               get_ptr<int>(sequence_lengths)}};
        decoding.forward(&output_tensors, &input_tensors, &decoding_weights);
    }

private:
    const int head_num_;
    const int size_per_head_;
    const int inter_size_;
    const int mem_hidden_dim_;
    const int layer_num_;
    const int vocab_size_;
    const int start_id_;
    const int end_id_;
    const float beam_search_diversity_rate_;
    const int top_k_;
    const float top_p_;
    const float temperature_;
    const float len_penalty_;
    const float repetition_penalty_;

    std::vector<th::Tensor> _weights;
    cublasLtHandle_t cublasltHandle_;
    std::mutex* cublas_wrapper_mutex_;
    ft::cublasAlgoMap* cublas_algo_map_;
    struct cudaDeviceProp prop_;
    ft::DecodingWeight<T> decoding_weights;
};

class FasterTransformerDecoding: public torch::jit::CustomClassHolder {
public:
    FasterTransformerDecoding(int64_t head_num,
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
                              th::Tensor embedding_bias);

    ~FasterTransformerDecoding();

    std::vector<th::Tensor>
    forward(int64_t beam_width, int64_t max_seq_len, th::Tensor memory, th::Tensor memory_seq_lens);

    std::vector<th::Tensor> get_pickle_info() const;

private:
    const at::ScalarType _st;
    torch_ext::IFTDecoding* ftdecoding;
    th::Tensor int_info_;
    th::Tensor float_info_;
    std::vector<th::Tensor> weights;
};

}  // namespace torch_ext