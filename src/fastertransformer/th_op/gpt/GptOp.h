/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include "src/fastertransformer/models/gpt/Gpt.h"

#include <cuda_fp16.h>
#include <iostream>
#include <nvToolsExt.h>
#include <vector>

#include "torch/csrc/cuda/Stream.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/custom_class.h>
#include <torch/script.h>

#include "src/fastertransformer/th_op/th_traits.h"
#include "src/fastertransformer/th_op/th_utils.h"

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

using std::vector;

class IFGpt {
public:
    virtual ~IFGpt() {}
    virtual void forward(th::Tensor& input_ids,
                         th::Tensor& input_lengths,
                         th::Tensor& output_ids,
                         th::Tensor& parent_ids,
                         th::Tensor& sequence_lengths,
                         size_t request_output_len) = 0;
};

template<typename T>
class FTGpt: public IFGpt {
public:
    FTGpt(const size_t max_batch_size,
          const size_t max_seq_len,
          const size_t beam_width,
          const size_t head_num,
          const size_t size_per_head,
          const size_t inter_size,
          const size_t layer_num,
          const size_t vocab_size,
          const int start_id,
          const int end_id,
          const float beam_search_diversity_rate,
          const int top_k,
          const float top_p,
          const unsigned long long random_seed,
          const float temperature,
          const float len_penalty,
          const float repetition_penalty,
          const bool sparse,
          const vector<th::Tensor> weights):
        max_batch_size_(max_batch_size),
        max_seq_len_(max_seq_len),
        beam_width_(beam_width),
        head_num_(head_num),
        size_per_head_(size_per_head),
        inter_size_(inter_size),
        layer_num_(layer_num),
        vocab_size_(vocab_size),
        start_id_(start_id),
        end_id_(end_id),
        beam_search_diversity_rate_(beam_search_diversity_rate),
        top_k_(top_k),
        top_p_(top_p),
        random_seed_(random_seed),
        temperature_(temperature),
        len_penalty_(len_penalty),
        repetition_penalty_(repetition_penalty),
#ifndef SPARSITY_ENABLED
        sparse_(false),
#else
        sparse_(sparse),
#endif
        weights_(weights)
    {
        check_cuda_error(cublasLtCreate(&cublasltHandle_));
        if (sparse) {
#ifdef SPARSITY_ENABLED
            CHECK_CUSPARSE(cusparseLtInit(&cusparseLtHandle_));
#else
            std::cout
                << "[WARNING] Sparsity support is not enabled. Will use dense GEMM instead. "
                   "To enabled sparisty, please provide `-DSUPPORT_SPARITY` flag for compliation."
                << std::endl;
#endif
        }

        std::string sp_config_fname = sparse ? SPGEMM_CONFIG : "";
        cublas_algo_map_ = new ft::cublasAlgoMap(GEMM_CONFIG, sp_config_fname);
        cublas_wrapper_mutex_ = new std::mutex();

        gpt_weights_.decoder_layer_weights.resize(layer_num_);
        for (int i = 0; i < (int)layer_num_; i++) {
            gpt_weights_.decoder_layer_weights[i].pre_layernorm_weights.gamma =
                get_ptr<T>(weights_[i + 0 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i].pre_layernorm_weights.beta = get_ptr<T>(weights_[i + 1 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i].self_attention_weights.query_weight.kernel =
                get_ptr<T>(weights_[i + 2 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i].self_attention_weights.query_weight.bias =
                get_ptr<T>(weights_[i + 3 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i].self_attention_weights.attention_output_weight.kernel =
                get_ptr<T>(weights_[i + 4 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i].self_attention_weights.attention_output_weight.bias =
                get_ptr<T>(weights_[i + 5 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i].self_attn_layernorm_weights.gamma =
                get_ptr<T>(weights_[i + 6 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i].self_attn_layernorm_weights.beta =
                get_ptr<T>(weights_[i + 7 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i].ffn_weights.intermediate_weight.kernel =
                get_ptr<T>(weights_[i + 8 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i].ffn_weights.intermediate_weight.bias =
                get_ptr<T>(weights_[i + 9 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i].ffn_weights.output_weight.kernel =
                get_ptr<T>(weights_[i + 10 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i].ffn_weights.output_weight.bias =
                get_ptr<T>(weights_[i + 11 * layer_num_]);
        }

        gpt_weights_.post_decoder_layernorm.gamma = get_ptr<T>(weights_[12 * layer_num_ + 0]);
        gpt_weights_.post_decoder_layernorm.beta = get_ptr<T>(weights_[12 * layer_num_ + 1]);
        gpt_weights_.position_encoding_table = get_ptr<T>(weights_[12 * layer_num_ + 2]);
        gpt_weights_.pre_decoder_embedding_table = get_ptr<T>(weights_[12 * layer_num_ + 3]);
        gpt_weights_.post_decoder_embedding.kernel = get_ptr<T>(weights_[12 * layer_num_ + 4]);
#ifdef SPARSITY_ENABLED
        if (sparse_) {
            auto stream = at::cuda::getCurrentCUDAStream().stream();
            cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
            cublasSetStream(cublas_handle, stream);
            ft::cublasMMWrapper cublas_wrapper = ft::cublasMMWrapper(cublas_handle,
                                                                     cublasltHandle_,
                                                                     cusparseLtHandle_,
                                                                     stream,
                                                                     cublas_algo_map_,
                                                                     cublas_wrapper_mutex_,
                                                                     nullptr);
            // Here we need to pass hidden_units to compress weights as sparse BERT did,
            // because GptWeights has no proper attribute value - like num_layer, dummy hidden_units,
            // or inter_size. Let me udpate an initalization of GptWeights in future.
            int hidden_units = head_num_ * size_per_head_;
            for (size_t i = 0; i < layer_num_; ++i) {
                gpt_weights_.decoder_layer_weights[i].compress_weights(cublas_wrapper, hidden_units);
            }
            is_spmm_compressed = true;
        }
#endif

        check_cuda_error(cudaGetDeviceProperties(&prop_, 0));
    }

    ~FTGpt() override
    {
        cublasLtDestroy(cublasltHandle_);
        delete cublas_algo_map_;
        delete cublas_wrapper_mutex_;
    }

    void forward(th::Tensor& input_ids,
                 th::Tensor& input_lengths,
                 th::Tensor& output_ids,
                 th::Tensor& parent_ids,
                 th::Tensor& sequence_lengths,
                 size_t request_output_len) override
    {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        cublasHandle_t cublasHandle = at::cuda::getCurrentCUDABlasHandle();
        cublasSetStream(cublasHandle, stream);
        fastertransformer::Allocator<AllocatorType::TH> allocator = fastertransformer::Allocator<AllocatorType::TH>();
        ft::cublasMMWrapper cublas_wrapper = ft::cublasMMWrapper(
            cublasHandle, cublasltHandle_,
#ifdef SPARSITY_ENABLED
            cusparseLtHandle_,
#endif
            stream, cublas_algo_map_, cublas_wrapper_mutex_, &allocator);


        if (std::is_same<T, half>::value) {
            cublas_wrapper.setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
        }
        else if (std::is_same<T, float>::value) {
            cublas_wrapper.setFP32GemmConfig();
        }

        const size_t request_batch_size = (size_t)input_ids.size(0) / beam_width_;
        const size_t max_input_length = (size_t)input_ids.size(1);
        const int total_output_len = (int)(max_input_length + request_output_len);

        Gpt<T> gpt = Gpt<T>(request_batch_size,
                            total_output_len,
                            max_input_length,
                            beam_width_,
                            head_num_,
                            size_per_head_,
                            inter_size_,
                            layer_num_,
                            vocab_size_,
                            start_id_,
                            end_id_,
                            0.0f,
                            top_k_,
                            top_p_,
                            random_seed_,
                            temperature_,
                            len_penalty_,
                            repetition_penalty_,
                            stream,
                            &cublas_wrapper,
                            &allocator,
                            false,
                            &prop_,
                            sparse_);

        std::vector<Tensor> input_tensors =
            std::vector<Tensor>{Tensor{MEMORY_GPU,
                                       TYPE_INT32,
                                       std::vector<size_t>{request_batch_size * beam_width_, max_input_length},
                                       get_ptr<int>(input_ids)},
                                Tensor{MEMORY_GPU,
                                       TYPE_INT32,
                                       std::vector<size_t>{request_batch_size * beam_width_},
                                       get_ptr<int>(input_lengths)},
                                Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{1}, &total_output_len}};

        std::vector<Tensor> output_tensors =
            std::vector<Tensor>{Tensor{MEMORY_GPU,
                                       TYPE_INT32,
                                       std::vector<size_t>{request_batch_size, beam_width_, (size_t)total_output_len},
                                       get_ptr<int>(output_ids)},
                                Tensor{MEMORY_GPU,
                                       TYPE_INT32,
                                       std::vector<size_t>{(size_t)total_output_len, request_batch_size, beam_width_},
                                       get_ptr<int>(parent_ids)},
                                Tensor{MEMORY_GPU,
                                       TYPE_INT32,
                                       std::vector<size_t>{request_batch_size, beam_width_},
                                       get_ptr<int>(sequence_lengths)},
                                Tensor{MEMORY_GPU,
                                       TYPE_FP32,
                                       std::vector<size_t>{request_output_len, request_batch_size, beam_width_},
                                       nullptr}};

        try {
            gpt.forward(&output_tensors, &input_tensors, &gpt_weights_);
        }
        catch (std::runtime_error& error) {
            std::cout << error.what();
            exit(-1);
        }
        catch (...) {
            std::cout << "Runtime error";
            exit(-1);
        }
    }

private:
    const size_t max_batch_size_;
    const size_t max_seq_len_;
    const size_t beam_width_;
    const size_t head_num_;
    const size_t size_per_head_;
    const size_t inter_size_;
    const size_t layer_num_;
    const size_t vocab_size_;
    const int start_id_;
    const int end_id_;
    const float beam_search_diversity_rate_;
    const int top_k_;
    const float top_p_;
    const unsigned long long random_seed_;
    const float temperature_;
    const float len_penalty_;
    const float repetition_penalty_;
    const bool sparse_;

    std::vector<th::Tensor> weights_;
    cublasLtHandle_t cublasltHandle_;
#ifdef SPARSITY_ENABLED
    cusparseLtHandle_t cusparseLtHandle_;
    bool is_spmm_compressed = false;
#endif
    std::mutex* cublas_wrapper_mutex_;
    ft::cublasAlgoMap* cublas_algo_map_;
    struct cudaDeviceProp prop_;
    GptWeight<T> gpt_weights_;
};

class GptOp: public th::jit::CustomClassHolder {
public:
    GptOp(const int64_t max_batch_size,
          const int64_t max_seq_len,
          const int64_t beam_width,
          const int64_t head_num,
          const int64_t size_per_head,
          const int64_t inter_size,
          const int64_t layer_num,
          const int64_t vocab_size,
          const int64_t start_id,
          const int64_t end_id,
          const double beam_search_diversity_rate,
          const int64_t top_k,
          const double top_p,
          const unsigned long long random_seed,
          const double temperature,
          const double len_penalty,
          const double repetition_penalty,
          const bool sparse,
          const vector<th::Tensor> weights);

    ~GptOp();

    vector<th::Tensor> forward(th::Tensor input_ids, th::Tensor input_lengths, const int64_t output_len);

private:
    const at::ScalarType st_;
    IFGpt* ftgpt;
    std::vector<th::Tensor> weights;
};

}  // namespace torch_ext
