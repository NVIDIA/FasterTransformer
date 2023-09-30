/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/llama/LLaMA.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include "src/fastertransformer/utils/nccl_utils.h"

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

using std::vector;

class IFLLaMA {
public:
    virtual ~IFLLaMA() {}
    virtual void forward(th::Tensor& hidden_vector,
                         th::Tensor& log_probs,
                         th::Tensor& cum_probs,
                         th::Tensor& input_ids,
                         th::Tensor& input_lengths,
                         th::Tensor& context_lengths,
                         const int   num_tokens,
                         const int   seq_len,
                         const int   attn_len,
                         const int   is_context) = 0;
};

template<typename T>
class FTLLaMA: public IFLLaMA {
public:
    FTLLaMA(const size_t             num_heads,
            const size_t             size_per_head,
            const size_t             inter_size,
            const size_t             num_layers,
            const size_t             vocab_size,
            const size_t             rotary_embedding_dim,
            const size_t             random_seed,
            const size_t             max_seq_len,
            const int64_t            tensor_para_size,
            const int64_t            pipeline_para_size,
            const vector<th::Tensor> weights):
        num_heads_(num_heads),
        size_per_head_(size_per_head),
        inter_size_(inter_size),
        num_layers_(num_layers),
        vocab_size_(vocab_size),
        rotary_embedding_dim_(rotary_embedding_dim),
        random_seed_(random_seed),
        max_seq_len_(max_seq_len),
        tensor_para_size_(tensor_para_size),
        pipeline_para_size_(pipeline_para_size),
        weights_(weights)
    {
        ft::Logger::getLogger().setLevel(ft::Logger::WARNING);

        ft::check_cuda_error(cublasLtCreate(&cublasltHandle_));
        cublas_algo_map_      = new ft::cublasAlgoMap(GEMM_CONFIG, "");
        cublas_wrapper_mutex_ = new std::mutex();

        ftNcclInitialize(tensor_para_, pipeline_para_, tensor_para_size, pipeline_para_size);

        llama_weights_.resizeLayer(num_layers_);
        for (int i = 0; i < (int)num_layers_; i++) {
            llama_weights_.decoder_layer_weights[i]->pre_layernorm_weights.beta =
                get_ptr<T>(weights_[i + 0 * num_layers_]);
            llama_weights_.decoder_layer_weights[i]->pre_layernorm_weights.gamma =
                get_ptr<T>(weights_[i + 1 * num_layers_]);
            llama_weights_.decoder_layer_weights[i]->self_attention_weights.query_weight.kernel =
                get_ptr<T>(weights_[i + 2 * num_layers_]);
            llama_weights_.decoder_layer_weights[i]->self_attention_weights.query_weight.bias =
                get_ptr<T>(weights_[i + 3 * num_layers_]);
            llama_weights_.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.kernel =
                get_ptr<T>(weights_[i + 4 * num_layers_]);
            llama_weights_.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.bias =
                get_ptr<T>(weights_[i + 5 * num_layers_]);
            llama_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight.kernel =
                get_ptr<T>(weights_[i + 6 * num_layers_]);
            llama_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight.bias =
                get_ptr<T>(weights_[i + 7 * num_layers_]);
            llama_weights_.decoder_layer_weights[i]->ffn_weights.output_weight.kernel =
                get_ptr<T>(weights_[i + 8 * num_layers_]);
            llama_weights_.decoder_layer_weights[i]->ffn_weights.output_weight.bias =
                get_ptr<T>(weights_[i + 9 * num_layers_]);
            llama_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight2.kernel =
                get_ptr<T>(weights_[i + 10 * num_layers_]);
            llama_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight2.bias =
                get_ptr<T>(weights_[i + 11 * num_layers_]);
            llama_weights_.decoder_layer_weights[i]->post_attention_layernorm_weights.beta =
                get_ptr<T>(weights_[i + 12 * num_layers_]);
            llama_weights_.decoder_layer_weights[i]->post_attention_layernorm_weights.gamma =
                get_ptr<T>(weights_[i + 13 * num_layers_]);
        }

        llama_weights_.pre_decoder_embedding_table   = get_ptr<T>(weights_[14 * num_layers_ + 0]);
        llama_weights_.post_decoder_layernorm.beta   = get_ptr<T>(weights_[14 * num_layers_ + 1]);
        llama_weights_.post_decoder_layernorm.gamma  = get_ptr<T>(weights_[14 * num_layers_ + 2]);
        llama_weights_.post_decoder_embedding.kernel = get_ptr<T>(weights_[14 * num_layers_ + 3]);

        ft::check_cuda_error(cudaGetDeviceProperties(&prop_, 0));
        // ft::check_cuda_error(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        ft::check_cuda_error(cudaStreamCreate(&stream_));

        for (int i = 0; i < num_events_; ++i) {
            ft::check_cuda_error(cudaEventCreate(&event_[i]));
        }

        cublasHandle_t cublasHandle = at::cuda::getCurrentCUDABlasHandle();
        cublasSetStream(cublasHandle, stream_);

        allocator_      = new ft::Allocator<ft::AllocatorType::TH>();
        cublas_wrapper_ = new ft::cublasMMWrapper(
            cublasHandle, cublasltHandle_, stream_, cublas_algo_map_, cublas_wrapper_mutex_, allocator_);

        if (std::is_same<T, half>::value) {
            cublas_wrapper_->setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
        }
        else if (std::is_same<T, float>::value) {
            cublas_wrapper_->setFP32GemmConfig();
        }

        ft::AttentionType attention_type = ft::getAttentionType<T>(size_per_head_,
                                                                   ft::getSMVersion(),
                                                                   true,   // remove_padding
                                                                   0,      // gpt supports any-seq-length fmha
                                                                   true,   // is_fuse
                                                                   false,  // with_relative_position_bias
                                                                   true);  // causal_mask
                                                                           //
        llama_ = new ft::LLaMA<T>(num_heads_,
                                  size_per_head_,
                                  inter_size_,
                                  num_layers_,
                                  vocab_size_,
                                  rotary_embedding_dim_,
                                  random_seed_,
                                  max_seq_len_,
                                  tensor_para_,
                                  pipeline_para_,
                                  stream_,
                                  cublas_wrapper_,
                                  allocator_,
                                  false,          // is_free_buffer_after_forward
                                  &prop_,         // cuda_device_prop
                                  attention_type  // attention_type
        );
    }

    ~FTLLaMA() override
    {
        for (int i = 0; i < num_events_; ++i) {
            ft::check_cuda_error(cudaEventDestroy(event_[i]));
        }
        ft::check_cuda_error(cudaStreamDestroy(stream_));

        delete llama_;
        delete cublas_wrapper_;
        delete allocator_;

        ft::ftNcclParamDestroy(tensor_para_);
        ft::ftNcclParamDestroy(pipeline_para_);
        cublasLtDestroy(cublasltHandle_);
        delete cublas_algo_map_;
        delete cublas_wrapper_mutex_;
    }

    virtual void forward(th::Tensor& hidden_vector,
                         th::Tensor& log_probs,
                         th::Tensor& cum_probs,
                         th::Tensor& input_ids,
                         th::Tensor& input_lengths,
                         th::Tensor& context_lengths,
                         const int   num_tokens,
                         const int   seq_len,
                         const int   attn_len,
                         const int   is_context) override
    {
        const size_t batch_size = (size_t)input_lengths.size(0);

        std::unordered_map<std::string, ft::Tensor> input_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"input_ids",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{batch_size, (size_t)seq_len},
                        get_ptr<int>(input_ids)}},
            {"input_lengths",
             ft::Tensor{ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{batch_size}, get_ptr<int>(input_lengths)}},
            {"context_lengths",
             ft::Tensor{
                 ft::MEMORY_CPU, ft::TYPE_INT32, std::vector<size_t>{batch_size}, get_ptr<int>(context_lengths)}},
            {"num_tokens", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, std::vector<size_t>{1}, &num_tokens}},
            {"seq_len", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, std::vector<size_t>{1}, &seq_len}},
            {"attn_len", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, std::vector<size_t>{1}, &attn_len}},
            {"is_context", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, std::vector<size_t>{1}, &is_context}}};

        std::unordered_map<std::string, ft::Tensor> output_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"hidden_vector",
             ft::Tensor{ft::MEMORY_GPU,
                        (std::is_same<T, half>::value) ? ft::TYPE_FP16 : ft::TYPE_FP32,
                        std::vector<size_t>{(size_t)num_tokens, num_heads_ * size_per_head_},
                        get_ptr<T>(hidden_vector)}},
            {"log_probs",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_FP32,
                        std::vector<size_t>{(size_t)num_tokens, vocab_size_},
                        get_ptr<float>(log_probs)}},
            {"cum_probs",
             ft::Tensor{ft::MEMORY_GPU, ft::TYPE_FP32, std::vector<size_t>{batch_size}, get_ptr<float>(cum_probs)}}};

        try {
            ft::check_cuda_error(cudaEventSynchronize(event_[ev_no_]));
            llama_->forward(&output_tensors, &input_tensors, &llama_weights_);
            ft::check_cuda_error(cudaEventRecord(event_[ev_no_], stream_));

            auto stream = at::cuda::getCurrentCUDAStream().stream();
            ft::check_cuda_error(cudaStreamWaitEvent(stream, event_[ev_no_]));
            ev_no_ = (ev_no_ + 1) % num_events_;
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
    const size_t num_heads_;
    const size_t size_per_head_;
    const size_t inter_size_;
    const size_t num_layers_;
    const size_t vocab_size_;
    const size_t rotary_embedding_dim_;
    const size_t random_seed_;
    const size_t max_seq_len_;
    int64_t      tensor_para_size_;
    int64_t      pipeline_para_size_;

    static constexpr int num_events_ = 5;
    int                  ev_no_      = 0;
    cudaEvent_t          event_[num_events_];
    cudaStream_t         stream_;

    std::vector<th::Tensor> weights_;
    cublasLtHandle_t        cublasltHandle_;
    std::mutex*             cublas_wrapper_mutex_;
    ft::cublasAlgoMap*      cublas_algo_map_;
    struct cudaDeviceProp   prop_;
    ft::LLaMAWeight<T>      llama_weights_;

    ft::NcclParam tensor_para_;
    ft::NcclParam pipeline_para_;

    ft::cublasMMWrapper* cublas_wrapper_;
    ft::IAllocator*      allocator_;
    ft::LLaMA<T>*        llama_ = nullptr;
};

class LLaMA: public th::jit::CustomClassHolder {
public:
    LLaMA(const int64_t            num_heads,
          const int64_t            size_per_head,
          const int64_t            inter_size,
          const int64_t            num_layers,
          const int64_t            vocab_size,
          const int64_t            rotary_embedding_dim,
          const int64_t            random_seed,
          const int64_t            max_seq_len,
          const int64_t            tensor_para_size,
          const int64_t            pipeline_para_size,
          const vector<th::Tensor> weights);

    ~LLaMA();

    std::vector<th::Tensor> forward(th::Tensor&   hidden_vector,
                                    th::Tensor&   log_probs,
                                    th::Tensor&   cum_probs,
                                    th::Tensor&   input_ids,
                                    th::Tensor&   input_lengths,
                                    th::Tensor&   context_lengths,
                                    const int64_t num_tokens,
                                    const int64_t seq_len,
                                    const int64_t attn_len,
                                    const int64_t is_context);

private:
    const at::ScalarType    st_;
    size_t                  vocab_size_;
    IFLLaMA*                ftllama;
    std::vector<th::Tensor> weights;
};

}  // namespace torch_ext
