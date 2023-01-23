/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "3rdparty/INIReader.h"

#include "src/fastertransformer/models/gpt_fp8/GptFP8.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include "src/fastertransformer/utils/mpi_utils.h"
#include "src/fastertransformer/utils/nvtx_utils.h"

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

using std::vector;

class IFGptFp8 {
public:
    virtual ~IFGptFp8() {}
    virtual void forward(th::Tensor&           input_ids,
                         th::Tensor&           input_lengths,
                         th::Tensor&           output_ids,
                         th::Tensor&           parent_ids,
                         th::Tensor&           sequence_lengths,
                         th::Tensor&           cum_log_probs,
                         const size_t          request_output_len,
                         const size_t          beam_width,
                         th::optional<int64_t> top_k_opt,
                         th::optional<double>  top_p_opt,
                         th::optional<double>  beam_search_diversity_rate_opt,
                         th::optional<double>  temperature_opt,
                         th::optional<double>  len_penalty_opt,
                         th::optional<double>  repetition_penalty_opt,
                         th::optional<int64_t> random_seed_opt,
                         th::optional<int64_t> return_cum_log_probs_opt) = 0;
};

template<typename T1, typename T2>
class FTGptFp8: public IFGptFp8 {
public:
    FTGptFp8(const size_t      head_num,
             const size_t      size_per_head,
             const size_t      inter_size,
             const size_t      layer_num,
             const size_t      vocab_size,
             const size_t      max_seq_len,
             const int         start_id,
             const int         end_id,
             const int         tensor_para_size,
             const int         pipeline_para_size,
             const std::string ckpt_path):
        head_num_(head_num),
        size_per_head_(size_per_head),
        inter_size_(inter_size),
        layer_num_(layer_num),
        vocab_size_(vocab_size),
        max_seq_len_(max_seq_len),
        start_id_(start_id),
        end_id_(end_id),
        ckpt_path_(ckpt_path),
        tensor_para_size_(tensor_para_size),
        pipeline_para_size_(pipeline_para_size)
    {
        ftNcclInitialize(tensor_para_, pipeline_para_, tensor_para_size_, pipeline_para_size_);
        ft::check_cuda_error(cublasLtCreate(&cublasltHandle_));

        cublas_algo_map_      = new ft::cublasAlgoMap(GEMM_CONFIG, "");
        cublas_wrapper_mutex_ = new std::mutex();

        ft::check_cuda_error(cudaGetDeviceProperties(&prop_, 0));

        const size_t hidden_units = head_num_ * size_per_head_;
        gpt_weights_              = new ft::GptFP8Weight<T1, T2>(hidden_units,
                                                    inter_size_,
                                                    vocab_size_,
                                                    layer_num_,
                                                    max_seq_len_,
                                                    tensor_para_size_,
                                                    tensor_para_rank_,
                                                    pipeline_para_size_,
                                                    pipeline_para_rank_);

        gpt_weights_->loadModel(ckpt_path_);
        gpt_weights_->transposeWeight();
    }

    ~FTGptFp8() override
    {
        cublasLtDestroy(cublasltHandle_);
        delete cublas_algo_map_;
        delete cublas_wrapper_mutex_;
    }

    void forward(th::Tensor&           input_ids,
                 th::Tensor&           input_lengths,
                 th::Tensor&           output_ids,
                 th::Tensor&           parent_ids,
                 th::Tensor&           sequence_lengths,
                 th::Tensor&           cum_log_probs,
                 size_t                request_output_len,
                 size_t                beam_width,
                 th::optional<int64_t> top_k_opt,
                 th::optional<double>  top_p_opt,
                 th::optional<double>  beam_search_diversity_rate_opt,
                 th::optional<double>  temperature_opt,
                 th::optional<double>  len_penalty_opt,
                 th::optional<double>  repetition_penalty_opt,
                 th::optional<int64_t> random_seed_opt,
                 th::optional<int64_t> return_cum_log_probs_opt) override
    {
        uint  top_k = top_k_opt.has_value() ? (uint)top_k_opt.value() : 1;
        float top_p = top_p_opt.has_value() ? (float)top_p_opt.value() : 0.0f;
        float beam_search_diversity_rate =
            beam_search_diversity_rate_opt.has_value() ? (float)beam_search_diversity_rate_opt.value() : 0.0f;
        float temperature        = temperature_opt.has_value() ? (float)temperature_opt.value() : 1.0f;
        float len_penalty        = len_penalty_opt.has_value() ? (float)len_penalty_opt.value() : 1.0f;
        float repetition_penalty = repetition_penalty_opt.has_value() ? (float)repetition_penalty_opt.value() : 1.0f;
        unsigned long long random_seed = random_seed_opt.has_value() ? (unsigned long long)random_seed_opt.value() : 0;
        int return_cum_log_probs = return_cum_log_probs_opt.has_value() ? (int)return_cum_log_probs_opt.value() : 0;

        const size_t hidden_units       = head_num_ * size_per_head_;
        const size_t request_batch_size = (size_t)input_ids.size(0);
        const size_t max_input_len      = (size_t)input_ids.size(1);
        const int    total_output_len   = max_input_len + request_output_len;
        if (total_output_len > (int)max_seq_len_) {
            printf("[ERROR] total_output_len (%d) should be <= max_seq_len (%ld). \n", total_output_len, max_seq_len_);
            exit(-1);
        }
        std::vector<uint32_t> output_seq_len_array(request_batch_size, total_output_len);

        auto           stream       = at::cuda::getCurrentCUDAStream().stream();
        cublasHandle_t cublasHandle = at::cuda::getCurrentCUDABlasHandle();
        cublasSetStream(cublasHandle, stream);
        ft::Allocator<ft::AllocatorType::TH> allocator      = ft::Allocator<ft::AllocatorType::TH>();
        ft::cublasFP8MMWrapper               cublas_wrapper = ft::cublasFP8MMWrapper(
            cublasHandle, cublasltHandle_, stream, cublas_algo_map_, cublas_wrapper_mutex_, &allocator);

        cublas_wrapper.setGemmConfig(CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F);

        struct cudaDeviceProp prop;
        ft::check_cuda_error(cudaGetDeviceProperties(&prop, 0));

        // Need to remove them in the future.
        ft::GptFP8<T1, T2> gpt = ft::GptFP8<__nv_fp8_e4m3, __nv_bfloat16>(beam_width,
                                                                          head_num_,
                                                                          size_per_head_,
                                                                          inter_size_,
                                                                          layer_num_,
                                                                          vocab_size_,
                                                                          start_id_,
                                                                          end_id_,
                                                                          tensor_para_,
                                                                          pipeline_para_,
                                                                          stream,
                                                                          &cublas_wrapper,
                                                                          &allocator,
                                                                          false,
                                                                          &prop,
                                                                          false);

        std::unordered_map<std::string, ft::Tensor> input_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"input_ids",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, (size_t)max_input_len},
                        get_ptr<int>(input_ids)}},
            {"input_lengths",
             ft::Tensor{
                 ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{request_batch_size}, get_ptr<int>(input_lengths)}},
            {"output_seq_len",
             ft::Tensor{ft::MEMORY_CPU,
                        ft::TYPE_UINT32,
                        std::vector<size_t>{request_batch_size},
                        output_seq_len_array.data()}},
            {"temperature", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &temperature}},
            {"len_penalty", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &len_penalty}},
            {"repetition_penalty",
             ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &repetition_penalty}}};
        if (top_k == 0 && top_p == 0.0f) {
            ft::FT_CHECK(beam_width > 1);
            input_tensors.insert(
                {"beam_search_diversity_rate",
                 ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &beam_search_diversity_rate}});
        }
        else {
            input_tensors.insert(
                {"random_seed", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_UINT64, std::vector<size_t>{1}, &random_seed}});
            if (top_p != 0.0f) {
                input_tensors.insert(
                    {"runtime_top_p", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &top_p}});
            }
            if (top_k != 0) {
                input_tensors.insert(
                    {"runtime_top_k", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_UINT32, std::vector<size_t>{1}, &top_k}});
            }
        }
        bool return_context_cum_log_probs = false;
        if (return_cum_log_probs == 2) {
            return_context_cum_log_probs = true;
            input_tensors.insert(
                {"is_return_context_cum_log_probs",
                 ft::Tensor{ft::MEMORY_CPU, ft::TYPE_BOOL, std::vector<size_t>{1}, &return_context_cum_log_probs}});
        }

        std::unordered_map<std::string, ft::Tensor> output_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"output_ids",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, beam_width, (size_t)total_output_len},
                        get_ptr<int>(output_ids)}},
            {"sequence_length",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, beam_width},
                        get_ptr<int>(sequence_lengths)}},
            {"output_cum_log_probs",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_FP32,
                        std::vector<size_t>{(size_t)request_output_len, request_batch_size, beam_width},
                        nullptr}}};
        if (return_cum_log_probs > 0) {
            output_tensors.insert({"cum_log_probs",
                                   ft::Tensor{ft::MEMORY_GPU,
                                              ft::TYPE_FP32,
                                              std::vector<size_t>{request_batch_size, beam_width},
                                              get_ptr<float>(cum_log_probs)}});
        }

        try {
            gpt.forward(&output_tensors, &input_tensors, gpt_weights_);
        }
        catch (std::runtime_error& error) {
            std::cout << error.what();
            exit(-1);
        }
        catch (...) {
            std::cout << "Runtime error";
            exit(-1);
        }

        return;
    }

private:
    const size_t head_num_;
    const size_t size_per_head_;
    const size_t inter_size_;
    const size_t layer_num_;
    const size_t vocab_size_;
    const size_t max_seq_len_;
    const int    start_id_;
    const int    end_id_;

    size_t        tensor_para_size_;
    size_t        pipeline_para_size_;
    size_t        tensor_para_rank_;
    ft::NcclParam tensor_para_;
    size_t        pipeline_para_rank_;
    ft::NcclParam pipeline_para_;
    int           world_size_ = 1;
    int           rank_       = 0;

    // std::vector<th::Tensor> weights_;
    cublasLtHandle_t                                cublasltHandle_;
    std::mutex*                                     cublas_wrapper_mutex_;
    ft::cublasAlgoMap*                              cublas_algo_map_;
    struct cudaDeviceProp                           prop_;
    const std::string                               ckpt_path_;
    ft::GptFP8Weight<__nv_fp8_e4m3, __nv_bfloat16>* gpt_weights_;
};

class GptFp8Op: public th::jit::CustomClassHolder {
public:
    GptFp8Op(const int64_t            head_num,
             const int64_t            size_per_head,
             const int64_t            inter_size,
             const int64_t            layer_num,
             const int64_t            vocab_size,
             const int64_t            max_seq_len,
             const int64_t            start_id,
             const int64_t            end_id,
             const int64_t            tensor_para_size,
             const int64_t            pipeline_para_size,
             const double             layernorm_eps,
             const std::string        layernorm_type,
             const std::string        activation_type,
             const std::string        ckpt_path,
             const bool               has_post_decoder_layernorm,
             const vector<th::Tensor> weights);

    ~GptFp8Op();

    vector<th::Tensor> forward(th::Tensor            input_ids,
                               th::Tensor            input_lengths,
                               const int64_t         output_len,
                               th::optional<int64_t> beam_width_opt,
                               th::optional<int64_t> top_k_opt,
                               th::optional<double>  top_p_opt,
                               th::optional<double>  beam_search_diversity_rate_opt,
                               th::optional<double>  temperature_opt,
                               th::optional<double>  len_penalty_opt,
                               th::optional<double>  repetition_penalty_opt,
                               th::optional<int64_t> random_seed_opt,
                               th::optional<int64_t> return_cum_log_probs_opt);

private:
    // const at::ScalarType st_;
    IFGptFp8* ftgpt;
    // std::vector<th::Tensor> weights;
};

}  // namespace torch_ext
