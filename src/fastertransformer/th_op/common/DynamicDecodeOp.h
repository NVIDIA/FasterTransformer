/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/layers/DynamicDecodeLayer.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include "src/fastertransformer/utils/nccl_utils.h"

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

class IFtDynamicDecode {
public:
    virtual ~IFtDynamicDecode() {}

    virtual void setup(size_t                   batch_size,
                       size_t                   beam_width,
                       th::optional<th::Tensor> runtime_top_k_opt,
                       th::optional<th::Tensor> runtime_top_p_opt,
                       th::optional<th::Tensor> temperature_opt,
                       th::optional<th::Tensor> repetition_penalty_opt,
                       th::optional<th::Tensor> length_penalty_opt,
                       th::optional<th::Tensor> beam_search_diversity_rate_opt,
                       th::optional<th::Tensor> random_seed_opt,
                       th::optional<th::Tensor> top_p_decay_opt,
                       th::optional<th::Tensor> top_p_min_opt,
                       th::optional<th::Tensor> top_p_reset_ids_opt) = 0;

    virtual void forward(th::Tensor&              logits,  // (batch_size, beam_width, hidden_size)
                         int                      step,
                         int                      max_input_length,
                         uint                     ite,
                         int                      local_batch_size,
                         th::Tensor               end_id,
                         th::optional<th::Tensor> runtime_top_k_opt,
                         th::optional<th::Tensor> runtime_top_p_opt,
                         th::optional<th::Tensor> temperature_opt,
                         th::optional<th::Tensor> repetition_penalty_opt,
                         th::optional<th::Tensor> length_penalty_opt,
                         th::optional<th::Tensor> beam_search_diversity_rate_opt,
                         th::optional<th::Tensor> top_p_decay_opt,
                         th::optional<th::Tensor> top_p_min_opt,
                         th::optional<th::Tensor> top_p_reset_ids_opt,
                         th::optional<th::Tensor> embedding_bias_opt,
                         th::optional<th::Tensor> input_lengths_opt,
                         th::optional<th::Tensor> sequence_limit_lengths_opt,
                         th::optional<th::Tensor> stop_words_list_opt,
                         th::optional<th::Tensor> bad_words_list_opt,
                         th::optional<th::Tensor> src_cache_indirection_opt,
                         // Outputs
                         th::Tensor&              output_token_ids,
                         th::Tensor&              should_stop_opt,
                         th::optional<th::Tensor> finished,
                         th::optional<th::Tensor> sequnce_lengths_opt,
                         th::optional<th::Tensor> cum_log_probs_opt,
                         th::optional<th::Tensor> output_log_probs_opt,
                         th::optional<th::Tensor> parent_ids_opt,
                         th::optional<th::Tensor> tgt_cache_indirection_opt) = 0;

    virtual void broadcastFromLastPipeline(std::vector<th::Tensor> tensors) = 0;
};

template<typename T>
class FtDynamicDecode: public IFtDynamicDecode {
public:
    FtDynamicDecode(const size_t vocab_size,
                    const size_t vocab_size_padded,
                    const int    tensor_para_size,
                    const int    pipeline_para_size);
    ~FtDynamicDecode() override;

    void setup(size_t                   batch_size,
               size_t                   beam_width,
               th::optional<th::Tensor> runtime_top_k_opt,
               th::optional<th::Tensor> runtime_top_p_opt,
               th::optional<th::Tensor> temperature_opt,
               th::optional<th::Tensor> repetition_penalty_opt,
               th::optional<th::Tensor> length_penalty_opt,
               th::optional<th::Tensor> beam_search_diversity_rate_opt,
               th::optional<th::Tensor> random_seed_opt,
               th::optional<th::Tensor> top_p_decay_opt,
               th::optional<th::Tensor> top_p_min_opt,
               th::optional<th::Tensor> top_p_reset_ids_opt) override;

    void forward(th::Tensor&              logits,  // (batch_size, beam_width, hidden_size)
                 int                      step,
                 int                      max_input_length,
                 uint                     ite,
                 int                      local_batch_size,
                 th::Tensor               end_id,
                 th::optional<th::Tensor> runtime_top_k_opt,
                 th::optional<th::Tensor> runtime_top_p_opt,
                 th::optional<th::Tensor> temperature_opt,
                 th::optional<th::Tensor> repetition_penalty_opt,
                 th::optional<th::Tensor> length_penalty_opt,
                 th::optional<th::Tensor> beam_search_diversity_rate_opt,
                 th::optional<th::Tensor> top_p_decay_opt,
                 th::optional<th::Tensor> top_p_min_opt,
                 th::optional<th::Tensor> top_p_reset_ids_opt,
                 th::optional<th::Tensor> embedding_bias_opt,
                 th::optional<th::Tensor> input_lengths_opt,
                 th::optional<th::Tensor> sequence_limit_lengths_opt,
                 th::optional<th::Tensor> stop_words_list_opt,
                 th::optional<th::Tensor> bad_words_list_opt,
                 th::optional<th::Tensor> src_cache_indirection_opt,
                 // Outputs
                 th::Tensor&              output_token_ids,
                 th::Tensor&              should_stop,
                 th::optional<th::Tensor> finished_opt,
                 th::optional<th::Tensor> sequence_lengths_opt,
                 th::optional<th::Tensor> cum_log_probs_opt,
                 th::optional<th::Tensor> output_log_probs_opt,
                 th::optional<th::Tensor> parent_ids_opt,
                 th::optional<th::Tensor> tgt_cache_indirection_opt) override;

    void broadcastFromLastPipeline(std::vector<th::Tensor> tensors) override;

private:
    const size_t vocab_size_;
    const size_t vocab_size_padded_;

    ft::NcclParam tensor_para_;
    ft::NcclParam pipeline_para_;

    ft::IAllocator*      allocator_;
    cublasLtHandle_t     cublaslt_handle_;
    std::mutex*          cublas_wrapper_mutex_;
    ft::cublasAlgoMap*   cublas_algo_map_;
    ft::cublasMMWrapper* cublas_wrapper_;
    cudaDeviceProp       prop_;

    ft::DynamicDecodeLayer<T>* dynamic_decode_layer_;
};

class DynamicDecodeOp: public th::jit::CustomClassHolder {
public:
    DynamicDecodeOp(const int64_t  vocab_size,
                    const int64_t  vocab_size_padded,
                    const int64_t  tensor_para_size,
                    const int64_t  pipeline_para_size,
                    at::ScalarType scalar_type);
    ~DynamicDecodeOp();

    void setup(int64_t                  batch_size,
               int64_t                  beam_width,
               th::optional<th::Tensor> runtime_top_k_opt,
               th::optional<th::Tensor> runtime_top_p_opt,
               th::optional<th::Tensor> temperature_opt,
               th::optional<th::Tensor> reptition_penalty_opt,
               th::optional<th::Tensor> length_penalty_opt,
               th::optional<th::Tensor> beam_search_diversity_rate_opt,
               th::optional<th::Tensor> random_seed_opt,
               th::optional<th::Tensor> top_p_decay_opt,
               th::optional<th::Tensor> top_p_min_opt,
               th::optional<th::Tensor> top_p_reset_ids_opt);

    th::Tensor forward(th::Tensor               logits,  // (batch_size, beam_width, vocab_size)
                       int64_t                  step,
                       int64_t                  max_input_length,
                       int64_t                  ite,
                       int64_t                  local_batch_size,
                       th::Tensor               end_id,
                       th::optional<th::Tensor> runtime_top_k_opt,
                       th::optional<th::Tensor> runtime_top_p_opt,
                       th::optional<th::Tensor> temperature_opt,
                       th::optional<th::Tensor> repetition_penalty_opt,
                       th::optional<th::Tensor> length_penalty_opt,
                       th::optional<th::Tensor> beam_search_diversity_rate_opt,
                       th::optional<th::Tensor> top_p_decay_opt,
                       th::optional<th::Tensor> top_p_min_opt,
                       th::optional<th::Tensor> top_p_reset_ids_opt,
                       th::optional<th::Tensor> embedding_bias_opt,
                       th::optional<th::Tensor> input_lengths_opt,  // length of input contexts.
                       th::optional<th::Tensor> sequence_limit_length_opt,
                       th::optional<th::Tensor> stop_words_list_opt,
                       th::optional<th::Tensor> bad_words_list_opt,
                       th::optional<th::Tensor> src_cache_indirection_opt,
                       // output buffers.
                       th::Tensor               output_token_ids,
                       th::optional<th::Tensor> finished_opt,
                       th::optional<th::Tensor> seuqence_lengths_opt,  // length of the current sequences.
                       th::optional<th::Tensor> cum_log_probs_opt,
                       th::optional<th::Tensor> output_log_probs_opt,
                       th::optional<th::Tensor> parent_ids_opt,
                       th::optional<th::Tensor> tgt_cache_indirection_opt);

    void broadcastFromLastPipeline(std::vector<th::Tensor> tensors);

private:
    size_t vocab_size_;
    size_t vocab_size_padded_;
    int    tensor_para_size_;
    int    pipeline_para_size_;

    // Data type of expected input logits.
    at::ScalarType scalar_type_;
    // FT Dynamice decode layer wrapper instance.
    std::unique_ptr<IFtDynamicDecode> dynamic_decode_;

    void createInstance();
};

}  // namespace torch_ext
