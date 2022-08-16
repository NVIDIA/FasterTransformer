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

#pragma once

#include <cstddef>
#include <vector>

#include "src/fastertransformer/layers/DynamicDecodeLayer.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptContextDecoder.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoder.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptWeight.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"

namespace fastertransformer {

template<typename T>
class ParallelGpt: public BaseLayer {
private:
    // meta data
    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t num_layer_;
    size_t vocab_size_;

    int    start_id_;
    int    end_id_;
    float  beam_search_diversity_rate_;
    size_t hidden_units_;

    const float layernorm_eps_;  // OPT
    float       shared_contexts_ratio_;

    // TODO(bhsueh) remove these member because they are runtime parameters
    size_t             top_k_;
    float              top_p_;
    unsigned long long random_seed_;

    float temperature_;
    float len_penalty_;
    float repetition_penalty_;

    size_t    local_head_num_;
    NcclParam tensor_para_;
    NcclParam pipeline_para_;

    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm_;
    int                                 enable_custom_all_reduce_;

    const bool is_context_qk_buf_float_ = true;
    size_t     vocab_size_padded_;
    const int  int8_mode_      = 0;
    bool       remove_padding_ = true;

    // Prompt Learning Parameters
    PromptLearningType prompt_learning_type_;
    int                prompt_learning_start_id_;  // start_id for prompt_learning (only needed by prefix prompts)
    bool               has_p_prompt_tuning_;
    bool               has_prefix_prompt_;
    bool               has_prefix_soft_prompt_;

    // GPT Variants parameters: e.g. Meta OPT
    gptVariantParams gpt_variant_params_;

    ParallelGptDecoder<T>*        gpt_decoder_;
    ParallelGptContextDecoder<T>* gpt_context_decoder_;
    DynamicDecodeLayer<float>*    dynamic_decode_layer_;

    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size,
                        size_t beam_width,
                        size_t max_seq_len,
                        size_t memory_len,
                        size_t max_input_len,
                        bool   is_return_context_cum_log_probs);
    void freeBuffer() override;

    void initialize();

    void computeContextCumLogProbs(float*                      cum_log_probs,
                                   const T*                    context_decoder_outputs,
                                   const int*                  input_ids,
                                   const int*                  input_lengths,
                                   const size_t                batch_size,
                                   const size_t                beam_width,
                                   const size_t                max_input_length,
                                   const ParallelGptWeight<T>* gpt_weights);

protected:
    // For stateful processing (interactive generation)
    int    step_;
    size_t session_len_;
    size_t memory_len_;
    int*   tiled_total_padding_count_ = nullptr;

    T*       padded_embedding_kernel_;
    const T* padded_embedding_kernel_ptr_;

    T* input_attention_mask_;

    T*        decoder_input_buf_;
    T*        decoder_output_buf_;
    T*        normed_decoder_output_buf_;
    float*    logits_buf_;
    float*    nccl_logits_buf_;
    float*    cum_log_probs_;
    bool*     finished_buf_;
    bool*     h_finished_buf_;
    int*      sequence_lengths_       = nullptr;
    uint32_t* seq_limit_len_          = nullptr;
    bool*     generation_should_stop_ = nullptr;

    int* shared_contexts_idx_      = nullptr;
    T*   compact_decoder_features_ = nullptr;
    int* compact_idx_              = nullptr;
    int* batch_to_compact_idx_     = nullptr;
    int* compact_size_             = nullptr;

    T*   key_cache_;
    T*   value_cache_;
    int* cache_indirections_[2] = {nullptr, nullptr};

    int* start_ids_buf_;
    int* end_ids_buf_;

    int* tiled_input_ids_buf_;
    int* tiled_input_lengths_buf_;

    // prompt_learning weight_batch ptrs
    const T** prompt_learning_weight_batch_;
    int*      tiled_prompt_lengths_buf_;  // only needed by prefix prompts

    int*  transposed_output_ids_buf_;
    int*  output_ids_buf_;
    int*  parent_ids_buf_;
    bool* masked_tokens_ = nullptr;

    T*     context_decoder_input_buf_;
    T*     context_decoder_output_buf_;
    float* output_log_probs_buf_;

    // buffers dedicated to log prob computation
    T*     lp_normed_decoder_output_buf_ = nullptr;
    float* lp_logits_buf_                = nullptr;
    float* lp_nccl_logits_buf_           = nullptr;
    float* lp_logprob_buf_               = nullptr;

    // function pointer callback
    using callback_sig                 = void(std::unordered_map<std::string, Tensor>*, void*);
    callback_sig* token_generated_cb_  = nullptr;
    void*         token_generated_ctx_ = nullptr;

    void setOutputTensors(std::unordered_map<std::string, Tensor>*       output_tensors,
                          const std::unordered_map<std::string, Tensor>* input_tensors,
                          const size_t                                   gen_len,
                          const size_t                                   session_len,
                          const size_t                                   max_context_len);
    void sendTensorsToFirstPipelineNode(std::unordered_map<std::string, Tensor>*       output_tensors,
                                        const std::unordered_map<std::string, Tensor>* input_tensors);

public:
    ParallelGpt(size_t                              max_batch_size,
                size_t                              max_seq_len,
                size_t                              max_input_len,
                size_t                              beam_width,
                size_t                              head_num,
                size_t                              size_per_head,
                size_t                              inter_size,
                size_t                              num_layer,
                size_t                              vocab_size,
                int                                 start_id,
                int                                 end_id,
                int                                 prompt_learning_start_id,  // only needed by p/prompt-tuning
                PromptLearningType                  prompt_learning_type,
                gptVariantParams                    gpt_variant_params,
                float                               beam_search_diversity_rate,
                size_t                              top_k,
                float                               top_p,
                unsigned long long                  random_seed,
                float                               temperature,
                float                               len_penalty,
                float                               repetition_penalty,
                NcclParam                           tensor_para,
                NcclParam                           pipeline_para,
                cudaStream_t                        stream,
                cublasMMWrapper*                    cublas_wrapper,
                IAllocator*                         allocator,
                bool                                is_free_buffer_after_forward,
                cudaDeviceProp*                     cuda_device_prop         = nullptr,
                bool                                sparse                   = false,
                int                                 int8_mode                = 0,
                std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm   = nullptr,
                int                                 enable_custom_all_reduce = 0,
                bool                                remove_padding           = true,
                float                               shared_contexts_ratio    = 1.0f);

    ParallelGpt(ParallelGpt<T> const& gpt);

    ~ParallelGpt();

    void forward(std::vector<Tensor>*        output_tensors,
                 const std::vector<Tensor>*  input_tensors,
                 const ParallelGptWeight<T>* gpt_weights);

    void forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                 const std::unordered_map<std::string, Tensor>* input_tensors,
                 const ParallelGptWeight<T>*                    gpt_weights);

    size_t getPipelineParallelRank();
    size_t getPipelineParallelSize();
    size_t getTensorParallelRank();
    size_t getTensorParallelSize();
    bool*  getFinishBuffer();

    void registerCallback(callback_sig* fn, void* ctx);
    void unRegisterCallback();
};

}  // namespace fastertransformer
