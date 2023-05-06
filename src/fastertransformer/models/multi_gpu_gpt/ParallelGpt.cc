/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 * Copyright (c) 2022, SK Telecom Authored by A. Dialog
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

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/decoding_kernels.h"
#include "src/fastertransformer/kernels/gen_relative_pos_bias.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/kernels/logprob_kernels.h"
#include "src/fastertransformer/layers/beam_search_layers/BaseBeamSearchLayer.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/utils/nvtx_utils.h"

namespace fastertransformer {

template<typename T>
void ParallelGpt<T>::initialize()
{
    gpt_context_decoder_ = new ParallelGptContextDecoder<T>(0,
                                                            0,
                                                            head_num_,
                                                            size_per_head_,
                                                            inter_size_,
                                                            num_layer_,
                                                            expert_num_,
                                                            moe_k_,
                                                            moe_layer_index_,
                                                            layernorm_eps_,
                                                            gpt_variant_params_,
                                                            tensor_para_,
                                                            pipeline_para_,
                                                            stream_,
                                                            cublas_wrapper_,
                                                            allocator_,
                                                            is_free_buffer_after_forward_,
                                                            is_context_qk_buf_float_,
                                                            attention_type_,
                                                            sparse_,
                                                            int8_mode_,
                                                            custom_all_reduce_comm_,
                                                            enable_custom_all_reduce_);

    gpt_decoder_ = new ParallelGptDecoder<T>(0,
                                             head_num_,
                                             size_per_head_,
                                             inter_size_,
                                             num_layer_,
                                             expert_num_,
                                             moe_k_,
                                             moe_layer_index_,
                                             layernorm_eps_,
                                             gpt_variant_params_,
                                             tensor_para_,
                                             pipeline_para_,
                                             stream_,
                                             cublas_wrapper_,
                                             allocator_,
                                             is_free_buffer_after_forward_,
                                             sparse_,
                                             int8_mode_,
                                             custom_all_reduce_comm_,
                                             enable_custom_all_reduce_);

    dynamic_decode_layer_ = new DynamicDecodeLayer<float>(vocab_size_,
                                                          vocab_size_padded_,
                                                          0,  // end_id, deprecated
                                                          stream_,
                                                          cublas_wrapper_,
                                                          allocator_,
                                                          is_free_buffer_after_forward_,
                                                          cuda_device_prop_);
}

template<typename T>
void ParallelGpt<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void ParallelGpt<T>::allocateBuffer(size_t batch_size,
                                    size_t beam_width,
                                    size_t max_session_len,
                                    size_t memory_len,
                                    size_t max_input_len,
                                    bool   is_return_context_cum_log_probs)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    const size_t batchxbeam       = batch_size * beam_width;
    const size_t local_batch_size = getLocalBatchSize(batch_size, 1, pipeline_para_.world_size_);
    FT_CHECK(batch_size % local_batch_size == 0);
    const size_t num_microbatches = batch_size / local_batch_size;

    const size_t self_cache_size =
        (num_layer_ / pipeline_para_.world_size_) * batchxbeam * memory_len * hidden_units_ / tensor_para_.world_size_;

    if (vocab_size_ != vocab_size_padded_) {
        padded_embedding_kernel_ =
            (T*)(allocator_->reMalloc(padded_embedding_kernel_, sizeof(T) * hidden_units_ * vocab_size_padded_, true));
        padded_embedding_kernel_ptr_ = padded_embedding_kernel_;
    }

    tiled_input_attention_mask_ = (T*)(allocator_->reMalloc(
        tiled_input_attention_mask_, sizeof(T) * batchxbeam * max_input_len * max_input_len, false));
    decoder_input_buf_ = (T*)(allocator_->reMalloc(decoder_input_buf_, sizeof(T) * batchxbeam * hidden_units_, false));
    decoder_normed_input_buf_ =
        (T*)(allocator_->reMalloc(decoder_normed_input_buf_, sizeof(T) * batchxbeam * hidden_units_, false));
    decoder_output_buf_ =
        (T*)(allocator_->reMalloc(decoder_output_buf_, sizeof(T) * batchxbeam * hidden_units_, false));
    normed_decoder_output_buf_ =
        (T*)(allocator_->reMalloc(normed_decoder_output_buf_, sizeof(T) * batchxbeam * hidden_units_, false));
    logits_buf_ = (float*)(allocator_->reMalloc(logits_buf_, sizeof(float) * batchxbeam * vocab_size_padded_, false));
    nccl_logits_buf_ =
        (float*)(allocator_->reMalloc(nccl_logits_buf_, sizeof(float) * batchxbeam * vocab_size_padded_, false));
    cum_log_probs_    = (float*)(allocator_->reMalloc(cum_log_probs_, sizeof(float) * batchxbeam, false));
    finished_buf_     = (bool*)(allocator_->reMalloc(finished_buf_, sizeof(bool) * batchxbeam, false));
    sequence_lengths_ = (int*)(allocator_->reMalloc(sequence_lengths_, sizeof(int) * batchxbeam, false));

    key_cache_   = (T*)(allocator_->reMalloc(key_cache_, sizeof(T) * self_cache_size * 2, true));
    value_cache_ = key_cache_ + self_cache_size;
    if (beam_width > 1) {
        cache_indirections_[0] =
            (int*)(allocator_->reMalloc(cache_indirections_[0], sizeof(int) * batchxbeam * memory_len * 2, true));
        cache_indirections_[1] = cache_indirections_[0] + batchxbeam * memory_len;
    }

    tiled_input_ids_buf_ =
        (int*)(allocator_->reMalloc(tiled_input_ids_buf_, sizeof(int) * batchxbeam * max_session_len, true));
    tiled_input_lengths_buf_ = (int*)(allocator_->reMalloc(tiled_input_lengths_buf_, sizeof(int) * batchxbeam, true));

    // prompt_learning weight batch ptrs
    prompt_learning_weight_batch_ =
        (const T**)(allocator_->reMalloc(prompt_learning_weight_batch_, sizeof(T*) * batchxbeam, false));
    tiled_prompt_lengths_buf_ =
        (int*)(allocator_->reMalloc(tiled_prompt_lengths_buf_, sizeof(int) * batchxbeam, false));

    start_ids_buf_ = (int*)(allocator_->reMalloc(start_ids_buf_, sizeof(int) * batch_size, false));
    end_ids_buf_   = (int*)(allocator_->reMalloc(end_ids_buf_, sizeof(int) * batch_size, false));

    transposed_output_ids_buf_ =
        (int*)(allocator_->reMalloc(transposed_output_ids_buf_, sizeof(int) * batchxbeam * max_session_len, true));
    output_ids_buf_ = (int*)(allocator_->reMalloc(output_ids_buf_, sizeof(int) * batchxbeam * max_session_len, true));
    parent_ids_buf_ = (int*)(allocator_->reMalloc(parent_ids_buf_, sizeof(int) * batchxbeam * max_session_len, true));
    seq_limit_len_  = (uint32_t*)(allocator_->reMalloc(seq_limit_len_, sizeof(uint32_t) * batch_size, false));
    tiled_masked_tokens_ =
        (bool*)(allocator_->reMalloc(tiled_masked_tokens_, sizeof(bool) * batchxbeam * memory_len, true));

    context_decoder_input_buf_  = (T*)(allocator_->reMalloc(
        context_decoder_input_buf_, sizeof(T) * batchxbeam * max_input_len * hidden_units_, false));
    context_decoder_output_buf_ = (T*)(allocator_->reMalloc(
        context_decoder_output_buf_, sizeof(T) * batchxbeam * max_input_len * hidden_units_, false));
    output_log_probs_buf_ =
        (float*)(allocator_->reMalloc(output_log_probs_buf_, sizeof(float) * batchxbeam * max_session_len, false));

    if (gpt_variant_params_.has_pre_decoder_layernorm) {
        context_decoder_normed_input_buf_ = (T*)allocator_->reMalloc(
            context_decoder_normed_input_buf_, sizeof(T) * batchxbeam * max_input_len * hidden_units_, false);
        decoder_normed_input_buf_ =
            (T*)allocator_->reMalloc(decoder_normed_input_buf_, sizeof(T) * batchxbeam * hidden_units_, false);
    }

    if (gpt_variant_params_.use_attention_linear_bias) {
        linear_bias_slopes_ = (T*)(allocator_->reMalloc(linear_bias_slopes_, sizeof(T) * head_num_, false));
    }

    if (is_return_context_cum_log_probs) {
        lp_normed_decoder_output_buf_ = (T*)allocator_->reMalloc(
            lp_normed_decoder_output_buf_, sizeof(T) * batchxbeam * max_input_len * hidden_units_);
        lp_logits_buf_      = (float*)allocator_->reMalloc(lp_logits_buf_,
                                                      sizeof(float) * batchxbeam * max_input_len * vocab_size_padded_);
        lp_nccl_logits_buf_ = (float*)allocator_->reMalloc(
            lp_nccl_logits_buf_, sizeof(float) * batchxbeam * max_input_len * vocab_size_padded_);
        lp_logprob_buf_ = (float*)allocator_->reMalloc(lp_logprob_buf_, sizeof(float) * batchxbeam * max_input_len);
    }
    if (shared_contexts_ratio_ > 0.0f) {
        shared_contexts_idx_  = (int*)allocator_->reMalloc(shared_contexts_idx_, batch_size * sizeof(int), false);
        batch_to_compact_idx_ = (int*)allocator_->reMalloc(batch_to_compact_idx_, batchxbeam * sizeof(int), false);
        compact_idx_          = (int*)allocator_->reMalloc(compact_idx_, batch_size * sizeof(int), false);
        compact_size_         = (int*)allocator_->reMalloc(compact_size_, sizeof(int), false);
    }
    microbatch_should_stop_ =
        (bool*)allocator_->reMalloc(microbatch_should_stop_, sizeof(bool) * num_microbatches, true, true);
    tiled_total_padding_count_ =
        (int*)allocator_->reMalloc(tiled_total_padding_count_, batchxbeam * sizeof(int), false);

    is_allocate_buffer_ = true;
}

template<typename T>
void ParallelGpt<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        if (vocab_size_ != vocab_size_padded_) {
            padded_embedding_kernel_ptr_ = nullptr;
            allocator_->free((void**)(&padded_embedding_kernel_));
        }

        allocator_->free((void**)(&tiled_input_attention_mask_));
        allocator_->free((void**)(&decoder_input_buf_));
        allocator_->free((void**)(&decoder_output_buf_));
        allocator_->free((void**)(&normed_decoder_output_buf_));
        allocator_->free((void**)(&logits_buf_));
        allocator_->free((void**)(&nccl_logits_buf_));
        allocator_->free((void**)(&cum_log_probs_));
        allocator_->free((void**)(&finished_buf_));
        allocator_->free((void**)(&sequence_lengths_));

        allocator_->free((void**)(&key_cache_));
        if (cache_indirections_[0] != nullptr) {
            allocator_->free((void**)(&cache_indirections_)[0]);
        }

        allocator_->free((void**)(&tiled_input_ids_buf_));
        allocator_->free((void**)(&tiled_input_lengths_buf_));

        allocator_->free((void**)(&prompt_learning_weight_batch_));
        allocator_->free((void**)(&tiled_prompt_lengths_buf_));

        allocator_->free((void**)(&transposed_output_ids_buf_));
        allocator_->free((void**)(&output_ids_buf_));
        allocator_->free((void**)(&parent_ids_buf_));
        allocator_->free((void**)(&tiled_masked_tokens_));

        allocator_->free((void**)(&seq_limit_len_));

        allocator_->free((void**)(&start_ids_buf_));
        allocator_->free((void**)(&end_ids_buf_));

        allocator_->free((void**)(&context_decoder_input_buf_));
        allocator_->free((void**)(&context_decoder_output_buf_));
        allocator_->free((void**)(&output_log_probs_buf_));

        if (gpt_variant_params_.has_pre_decoder_layernorm) {
            allocator_->free((void**)(&context_decoder_normed_input_buf_));
            allocator_->free((void**)(&decoder_normed_input_buf_));
        }
        if (gpt_variant_params_.use_attention_linear_bias) {
            allocator_->free((void**)(&linear_bias_slopes_));
        }

        allocator_->free((void**)(&lp_normed_decoder_output_buf_));
        allocator_->free((void**)(&lp_logits_buf_));
        allocator_->free((void**)(&lp_nccl_logits_buf_));
        allocator_->free((void**)(&lp_logprob_buf_));

        allocator_->free((void**)(&microbatch_should_stop_), true);

        if (shared_contexts_ratio_ > 0.0f) {
            allocator_->free((void**)(&shared_contexts_idx_));
            allocator_->free((void**)(&compact_size_));
        }
        allocator_->free((void**)(&tiled_total_padding_count_));

        is_allocate_buffer_ = false;
    }
}

template<typename T>
ParallelGpt<T>::ParallelGpt(size_t                              max_batch_size,
                            size_t                              max_seq_len,
                            size_t                              max_input_len,
                            size_t                              beam_width,
                            size_t                              head_num,
                            size_t                              size_per_head,
                            size_t                              inter_size,
                            size_t                              num_layer,
                            size_t                              expert_num,
                            size_t                              moe_k,
                            std::vector<int64_t>                moe_layer_index,
                            size_t                              vocab_size,
                            int                                 start_id,
                            int                                 end_id,
                            int                                 prompt_learning_start_id,
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
                            cudaDeviceProp*                     cuda_device_prop,
                            AttentionType                       attention_type,
                            bool                                sparse,
                            int                                 int8_mode,
                            std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                            int                                 enable_custom_all_reduce,
                            float                               shared_contexts_ratio):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop, sparse),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    expert_num_(expert_num),
    moe_k_(moe_k),
    moe_layer_index_(moe_layer_index),
    vocab_size_(vocab_size),
    start_id_(start_id),
    end_id_(end_id),
    prompt_learning_start_id_(prompt_learning_start_id),
    prompt_learning_type_(prompt_learning_type),
    layernorm_eps_(gpt_variant_params.layernorm_eps),
    gpt_variant_params_(gpt_variant_params),
    beam_search_diversity_rate_(beam_search_diversity_rate),
    hidden_units_(head_num_ * size_per_head),
    top_k_(top_k),
    top_p_(top_p),
    random_seed_(random_seed),
    temperature_(temperature),
    len_penalty_(len_penalty),
    repetition_penalty_(repetition_penalty),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para),
    local_head_num_(head_num / tensor_para.world_size_),
    attention_type_(attention_type),
    int8_mode_(int8_mode),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce),
    shared_contexts_ratio_(shared_contexts_ratio)
{
    int local_vacab_size = ceil(vocab_size_ / 1.f / tensor_para_.world_size_);
    if (std::is_same<half, T>::value
#ifdef ENABLE_BF16
        || std::is_same<__nv_bfloat16, T>::value
#endif
    ) {
        local_vacab_size = ceil(local_vacab_size / 8.f) * 8;
    }
    vocab_size_padded_ = (size_t)local_vacab_size * tensor_para_.world_size_;
    initialize();
}

template<typename T>
ParallelGpt<T>::ParallelGpt(ParallelGpt<T> const& gpt):
    BaseLayer(gpt),
    head_num_(gpt.head_num_),
    size_per_head_(gpt.size_per_head_),
    inter_size_(gpt.inter_size_),
    num_layer_(gpt.num_layer_),
    expert_num_(gpt.expert_num_),
    moe_k_(gpt.moe_k_),
    moe_layer_index_(gpt.moe_layer_index_),
    vocab_size_(gpt.vocab_size_),
    start_id_(gpt.start_id_),
    end_id_(gpt.end_id_),
    prompt_learning_start_id_(gpt.prompt_learning_start_id_),
    prompt_learning_type_(gpt.prompt_learning_type_),
    beam_search_diversity_rate_(gpt.beam_search_diversity_rate_),
    layernorm_eps_(gpt.gpt_variant_params_.layernorm_eps),
    gpt_variant_params_(gpt.gpt_variant_params_),
    hidden_units_(gpt.hidden_units_),
    top_k_(gpt.top_k_),
    top_p_(gpt.top_p_),
    random_seed_(gpt.random_seed_),
    temperature_(gpt.temperature_),
    len_penalty_(gpt.len_penalty_),
    repetition_penalty_(gpt.repetition_penalty_),
    tensor_para_(gpt.tensor_para_),
    pipeline_para_(gpt.pipeline_para_),
    local_head_num_(gpt.local_head_num_),
    vocab_size_padded_(gpt.vocab_size_padded_),
    attention_type_(gpt.attention_type_),
    int8_mode_(gpt.int8_mode_),
    custom_all_reduce_comm_(gpt.custom_all_reduce_comm_),
    enable_custom_all_reduce_(gpt.enable_custom_all_reduce_),
    shared_contexts_ratio_(gpt.shared_contexts_ratio_)
{
    initialize();
}

template<typename T>
ParallelGpt<T>::~ParallelGpt()
{
    delete gpt_decoder_;
    delete gpt_context_decoder_;
    delete dynamic_decode_layer_;
    freeBuffer();
}

template<typename T>
void ParallelGpt<T>::computeContextCumLogProbs(float*                      cum_log_probs,
                                               const T*                    context_decoder_outputs,
                                               const int*                  input_ids,
                                               const int*                  input_lengths,
                                               const size_t                batch_size,
                                               const size_t                beam_width,
                                               const size_t                max_input_length,
                                               const ParallelGptWeight<T>* gpt_weights)
{
    // Compute the log probabilties of prompt inputs.
    //
    // cum_log_probs [batch_size, beam_width]
    // context_decoder_outputs [batch_size * beam_width, max_input_length, hidden_units]
    // input_ids [batch_size * beam_width, max_input_length]; input ids.
    // input_lengths [batch_size, beam_width]; input lengths.
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    const size_t batchxbeam      = batch_size * beam_width;
    const size_t n_hidden_states = batchxbeam * max_input_length;

    const cudaDataType_t cublas_type = getCudaDataType<T>();

    if (pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) {
        // normed decoder output [batch_size * beam_width, max_input_length, hidden_units_]
        invokeGeneralLayerNorm(lp_normed_decoder_output_buf_,
                               context_decoder_outputs,
                               gpt_weights->post_decoder_layernorm.gamma,
                               gpt_weights->post_decoder_layernorm.beta,
                               layernorm_eps_,
                               n_hidden_states,
                               hidden_units_,
                               (float*)nullptr,
                               0,
                               stream_);
        sync_check_cuda_error();
        if (tensor_para_.world_size_ == 1) {
            float alpha = 1.0f;
            float beta  = 0.0f;
            cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                  CUBLAS_OP_N,
                                  vocab_size_padded_,  // n
                                  n_hidden_states,
                                  hidden_units_,  // k
                                  &alpha,
                                  padded_embedding_kernel_ptr_,
                                  cublas_type,
                                  hidden_units_,  // k
                                  lp_normed_decoder_output_buf_,
                                  cublas_type,
                                  hidden_units_,  // k
                                  &beta,
                                  lp_logits_buf_,
                                  CUDA_R_32F,
                                  vocab_size_padded_, /* n */
                                  CUDA_R_32F,
                                  cublasGemmAlgo_t(-1));
            sync_check_cuda_error();
        }
        else {
            FT_CHECK(vocab_size_padded_ % tensor_para_.world_size_ == 0);
            const int local_vocab_size = vocab_size_padded_ / tensor_para_.world_size_;
            float     alpha            = 1.0f;
            float     beta             = 0.0f;
            cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                  CUBLAS_OP_N,
                                  local_vocab_size,  // n
                                  n_hidden_states,
                                  hidden_units_,  // k
                                  &alpha,
                                  padded_embedding_kernel_ptr_ + tensor_para_.rank_ * local_vocab_size * hidden_units_,
                                  cublas_type,
                                  hidden_units_,  // k
                                  lp_normed_decoder_output_buf_,
                                  cublas_type,
                                  hidden_units_,  // k
                                  &beta,
                                  lp_nccl_logits_buf_ + tensor_para_.rank_ * n_hidden_states * local_vocab_size,
                                  CUDA_R_32F,
                                  local_vocab_size, /* n */
                                  CUDA_R_32F,
                                  cublasGemmAlgo_t(-1));
            sync_check_cuda_error();
            ftNcclAllGather(lp_nccl_logits_buf_,
                            lp_nccl_logits_buf_,
                            n_hidden_states * local_vocab_size,
                            tensor_para_.rank_,
                            tensor_para_,
                            stream_);
            check_cuda_error(cudaStreamSynchronize(stream_));
            sync_check_cuda_error();

            invokeTransposeAxis01(lp_logits_buf_,
                                  lp_nccl_logits_buf_,
                                  tensor_para_.world_size_,
                                  n_hidden_states,
                                  local_vocab_size,
                                  stream_);
            sync_check_cuda_error();
        }
    }

    invokeLogProbFromLogits(cum_log_probs,
                            lp_logits_buf_,
                            input_ids,
                            input_lengths,
                            max_input_length,
                            batchxbeam,
                            vocab_size_,
                            vocab_size_padded_,
                            lp_logprob_buf_,
                            sizeof(float) * batchxbeam * max_input_length,
                            stream_,
                            true);
    sync_check_cuda_error();
}

template<typename T>
void ParallelGpt<T>::registerCallback(callback_sig* fn, void* ctx)
{
    token_generated_cb_  = fn;
    token_generated_ctx_ = ctx;
}

template<typename T>
void ParallelGpt<T>::unRegisterCallback()
{
    token_generated_cb_  = nullptr;
    token_generated_ctx_ = nullptr;
}

template<typename T>
void ParallelGpt<T>::forward(std::vector<Tensor>*        output_tensors,
                             const std::vector<Tensor>*  input_tensors,
                             const ParallelGptWeight<T>* gpt_weights)
{
    // input_tensors:
    //      input_ids [batch_size, max_input_length]
    //      input_lengths [batch_size]
    //      max_output_seq_len [1] on cpu

    // output_tensors:
    //      output_ids [batch_size, beam, max_output_seq_len]
    //      sequence_length [batch_size, beam]
    //      output_log_probs [batch_size, beam, request_output_seq_len], must be float*.
    //          It leads to additional computing cost. If we don't need this result, please put nullptr
    //      cum_log_probs [batch_size, beam], must be float*, optional
    //          The cumulative log probability of generated sequences. It leads additional computing cost.

    // Step is from max_input_length ~ max_output_seq_len,
    // When step = k,  we put output ids and caches at step k, and the sequence_length would be k - 1 before
    // complete this step.
    // When there is no input_ids, put the start token at step 0 of output_ids_buf_. After forward, only copy
    // the step 1 ~ max_output_seq_len of output_ids_buf_ to output_tensors->at(0).data

    std::unordered_map<std::string, Tensor> input_tensors_map{{"input_ids", input_tensors->at(0)},
                                                              {"input_lengths", input_tensors->at(1)},
                                                              {"max_output_seq_len", input_tensors->at(2)}};
    input_tensors_map.insert({"random_seed", {MEMORY_CPU, TYPE_INT32, {1}, &random_seed_}});
    input_tensors_map.insert({"runtime_top_k", {MEMORY_CPU, TYPE_UINT32, {1}, &top_k_}});
    input_tensors_map.insert({"runtime_top_p", {MEMORY_CPU, TYPE_FP32, {1}, &top_p_}});

    std::unordered_map<std::string, Tensor> output_tensors_map{{"output_ids", output_tensors->at(0)},
                                                               {"sequence_length", output_tensors->at(1)},
                                                               {"output_log_probs", output_tensors->at(2)}};
    if (output_tensors->size() > 3) {
        output_tensors_map.insert({"cum_log_probs", output_tensors->at(4)});
    }
    forward(&output_tensors_map, &input_tensors_map, gpt_weights);
}

template<typename T>
void ParallelGpt<T>::forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                             const std::unordered_map<std::string, Tensor>* input_tensors,
                             const ParallelGptWeight<T>*                    gpt_weights)
{
    // input_tensors:
    //      input_ids [batch_size, max_input_length]
    //      input_lengths [batch_size]
    //      input_lengths_h [batch_size] on cpu, optional
    //      prompt_learning_task_name_ids [batch_size] on cpu
    //      output_seq_len [batch_size] on cpu
    //      stop_words_list [batch_size, 2, stop_words_length], optional
    //      bad_words_list [2, bad_words_length] or [batch_size, 2, bad_words_length], optional
    //      start_id [batch_size] on cpu, optional
    //      end_id [batch_size] on cpu, optional
    //      runtime_top_k [1] or [batch_size] on cpu, optional, uint.
    //      runtime_top_p [1] or [batch_size] on cpu, optional, float.
    //      beam_search_diversity_rate [1] or [batch_size] on cpu, optional, float.
    //      temperature [1] or [batch_size] on cpu, optional, float.
    //      len_penalty [1] or [batch_size] on cpu, optional, float.
    //      repetition_penalty [1] or [batch_size] on cpu, optional, float.
    //      presence_penalty [1] or [batch_size] on cpu, optional, float.
    //          Only one of repetition and presence penalties is allowed.
    //      min_length [1] or [batch_size] on cpu, optional, int
    //      random_seed [1] or [batch_size] on cpu, optional, unsigned long long int.
    //      request_prompt_lengths [batch_size], optional
    //      request_prompt_lengths_h [batch_size], cpu, optional
    //      request_prompt_embedding [batch_size, max_prompt_length, hidden_units], float, optional
    //      request_prompt_type [batch_size], int, optional
    //      is_return_context_cum_log_probs [1] on cpu, bool, optional
    //      session_len [1] on cpu, uint32, optional
    //      memory_len [1] on cpu, uint32, optional
    //      continue_gen [1] on cpu, bool, optional
    //      is_return_context_embeddings [1] on cpu, bool, optional
    //      top_p_decay [batch_size] on gpu, float, optional
    //      top_p_min [batch_size] on gpu, float, optional
    //      top_p_reset_ids [batch_size] on gpu, uint32, optional

    // output_tensors:
    //      output_ids [batch_size, beam_width, max_output_seq_len]
    //      sequence_length [batch_size, beam_width]
    //      response_input_lengths [batch_size, beam_width], optional
    //      output_log_probs [batch_size, beam_width, request_output_seq_len], must be float*.
    //          optional. It leads to additional computing cost. If we don't need this result, don't put it.
    //      cum_log_probs [batch_size, beam_width], must be float*. optional.
    //          The cumulative log probability of generated sequences. It may lead to additional computing cost.
    //      context_embeddings [batch_size, hidden_units], must be float*, optional

    // Step is from max_input_length ~ max_output_seq_len,
    // When step = k,  we put output ids and caches at step k, and the sequence_length would be k - 1 before
    // complete this step.
    // When there is no input_ids, put the start token at step 0 of output_ids_buf_. After forward, only copy
    // the step 1 ~ max_output_seq_len of output_ids_buf_ to output_tensors->at(0).data

    FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    FT_CHECK_WITH_INFO(input_tensors->size() >= 3, "input_tensors->size() >= 3");
    FT_CHECK_WITH_INFO(output_tensors->size() >= 2, "output_tensors->size() >= 2");
    FT_CHECK(input_tensors->at("input_ids").shape.size() == 2);
    FT_CHECK(input_tensors->at("input_lengths").shape.size() == 1);
    FT_CHECK(input_tensors->find("output_seq_len") != input_tensors->end()
             && input_tensors->at("output_seq_len").shape.size() == 1);
    FT_CHECK(output_tensors->at("output_ids").shape.size() == 3);
    FT_CHECK(output_tensors->at("sequence_length").shape.size() == 2);
    FT_CHECK_WITH_INFO(input_tensors->at("input_ids").shape[0] == output_tensors->at("output_ids").shape[0],
                       "input_tensors->at(\"input_ids\").shape[0] == output_tensors->at(\"output_ids\").shape[0]");

    // Used when inputs do not contain random_seed
    const size_t batch_size = output_tensors->at("output_ids").shape[0];
    const size_t beam_width = output_tensors->at("output_ids").shape[1];
    FT_CHECK_WITH_INFO(output_tensors->count("cum_log_probs") == 0
                           || output_tensors->at("cum_log_probs").size() == batch_size * beam_width,
                       "The shape of cum_log_probs should match with batch_size x beam_width if provided.");
    int max_input_length = input_tensors->at("input_ids").shape[1];

    bool       continue_gen = input_tensors->find("continue_gen") != input_tensors->end() ?
                                  input_tensors->at("continue_gen").getVal<bool>() :
                                  false;
    const bool is_return_context_embeddings =
        input_tensors->find("is_return_context_embeddings") != input_tensors->end()
        && input_tensors->at("is_return_context_embeddings").getVal<bool>();
    if (is_return_context_embeddings) {
        FT_CHECK_WITH_INFO(output_tensors->find("context_embeddings") != output_tensors->end(),
                           "When requesting context embeddings, a context embeddings output tensors must be provided");
    }

    const int initial_step    = continue_gen ? step_ : 0;
    int       max_context_len = max_input_length + initial_step;

    // NOTE: the input already contains the p/prompt-tunning tokens ids for p/prompt tuning task
    // prompt_learning_task_name_ids are used by both p/prompt-tunning and prefix_prompt task
    const int* prompt_learning_task_name_ids =
        input_tensors->count("prompt_learning_task_name_ids") ?
            input_tensors->at("prompt_learning_task_name_ids").getPtr<const int>() :
            nullptr;

    FT_CHECK_WITH_INFO(
        !(prompt_learning_task_name_ids != nullptr
          && (prompt_learning_type_ == PromptLearningType::no_prompt
              || prompt_learning_type_ == PromptLearningType::soft_prompt)),
        "prompt_learning_type is prefix_prompt either p_prompt_tuning when prompt_learning_task_name_ids are provided.");

    PromptLearningType request_prompt_type = PromptLearningType::no_prompt;
    int                valid_prompt_inputs = input_tensors->count("request_prompt_type")
                              + input_tensors->count("request_prompt_lengths")
                              + input_tensors->count("request_prompt_embedding");

    if (valid_prompt_inputs == 3) {
        request_prompt_type = static_cast<PromptLearningType>(input_tensors->at("request_prompt_type").getVal<int>());
        if (prompt_learning_task_name_ids != nullptr) {
            FT_LOG_INFO("Apply prompt embedding from input, will ignore task name ids");
        }
    }
    else if (valid_prompt_inputs > 0) {
        FT_LOG_WARNING(
            "Prompts not applied: request_prompt_embedding, request_prompt_lengths, request_prompt_type are all needed!");
    }
    if (request_prompt_type == PromptLearningType::prefix_prompt) {
        FT_LOG_WARNING("Request prompt doesn't support prefix prompt currently!");
    }

    // whether or not use prompt embeddings from the request.
    // If true, staticlly loaded prompts weights during model loading and task name ids will be ignored
    bool use_request_p_prompt_embedding = request_prompt_type == PromptLearningType::p_prompt_tuning;
    int  max_request_p_prompt_length =
        use_request_p_prompt_embedding ? input_tensors->at("request_prompt_embedding").shape[1] : 0;
    // p_prompt tuning: input and prompt are concatnenated (not separate),
    const uint32_t* input_lengths_h = input_tensors->count("input_lengths_h") ?
                                          input_tensors->at("input_lengths_h").getPtr<const uint32_t>() :
                                          nullptr;

    size_t max_input_without_prompt_length = max_context_len;
    if (use_request_p_prompt_embedding && input_lengths_h != nullptr
        && input_tensors->count("request_prompt_lengths_h")) {

        const uint32_t* request_prompt_lengths_h =
            input_tensors->at("request_prompt_lengths_h").getPtr<const uint32_t>();
        max_input_without_prompt_length = input_lengths_h[0] - request_prompt_lengths_h[0];
        for (int bs_id = 1; bs_id < batch_size; ++bs_id) {
            max_input_without_prompt_length = std::max(size_t(input_lengths_h[bs_id] - request_prompt_lengths_h[bs_id]),
                                                       max_input_without_prompt_length);
        }
    }

    has_prefix_prompt_ =
        (prompt_learning_task_name_ids != nullptr && prompt_learning_type_ == PromptLearningType::prefix_prompt);
    has_p_prompt_tuning_ =
        prompt_learning_task_name_ids != nullptr && prompt_learning_type_ == PromptLearningType::p_prompt_tuning
        || use_request_p_prompt_embedding;
    bool use_loaded_p_prompt_embedding = has_p_prompt_tuning_ && !use_request_p_prompt_embedding;
    has_prefix_soft_prompt_            = request_prompt_type == PromptLearningType::soft_prompt;

    // NOTE: soft prompt
    FT_CHECK_WITH_INFO(!(has_prefix_soft_prompt_ && continue_gen),
                       "Interactive Generations cannot work with prefix_soft_prompt !");
    const size_t max_prefix_soft_prompt_length =
        has_prefix_soft_prompt_ ? input_tensors->at("request_prompt_embedding").shape[1] : 0;
    const size_t limit_len_offset = max_prefix_soft_prompt_length + (max_input_length == 0 ? 1 : 0);
    const size_t gen_len          = input_tensors->at("output_seq_len").max<uint32_t>() + limit_len_offset;

    size_t session_len = 0;
    if (continue_gen) {
        session_len = session_len_;  // Record the size of allocated buffer in previous round.
    }
    else if (input_tensors->find("session_len") != input_tensors->end()) {
        session_len = input_tensors->at("session_len").getVal<uint32_t>();  // Use for allocate buffer in first round.
    }
    else {
        session_len = gen_len;  // When the interactive generation mode is disabled.
    }
    session_len_ = session_len;
    FT_CHECK_WITH_INFO(
        gen_len + initial_step <= session_len,
        fmtstr("Session size too low (%d) vs. total output size (%d)", session_len, gen_len + initial_step));
    size_t memory_len = 0;
    if (continue_gen) {
        memory_len = memory_len_;  // Record the size of allocated buffer in previous round.
    }
    else if (input_tensors->find("memory_len") != input_tensors->end()) {
        memory_len = input_tensors->at("memory_len").getVal<uint32_t>();  // Use for allocate buffer in first round.
    }
    else {
        memory_len = session_len;  // When the interactive generation mode is disabled.
    }
    memory_len_ = memory_len;
    /* TODO: could remove this constraint by changing how context decoder operates */
    FT_CHECK_WITH_INFO(max_input_length <= memory_len,
                       fmtstr("Memory size too low (%d) vs. input length (%d)", memory_len, max_input_length));

    if (memory_len < session_len) {
        FT_LOG_WARNING("memory_len (%d) is less than session_len (%d). "
                       "Note that this reduces the memory cost of k/v cache, but may hurt the accuracy.",
                       memory_len,
                       session_len);
    }
    else if (memory_len > session_len) {
        FT_LOG_WARNING("memory_len (%d) is larger than session_len (%d). "
                       "This may lead to additional memory cost. Suggest to use smaller memory_len.",
                       memory_len,
                       session_len);
    }

    if (gpt_variant_params_.has_positional_encoding && session_len_ > gpt_weights->getMaxSeqLen()) {
        FT_LOG_ERROR("The session_len_ (%d) of request is longer than max_seq_len (%d) of embedding table."
                     " This is a invalid input. Setting the session_len_ to %d.",
                     session_len_,
                     gpt_weights->getMaxSeqLen(),
                     gpt_weights->getMaxSeqLen());
        session_len_ = gpt_weights->getMaxSeqLen();
    }

    const bool is_return_context_cum_log_probs = input_tensors->count("is_return_context_cum_log_probs") > 0
                                                 && input_tensors->at("is_return_context_cum_log_probs").getVal<bool>();
    if (is_return_context_cum_log_probs) {
        FT_CHECK_WITH_INFO(output_tensors->count("cum_log_probs")
                               && output_tensors->at("cum_log_probs").data != nullptr,
                           "`cum_log_probs` must be provided in `output_tensors` in order to enable "
                           "the cumulative log probability computation of input contexts.");
    }

    PUSH_RANGE("buffer allocation");
    if (!continue_gen) {
        allocateBuffer(batch_size,
                       beam_width,
                       session_len,
                       memory_len,
                       max_input_length + max_prefix_soft_prompt_length,
                       is_return_context_cum_log_probs);
        sync_check_cuda_error();
    }
    setSeqLimitLen(seq_limit_len_, input_tensors->at("output_seq_len"), limit_len_offset, batch_size);
    POP_RANGE;

    const DataType       data_type      = getTensorType<T>();
    const cudaDataType_t gemm_data_type = getCudaDataType<T>();

    const std::vector<size_t> self_k_cache_shape = {num_layer_ / pipeline_para_.world_size_,
                                                    batch_size * beam_width,
                                                    local_head_num_,
                                                    size_per_head_ / (16 / sizeof(T)),
                                                    memory_len,
                                                    16 / sizeof(T)};
    const std::vector<size_t> self_v_cache_shape = {
        num_layer_ / pipeline_para_.world_size_, batch_size * beam_width, local_head_num_, memory_len, size_per_head_};

    {
        TensorMap input_map(*input_tensors);

        PUSH_RANGE("dynamic decode setup");
        dynamic_decode_layer_->setup(batch_size, beam_width, &input_map);
        handleOptArg(&input_map, "start_id", start_ids_buf_, start_id_, batch_size);
        handleOptArg(&input_map, "end_id", end_ids_buf_, end_id_, batch_size);
        POP_RANGE;
    }

    if (gpt_variant_params_.use_attention_linear_bias) {
        PUSH_RANGE("build alibi slopes");
        invokeBuildAlibiSlopes(linear_bias_slopes_, head_num_, stream_);
        POP_RANGE;
    }

    if (continue_gen) {
        PUSH_RANGE("input tiling and init");
        invokeTileGptInputs(tiled_input_ids_buf_,
                            tiled_input_lengths_buf_,
                            input_tensors->at("input_ids").getPtr<int>(),
                            input_tensors->at("input_lengths").getPtr<const int>(),
                            batch_size,
                            beam_width,
                            max_input_length,
                            stream_);
        invokePlusScalar(tiled_input_lengths_buf_, initial_step, batch_size * beam_width, stream_);
        sync_check_cuda_error();
        invokeDecodingInitialize(finished_buf_,
                                 sequence_lengths_,
                                 nullptr,
                                 cum_log_probs_,
                                 start_ids_buf_,
                                 batch_size,
                                 beam_width,
                                 initial_step - 1,
                                 stream_);
        invokeTransposeAxis01(output_ids_buf_ + initial_step * batch_size * beam_width,
                              tiled_input_ids_buf_,
                              batch_size * beam_width,
                              max_input_length,
                              1,
                              stream_);
        POP_RANGE;
    }
    else {
        // TODO(bhsueh) Initilaize them in one kernel
        // initialize the output ids and parent ids
        PUSH_RANGE("initialize output and parent ids");
        cudaMemsetAsync(output_ids_buf_, 0, sizeof(int) * batch_size * beam_width * session_len, stream_);
        cudaMemsetAsync(parent_ids_buf_, 0, sizeof(int) * batch_size * beam_width * session_len, stream_);
        cudaMemsetAsync(tiled_masked_tokens_, false, sizeof(bool) * batch_size * beam_width * memory_len, stream_);
        cudaMemsetAsync(tiled_total_padding_count_, 0, sizeof(int) * batch_size * beam_width, stream_);
        if (beam_width > 1) {
            cudaMemsetAsync(cache_indirections_[0], 0, 2 * sizeof(int) * batch_size * beam_width * memory_len, stream_);
        }
        sync_check_cuda_error();
        POP_RANGE;

        PUSH_RANGE("padded embedding kernel init");
        if (vocab_size_ == vocab_size_padded_) {
            padded_embedding_kernel_ptr_ = gpt_weights->post_decoder_embedding.kernel;
        }
        else {
            cudaAutoCpy(padded_embedding_kernel_,
                        gpt_weights->post_decoder_embedding.kernel,
                        vocab_size_ * hidden_units_,
                        stream_);
            sync_check_cuda_error();
        }
        POP_RANGE;

        int  compact_size;
        bool use_shared_contexts = (shared_contexts_ratio_ > 0.0f) && (max_input_length >= 1) && (batch_size > 1);
        PUSH_RANGE("find context dups");
        if (use_shared_contexts) {
            invokeFindContextDups(shared_contexts_idx_,
                                  batch_to_compact_idx_,
                                  compact_idx_,
                                  compact_size_,
                                  input_tensors->at("input_ids").getPtr<int>(),
                                  batch_size,
                                  beam_width,
                                  max_input_length,
                                  stream_);
            cudaD2Hcpy(&compact_size, compact_size_, 1);
            use_shared_contexts = compact_size <= shared_contexts_ratio_ * batch_size;
            sync_check_cuda_error();
        }
        POP_RANGE;

        // NOTE: p/prompt-tuning process here (lookup prompt embedding tables by task name ids)
        // get p/prompt-tuning weight for each batch --> shape [batch, beam_width]
        // --> ptrs with shape [prompt_len, hidden_size]
        std::vector<const T*> p_prompt_tuning_batch_ptrs;
        std::vector<int>      p_prompt_tuning_lengths;
        PUSH_RANGE("prompt embedding lookup");
        if (use_loaded_p_prompt_embedding) {
            for (int bs_id = 0; bs_id < batch_size; ++bs_id) {
                int                      task_id              = prompt_learning_task_name_ids[bs_id];
                std::pair<const T*, int> p_prompt_tuning_pair = {};
                bool                     valid_task_name_id   = task_id < gpt_weights->prompt_learning_table.size();
                if (valid_task_name_id) {
                    p_prompt_tuning_pair = gpt_weights->prompt_learning_table.at(task_id);
                }
                else {
                    // don't throw oor in case of model server failing
                    FT_LOG_ERROR("p_prompt_tuning_weights not found for task id: " + std::to_string(task_id)
                                 + "\n return with invalid output tensors");
                    return;
                }
                if (input_lengths_h != nullptr) {
                    if (bs_id == 0) {
                        max_input_without_prompt_length = input_lengths_h[bs_id] - p_prompt_tuning_pair.second;
                    }
                    else {
                        max_input_without_prompt_length =
                            std::max(size_t(input_lengths_h[bs_id] - p_prompt_tuning_pair.second),
                                     max_input_without_prompt_length);
                    }
                }
                for (int bw_id = 0; bw_id < beam_width; ++bw_id) {
                    // only weight ptrs needed here
                    p_prompt_tuning_batch_ptrs.push_back(p_prompt_tuning_pair.first);
                    p_prompt_tuning_lengths.push_back(p_prompt_tuning_pair.second);
                }
            }

            cudaAutoCpy(
                prompt_learning_weight_batch_, p_prompt_tuning_batch_ptrs.data(), batch_size * beam_width, stream_);

            cudaAutoCpy(tiled_prompt_lengths_buf_, p_prompt_tuning_lengths.data(), batch_size * beam_width, stream_);

            sync_check_cuda_error();
        }
        POP_RANGE;

        // handle first step
        if (has_p_prompt_tuning_ || has_prefix_prompt_ || has_prefix_soft_prompt_ || max_input_length > 1) {
            PUSH_RANGE("input tiling and init");
            invokeTileGptPromptInputs(tiled_input_ids_buf_,
                                      tiled_input_lengths_buf_,
                                      use_request_p_prompt_embedding ? tiled_prompt_lengths_buf_ : nullptr,
                                      input_tensors->at("input_ids").getPtr<int>(),
                                      input_tensors->at("input_lengths").getPtr<const int>(),
                                      use_request_p_prompt_embedding ?
                                          input_tensors->at("request_prompt_lengths").getPtr<const int>() :
                                          nullptr,
                                      batch_size,
                                      beam_width,
                                      max_input_length,
                                      stream_);
            sync_check_cuda_error();
            POP_RANGE;

            if (has_prefix_soft_prompt_) {
                PUSH_RANGE("input id embedding lookup");
                inputIdsEmbeddingLookupPosEncodingSoftPromptParam<T> param;
                param.from_tensor                   = context_decoder_input_buf_;
                param.output_ids                    = output_ids_buf_;
                param.input_lengths                 = tiled_input_lengths_buf_;
                param.embedding_table               = gpt_weights->pre_decoder_embedding_table;
                param.pos_table                     = gpt_weights->position_encoding_table;
                param.prefix_soft_prompt_embedding  = input_tensors->at("request_prompt_embedding").getPtr<float>();
                param.prefix_soft_prompt_lengths    = input_tensors->at("request_prompt_lengths").getPtr<int>();
                param.input_ids                     = tiled_input_ids_buf_;
                param.start_step                    = 1;
                param.max_input_length              = max_input_length;
                param.max_prefix_soft_prompt_length = max_prefix_soft_prompt_length;
                param.batch_size                    = batch_size;
                param.beam_width                    = beam_width;
                param.hidden_units                  = hidden_units_;
                param.stream                        = stream_;

                invokeInputIdsEmbeddingLookupPosEncodingSoftPrompt(param);
                sync_check_cuda_error();
                POP_RANGE;

                max_input_length += max_prefix_soft_prompt_length;  // view soft_prompt as input
                max_context_len += max_prefix_soft_prompt_length;
            }
            else {
                // NOTE: add prompt embeddings here (for p/prompt tuning)
                PUSH_RANGE("input id embedding lookup");
                pPromptTuningParam<T> prompt_param{
                    use_loaded_p_prompt_embedding ? prompt_learning_weight_batch_ : (const T**)nullptr,
                    prompt_learning_start_id_,
                    max_request_p_prompt_length,
                    use_request_p_prompt_embedding,
                    use_request_p_prompt_embedding ? input_tensors->at("request_prompt_embedding").getPtr<T>() :
                                                     nullptr};
                invokeInputIdsEmbeddingLookupPosEncoding(context_decoder_input_buf_,
                                                         output_ids_buf_,
                                                         gpt_weights->pre_decoder_embedding_table,
                                                         gpt_weights->position_encoding_table,
                                                         prompt_param,
                                                         tiled_input_ids_buf_,
                                                         1,
                                                         max_input_length,
                                                         max_input_length,
                                                         batch_size * beam_width,
                                                         hidden_units_,
                                                         stream_);
                sync_check_cuda_error();
                POP_RANGE;
            }

            if (gpt_variant_params_.has_pre_decoder_layernorm) {
                PUSH_RANGE("pre-decoder layernorm");
                invokeGeneralLayerNorm(context_decoder_normed_input_buf_,
                                       context_decoder_input_buf_,
                                       gpt_weights->pre_decoder_layernorm.gamma,
                                       gpt_weights->pre_decoder_layernorm.beta,
                                       layernorm_eps_,
                                       batch_size * beam_width * max_input_length,
                                       hidden_units_,
                                       (float*)nullptr,
                                       0,
                                       stream_);
                POP_RANGE;
            }
            PUSH_RANGE("build decoder attention mask");
            invokeBuildDecoderAttentionMask(tiled_input_attention_mask_,
                                            tiled_input_lengths_buf_,
                                            nullptr,
                                            batch_size * beam_width,
                                            max_input_length,
                                            0,
                                            stream_);
            sync_check_cuda_error();
            POP_RANGE;

            TensorMap decoder_input_tensors(
                {{"decoder_input",
                  Tensor(MEMORY_GPU,
                         data_type,
                         {batch_size * beam_width, (size_t)max_input_length, hidden_units_},
                         gpt_variant_params_.has_pre_decoder_layernorm ? context_decoder_normed_input_buf_ :
                                                                         context_decoder_input_buf_)},
                 {"attention_mask",
                  Tensor(MEMORY_GPU,
                         data_type,
                         {batch_size * beam_width, 1, (size_t)max_input_length, (size_t)max_input_length},
                         tiled_input_attention_mask_)},
                 {"input_lengths",
                  Tensor(MEMORY_GPU, TYPE_INT32, {batch_size * beam_width}, tiled_input_lengths_buf_)}});

            if (use_shared_contexts) {
                decoder_input_tensors.insert("compact_idx",
                                             Tensor(MEMORY_GPU, TYPE_INT32, {(size_t)compact_size}, compact_idx_));
                decoder_input_tensors.insert(
                    "batch_to_compact_idx",
                    Tensor(MEMORY_GPU, TYPE_INT32, {batch_size * beam_width}, batch_to_compact_idx_));
            }
            if (gpt_variant_params_.use_attention_linear_bias) {
                decoder_input_tensors.insert("linear_bias_slopes",
                                             Tensor(MEMORY_GPU,
                                                    data_type,
                                                    {local_head_num_},
                                                    linear_bias_slopes_ + local_head_num_ * tensor_para_.rank_));
            }

            TensorMap decoder_output_tensors(
                {{"decoder_output",
                  Tensor(MEMORY_GPU,
                         data_type,
                         {batch_size * beam_width, (size_t)max_input_length, hidden_units_},
                         context_decoder_output_buf_)},
                 {"key_cache", Tensor(MEMORY_GPU, data_type, self_k_cache_shape, key_cache_)},
                 {"value_cache", Tensor(MEMORY_GPU, data_type, self_v_cache_shape, value_cache_)},
                 {"last_token_hidden_units",
                  Tensor(MEMORY_GPU, data_type, {batch_size * beam_width, hidden_units_}, decoder_output_buf_)}});

            gpt_context_decoder_->forward(
                &decoder_output_tensors, &decoder_input_tensors, &gpt_weights->decoder_layer_weights);

            if (is_return_context_embeddings) {
                PUSH_RANGE("context embedding sum length dim");
                invokeSumLengthDimension(output_tensors->at("context_embeddings").getPtr<float>(),
                                         context_decoder_output_buf_,
                                         batch_size * beam_width,
                                         max_input_length,
                                         hidden_units_,
                                         stream_);
                POP_RANGE;
            }

            PUSH_RANGE("decoding init");
            invokeDecodingInitialize(finished_buf_,
                                     sequence_lengths_,
                                     nullptr,
                                     cum_log_probs_,
                                     start_ids_buf_,
                                     batch_size,
                                     beam_width,
                                     max_input_length - 1,
                                     stream_);
            POP_RANGE;

            if (is_return_context_cum_log_probs) {
                PUSH_RANGE("compute context cumulative log probs");
                computeContextCumLogProbs(cum_log_probs_,
                                          context_decoder_output_buf_,
                                          tiled_input_ids_buf_,
                                          tiled_input_lengths_buf_,
                                          batch_size,
                                          beam_width,
                                          (size_t)max_input_length,
                                          gpt_weights);
                POP_RANGE;
            }
            sync_check_cuda_error();
        }
        else if (max_input_length == 0) {
            FT_CHECK(prompt_learning_type_ == PromptLearningType::no_prompt
                     && request_prompt_type == PromptLearningType::no_prompt);
            max_input_length++;
            PUSH_RANGE("decoding init");
            invokeDecodingInitialize(finished_buf_,
                                     sequence_lengths_,
                                     output_ids_buf_,
                                     cum_log_probs_,
                                     start_ids_buf_,
                                     batch_size,
                                     beam_width,
                                     max_input_length - 1,
                                     stream_);
            std::vector<int> h_input_lengths(batch_size * beam_width, 1);
            cudaAutoCpy(tiled_input_lengths_buf_, h_input_lengths.data(), batch_size * beam_width, stream_);
            sync_check_cuda_error();
            POP_RANGE;
        }
        else if (max_input_length == 1) {
            FT_CHECK(prompt_learning_type_ == PromptLearningType::no_prompt
                     && request_prompt_type == PromptLearningType::no_prompt);
            PUSH_RANGE("decoding init");
            invokeDecodingInitialize(finished_buf_,
                                     sequence_lengths_,
                                     nullptr,
                                     cum_log_probs_,
                                     start_ids_buf_,
                                     batch_size,
                                     beam_width,
                                     max_input_length - 1,
                                     stream_);
            sync_check_cuda_error();
            POP_RANGE;
            PUSH_RANGE("input tiling and init");
            invokeTileGptInputs(tiled_input_ids_buf_,
                                tiled_input_lengths_buf_,
                                input_tensors->at("input_ids").getPtr<int>(),
                                input_tensors->at("input_lengths").getPtr<int>(),
                                batch_size,
                                beam_width,
                                max_input_length,
                                stream_);
            sync_check_cuda_error();

            cudaAutoCpy(output_ids_buf_, tiled_input_ids_buf_, batch_size * beam_width, stream_);
            POP_RANGE;
        }
    }

    PUSH_RANGE("mask padding tokens");
    invokeMaskPaddingTokens(tiled_masked_tokens_,
                            input_tensors->at("input_lengths").getPtr<int>(),
                            memory_len,
                            max_input_length,
                            initial_step,
                            batch_size,
                            beam_width,
                            stream_);
    POP_RANGE;

    // If continue, we restart from initial_step because last token hasn't been processed in decoder
    const int step_start = continue_gen ? initial_step : max_input_length;

    const size_t local_batch_size = getLocalBatchSize(batch_size, 1, pipeline_para_.world_size_);
    FT_CHECK(batch_size % local_batch_size == 0);
    const size_t iteration_num = batch_size / local_batch_size;
    for (int microbatch = 0; microbatch < iteration_num; ++microbatch) {
        microbatch_should_stop_[microbatch] = false;
    }

    for (step_ = step_start; step_ < (int)gen_len; step_++) {
        // Loop body produces Nth token by embedding && encoding token (N-1)
        // if necessary.
        const bool fill_caches_only = continue_gen && (step_ < max_context_len);
        const int  src_indir_idx    = (step_ - step_start) % 2;
        const int  tgt_indir_idx    = 1 - src_indir_idx;

        bool generation_should_stop = !fill_caches_only;

        PUSH_RANGE(fmtstr("token_%d", step_ - step_start));
        for (uint ite = 0; ite < iteration_num; ++ite) {
            // skip the finished microbatch in previous steps
            if (microbatch_should_stop_[ite]) {
                continue;
            }
            const int id_offset               = ite * local_batch_size * beam_width;
            const int hidden_units_offset     = id_offset * hidden_units_;
            const int vocab_size_units_offset = id_offset * vocab_size_padded_;

            // Rank 0~N-1 needs to update the buffer by the results of last rank when the pipeline parallelism is
            // enabled (pipeline_para_.world_size_ > 1). And if step_ == step_start, then this is the first step and
            // these buffers are initialized by context directly.
            if (step_ != step_start && pipeline_para_.rank_ != pipeline_para_.world_size_ - 1
                && pipeline_para_.world_size_ > 1) {
                ftNcclGroupStart();
                // receive updated sequence_length_ from last rank
                ftNcclRecv(sequence_lengths_ + id_offset,
                           local_batch_size * beam_width,
                           pipeline_para_.world_size_ - 1,
                           pipeline_para_,
                           stream_);

                // receive updated microbatch_should_stop_ from last rank
                ftNcclRecv(microbatch_should_stop_ + ite, 1, pipeline_para_.world_size_ - 1, pipeline_para_, stream_);
                generation_should_stop &= microbatch_should_stop_[ite];

                // receive updated cache_indirections from last rank
                if (beam_width > 1) {
                    ftNcclRecv(cache_indirections_[tgt_indir_idx] + id_offset * memory_len,
                               local_batch_size * beam_width * memory_len,
                               pipeline_para_.world_size_ - 1,
                               pipeline_para_,
                               stream_);
                }

                // for ids of next step, only first rank needs to receive updated ids
                if (pipeline_para_.rank_ == 0) {
                    ftNcclRecv(output_ids_buf_ + (step_ - 1) * batch_size * beam_width + id_offset,
                               local_batch_size * beam_width,
                               pipeline_para_.world_size_ - 1,
                               pipeline_para_,
                               stream_);
                }

                ftNcclGroupEnd();
                // throw errors when detected
                ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream_);
                sync_check_cuda_error();
            }
            // skip the microbatch for last step, which is updated by last rank
            if (microbatch_should_stop_[ite]) {
                continue;
            }

            if ((max_input_length <= 1) || (step_ > step_start) || continue_gen) {
                if (pipeline_para_.rank_ == 0) {
                    invokeEmbeddingLookupPosEncodingPadCount(decoder_input_buf_ + hidden_units_offset,
                                                             gpt_weights->pre_decoder_embedding_table,
                                                             gpt_weights->position_encoding_table,
                                                             output_ids_buf_ + id_offset,
                                                             tiled_total_padding_count_ + id_offset,
                                                             local_batch_size * beam_width,
                                                             hidden_units_,
                                                             (T)(1.0f),
                                                             step_ - 1,
                                                             batch_size * beam_width,
                                                             0,
                                                             stream_);
                    sync_check_cuda_error();

                    if (gpt_variant_params_.has_pre_decoder_layernorm) {
                        invokeGeneralLayerNorm(decoder_normed_input_buf_ + hidden_units_offset,
                                               decoder_input_buf_ + hidden_units_offset,
                                               gpt_weights->pre_decoder_layernorm.gamma,
                                               gpt_weights->pre_decoder_layernorm.beta,
                                               layernorm_eps_,
                                               batch_size * beam_width,
                                               hidden_units_,
                                               (float*)nullptr,
                                               0,
                                               stream_);
                    }
                    sync_check_cuda_error();
                }

                std::unordered_map<std::string, Tensor> decoder_input_tensors(
                    {{"decoder_input",
                      Tensor(MEMORY_GPU,
                             data_type,
                             {local_batch_size * beam_width, hidden_units_},
                             gpt_variant_params_.has_pre_decoder_layernorm ?
                                 decoder_normed_input_buf_ + hidden_units_offset :
                                 decoder_input_buf_ + hidden_units_offset)},
                     {"finished",
                      Tensor(MEMORY_GPU, TYPE_BOOL, {local_batch_size * beam_width}, finished_buf_ + id_offset)},
                     {"input_lengths",
                      Tensor(MEMORY_GPU, TYPE_INT32, {local_batch_size * beam_width}, sequence_lengths_ + id_offset)},
                     {"total_padding_tokens",
                      Tensor(MEMORY_GPU,
                             TYPE_INT32,
                             {local_batch_size * beam_width},
                             tiled_total_padding_count_ + id_offset)},
                     {"max_input_length", Tensor(MEMORY_CPU, TYPE_INT32, {1}, &max_context_len)},
                     {"step", Tensor(MEMORY_CPU, TYPE_INT32, {1}, &step_)},
                     {"ite", Tensor(MEMORY_CPU, TYPE_INT32, {1}, &ite)},
                     {"masked_tokens",
                      Tensor(MEMORY_GPU,
                             TYPE_BOOL,
                             {local_batch_size * beam_width, memory_len},
                             tiled_masked_tokens_ + id_offset * memory_len)}});
                if (beam_width > 1) {
                    decoder_input_tensors.insert({"cache_indirection",
                                                  Tensor(MEMORY_GPU,
                                                         TYPE_INT32,
                                                         {local_batch_size, beam_width, memory_len},
                                                         cache_indirections_[src_indir_idx] + id_offset * memory_len)});
                }

                if (gpt_variant_params_.use_attention_linear_bias) {
                    decoder_input_tensors.insert({"linear_bias_slopes",
                                                  Tensor(MEMORY_GPU,
                                                         data_type,
                                                         {local_head_num_},
                                                         linear_bias_slopes_ + local_head_num_ * tensor_para_.rank_)});
                }

                std::unordered_map<std::string, Tensor> decoder_output_tensors(
                    {{"decoder_output",
                      Tensor(MEMORY_GPU,
                             data_type,
                             {local_batch_size * beam_width, hidden_units_},
                             decoder_output_buf_ + hidden_units_offset)},
                     {"key_cache", Tensor(MEMORY_GPU, data_type, self_k_cache_shape, key_cache_)},
                     {"value_cache", Tensor(MEMORY_GPU, data_type, self_v_cache_shape, value_cache_)}});

                gpt_decoder_->forward(
                    &decoder_output_tensors, &decoder_input_tensors, &gpt_weights->decoder_layer_weights);
            }

            if (!fill_caches_only && pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) {
                // OPT
                PUSH_RANGE("Token Final Layer Norm");
                T* decoder_output_final_buf =
                    gpt_variant_params_.has_post_decoder_layernorm ? normed_decoder_output_buf_ : decoder_output_buf_;
                if (gpt_variant_params_.has_post_decoder_layernorm) {
                    invokeGeneralLayerNorm(normed_decoder_output_buf_ + hidden_units_offset,
                                           decoder_output_buf_ + hidden_units_offset,
                                           gpt_weights->post_decoder_layernorm.gamma,
                                           gpt_weights->post_decoder_layernorm.beta,
                                           layernorm_eps_,
                                           local_batch_size * beam_width,
                                           hidden_units_,
                                           (float*)nullptr,
                                           0,
                                           stream_);
                }
                sync_check_cuda_error();
                POP_RANGE;

                if (tensor_para_.world_size_ == 1) {
                    float alpha = 1.0f;
                    float beta  = 0.0f;
                    PUSH_RANGE("logits gemm");
                    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                          CUBLAS_OP_N,
                                          vocab_size_padded_,  // n
                                          local_batch_size * beam_width,
                                          hidden_units_,  // k
                                          &alpha,
                                          padded_embedding_kernel_ptr_,
                                          gemm_data_type,
                                          hidden_units_,                                   // k
                                          decoder_output_final_buf + hidden_units_offset,  // OPT: no final layer norm
                                          gemm_data_type,
                                          hidden_units_,  // k
                                          &beta,
                                          logits_buf_ + vocab_size_units_offset,
                                          CUDA_R_32F,
                                          vocab_size_padded_, /* n */
                                          CUDA_R_32F,
                                          cublasGemmAlgo_t(-1));
                    POP_RANGE;
                }
                else {
                    FT_CHECK(vocab_size_padded_ % tensor_para_.world_size_ == 0);
                    const int local_vocab_size = vocab_size_padded_ / tensor_para_.world_size_;
                    float     alpha            = 1.0f;
                    float     beta             = 0.0f;
                    PUSH_RANGE("logits gemm");
                    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                          CUBLAS_OP_N,
                                          local_vocab_size,  // n
                                          local_batch_size * beam_width,
                                          hidden_units_,  // k
                                          &alpha,
                                          padded_embedding_kernel_ptr_
                                              + tensor_para_.rank_ * local_vocab_size * hidden_units_,
                                          gemm_data_type,
                                          hidden_units_,                                   // k
                                          decoder_output_final_buf + hidden_units_offset,  // OPT: no final layer norm
                                          gemm_data_type,
                                          hidden_units_,  // k
                                          &beta,
                                          nccl_logits_buf_ + vocab_size_units_offset
                                              + tensor_para_.rank_ * local_batch_size * beam_width * local_vocab_size,
                                          CUDA_R_32F,
                                          local_vocab_size, /* n */
                                          CUDA_R_32F,
                                          cublasGemmAlgo_t(-1));
                    POP_RANGE;
                    PUSH_RANGE("logits all gather");
                    ftNcclAllGather(nccl_logits_buf_ + vocab_size_units_offset,
                                    nccl_logits_buf_ + vocab_size_units_offset,
                                    local_batch_size * beam_width * local_vocab_size,
                                    tensor_para_.rank_,
                                    tensor_para_,
                                    stream_);
                    invokeTransposeAxis01(logits_buf_ + vocab_size_units_offset,
                                          nccl_logits_buf_ + vocab_size_units_offset,
                                          tensor_para_.world_size_,
                                          local_batch_size * beam_width,
                                          local_vocab_size,
                                          stream_);
                    POP_RANGE;
                }

                int  tmp_local_batch_size       = local_batch_size;
                bool is_initialize_random_table = step_ == max_context_len;

                std::unordered_map<std::string, Tensor> dynamic_decode_input_tensors{
                    {"logits",
                     Tensor{MEMORY_GPU, TYPE_FP32, {batch_size, beam_width, vocab_size_padded_}, logits_buf_}},
                    {"step", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step_}},
                    {"max_input_length", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_context_len}},
                    {"sequence_limit_length", Tensor{MEMORY_GPU, TYPE_UINT32, {batch_size}, seq_limit_len_}},
                    {"end_id", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size}, end_ids_buf_}},
                    {"input_lengths",
                     Tensor{MEMORY_GPU, TYPE_INT32, {batch_size, beam_width}, tiled_input_lengths_buf_}},
                    {"ite", Tensor{MEMORY_CPU, TYPE_UINT32, {1}, &ite}},
                    {"src_cache_indirection",
                     Tensor{MEMORY_GPU,
                            TYPE_INT32,
                            {local_batch_size, beam_width, memory_len},
                            cache_indirections_[src_indir_idx] + id_offset * memory_len}},
                    {"local_batch_size", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &tmp_local_batch_size}},
                    {"is_initialize_random_table", Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &is_initialize_random_table}}};

                for (auto t = input_tensors->begin(); t != input_tensors->end(); ++t) {
                    if (dynamic_decode_input_tensors.find(t->first) == dynamic_decode_input_tensors.end()) {
                        dynamic_decode_input_tensors.insert(*t);
                    }
                }

                // common outputs
                bool                                    subbatch_should_stop = false;
                std::unordered_map<std::string, Tensor> dynamic_decode_output_tensors{
                    {"output_ids", Tensor{MEMORY_GPU, TYPE_INT32, {gen_len, batch_size, beam_width}, output_ids_buf_}},
                    {"finished", Tensor{MEMORY_GPU, TYPE_BOOL, {batch_size * beam_width}, finished_buf_}},
                    // cum_log_probs is necessary for beam search, while it is optional for sampling.
                    {"cum_log_probs",
                     Tensor{MEMORY_GPU,
                            TYPE_FP32,
                            {batch_size * beam_width},
                            ((beam_width > 1) || (output_tensors->count("cum_log_probs") > 0)) ? cum_log_probs_ :
                                                                                                 nullptr}},
                    {"output_log_probs",
                     Tensor{MEMORY_GPU,
                            TYPE_FP32,
                            {gen_len, batch_size, beam_width},
                            output_tensors->count("output_log_probs") > 0
                                    && output_tensors->at("output_log_probs").data != nullptr ?
                                output_log_probs_buf_ :
                                nullptr}},
                    {"parent_ids", Tensor{MEMORY_GPU, TYPE_INT32, {gen_len, batch_size, beam_width}, parent_ids_buf_}},
                    {"sequence_length", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size * beam_width}, sequence_lengths_}},
                    {"tgt_cache_indirection",
                     Tensor{MEMORY_GPU,
                            TYPE_INT32,
                            {local_batch_size, beam_width, memory_len},
                            cache_indirections_[tgt_indir_idx] + id_offset * memory_len}},
                    {"should_stop", Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &subbatch_should_stop}}};
                for (auto t = output_tensors->begin(); t != output_tensors->end(); ++t) {
                    // Handle exceptions.
                    if (t->first == "cum_log_probs" || t->first == "output_log_probs") {
                        continue;
                    }
                    dynamic_decode_output_tensors.insert(*t);
                }

                PUSH_RANGE("result sampling and stop check");
                dynamic_decode_layer_->forward(&dynamic_decode_output_tensors, &dynamic_decode_input_tensors);
                generation_should_stop &= subbatch_should_stop;
                microbatch_should_stop_[ite] = subbatch_should_stop;
                POP_RANGE;
            }
            else {
                // for other ranks, they cannot update generation_should_stop by DynamicDecode, set to false directly;
                generation_should_stop &= microbatch_should_stop_[ite];
            }

            PUSH_RANGE("result communication");
            // send results to other rank
            if (fill_caches_only) {
                invokePlusScalar(sequence_lengths_, 1, batch_size * beam_width, stream_);
            }

            // When pipeline parallelism is enabled (pipeline_para_.world_size_ > 1), last rank needs to send updates
            // to other ranks.
            if (step_ < gen_len - 1 && pipeline_para_.world_size_ > 1
                && pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) {
                ftNcclGroupStart();
                for (int i = 0; i < pipeline_para_.world_size_ - 1; i++) {
                    // send updated sequence_length_ to other rank
                    ftNcclSend(
                        sequence_lengths_ + id_offset, local_batch_size * beam_width, i, pipeline_para_, stream_);

                    // send updated microbatch_should_stop_
                    ftNcclSend(microbatch_should_stop_ + ite, 1, i, pipeline_para_, stream_);

                    // send updated cache_indirections
                    if (beam_width > 1) {
                        ftNcclSend(cache_indirections_[tgt_indir_idx] + id_offset * memory_len,
                                   local_batch_size * beam_width * memory_len,
                                   i,
                                   pipeline_para_,
                                   stream_);
                    }
                }

                // for ids of next step, only need to send updated ids to first rank
                ftNcclSend(output_ids_buf_ + step_ * batch_size * beam_width + id_offset,
                           local_batch_size * beam_width,
                           0,
                           pipeline_para_,
                           stream_);

                ftNcclGroupEnd();
                // throw errors when detected
                ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream_);
                sync_check_cuda_error();
            }
            POP_RANGE;
        }

        if (token_generated_cb_ && step_ + 1 < (int)gen_len) {
            setOutputTensors(
                output_tensors, input_tensors, gen_len, session_len, max_context_len, max_input_without_prompt_length);
            sendTensorsToFirstPipelineNode(output_tensors, input_tensors);

            if (pipeline_para_.rank_ == 0 && tensor_para_.rank_ == 0) {
                token_generated_cb_(output_tensors, token_generated_ctx_);
            }
        }

        if (step_ == initial_step + max_input_length) {
            /* We have just finished processing input: update the padding count:
             * total_padding_count += (max_input_length - input_lengths) */
            PUSH_RANGE("Update padding count");
            invokeUpdatePaddingCount(tiled_total_padding_count_,
                                     input_tensors->at("input_lengths").getPtr<int>(),
                                     max_input_length,
                                     batch_size,
                                     beam_width,
                                     stream_);
            POP_RANGE;
        }

        if (generation_should_stop) {
            break;
        }

        POP_RANGE;
    }
    PUSH_RANGE("communicate tensors");
    setOutputTensors(
        output_tensors, input_tensors, gen_len, session_len, max_context_len, max_input_without_prompt_length);
    sendTensorsToFirstPipelineNode(output_tensors, input_tensors);
    POP_RANGE;
}

template<typename T>
void ParallelGpt<T>::sendTensorsToFirstPipelineNode(std::unordered_map<std::string, Tensor>*       output_tensors,
                                                    const std::unordered_map<std::string, Tensor>* input_tensors)
{
    if (pipeline_para_.world_size_ == 1) {
        // throw errors when detected
        ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream_);
        return;
    }

    const auto pp_rank = pipeline_para_.rank_;

    ftNcclGroupStart();
    for (auto const& it : *output_tensors) {
        if (it.second.data == nullptr) {
            continue;
        }

        if (pp_rank == pipeline_para_.world_size_ - 1) {
            ftNcclSend(it.second.getPtr<char>(), it.second.sizeBytes(), 0, pipeline_para_, stream_);
        }
        else if (pp_rank == 0) {
            ftNcclRecv(it.second.getPtr<char>(),
                       it.second.sizeBytes(),
                       pipeline_para_.world_size_ - 1,
                       pipeline_para_,
                       stream_);
        }
    }
    ftNcclGroupEnd();
    // throw errors when detected
    ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream_);
}

template<typename T>
void ParallelGpt<T>::setOutputTensors(std::unordered_map<std::string, Tensor>*       output_tensors,
                                      const std::unordered_map<std::string, Tensor>* input_tensors,
                                      const size_t                                   gen_len,
                                      const size_t                                   session_len,
                                      const size_t                                   max_context_len,
                                      const size_t                                   max_input_without_prompt_length)
{
    PUSH_RANGE("Resolve output tensors");
    if (pipeline_para_.rank_ != pipeline_para_.world_size_ - 1) {
        return;
    }

    const size_t batch_size       = output_tensors->at("output_ids").shape[0];
    const size_t beam_width       = output_tensors->at("output_ids").shape[1];
    int*         sequence_lengths = output_tensors->at("sequence_length").getPtr<int>();
    const size_t max_prefix_soft_prompt_length =
        has_prefix_soft_prompt_ ? input_tensors->at("request_prompt_embedding").shape[1] : 0;

    cudaAutoCpy(sequence_lengths, sequence_lengths_, output_tensors->at("sequence_length").size(), stream_);
    if (input_tensors->at("input_ids").shape[1] == 0) {
        // TODO: D2D sequence_lenghts
        if (beam_width > 1) {
            // For beam search, do gather_tree
            // take output_parent_ids as inter buffer
            invokeGatherTree(transposed_output_ids_buf_,
                             sequence_lengths_,
                             session_len,
                             batch_size,
                             beam_width,
                             output_ids_buf_ + batch_size * beam_width,
                             parent_ids_buf_ + batch_size * beam_width,
                             end_ids_buf_,
                             stream_);

            // transpose and take output_parent_ids as inter buffer
            invokeTransposeAxis01(output_tensors->at("output_ids").getPtr<int>(),
                                  transposed_output_ids_buf_,
                                  gen_len - 1,
                                  batch_size * beam_width,
                                  1,
                                  stream_);
        }
        else {
            // For sampling, only copy the results to output_tensor
            invokeTransposeAxis01(output_tensors->at("output_ids").getPtr<int>(),
                                  output_ids_buf_ + batch_size * beam_width,
                                  gen_len - 1,
                                  batch_size * beam_width,
                                  1,
                                  stream_);
        }
    }
    else {
        // For sampling, it is equivalent to all parent ids are 0.
        gatherTreeParam param;
        param.beams = transposed_output_ids_buf_;
        // Remove prompt length if possible
        param.max_sequence_lengths = sequence_lengths;
        // add sequence_length 1 here because the sequence_length of time step t is t - 1
        param.max_sequence_length_final_step = 1;
        // response input lengths (used to slice the ids during postprocessing)
        param.response_input_lengths = output_tensors->count("response_input_lengths") ?
                                           output_tensors->at("response_input_lengths").getPtr<int>() :
                                           nullptr;
        param.max_time               = gen_len;
        param.batch_size             = batch_size;
        param.beam_width             = beam_width;
        param.step_ids               = output_ids_buf_;
        param.parent_ids             = beam_width == 1 ? nullptr : parent_ids_buf_;
        param.end_tokens             = end_ids_buf_;
        param.max_input_length       = max_context_len;
        param.prefix_soft_prompt_lengths =
            has_prefix_soft_prompt_ ? input_tensors->at("request_prompt_lengths").getPtr<int>() : nullptr;
        param.input_lengths                   = tiled_input_lengths_buf_;
        param.p_prompt_tuning_prompt_lengths  = has_p_prompt_tuning_ ? tiled_prompt_lengths_buf_ : nullptr;
        param.max_input_without_prompt_length = max_input_without_prompt_length;
        param.max_prefix_soft_prompt_length   = max_prefix_soft_prompt_length;
        param.stream                          = stream_;
        param.output_ids                      = output_tensors->at("output_ids").getPtr<int>();
        // NOTE: need to remove all prompt virtual tokens
        invokeGatherTree(param);
        sync_check_cuda_error();
    }

    // remove p_prompt virtual tokens and update output tensors shape
    if (has_p_prompt_tuning_) {  // remove p_prompt virtual tokens and update output tensors shape
        output_tensors->at("output_ids").updateShape(2, gen_len - (max_context_len - max_input_without_prompt_length));
    }

    if ((output_tensors->count("output_log_probs") > 0 && output_tensors->at("output_log_probs").data != nullptr)) {
        invokeTransposeAxis01(output_tensors->at("output_log_probs").getPtr<float>(),
                              output_log_probs_buf_,
                              input_tensors->at("output_seq_len").max<uint32_t>() - max_context_len,
                              batch_size * beam_width,
                              1,
                              stream_);
    }
    // Return the cumulative log probability if requested.
    if (output_tensors->count("cum_log_probs") > 0) {
        Tensor cum_log_probs = output_tensors->at("cum_log_probs");
        FT_CHECK_WITH_INFO(cum_log_probs.size() == batch_size * beam_width,
                           "The shape of cum_log_probs does not match with batch_size x beam_width.");
        cudaAutoCpy(cum_log_probs.getPtr<float>(), cum_log_probs_, cum_log_probs.size(), stream_);
    }

    if (output_tensors->count("is_finished")) {
        cudaD2Dcpy(
            output_tensors->at("is_finished").getPtr<bool>(), finished_buf_, output_tensors->at("is_finished").size());
    }
    POP_RANGE;
}

template<typename T>
size_t ParallelGpt<T>::getPipelineParallelRank()
{
    return pipeline_para_.rank_;
}

template<typename T>
size_t ParallelGpt<T>::getPipelineParallelSize()
{
    return pipeline_para_.world_size_;
}

template<typename T>
size_t ParallelGpt<T>::getTensorParallelRank()
{
    return tensor_para_.rank_;
}

template<typename T>
size_t ParallelGpt<T>::getTensorParallelSize()
{
    return tensor_para_.world_size_;
}

template<typename T>
size_t ParallelGpt<T>::getHiddenUnits()
{
    return hidden_units_;
}

template<typename T>
bool* ParallelGpt<T>::getFinishBuffer()
{
    return finished_buf_;
}

template<typename T>
size_t ParallelGpt<T>::getStep()
{
    return step_;
}

template class ParallelGpt<float>;
template class ParallelGpt<half>;
#ifdef ENABLE_BF16
template class ParallelGpt<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
