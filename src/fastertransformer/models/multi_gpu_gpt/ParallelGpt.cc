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

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/decoding_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/kernels/logprob_kernels.h"
#include "src/fastertransformer/layers/beam_search_layers/BaseBeamSearchLayer.h"
#include "src/fastertransformer/utils/logger.h"

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
                                                            tensor_para_,
                                                            pipeline_para_,
                                                            stream_,
                                                            cublas_wrapper_,
                                                            allocator_,
                                                            is_free_buffer_after_forward_,
                                                            is_context_qk_buf_float_,
                                                            sparse_,
                                                            custom_all_reduce_comm_,
                                                            enable_custom_all_reduce_);

    gpt_decoder_ = new ParallelGptDecoder<T>(0,
                                             head_num_,
                                             size_per_head_,
                                             inter_size_,
                                             num_layer_,
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
                                    size_t max_seq_len,
                                    size_t max_input_len,
                                    bool is_return_context_cum_log_probs)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    const size_t batchxbeam = batch_size * beam_width;
    const size_t self_cache_size =
        (num_layer_ / pipeline_para_.world_size_) * batchxbeam * max_seq_len * hidden_units_ / tensor_para_.world_size_;

    if (vocab_size_ != vocab_size_padded_) {
        padded_embedding_kernel_ =
            (T*)(allocator_->reMalloc(padded_embedding_kernel_, sizeof(T) * hidden_units_ * vocab_size_padded_, true));
        padded_embedding_kernel_ptr_ = padded_embedding_kernel_;
    }

    input_attention_mask_ =
        (T*)(allocator_->reMalloc(input_attention_mask_, sizeof(T) * batchxbeam * max_seq_len * max_seq_len, false));
    decoder_input_buf_ = (T*)(allocator_->reMalloc(decoder_input_buf_, sizeof(T) * batchxbeam * hidden_units_, false));
    decoder_output_buf_ =
        (T*)(allocator_->reMalloc(decoder_output_buf_, sizeof(T) * batchxbeam * hidden_units_, false));
    normed_decoder_output_buf_ =
        (T*)(allocator_->reMalloc(normed_decoder_output_buf_, sizeof(T) * batchxbeam * hidden_units_, false));
    logits_buf_ = (float*)(allocator_->reMalloc(logits_buf_, sizeof(float) * batchxbeam * vocab_size_padded_, false));
    nccl_logits_buf_ =
        (float*)(allocator_->reMalloc(nccl_logits_buf_, sizeof(float) * batchxbeam * vocab_size_padded_, false));
    cum_log_probs_ = (float*)(allocator_->reMalloc(cum_log_probs_, sizeof(float) * batchxbeam, false));
    finished_buf_ = (bool*)(allocator_->reMalloc(finished_buf_, sizeof(bool) * batchxbeam, false));
    h_finished_buf_ = new bool[batchxbeam];

    key_cache_ = (T*)(allocator_->reMalloc(key_cache_, sizeof(T) * self_cache_size * 2, true));
    value_cache_ = key_cache_ + self_cache_size;
    if (beam_width > 1) {
        cache_indirections_[0] =
            (int*)(allocator_->reMalloc(cache_indirections_[0], sizeof(int) * batchxbeam * max_seq_len * 2, true));
        cache_indirections_[1] = cache_indirections_[0] + batchxbeam * max_seq_len;
    }

    tiled_input_ids_buf_ =
        (int*)(allocator_->reMalloc(tiled_input_ids_buf_, sizeof(int) * batchxbeam * max_input_len, true));
    tiled_input_lengths_buf_ = (int*)(allocator_->reMalloc(tiled_input_lengths_buf_, sizeof(int) * batchxbeam, true));

    start_ids_buf_ = (int*)(allocator_->reMalloc(start_ids_buf_, sizeof(int) * batch_size, false));
    end_ids_buf_ = (int*)(allocator_->reMalloc(end_ids_buf_, sizeof(int) * batch_size, false));

    transposed_output_ids_buf_ =
        (int*)(allocator_->reMalloc(transposed_output_ids_buf_, sizeof(int) * batchxbeam * max_seq_len, true));
    output_ids_buf_ = (int*)(allocator_->reMalloc(output_ids_buf_, sizeof(int) * batchxbeam * max_seq_len, true));
    parent_ids_buf_ = (int*)(allocator_->reMalloc(parent_ids_buf_, sizeof(int) * batchxbeam * max_seq_len, true));

    context_decoder_input_buf_ = (T*)(allocator_->reMalloc(
        context_decoder_input_buf_, sizeof(T) * batchxbeam * max_input_len * hidden_units_, false));
    context_decoder_output_buf_ = (T*)(allocator_->reMalloc(
        context_decoder_output_buf_, sizeof(T) * batchxbeam * max_input_len * hidden_units_, false));
    output_log_probs_buf_ =
        (float*)(allocator_->reMalloc(output_log_probs_buf_, sizeof(float) * batchxbeam * max_seq_len, false));

    if (is_return_context_cum_log_probs) {
        lp_normed_decoder_output_buf_ = (T*)allocator_->reMalloc(
            lp_normed_decoder_output_buf_, sizeof(T) * batchxbeam * max_input_len * hidden_units_);
        lp_logits_buf_ = (float*)allocator_->reMalloc(lp_logits_buf_,
                                                      sizeof(float) * batchxbeam * max_input_len * vocab_size_padded_);
        lp_nccl_logits_buf_ = (float*)allocator_->reMalloc(
            lp_nccl_logits_buf_, sizeof(float) * batchxbeam * max_input_len * vocab_size_padded_);
        lp_logprob_buf_ = (float*)allocator_->reMalloc(lp_logprob_buf_, sizeof(float) * batchxbeam * max_input_len);
    }

    is_allocate_buffer_ = true;
}

template<typename T>
void ParallelGpt<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        if (vocab_size_ != vocab_size_padded_) {
            padded_embedding_kernel_ptr_ = nullptr;
            allocator_->free(padded_embedding_kernel_);
        }

        allocator_->free(input_attention_mask_);
        allocator_->free(decoder_input_buf_);
        allocator_->free(decoder_output_buf_);
        allocator_->free(normed_decoder_output_buf_);
        allocator_->free(logits_buf_);
        allocator_->free(nccl_logits_buf_);
        allocator_->free(cum_log_probs_);
        allocator_->free(finished_buf_);
        delete[] h_finished_buf_;

        allocator_->free(key_cache_);
        if (cache_indirections_[0] != nullptr) {
            allocator_->free(cache_indirections_[0]);
        }

        allocator_->free(tiled_input_ids_buf_);
        allocator_->free(tiled_input_lengths_buf_);

        allocator_->free(transposed_output_ids_buf_);
        allocator_->free(output_ids_buf_);
        allocator_->free(parent_ids_buf_);

        allocator_->free(start_ids_buf_);
        allocator_->free(end_ids_buf_);

        allocator_->free(context_decoder_input_buf_);
        allocator_->free(context_decoder_output_buf_);
        allocator_->free(output_log_probs_buf_);

        allocator_->free(lp_normed_decoder_output_buf_);
        allocator_->free(lp_logits_buf_);
        allocator_->free(lp_nccl_logits_buf_);
        allocator_->free(lp_logprob_buf_);

        is_allocate_buffer_ = false;
    }
}

template<typename T>
ParallelGpt<T>::ParallelGpt(size_t max_batch_size,
                            size_t max_seq_len,
                            size_t max_input_len,
                            size_t beam_width,
                            size_t head_num,
                            size_t size_per_head,
                            size_t inter_size,
                            size_t num_layer,
                            size_t vocab_size,
                            int start_id,
                            int end_id,
                            float beam_search_diversity_rate,
                            size_t top_k,
                            float top_p,
                            unsigned long long random_seed,
                            float temperature,
                            float len_penalty,
                            float repetition_penalty,
                            NcclParam tensor_para,
                            NcclParam pipeline_para,
                            cudaStream_t stream,
                            cublasMMWrapper* cublas_wrapper,
                            IAllocator* allocator,
                            bool is_free_buffer_after_forward,
                            cudaDeviceProp* cuda_device_prop,
                            bool sparse,
                            int int8_mode,
                            std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                            int enable_custom_all_reduce):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop, sparse),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    vocab_size_(vocab_size),
    start_id_(start_id),
    end_id_(end_id),
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
    int8_mode_(int8_mode),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce)
{
    int local_vacab_size = ceil(vocab_size_ / 1.f / tensor_para_.world_size_);
    if (std::is_same<half, T>::value) {
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
    vocab_size_(gpt.vocab_size_),
    start_id_(gpt.start_id_),
    end_id_(gpt.end_id_),
    beam_search_diversity_rate_(gpt.beam_search_diversity_rate_),
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
    int8_mode_(gpt.int8_mode_),
    custom_all_reduce_comm_(gpt.custom_all_reduce_comm_),
    enable_custom_all_reduce_(gpt.enable_custom_all_reduce_)
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
void ParallelGpt<T>::computeContextCumLogProbs(float* cum_log_probs,
                                               const T* context_decoder_outputs,
                                               const int* input_ids,
                                               const int* input_lengths,
                                               const size_t batch_size,
                                               const size_t beam_width,
                                               const size_t max_input_length,
                                               const ParallelGptWeight<T>* gpt_weights)
{
    // Compute the log probabilties of prompt inputs.
    //
    // cum_log_probs [batch_size, beam_width]
    // context_decoder_outputs [batch_size * beam_width, max_input_length, hidden_units]
    // input_ids [batch_size * beam_width, max_input_length]; input ids.
    // input_lengths [batch_size, beam_width]; input lengths.
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    const size_t batchxbeam = batch_size * beam_width;
    const size_t n_hidden_states = batchxbeam * max_input_length;

    if (pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) {
        // normed decoder output [batch_size * beam_width, max_input_length, hidden_units_]
        invokeGeneralLayerNorm(lp_normed_decoder_output_buf_,
                               context_decoder_outputs,
                               gpt_weights->post_decoder_layernorm.gamma,
                               gpt_weights->post_decoder_layernorm.beta,
                               n_hidden_states,
                               hidden_units_,
                               stream_);
        sync_check_cuda_error();
        if (tensor_para_.world_size_ == 1) {
            float alpha = 1.0f;
            float beta = 0.0f;
            cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                  CUBLAS_OP_N,
                                  vocab_size_padded_,  // n
                                  n_hidden_states,
                                  hidden_units_,  // k
                                  &alpha,
                                  padded_embedding_kernel_ptr_,
                                  sizeof(T) == 2 ? CUDA_R_16F : CUDA_R_32F,
                                  hidden_units_,  // k
                                  lp_normed_decoder_output_buf_,
                                  sizeof(T) == 2 ? CUDA_R_16F : CUDA_R_32F,
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
            float alpha = 1.0f;
            float beta = 0.0f;
            cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                  CUBLAS_OP_N,
                                  local_vocab_size,  // n
                                  n_hidden_states,
                                  hidden_units_,  // k
                                  &alpha,
                                  padded_embedding_kernel_ptr_ + tensor_para_.rank_ * local_vocab_size * hidden_units_,
                                  sizeof(T) == 2 ? CUDA_R_16F : CUDA_R_32F,
                                  hidden_units_,  // k
                                  lp_normed_decoder_output_buf_,
                                  sizeof(T) == 2 ? CUDA_R_16F : CUDA_R_32F,
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
void ParallelGpt<T>::forward(std::vector<Tensor>* output_tensors,
                             const std::vector<Tensor>* input_tensors,
                             const ParallelGptWeight<T>* gpt_weights)
{
    // input_tensors:
    //      input_ids [batch_size, max_input_length]
    //      input_lengths [batch_size]
    //      max_output_seq_len [1] on cpu

    // output_tensors:
    //      output_ids [batch_size, beam, max_output_seq_len]
    //      parent_ids [max_output_seq_len, batch_size, beam]
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
    input_tensors_map.insert({"runtime_top_k", {MEMORY_CPU, TYPE_INT32, {1}, &top_k_}});
    input_tensors_map.insert({"runtime_top_p", {MEMORY_CPU, TYPE_FP32, {1}, &top_p_}});

    std::unordered_map<std::string, Tensor> output_tensors_map{{"output_ids", output_tensors->at(0)},
                                                               {"parent_ids", output_tensors->at(1)},
                                                               {"sequence_length", output_tensors->at(2)},
                                                               {"output_log_probs", output_tensors->at(3)}};
    if (output_tensors->size() > 3) {
        output_tensors_map.insert({"cum_log_probs", output_tensors->at(4)});
    }
    forward(&output_tensors_map, &input_tensors_map, gpt_weights);
}

template<typename T>
void ParallelGpt<T>::forward(std::unordered_map<std::string, Tensor>* output_tensors,
                             const std::unordered_map<std::string, Tensor>* input_tensors,
                             const ParallelGptWeight<T>* gpt_weights)
{
    // input_tensors:
    //      input_ids [batch_size, max_input_length]
    //      input_lengths [batch_size]
    //      max_output_seq_len [1] on cpu
    //      stop_words_list [batch_size, 2, stop_words_length], optional
    //      start_id [batch_size] on cpu, optional
    //      end_id [batch_size] on cpu, optional
    //      runtime_top_k [1] or [batch_size] on cpu, optional
    //      runtime_top_p [1] or [batch_size] on cpu, optional
    //      beam_search_diversity_rate [1] or [batch_size] on cpu, optional
    //      temperature [1] or [batch_size] on cpu, optional
    //      len_penalty [1] or [batch_size] on cpu, optional
    //      repetition_penalty [1] or [batch_size] on cpu, optional
    //      random_seed [1] or [batch_size] on cpu, optional
    //      prefix_soft_prompt_lengths [batch_size], optional
    //      prefix_soft_prompt_embedding [batch_size, max_prefix_soft_prompt_length, hidden_units], float, optional
    //      is_return_context_cum_log_probs [1] on cpu, bool, optional

    // output_tensors:
    //      output_ids [batch_size, beam_width, max_output_seq_len]
    //      parent_ids [max_output_seq_len, batch_size, beam_width]
    //      sequence_length [batch_size, beam_width]
    //      output_log_probs [batch_size, beam_width, request_output_seq_len], must be float*.
    //          optional. It leads to additional computing cost. If we don't need this result, don't put it.
    //      cum_log_probs [batch_size, beam_width], must be float*. optional.
    //          The cumulative log probability of generated sequences. It may lead to additional computing cost.

    // Step is from max_input_length ~ max_output_seq_len,
    // When step = k,  we put output ids and caches at step k, and the sequence_length would be k - 1 before
    // complete this step.
    // When there is no input_ids, put the start token at step 0 of output_ids_buf_. After forward, only copy
    // the step 1 ~ max_output_seq_len of output_ids_buf_ to output_tensors->at(0).data

    FT_CHECK_WITH_INFO(input_tensors->size() >= 3, "input_tensors->size() >= 3");
    FT_CHECK_WITH_INFO(output_tensors->size() >= 3, "output_tensors->size() >= 3");
    FT_CHECK(input_tensors->at("input_ids").shape.size() == 2);
    FT_CHECK(input_tensors->at("input_lengths").shape.size() == 1);
    FT_CHECK(input_tensors->at("max_output_seq_len").shape.size() == 1);
    FT_CHECK(output_tensors->at("output_ids").shape.size() == 3);
    FT_CHECK(output_tensors->at("parent_ids").shape.size() == 3);
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
    const size_t max_prefix_soft_prompt_length = input_tensors->count("prefix_soft_prompt_embedding") ?
                                                     input_tensors->at("prefix_soft_prompt_embedding").shape[1] :
                                                     0;
    const size_t max_output_seq_len = (*((int*)input_tensors->at("max_output_seq_len").data))
                                      + (max_input_length == 0 ? 1 : 0)  // additional 1 to put start token
                                      + max_prefix_soft_prompt_length;
    const size_t max_seq_len = max_output_seq_len;
    const bool is_return_context_cum_log_probs = input_tensors->count("is_return_context_cum_log_probs") > 0
                                                 && input_tensors->at("is_return_context_cum_log_probs").getVal<bool>();
    if (is_return_context_cum_log_probs) {
        FT_CHECK_WITH_INFO(output_tensors->count("cum_log_probs")
                               && output_tensors->at("cum_log_probs").data != nullptr,
                           "`cum_log_probs` must be provided in `output_tensors` in order to enable "
                           "the cumulative log probability computation of input contexts.");
    }

    allocateBuffer(batch_size,
                   beam_width,
                   max_seq_len,
                   max_input_length + max_prefix_soft_prompt_length,
                   is_return_context_cum_log_probs);
    sync_check_cuda_error();
    bool has_diff_runtime_args = hasDiffRuntimeArgs(input_tensors);

    int* sequence_lengths = (int*)(output_tensors->at("sequence_length").data);
    const DataType data_type = getTensorType<T>();
    const cudaDataType_t gemm_data_type = getCudaDataType<T>();

    const std::vector<size_t> self_k_cache_shape = {num_layer_ / pipeline_para_.world_size_,
                                                    batch_size * beam_width,
                                                    local_head_num_,
                                                    size_per_head_ / (16 / sizeof(T)),
                                                    max_output_seq_len,
                                                    16 / sizeof(T)};
    const std::vector<size_t> self_v_cache_shape = {num_layer_ / pipeline_para_.world_size_,
                                                    batch_size * beam_width,
                                                    local_head_num_,
                                                    max_output_seq_len,
                                                    size_per_head_};

    handleOptArg(input_tensors, "start_id", start_ids_buf_, start_id_, batch_size);
    handleOptArg(input_tensors, "end_id", end_ids_buf_, end_id_, batch_size);

    // TODO(bhsueh) Initilaize them in one kernel
    // initialize the output ids and parent ids
    cudaMemsetAsync(output_ids_buf_, 0, sizeof(int) * batch_size * beam_width * max_seq_len, stream_);
    cudaMemsetAsync(parent_ids_buf_, 0, sizeof(int) * batch_size * beam_width * max_seq_len, stream_);
    if (beam_width > 1) {
        cudaMemsetAsync(cache_indirections_[0], 0, 2 * sizeof(int) * batch_size * beam_width * max_seq_len, stream_);
    }
    sync_check_cuda_error();

    if (vocab_size_ == vocab_size_padded_) {
        padded_embedding_kernel_ptr_ = gpt_weights->post_decoder_embedding.kernel;
    }
    else {
        cudaMemcpyAsync(padded_embedding_kernel_,
                        gpt_weights->post_decoder_embedding.kernel,
                        sizeof(T) * vocab_size_ * hidden_units_,
                        cudaMemcpyDeviceToDevice,
                        stream_);
        sync_check_cuda_error();
    }

    // handle first step
    if (input_tensors->count("prefix_soft_prompt_embedding") || max_input_length >= 1) {
        invokeTileGptInputs(tiled_input_ids_buf_,
                            tiled_input_lengths_buf_,
                            (int*)input_tensors->at("input_ids").data,
                            (const int*)(input_tensors->at("input_lengths").data),
                            batch_size,
                            beam_width,
                            max_input_length,
                            stream_);
        sync_check_cuda_error();

        if (input_tensors->count("prefix_soft_prompt_embedding")) {
            inputIdsEmbeddingLookupPosEncodingSoftPromptParam<T> param;
            param.from_tensor = context_decoder_input_buf_;
            param.output_ids = output_ids_buf_;
            param.input_lengths = tiled_input_lengths_buf_;
            param.embedding_table = gpt_weights->pre_decoder_embedding_table;
            param.pos_table = gpt_weights->position_encoding_table;
            param.prefix_soft_prompt_embedding = input_tensors->at("prefix_soft_prompt_embedding").getPtr<float>();
            param.prefix_soft_prompt_lengths = input_tensors->at("prefix_soft_prompt_lengths").getPtr<int>();
            param.input_ids = tiled_input_ids_buf_;
            param.start_step = 1;
            param.max_input_length = max_input_length;
            param.max_prefix_soft_prompt_length = max_prefix_soft_prompt_length;
            param.batch_size = batch_size;
            param.beam_width = beam_width;
            param.hidden_units = hidden_units_;
            param.stream = stream_;

            invokeInputIdsEmbeddingLookupPosEncodingSoftPrompt(param);
            sync_check_cuda_error();
            max_input_length += max_prefix_soft_prompt_length;  // view soft_prompt as input
        }
        else {
            invokeInputIdsEmbeddingLookupPosEncoding(context_decoder_input_buf_,
                                                     output_ids_buf_,
                                                     gpt_weights->pre_decoder_embedding_table,
                                                     gpt_weights->position_encoding_table,
                                                     tiled_input_ids_buf_,
                                                     1,
                                                     max_input_length,
                                                     max_input_length,
                                                     batch_size * beam_width,
                                                     hidden_units_,
                                                     stream_);
            sync_check_cuda_error();
        }

        invokeBuildDecoderAttentionMask(
            input_attention_mask_, tiled_input_lengths_buf_, batch_size * beam_width, max_input_length, stream_);
        sync_check_cuda_error();

        std::vector<Tensor> decoder_input_tensors{
            Tensor{MEMORY_GPU,
                   data_type,
                   {batch_size * beam_width, (size_t)max_input_length, hidden_units_},
                   context_decoder_input_buf_},
            Tensor{MEMORY_GPU,
                   data_type,
                   {batch_size * beam_width, 1, (size_t)max_input_length, (size_t)max_input_length},
                   input_attention_mask_},
            Tensor{MEMORY_GPU, TYPE_INT32, {batch_size * beam_width}, tiled_input_lengths_buf_}};

        std::vector<Tensor> decoder_output_tensors{
            Tensor{MEMORY_GPU,
                   data_type,
                   {batch_size * beam_width, (size_t)max_input_length, hidden_units_},
                   context_decoder_output_buf_},
            Tensor{MEMORY_GPU, data_type, self_k_cache_shape, key_cache_},
            Tensor{MEMORY_GPU, data_type, self_v_cache_shape, value_cache_},
            Tensor{MEMORY_GPU, data_type, {batch_size * beam_width, hidden_units_}, decoder_output_buf_}};

        gpt_context_decoder_->forward(
            &decoder_output_tensors, &decoder_input_tensors, &gpt_weights->decoder_layer_weights);

        invokeDecodingInitialize(finished_buf_,
                                 sequence_lengths,
                                 nullptr,
                                 cum_log_probs_,
                                 start_ids_buf_,
                                 batch_size,
                                 beam_width,
                                 max_input_length - 1,
                                 stream_);

        if (is_return_context_cum_log_probs) {
            computeContextCumLogProbs(cum_log_probs_,
                                      context_decoder_output_buf_,
                                      tiled_input_ids_buf_,
                                      tiled_input_lengths_buf_,
                                      batch_size,
                                      beam_width,
                                      (size_t)max_input_length,
                                      gpt_weights);
        }
        sync_check_cuda_error();
    }
    else if (max_input_length == 0) {
        max_input_length++;
        invokeDecodingInitialize(finished_buf_,
                                 sequence_lengths,
                                 output_ids_buf_,
                                 cum_log_probs_,
                                 start_ids_buf_,
                                 batch_size,
                                 beam_width,
                                 max_input_length - 1,
                                 stream_);
        std::vector<int> h_input_lengths(batch_size * beam_width, 1);
        cudaMemcpyAsync(tiled_input_lengths_buf_,
                        h_input_lengths.data(),
                        sizeof(int) * batch_size * beam_width,
                        cudaMemcpyHostToDevice,
                        stream_);
        sync_check_cuda_error();
    }
    else if (max_input_length == 1) {
        invokeDecodingInitialize(finished_buf_,
                                 sequence_lengths,
                                 nullptr,
                                 cum_log_probs_,
                                 start_ids_buf_,
                                 batch_size,
                                 beam_width,
                                 max_input_length - 1,
                                 stream_);
        sync_check_cuda_error();
        invokeTileGptInputs(tiled_input_ids_buf_,
                            tiled_input_lengths_buf_,
                            (int*)input_tensors->at("input_ids").data,
                            (const int*)(input_tensors->at("input_lengths").data),
                            batch_size,
                            beam_width,
                            max_input_length,
                            stream_);
        sync_check_cuda_error();

        cudaMemcpyAsync(output_ids_buf_,
                        tiled_input_ids_buf_,
                        sizeof(int) * batch_size * beam_width,
                        cudaMemcpyDeviceToDevice,
                        stream_);
    }

    for (int step = max_input_length; step < (int)max_output_seq_len; step++) {
        const int src_indir_idx = (step - max_input_length) % 2;
        const int tgt_indir_idx = 1 - src_indir_idx;

        const size_t local_batch_size = getLocalBatchSize(batch_size, 1, pipeline_para_.world_size_);
        FT_CHECK(batch_size % local_batch_size == 0);
        const size_t iteration_num = batch_size / local_batch_size;

        for (uint ite = 0; ite < iteration_num; ++ite) {
            const int id_offset = ite * local_batch_size * beam_width;
            const int hidden_units_offset = id_offset * hidden_units_;
            const int vocab_size_units_offset = id_offset * vocab_size_padded_;

            if (!(max_input_length > 1 && step == max_input_length)) {
                if (pipeline_para_.rank_ == 0) {
                    invokeEmbeddingLookupPosEncoding(decoder_input_buf_ + hidden_units_offset,
                                                     gpt_weights->pre_decoder_embedding_table,
                                                     gpt_weights->position_encoding_table,
                                                     output_ids_buf_ + id_offset,
                                                     tiled_input_lengths_buf_ + id_offset,
                                                     local_batch_size * beam_width,
                                                     hidden_units_,
                                                     (T)(1.0f),
                                                     step - 1,
                                                     max_input_length,
                                                     batch_size * beam_width,
                                                     0,
                                                     stream_);
                    sync_check_cuda_error();
                }

                std::vector<Tensor> decoder_input_tensors{
                    Tensor{MEMORY_GPU,
                           data_type,
                           {local_batch_size * beam_width, hidden_units_},
                           decoder_input_buf_ + hidden_units_offset},
                    Tensor{MEMORY_GPU, TYPE_BOOL, {local_batch_size * beam_width}, finished_buf_ + id_offset},
                    Tensor{MEMORY_GPU, TYPE_INT32, {local_batch_size * beam_width}, sequence_lengths + id_offset},
                    Tensor{
                        MEMORY_GPU, TYPE_INT32, {local_batch_size * beam_width}, tiled_input_lengths_buf_ + id_offset},
                    Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_length},
                    Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step},
                    Tensor{MEMORY_CPU, TYPE_INT32, {1}, &ite},
                    Tensor{MEMORY_GPU,
                           TYPE_INT32,
                           {local_batch_size, beam_width, max_output_seq_len},
                           beam_width > 1 ? cache_indirections_[src_indir_idx] + id_offset * max_output_seq_len :
                                            nullptr}};

                std::vector<Tensor> decoder_output_tensors{
                    Tensor{MEMORY_GPU,
                           data_type,
                           {local_batch_size * beam_width, hidden_units_},
                           decoder_output_buf_ + hidden_units_offset},
                    Tensor{MEMORY_GPU, data_type, self_k_cache_shape, key_cache_},
                    Tensor{MEMORY_GPU, data_type, self_v_cache_shape, value_cache_}};
                gpt_decoder_->forward(
                    &decoder_output_tensors, &decoder_input_tensors, &gpt_weights->decoder_layer_weights);
            }

            if (pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) {
                invokeGeneralLayerNorm(normed_decoder_output_buf_ + hidden_units_offset,
                                       decoder_output_buf_ + hidden_units_offset,
                                       gpt_weights->post_decoder_layernorm.gamma,
                                       gpt_weights->post_decoder_layernorm.beta,
                                       local_batch_size * beam_width,
                                       hidden_units_,
                                       stream_);
                sync_check_cuda_error();

                if (tensor_para_.world_size_ == 1) {
                    float alpha = 1.0f;
                    float beta = 0.0f;
                    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                          CUBLAS_OP_N,
                                          vocab_size_padded_,  // n
                                          local_batch_size * beam_width,
                                          hidden_units_,  // k
                                          &alpha,
                                          padded_embedding_kernel_ptr_,
                                          gemm_data_type,
                                          hidden_units_,  // k
                                          normed_decoder_output_buf_ + hidden_units_offset,
                                          gemm_data_type,
                                          hidden_units_,  // k
                                          &beta,
                                          logits_buf_ + vocab_size_units_offset,
                                          CUDA_R_32F,
                                          vocab_size_padded_, /* n */
                                          CUDA_R_32F,
                                          cublasGemmAlgo_t(-1));
                }
                else {
                    FT_CHECK(vocab_size_padded_ % tensor_para_.world_size_ == 0);
                    const int local_vocab_size = vocab_size_padded_ / tensor_para_.world_size_;
                    float alpha = 1.0f;
                    float beta = 0.0f;
                    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                          CUBLAS_OP_N,
                                          local_vocab_size,  // n
                                          local_batch_size * beam_width,
                                          hidden_units_,  // k
                                          &alpha,
                                          padded_embedding_kernel_ptr_
                                              + tensor_para_.rank_ * local_vocab_size * hidden_units_,
                                          gemm_data_type,
                                          hidden_units_,  // k
                                          normed_decoder_output_buf_ + hidden_units_offset,
                                          gemm_data_type,
                                          hidden_units_,  // k
                                          &beta,
                                          nccl_logits_buf_ + vocab_size_units_offset
                                              + tensor_para_.rank_ * local_batch_size * beam_width * local_vocab_size,
                                          CUDA_R_32F,
                                          local_vocab_size, /* n */
                                          CUDA_R_32F,
                                          cublasGemmAlgo_t(-1));
                    ftNcclAllGather(nccl_logits_buf_ + vocab_size_units_offset,
                                    nccl_logits_buf_ + vocab_size_units_offset,
                                    local_batch_size * beam_width * local_vocab_size,
                                    tensor_para_.rank_,
                                    tensor_para_,
                                    stream_);
                    check_cuda_error(cudaStreamSynchronize(stream_));
                    invokeTransposeAxis01(logits_buf_ + vocab_size_units_offset,
                                          nccl_logits_buf_ + vocab_size_units_offset,
                                          tensor_para_.world_size_,
                                          local_batch_size * beam_width,
                                          local_vocab_size,
                                          stream_);
                }

                int tmp_local_batch_size = local_batch_size;
                bool is_initialize_random_table = step == max_input_length;
                std::unordered_map<std::string, Tensor> dynamic_decode_input_tensors{
                    {"logits",
                     Tensor{MEMORY_GPU, TYPE_FP32, {batch_size, beam_width, vocab_size_padded_}, logits_buf_}},
                    {"embedding_bias", Tensor{MEMORY_GPU, data_type, {vocab_size_padded_}, nullptr}},
                    {"step", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step}},
                    {"max_input_length", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_length}},
                    {"end_id", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size}, end_ids_buf_}},
                    {"input_lengths",
                     Tensor{MEMORY_GPU, TYPE_INT32, {batch_size, beam_width}, tiled_input_lengths_buf_}},
                    {"ite", Tensor{MEMORY_CPU, TYPE_UINT32, {1}, &ite}},
                    {"has_diff_runtime_args", Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &has_diff_runtime_args}},
                    {"src_key_cache", Tensor{MEMORY_GPU, data_type, self_k_cache_shape, key_cache_}},
                    {"src_value_cache", Tensor{MEMORY_GPU, data_type, self_v_cache_shape, value_cache_}},
                    {"src_cache_indirection",
                     Tensor{MEMORY_GPU,
                            TYPE_INT32,
                            {local_batch_size, beam_width, max_output_seq_len},
                            cache_indirections_[src_indir_idx] + id_offset * max_output_seq_len}},
                    {"local_batch_size", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &tmp_local_batch_size}},
                    {"is_initialize_random_table", Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &is_initialize_random_table}}};

                for (auto t = input_tensors->begin(); t != input_tensors->end(); ++t) {
                    if (dynamic_decode_input_tensors.find(t->first) == dynamic_decode_input_tensors.end()) {
                        dynamic_decode_input_tensors.insert(*t);
                    }
                }

                // common outputs
                std::unordered_map<std::string, Tensor> dynamic_decode_output_tensors{
                    {"output_ids",
                     Tensor{MEMORY_GPU, TYPE_INT32, {max_seq_len, batch_size, beam_width}, output_ids_buf_}},
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
                            {max_seq_len, batch_size, beam_width},
                            output_tensors->count("output_log_probs") > 0
                                    && output_tensors->at("output_log_probs").data != nullptr ?
                                output_log_probs_buf_ :
                                nullptr}},
                    {"parent_ids",
                     Tensor{MEMORY_GPU, TYPE_INT32, {max_seq_len, batch_size, beam_width}, parent_ids_buf_}},
                    {"sequence_length", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size * beam_width}, sequence_lengths}},
                    {"tgt_cache_indirection",
                     Tensor{MEMORY_GPU,
                            TYPE_INT32,
                            {local_batch_size, beam_width, max_output_seq_len},
                            cache_indirections_[tgt_indir_idx] + id_offset * max_output_seq_len}}};
                for (auto t = output_tensors->begin(); t != output_tensors->end(); ++t) {
                    // Handle exceptions.
                    if (t->first == "cum_log_probs" || t->first == "output_log_probs") {
                        continue;
                    }
                    dynamic_decode_output_tensors.insert(*t);
                }

                dynamic_decode_layer_->forward(&dynamic_decode_output_tensors, &dynamic_decode_input_tensors);
            }
        }

        if (pipeline_para_.world_size_ > 1) {
            NCCLCHECK(ncclGroupStart());
            ftNcclBroadCast(output_ids_buf_ + step * batch_size * beam_width,
                            batch_size * beam_width,
                            pipeline_para_.world_size_ - 1,
                            pipeline_para_,
                            stream_);

            ftNcclBroadCast(
                sequence_lengths, batch_size * beam_width, pipeline_para_.world_size_ - 1, pipeline_para_, stream_);

            ftNcclBroadCast(
                finished_buf_, batch_size * beam_width, pipeline_para_.world_size_ - 1, pipeline_para_, stream_);

            if (beam_width > 1) {

                ftNcclBroadCast(cache_indirections_[tgt_indir_idx],
                                batch_size * beam_width * max_output_seq_len,
                                pipeline_para_.world_size_ - 1,
                                pipeline_para_,
                                stream_);
            }
            NCCLCHECK(ncclGroupEnd());
            check_cuda_error(cudaStreamSynchronize(stream_));
            sync_check_cuda_error();
        }

        cudaD2Hcpy(h_finished_buf_, finished_buf_, batch_size * beam_width);
        uint sum = 0;
        for (uint i = 0; i < batch_size * beam_width; i++) {
            sum += (int)h_finished_buf_[i];
        }
        if (sum == batch_size * beam_width) {
            break;
        }
    }

    if (pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) {
        if (input_tensors->at("input_ids").shape[1] == 0) {
            if (beam_width > 1) {
                // For beam search, do gather_tree
                invokeGatherTree(transposed_output_ids_buf_,
                                 sequence_lengths,
                                 max_output_seq_len,
                                 batch_size,
                                 beam_width,
                                 output_ids_buf_ + batch_size * beam_width,
                                 parent_ids_buf_ + batch_size * beam_width,
                                 end_ids_buf_,
                                 stream_);

                // transpose and take output_parent_ids as inter buffer
                invokeTransposeAxis01((int*)output_tensors->at("output_ids").data,
                                      transposed_output_ids_buf_,
                                      max_output_seq_len - 1,
                                      batch_size * beam_width,
                                      1,
                                      stream_);

                cudaD2Dcpy((int*)output_tensors->at("parent_ids").data,
                           parent_ids_buf_ + batch_size * beam_width,
                           batch_size * beam_width * (max_output_seq_len - 1));
            }
            else {
                // For sampling, only transpose the results to output_tensor
                invokeTransposeAxis01((int*)output_tensors->at("output_ids").data,
                                      output_ids_buf_ + batch_size * beam_width,
                                      max_output_seq_len - 1,
                                      batch_size * beam_width,
                                      1,
                                      stream_);
            }
        }
        else {
            // add sequence_length 1 here because the sequence_length of time step t is t - 1
            invokePlusScalar(sequence_lengths, 1, batch_size * beam_width, stream_);

            // For sampling, it is equivalent to all parent ids are 0.
            gatherTreeParam param;
            param.beams = transposed_output_ids_buf_;
            param.max_sequence_lengths = sequence_lengths;
            param.max_time = max_output_seq_len;
            param.batch_size = batch_size;
            param.beam_width = beam_width;
            param.step_ids = output_ids_buf_;
            param.parent_ids = beam_width == 1 ? nullptr : parent_ids_buf_;
            param.end_tokens = end_ids_buf_;
            param.max_input_length = max_input_length;
            param.prefix_soft_prompt_lengths = input_tensors->count("prefix_soft_prompt_lengths") ?
                                                   input_tensors->at("prefix_soft_prompt_lengths").getPtr<int>() :
                                                   nullptr;
            param.input_lengths = tiled_input_lengths_buf_;
            param.max_prefix_soft_prompt_length = max_prefix_soft_prompt_length;
            param.stream = stream_;
            param.output_ids = (int*)output_tensors->at("output_ids").data;
            invokeGatherTree(param);
            sync_check_cuda_error();
        }

        if (output_tensors->count("output_log_probs")) {
            invokeTransposeAxis01(output_tensors->at("output_log_probs").getPtr<float>(),
                                  output_log_probs_buf_,
                                  input_tensors->at("max_output_seq_len").getVal<int>() - max_input_length,
                                  batch_size * beam_width,
                                  1,
                                  stream_);
        }
        // Return the cumulative log probability if requested.
        if (output_tensors->count("cum_log_probs") > 0 && output_tensors->at("cum_log_probs").data != nullptr) {
            Tensor cum_log_probs = output_tensors->at("cum_log_probs");
            FT_CHECK_WITH_INFO(cum_log_probs.size() == batch_size * beam_width,
                               "The shape of cum_log_probs does not match with batch_size x beam_width.");
            cudaD2Dcpy(cum_log_probs.getPtr<float>(), cum_log_probs_, batch_size * beam_width);
        }
    }

    if (pipeline_para_.world_size_ > 1) {
        NCCLCHECK(ncclGroupStart());
        if (pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) {
            ftNcclSend(output_tensors->at("output_ids").getPtr<int>(),
                       batch_size * beam_width * max_output_seq_len,
                       0,
                       pipeline_para_,
                       stream_);

            ftNcclSend(output_tensors->at("sequence_length").getPtr<int>(),
                       batch_size * beam_width,
                       0,
                       pipeline_para_,
                       stream_);

            if (output_tensors->count("cum_log_probs") > 0 && output_tensors->at("cum_log_probs").data != nullptr) {
                ftNcclSend(output_tensors->at("cum_log_probs").getPtr<float>(),
                           batch_size * beam_width,
                           0,
                           pipeline_para_,
                           stream_);
            }

            if (output_tensors->count("output_log_probs") > 0
                && output_tensors->at("output_log_probs").data != nullptr) {
                ftNcclSend(output_tensors->at("output_log_probs").getPtr<float>(),
                           output_tensors->at("output_log_probs").size(),
                           0,
                           pipeline_para_,
                           stream_);
            }
        }
        else if (pipeline_para_.rank_ == 0) {
            ftNcclRecv(output_tensors->at("output_ids").getPtr<int>(),
                       batch_size * beam_width * max_output_seq_len,
                       pipeline_para_.world_size_ - 1,
                       pipeline_para_,
                       stream_);

            ftNcclRecv(output_tensors->at("sequence_length").getPtr<int>(),
                       batch_size * beam_width,
                       pipeline_para_.world_size_ - 1,
                       pipeline_para_,
                       stream_);

            if (output_tensors->count("cum_log_probs") > 0 && output_tensors->at("cum_log_probs").data != nullptr) {
                ftNcclRecv(output_tensors->at("cum_log_probs").getPtr<float>(),
                           batch_size * beam_width,
                           pipeline_para_.world_size_ - 1,
                           pipeline_para_,
                           stream_);
            }

            if (output_tensors->count("output_log_probs") > 0
                && output_tensors->at("output_log_probs").data != nullptr) {
                ftNcclRecv(output_tensors->at("output_log_probs").getPtr<float>(),
                           output_tensors->at("output_log_probs").size(),
                           pipeline_para_.world_size_ - 1,
                           pipeline_para_,
                           stream_);
            }
        }
        NCCLCHECK(ncclGroupEnd());
        check_cuda_error(cudaStreamSynchronize(stream_));
    }
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
bool* ParallelGpt<T>::getFinishBuffer()
{
    return finished_buf_;
}

template class ParallelGpt<float>;
template class ParallelGpt<half>;
#ifdef ENABLE_BF16
template class ParallelGpt<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
