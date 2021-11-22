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

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/decoding_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/layers/beam_search_layers/BeamSearchLayer.h"
#include "src/fastertransformer/layers/beam_search_layers/OnlineBeamSearchLayer.h"
#include "src/fastertransformer/layers/sampling_layers/TopKSamplingLayer.h"
#include "src/fastertransformer/layers/sampling_layers/TopKTopPSamplingLayer.h"
#include "src/fastertransformer/layers/sampling_layers/TopPSamplingLayer.h"

namespace fastertransformer {

template<typename T>
void ParallelGpt<T>::initialize()
{
    gpt_context_decoder_ = new ParallelGptContextDecoder<T>(max_batch_size_ * beam_width_,
                                                            max_input_len_,
                                                            head_num_,
                                                            size_per_head_,
                                                            inter_size_,
                                                            num_layer_,
                                                            tensor_para_size_,
                                                            tensor_para_rank_,
                                                            tensor_para_comm_,
                                                            pipeline_para_size_,
                                                            pipeline_para_rank_,
                                                            pipeline_para_comm_,
                                                            stream_,
                                                            cublas_wrapper_,
                                                            allocator_,
                                                            is_free_buffer_after_forward_,
                                                            is_context_qk_buf_float_,
                                                            sparse_);

    gpt_decoder_ = new ParallelGptDecoder<T>(max_batch_size_ * beam_width_,
                                             head_num_,
                                             size_per_head_,
                                             inter_size_,
                                             num_layer_,
                                             tensor_para_size_,
                                             tensor_para_rank_,
                                             tensor_para_comm_,
                                             pipeline_para_size_,
                                             pipeline_para_rank_,
                                             pipeline_para_comm_,
                                             stream_,
                                             cublas_wrapper_,
                                             allocator_,
                                             is_free_buffer_after_forward_,
                                             sparse_,
                                             int8_mode_);

    if (beam_width_ > 1) {
        if (beam_width_ < 16) {
            dynamic_decode_ = new OnlineBeamSearchLayer<float>(max_batch_size_,
                                                               local_head_num_,
                                                               size_per_head_,
                                                               beam_width_,
                                                               vocab_size_,
                                                               vocab_size_padded_,
                                                               end_id_,
                                                               beam_search_diversity_rate_,
                                                               temperature_,
                                                               len_penalty_,
                                                               repetition_penalty_,
                                                               stream_,
                                                               cublas_wrapper_,
                                                               allocator_,
                                                               is_free_buffer_after_forward_);
        }
        else {
            dynamic_decode_ = new BeamSearchLayer<float>(max_batch_size_,
                                                         local_head_num_,
                                                         size_per_head_,
                                                         beam_width_,
                                                         vocab_size_,
                                                         vocab_size_padded_,
                                                         end_id_,
                                                         beam_search_diversity_rate_,
                                                         temperature_,
                                                         len_penalty_,
                                                         repetition_penalty_,
                                                         stream_,
                                                         cublas_wrapper_,
                                                         allocator_,
                                                         is_free_buffer_after_forward_);
        }
    }
    else if (top_p_ == 0 && top_k_ != 0) {
        // we sugguest set the is_free_buffer_after_forward_ of sampling to false
        // since we need to initialize some buffers if we allocate buffer
        // every time.
        dynamic_decode_ = new TopKSamplingLayer<float>(max_batch_size_,
                                                       vocab_size_,
                                                       vocab_size_padded_,
                                                       end_id_,
                                                       top_k_,
                                                       random_seed_,
                                                       temperature_,
                                                       len_penalty_,
                                                       repetition_penalty_,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       false);
    }
    else if (top_k_ == 0 && top_p_ != 0.0f) {
        // we sugguest set the is_free_buffer_after_forward_ of sampling to false
        // since we need to initialize some buffers if we allocate buffer
        // every time.
        dynamic_decode_ = new TopPSamplingLayer<float>(max_batch_size_,
                                                       vocab_size_,
                                                       vocab_size_padded_,
                                                       end_id_,
                                                       top_p_,
                                                       random_seed_,
                                                       temperature_,
                                                       len_penalty_,
                                                       repetition_penalty_,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       false,
                                                       cuda_device_prop_);
    }
    else {
        // we sugguest set the is_free_buffer_after_forward_ of sampling to false
        // since we need to initialize some buffers if we allocate buffer
        // every time.
        dynamic_decode_ = new TopKTopPSamplingLayer<float>(max_batch_size_,
                                                           vocab_size_,
                                                           vocab_size_padded_,
                                                           end_id_,
                                                           top_k_,
                                                           top_p_,
                                                           random_seed_,
                                                           temperature_,
                                                           len_penalty_,
                                                           repetition_penalty_,
                                                           stream_,
                                                           cublas_wrapper_,
                                                           allocator_,
                                                           false);
    }
}

template<typename T>
void ParallelGpt<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {

        const size_t batchxbeam = max_batch_size_ * beam_width_;
        const size_t self_cache_size =
            (num_layer_ / pipeline_para_size_) * batchxbeam * max_seq_len_ * hidden_units_ / tensor_para_size_;

        if (vocab_size_ != vocab_size_padded_) {
            padded_embedding_kernel_ = (T*)(allocator_->malloc(sizeof(T) * hidden_units_ * vocab_size_padded_, true));
            padded_embedding_kernel_ptr_ = padded_embedding_kernel_;
        }

        input_attention_mask_ = (T*)(allocator_->malloc(sizeof(T) * batchxbeam * max_seq_len_ * max_seq_len_, false));
        decoder_input_buf_ = (T*)(allocator_->malloc(sizeof(T) * batchxbeam * hidden_units_, false));
        decoder_output_buf_ = (T*)(allocator_->malloc(sizeof(T) * batchxbeam * hidden_units_, false));
        normed_decoder_output_buf_ = (T*)(allocator_->malloc(sizeof(T) * batchxbeam * hidden_units_, false));
        logits_buf_ = (float*)(allocator_->malloc(sizeof(float) * batchxbeam * vocab_size_padded_, false));
        nccl_logits_buf_ = (float*)(allocator_->malloc(sizeof(float) * batchxbeam * vocab_size_padded_, false));
        cum_log_probs_ = (float*)(allocator_->malloc(sizeof(float) * batchxbeam, false));
        finished_buf_ = (bool*)(allocator_->malloc(sizeof(bool) * batchxbeam, false));
        h_finished_buf_ = new bool[batchxbeam];

        key_caches_[0] = (T*)(allocator_->malloc(sizeof(T) * self_cache_size, true));
        value_caches_[0] = (T*)(allocator_->malloc(sizeof(T) * self_cache_size, true));
        if (beam_width_ > 1) {
            key_caches_[1] = (T*)(allocator_->malloc(sizeof(T) * self_cache_size, true));
            value_caches_[1] = (T*)(allocator_->malloc(sizeof(T) * self_cache_size, true));
        }

        padded_pos_embedding_bias_ = (T*)(allocator_->malloc(sizeof(T) * vocab_size_padded_, false));
        output_ids_buf_ = (int*)(allocator_->malloc(sizeof(int) * batchxbeam * max_seq_len_, true));
        parent_ids_buf_ = (int*)(allocator_->malloc(sizeof(int) * batchxbeam * max_seq_len_, true));
        input_length_buf_ = (int*)(allocator_->malloc(sizeof(int) * batchxbeam, false));

        context_decoder_input_buf_ =
            (T*)(allocator_->malloc(sizeof(T) * batchxbeam * max_input_len_ * hidden_units_, false));
        context_decoder_output_buf_ =
            (T*)(allocator_->malloc(sizeof(T) * batchxbeam * max_input_len_ * hidden_units_, false));
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void ParallelGpt<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
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

        allocator_->free(key_caches_[0]);
        allocator_->free(value_caches_[0]);
        if (beam_width_ > 1) {
            allocator_->free(key_caches_[1]);
            allocator_->free(value_caches_[1]);
        }

        allocator_->free(padded_pos_embedding_bias_);

        allocator_->free(output_ids_buf_);
        allocator_->free(parent_ids_buf_);

        allocator_->free(context_decoder_input_buf_);
        allocator_->free(context_decoder_output_buf_);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool ParallelGpt<T>::isValidBatchSize(size_t batch_size)
{
    if (batch_size <= max_batch_size_) {
        return true;
    }
    else {
        freeBuffer();
        max_batch_size_ = batch_size * 1.2;
        return true;
    }
}

template<typename T>
bool ParallelGpt<T>::isValidSeqLen(size_t seq_len)
{
    if (seq_len + 1 <= max_seq_len_) {
        return true;
    }
    else {
        // Since the max_seq_len_ is related to the position embedding table,
        // we cannot dynamic adjust it. 
        // return false;
        freeBuffer();
        max_seq_len_ = (seq_len + 1) * 1.2;
        return true;
    }
}

template<typename T>
bool ParallelGpt<T>::isValidInputSeqLen(size_t seq_len)
{
    if (seq_len <= max_input_len_) {
        return true;
    }
    else {
        freeBuffer();
        max_input_len_ = seq_len * 1.2;
        return true;
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
                            size_t tensor_para_size,
                            size_t tensor_para_rank,
                            ncclComm_t tensor_para_comm,
                            size_t pipeline_para_size,
                            size_t pipeline_para_rank,
                            ncclComm_t pipeline_para_comm,
                            cudaStream_t stream,
                            cublasMMWrapper* cublas_wrapper,
                            IAllocator* allocator,
                            bool is_free_buffer_after_forward,
                            cudaDeviceProp* cuda_device_prop,
                            bool sparse,
                            int int8_mode):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop, sparse),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len + 1),
    max_input_len_(max_input_len),
    beam_width_(beam_width),
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
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    tensor_para_comm_(tensor_para_comm),
    local_head_num_(head_num_ / tensor_para_size_),
    pipeline_para_size_(pipeline_para_size),
    pipeline_para_rank_(pipeline_para_rank),
    pipeline_para_comm_(pipeline_para_comm),
    int8_mode_(int8_mode)
{
    vocab_size_padded_ = ((size_t)ceil(vocab_size_ / 1. / tensor_para_size_) * tensor_para_size_);
    if (std::is_same<half, T>::value) {
        vocab_size_padded_ = ((size_t)ceil(vocab_size_padded_ / 8.) * 8);
    }

    initialize();
}

template<typename T>
ParallelGpt<T>::ParallelGpt(ParallelGpt<T> const& gpt):
    BaseLayer(gpt),
    max_batch_size_(gpt.max_batch_size_),
    max_seq_len_(gpt.max_seq_len_),
    max_input_len_(gpt.max_input_len_),
    beam_width_(gpt.beam_width_),
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
    tensor_para_size_(gpt.tensor_para_size_),
    tensor_para_rank_(gpt.tensor_para_rank_),
    tensor_para_comm_(gpt.tensor_para_comm_),
    local_head_num_(gpt.local_head_num_),
    pipeline_para_size_(gpt.pipeline_para_size_),
    pipeline_para_rank_(gpt.pipeline_para_rank_),
    pipeline_para_comm_(gpt.pipeline_para_comm_),
    vocab_size_padded_(gpt.vocab_size_padded_),
    int8_mode_(gpt.int8_mode_)
{
    initialize();
}

template<typename T>
ParallelGpt<T>::~ParallelGpt()
{
    delete gpt_decoder_;
    delete dynamic_decode_;
    freeBuffer();
}

template<typename T>
void ParallelGpt<T>::forward(std::vector<Tensor>* output_tensors,
                             const std::vector<Tensor>* input_tensors,
                             const ParallelGptWeight<T>* gpt_weights)
{
    // input_tensors:
    //      input_ids [batch_size * beam, max_input_length]
    //      input_lengths [batch_size * beam]
    //      max_output_seq_len [1] on cpu

    // output_tensors:
    //      output_ids [batch_size, beam, max_output_seq_len]
    //      parent_ids [max_output_seq_len, batch_size, beam]
    //      sequence_length [batch_size, beam]
    //      output_cum_log_probs [request_output_seq_len, batch_size, beam], must be float*.
    //          It leads to additional computing cost. If we don't need this result, please put nullptr

    // Step is from max_input_length ~ max_output_seq_len,
    // When step = k,  we put output ids and caches at step k, and the sequence_length would be k - 1 before
    // complete this step.
    // When there is no input_ids, put the start token at step 0 of output_ids_buf_. After forward, only copy
    // the step 1 ~ max_output_seq_len of output_ids_buf_ to output_tensors->at(0).data

    FT_CHECK(input_tensors->size() == 3);
    FT_CHECK(output_tensors->size() == 4);
    FT_CHECK(input_tensors->at(0).shape.size() == 2);
    FT_CHECK(input_tensors->at(1).shape.size() == 1);
    FT_CHECK(input_tensors->at(2).shape.size() == 1);
    FT_CHECK(output_tensors->at(0).shape.size() == 3);
    FT_CHECK(output_tensors->at(1).shape.size() == 3);
    FT_CHECK(output_tensors->at(2).shape.size() == 2);

    isValidSeqLen(output_tensors->at(0).shape[2]);
    isValidBatchSize(output_tensors->at(0).shape[0]);
    isValidInputSeqLen(input_tensors->at(0).shape[1]);
    allocateBuffer();
    sync_check_cuda_error();

    int max_input_length = input_tensors->at(0).shape[1];
    const int* input_length_ptr = (const int*)(input_tensors->at(1).data);
    const size_t max_output_seq_len =
        (*((int*)input_tensors->at(2).data)) + (max_input_length == 0 ? 1 : 0);  // additional 1 to put start token
    FT_CHECK(max_output_seq_len <= max_seq_len_);

    const size_t batch_size = output_tensors->at(0).shape[0];
    int* sequence_lengths = (int*)(output_tensors->at(2).data);
    const DataType data_type = getTensorType<T>();

    const std::vector<size_t> self_k_cache_size = {num_layer_ / pipeline_para_size_,
                                                   batch_size * beam_width_,
                                                   local_head_num_,
                                                   size_per_head_ / (16 / sizeof(T)),
                                                   max_output_seq_len,
                                                   16 / sizeof(T)};
    const std::vector<size_t> self_v_cache_size = {num_layer_ / pipeline_para_size_,
                                                   batch_size * beam_width_,
                                                   local_head_num_,
                                                   max_output_seq_len,
                                                   size_per_head_};

    float* output_cum_log_probs = (float*)(output_tensors->at(3).data);

    // initialize the output ids and parent ids
    cudaMemsetAsync(output_ids_buf_, 0, sizeof(int) * batch_size * beam_width_ * max_seq_len_, stream_);
    cudaMemsetAsync(parent_ids_buf_, 0, sizeof(int) * batch_size * beam_width_ * max_seq_len_, stream_);
    sync_check_cuda_error();

    // handle first step
    if (max_input_length == 0) {
        max_input_length++;
        invokeDecodingInitialize(finished_buf_,
                                 sequence_lengths,
                                 output_ids_buf_,
                                 cum_log_probs_,
                                 start_id_,
                                 batch_size,
                                 beam_width_,
                                 max_input_length - 1,
                                 stream_);
        std::vector<int> h_input_lengths(batch_size * beam_width_, 1);
        cudaMemcpyAsync(input_length_buf_,
                        h_input_lengths.data(),
                        sizeof(int) * batch_size * beam_width_,
                        cudaMemcpyHostToDevice,
                        stream_);
        input_length_ptr = input_length_buf_;
        sync_check_cuda_error();
    }
    else if (max_input_length == 1) {
        invokeDecodingInitialize(finished_buf_,
                                 sequence_lengths,
                                 nullptr,
                                 cum_log_probs_,
                                 start_id_,
                                 batch_size,
                                 beam_width_,
                                 max_input_length - 1,
                                 stream_);
        sync_check_cuda_error();

        cudaMemcpyAsync(output_ids_buf_,
                        (int*)input_tensors->at(0).data,
                        sizeof(int) * batch_size * beam_width_,
                        cudaMemcpyDeviceToDevice,
                        stream_);
    }
    else if (max_input_length > 1) {
        invokeBuildDecoderAttentionMask(
            input_attention_mask_, input_length_ptr, batch_size * beam_width_, max_input_length, stream_);
        sync_check_cuda_error();

        invokeInputIdsEmbeddingLookupPosEncoding(context_decoder_input_buf_,
                                                 output_ids_buf_,
                                                 gpt_weights->pre_decoder_embedding_table,
                                                 gpt_weights->position_encoding_table,
                                                 (int*)input_tensors->at(0).data,
                                                 1,
                                                 max_input_length,
                                                 max_input_length,
                                                 batch_size * beam_width_,
                                                 hidden_units_,
                                                 stream_);

        std::vector<Tensor> decoder_input_tensors{
            Tensor{MEMORY_GPU,
                   data_type,
                   {batch_size * beam_width_, (size_t)max_input_length, hidden_units_},
                   context_decoder_input_buf_},
            Tensor{MEMORY_GPU,
                   data_type,
                   {batch_size * beam_width_, 1, (size_t)max_input_length, (size_t)max_input_length},
                   input_attention_mask_},
            Tensor{MEMORY_GPU, TYPE_INT32, {batch_size * beam_width_}, input_length_ptr}};

        const int src_cache_id = beam_width_ > 1 ? (max_input_length - 1) & 0x1 : 0;
        std::vector<Tensor> decoder_output_tensors{
            Tensor{MEMORY_GPU,
                   data_type,
                   {batch_size * beam_width_, (size_t)max_input_length, hidden_units_},
                   context_decoder_output_buf_},
            Tensor{MEMORY_GPU, data_type, self_k_cache_size, key_caches_[src_cache_id]},
            Tensor{MEMORY_GPU, data_type, self_v_cache_size, value_caches_[src_cache_id]},
            Tensor{MEMORY_GPU, data_type, {batch_size * beam_width_, hidden_units_}, decoder_output_buf_}};

        gpt_context_decoder_->forward(
            &decoder_output_tensors, &decoder_input_tensors, &gpt_weights->decoder_layer_weights);

        invokeDecodingInitialize(finished_buf_,
                                 sequence_lengths,
                                 nullptr,
                                 cum_log_probs_,
                                 start_id_,
                                 batch_size,
                                 beam_width_,
                                 max_input_length - 1,
                                 stream_);
        sync_check_cuda_error();
    }

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

    for (int step = max_input_length; step < (int)max_output_seq_len; step++) {
        const int src_cache_id = beam_width_ > 1 ? (step - 1) & 0x1 : 0;
        const int tgt_cache_id = 1 - src_cache_id;

        const size_t local_batch_size = getLocalBatchSize(batch_size, 1, pipeline_para_size_);
        FT_CHECK(batch_size % local_batch_size == 0);
        const size_t iteration_num = batch_size / local_batch_size;

        for (uint ite = 0; ite < iteration_num; ++ite) {
            const int id_offset = ite * local_batch_size * beam_width_;
            const int hidden_units_offset = id_offset * hidden_units_;
            const int vocab_size_units_offset = id_offset * vocab_size_padded_;

            if (!(max_input_length > 1 && step == max_input_length)) {
                if (pipeline_para_rank_ == 0) {
                    invokeEmbeddingLookupPosEncoding(decoder_input_buf_ + hidden_units_offset,
                                                     gpt_weights->pre_decoder_embedding_table,
                                                     gpt_weights->position_encoding_table,
                                                     output_ids_buf_ + id_offset,
                                                     input_length_ptr + id_offset,
                                                     local_batch_size * beam_width_,
                                                     hidden_units_,
                                                     (T)(1.0f),
                                                     step - 1,
                                                     max_input_length,
                                                     batch_size * beam_width_,
                                                     0,
                                                     stream_);
                    sync_check_cuda_error();
                }

                std::vector<Tensor> decoder_input_tensors{
                    Tensor{MEMORY_GPU,
                           data_type,
                           {local_batch_size * beam_width_, hidden_units_},
                           decoder_input_buf_ + hidden_units_offset},
                    Tensor{MEMORY_GPU, TYPE_BOOL, {local_batch_size * beam_width_}, finished_buf_ + id_offset},
                    Tensor{MEMORY_GPU, TYPE_INT32, {local_batch_size * beam_width_}, sequence_lengths + id_offset},
                    Tensor{MEMORY_GPU, TYPE_INT32, {local_batch_size * beam_width_}, input_length_ptr + id_offset},
                    Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_length},
                    Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step},
                    Tensor{MEMORY_CPU, TYPE_INT32, {1}, &ite}};
                std::vector<Tensor> decoder_output_tensors{
                    Tensor{MEMORY_GPU,
                           data_type,
                           {local_batch_size * beam_width_, hidden_units_},
                           decoder_output_buf_ + hidden_units_offset},
                    Tensor{MEMORY_GPU, data_type, self_k_cache_size, key_caches_[src_cache_id]},
                    Tensor{MEMORY_GPU, data_type, self_v_cache_size, value_caches_[src_cache_id]}};
                gpt_decoder_->forward(
                    &decoder_output_tensors, &decoder_input_tensors, &gpt_weights->decoder_layer_weights);
            }

            if (pipeline_para_rank_ == pipeline_para_size_ - 1) {
                invokeGeneralLayerNorm(normed_decoder_output_buf_ + hidden_units_offset,
                                       decoder_output_buf_ + hidden_units_offset,
                                       gpt_weights->post_decoder_layernorm.gamma,
                                       gpt_weights->post_decoder_layernorm.beta,
                                       local_batch_size * beam_width_,
                                       hidden_units_,
                                       stream_);
                sync_check_cuda_error();

                if (tensor_para_size_ == 1) {
                    float alpha = 1.0f;
                    float beta = 0.0f;
                    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                          CUBLAS_OP_N,
                                          vocab_size_padded_,  // n
                                          local_batch_size * beam_width_,
                                          hidden_units_,  // k
                                          &alpha,
                                          padded_embedding_kernel_ptr_,
                                          sizeof(T) == 2 ? CUDA_R_16F : CUDA_R_32F,
                                          hidden_units_,  // k
                                          normed_decoder_output_buf_ + hidden_units_offset,
                                          sizeof(T) == 2 ? CUDA_R_16F : CUDA_R_32F,
                                          hidden_units_,  // k
                                          &beta,
                                          logits_buf_ + vocab_size_units_offset,
                                          CUDA_R_32F,
                                          vocab_size_padded_, /* n */
                                          CUDA_R_32F,
                                          cublasGemmAlgo_t(-1));
                }
                else {
                    FT_CHECK(vocab_size_padded_ % tensor_para_size_ == 0);
                    const int local_vocab_size = vocab_size_padded_ / tensor_para_size_;
                    float alpha = 1.0f;
                    float beta = 0.0f;
                    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                          CUBLAS_OP_N,
                                          local_vocab_size,  // n
                                          local_batch_size * beam_width_,
                                          hidden_units_,  // k
                                          &alpha,
                                          padded_embedding_kernel_ptr_
                                              + tensor_para_rank_ * local_vocab_size * hidden_units_,
                                          sizeof(T) == 2 ? CUDA_R_16F : CUDA_R_32F,
                                          hidden_units_,  // k
                                          normed_decoder_output_buf_ + hidden_units_offset,
                                          sizeof(T) == 2 ? CUDA_R_16F : CUDA_R_32F,
                                          hidden_units_,  // k
                                          &beta,
                                          nccl_logits_buf_ + vocab_size_units_offset
                                              + tensor_para_rank_ * local_batch_size * beam_width_ * local_vocab_size,
                                          CUDA_R_32F,
                                          local_vocab_size, /* n */
                                          CUDA_R_32F,
                                          cublasGemmAlgo_t(-1));
                    ftNcclAllGather(nccl_logits_buf_ + vocab_size_units_offset,
                                    nccl_logits_buf_ + vocab_size_units_offset,
                                    local_batch_size * beam_width_ * local_vocab_size,
                                    tensor_para_rank_,
                                    tensor_para_comm_,
                                    stream_);
                    check_cuda_error(cudaStreamSynchronize(stream_));
                    invokeTransposeAxis01(logits_buf_ + vocab_size_units_offset,
                                          nccl_logits_buf_ + vocab_size_units_offset,
                                          tensor_para_size_,
                                          local_batch_size * beam_width_,
                                          local_vocab_size,
                                          stream_);
                }

                std::vector<Tensor>* dynamic_decode_input_tensors;
                std::vector<Tensor>* dynamic_decode_output_tensors;
                if (beam_width_ > 1) {
                    dynamic_decode_input_tensors = new std::vector<Tensor>{
                        Tensor{MEMORY_GPU,
                               TYPE_FP32,
                               {local_batch_size, beam_width_, vocab_size_padded_},
                               logits_buf_ + vocab_size_units_offset},
                        Tensor{MEMORY_GPU, data_type, {vocab_size_padded_}, nullptr},
                        Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step},
                        Tensor{MEMORY_GPU, data_type, self_k_cache_size, key_caches_[src_cache_id]},
                        Tensor{MEMORY_GPU, data_type, self_v_cache_size, value_caches_[src_cache_id]},
                        Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_length},
                        Tensor{MEMORY_GPU, TYPE_INT32, {local_batch_size, beam_width_}, input_length_ptr + id_offset},
                        Tensor{MEMORY_CPU, TYPE_INT32, {1}, &ite}};

                    dynamic_decode_output_tensors = new std::vector<Tensor>{
                        Tensor{MEMORY_GPU, TYPE_INT32, {max_output_seq_len, batch_size, beam_width_}, output_ids_buf_},
                        Tensor{MEMORY_GPU, TYPE_BOOL, {local_batch_size * beam_width_}, finished_buf_ + id_offset},
                        Tensor{MEMORY_GPU, TYPE_FP32, {local_batch_size * beam_width_}, cum_log_probs_ + id_offset},
                        Tensor{MEMORY_GPU, TYPE_INT32, {max_output_seq_len, batch_size, beam_width_}, parent_ids_buf_},
                        Tensor{MEMORY_GPU, TYPE_INT32, {local_batch_size * beam_width_}, sequence_lengths + id_offset},
                        Tensor{MEMORY_GPU, data_type, self_k_cache_size, key_caches_[tgt_cache_id]},
                        Tensor{MEMORY_GPU, data_type, self_v_cache_size, value_caches_[tgt_cache_id]}};
                }
                else {
                    dynamic_decode_input_tensors = new std::vector<Tensor>{
                        Tensor{MEMORY_GPU,
                               TYPE_FP32,
                               {local_batch_size, vocab_size_padded_},
                               logits_buf_ + vocab_size_units_offset},
                        Tensor{MEMORY_GPU, data_type, {vocab_size_padded_}, nullptr},
                        Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step},
                        Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_length},
                        Tensor{MEMORY_GPU, TYPE_INT32, {local_batch_size}, input_length_ptr + id_offset},
                        Tensor{MEMORY_GPU, TYPE_INT32, {1}, &ite}};

                    dynamic_decode_output_tensors = new std::vector<Tensor>{
                        Tensor{MEMORY_GPU,
                               TYPE_INT32,
                               {max_output_seq_len, batch_size},
                               output_ids_buf_},  // note that we need to put whole output_ids here
                        Tensor{MEMORY_GPU, TYPE_BOOL, {local_batch_size}, finished_buf_ + id_offset},
                        Tensor{MEMORY_GPU, TYPE_INT32, {local_batch_size}, sequence_lengths + id_offset},
                        Tensor{MEMORY_GPU,
                               TYPE_FP32,
                               {local_batch_size},
                               output_cum_log_probs == nullptr ?
                                   nullptr :
                                   output_cum_log_probs + (step - max_input_length) * batch_size * beam_width_
                                       + id_offset}};
                }
                dynamic_decode_->forward(dynamic_decode_output_tensors, dynamic_decode_input_tensors);

                delete dynamic_decode_input_tensors;
                delete dynamic_decode_output_tensors;
            }
        }

        if (pipeline_para_size_ > 1) {
            NCCLCHECK(ncclGroupStart());
            ftNcclBroadCast(output_ids_buf_ + step * batch_size * beam_width_,
                            batch_size * beam_width_,
                            pipeline_para_size_ - 1,
                            pipeline_para_comm_,
                            stream_);

            ftNcclBroadCast(
                sequence_lengths, batch_size * beam_width_, pipeline_para_size_ - 1, pipeline_para_comm_, stream_);

            ftNcclBroadCast(
                finished_buf_, batch_size * beam_width_, pipeline_para_size_ - 1, pipeline_para_comm_, stream_);

            if (beam_width_ > 1) {
                ftNcclBroadCast(parent_ids_buf_ + step * batch_size * beam_width_,
                                batch_size * beam_width_,
                                pipeline_para_size_ - 1,
                                pipeline_para_comm_,
                                stream_);
            }
            NCCLCHECK(ncclGroupEnd());
            check_cuda_error(cudaStreamSynchronize(stream_));
            sync_check_cuda_error();
        }

        if (pipeline_para_rank_ != pipeline_para_size_ - 1 && beam_width_ > 1) {
            // update kv cache of other pipeline_para_rank since they don't
            // enter the beam search
            update_KV_cache_kernelLauncher(key_caches_[tgt_cache_id],
                                           value_caches_[tgt_cache_id],
                                           key_caches_[src_cache_id],
                                           value_caches_[src_cache_id],
                                           parent_ids_buf_ + step * batch_size * beam_width_,
                                           finished_buf_,
                                           batch_size,
                                           batch_size,
                                           beam_width_,
                                           local_head_num_,
                                           size_per_head_,
                                           step,
                                           max_output_seq_len,
                                           batch_size * beam_width_ * local_head_num_ * size_per_head_
                                               * max_output_seq_len,
                                           num_layer_ / pipeline_para_size_,
                                           0,
                                           stream_);
            sync_check_cuda_error();
        }

        cudaD2Hcpy(h_finished_buf_, finished_buf_, batch_size * beam_width_);
        uint sum = 0;
        for (uint i = 0; i < batch_size * beam_width_; i++) {
            sum += (int)h_finished_buf_[i];
        }
        if (sum == batch_size * beam_width_) {
            break;
        }
    }

    if (input_tensors->at(0).shape[1] == 0) {
        if (beam_width_ > 1) {
            // For beam search, do gather_tree
            invokeGatherTree((int*)output_tensors->at(1).data,
                             (int*)output_tensors->at(2).data,
                             max_output_seq_len - 1,
                             batch_size,
                             beam_width_,
                             output_ids_buf_ + batch_size * beam_width_,
                             parent_ids_buf_ + batch_size * beam_width_,
                             end_id_,
                             stream_);

            // transpose and take output_parent_ids as inter buffer
            invokeTransposeAxis01((int*)output_tensors->at(0).data,
                                  (int*)output_tensors->at(1).data,
                                  max_output_seq_len - 1,
                                  batch_size * beam_width_,
                                  1,
                                  stream_);

            cudaD2Dcpy((int*)output_tensors->at(1).data,
                       parent_ids_buf_ + batch_size * beam_width_,
                       batch_size * beam_width_ * (max_output_seq_len - 1));
        }
        else {
            // For sampling, only transpose the results to output_tensor
            invokeTransposeAxis01((int*)output_tensors->at(0).data,
                                  output_ids_buf_ + batch_size * beam_width_,
                                  max_output_seq_len - 1,
                                  batch_size * beam_width_,
                                  1,
                                  stream_);
        }
    }
    else {
        if (beam_width_ > 1) {
            // For beam search, do gather_tree
            invokeGatherTree((int*)output_tensors->at(1).data,
                             (int*)output_tensors->at(2).data,
                             max_output_seq_len,
                             batch_size,
                             beam_width_,
                             output_ids_buf_,
                             parent_ids_buf_,
                             -1,
                             stream_);

            // transpose and take output_parent_ids as inter buffer
            invokeTransposeAxis01((int*)output_tensors->at(0).data,
                                  (int*)output_tensors->at(1).data,
                                  max_output_seq_len,
                                  batch_size * beam_width_,
                                  1,
                                  stream_);

            cudaD2Dcpy(
                (int*)output_tensors->at(1).data, parent_ids_buf_, batch_size * beam_width_ * max_output_seq_len);
        }
        else {
            // For sampling, only transpose the results to output_tensor
            invokeTransposeAxis01((int*)output_tensors->at(0).data,
                                  output_ids_buf_,
                                  max_output_seq_len,
                                  batch_size * beam_width_,
                                  1,
                                  stream_);
        }
    }
}

template<typename T>
size_t ParallelGpt<T>::getPipelineParallelRank()
{
    return pipeline_para_rank_;
}

template<typename T>
size_t ParallelGpt<T>::getPipelineParallelSize()
{
    return pipeline_para_size_;
}

template<typename T>
size_t ParallelGpt<T>::getTensorParallelRank()
{
    return tensor_para_rank_;
}

template<typename T>
size_t ParallelGpt<T>::getTensorParallelSize()
{
    return tensor_para_size_;
}

template<typename T>
bool* ParallelGpt<T>::getFinishBuffer()
{
    return finished_buf_;
}

template class ParallelGpt<float>;
template class ParallelGpt<half>;

}  // namespace fastertransformer
