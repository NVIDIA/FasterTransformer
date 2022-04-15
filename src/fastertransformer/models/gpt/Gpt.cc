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
void Gpt<T>::initialize()
{
    gpt_context_decoder_ = new GptContextDecoder<T>(max_batch_size_ * beam_width_,
                                                    max_input_len_,
                                                    head_num_,
                                                    size_per_head_,
                                                    inter_size_,
                                                    num_layer_,
                                                    stream_,
                                                    cublas_wrapper_,
                                                    allocator_,
                                                    is_free_buffer_after_forward_,
                                                    is_context_qk_buf_float_,
                                                    sparse_);

    gpt_decoder_ = new GptDecoder<T>(max_batch_size_ * beam_width_,
                                     head_num_,
                                     size_per_head_,
                                     inter_size_,
                                     num_layer_,
                                     stream_,
                                     cublas_wrapper_,
                                     allocator_,
                                     is_free_buffer_after_forward_,
                                     sparse_);

    if (beam_width_ > 1) {
        if (beam_width_ < 16) {
            dynamic_decode_ = new OnlineBeamSearchLayer<T>(max_batch_size_,
                                                           head_num_,
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
            dynamic_decode_ = new BeamSearchLayer<T>(max_batch_size_,
                                                     head_num_,
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
        dynamic_decode_ = new TopKSamplingLayer<T>(max_batch_size_,
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
        dynamic_decode_ = new TopPSamplingLayer<T>(max_batch_size_,
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
        dynamic_decode_ = new TopKTopPSamplingLayer<T>(max_batch_size_,
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
void Gpt<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {

        const size_t batchxbeam = max_batch_size_ * beam_width_;
        const size_t self_cache_size = num_layer_ * batchxbeam * max_seq_len_ * hidden_units_;

        if (vocab_size_ != vocab_size_padded_) {
            padded_embedding_kernel_ = (T*)(allocator_->malloc(sizeof(T) * hidden_units_ * vocab_size_padded_, true));
            padded_embedding_kernel_ptr_ = padded_embedding_kernel_;
        }

        input_attention_mask_ = (T*)(allocator_->malloc(sizeof(T) * batchxbeam * max_seq_len_ * max_seq_len_, false));
        decoder_input_buf_ = (T*)(allocator_->malloc(sizeof(T) * batchxbeam * hidden_units_, false));
        decoder_output_buf_ = (T*)(allocator_->malloc(sizeof(T) * batchxbeam * hidden_units_, false));
        normed_decoder_output_buf_ = (T*)(allocator_->malloc(sizeof(T) * batchxbeam * hidden_units_, false));
        logits_buf_ = (T*)(allocator_->malloc(sizeof(T) * batchxbeam * vocab_size_padded_, false));
        cum_log_probs_ = (float*)(allocator_->malloc(sizeof(float) * batchxbeam, false));
        finished_buf_ = (bool*)(allocator_->malloc(sizeof(bool) * batchxbeam, false));
        h_finished_buf_ = new bool[batchxbeam];

        key_caches_[0] = (T*)(allocator_->malloc(sizeof(T) * self_cache_size, false));
        value_caches_[0] = (T*)(allocator_->malloc(sizeof(T) * self_cache_size, false));
        if (beam_width_ > 1) {
            key_caches_[1] = (T*)(allocator_->malloc(sizeof(T) * self_cache_size, false));
            value_caches_[1] = (T*)(allocator_->malloc(sizeof(T) * self_cache_size, false));
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
void Gpt<T>::freeBuffer()
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
bool Gpt<T>::isValidBatchSize(size_t batch_size)
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
bool Gpt<T>::isValidSeqLen(size_t seq_len)
{
    if (seq_len + 1 <= max_seq_len_) {
        return true;
    }
    else {
        freeBuffer();
        max_seq_len_ = (seq_len + 1) * 1.2;
        return true;
    }
}

template<typename T>
bool Gpt<T>::isValidInputSeqLen(size_t seq_len)
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
Gpt<T>::Gpt(size_t max_batch_size,
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
            cudaStream_t stream,
            cublasMMWrapper* cublas_wrapper,
            IAllocator* allocator,
            bool is_free_buffer_after_forward,
            cudaDeviceProp* cuda_device_prop,
            bool sparse):
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
    repetition_penalty_(repetition_penalty)
{
    vocab_size_padded_ = vocab_size_;
    if (std::is_same<half, T>::value) {
        vocab_size_padded_ = ((size_t)ceil(vocab_size_padded_ / 8.) * 8);
    }

    initialize();
}

template<typename T>
Gpt<T>::Gpt(Gpt<T> const& gpt):
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
    vocab_size_padded_(gpt.vocab_size_padded_)
{
    initialize();
}

template<typename T>
Gpt<T>::~Gpt()
{
    delete gpt_decoder_;
    delete dynamic_decode_;
    freeBuffer();
}

template<typename T>
void Gpt<T>::forward(std::vector<Tensor>* output_tensors,
                     const std::vector<Tensor>* input_tensors,
                     const GptWeight<T>* gpt_weights)
{
    // input_tensors:
    //      input_ids [batch_size * beam, max_input_length]
    //      input_lengths [batch_size * beam]
    //      max_output_seq_len [1] on cpu

    // output_tensors:
    //      output_ids [batch_size, beam, max_output_seq_len]
    //      parent_ids [max_output_seq_len, batch_size, beam]
    //      sequence_length [batch_size * beam]
    //      output_cum_log_probs [request_output_seq_len, batch_size, beam], must be float*.
    //          It leads to additional computing cost. If we don't need this result, please put nullptr

    // Step is from max_input_length ~ max_output_seq_len,
    // When step = k,  we put output ids and caches at step k, and the sequence_length would be k - 1 before
    // complete this step.
    // When there is no input_ids, put the start token at step 0 of output_ids_buf_. After forward, only copy
    // the step 1 ~ max_output_seq_len of output_ids_buf_ to output_tensors->at(0).data

    FT_CHECK(input_tensors->size() == 3);
    FT_CHECK(output_tensors->size() == 4);
    isValidSeqLen(output_tensors->at(0).shape[2]);
    isValidBatchSize(output_tensors->at(0).shape[0]);
    isValidInputSeqLen(input_tensors->at(0).shape[1]);
    allocateBuffer();

    int max_input_length = input_tensors->at(0).shape[1];
    const int* input_length_ptr = (const int*)(input_tensors->at(1).data);
    const size_t max_output_seq_len = (size_t)(*(int*)input_tensors->at(2).data)
                                      + (max_input_length == 0 ? 1 : 0);  // additional 1 to put start token

    const size_t batch_size = output_tensors->at(0).shape[0];
    int* sequence_lengths = (int*)(output_tensors->at(2).data);
    const DataType data_type = getTensorType<T>();

    float* output_cum_log_probs = (float*)(output_tensors->at(3).data);

    // initialize the output ids and parent ids
    cudaMemsetAsync(output_ids_buf_, 0, sizeof(int) * batch_size * beam_width_ * max_seq_len_, stream_);
    cudaMemsetAsync(parent_ids_buf_, 0, sizeof(int) * batch_size * beam_width_ * max_seq_len_, stream_);

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
        const std::vector<size_t> self_k_cache_size = {num_layer_,
                                                       batch_size * beam_width_,
                                                       head_num_,
                                                       size_per_head_ / (16 / sizeof(T)),
                                                       max_output_seq_len,
                                                       16 / sizeof(T)};
        const std::vector<size_t> self_v_cache_size = {
            num_layer_, batch_size * beam_width_, head_num_, max_output_seq_len, size_per_head_};

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
        sync_check_cuda_error();

        std::vector<Tensor> decoder_input_tensors{
            Tensor{MEMORY_GPU,
                   data_type,
                   {batch_size * beam_width_, (size_t)max_input_length, hidden_units_},
                   context_decoder_input_buf_},
            Tensor{MEMORY_GPU,
                   data_type,
                   {batch_size * beam_width_, 1, (size_t)max_input_length, (size_t)max_input_length},
                   input_attention_mask_}};

        const int src_cache_id = beam_width_ > 1 ? (max_input_length - 1) & 0x1 : 0;
        std::vector<Tensor> decoder_output_tensors{
            Tensor{MEMORY_GPU,
                   data_type,
                   {batch_size * beam_width_, (size_t)max_input_length, hidden_units_},
                   context_decoder_output_buf_},
            Tensor{MEMORY_GPU, data_type, self_k_cache_size, key_caches_[src_cache_id]},
            Tensor{MEMORY_GPU, data_type, self_v_cache_size, value_caches_[src_cache_id]}};

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

    const std::vector<size_t> self_k_cache_size = {num_layer_,
                                                   batch_size * beam_width_,
                                                   head_num_,
                                                   size_per_head_ / (16 / sizeof(T)),
                                                   max_output_seq_len,
                                                   16 / sizeof(T)};
    const std::vector<size_t> self_v_cache_size = {
        num_layer_, batch_size * beam_width_, head_num_, max_output_seq_len, size_per_head_};

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
        cudaD2Hcpy(h_finished_buf_, finished_buf_, batch_size * beam_width_);
        uint sum = 0;
        for (uint i = 0; i < batch_size * beam_width_; i++) {
            sum += (int)h_finished_buf_[i];
        }
        if (sum == batch_size * beam_width_) {
            break;
        }

        const int src_cache_id = beam_width_ > 1 ? (step - 1) & 0x1 : 0;
        const int tgt_cache_id = 1 - src_cache_id;
        sync_check_cuda_error();
        invokeEmbeddingLookupPosEncoding(decoder_input_buf_,
                                         gpt_weights->pre_decoder_embedding_table,
                                         gpt_weights->position_encoding_table,
                                         output_ids_buf_,
                                         input_length_ptr,
                                         batch_size * beam_width_,
                                         hidden_units_,
                                         (T)(1.0f),
                                         step - 1,
                                         max_input_length,
                                         batch_size * beam_width_,
                                         0,
                                         stream_);
        sync_check_cuda_error();

        std::vector<Tensor> decoder_input_tensors{
            Tensor{MEMORY_GPU, data_type, {batch_size * beam_width_, hidden_units_}, decoder_input_buf_},
            Tensor{MEMORY_GPU, TYPE_BOOL, {batch_size * beam_width_}, finished_buf_},
            output_tensors->at(2),
            Tensor{MEMORY_GPU, TYPE_INT32, {batch_size, beam_width_}, input_length_ptr},
            Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_length},
            Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step}};
        std::vector<Tensor> decoder_output_tensors{
            Tensor{MEMORY_GPU, data_type, {batch_size * beam_width_, hidden_units_}, decoder_output_buf_},
            Tensor{MEMORY_GPU, data_type, self_k_cache_size, key_caches_[src_cache_id]},
            Tensor{MEMORY_GPU, data_type, self_v_cache_size, value_caches_[src_cache_id]}};
        gpt_decoder_->forward(&decoder_output_tensors, &decoder_input_tensors, &gpt_weights->decoder_layer_weights);

        invokeGeneralLayerNorm(normed_decoder_output_buf_,
                               decoder_output_buf_,
                               gpt_weights->post_decoder_layernorm.gamma,
                               gpt_weights->post_decoder_layernorm.beta,
                               batch_size * beam_width_,
                               hidden_units_,
                               stream_);
        sync_check_cuda_error();

        cublas_wrapper_->Gemm(CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              vocab_size_padded_,  // n
                              batch_size * beam_width_,
                              hidden_units_,  // k
                              padded_embedding_kernel_ptr_,
                              hidden_units_,  // k
                              normed_decoder_output_buf_,
                              hidden_units_,  // k
                              logits_buf_,
                              vocab_size_padded_ /* n */);

        std::vector<Tensor>* dynamic_decode_input_tensors;
        std::vector<Tensor>* dynamic_decode_output_tensors;

        const int tmp_ite = 0;
        if (beam_width_ > 1) {
            dynamic_decode_input_tensors = new std::vector<Tensor>{
                Tensor{MEMORY_GPU, data_type, {batch_size, beam_width_, vocab_size_padded_}, logits_buf_},
                Tensor{MEMORY_GPU, data_type, {vocab_size_padded_}, nullptr},
                Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step},
                Tensor{MEMORY_GPU, data_type, self_k_cache_size, key_caches_[src_cache_id]},
                Tensor{MEMORY_GPU, data_type, self_v_cache_size, value_caches_[src_cache_id]},
                Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_length},
                Tensor{MEMORY_GPU, TYPE_INT32, {batch_size, beam_width_}, input_length_ptr},
                Tensor{MEMORY_CPU, TYPE_INT32, {1}, &tmp_ite}};

            dynamic_decode_output_tensors = new std::vector<Tensor>{
                Tensor{MEMORY_GPU, TYPE_INT32, {max_output_seq_len, batch_size, beam_width_}, output_ids_buf_},
                Tensor{MEMORY_GPU, TYPE_BOOL, {batch_size * beam_width_}, finished_buf_},
                Tensor{MEMORY_GPU, TYPE_FP32, {batch_size * beam_width_}, cum_log_probs_},
                Tensor{MEMORY_GPU, TYPE_INT32, {max_output_seq_len, batch_size, beam_width_}, parent_ids_buf_},
                output_tensors->at(2),
                Tensor{MEMORY_GPU, data_type, self_k_cache_size, key_caches_[tgt_cache_id]},
                Tensor{MEMORY_GPU, data_type, self_v_cache_size, value_caches_[tgt_cache_id]}};
        }
        else {
            dynamic_decode_input_tensors = new std::vector<Tensor>{
                Tensor{MEMORY_GPU, data_type, {batch_size, beam_width_, vocab_size_padded_}, logits_buf_},
                Tensor{MEMORY_GPU, data_type, {vocab_size_padded_}, nullptr},
                Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step},
                Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_length},
                Tensor{MEMORY_GPU, TYPE_INT32, {batch_size, beam_width_}, input_length_ptr},
                Tensor{MEMORY_CPU, TYPE_INT32, {1}, &tmp_ite}};

            dynamic_decode_output_tensors = new std::vector<Tensor>{
                Tensor{MEMORY_GPU, TYPE_INT32, {max_output_seq_len, batch_size, beam_width_}, output_ids_buf_},
                Tensor{MEMORY_GPU, TYPE_BOOL, {batch_size * beam_width_}, finished_buf_},
                output_tensors->at(2),
                Tensor{MEMORY_GPU,
                       TYPE_FP32,
                       {max_seq_len_, batch_size, beam_width_},
                       output_cum_log_probs == nullptr ?
                           nullptr :
                           output_cum_log_probs + (step - max_input_length) * batch_size * beam_width_}};
        }

        dynamic_decode_->forward(dynamic_decode_output_tensors, dynamic_decode_input_tensors);

        delete dynamic_decode_input_tensors;
        delete dynamic_decode_output_tensors;
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

            //transpose and take output_parent_ids as inter buffer
            invokeTransposeAxis01((int*)output_tensors->at(0).data,
                                  (int*)output_tensors->at(1).data,
                                  max_output_seq_len - 1, batch_size * beam_width_, 1, stream_);

            cudaD2Dcpy((int*)output_tensors->at(1).data,
                       parent_ids_buf_ + batch_size * beam_width_,
                       batch_size * beam_width_ * (max_output_seq_len - 1));

        }
        else {
            // For sampling, only transpose the results to output_tensor
            invokeTransposeAxis01((int*)output_tensors->at(0).data,
                                  output_ids_buf_ + batch_size * beam_width_,
                                  max_output_seq_len - 1, batch_size * beam_width_, 1, stream_);
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

            //transpose and take output_parent_ids as inter buffer
            invokeTransposeAxis01((int*)output_tensors->at(0).data,
                                  (int*)output_tensors->at(1).data,
                                  max_output_seq_len, batch_size * beam_width_, 1, stream_);

            cudaD2Dcpy((int*)output_tensors->at(1).data,
                       parent_ids_buf_,
                       batch_size * beam_width_ * max_output_seq_len);

        }
        else {
            // For sampling, only transpose the results to output_tensor
            invokeTransposeAxis01((int*)output_tensors->at(0).data,
                                  output_ids_buf_,
                                  max_output_seq_len, batch_size * beam_width_, 1, stream_);
        }
    }
}

template class Gpt<float>;
template class Gpt<half>;

}  // namespace fastertransformer
