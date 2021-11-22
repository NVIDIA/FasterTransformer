/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/t5/T5Decoding.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/layers/beam_search_layers/BaseBeamSearchLayer.h"
#include "src/fastertransformer/layers/beam_search_layers/BeamSearchLayer.h"
#include "src/fastertransformer/layers/beam_search_layers/OnlineBeamSearchLayer.h"
#include "src/fastertransformer/layers/sampling_layers/TopKSamplingLayer.h"
#include "src/fastertransformer/layers/sampling_layers/TopKTopPSamplingLayer.h"
#include "src/fastertransformer/layers/sampling_layers/TopPSamplingLayer.h"

namespace fastertransformer {

template<typename T>
void T5Decoding<T>::initialize()
{
    decoder_ = new T5Decoder<T>(max_batch_size_ * beam_width_,
                                head_num_,
                                size_per_head_,
                                inter_size_,
                                d_model_,
                                num_layer_,
                                stream_,
                                cublas_wrapper_,
                                allocator_,
                                is_free_buffer_after_forward_,
                                tensor_para_,
                                pipeline_para_);

    if (beam_width_ > 1) {
        if (beam_width_ < 16) {
            dynamic_decode_ = new OnlineBeamSearchLayer<T>(max_batch_size_,
                                                           head_num_ / tensor_para_.world_size_,
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
                                                     head_num_ / tensor_para_.world_size_,
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
                                                   0,
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
                                                   0,
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
                                                       0,
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
void T5Decoding<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {

        const size_t batchxbeam = max_batch_size_ * beam_width_;
        const size_t self_cache_size = (num_layer_ / pipeline_para_.world_size_) * batchxbeam * max_seq_len_
                                       * (hidden_units_ / tensor_para_.world_size_);
        const size_t mem_cache_size = (num_layer_ / pipeline_para_.world_size_) * batchxbeam * mem_max_seq_len_
                                      * (hidden_units_ / tensor_para_.world_size_);

        // Assume the dimension of encoder output is d_model_ too.
        if (beam_width_ > 1) {
            tiled_encoder_output_ = (T*)(allocator_->malloc(sizeof(T) * batchxbeam * mem_max_seq_len_ * d_model_));
            tiled_encoder_sequence_length_ = (int*)(allocator_->malloc(sizeof(int) * batchxbeam));
        }

        if (vocab_size_ != vocab_size_padded_) {
            padded_embedding_kernel_ = (T*)(allocator_->malloc(sizeof(T) * d_model_ * vocab_size_padded_, true));
            padded_embedding_kernel_ptr_ = padded_embedding_kernel_;
        }
        relative_attention_bias_ = (T*)(allocator_->malloc(sizeof(T) * head_num_ * max_seq_len_ * max_seq_len_, false));

        decoder_input_buf_ = (T*)(allocator_->malloc(sizeof(T) * batchxbeam * d_model_, false));
        decoder_output_buf_ = (T*)(allocator_->malloc(sizeof(T) * batchxbeam * d_model_, false));
        normed_decoder_output_buf_ = (T*)(allocator_->malloc(sizeof(T) * batchxbeam * d_model_, false));
        logits_buf_ = (T*)(allocator_->malloc(sizeof(T) * batchxbeam * vocab_size_padded_, false));
        nccl_logits_buf_ = (T*)(allocator_->malloc(sizeof(T) * batchxbeam * vocab_size_padded_, false));
        cum_log_probs_ = (float*)(allocator_->malloc(sizeof(float) * batchxbeam, false));
        finished_buf_ = (bool*)(allocator_->malloc(sizeof(bool) * batchxbeam, false));
        h_finished_buf_ = new bool[batchxbeam];

        key_caches_[0] = (T*)(allocator_->malloc(sizeof(T) * self_cache_size, false));
        value_caches_[0] = (T*)(allocator_->malloc(sizeof(T) * self_cache_size, false));
        if (beam_width_ > 1) {
            key_caches_[1] = (T*)(allocator_->malloc(sizeof(T) * self_cache_size, false));
            value_caches_[1] = (T*)(allocator_->malloc(sizeof(T) * self_cache_size, false));
        }
        key_mem_caches_ = (T*)(allocator_->malloc(sizeof(T) * mem_cache_size, false));
        value_mem_caches_ = (T*)(allocator_->malloc(sizeof(T) * mem_cache_size, false));

        output_ids_buf_ = (int*)(allocator_->malloc(sizeof(int) * batchxbeam * max_seq_len_, false));
        parent_ids_buf_ = (int*)(allocator_->malloc(sizeof(int) * batchxbeam * max_seq_len_, false));

        is_allocate_buffer_ = true;
    }
}

template<typename T>
void T5Decoding<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        if (vocab_size_ != vocab_size_padded_) {
            padded_embedding_kernel_ptr_ = nullptr;
            allocator_->free(padded_embedding_kernel_);
        }

        if (beam_width_ > 1) {
            allocator_->free(tiled_encoder_output_);
            allocator_->free(tiled_encoder_sequence_length_);
        }

        allocator_->free(relative_attention_bias_);

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
        allocator_->free(key_mem_caches_);
        allocator_->free(value_mem_caches_);

        allocator_->free(output_ids_buf_);
        allocator_->free(parent_ids_buf_);

        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool T5Decoding<T>::isValidBatchSize(size_t batch_size)
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
bool T5Decoding<T>::isValidSeqLen(size_t seq_len)
{
    if ((seq_len + 1) <= max_seq_len_) {
        return true;
    }
    else {
        freeBuffer();
        max_seq_len_ = (seq_len + 1) * 1.2;
        return true;
    }
}

template<typename T>
bool T5Decoding<T>::isValidMemSeqLen(size_t seq_len)
{
    if (seq_len <= mem_max_seq_len_) {
        return true;
    }
    else {
        freeBuffer();
        mem_max_seq_len_ = seq_len * 1.2;
        return true;
    }
}

template<typename T>
T5Decoding<T>::T5Decoding(size_t max_batch_size,
                          size_t max_seq_len,
                          size_t mem_max_seq_len,
                          size_t beam_width,
                          size_t head_num,
                          size_t size_per_head,
                          size_t inter_size,
                          size_t d_model,
                          size_t num_layer,
                          size_t vocab_size,
                          size_t num_bucket,
                          size_t max_distance,
                          int start_id,
                          int end_id,
                          float beam_search_diversity_rate,
                          size_t top_k,
                          float top_p,
                          float temperature,
                          float len_penalty,
                          float repetition_penalty,
                          cudaStream_t stream,
                          cublasMMWrapper* cublas_wrapper,
                          IAllocator* allocator,
                          bool is_free_buffer_after_forward,
                          cudaDeviceProp* cuda_device_prop,
                          NcclParam tensor_para,
                          NcclParam pipeline_para):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len + 1),  // allocater additional one to put the start token
    mem_max_seq_len_(mem_max_seq_len),
    beam_width_(beam_width),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    d_model_(d_model),
    num_layer_(num_layer),
    vocab_size_(vocab_size),
    num_bucket_(num_bucket),
    max_distance_(max_distance),
    start_id_(start_id),
    end_id_(end_id),
    beam_search_diversity_rate_(beam_search_diversity_rate),
    hidden_units_(head_num_ * size_per_head_),
    top_k_(top_k),
    top_p_(top_p),
    temperature_(temperature),
    len_penalty_(len_penalty),
    repetition_penalty_(repetition_penalty),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para)
{
    vocab_size_padded_ = ((size_t)ceil(vocab_size_ / 1. / tensor_para_.world_size_) * tensor_para_.world_size_);
    if (std::is_same<half, T>::value) {
        vocab_size_padded_ = ((size_t)ceil(vocab_size_padded_ / 8.) * 8);
    }

    initialize();
}

template<typename T>
T5Decoding<T>::T5Decoding(T5Decoding<T> const& decoding):
    BaseLayer(decoding),
    max_batch_size_(decoding.max_batch_size_),
    max_seq_len_(decoding.max_seq_len_),
    mem_max_seq_len_(decoding.mem_max_seq_len_),
    beam_width_(decoding.beam_width_),
    head_num_(decoding.head_num_),
    size_per_head_(decoding.size_per_head_),
    inter_size_(decoding.inter_size_),
    d_model_(decoding.d_model_),
    num_layer_(decoding.num_layer_),
    vocab_size_(decoding.vocab_size_),
    num_bucket_(decoding.num_bucket_),
    max_distance_(decoding.max_distance_),
    start_id_(decoding.start_id_),
    end_id_(decoding.end_id_),
    beam_search_diversity_rate_(decoding.beam_search_diversity_rate_),
    hidden_units_(decoding.hidden_units_),
    top_k_(decoding.top_k_),
    top_p_(decoding.top_p_),
    temperature_(decoding.temperature_),
    len_penalty_(decoding.len_penalty_),
    repetition_penalty_(decoding.repetition_penalty_),
    vocab_size_padded_(decoding.vocab_size_padded_),
    tensor_para_(decoding.tensor_para_),
    pipeline_para_(decoding.pipeline_para_)
{
    initialize();
}

template<typename T>
T5Decoding<T>::~T5Decoding()
{
    delete decoder_;
    delete dynamic_decode_;
    freeBuffer();
}

template<typename T>
void T5Decoding<T>::forward(std::vector<Tensor>* output_tensors,
                            const std::vector<Tensor>* input_tensors,
                            const T5DecodingWeight<T>* decoding_weights)
{
    // input_tensors:
    //      encoder_output [batch_size, mem_max_seq_len, memory_hidden_dimension]
    //      encoder_sequence_length [batch_size]

    // output_tensors:
    //      output_ids [batch_size, beam, max_seq_len]
    //      parent_ids [batch_size, beam, max_seq_len]
    //      sequence_length [batch_size, beam], record the number of generated token, except the start token

    // Step is from 1 ~ max_seq_len,
    // When step = k,  we put output ids and caches at step k, and the sequence_length would be k - 1 before
    // complete this step.

    FT_CHECK(input_tensors->size() == 2);
    FT_CHECK(output_tensors->size() == 3);
    FT_CHECK(input_tensors->at(0).shape.size() == 3);
    isValidSeqLen(output_tensors->at(0).shape[2]);
    isValidBatchSize(output_tensors->at(0).shape[0]);
    isValidMemSeqLen(input_tensors->at(0).shape[1]);
    allocateBuffer();

    FT_CHECK(input_tensors->at(0).shape[2] == d_model_);

    const size_t batch_size = output_tensors->at(0).shape[0];
    const int max_input_length = 0;
    const DataType data_type = getTensorType<T>();
    const size_t mem_max_seq_len = input_tensors->at(0).shape[1];
    int* sequence_lengths = (int*)output_tensors->at(2).data;

    cudaMemset((int*)output_tensors->at(0).data, 0, sizeof(int) * batch_size * beam_width_ * (max_seq_len_ - 1));
    cudaMemset((int*)output_tensors->at(1).data, 0, sizeof(int) * batch_size * beam_width_ * (max_seq_len_ - 1));

    if (beam_width_ > 1) {
        invokeTileEncoderResults(tiled_encoder_output_,
                                 tiled_encoder_sequence_length_,
                                 (const T*)(input_tensors->at(0).data),
                                 (const int*)(input_tensors->at(1).data),
                                 batch_size,
                                 beam_width_,
                                 mem_max_seq_len,
                                 d_model_,
                                 stream_);
        sync_check_cuda_error();
        encoder_output_ptr_ = tiled_encoder_output_;
        encoder_sequence_length_ptr_ = tiled_encoder_sequence_length_;
    }
    else {
        encoder_output_ptr_ = (const T*)(input_tensors->at(0).data);
        encoder_sequence_length_ptr_ = (const int*)(input_tensors->at(1).data);
    }

    invokeDecodingInitialize(finished_buf_,
                             sequence_lengths,
                             output_ids_buf_,
                             cum_log_probs_,
                             start_id_,
                             batch_size,
                             beam_width_,
                             max_input_length,
                             stream_);
    sync_check_cuda_error();
    invokeBuildRelativeAttentionBias(relative_attention_bias_,
                                     decoding_weights->relative_attention_bias,
                                     head_num_,
                                     max_seq_len_,
                                     num_bucket_,
                                     false,
                                     max_distance_,
                                     stream_);
    sync_check_cuda_error();

    if (vocab_size_ == vocab_size_padded_) {
        padded_embedding_kernel_ptr_ = decoding_weights->post_decoder_embedding.kernel;
    }
    else {
        invokePaddingEmbeddingKernel(padded_embedding_kernel_,
                                     decoding_weights->post_decoder_embedding.kernel,
                                     d_model_,
                                     vocab_size_,
                                     vocab_size_padded_,
                                     stream_);
        sync_check_cuda_error();
    }

    const std::vector<size_t> self_k_cache_shape = {num_layer_ / pipeline_para_.world_size_,
                                                    batch_size * beam_width_,
                                                    head_num_ / tensor_para_.world_size_,
                                                    size_per_head_ / (16 / sizeof(T)),
                                                    max_seq_len_,
                                                    16 / sizeof(T)};
    const std::vector<size_t> self_v_cache_shape = {num_layer_ / pipeline_para_.world_size_,
                                                    batch_size * beam_width_,
                                                    head_num_ / tensor_para_.world_size_,
                                                    (size_t)(max_seq_len_),
                                                    size_per_head_};
    const std::vector<size_t> mem_cache_shape = {num_layer_ / pipeline_para_.world_size_,
                                                 batch_size * beam_width_,
                                                 mem_max_seq_len,
                                                 head_num_ / tensor_para_.world_size_ * size_per_head_};

    for (int step = 1; step <= (int)max_seq_len_; step++) {
        // cudaD2Hcpy(h_finished_buf_, finished_buf_, batch_size * beam_width_);
        // uint sum = 0;
        // for (uint i = 0; i < batch_size * beam_width_; i++) {
        //     sum += (int)h_finished_buf_[i];
        // }
        // if (sum == batch_size * beam_width_) {
        //     break;
        // }

        const int src_cache_id = beam_width_ > 1 ? (step - 1) & 0x1 : 0;
        const int tgt_cache_id = 1 - src_cache_id;

        const size_t local_batch_size = getLocalBatchSize(batch_size, 1, pipeline_para_.world_size_);
        // const size_t local_batch_size = batch_size / 2;
        FT_CHECK(batch_size % local_batch_size == 0);
        const size_t iteration_num = batch_size / local_batch_size;

        for (uint ite = 0; ite < iteration_num; ++ite) {
            const int id_offset = ite * local_batch_size * beam_width_;
            const int d_model_offset = id_offset * d_model_;
            const int vocab_size_units_offset = id_offset * vocab_size_padded_;

            if (pipeline_para_.rank_ == 0) {
                invokeEmbeddingLookupPosEncoding(decoder_input_buf_ + d_model_offset,
                                                 decoding_weights->pre_decoder_embedding_table,
                                                 (T*)nullptr,
                                                 output_ids_buf_ + id_offset,
                                                 nullptr,
                                                 local_batch_size * beam_width_,
                                                 d_model_,
                                                 (T)1.0f,
                                                 step - 1,
                                                 0,
                                                 batch_size * beam_width_,
                                                 0,
                                                 stream_);
                sync_check_cuda_error();
            }

            std::vector<Tensor> decoder_input_tensors{
                Tensor{MEMORY_GPU,
                       data_type,
                       {local_batch_size * beam_width_, d_model_},
                       decoder_input_buf_ + d_model_offset},
                Tensor{MEMORY_GPU,
                       data_type,
                       {local_batch_size * beam_width_, input_tensors->at(0).shape[1], input_tensors->at(0).shape[2]},
                       encoder_output_ptr_ + id_offset * input_tensors->at(0).shape[1] * input_tensors->at(0).shape[2]},
                Tensor{
                    MEMORY_GPU, TYPE_INT32, {local_batch_size * beam_width_}, encoder_sequence_length_ptr_ + id_offset},
                Tensor{MEMORY_GPU, TYPE_BOOL, {local_batch_size * beam_width_}, finished_buf_ + id_offset},
                Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step},
                Tensor{MEMORY_GPU, TYPE_INT32, {local_batch_size * beam_width_}, sequence_lengths + id_offset},
                Tensor{MEMORY_GPU, data_type, {1, head_num_, max_seq_len_, max_seq_len_}, relative_attention_bias_},
                Tensor{MEMORY_CPU, TYPE_UINT32, {1}, &ite}};

            std::vector<Tensor> decoder_output_tensors{
                Tensor{MEMORY_GPU,
                       data_type,
                       {local_batch_size * beam_width_, d_model_},
                       decoder_output_buf_ + d_model_offset},
                Tensor{MEMORY_GPU, data_type, self_k_cache_shape, key_caches_[src_cache_id]},
                Tensor{MEMORY_GPU, data_type, self_v_cache_shape, value_caches_[src_cache_id]},
                Tensor{MEMORY_GPU, data_type, mem_cache_shape, key_mem_caches_},
                Tensor{MEMORY_GPU, data_type, mem_cache_shape, value_mem_caches_}};
            decoder_->forward(
                &decoder_output_tensors, &decoder_input_tensors, &decoding_weights->decoder_layer_weights);

            if (pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) {
                invokeGeneralT5LayerNorm(normed_decoder_output_buf_ + d_model_offset,
                                         decoder_output_buf_ + d_model_offset,
                                         decoding_weights->post_decoder_layernorm.gamma,
                                         local_batch_size * beam_width_,
                                         d_model_,
                                         stream_);
                sync_check_cuda_error();

                if (tensor_para_.world_size_ == 1) {
                    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                          CUBLAS_OP_N,
                                          vocab_size_padded_,  // n
                                          local_batch_size * beam_width_,
                                          d_model_,  // k
                                          padded_embedding_kernel_ptr_,
                                          d_model_,  // k
                                          normed_decoder_output_buf_ + d_model_offset,
                                          d_model_,  // k
                                          logits_buf_ + vocab_size_units_offset,
                                          vocab_size_padded_ /* n */,
                                          1.0f / sqrt(d_model_),
                                          0.0f);
                }
                else {
                    const int local_vocab_size = vocab_size_padded_ / tensor_para_.world_size_;
                    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                          CUBLAS_OP_N,
                                          local_vocab_size,  // n
                                          local_batch_size * beam_width_,
                                          d_model_,  // k
                                          padded_embedding_kernel_ptr_
                                              + tensor_para_.rank_ * local_vocab_size * d_model_,
                                          d_model_,  // k
                                          normed_decoder_output_buf_ + d_model_offset,
                                          d_model_,  // k
                                          nccl_logits_buf_ + vocab_size_units_offset
                                              + tensor_para_.rank_ * local_batch_size * beam_width_ * local_vocab_size,
                                          local_vocab_size /* n */,
                                          1.0f / sqrt(d_model_),
                                          0.0f);
                    ftNcclAllGather(nccl_logits_buf_ + vocab_size_units_offset,
                                    nccl_logits_buf_ + vocab_size_units_offset,
                                    local_batch_size * beam_width_ * local_vocab_size,
                                    tensor_para_.rank_,
                                    tensor_para_.nccl_comm_,
                                    stream_);
                    check_cuda_error(cudaStreamSynchronize(stream_));
                    invokeTransposeAxis01(logits_buf_ + vocab_size_units_offset,
                                          nccl_logits_buf_ + vocab_size_units_offset,
                                          tensor_para_.world_size_,
                                          local_batch_size * beam_width_,
                                          local_vocab_size,
                                          stream_);
                }

                std::vector<Tensor>* dynamic_decode_input_tensors;
                std::vector<Tensor>* dynamic_decode_output_tensors;

                if (beam_width_ > 1) {
                    dynamic_decode_input_tensors = new std::vector<Tensor>{
                        Tensor{MEMORY_GPU,
                               data_type,
                               {local_batch_size, beam_width_, vocab_size_padded_},
                               logits_buf_ + vocab_size_units_offset},
                        Tensor{MEMORY_GPU, data_type, {vocab_size_padded_}, nullptr},
                        Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step},
                        Tensor{MEMORY_GPU, data_type, self_k_cache_shape, key_caches_[src_cache_id]},
                        Tensor{MEMORY_GPU, data_type, self_v_cache_shape, value_caches_[src_cache_id]},
                        Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_length},
                        Tensor{MEMORY_GPU, TYPE_INT32, {local_batch_size, beam_width_}, nullptr},
                        Tensor{MEMORY_CPU, TYPE_UINT32, {1}, &ite}};

                    dynamic_decode_output_tensors = new std::vector<Tensor>{
                        Tensor{MEMORY_GPU, TYPE_INT32, {max_seq_len_, batch_size, beam_width_}, output_ids_buf_},
                        Tensor{MEMORY_GPU, TYPE_BOOL, {local_batch_size * beam_width_}, finished_buf_ + id_offset},
                        Tensor{MEMORY_GPU, TYPE_FP32, {local_batch_size * beam_width_}, cum_log_probs_ + id_offset},
                        Tensor{MEMORY_GPU, TYPE_INT32, {max_seq_len_, batch_size, beam_width_}, parent_ids_buf_},
                        Tensor{MEMORY_GPU, TYPE_INT32, {local_batch_size * beam_width_}, sequence_lengths + id_offset},
                        Tensor{MEMORY_GPU, data_type, self_k_cache_shape, key_caches_[tgt_cache_id]},
                        Tensor{MEMORY_GPU, data_type, self_v_cache_shape, value_caches_[tgt_cache_id]}};
                }
                else {
                    dynamic_decode_input_tensors = new std::vector<Tensor>{
                        Tensor{MEMORY_GPU,
                               data_type,
                               {local_batch_size, beam_width_, vocab_size_padded_},
                               logits_buf_ + vocab_size_units_offset},
                        Tensor{MEMORY_GPU, data_type, {vocab_size_padded_}, nullptr},
                        Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step},
                        Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_length},
                        Tensor{MEMORY_GPU, TYPE_INT32, {local_batch_size, beam_width_}, nullptr},
                        Tensor{MEMORY_CPU, TYPE_UINT32, {1}, &ite}};

                    dynamic_decode_output_tensors = new std::vector<Tensor>{
                        Tensor{MEMORY_GPU, TYPE_INT32, {max_seq_len_, batch_size, beam_width_}, output_ids_buf_},
                        Tensor{MEMORY_GPU, TYPE_BOOL, {local_batch_size * beam_width_}, finished_buf_ + id_offset},
                        Tensor{MEMORY_GPU, TYPE_INT32, {local_batch_size * beam_width_}, sequence_lengths + id_offset},
                        Tensor{MEMORY_GPU, TYPE_FP32, {max_seq_len_, batch_size, beam_width_}, nullptr}};
                }

                dynamic_decode_->forward(dynamic_decode_output_tensors, dynamic_decode_input_tensors);

                delete dynamic_decode_input_tensors;
                delete dynamic_decode_output_tensors;
            }
        }

        if (pipeline_para_.world_size_ > 1) {
            NCCLCHECK(ncclGroupStart());
            ftNcclBroadCast(output_ids_buf_ + step * batch_size * beam_width_,
                            batch_size * beam_width_,
                            pipeline_para_.world_size_ - 1,
                            pipeline_para_.nccl_comm_,
                            stream_);

            ftNcclBroadCast(sequence_lengths,
                            batch_size * beam_width_,
                            pipeline_para_.world_size_ - 1,
                            pipeline_para_.nccl_comm_,
                            stream_);

            ftNcclBroadCast(finished_buf_,
                            batch_size * beam_width_,
                            pipeline_para_.world_size_ - 1,
                            pipeline_para_.nccl_comm_,
                            stream_);

            if (beam_width_ > 1) {
                ftNcclBroadCast(parent_ids_buf_ + step * batch_size * beam_width_,
                                batch_size * beam_width_,
                                pipeline_para_.world_size_ - 1,
                                pipeline_para_.nccl_comm_,
                                stream_);
            }
            NCCLCHECK(ncclGroupEnd());
            check_cuda_error(cudaStreamSynchronize(stream_));
            sync_check_cuda_error();
        }

        if (pipeline_para_.rank_ != pipeline_para_.world_size_ - 1 && beam_width_ > 1) {
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
                                           head_num_ / tensor_para_.world_size_,
                                           size_per_head_,
                                           step,
                                           max_seq_len_,
                                           batch_size * beam_width_ * head_num_ / tensor_para_.world_size_
                                               * size_per_head_ * max_seq_len_,
                                           num_layer_ / pipeline_para_.world_size_,
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

    // minus the sequence length of unfinished sentence by 1 since we start from 1.
    invokeMinusUnfinishedSeqlen(sequence_lengths, finished_buf_, batch_size * beam_width_, stream_);

    if (beam_width_ > 1) {
        // For beam search, do gather_tree
        invokeGatherTree((int*)output_tensors->at(1).data,
                         (int*)output_tensors->at(2).data,
                         max_seq_len_ - 1,
                         batch_size,
                         beam_width_,
                         output_ids_buf_ + batch_size * beam_width_,
                         parent_ids_buf_ + batch_size * beam_width_,
                         end_id_,
                         stream_);

        // transpose and take output_parent_ids as inter buffer
        invokeTransposeAxis01((int*)output_tensors->at(0).data,
                              (int*)output_tensors->at(1).data,
                              max_seq_len_ - 1,
                              batch_size * beam_width_,
                              1,
                              stream_);

        cudaD2Dcpy((int*)output_tensors->at(1).data,
                   parent_ids_buf_ + batch_size * beam_width_,
                   batch_size * beam_width_ * (max_seq_len_ - 1));
    }
    else {
        // For sampling, only transpose the results to output_tensor
        invokeTransposeAxis01((int*)output_tensors->at(0).data,
                              output_ids_buf_ + batch_size * beam_width_,
                              max_seq_len_ - 1,
                              batch_size * beam_width_,
                              1,
                              stream_);
    }
}

template class T5Decoding<float>;
template class T5Decoding<half>;

}  // namespace fastertransformer