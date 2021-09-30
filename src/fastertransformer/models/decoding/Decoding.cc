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

#include "src/fastertransformer/models/decoding/Decoding.h"
#include "src/fastertransformer/layers/beam_search_layers/BeamSearchLayer.h"
#include "src/fastertransformer/layers/beam_search_layers/OnlineBeamSearchLayer.h"
#include "src/fastertransformer/layers/sampling_layers/TopKSamplingLayer.h"
#include "src/fastertransformer/layers/sampling_layers/TopKTopPSamplingLayer.h"
#include "src/fastertransformer/layers/sampling_layers/TopPSamplingLayer.h"

namespace fastertransformer {

template<typename T>
void Decoding<T>::initialize()
{
    decoder_ = new Decoder<T>(max_batch_size_ * beam_width_,
                              head_num_,
                              size_per_head_,
                              inter_size_,
                              num_layer_,
                              stream_,
                              cublas_wrapper_,
                              allocator_,
                              is_free_buffer_after_forward_);

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
void Decoding<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {

        const size_t batchxbeam = max_batch_size_ * beam_width_;
        const size_t self_cache_size = num_layer_ * batchxbeam * max_seq_len_ * hidden_units_;
        const size_t mem_cache_size = num_layer_ * batchxbeam * mem_max_seq_len_ * hidden_units_;

        if (vocab_size_ != vocab_size_padded_) {
            padded_embedding_kernel_ = (T*)(allocator_->malloc(sizeof(T) * hidden_units_ * vocab_size_padded_, true));
            padded_embedding_bias_ = (T*)(allocator_->malloc(sizeof(T) * vocab_size_padded_, true));
            padded_embedding_kernel_ptr_ = padded_embedding_kernel_;
            padded_embedding_bias_ptr_ = padded_embedding_bias_;
        }

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
        key_mem_caches_ = (T*)(allocator_->malloc(sizeof(T) * mem_cache_size, false));
        value_mem_caches_ = (T*)(allocator_->malloc(sizeof(T) * mem_cache_size, false));

        padded_pos_embedding_bias_ = (T*)(allocator_->malloc(sizeof(T) * vocab_size_padded_, false));
        output_ids_buf_ = (int*)(allocator_->malloc(sizeof(int) * batchxbeam * max_seq_len_, false));
        parent_ids_buf_ = (int*)(allocator_->malloc(sizeof(int) * batchxbeam * max_seq_len_, false));

        is_allocate_buffer_ = true;
    }
}

template<typename T>
void Decoding<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        if (vocab_size_ != vocab_size_padded_) {
            padded_embedding_kernel_ptr_ = nullptr;
            padded_embedding_bias_ptr_ = nullptr;
            allocator_->free(padded_embedding_kernel_);
            allocator_->free(padded_embedding_bias_);
        }

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
        allocator_->free(key_mem_caches_);
        allocator_->free(value_mem_caches_);

        allocator_->free(padded_pos_embedding_bias_);

        allocator_->free(output_ids_buf_);
        allocator_->free(parent_ids_buf_);

        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool Decoding<T>::isValidBatchSize(size_t batch_size)
{
    if (max_batch_size_ == 0) {
        max_batch_size_ = batch_size;
        return true;
    }
    else {
        return batch_size <= max_batch_size_;
    }
}

template<typename T>
bool Decoding<T>::isValidSeqLen(size_t seq_len)
{
    if (max_seq_len_ == 0) {
        // allocater additional one to put the start token
        max_seq_len_ = seq_len + 1;
        return true;
    }
    else {
        return seq_len <= max_seq_len_;
    }
}

template<typename T>
bool Decoding<T>::isValidMemSeqLen(size_t seq_len)
{
    if (mem_max_seq_len_ == 0) {
        mem_max_seq_len_ = seq_len;
        return true;
    }
    else {
        return seq_len <= mem_max_seq_len_;
    }
}

template<typename T>
Decoding<T>::Decoding(size_t max_batch_size,
                      size_t max_seq_len,
                      size_t mem_max_seq_len,
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
                      float temperature,
                      float len_penalty,
                      float repetition_penalty,
                      cudaStream_t stream,
                      cublasMMWrapper* cublas_wrapper,
                      IAllocator* allocator,
                      bool is_free_buffer_after_forward,
                      cudaDeviceProp* cuda_device_prop):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len + 1),  // allocater additional one to put the start token
    mem_max_seq_len_(mem_max_seq_len),
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
Decoding<T>::Decoding(Decoding<T> const& decoding):
    BaseLayer(decoding),
    max_batch_size_(decoding.max_batch_size_),
    max_seq_len_(decoding.max_seq_len_),
    mem_max_seq_len_(decoding.mem_max_seq_len_),
    beam_width_(decoding.beam_width_),
    head_num_(decoding.head_num_),
    size_per_head_(decoding.size_per_head_),
    inter_size_(decoding.inter_size_),
    num_layer_(decoding.num_layer_),
    vocab_size_(decoding.vocab_size_),
    start_id_(decoding.start_id_),
    end_id_(decoding.end_id_),
    beam_search_diversity_rate_(decoding.beam_search_diversity_rate_),
    hidden_units_(decoding.hidden_units_),
    top_k_(decoding.top_k_),
    top_p_(decoding.top_p_),
    temperature_(decoding.temperature_),
    len_penalty_(decoding.len_penalty_),
    repetition_penalty_(decoding.repetition_penalty_),
    vocab_size_padded_(decoding.vocab_size_padded_)
{
    initialize();
}

template<typename T>
Decoding<T>::~Decoding()
{
    delete decoder_;
    delete dynamic_decode_;
    freeBuffer();
}

template<typename T>
void Decoding<T>::forward(std::vector<Tensor>* output_tensors,
                          const std::vector<Tensor>* input_tensors,
                          const DecodingWeight<T>* decoding_weights)
{
    // input_tensors:
    //      encoder_output [batch_size * beam, mem_max_seq_len, memory_hidden_dimension]
    //      encoder_sequence_length [batch_size * beam]

    // output_tensors:
    //      output_ids [max_seq_len, batch_size, beam]
    //      parent_ids [max_seq_len, batch_size, beam]
    //      sequence_length [batch_size, beam], record the number of generated token, except the start token

    // Step is from 1 ~ max_seq_len,
    // When step = k,  we put output ids and caches at step k, and the sequence_length would be k - 1 before
    // complete this step.

    FT_CHECK(input_tensors->size() == 2);
    FT_CHECK(output_tensors->size() == 3);
    isValidSeqLen(output_tensors->at(0).shape[0]);
    isValidBatchSize(output_tensors->at(0).shape[1]);
    isValidMemSeqLen(input_tensors->at(0).shape[1]);
    allocateBuffer();

    const size_t batch_size = output_tensors->at(0).shape[1];
    const int max_input_length = 0;
    const DataType data_type = getTensorType<T>();
    const size_t mem_max_seq_len = input_tensors->at(0).shape[1];

    invokeDecodingInitialize(finished_buf_,
                             (int*)output_tensors->at(2).data,
                             output_ids_buf_,
                             cum_log_probs_,
                             start_id_,
                             batch_size,
                             beam_width_,
                             max_input_length,
                             stream_);
    sync_check_cuda_error();

    if (vocab_size_ == vocab_size_padded_) {
        padded_embedding_kernel_ptr_ = decoding_weights->post_decoder_embedding.kernel;
        padded_embedding_bias_ptr_ = decoding_weights->post_decoder_embedding.bias;
    }
    else {
        invokePaddingEmbedding(padded_embedding_kernel_,
                               padded_embedding_bias_,
                               decoding_weights->post_decoder_embedding.kernel,
                               decoding_weights->post_decoder_embedding.bias,
                               hidden_units_,
                               vocab_size_,
                               vocab_size_padded_,
                               stream_);
        sync_check_cuda_error();
    }

    const std::vector<size_t> self_k_cache_size = {num_layer_,
                                                   batch_size * beam_width_,
                                                   head_num_,
                                                   size_per_head_ / (16 / sizeof(T)),
                                                   max_seq_len_,
                                                   16 / sizeof(T)};
    const std::vector<size_t> self_v_cache_size = {
        num_layer_, batch_size * beam_width_, head_num_, (size_t)(max_seq_len_), size_per_head_};

    for (int step = 1; step <= (int)max_seq_len_; step++) {
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

        invokeEmbeddingLookupPosEncoding(decoder_input_buf_,
                                         decoding_weights->pre_decoder_embedding_table,
                                         decoding_weights->position_encoding_table,
                                         output_ids_buf_,
                                         nullptr,
                                         batch_size * beam_width_,
                                         hidden_units_,
                                         (T)sqrtf(float(hidden_units_)),
                                         step - 1,
                                         0,
                                         batch_size * beam_width_,
                                         0,
                                         stream_);
        sync_check_cuda_error();

        std::vector<Tensor> decoder_input_tensors{
            Tensor{MEMORY_GPU, data_type, {batch_size * beam_width_, hidden_units_}, decoder_input_buf_},
            input_tensors->at(0),
            input_tensors->at(1),
            Tensor{MEMORY_GPU, TYPE_BOOL, {batch_size * beam_width_}, finished_buf_},
            Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step},
            output_tensors->at(2)};

        std::vector<Tensor> decoder_output_tensors{
            Tensor{MEMORY_GPU, data_type, {batch_size * beam_width_, hidden_units_}, decoder_output_buf_},
            Tensor{MEMORY_GPU, data_type, self_k_cache_size, key_caches_[src_cache_id]},
            Tensor{MEMORY_GPU, data_type, self_v_cache_size, value_caches_[src_cache_id]},
            Tensor{MEMORY_GPU,
                   data_type,
                   {num_layer_, batch_size * beam_width_, mem_max_seq_len, hidden_units_},
                   key_mem_caches_},
            Tensor{MEMORY_GPU,
                   data_type,
                   {num_layer_, batch_size * beam_width_, mem_max_seq_len, hidden_units_},
                   value_mem_caches_}};
        decoder_->forward(&decoder_output_tensors, &decoder_input_tensors, &decoding_weights->decoder_layer_weights);

        invokeGeneralLayerNorm(normed_decoder_output_buf_,
                               decoder_output_buf_,
                               decoding_weights->post_decoder_layernorm.gamma,
                               decoding_weights->post_decoder_layernorm.beta,
                               batch_size * beam_width_,
                               hidden_units_,
                               stream_);
        sync_check_cuda_error();

        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              vocab_size_padded_,  // n
                              batch_size * beam_width_,
                              hidden_units_,  // k
                              padded_embedding_kernel_ptr_,
                              vocab_size_padded_,  // n
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
                Tensor{MEMORY_GPU, data_type, {vocab_size_padded_}, padded_embedding_bias_ptr_},
                Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step},
                Tensor{MEMORY_GPU, data_type, self_k_cache_size, key_caches_[src_cache_id]},
                Tensor{MEMORY_GPU, data_type, self_v_cache_size, value_caches_[src_cache_id]},
                Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_length},
                Tensor{MEMORY_GPU, TYPE_INT32, {batch_size, beam_width_}, nullptr},
                Tensor{MEMORY_CPU, TYPE_INT32, {1}, &tmp_ite}};

            dynamic_decode_output_tensors = new std::vector<Tensor>{
                Tensor{MEMORY_GPU, TYPE_INT32, {max_seq_len_, batch_size, beam_width_}, output_ids_buf_},
                Tensor{MEMORY_GPU, TYPE_BOOL, {batch_size * beam_width_}, finished_buf_},
                Tensor{MEMORY_GPU, TYPE_FP32, {batch_size * beam_width_}, cum_log_probs_},
                Tensor{MEMORY_GPU, TYPE_INT32, {max_seq_len_, batch_size, beam_width_}, parent_ids_buf_},
                output_tensors->at(2),
                Tensor{MEMORY_GPU, data_type, self_k_cache_size, key_caches_[tgt_cache_id]},
                Tensor{MEMORY_GPU, data_type, self_v_cache_size, value_caches_[tgt_cache_id]}};
        }
        else {

            dynamic_decode_input_tensors = new std::vector<Tensor>{
                Tensor{MEMORY_GPU, data_type, {batch_size, beam_width_, vocab_size_padded_}, logits_buf_},
                Tensor{MEMORY_GPU, data_type, {vocab_size_padded_}, padded_embedding_bias_ptr_},
                Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step},
                Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_length},
                Tensor{MEMORY_GPU, TYPE_INT32, {batch_size, beam_width_}, nullptr},
                Tensor{MEMORY_CPU, TYPE_INT32, {1}, &tmp_ite}};

            dynamic_decode_output_tensors = new std::vector<Tensor>{
                Tensor{MEMORY_GPU, TYPE_INT32, {max_seq_len_, batch_size, beam_width_}, output_ids_buf_},
                Tensor{MEMORY_GPU, TYPE_BOOL, {batch_size * beam_width_}, finished_buf_},
                output_tensors->at(2),
                Tensor{MEMORY_GPU, TYPE_FP32, {max_seq_len_, batch_size, beam_width_}, nullptr}};
        }

        dynamic_decode_->forward(dynamic_decode_output_tensors, dynamic_decode_input_tensors);

        delete dynamic_decode_input_tensors;
        delete dynamic_decode_output_tensors;
    }

    // minus the sequence length of unfinished sentence by 1 since we start from 1.
    invokeMinusUnfinishedSeqlen((int*)output_tensors->at(2).data, finished_buf_, batch_size * beam_width_, stream_);

    if (beam_width_ > 1) {
        // For beam search, do gather_tree
        // TODO(bhsueh) remove the output of parent_ids
        cudaD2Dcpy((int*)output_tensors->at(1).data,
                   parent_ids_buf_ + batch_size * beam_width_,
                   batch_size * beam_width_ * (max_seq_len_ - 1));

        invokeGatherTree((int*)output_tensors->at(0).data,
                         (int*)output_tensors->at(2).data,
                         max_seq_len_ - 1,
                         batch_size,
                         beam_width_,
                         output_ids_buf_ + batch_size * beam_width_,
                         parent_ids_buf_ + batch_size * beam_width_,
                         end_id_,
                         stream_);
    }
    else {
        // For sampling, only copy the results to output_tensor
        cudaD2Dcpy((int*)output_tensors->at(0).data,
                   output_ids_buf_ + batch_size * beam_width_,
                   batch_size * beam_width_ * (max_seq_len_ - 1));
    }
}

template class Decoding<float>;
template class Decoding<half>;

}  // namespace fastertransformer
