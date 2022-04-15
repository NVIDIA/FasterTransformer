/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <string>
#include <unordered_map>

#include "src/fastertransformer/kernels/beam_search_topk_kernels.h"
#include "src/fastertransformer/layers/beam_search_layers/BaseBeamSearchLayer.h"
#include "src/fastertransformer/models/decoding/Decoding.h"

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

    dynamic_decode_layer_ = new DynamicDecodeLayer<T>(vocab_size_,
                                                      vocab_size_padded_,
                                                      end_id_,
                                                      stream_,
                                                      cublas_wrapper_,
                                                      allocator_,
                                                      is_free_buffer_after_forward_,
                                                      cuda_device_prop_);
}

template<typename T>
void Decoding<T>::allocateBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
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

        start_ids_buf_ = (int*)(allocator_->malloc(sizeof(int) * max_batch_size_, false));
        end_ids_buf_ = (int*)(allocator_->malloc(sizeof(int) * max_batch_size_, false));

        key_cache_ = (T*)(allocator_->malloc(sizeof(T) * self_cache_size, false));
        value_cache_ = (T*)(allocator_->malloc(sizeof(T) * self_cache_size, false));
        if (beam_width_ > 1) {
            cache_indirections_[0] = (int*)(allocator_->malloc(sizeof(int) * batchxbeam * max_seq_len_ * 2, true));
            cache_indirections_[1] = cache_indirections_[0] + batchxbeam * max_seq_len_;
        }
        key_mem_cache_ = (T*)(allocator_->malloc(sizeof(T) * mem_cache_size, false));
        value_mem_cache_ = (T*)(allocator_->malloc(sizeof(T) * mem_cache_size, false));

        padded_pos_embedding_bias_ = (T*)(allocator_->malloc(sizeof(T) * vocab_size_padded_, false));
        output_ids_buf_ = (int*)(allocator_->malloc(sizeof(int) * batchxbeam * max_seq_len_, false));
        parent_ids_buf_ = (int*)(allocator_->malloc(sizeof(int) * batchxbeam * max_seq_len_, false));

        is_allocate_buffer_ = true;
    }
}

template<typename T>
void Decoding<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_ == true) {
        if (vocab_size_ != vocab_size_padded_) {
            padded_embedding_kernel_ptr_ = nullptr;
            padded_embedding_bias_ptr_ = nullptr;
            allocator_->free(padded_embedding_kernel_);
            allocator_->free(padded_embedding_bias_);
        }
        allocator_->free(start_ids_buf_);
        allocator_->free(end_ids_buf_);

        allocator_->free(decoder_input_buf_);
        allocator_->free(decoder_output_buf_);
        allocator_->free(normed_decoder_output_buf_);
        allocator_->free(logits_buf_);
        allocator_->free(cum_log_probs_);
        allocator_->free(finished_buf_);
        delete[] h_finished_buf_;

        allocator_->free(key_cache_);
        allocator_->free(value_cache_);
        if (cache_indirections_[0] != nullptr) {
            allocator_->free(cache_indirections_[0]);
        }
        allocator_->free(key_mem_cache_);
        allocator_->free(value_mem_cache_);

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
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    delete decoder_;
    delete dynamic_decode_layer_;
    freeBuffer();
}

template<typename T>
void Decoding<T>::forward(std::vector<Tensor>* output_tensors,
                          const std::vector<Tensor>* input_tensors,
                          const DecodingWeight<T>* decoding_weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
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

    deviceFill(start_ids_buf_, batch_size, start_id_);
    deviceFill(end_ids_buf_, batch_size, end_id_);

    if (beam_width_ > 1) {
        cudaMemsetAsync(cache_indirections_[0], 0, 2 * sizeof(int) * batch_size * beam_width_ * max_seq_len_, stream_);
    }

    invokeDecodingInitialize(finished_buf_,
                             (int*)output_tensors->at(2).data,
                             output_ids_buf_,
                             cum_log_probs_,
                             start_ids_buf_,
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
        const int ite = 0;
        const int local_batch_size = batch_size;
        const int id_offset = ite * local_batch_size * beam_width_;

        cudaD2Hcpy(h_finished_buf_, finished_buf_, batch_size * beam_width_);
        uint sum = 0;
        for (uint i = 0; i < batch_size * beam_width_; i++) {
            sum += (int)h_finished_buf_[i];
        }
        if (sum == batch_size * beam_width_) {
            break;
        }

        const int src_indir_idx = (step - 1) % 2;
        const int tgt_indir_idx = 1 - src_indir_idx;

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
            output_tensors->at(2),
            Tensor{MEMORY_GPU,
                   TYPE_INT32,
                   {(size_t)local_batch_size, beam_width_, max_seq_len_},
                   beam_width_ > 1 ? cache_indirections_[src_indir_idx] + id_offset * max_seq_len_ : nullptr}};

        std::vector<Tensor> decoder_output_tensors{
            Tensor{MEMORY_GPU, data_type, {batch_size * beam_width_, hidden_units_}, decoder_output_buf_},
            Tensor{MEMORY_GPU, data_type, self_k_cache_size, key_cache_},
            Tensor{MEMORY_GPU, data_type, self_v_cache_size, value_cache_},
            Tensor{MEMORY_GPU,
                   data_type,
                   {num_layer_, batch_size * beam_width_, mem_max_seq_len, hidden_units_},
                   key_mem_cache_},
            Tensor{MEMORY_GPU,
                   data_type,
                   {num_layer_, batch_size * beam_width_, mem_max_seq_len, hidden_units_},
                   value_mem_cache_}};
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

        const int tmp_ite = 0;
        const int tmp_local_batch_size = batch_size;
        const bool has_diff_runtime_args = false;
        const int runtime_top_k = (int)top_k_;
        const float runtime_top_p = (float)top_p_;
        const unsigned long long int random_seed = 0;
        const bool is_initialize_random_table = step == 1;

        std::unordered_map<std::string, Tensor> dynamic_decode_input_tensors{
            {"logits", Tensor{MEMORY_GPU, data_type, {batch_size, beam_width_, vocab_size_padded_}, logits_buf_}},
            {"embedding_bias", Tensor{MEMORY_GPU, data_type, {vocab_size_padded_}, padded_embedding_bias_ptr_}},
            {"end_id", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size}, end_ids_buf_}},
            {"step", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step}},
            {"max_input_length", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_length}},
            {"input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size, beam_width_}, nullptr}},
            {"ite", Tensor{MEMORY_CPU, TYPE_UINT32, {1}, &tmp_ite}},
            {"has_diff_runtime_args", Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &has_diff_runtime_args}},
            {"src_key_cache", Tensor{MEMORY_GPU, data_type, self_k_cache_size, key_cache_}},
            {"src_value_cache", Tensor{MEMORY_GPU, data_type, self_v_cache_size, value_cache_}},
            {"src_cache_indirection",
             Tensor{
                 MEMORY_GPU, TYPE_INT32, {batch_size, beam_width_, max_seq_len_}, cache_indirections_[src_indir_idx]}},
            {"local_batch_size", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &tmp_local_batch_size}},
            {"random_seed", Tensor{MEMORY_CPU, TYPE_UINT64, {1}, &random_seed}},
            {"beam_search_diversity_rate", Tensor{MEMORY_CPU, TYPE_FP32, {1}, &beam_search_diversity_rate_}},
            {"temperature", Tensor{MEMORY_CPU, TYPE_FP32, {1}, &temperature_}},
            {"len_penalty", Tensor{MEMORY_CPU, TYPE_FP32, {1}, &len_penalty_}},
            {"repetition_penalty", Tensor{MEMORY_CPU, TYPE_FP32, {1}, &repetition_penalty_}},
            {"runtime_top_k", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &runtime_top_k}},
            {"runtime_top_p", Tensor{MEMORY_CPU, TYPE_FP32, {1}, &runtime_top_p}},
            {"is_initialize_random_table", Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &is_initialize_random_table}}};

        // TODO(bhsueh) Need to modify the forward function to use unordered_map
        // for (auto t = input_tensors->begin(); t != input_tensors->end(); ++t) {
        //     dynamic_decode_input_tensors.insert(*t);
        // }

        // common outputs
        std::unordered_map<std::string, Tensor> dynamic_decode_output_tensors{
            {"output_ids", Tensor{MEMORY_GPU, TYPE_INT32, {max_seq_len_, batch_size, beam_width_}, output_ids_buf_}},
            {"finished", Tensor{MEMORY_GPU, TYPE_BOOL, {batch_size * beam_width_}, finished_buf_}},
            {"cum_log_probs", Tensor{MEMORY_GPU, TYPE_FP32, {batch_size * beam_width_}, cum_log_probs_}},
            {"parent_ids", Tensor{MEMORY_GPU, TYPE_INT32, {max_seq_len_, batch_size, beam_width_}, parent_ids_buf_}},
            {"sequence_length", output_tensors->at(2)},
            {"tgt_key_cache", Tensor{MEMORY_GPU, data_type, self_k_cache_size, key_cache_}},
            {"tgt_value_cache", Tensor{MEMORY_GPU, data_type, self_v_cache_size, value_cache_}},
            {"tgt_cache_indirection",
             Tensor{
                 MEMORY_GPU, TYPE_INT32, {batch_size, beam_width_, max_seq_len_}, cache_indirections_[tgt_indir_idx]}}};

        // TODO(bhsueh) Need to modify the forward function to use unordered_map
        // for (auto t = output_tensors->begin(); t != output_tensors->end(); ++t) {
        //     dynamic_decode_output_tensors.insert(*t);
        // }

        dynamic_decode_layer_->forward(&dynamic_decode_output_tensors, &dynamic_decode_input_tensors);
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
                         end_ids_buf_,
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
