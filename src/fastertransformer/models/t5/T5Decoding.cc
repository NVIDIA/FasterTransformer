/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
#include "src/fastertransformer/kernels/beam_search_topk_kernels.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/layers/beam_search_layers/BaseBeamSearchLayer.h"

namespace fastertransformer {

template<typename T>
void T5Decoding<T>::initialize()
{
    decoder_ = new T5Decoder<T>(0,  // max_batch_size_ * beam_width_,
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
                                pipeline_para_,
                                activation_type_,
                                q_scaling_,
                                custom_all_reduce_comm_,
                                enable_custom_all_reduce_);

    dynamic_decode_layer_ = new DynamicDecodeLayer<T>(vocab_size_,
                                                      vocab_size_padded_,
                                                      0,  // end_id, deprecated
                                                      stream_,
                                                      cublas_wrapper_,
                                                      allocator_,
                                                      is_free_buffer_after_forward_,
                                                      cuda_device_prop_);
}

template<typename T>
void T5Decoding<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void T5Decoding<T>::allocateBuffer(
    size_t batch_size, size_t beam_width, size_t max_seq_len, size_t max_mem_seq_len, size_t encoder_d_model)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // Note: To put the start_ids, we use max_seq_len + 1 for ouptut_ids_buf_
    // And to consistent to the output_ids_buf_, some related buffers are also
    // use max_seq_len + 1, but not max_seq_len.
    // This only affects the buffer size, not affect the performance.

    const size_t batchxbeam = batch_size * beam_width;
    const size_t self_cache_size = (num_layer_ / pipeline_para_.world_size_) * batchxbeam * (max_seq_len + 1)
                                   * (hidden_units_ / tensor_para_.world_size_);
    const size_t mem_cache_size = (num_layer_ / pipeline_para_.world_size_) * batchxbeam * max_mem_seq_len
                                  * (hidden_units_ / tensor_para_.world_size_);

    if (vocab_size_ != vocab_size_padded_) {
        padded_embedding_kernel_ =
            (T*)(allocator_->reMalloc(padded_embedding_kernel_, sizeof(T) * d_model_ * vocab_size_padded_, true));
        padded_embedding_kernel_ptr_ = padded_embedding_kernel_;

        padded_post_decoder_embedding_bias_ =
            (T*)(allocator_->reMalloc(padded_post_decoder_embedding_bias_, sizeof(T) * vocab_size_padded_, true));
        padded_post_decoder_embedding_bias_ptr_ = padded_post_decoder_embedding_bias_;
    }
    relative_attention_bias_ = (T*)(allocator_->reMalloc(
        relative_attention_bias_, sizeof(T) * head_num_ * (max_seq_len + 1) * (max_seq_len + 1), false));

    decoder_input_buf_ = (T*)(allocator_->reMalloc(decoder_input_buf_, sizeof(T) * batchxbeam * d_model_, false));
    decoder_output_buf_ = (T*)(allocator_->reMalloc(decoder_output_buf_, sizeof(T) * batchxbeam * d_model_, false));
    normed_decoder_output_buf_ =
        (T*)(allocator_->reMalloc(normed_decoder_output_buf_, sizeof(T) * batchxbeam * d_model_, false));
    logits_buf_ = (T*)(allocator_->reMalloc(logits_buf_, sizeof(T) * batchxbeam * vocab_size_padded_, false));
    nccl_logits_buf_ = (T*)(allocator_->reMalloc(nccl_logits_buf_, sizeof(T) * batchxbeam * vocab_size_padded_, false));
    cum_log_probs_ = (float*)(allocator_->reMalloc(cum_log_probs_, sizeof(float) * batchxbeam, false));
    finished_buf_ = (bool*)(allocator_->reMalloc(finished_buf_, sizeof(bool) * batchxbeam, false));
    h_finished_buf_ = (bool*)realloc(h_finished_buf_, sizeof(bool) * batchxbeam);

    key_cache_ = (T*)(allocator_->reMalloc(key_cache_, sizeof(T) * (2 * self_cache_size + 2 * mem_cache_size), false));
    value_cache_ = key_cache_ + self_cache_size;
    key_mem_cache_ = value_cache_ + self_cache_size;
    value_mem_cache_ = key_mem_cache_ + mem_cache_size;
    if (beam_width > 1) {
        cache_indirections_[0] = (int*)(allocator_->reMalloc(
            cache_indirections_[0], sizeof(int) * batchxbeam * (max_seq_len + 1) * 2, true));
        cache_indirections_[1] = cache_indirections_[0] + batchxbeam * (max_seq_len + 1);
    }
    tiled_encoder_output_ = (T*)(allocator_->reMalloc(
        tiled_encoder_output_, sizeof(T) * batchxbeam * max_mem_seq_len * encoder_d_model, false));
    tiled_encoder_sequence_length_ =
        (int*)(allocator_->reMalloc(tiled_encoder_sequence_length_, sizeof(int) * batchxbeam, false));

    start_ids_buf_ = (int*)(allocator_->reMalloc(start_ids_buf_, sizeof(int) * batch_size, false));
    end_ids_buf_ = (int*)(allocator_->reMalloc(end_ids_buf_, sizeof(int) * batch_size, false));

    output_ids_buf_ =
        (int*)(allocator_->reMalloc(output_ids_buf_, sizeof(int) * batchxbeam * (max_seq_len + 1), false));
    parent_ids_buf_ =
        (int*)(allocator_->reMalloc(parent_ids_buf_, sizeof(int) * batchxbeam * (max_seq_len + 1), false));
    output_ids_transpose_buf_ =
        (int*)(allocator_->reMalloc(output_ids_transpose_buf_, sizeof(int) * batchxbeam * max_seq_len, false));
    output_log_probs_buf_ =
        (float*)(allocator_->reMalloc(output_log_probs_buf_, sizeof(float) * batchxbeam * max_seq_len, false));
    is_allocate_buffer_ = true;
}

template<typename T>
void T5Decoding<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        if (vocab_size_ != vocab_size_padded_) {
            padded_embedding_kernel_ptr_ = nullptr;
            allocator_->free(padded_embedding_kernel_);

            padded_post_decoder_embedding_bias_ptr_ = nullptr;
            allocator_->free(padded_post_decoder_embedding_bias_);
        }

        allocator_->free(relative_attention_bias_);

        allocator_->free(decoder_input_buf_);
        allocator_->free(decoder_output_buf_);
        allocator_->free(normed_decoder_output_buf_);
        allocator_->free(logits_buf_);
        allocator_->free(nccl_logits_buf_);
        allocator_->free(cum_log_probs_);
        allocator_->free(finished_buf_);
        free(h_finished_buf_);

        allocator_->free(key_cache_);
        if (cache_indirections_[0] != nullptr) {
            allocator_->free(cache_indirections_[0]);
        }

        allocator_->free(start_ids_buf_);
        allocator_->free(end_ids_buf_);

        allocator_->free(output_ids_buf_);
        allocator_->free(parent_ids_buf_);
        allocator_->free(output_ids_transpose_buf_);
        allocator_->free(output_log_probs_buf_);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
void T5Decoding<T>::setStream(cudaStream_t stream)
{
    decoder_->setStream(stream);
    dynamic_decode_layer_->setStream(stream);
    BaseLayer::setStream(stream);
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
                          float q_scaling,
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
                          NcclParam pipeline_para,
                          ActivationType activation_type,
                          std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                          int enable_custom_all_reduce):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    d_model_(d_model),
    num_layer_(num_layer),
    vocab_size_(vocab_size),
    num_bucket_(num_bucket),
    max_distance_(max_distance),
    q_scaling_(q_scaling),
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
    pipeline_para_(pipeline_para),
    activation_type_(activation_type),
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
T5Decoding<T>::T5Decoding(T5Decoding<T> const& decoding):
    BaseLayer(decoding),
    head_num_(decoding.head_num_),
    size_per_head_(decoding.size_per_head_),
    inter_size_(decoding.inter_size_),
    d_model_(decoding.d_model_),
    num_layer_(decoding.num_layer_),
    vocab_size_(decoding.vocab_size_),
    num_bucket_(decoding.num_bucket_),
    max_distance_(decoding.max_distance_),
    q_scaling_(decoding.q_scaling_),
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
    pipeline_para_(decoding.pipeline_para_),
    activation_type_(decoding.activation_type_),
    custom_all_reduce_comm_(decoding.custom_all_reduce_comm_),
    enable_custom_all_reduce_(decoding.enable_custom_all_reduce_)
{
    initialize();
}

template<typename T>
T5Decoding<T>::~T5Decoding()
{
    delete decoder_;
    delete dynamic_decode_layer_;
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
    //      sequence_length [batch_size, beam], record the number of generated token, except the start token

    // Step is from 1 ~ max_seq_len,
    // When step = k,  we put output ids and caches at step k, and the sequence_length would be k - 1 before
    // complete this step.

    std::unordered_map<std::string, Tensor> input_tensors_map{{"encoder_output", input_tensors->at(0)},
                                                              {"encoder_sequence_length", input_tensors->at(1)}};

    std::unordered_map<std::string, Tensor> output_tensors_map{{"output_ids", output_tensors->at(0)},
                                                               {"sequence_length", output_tensors->at(1)}};
    forward(&output_tensors_map, &input_tensors_map, decoding_weights);
}

template<typename T>
bool T5Decoding<T>::hasDiffRuntimeArgs(const std::unordered_map<std::string, Tensor>* input_tensors)
{
    //      runtime_top_k [1] or [batch_size] on cpu, optional.
    //      runtime_top_p [1] or [batch_size] on cpu, optional
    //      beam_search_diversity_rate [1] or [batch_size] on cpu, optional
    //      temperature [1] or [batch_size] on cpu, optional
    //      len_penalty [1] or [batch_size] on cpu, optional
    //      repetition_penalty [1] or [batch_size] on cpu, optional
    //      random_seed [1] or [batch_size] on cpu, optional

    std::vector<std::string> check_list = {"runtime_top_k",
                                           "runtime_top_p",
                                           "beam_search_diversity_rate",
                                           "temperature",
                                           "len_penalty",
                                           "repetition_penalty",
                                           "random_seed"};

    for (int i = 0; i < check_list.size(); i++) {
        if (input_tensors->count(check_list[i])) {
            auto tensor = input_tensors->at(check_list[i]);
            if (tensor.shape.size() > 1) {
                FT_CHECK(tensor.shape[1] == 1);
                for (int i = 1; i < tensor.shape[0]; i++) {
                    const void* data = tensor.data;
                    switch (tensor.type) {
                        case TYPE_FP32:
                            if (((const float*)data)[0] != ((const float*)data)[i]) {
                                return true;
                            }
                            break;
                        case TYPE_INT32:
                            if (((const int*)data)[0] != ((const int*)data)[i]) {
                                return true;
                            }
                            break;
                        case TYPE_UINT32:
                            if (((const uint*)data)[0] != ((const uint*)data)[i]) {
                                return true;
                            }
                            break;
                        case TYPE_UINT64:
                            if (((const unsigned long long int*)data)[0] != ((const unsigned long long int*)data)[i]) {
                                return true;
                            }
                            break;
                        default:
                            FT_CHECK(false);
                            break;
                    }
                }
            }
        }
    }
    return false;
}

template<typename T>
void T5Decoding<T>::forward(std::unordered_map<std::string, Tensor>* output_tensors,
                            const std::unordered_map<std::string, Tensor>* input_tensors,
                            const T5DecodingWeight<T>* decoding_weights)
{
    // input_tensors:
    //      encoder_output [batch_size, mem_max_seq_len, memory_hidden_dimension]
    //      encoder_sequence_length [batch_size]
    //      stop_words_list [batch_size, 2, stop_words_length], optional
    //      start_id [batch_size] on cpu, optional
    //      end_id [batch_size] on cpu, optional
    //      runtime_top_k [1] or [batch_size] on cpu, optional.
    //      runtime_top_p [1] or [batch_size] on cpu, optional
    //      beam_search_diversity_rate [1] or [batch_size] on cpu, optional
    //      temperature [1] or [batch_size] on cpu, optional
    //      len_penalty [1] or [batch_size] on cpu, optional
    //      repetition_penalty [1] or [batch_size] on cpu, optional
    //      random_seed [1] or [batch_size] on cpu, optional

    // output_tensors:
    //      output_ids [batch_size, beam, max_seq_len]
    //      sequence_length [batch_size, beam], record the number of generated token, except the start token
    //      output_log_probs [batch_size, beam, max_seq_len], optional, must be float*.
    //      cum_log_probs [batch_size, beam], optional, must be float*.

    // Step is from 1 ~ max_seq_len,
    // When step = k,  we put output ids and caches at step k, and the sequence_length would be k - 1 before
    // complete this step.

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() >= 2);
    FT_CHECK(output_tensors->size() >= 2);
    FT_CHECK(input_tensors->at("encoder_output").shape.size() == 3);
    const size_t batch_size = output_tensors->at("output_ids").shape[0];
    const size_t beam_width = output_tensors->at("output_ids").shape[1];
    const size_t max_seq_len = output_tensors->at("output_ids").shape[2];
    const size_t mem_max_seq_len = input_tensors->at("encoder_output").shape[1];
    allocateBuffer(batch_size, beam_width, max_seq_len, mem_max_seq_len, input_tensors->at("encoder_output").shape[2]);

    bool has_diff_runtime_args = hasDiffRuntimeArgs(input_tensors);

    handleOptArg(input_tensors, "start_id", start_ids_buf_, start_id_, batch_size);
    handleOptArg(input_tensors, "end_id", end_ids_buf_, end_id_, batch_size);

    FT_CHECK(input_tensors->at("encoder_output").shape[2] == d_model_);

    const int max_input_length = 1;
    const DataType data_type = getTensorType<T>();
    int* sequence_lengths = (int*)output_tensors->at("sequence_length").data;

    cudaMemset((int*)output_tensors->at("output_ids").data, 0, sizeof(int) * batch_size * beam_width * max_seq_len);
    if (beam_width > 1) {
        cudaMemsetAsync(
            cache_indirections_[0], 0, 2 * sizeof(int) * batch_size * beam_width * (max_seq_len + 1), stream_);
    }

    if (beam_width > 1) {
        invokeTileEncoderResults(tiled_encoder_output_,
                                 tiled_encoder_sequence_length_,
                                 (const T*)(input_tensors->at("encoder_output").data),
                                 (const int*)(input_tensors->at("encoder_sequence_length").data),
                                 batch_size,
                                 beam_width,
                                 mem_max_seq_len,
                                 d_model_,
                                 stream_);
        sync_check_cuda_error();
        encoder_output_ptr_ = tiled_encoder_output_;
        encoder_sequence_length_ptr_ = tiled_encoder_sequence_length_;
    }
    else {
        encoder_output_ptr_ = (const T*)(input_tensors->at("encoder_output").data);
        encoder_sequence_length_ptr_ = (const int*)(input_tensors->at("encoder_sequence_length").data);
    }

    invokeDecodingInitialize(finished_buf_,
                             sequence_lengths,
                             output_ids_buf_,
                             cum_log_probs_,
                             start_ids_buf_,
                             batch_size,
                             beam_width,
                             max_input_length - 1,
                             stream_);
    sync_check_cuda_error();

    invokeBuildRelativeAttentionBias(relative_attention_bias_,
                                     decoding_weights->absolute_or_relative_position_embedding,
                                     head_num_,
                                     (max_seq_len + 1),
                                     num_bucket_,
                                     false,
                                     max_distance_,
                                     decoding_weights->position_embedding_type,
                                     stream_);
    sync_check_cuda_error();

    if (vocab_size_ == vocab_size_padded_) {
        padded_embedding_kernel_ptr_ = decoding_weights->post_decoder_embedding.kernel;
        padded_post_decoder_embedding_bias_ptr_ = decoding_weights->post_decoder_embedding.bias;
    }
    else {
        invokePaddingEmbeddingKernel(padded_embedding_kernel_,
                                     decoding_weights->post_decoder_embedding.kernel,
                                     d_model_,
                                     vocab_size_,
                                     vocab_size_padded_,
                                     stream_);
        sync_check_cuda_error();
        FT_CHECK(decoding_weights->post_decoder_embedding.bias != nullptr);
        cudaD2Dcpy(padded_post_decoder_embedding_bias_, decoding_weights->post_decoder_embedding.bias, vocab_size_);
    }

    const std::vector<size_t> self_k_cache_shape = {num_layer_ / pipeline_para_.world_size_,
                                                    batch_size * beam_width,
                                                    head_num_ / tensor_para_.world_size_,
                                                    size_per_head_ / (16 / sizeof(T)),
                                                    max_seq_len + 1,
                                                    16 / sizeof(T)};
    const std::vector<size_t> self_v_cache_shape = {num_layer_ / pipeline_para_.world_size_,
                                                    batch_size * beam_width,
                                                    head_num_ / tensor_para_.world_size_,
                                                    (size_t)(max_seq_len + 1),
                                                    size_per_head_};
    const std::vector<size_t> mem_cache_shape = {num_layer_ / pipeline_para_.world_size_,
                                                 batch_size * beam_width,
                                                 mem_max_seq_len,
                                                 head_num_ / tensor_para_.world_size_ * size_per_head_};

    const size_t local_batch_size = getLocalBatchSize(batch_size, 1, pipeline_para_.world_size_);
    FT_CHECK(batch_size % local_batch_size == 0);
    const size_t iteration_num = batch_size / local_batch_size;
    for (int step = max_input_length; step <= (int)max_seq_len; step++) {
        const int src_indir_idx = beam_width > 1 ? (step - 1) & 0x1 : 0;
        const int tgt_indir_idx = 1 - src_indir_idx;

        for (uint ite = 0; ite < iteration_num; ++ite) {
            const int id_offset = ite * local_batch_size * beam_width;
            const int d_model_offset = id_offset * d_model_;
            const int vocab_size_units_offset = id_offset * vocab_size_padded_;

            if (pipeline_para_.rank_ == 0) {
                invokeEmbeddingLookupPosEncoding(decoder_input_buf_ + d_model_offset,
                                                 decoding_weights->pre_decoder_embedding_table,
                                                 decoding_weights->position_embedding_type
                                                         == PositionEmbeddingType::relative ?
                                                     (T*)nullptr :
                                                     decoding_weights->absolute_or_relative_position_embedding,
                                                 output_ids_buf_ + id_offset,
                                                 nullptr,
                                                 local_batch_size * beam_width,
                                                 d_model_,
                                                 (T)1.0f,
                                                 step - 1,
                                                 0,
                                                 batch_size * beam_width,
                                                 0,
                                                 stream_);
                sync_check_cuda_error();
            }

            std::vector<Tensor> decoder_input_tensors{
                Tensor{MEMORY_GPU,
                       data_type,
                       {local_batch_size * beam_width, d_model_},
                       decoder_input_buf_ + d_model_offset},
                Tensor{MEMORY_GPU,
                       data_type,
                       {local_batch_size * beam_width,
                        input_tensors->at("encoder_output").shape[1],
                        input_tensors->at("encoder_output").shape[2]},
                       encoder_output_ptr_
                           + id_offset * input_tensors->at("encoder_output").shape[1]
                                 * input_tensors->at("encoder_output").shape[2]},
                Tensor{
                    MEMORY_GPU, TYPE_INT32, {local_batch_size * beam_width}, encoder_sequence_length_ptr_ + id_offset},
                Tensor{MEMORY_GPU, TYPE_BOOL, {local_batch_size * beam_width}, finished_buf_ + id_offset},
                Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step},
                Tensor{MEMORY_GPU, TYPE_INT32, {local_batch_size * beam_width}, sequence_lengths + id_offset},
                Tensor{MEMORY_GPU,
                       data_type,
                       {1, head_num_, max_seq_len + 1, max_seq_len + 1},
                       decoding_weights->position_embedding_type == PositionEmbeddingType::relative ?
                           relative_attention_bias_ :
                           nullptr},
                Tensor{MEMORY_CPU, TYPE_UINT32, {1}, &ite},
                Tensor{MEMORY_GPU,
                       TYPE_INT32,
                       {local_batch_size, beam_width, max_seq_len + 1},
                       beam_width > 1 ? cache_indirections_[src_indir_idx] + id_offset * (max_seq_len + 1) : nullptr}};

            std::vector<Tensor> decoder_output_tensors{
                Tensor{MEMORY_GPU,
                       data_type,
                       {local_batch_size * beam_width, d_model_},
                       decoder_output_buf_ + d_model_offset},
                Tensor{MEMORY_GPU, data_type, self_k_cache_shape, key_cache_},
                Tensor{MEMORY_GPU, data_type, self_v_cache_shape, value_cache_},
                Tensor{MEMORY_GPU, data_type, mem_cache_shape, key_mem_cache_},
                Tensor{MEMORY_GPU, data_type, mem_cache_shape, value_mem_cache_}};
            decoder_->forward(
                &decoder_output_tensors, &decoder_input_tensors, &decoding_weights->decoder_layer_weights);

            bool t5_with_bias = decoding_weights->t5_with_bias;

            if (pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) {
                invokeGeneralT5LayerNorm(normed_decoder_output_buf_ + d_model_offset,
                                         decoder_output_buf_ + d_model_offset,
                                         decoding_weights->post_decoder_layernorm.gamma,
                                         decoding_weights->post_decoder_layernorm.beta,
                                         local_batch_size * beam_width,
                                         d_model_,
                                         stream_);
                sync_check_cuda_error();

                if (tensor_para_.world_size_ == 1) {
                    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                          CUBLAS_OP_N,
                                          vocab_size_padded_,  // n
                                          local_batch_size * beam_width,
                                          d_model_,  // k
                                          padded_embedding_kernel_ptr_,
                                          d_model_,  // k
                                          normed_decoder_output_buf_ + d_model_offset,
                                          d_model_,  // k
                                          logits_buf_ + vocab_size_units_offset,
                                          vocab_size_padded_ /* n */,
                                          t5_with_bias ? 1.0f : 1.0f / sqrt(d_model_),
                                          0.0f);
                }
                else {
                    const int local_vocab_size = vocab_size_padded_ / tensor_para_.world_size_;
                    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                          CUBLAS_OP_N,
                                          local_vocab_size,  // n
                                          local_batch_size * beam_width,
                                          d_model_,  // k
                                          padded_embedding_kernel_ptr_
                                              + tensor_para_.rank_ * local_vocab_size * d_model_,
                                          d_model_,  // k
                                          normed_decoder_output_buf_ + d_model_offset,
                                          d_model_,  // k
                                          nccl_logits_buf_ + vocab_size_units_offset
                                              + tensor_para_.rank_ * local_batch_size * beam_width * local_vocab_size,
                                          local_vocab_size /* n */,
                                          t5_with_bias ? 1.0f : 1.0f / sqrt(d_model_),
                                          0.0f);
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

                if (t5_with_bias) {
                    invokeAddBias(logits_buf_ + vocab_size_units_offset,
                                  padded_post_decoder_embedding_bias_ptr_,
                                  local_batch_size * beam_width,
                                  vocab_size_padded_,
                                  stream_);
                }

                int tmp_local_batch_size = local_batch_size;
                bool is_initialize_random_table = step == 1;
                std::unordered_map<std::string, Tensor> dynamic_decode_input_tensors{
                    {"logits",
                     Tensor{MEMORY_GPU, data_type, {batch_size, beam_width, vocab_size_padded_}, logits_buf_}},
                    {"embedding_bias", Tensor{MEMORY_GPU, data_type, {vocab_size_padded_}, nullptr}},
                    {"step", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step}},
                    {"max_input_length", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_length}},
                    {"input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size, beam_width}, nullptr}},
                    {"ite", Tensor{MEMORY_CPU, TYPE_UINT32, {1}, &ite}},
                    {"end_id", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size}, end_ids_buf_}},
                    {"has_diff_runtime_args", Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &has_diff_runtime_args}},
                    {"src_key_cache", Tensor{MEMORY_GPU, data_type, self_k_cache_shape, key_cache_}},
                    {"src_value_cache", Tensor{MEMORY_GPU, data_type, self_v_cache_shape, value_cache_}},
                    {"src_cache_indirection",
                     Tensor{MEMORY_GPU,
                            TYPE_INT32,
                            {local_batch_size, beam_width, (max_seq_len + 1)},
                            cache_indirections_[src_indir_idx] + id_offset * (max_seq_len + 1)}},
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
                     Tensor{MEMORY_GPU, TYPE_INT32, {(max_seq_len + 1), batch_size, beam_width}, output_ids_buf_}},
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
                            {(max_seq_len + 1), batch_size, beam_width},
                            output_tensors->count("output_log_probs") > 0
                                    && output_tensors->at("output_log_probs").data != nullptr ?
                                output_log_probs_buf_ :
                                nullptr}},
                    {"parent_ids",
                     Tensor{MEMORY_GPU, TYPE_INT32, {(max_seq_len + 1), batch_size, beam_width}, parent_ids_buf_}},
                    {"sequence_length", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size * beam_width}, sequence_lengths}},
                    {"tgt_cache_indirection",
                     Tensor{MEMORY_GPU,
                            TYPE_INT32,
                            {local_batch_size, beam_width, (max_seq_len + 1)},
                            cache_indirections_[tgt_indir_idx] + id_offset * (max_seq_len + 1)}}};

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
                                batch_size * beam_width * (max_seq_len + 1),
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
        if (beam_width > 1) {
            // For beam search, do gather_tree
            invokeGatherTree(output_ids_transpose_buf_,
                             (int*)output_tensors->at("sequence_length").data,
                             max_seq_len,
                             batch_size,
                             beam_width,
                             output_ids_buf_ + batch_size * beam_width,
                             parent_ids_buf_ + batch_size * beam_width,
                             end_ids_buf_,
                             stream_);

            // transpose and take output_parent_ids as inter buffer
            invokeTransposeAxis01((int*)output_tensors->at("output_ids").data,
                                  output_ids_transpose_buf_,
                                  max_seq_len,
                                  batch_size * beam_width,
                                  1,
                                  stream_);
        }
        else {
            // For sampling, only transpose the results to output_tensor
            invokeTransposeAxis01((int*)output_tensors->at("output_ids").data,
                                  output_ids_buf_ + batch_size * beam_width,
                                  max_seq_len,
                                  batch_size * beam_width,
                                  1,
                                  stream_);
        }
        if (output_tensors->find("output_log_probs") != output_tensors->end()) {
            invokeTransposeAxis01(output_tensors->at("output_log_probs").getPtr<float>(),
                                  output_log_probs_buf_,
                                  max_seq_len,
                                  batch_size * beam_width,
                                  1,
                                  stream_);
        }

        // Return the cumulative log probability if requested.
        if (output_tensors->count("cum_log_probs") > 0) {
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
                       batch_size * beam_width * max_seq_len,
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
                           batch_size * beam_width * max_seq_len,
                           0,
                           pipeline_para_,
                           stream_);
            }
        }
        else if (pipeline_para_.rank_ == 0) {
            ftNcclRecv(output_tensors->at("output_ids").getPtr<int>(),
                       batch_size * beam_width * max_seq_len,
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
                           batch_size * beam_width * max_seq_len,
                           pipeline_para_.world_size_ - 1,
                           pipeline_para_,
                           stream_);
            }
        }
        NCCLCHECK(ncclGroupEnd());
        check_cuda_error(cudaStreamSynchronize(stream_));
    }

    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
}

template class T5Decoding<float>;
template class T5Decoding<half>;

}  // namespace fastertransformer
