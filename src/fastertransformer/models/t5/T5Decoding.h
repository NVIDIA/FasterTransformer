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

#pragma once

#include <cstddef>
#include <vector>

#include "src/fastertransformer/kernels/decoding_kernels.h"
#include "src/fastertransformer/layers/DynamicDecodeLayer.h"
#include "src/fastertransformer/models/t5/T5Decoder.h"
#include "src/fastertransformer/models/t5/T5DecodingWeight.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"

namespace fastertransformer {

template<typename T>
class T5Decoding: public BaseLayer {
private:
    // meta data
    const size_t head_num_;
    const size_t size_per_head_;
    const size_t inter_size_;
    const size_t d_model_;
    const size_t num_layer_;
    const size_t vocab_size_;
    const size_t num_bucket_;
    const size_t max_distance_;
    const ActivationType activation_type_;
    float q_scaling_;

    const int start_id_;
    const int end_id_;

    // TODO(bhsueh) remove
    const float beam_search_diversity_rate_;
    const size_t hidden_units_;
    const size_t top_k_;
    const float top_p_;
    const float temperature_;
    const float len_penalty_;
    const float repetition_penalty_;

    // calculated data
    size_t vocab_size_padded_;

    T5Decoder<T>* decoder_;
    DynamicDecodeLayer<T>* dynamic_decode_layer_;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(
        size_t batch_size, size_t beam_width, size_t max_seq_len, size_t max_mem_seq_len, size_t encoder_d_model);

    void initialize();
    bool hasDiffRuntimeArgs(const std::unordered_map<std::string, Tensor>* input_tensors);

    NcclParam tensor_para_;
    NcclParam pipeline_para_;

    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm_;
    int enable_custom_all_reduce_;

protected:
    T* padded_embedding_kernel_ = nullptr;
    const T* padded_embedding_kernel_ptr_ = nullptr;
    T* padded_post_decoder_embedding_bias_ = nullptr;
    const T* padded_post_decoder_embedding_bias_ptr_ = nullptr;
    T* relative_attention_bias_ = nullptr;

    T* decoder_input_buf_ = nullptr;
    T* decoder_output_buf_ = nullptr;
    T* normed_decoder_output_buf_ = nullptr;
    T* logits_buf_ = nullptr;
    T* nccl_logits_buf_ = nullptr;
    float* cum_log_probs_ = nullptr;
    bool* finished_buf_ = nullptr;
    bool* h_finished_buf_ = nullptr;

    int* start_ids_buf_ = nullptr;
    int* end_ids_buf_ = nullptr;

    T* key_cache_ = nullptr;
    T* value_cache_ = nullptr;
    T* key_mem_cache_ = nullptr;
    T* value_mem_cache_ = nullptr;
    int* cache_indirections_[2] = {nullptr, nullptr};

    int* output_ids_buf_ = nullptr;
    int* parent_ids_buf_ = nullptr;
    int* output_ids_transpose_buf_ = nullptr;
    float* output_log_probs_buf_ = nullptr;

    T* tiled_encoder_output_ = nullptr;
    int* tiled_encoder_sequence_length_ = nullptr;

    const T* encoder_output_ptr_ = nullptr;
    const int* encoder_sequence_length_ptr_ = nullptr;

public:
    T5Decoding(size_t max_batch_size,
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
               ActivationType activation_type = ActivationType::Relu,
               std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm = nullptr,
               int enable_custom_all_reduce = 0);

    T5Decoding(T5Decoding<T> const& T5Decoding);

    ~T5Decoding();

    void forward(std::vector<Tensor>* output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 const T5DecodingWeight<T>* Decoding_weights);

    void forward(std::unordered_map<std::string, Tensor>* output_tensors,
                 const std::unordered_map<std::string, Tensor>* input_tensors,
                 const T5DecodingWeight<T>* Decoding_weights);

    void setStream(cudaStream_t stream) override;
};

}  // namespace fastertransformer
