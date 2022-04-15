/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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
#include "src/fastertransformer/models/decoder/Decoder.h"
#include "src/fastertransformer/models/decoding/DecodingWeight.h"

namespace fastertransformer {

template<typename T>
class Decoding: public BaseLayer {
private:
    // buffer handling
    size_t max_batch_size_ = 0;
    size_t max_seq_len_ = 0;
    size_t mem_max_seq_len_ = 0;
    // meta data
    size_t beam_width_;
    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t num_layer_;
    size_t vocab_size_;

    int start_id_;
    int end_id_;
    float beam_search_diversity_rate_;
    size_t hidden_units_;
    size_t top_k_;
    float top_p_;
    float temperature_;
    float len_penalty_;
    float repetition_penalty_;

    // calculated data
    size_t vocab_size_padded_;

    Decoder<T>* decoder_;
    DynamicDecodeLayer<T>* dynamic_decode_layer_;

    void allocateBuffer() override;
    void freeBuffer() override;
    bool isValidBatchSize(size_t batch_size);
    bool isValidSeqLen(size_t seq_len);
    bool isValidMemSeqLen(size_t seq_len);

    void initialize();

protected:
    T* padded_embedding_kernel_ = nullptr;
    T* padded_embedding_bias_ = nullptr;
    const T* padded_embedding_kernel_ptr_ = nullptr;
    const T* padded_embedding_bias_ptr_ = nullptr;

    T* decoder_input_buf_ = nullptr;
    T* decoder_output_buf_ = nullptr;
    T* normed_decoder_output_buf_ = nullptr;
    T* logits_buf_ = nullptr;
    float* cum_log_probs_ = nullptr;
    bool* finished_buf_ = nullptr;
    bool* h_finished_buf_ = nullptr;

    int* start_ids_buf_;
    int* end_ids_buf_;

    T* key_cache_ = nullptr;
    T* value_cache_ = nullptr;
    T* key_mem_cache_ = nullptr;
    T* value_mem_cache_ = nullptr;
    int* cache_indirections_[2] = {nullptr, nullptr};

    T* padded_pos_embedding_bias_ = nullptr;

    int* output_ids_buf_ = nullptr;
    int* parent_ids_buf_ = nullptr;

public:
    Decoding(size_t max_batch_size,
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
             cudaDeviceProp* cuda_device_prop);

    Decoding(Decoding<T> const& Decoding);

    ~Decoding();

    void forward(std::vector<Tensor>* output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 const DecodingWeight<T>* Decoding_weights);
};

}  // namespace fastertransformer
