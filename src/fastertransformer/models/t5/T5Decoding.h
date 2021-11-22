/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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
#include "src/fastertransformer/layers/DynamicDecodeBaseLayer.h"
#include "src/fastertransformer/models/t5/T5Decoder.h"
#include "src/fastertransformer/models/t5/T5DecodingWeight.h"

namespace fastertransformer {

template<typename T>
class T5Decoding: public BaseLayer {
private:
    // buffer handling
    size_t max_batch_size_ = 0;
    size_t max_seq_len_ = 0;
    size_t mem_max_seq_len_ = 0;
    // meta data
    const size_t beam_width_;
    const size_t head_num_;
    const size_t size_per_head_;
    const size_t inter_size_;
    const size_t d_model_;
    const size_t num_layer_;
    const size_t vocab_size_;
    const size_t num_bucket_;
    const size_t max_distance_;

    const int start_id_;
    const int end_id_;
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
    DynamicDecodeBaseLayer* dynamic_decode_;

    void allocateBuffer() override;
    void freeBuffer() override;
    bool isValidBatchSize(size_t batch_size);
    bool isValidSeqLen(size_t seq_len);
    bool isValidMemSeqLen(size_t seq_len);

    void initialize();

    NcclParam tensor_para_;
    NcclParam pipeline_para_;

protected:
    T* padded_embedding_kernel_;
    const T* padded_embedding_kernel_ptr_;
    T* relative_attention_bias_;

    T* decoder_input_buf_;
    T* decoder_output_buf_;
    T* normed_decoder_output_buf_;
    T* logits_buf_;
    T* nccl_logits_buf_;
    float* cum_log_probs_;
    bool* finished_buf_;
    bool* h_finished_buf_;

    T* key_caches_[2];    // ping-pong buffer
    T* value_caches_[2];  // ping-pong buffer
    T* key_mem_caches_;
    T* value_mem_caches_;

    int* output_ids_buf_;
    int* parent_ids_buf_;

    T* tiled_encoder_output_;
    int* tiled_encoder_sequence_length_;

    const T* encoder_output_ptr_;
    const int* encoder_sequence_length_ptr_;

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
               NcclParam pipeline_para);

    T5Decoding(T5Decoding<T> const& T5Decoding);

    ~T5Decoding();

    void forward(std::vector<Tensor>* output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 const T5DecodingWeight<T>* Decoding_weights);

    inline size_t getMaxSeqLen()
    {
        return max_seq_len_ - 1;
    }
};

}  // namespace fastertransformer
