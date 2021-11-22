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

#pragma once

#include <cstddef>
#include <vector>

#include "src/fastertransformer/layers/DynamicDecodeBaseLayer.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptContextDecoder.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoder.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptWeight.h"

namespace fastertransformer {

template<typename T>
class ParallelGpt: public BaseLayer {
private:
    // buffer handling
    size_t max_batch_size_ = 0;
    size_t max_seq_len_ = 0;
    size_t max_input_len_ = 0;

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
    unsigned long long random_seed_;

    float temperature_;
    float len_penalty_;
    float repetition_penalty_;

    size_t tensor_para_size_;
    size_t tensor_para_rank_;
    ncclComm_t tensor_para_comm_;
    size_t local_head_num_;
    size_t pipeline_para_size_;
    size_t pipeline_para_rank_;
    ncclComm_t pipeline_para_comm_;

    const bool is_context_qk_buf_float_ = true;
    size_t vocab_size_padded_;
    const int int8_mode_ = 0;

    ParallelGptDecoder<T>* gpt_decoder_;
    DynamicDecodeBaseLayer* dynamic_decode_;
    ParallelGptContextDecoder<T>* gpt_context_decoder_;

    void allocateBuffer() override;
    void freeBuffer() override;
    bool isValidBatchSize(size_t batch_size);
    bool isValidSeqLen(size_t seq_len);
    bool isValidInputSeqLen(size_t seq_len);

    void initialize();

protected:
    T* padded_embedding_kernel_;
    const T* padded_embedding_kernel_ptr_;

    T* input_attention_mask_;

    T* decoder_input_buf_;
    T* decoder_output_buf_;
    T* normed_decoder_output_buf_;
    float* logits_buf_;
    float* nccl_logits_buf_;
    float* cum_log_probs_;
    bool* finished_buf_;
    bool* h_finished_buf_;

    T* key_caches_[2];    // ping-pong buffer
    T* value_caches_[2];  // ping-pong buffer

    T* padded_pos_embedding_bias_;

    int* output_ids_buf_;
    int* parent_ids_buf_;
    int* input_length_buf_;

    T* context_decoder_input_buf_;
    T* context_decoder_output_buf_;

public:
    ParallelGpt(size_t max_batch_size,
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
                cudaDeviceProp* cuda_device_prop = nullptr,
                bool sparse = false,
                int int8_mode = 0);

    ParallelGpt(ParallelGpt<T> const& gpt);

    ~ParallelGpt();

    void forward(std::vector<Tensor>* output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 const ParallelGptWeight<T>* gpt_weights);

    size_t getPipelineParallelRank();
    size_t getPipelineParallelSize();
    size_t getTensorParallelRank();
    size_t getTensorParallelSize();
    bool* getFinishBuffer();
};

}  // namespace fastertransformer
