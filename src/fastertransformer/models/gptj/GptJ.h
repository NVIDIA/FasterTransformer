/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/layers/DynamicDecodeLayer.h"
#include "src/fastertransformer/models/gptj/GptJContextDecoder.h"
#include "src/fastertransformer/models/gptj/GptJDecoder.h"
#include "src/fastertransformer/models/gptj/GptJWeight.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"

namespace fastertransformer {

template<typename T>
class GptJ: public BaseLayer {
private:
    // meta data
    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t num_layer_;
    size_t vocab_size_;
    size_t rotary_embedding_dim_;

    int start_id_;
    int end_id_;
    size_t hidden_units_;

    size_t local_head_num_;
    NcclParam tensor_para_;
    NcclParam pipeline_para_;

    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm_;
    int enable_custom_all_reduce_;

    size_t vocab_size_padded_;
    const bool is_context_qk_buf_float_ = true;

    GptJDecoder<T>* gpt_decoder_;
    GptJContextDecoder<T>* gpt_context_decoder_;
    DynamicDecodeLayer<float>* dynamic_decode_layer_;

    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size, size_t beam_width, size_t max_seq_len, size_t max_input_len);
    void freeBuffer() override;

    void initialize();

protected:
    T* padded_embedding_kernel_;
    T* padded_embedding_bias_;
    const T* padded_embedding_kernel_ptr_;
    const T* padded_embedding_bias_ptr_;

    T* input_attention_mask_;

    T* decoder_input_buf_;
    T* decoder_output_buf_;
    T* normed_decoder_output_buf_;

    float* logits_buf_;
    float* nccl_logits_buf_;
    float* cum_log_probs_;

    bool* finished_buf_;
    bool* h_finished_buf_;

    T* key_cache_;
    T* value_cache_;
    int* cache_indirections_[2] = {nullptr, nullptr};

    int* tiled_input_ids_buf_;
    int* tiled_input_lengths_buf_;
    int* transposed_output_ids_buf_;
    int* output_ids_buf_;
    int* parent_ids_buf_;
    int* start_ids_buf_;
    int* end_ids_buf_;

    T* context_decoder_input_buf_;
    T* context_decoder_output_buf_;
    float* output_log_probs_buf_;

public:
    GptJ(size_t max_batch_size,
         size_t max_seq_len,
         size_t max_input_len,
         size_t beam_width,
         size_t head_num,
         size_t size_per_head,
         size_t inter_size,
         size_t num_layer,
         size_t vocab_size,
         size_t rotary_embedding_dim,
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
         std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm = nullptr,
         int enable_custom_all_reduce = 0);

    GptJ(size_t max_batch_size,
         size_t max_seq_len,
         size_t max_input_len,
         size_t beam_width,
         size_t head_num,
         size_t size_per_head,
         size_t inter_size,
         size_t num_layer,
         size_t vocab_size,
         size_t rotary_embedding_dim,
         int start_id,
         int end_id,
         float beam_search_diversity_rate,
         size_t top_k,
         float top_p,
         unsigned long long random_seed,
         float temperature,
         float len_penalty,
         float repetition_penalty,
         NcclParam tensor_para,
         NcclParam pipeline_para,
         cudaStream_t stream,
         cublasMMWrapper* cublas_wrapper,
         IAllocator* allocator,
         bool is_free_buffer_after_forward,
         cudaDeviceProp* cuda_device_prop = nullptr,
         std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm = nullptr,
         int enable_custom_all_reduce = 0);

    GptJ(GptJ<T> const& GptJ);

    ~GptJ();

    void forward(std::vector<Tensor>* output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 const GptJWeight<T>* gpt_weights);

    void forward(std::unordered_map<std::string, Tensor>* output_tensors,
                 const std::unordered_map<std::string, Tensor>* input_tensors,
                 const GptJWeight<T>* gpt_weights);

    size_t getPipelineParallelRank();
    size_t getPipelineParallelSize();
    size_t getTensorParallelRank();
    size_t getTensorParallelSize();
    bool* getFinishBuffer();
};

}  // namespace fastertransformer
