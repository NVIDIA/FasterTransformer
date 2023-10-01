/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/llama/LLaMAContextDecoder.h"
#include "src/fastertransformer/models/llama/LLaMAWeight.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"

namespace fastertransformer {

template<typename T>
class LLaMA: public BaseLayer {
private:
    // meta data
    size_t                 head_num_;
    size_t                 size_per_head_;
    size_t                 inter_size_;
    size_t                 num_layer_;
    size_t                 vocab_size_;
    size_t                 rotary_embedding_dim_;
    size_t                 random_seed_;
    size_t                 max_seq_len_;
    size_t                 hidden_units_;
    size_t                 rank_;
    size_t                 world_size_;
    static constexpr float layernorm_eps_ = 1e-6f;
    AttentionType          attention_type_;
    const bool             is_context_qk_buf_float_ = (std::getenv("CONTEXT_ATTENTION_BMM1_HALF_ACCUM") == nullptr
                                           || std::string(std::getenv("CONTEXT_ATTENTION_BMM1_HALF_ACCUM")) != "ON");

    LLaMAContextDecoder<T>* llama_context_decoder_;

    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size, size_t seq_len, size_t attn_len, int is_context);
    void freeBuffer() override;

    void initialize();

protected:
    int*   padding_offset_             = nullptr;
    int*   cu_seqlens_                 = nullptr;
    T*     input_attention_mask_       = nullptr;
    T*     key_cache_                  = nullptr;
    T*     value_cache_                = nullptr;
    T*     context_output_buf_         = nullptr;
    T*     normed_decoder_output_buf_  = nullptr;
    float* logits_buf_                 = nullptr;
    float* log_likelihood_buf_         = nullptr;
    T*     context_decoder_input_buf_  = nullptr;
    T*     context_decoder_output_buf_ = nullptr;

    void sendTensorsToFirstPipelineNode(std::unordered_map<std::string, Tensor>*       output_tensors,
                                        const std::unordered_map<std::string, Tensor>* input_tensors);

public:
    LLaMA(size_t           head_num,
          size_t           size_per_head,
          size_t           inter_size,
          size_t           num_layer,
          size_t           vocab_size,
          size_t           rotary_embedding_dim,
          size_t           random_seed,
          size_t           max_seq_len,
          size_t           rank,
          size_t           world_size,
          cudaStream_t     stream,
          cublasMMWrapper* cublas_wrapper,
          IAllocator*      allocator,
          bool             is_free_buffer_after_forward,
          cudaDeviceProp*  cuda_device_prop = nullptr,
          AttentionType    attention_type   = AttentionType::UNFUSED_MHA);

    LLaMA(LLaMA<T> const& LLaMA);

    ~LLaMA();

    void forward(std::vector<Tensor>*       output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 const LLaMAWeight<T>*      llama_weights);

    void forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                 const std::unordered_map<std::string, Tensor>* input_tensors,
                 const LLaMAWeight<T>*                          llama_weights);

};

}  // namespace fastertransformer
