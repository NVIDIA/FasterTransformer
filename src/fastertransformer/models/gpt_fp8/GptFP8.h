/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "src/fastertransformer/models/gpt_fp8/GptFP8ContextDecoder.h"
#include "src/fastertransformer/models/gpt_fp8/GptFP8Decoder.h"
#include "src/fastertransformer/models/gpt_fp8/GptFP8Weight.h"

namespace fastertransformer {

template<typename T1, typename T2>
class GptFP8: public BaseLayer {
private:
    // meta data
    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t num_layer_;
    size_t vocab_size_;

    int    start_id_;
    int    end_id_;
    size_t hidden_units_;

    size_t    local_head_num_;
    NcclParam tensor_para_;
    NcclParam pipeline_para_;

    const bool is_context_qk_buf_float_ = true;
    size_t     vocab_size_padded_;
    const int  int8_mode_ = 0;

    GptFP8Decoder<T1, T2>*        gpt_decoder_;
    GptFP8ContextDecoder<T1, T2>* gpt_context_decoder_;
    DynamicDecodeLayer<float>*    dynamic_decode_layer_;

    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size, size_t beam_width, size_t max_seq_len, size_t max_input_len);
    void freeBuffer() override;

    void initialize();

protected:
    T2*       padded_embedding_kernel_;
    const T2* padded_embedding_kernel_ptr_;

    T1* input_attention_mask_;

    T2*    decoder_input_buf_;
    T2*    decoder_output_buf_;
    T2*    normed_decoder_output_buf_;
    float* logits_buf_;
    float* nccl_logits_buf_;
    float* cum_log_probs_;
    bool*  finished_buf_;
    bool*  h_finished_buf_;

#ifdef FP8_MHA
    T1* key_cache_;
    T1* value_cache_;
#else
    T2* key_cache_;
    T2* value_cache_;
#endif
    int* cache_indirections_[2] = {nullptr, nullptr};

    int* tiled_input_ids_buf_;
    int* tiled_input_lengths_buf_;

    int* start_ids_buf_;
    int* end_ids_buf_;

    int*  transposed_output_ids_buf_;
    int*  output_ids_buf_;
    int*  parent_ids_buf_;
    int*  tiled_total_padding_count_;
    bool* masked_tokens_;

    T2* context_decoder_input_buf_;
    T2* context_decoder_output_buf_;

    float layernorm_eps_ = 1e-6f;

public:
    GptFP8(size_t           beam_width,
           size_t           head_num,
           size_t           size_per_head,
           size_t           inter_size,
           size_t           num_layer,
           size_t           vocab_size,
           int              start_id,
           int              end_id,
           NcclParam        tensor_para,
           NcclParam        pipeline_para,
           cudaStream_t     stream,
           cublasMMWrapper* cublas_wrapper,
           IAllocator*      allocator,
           bool             is_free_buffer_after_forward,
           cudaDeviceProp*  cuda_device_prop = nullptr,
           bool             sparse           = false);

    GptFP8(GptFP8<T1, T2> const& gpt);

    ~GptFP8();

    void forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                 const std::unordered_map<std::string, Tensor>* input_tensors,
                 const GptFP8Weight<T1, T2>*                    gpt_weights);

    size_t getPipelineParallelRank();
    size_t getPipelineParallelSize();
    size_t getTensorParallelRank();
    size_t getTensorParallelSize();
    bool*  getFinishBuffer();
};

}  // namespace fastertransformer
