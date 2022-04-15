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

#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/longformer_kernels.h"
#include "src/fastertransformer/layers/DenseWeight.h"
#include "src/fastertransformer/layers/FfnLayer.h"
#include "src/fastertransformer/layers/attention_layers/LongformerAttentionLayer.h"

namespace fastertransformer {

template<typename T>
struct LongformerLayerWeight {
    DenseWeight<T> query_weights;
    DenseWeight<T> key_weights;
    DenseWeight<T> value_weights;
    DenseWeight<T> global_query_weights;
    DenseWeight<T> global_key_weights;
    DenseWeight<T> global_value_weights;
    DenseWeight<T> attention_output_weights;
    FfnWeight<T> ffn_weights;
    LayerNormWeight<T> attention_output_layernorm_weights;
    LayerNormWeight<T> output_layernorm_weights;
};

template<typename T>
class LongformerEncoder {
private:
    size_t layers_num_;
    size_t in_dim_;
    size_t head_num_;
    size_t size_per_head_;
    size_t hidden_units_;
    size_t intermediate_size_;
    size_t max_batch_size_;
    size_t max_global_token_num_;
    size_t max_seq_len_;

    // internal buffers
    void* cub_storage_;
    int* global_idx_;
    int* global_token_nums_;
    int* seq_idx_;
    T* local_attn_mask_shifted_;
    T* qkv_buffer_;
    T* mha_qkv_buffer_;
    T* mha_out_buffer_;
    T* attn_out_buffer_;
    T* attn_output_buffer_;
    T* intermediate_buffer_;

    GeluFfnLayer<T>* inter_gelu_out_ffn_;
    LongformerAttentionLayer<T>* longformer_attn_layer_;

    std::vector<LongformerLayerWeight<T>> weights_;

    cublasMMWrapper* cublas_wrapper_;
    IAllocator* allocator_;
    cudaStream_t stream_;
    bool is_free_buffer_after_forward_;
    bool is_allocate_buffer_ = false;

public:
    LongformerEncoder(size_t layers_num,
                      size_t in_dim,
                      size_t head_num,
                      size_t size_per_head,
                      size_t intermediate_size,
                      size_t local_attn_window_size,
                      size_t max_global_token_num,
                      size_t max_batch_size,
                      size_t max_seq_len,
                      float attn_scaler,
                      cudaStream_t stream,
                      cublasMMWrapper* cublas_wrapper,
                      IAllocator* allocator,
                      bool is_free_buffer_after_forward);
    ~LongformerEncoder();
    void forward(std::vector<Tensor>* output_tensors, std::vector<Tensor>* input_tensors);
    std::vector<LongformerLayerWeight<T>>* getWeightsPtr();

private:
    size_t getInitCubStorage(const int seq_len);
    void initLongformerIdx(const T* global_attn_mask, const int seq_len, const int batch_size);
    void allocateBuffer();
    void freeBuffer();
    void forwardLayer(T* input,
                      T* output,
                      const T* local_attn_mask,
                      const T* global_attn_mask,
                      const int* global_idx,
                      const int* global_token_nums,
                      const LongformerLayerWeight<T>* weight,
                      const size_t batch_size,
                      const size_t seq_len,
                      const size_t in_dim);
};

}  // namespace fastertransformer
