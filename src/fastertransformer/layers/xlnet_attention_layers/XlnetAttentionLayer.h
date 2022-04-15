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

#include "src/fastertransformer/kernels/xlnet_attention_kernels.h"
#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/layers/xlnet_attention_layers/XlnetAttentionWeight.h"

namespace fastertransformer {

template<typename T>
class XlnetAttentionLayer: public BaseLayer {
private:
    // buffer handling
    size_t max_batch_size_ = 0;
    size_t max_seq_len_ = 0;

    // metadata
    int head_num_;
    int size_per_head_;
    float q_scaling_;

    // calculated params
    size_t hidden_units_;

    void allocateBuffer() override;
    void freeBuffer() override;
    bool isValidBatchSize(size_t batch_size);
    bool isValidSeqLen(size_t seq_len);

    using BaseLayer::stream_;
    using BaseLayer::is_free_buffer_after_forward_;
    using BaseLayer::is_allocate_buffer_;
    using BaseLayer::cublas_wrapper_;
    using BaseLayer::allocator_;

protected:
    T* k_head_r_;
    T* query_buf_;
    T* key_buf_;
    T* value_buf_;
    T* q_buf_;
    T* k_buf_;
    T* qk_buf_;
    T* q_buf_bd_;
    T* k_buf_bd_;
    T* qk_buf_bd_;
    T* qk_buf_bd_shift_;
    T* q_buf_ef_;
    T* k_buf_ef_;
    T* qk_buf_ef_;
    T* qk_buf_ef_trans_;
    T* qk_buf_ef_seg_;
    T* qk_buf_ef_seg_trans_;
    T* attn_score_;
    T* value_buf_trans_;
    T* attn_vec_;
    T* attn_vec_trans_;
    T* attn_out_;

public:
    XlnetAttentionLayer(size_t max_batch_size,
                        size_t max_seq_len,
                        size_t head_num,
                        size_t size_per_head,
                        float q_scaling,
                        cudaStream_t stream,
                        cublasMMWrapper* cublas_wrapper,
                        IAllocator* allocator,
                        bool is_free_buffer_after_forward);

    XlnetAttentionLayer(XlnetAttentionLayer<T> const& attention_layer);

    ~XlnetAttentionLayer();

    void oneToManyCublasGemm(T* d_A,
                             T* d_B,
                             T* d_C,
                             cublasOperation_t transa,
                             cublasOperation_t transb,
                             int v_m,
                             int v_n,
                             int v_k,
                             int lda,
                             int strideA,
                             int ldb,
                             int strideB,
                             int ldc,
                             int strideC,
                             int batch);

    void forward(std::vector<fastertransformer::Tensor>* output_tensors,
                 const std::vector<fastertransformer::Tensor>* input_tensors,
                 const XlnetAttentionWeight<T>* xlnet_attention_weights);
};

}  // namespace fastertransformer
