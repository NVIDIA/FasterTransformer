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

#include "BertLayerINT8.h"

namespace fastertransformer {

template<typename T>
void BertLayerINT8<T>::initialize()
{
    if ((attention_type_ == AttentionType::FUSED_MHA || attention_type_ == AttentionType::FUSED_PADDED_MHA)
        && max_seq_len_ <= 384) {
        attention_layer_ = new FusedAttentionLayerINT8<T>(max_batch_size_,
                                                          max_seq_len_,
                                                          head_num_,
                                                          size_per_head_,
                                                          sm_,
                                                          q_scaling_,
                                                          int8_mode_,
                                                          stream_,
                                                          cublas_wrapper_,
                                                          allocator_,
                                                          is_free_buffer_after_forward_,
                                                          sparse_);
    }
    else if (attention_type_ == AttentionType::UNFUSED_MHA || attention_type_ == AttentionType::UNFUSED_PADDED_MHA) {
        attention_layer_ = new UnfusedAttentionLayerINT8<T>(max_batch_size_,
                                                            max_seq_len_,
                                                            head_num_,
                                                            size_per_head_,
                                                            q_scaling_,
                                                            int8_mode_,
                                                            stream_,
                                                            cublas_wrapper_,
                                                            allocator_,
                                                            is_free_buffer_after_forward_,
                                                            sparse_);
    }
    else {
        throw std::runtime_error(std::string("[FT][ERROR] Invalid attention type \n"));
    }
    ffn_layer_ = new GeluFfnLayerINT8<T>(max_batch_size_,
                                         max_seq_len_,
                                         head_num_,
                                         size_per_head_,
                                         inter_size_,
                                         int8_mode_,
                                         stream_,
                                         cublas_wrapper_,
                                         allocator_,
                                         is_free_buffer_after_forward_,
                                         sparse_);
}

template<typename T>
BertLayerINT8<T>::BertLayerINT8(size_t max_batch_size,
                                size_t max_seq_len,
                                size_t head_num,
                                size_t size_per_head,
                                size_t inter_size,
                                int sm,
                                float q_scaling,
                                int int8_mode,
                                cudaStream_t stream,
                                cublasMMWrapper* cublas_wrapper,
                                IAllocator* allocator,
                                bool is_free_buffer_after_forward,
                                AttentionType attention_type,
                                bool sparse):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    sm_(sm),
    q_scaling_(q_scaling),
    hidden_units_(head_num_ * size_per_head),
    attention_type_(attention_type),
    int8_mode_(int8_mode),
    sparse_(sparse)
{
    initialize();
}

template<typename T>
BertLayerINT8<T>::BertLayerINT8(BertLayerINT8<T> const& bert_layer):
    BaseLayer(bert_layer.stream_,
              bert_layer.cublas_wrapper_,
              bert_layer.allocator_,
              bert_layer.is_free_buffer_after_forward_),
    max_batch_size_(bert_layer.max_batch_size_),
    max_seq_len_(bert_layer.max_seq_len_),
    head_num_(bert_layer.head_num_),
    size_per_head_(bert_layer.size_per_head_),
    inter_size_(bert_layer.inter_size_),
    sm_(bert_layer.sm_),
    q_scaling_(bert_layer.q_scaling_),
    hidden_units_(bert_layer.hidden_units_),
    attention_type_(bert_layer.attention_type_),
    int8_mode_(bert_layer.int8_mode_),
    sparse_(bert_layer.sparse_)
{
    initialize();
}

template<typename T>
BertLayerINT8<T>::~BertLayerINT8()
{
    delete attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void BertLayerINT8<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        attn_out_buf_ = reinterpret_cast<int32_t*>(
            allocator_->malloc(sizeof(int32_t) * max_batch_size_ * max_seq_len_ * hidden_units_, false));

        int8_buf_ = reinterpret_cast<int8_t*>(
            allocator_->malloc(sizeof(int8_t) * max_batch_size_ * max_seq_len_ * hidden_units_, false));

        layer_norm_tmp_buf_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false));

        transformer_out_tmp_DataType_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false));

        // re-use transformer_out_tmp_DataType_ as col32_from_tensor_
        col32_from_tensor_ = transformer_out_tmp_DataType_;
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void BertLayerINT8<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free(attn_out_buf_);
        allocator_->free(int8_buf_);
        allocator_->free(layer_norm_tmp_buf_);
        allocator_->free(transformer_out_tmp_DataType_);
        is_allocate_buffer_ = false;
    }
}

// for layer_idx == 0, the data_type of input should be T, we need to quantize the input;
// for layer_idx == 0, the layout of input should be row-major, we need to transform the input;
// for layer_idx != 0, the data_type of input should be int8 for int8_mode=2/3, and T for int8_mode=1 (we need to
// quantize for int8_mode=1); for layer_idx != 0, the layout of input should be COL32.

template<typename T>
void BertLayerINT8<T>::forward(std::vector<Tensor>* output_tensors,
                               const std::vector<Tensor>* input_tensors,
                               const BertLayerWeight<T>* bert_layer_weight)
{
    const BertLayerINT8Weight<T>* bert_layer_int8_weight = (const BertLayerINT8Weight<T>*)bert_layer_weight;
    const ScaleList* scale_list = &(bert_layer_int8_weight->scale_list_);

    // input_tensors: [input_query (token_num, hidden_dimension),
    //                 attention_mask (batch, 1, seqlen, seqlen),
    //                 padding_offset (token_num),
    //                 layer_idx (1),
    //                 num_layer (1)]
    // output_tensors: [output (token_num, hidden_dimension)]
    // If padding_offset.data is nullptr, then not remove padding

    FT_CHECK(input_tensors->size() == 5);
    FT_CHECK(input_tensors->at(0).shape.size() == 2);
    FT_CHECK(input_tensors->at(1).shape.size() == 4);

    FT_CHECK(isValidBatchSize(input_tensors->at(1).shape[0]));
    FT_CHECK(isValidSeqLen(input_tensors->at(1).shape[2]));
    allocateBuffer();

    T* from_tensor = (T*)input_tensors->at(0).data;
    T* out_tensor = (T*)(output_tensors->at(0).data);

    const size_t m = input_tensors->at(0).shape[0];
    const size_t n = hidden_units_;

    const int layer_idx = *(int*)(input_tensors->at(3).data);
    const int num_layer = *(int*)(input_tensors->at(4).data);

    std::vector<Tensor> attn_output_tensors{
        Tensor{MEMORY_GPU, getTensorType<int>(), std::vector<size_t>{m, n}, attn_out_buf_},
    };

    if (int8_mode_ == 1) {

        if (layer_idx == 0) {
            invokeTransposeMatrixColMajorToCOL32(col32_from_tensor_, from_tensor, n, m, stream_);
            from_tensor = col32_from_tensor_;
        }
        invokeQuantization(int8_buf_, from_tensor, m * n, &(scale_list->d_scale_list_[3]), stream_);
        std::vector<Tensor> int8_input_tensors{Tensor{MEMORY_GPU, TYPE_INT8, std::vector<size_t>{m, n}, int8_buf_},
                                               input_tensors->at(1),
                                               input_tensors->at(2)};
        attention_layer_->forward(
            &attn_output_tensors, &int8_input_tensors, &bert_layer_int8_weight->attention_weights);
        // int32 I ; DataType O

        invokeAddBiasResidualLayerNormCol32(layer_norm_tmp_buf_,
                                            attn_out_buf_,
                                            from_tensor,
                                            bert_layer_int8_weight->attention_weights.attention_output_weight.bias,
                                            bert_layer_int8_weight->attn_layernorm_weights.gamma,
                                            bert_layer_int8_weight->attn_layernorm_weights.beta,
                                            m,
                                            n,
                                            stream_,
                                            &(scale_list->d_scale_list_[scale_list->p2_offset_ + 3 * hidden_units_]),
                                            &(scale_list->d_scale_list_[36]));
        invokeQuantization(int8_buf_, layer_norm_tmp_buf_, m * n, &(scale_list->d_scale_list_[44 + 3]), stream_);
        std::vector<Tensor> ffn_input_tensors{Tensor{MEMORY_GPU, TYPE_INT8, std::vector<size_t>{m, n}, int8_buf_}};
        // reuse attn_output_tensors as ffn_output_tensors
        ffn_layer_->forward(&attn_output_tensors, &ffn_input_tensors, &bert_layer_int8_weight->ffn_weights);
        if (layer_idx != num_layer - 1) {
            // int32 I ; DataType O
            invokeAddBiasResidualLayerNormCol32(
                out_tensor,
                attn_out_buf_,
                layer_norm_tmp_buf_,
                bert_layer_int8_weight->ffn_weights.output_weight.bias,
                bert_layer_int8_weight->ffn_layernorm_weights.gamma,
                bert_layer_int8_weight->ffn_layernorm_weights.beta,
                m,
                n,
                stream_,
                &(scale_list->d_scale_list_[scale_list->p2_offset_ + 8 * hidden_units_]),
                &(scale_list->d_scale_list_[52]));
        }
        else {
            // int32 I ; DataType O
            invokeAddBiasResidualLayerNormCol32(
                transformer_out_tmp_DataType_,
                attn_out_buf_,
                layer_norm_tmp_buf_,
                bert_layer_int8_weight->ffn_weights.output_weight.bias,
                bert_layer_int8_weight->ffn_layernorm_weights.gamma,
                bert_layer_int8_weight->ffn_layernorm_weights.beta,
                m,
                n,
                stream_,
                &(scale_list->d_scale_list_[scale_list->p2_offset_ + 8 * hidden_units_]),
                &(scale_list->d_scale_list_[52]));

            invokeTransposeMatrixCOL32ToColMajor(out_tensor, transformer_out_tmp_DataType_, m, n, stream_);
        }
    }
    else if (int8_mode_ == 2 || int8_mode_ == 3) {

        if (layer_idx == 0) {
#ifdef SPARSITY_ENABLED
            if (sparse_) {
                invokeQuantization(int8_buf_, from_tensor, m * n, &(scale_list->d_scale_list_[3]), stream_);
            }
            else {
#endif
                invokeTransposeMatrixColMajorToCOL32Quantize(
                    int8_buf_, from_tensor, n, m, &(scale_list->d_scale_list_[3]), stream_);
#ifdef SPARSITY_ENABLED
            }
#endif
            std::vector<Tensor> int8_input_tensors{Tensor{MEMORY_GPU, TYPE_INT8, std::vector<size_t>{m, n}, int8_buf_},
                                                   input_tensors->at(1),
                                                   input_tensors->at(2)};
            attention_layer_->forward(
                &attn_output_tensors, &int8_input_tensors, &bert_layer_int8_weight->attention_weights);
        }
        else {
            attention_layer_->forward(&attn_output_tensors, input_tensors, &bert_layer_int8_weight->attention_weights);
        }

        const int8_t* residual = layer_idx == 0 ? int8_buf_ : (const int8_t*)from_tensor;
        // int8 IO
#ifdef SPARSITY_ENABLED
        if (sparse_) {
            invokeAddBiasResidualLayerNormRow((int8_t*)layer_norm_tmp_buf_,
                                              (const int8_t*)attn_out_buf_,
                                              residual,
                                              bert_layer_int8_weight->attention_weights.attention_output_weight.bias,
                                              bert_layer_int8_weight->attn_layernorm_weights.gamma,
                                              bert_layer_int8_weight->attn_layernorm_weights.beta,
                                              m,
                                              n,
                                              stream_,
                                              &(scale_list->d_scale_list_[40 + 1]),
                                              &(scale_list->d_scale_list_[0 + 1]),
                                              &(scale_list->d_scale_list_[44 + 3]));
        }
        else {
#endif
            invokeAddBiasResidualLayerNormCol32((int8_t*)layer_norm_tmp_buf_,
                                                (const int8_t*)attn_out_buf_,
                                                residual,
                                                bert_layer_int8_weight->attention_weights.attention_output_weight.bias,
                                                bert_layer_int8_weight->attn_layernorm_weights.gamma,
                                                bert_layer_int8_weight->attn_layernorm_weights.beta,
                                                m,
                                                n,
                                                stream_,
                                                &(scale_list->d_scale_list_[40 + 1]),
                                                &(scale_list->d_scale_list_[0 + 1]),
                                                &(scale_list->d_scale_list_[44 + 3]));
#ifdef SPARSITY_ENABLED
        }
#endif
        std::vector<Tensor> ffn_input_tensors{
            Tensor{MEMORY_GPU, TYPE_INT8, std::vector<size_t>{m, n}, layer_norm_tmp_buf_}};
        // reuse attn_output_tensors as ffn_output_tensors
        ffn_layer_->forward(&attn_output_tensors, &ffn_input_tensors, &bert_layer_int8_weight->ffn_weights);
        if (layer_idx != num_layer - 1) {
            // int8 IO
#ifdef SPARSITY_ENABLED
            if (sparse_) {
                invokeAddBiasResidualLayerNormRow((int8_t*)out_tensor,
                                                  (int8_t*)attn_out_buf_,
                                                  (int8_t*)layer_norm_tmp_buf_,
                                                  bert_layer_int8_weight->ffn_weights.output_weight.bias,
                                                  bert_layer_int8_weight->ffn_layernorm_weights.gamma,
                                                  bert_layer_int8_weight->ffn_layernorm_weights.beta,
                                                  m,
                                                  n,
                                                  stream_,
                                                  &(scale_list->d_scale_list_[56 + 1]),
                                                  &(scale_list->d_scale_list_[44 + 1]),
                                                  &(scale_list->d_scale_list_[60 + 3]));
            }
            else {
#endif
                invokeAddBiasResidualLayerNormCol32((int8_t*)out_tensor,
                                                    (int8_t*)attn_out_buf_,
                                                    (int8_t*)layer_norm_tmp_buf_,
                                                    bert_layer_int8_weight->ffn_weights.output_weight.bias,
                                                    bert_layer_int8_weight->ffn_layernorm_weights.gamma,
                                                    bert_layer_int8_weight->ffn_layernorm_weights.beta,
                                                    m,
                                                    n,
                                                    stream_,
                                                    &(scale_list->d_scale_list_[56 + 1]),
                                                    &(scale_list->d_scale_list_[44 + 1]),
                                                    &(scale_list->d_scale_list_[60 + 3]));
#ifdef SPARSITY_ENABLED
            }
#endif
        }
        else {
#ifdef SPARSITY_ENABLED
            if (sparse_) {
                invokeAddBiasResidualLayerNormRow(out_tensor,
                                                  (int8_t*)attn_out_buf_,
                                                  (int8_t*)layer_norm_tmp_buf_,
                                                  bert_layer_int8_weight->ffn_weights.output_weight.bias,
                                                  bert_layer_int8_weight->ffn_layernorm_weights.gamma,
                                                  bert_layer_int8_weight->ffn_layernorm_weights.beta,
                                                  m,
                                                  n,
                                                  stream_,
                                                  &(scale_list->d_scale_list_[56 + 1]),
                                                  &(scale_list->d_scale_list_[44 + 1]));
            }
            else {
#endif
                // int8 I ; DataType O
                invokeAddBiasResidualLayerNormCol32(transformer_out_tmp_DataType_,
                                                    (int8_t*)attn_out_buf_,
                                                    (int8_t*)layer_norm_tmp_buf_,
                                                    bert_layer_int8_weight->ffn_weights.output_weight.bias,
                                                    bert_layer_int8_weight->ffn_layernorm_weights.gamma,
                                                    bert_layer_int8_weight->ffn_layernorm_weights.beta,
                                                    m,
                                                    n,
                                                    stream_,
                                                    &(scale_list->d_scale_list_[56 + 1]),
                                                    &(scale_list->d_scale_list_[44 + 1]));

                invokeTransposeMatrixCOL32ToColMajor(out_tensor, transformer_out_tmp_DataType_, m, n, stream_);
#ifdef SPARSITY_ENABLED
            }
#endif
        }
    }

    sync_check_cuda_error();
    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
bool BertLayerINT8<T>::isValidBatchSize(size_t batch_size)
{
    if (max_batch_size_ == 0) {
        max_batch_size_ = batch_size;
        return true;
    }
    else {
        return batch_size <= max_batch_size_;
    }
}

template<typename T>
bool BertLayerINT8<T>::isValidSeqLen(size_t seq_len)
{
    if (max_seq_len_ == 0) {
        max_seq_len_ = seq_len;
        return true;
    }
    else {
        return seq_len <= max_seq_len_;
    }
}

template class BertLayerINT8<float>;
template class BertLayerINT8<half>;

}  // namespace fastertransformer
