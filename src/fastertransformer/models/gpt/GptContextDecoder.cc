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

#include "src/fastertransformer/models/gpt/GptContextDecoder.h"

namespace fastertransformer {

template<typename T>
void GptContextDecoder<T>::initialize()
{
    self_attention_layer_ = new GptContextAttentionLayer<T>(max_batch_size_,
                                                            max_seq_len_,
                                                            head_num_,
                                                            size_per_head_,
                                                            stream_,
                                                            cublas_wrapper_,
                                                            allocator_,
                                                            is_free_buffer_after_forward_,
                                                            is_qk_buf_float_,
                                                            sparse_);

    ffn_layer_ = new GeluFfnLayer<T>(max_batch_size_,
                                     max_seq_len_,
                                     head_num_,
                                     size_per_head_,
                                     inter_size_,
                                     stream_,
                                     cublas_wrapper_,
                                     allocator_,
                                     is_free_buffer_after_forward_,
                                     sparse_);
}

template<typename T>
void GptContextDecoder<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        decoder_normed_input_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false));
        self_attn_output_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false));
        normed_self_attn_output_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false));
        decoder_layer_output_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false));
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void GptContextDecoder<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free(decoder_normed_input_);
        allocator_->free(self_attn_output_);
        allocator_->free(normed_self_attn_output_);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool GptContextDecoder<T>::isValidBatchSize(size_t batch_size)
{
    if (batch_size <= max_batch_size_) {
        return true;
    }
    else {
        freeBuffer();
        max_batch_size_ = batch_size * 1.2;
        return true;
    }
}

template<typename T>
bool GptContextDecoder<T>::isValidSeqLen(size_t seq_len)
{
    if (seq_len <= max_seq_len_) {
        return true;
    }
    else {
        freeBuffer();
        max_seq_len_ = seq_len * 1.2;
        return true;
    }
}

template<typename T>
GptContextDecoder<T>::GptContextDecoder(size_t max_batch_size,
                                        size_t max_seq_len,
                                        size_t head_num,
                                        size_t size_per_head,
                                        size_t inter_size,
                                        size_t num_layer,
                                        cudaStream_t stream,
                                        cublasMMWrapper* cublas_wrapper,
                                        IAllocator* allocator,
                                        bool is_free_buffer_after_forward,
                                        bool is_qk_buf_float,
                                        bool sparse):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    hidden_units_(head_num * size_per_head),
    is_qk_buf_float_(is_qk_buf_float)
{
    initialize();
}

template<typename T>
GptContextDecoder<T>::GptContextDecoder(GptContextDecoder<T> const& decoder):
    BaseLayer(decoder.stream_, decoder.cublas_wrapper_, decoder.allocator_, decoder.is_free_buffer_after_forward_),
    max_batch_size_(decoder.max_batch_size_),
    max_seq_len_(decoder.max_seq_len_),
    head_num_(decoder.head_num_),
    size_per_head_(decoder.size_per_head_),
    inter_size_(decoder.inter_size_),
    num_layer_(decoder.num_layer_),
    hidden_units_(decoder.hidden_units_),
    is_qk_buf_float_(decoder.is_qk_buf_float_)
{
    initialize();
}

template<typename T>
GptContextDecoder<T>::~GptContextDecoder()
{
    delete self_attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void GptContextDecoder<T>::forward(std::vector<Tensor>* output_tensors,
                                   const std::vector<Tensor>* input_tensors,
                                   const std::vector<GptDecoderLayerWeight<T>>* gpt_decoder_layer_weight)
{
    // input tensors:
    //      decoder_input [batch_size, seq_len, hidden_dimension],
    //      attention_mask [batch_size, 1, seq_len, seq_len]

    // output tensors:
    //      decoder_output [batch_size, seq_len, hidden_dimension],
    //      key_cache [num_layer, batch, head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [num_layer, batch, head_num, max_seq_len, size_per_head]

    FT_CHECK(input_tensors->size() == 2);
    FT_CHECK(output_tensors->size() == 3);
    isValidBatchSize(input_tensors->at(0).shape[0]);
    isValidSeqLen(input_tensors->at(0).shape[1]);
    allocateBuffer();

    const size_t batch_size = (size_t)input_tensors->at(0).shape[0];
    const size_t seq_len = (size_t)input_tensors->at(0).shape[1];
    const DataType data_type = getTensorType<T>();

    for (uint l = 0; l < num_layer_; l++) {
        const bool is_final = l == (num_layer_ - 1);
        const T* decoder_input = (const T*)((l == 0) ? input_tensors->at(0).data : decoder_layer_output_);
        T* decoder_output = (T*)((l == (num_layer_ - 1)) ? output_tensors->at(0).data : decoder_layer_output_);

        size_t self_key_cache_offset = l;
        for (auto t = output_tensors->at(1).shape.begin() + 1; t != output_tensors->at(1).shape.end(); ++t) {
            self_key_cache_offset *= (*t);
        }
        size_t self_value_cache_offset = l;
        for (auto t = output_tensors->at(2).shape.begin() + 1; t != output_tensors->at(2).shape.end(); ++t) {
            self_value_cache_offset *= (*t);
        }

        invokeGeneralLayerNorm(decoder_normed_input_,
                               decoder_input,
                               gpt_decoder_layer_weight->at(l).pre_layernorm_weights.gamma,
                               gpt_decoder_layer_weight->at(l).pre_layernorm_weights.beta,
                               batch_size * seq_len,
                               hidden_units_,
                               stream_);
        sync_check_cuda_error();

        std::vector<Tensor> self_attention_input_tensors{
            Tensor{MEMORY_GPU, data_type, {batch_size * seq_len, hidden_units_}, decoder_normed_input_},
            input_tensors->at(1),
            Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &is_final}};
        std::vector<Tensor> self_attention_output_tensors{
            Tensor{MEMORY_GPU, data_type, {batch_size * seq_len, hidden_units_}, self_attn_output_},
            Tensor{MEMORY_GPU,
                   data_type,
                   std::vector<size_t>(output_tensors->at(1).shape.begin() + 1, output_tensors->at(1).shape.end()),
                   ((const T*)output_tensors->at(1).data) + self_key_cache_offset},
            Tensor{MEMORY_GPU,
                   data_type,
                   std::vector<size_t>(output_tensors->at(2).shape.begin() + 1, output_tensors->at(2).shape.end()),
                   ((const T*)output_tensors->at(2).data) + self_value_cache_offset}};
        self_attention_layer_->forward(&self_attention_output_tensors,
                                       &self_attention_input_tensors,
                                       &gpt_decoder_layer_weight->at(l).self_attention_weights);

        if (is_final == false) {
            invokeGeneralAddBiasResidualPreLayerNorm(
                self_attn_output_,
                normed_self_attn_output_,
                decoder_input,
                gpt_decoder_layer_weight->at(l).self_attn_layernorm_weights.gamma,
                gpt_decoder_layer_weight->at(l).self_attn_layernorm_weights.beta,
                gpt_decoder_layer_weight->at(l).self_attention_weights.attention_output_weight.bias,
                batch_size * seq_len,
                hidden_units_,
                stream_);
            sync_check_cuda_error();

            std::vector<Tensor> ffn_input_tensors{
                Tensor{MEMORY_GPU, data_type, {batch_size * seq_len, hidden_units_}, normed_self_attn_output_}};
            std::vector<Tensor> ffn_output_tensors{
                Tensor{MEMORY_GPU, data_type, {batch_size * seq_len, hidden_units_}, decoder_output}};

            ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &gpt_decoder_layer_weight->at(l).ffn_weights);
            invokeAddBiasResidual(decoder_output,
                                  self_attn_output_,
                                  gpt_decoder_layer_weight->at(l).ffn_weights.output_weight.bias,
                                  batch_size * seq_len,
                                  hidden_units_,
                                  stream_);
            sync_check_cuda_error();
        }
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template class GptContextDecoder<float>;
template class GptContextDecoder<half>;

}  // namespace fastertransformer