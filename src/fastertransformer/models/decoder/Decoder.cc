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

#include "src/fastertransformer/models/decoder/Decoder.h"

namespace fastertransformer {

template<typename T>
void Decoder<T>::initialize()
{
    self_attention_layer_ = new DecoderSelfAttentionLayer<T>(max_batch_size_,
                                                             head_num_,
                                                             size_per_head_,
                                                             stream_,
                                                             cublas_wrapper_,
                                                             allocator_,
                                                             is_free_buffer_after_forward_);

    cross_attention_layer_ = new DecoderCrossAttentionLayer<T>(max_batch_size_,
                                                               head_num_,
                                                               size_per_head_,
                                                               stream_,
                                                               cublas_wrapper_,
                                                               allocator_,
                                                               is_free_buffer_after_forward_);

    ffn_layer_ = new ReluFfnLayer<T>(max_batch_size_,
                                     1,
                                     head_num_,
                                     size_per_head_,
                                     inter_size_,
                                     stream_,
                                     cublas_wrapper_,
                                     allocator_,
                                     is_free_buffer_after_forward_);
}

template<typename T>
void Decoder<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        decoder_normed_input_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * hidden_units_, false));
        self_attn_output_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * hidden_units_, false));
        normed_self_attn_output_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * hidden_units_, false));
        cross_attn_output_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * hidden_units_, false));
        normed_cross_attn_output_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * hidden_units_, false));
        decoder_layer_output_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * hidden_units_, false));
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void Decoder<T>::allocateBuffer(size_t batch_size)
{
    decoder_normed_input_ =
        reinterpret_cast<T*>(decoder_normed_input_, allocator_->malloc(sizeof(T) * batch_size * hidden_units_, false));
    self_attn_output_ =
        reinterpret_cast<T*>(self_attn_output_, allocator_->malloc(sizeof(T) * batch_size * hidden_units_, false));
    normed_self_attn_output_ = reinterpret_cast<T*>(normed_self_attn_output_,
                                                    allocator_->malloc(sizeof(T) * batch_size * hidden_units_, false));
    cross_attn_output_ =
        reinterpret_cast<T*>(cross_attn_output_, allocator_->malloc(sizeof(T) * batch_size * hidden_units_, false));
    normed_cross_attn_output_ = reinterpret_cast<T*>(normed_cross_attn_output_,
                                                     allocator_->malloc(sizeof(T) * batch_size * hidden_units_, false));
    decoder_layer_output_ =
        reinterpret_cast<T*>(decoder_layer_output_, allocator_->malloc(sizeof(T) * batch_size * hidden_units_, false));
}

template<typename T>
void Decoder<T>::freeBuffer()
{
    allocator_->free(decoder_normed_input_);
    allocator_->free(self_attn_output_);
    allocator_->free(normed_self_attn_output_);
    allocator_->free(cross_attn_output_);
    allocator_->free(normed_cross_attn_output_);
    allocator_->free(decoder_layer_output_);
}

template<typename T>
bool Decoder<T>::isValidBatchSize(size_t batch_size)
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
Decoder<T>::Decoder(size_t max_batch_size,
                    size_t head_num,
                    size_t size_per_head,
                    size_t inter_size,
                    size_t num_layer,
                    cudaStream_t stream,
                    cublasMMWrapper* cublas_wrapper,
                    IAllocator* allocator,
                    bool is_free_buffer_after_forward):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_size_(max_batch_size),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    hidden_units_(head_num_ * size_per_head)
{
    initialize();
}

template<typename T>
Decoder<T>::Decoder(Decoder<T> const& decoder):
    BaseLayer(decoder.stream_, decoder.cublas_wrapper_, decoder.allocator_, decoder.is_free_buffer_after_forward_),
    max_batch_size_(decoder.max_batch_size_),
    head_num_(decoder.head_num_),
    size_per_head_(decoder.size_per_head_),
    inter_size_(decoder.inter_size_),
    num_layer_(decoder.num_layer_),
    hidden_units_(decoder.hidden_units_)
{
    initialize();
}

template<typename T>
Decoder<T>::~Decoder()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    delete self_attention_layer_;
    delete cross_attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void Decoder<T>::forward(std::vector<Tensor>* output_tensors,
                         const std::vector<Tensor>* input_tensors,
                         const std::vector<DecoderLayerWeight<T>>* decoder_layer_weight)
{
    // input tensors:
    //      decoder_input [batch_size, hidden_dimension],
    //      encoder_output [batch_size, mem_max_seq_len, memory_hidden_dimension],
    //      encoder_sequence_length [batch_size],
    //      finished [batch_size],
    //      step [1] on cpu
    //      sequence_lengths [batch_size]
    //      cache_indirection [local_batch_size / beam_width, beam_width, max_seq_len]
    //              Here, local_batch_size contains the beam_width, so local_batch_size / beam_width
    //              is real local_batch_size.

    // output tensors:
    //      decoder_output [batch_size, hidden_dimension],
    //      key_cache [num_layer, batch, head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [num_layer, batch, head_num, max_seq_len, size_per_head]
    //      key_mem_cache [num_layer, batch_size, mem_max_seq_len, hidden_dimension],
    //      value_mem_cache [num_layer, batch_size, mem_max_seq_len, hidden_dimension]

    FT_CHECK(input_tensors->size() == 7);
    FT_CHECK(output_tensors->size() == 5);
    isValidBatchSize(input_tensors->at(0).shape[0]);
    allocateBuffer(input_tensors->at(0).shape[0]);

    const size_t batch_size = (size_t)input_tensors->at(0).shape[0];
    const size_t mem_max_seq_len = (size_t)input_tensors->at(1).shape[1];
    const DataType data_type = getTensorType<T>();

    for (uint l = 0; l < num_layer_; l++) {
        const T* decoder_input = (const T*)((l == 0) ? input_tensors->at(0).data : decoder_layer_output_);
        T* decoder_output = (T*)((l == num_layer_ - 1) ? output_tensors->at(0).data : decoder_layer_output_);

        size_t self_key_cache_offset = l;
        for (auto t = output_tensors->at(1).shape.begin() + 1; t != output_tensors->at(1).shape.end(); ++t) {
            self_key_cache_offset *= (*t);
        }
        size_t self_value_cache_offset = l;
        for (auto t = output_tensors->at(2).shape.begin() + 1; t != output_tensors->at(2).shape.end(); ++t) {
            self_value_cache_offset *= (*t);
        }
        const size_t mem_cache_offset = l * batch_size * mem_max_seq_len * hidden_units_;

        invokeGeneralLayerNorm(decoder_normed_input_,
                               decoder_input,
                               decoder_layer_weight->at(l).pre_layernorm_weights.gamma,
                               decoder_layer_weight->at(l).pre_layernorm_weights.beta,
                               batch_size,
                               hidden_units_,
                               stream_);
        sync_check_cuda_error();

        int tmp_0 = 0;
        std::vector<Tensor> self_attention_input_tensors{
            Tensor{MEMORY_GPU, data_type, {batch_size, hidden_units_}, decoder_normed_input_},
            input_tensors->at(3),
            input_tensors->at(5),
            Tensor{MEMORY_GPU, TYPE_INT32, {batch_size}, nullptr},
            Tensor{MEMORY_CPU, TYPE_INT32, {1}, &tmp_0},
            input_tensors->at(4),
            input_tensors->at(6)};
        std::vector<Tensor> self_attention_output_tensors{
            Tensor{MEMORY_GPU, data_type, {batch_size, hidden_units_}, self_attn_output_},
            Tensor{MEMORY_GPU,
                   data_type,
                   std::vector<size_t>(output_tensors->at(1).shape.begin() + 1, output_tensors->at(1).shape.end()),
                   ((const T*)output_tensors->at(1).data) + (int)self_key_cache_offset},
            Tensor{MEMORY_GPU,
                   data_type,
                   std::vector<size_t>(output_tensors->at(2).shape.begin() + 1, output_tensors->at(2).shape.end()),
                   ((const T*)output_tensors->at(2).data) + (int)self_value_cache_offset}};
        self_attention_layer_->forward(&self_attention_output_tensors,
                                       &self_attention_input_tensors,
                                       &decoder_layer_weight->at(l).self_attention_weights);

        invokeGeneralAddBiasResidualPreLayerNorm(
            self_attn_output_,
            normed_self_attn_output_,
            decoder_input,
            decoder_layer_weight->at(l).self_attn_layernorm_weights.gamma,
            decoder_layer_weight->at(l).self_attn_layernorm_weights.beta,
            decoder_layer_weight->at(l).self_attention_weights.attention_output_weight.bias,
            batch_size,
            hidden_units_,
            stream_);
        sync_check_cuda_error();

        std::vector<Tensor> cross_attention_input_tensors{
            Tensor{MEMORY_GPU, data_type, {batch_size, hidden_units_}, normed_self_attn_output_},
            input_tensors->at(1),
            input_tensors->at(2),
            input_tensors->at(3),
            input_tensors->at(4)};
        std::vector<Tensor> cross_attention_output_tensors{
            Tensor{MEMORY_GPU, data_type, {batch_size, hidden_units_}, cross_attn_output_},
            Tensor{MEMORY_GPU,
                   data_type,
                   std::vector<size_t>(output_tensors->at(3).shape.begin() + 1, output_tensors->at(3).shape.end()),
                   ((const T*)output_tensors->at(3).data) + (int)mem_cache_offset},
            Tensor{MEMORY_GPU,
                   data_type,
                   std::vector<size_t>(output_tensors->at(4).shape.begin() + 1, output_tensors->at(4).shape.end()),
                   ((const T*)output_tensors->at(4).data) + (int)mem_cache_offset}};
        cross_attention_layer_->forward(&cross_attention_output_tensors,
                                        &cross_attention_input_tensors,
                                        &decoder_layer_weight->at(l).cross_attention_weights);

        invokeGeneralAddBiasResidualPreLayerNorm(
            cross_attn_output_,
            normed_cross_attn_output_,
            self_attn_output_,
            decoder_layer_weight->at(l).cross_attn_layernorm_weights.gamma,
            decoder_layer_weight->at(l).cross_attn_layernorm_weights.beta,
            decoder_layer_weight->at(l).cross_attention_weights.attention_output_weight.bias,
            batch_size,
            hidden_units_,
            stream_);
        sync_check_cuda_error();

        std::vector<Tensor> ffn_input_tensors{
            Tensor{MEMORY_GPU, data_type, {batch_size, hidden_units_}, normed_cross_attn_output_}};
        std::vector<Tensor> ffn_output_tensors{
            Tensor{MEMORY_GPU, data_type, {batch_size, hidden_units_}, decoder_output}};
        ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &decoder_layer_weight->at(l).ffn_weights);

        invokeAddBiasResidual(decoder_output,
                              cross_attn_output_,
                              decoder_layer_weight->at(l).ffn_weights.output_weight.bias,
                              batch_size,
                              hidden_units_,
                              stream_);
        sync_check_cuda_error();
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template class Decoder<float>;
template class Decoder<half>;

}  // namespace fastertransformer