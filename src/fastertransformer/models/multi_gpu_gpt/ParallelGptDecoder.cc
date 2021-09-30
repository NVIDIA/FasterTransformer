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

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoder.h"

namespace fastertransformer {

template<typename T>
void ParallelGptDecoder<T>::initialize()
{
    self_attention_layer_ = new TensorParallelDecoderSelfAttentionLayer<T>(max_batch_size_,
                                                                           head_num_,
                                                                           size_per_head_,
                                                                           tensor_para_size_,
                                                                           tensor_para_comm_,
                                                                           stream_,
                                                                           cublas_wrapper_,
                                                                           allocator_,
                                                                           is_free_buffer_after_forward_);

    ffn_layer_ = new TensorParallelGeluFfnLayer<T>(max_batch_size_,
                                                   1,
                                                   head_num_,
                                                   size_per_head_,
                                                   inter_size_,
                                                   tensor_para_size_,
                                                   tensor_para_comm_,
                                                   stream_,
                                                   cublas_wrapper_,
                                                   allocator_,
                                                   is_free_buffer_after_forward_);
    allocateBuffer();
}

template<typename T>
ParallelGptDecoder<T>::ParallelGptDecoder(size_t max_batch_size,
                                          size_t head_num,
                                          size_t size_per_head,
                                          size_t inter_size,
                                          size_t num_layer,
                                          size_t tensor_para_size,
                                          size_t tensor_para_rank,
                                          ncclComm_t tensor_para_comm,
                                          size_t layer_para_size,
                                          size_t layer_para_rank,
                                          ncclComm_t layer_para_comm,
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
    hidden_units_(head_num_ * size_per_head_),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    tensor_para_comm_(tensor_para_comm),
    layer_para_size_(layer_para_size),
    layer_para_rank_(layer_para_rank),
    layer_para_comm_(layer_para_comm)
{
    initialize();
}

template<typename T>
ParallelGptDecoder<T>::ParallelGptDecoder(ParallelGptDecoder<T> const& decoder):
    BaseLayer(decoder.stream_, decoder.cublas_wrapper_, decoder.allocator_, decoder.is_free_buffer_after_forward_),
    max_batch_size_(decoder.max_batch_size_),
    head_num_(decoder.head_num_),
    size_per_head_(decoder.size_per_head_),
    inter_size_(decoder.inter_size_),
    num_layer_(decoder.num_layer_),
    hidden_units_(decoder.hidden_units_),
    tensor_para_size_(decoder.tensor_para_size_),
    tensor_para_rank_(decoder.tensor_para_rank_),
    tensor_para_comm_(decoder.tensor_para_comm_),
    layer_para_size_(decoder.layer_para_size_),
    layer_para_rank_(decoder.layer_para_rank_),
    layer_para_comm_(decoder.layer_para_comm_)
{
    initialize();
}

template<typename T>
void ParallelGptDecoder<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        decoder_layer_output_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * hidden_units_, false));
        decoder_normed_input_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * hidden_units_, false));
        self_attn_output_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * hidden_units_, false));
        normed_self_attn_output_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * hidden_units_, false));
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void ParallelGptDecoder<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free(decoder_layer_output_);
        allocator_->free(decoder_normed_input_);
        allocator_->free(self_attn_output_);
        allocator_->free(normed_self_attn_output_);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool ParallelGptDecoder<T>::isValidBatchSize(size_t batch_size)
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
bool ParallelGptDecoder<T>::isValidLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / layer_para_size_));
    return l < num_layer_ && (l >= local_num_layer * layer_para_rank_)
           && (l < local_num_layer * (layer_para_rank_ + 1));
}

template<typename T>
bool ParallelGptDecoder<T>::isFirstLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / layer_para_size_));
    return l < num_layer_ && (l == local_num_layer * layer_para_rank_);
}

template<typename T>
bool ParallelGptDecoder<T>::isLastLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / layer_para_size_));
    return l < num_layer_ && (l == local_num_layer * (layer_para_rank_ + 1) - 1);
}

template<typename T>
int ParallelGptDecoder<T>::getFirstLayerParallelId()
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / layer_para_size_));
    return local_num_layer * layer_para_rank_;
}

template<typename T>
ParallelGptDecoder<T>::~ParallelGptDecoder()
{
    delete self_attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void ParallelGptDecoder<T>::forward(std::vector<Tensor>* output_tensors,
                                    const std::vector<Tensor>* input_tensors,
                                    const std::vector<ParallelGptDecoderLayerWeight<T>*>* gpt_decoder_layer_weight)
{
    // input tensors:
    //      decoder_input [local_batch_size, hidden_dimension],
    //      finished [local_batch_size],
    //      sequence_lengths [local_batch_size]
    //      input_lengths [local_batch_size],
    //      max_input_length [1] on cpu
    //      step [1] on cpu
    //      ite [1] on cpu

    // output tensors:
    //      decoder_output [local_batch_size, hidden_dimension],
    //      key_cache [num_layer, batch_size, head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [num_layer, batch_size, head_num, max_seq_len, size_per_head]

    FT_CHECK(input_tensors->size() == 7);
    FT_CHECK(output_tensors->size() == 3);
    isValidBatchSize(input_tensors->at(0).shape[0]);
    allocateBuffer();

    const DataType data_type = getTensorType<T>();
    const size_t local_batch_size = input_tensors->at(0).shape[0];
    const int ite = *((int*)(input_tensors->at(6).data));

    std::vector<size_t> self_k_cache_size;
    self_k_cache_size.push_back(local_batch_size);
    for (auto t = output_tensors->at(1).shape.begin() + 2; t != output_tensors->at(1).shape.end(); ++t) {
        self_k_cache_size.push_back(*t);
    }
    std::vector<size_t> self_v_cache_size;
    self_v_cache_size.push_back(local_batch_size);
    for (auto t = output_tensors->at(2).shape.begin() + 2; t != output_tensors->at(2).shape.end(); ++t) {
        self_v_cache_size.push_back(*t);
    }

    for (uint l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l) == false) {
            continue;
        }
        T* decoder_input = (T*)((l == 0) ? input_tensors->at(0).data : decoder_layer_output_);
        T* decoder_output = (T*)((l == num_layer_ - 1) ? output_tensors->at(0).data : decoder_layer_output_);

        if (isFirstLayerParallelId(l) == true && layer_para_rank_ != 0 && layer_para_size_ > 1) {
            // ftNcclRecv(decoder_input, local_batch_size * hidden_units_, layer_para_rank_ - 1, layer_para_comm_,
            // stream_);

            ftNcclRecv(decoder_input + local_batch_size * hidden_units_ / tensor_para_size_ * tensor_para_rank_,
                       local_batch_size * hidden_units_ / tensor_para_size_,
                       layer_para_rank_ - 1,
                       layer_para_comm_,
                       stream_);
            if (tensor_para_size_ > 1) {
                ftNcclAllGather(decoder_input,
                                decoder_input,
                                local_batch_size * hidden_units_ / tensor_para_size_,
                                tensor_para_rank_,
                                tensor_para_comm_,
                                stream_);
            }
        }

        size_t cache_offset = l - getFirstLayerParallelId();
        for (auto t = output_tensors->at(1).shape.begin() + 1; t != output_tensors->at(1).shape.end(); ++t) {
            cache_offset *= *t;
        };
        size_t ite_cache_offset = ite * local_batch_size;
        for (auto t = output_tensors->at(1).shape.begin() + 2; t != output_tensors->at(1).shape.end(); ++t) {
            ite_cache_offset *= *t;
        }
        cache_offset += ite_cache_offset;

        invokeGeneralLayerNorm(decoder_normed_input_,
                               decoder_input,
                               gpt_decoder_layer_weight->at(l)->pre_layernorm_weights.gamma,
                               gpt_decoder_layer_weight->at(l)->pre_layernorm_weights.beta,
                               local_batch_size,
                               hidden_units_,
                               stream_);
        sync_check_cuda_error();

        std::vector<Tensor> self_attention_input_tensors{
            Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, decoder_normed_input_},
            input_tensors->at(1),
            input_tensors->at(2),
            input_tensors->at(3),
            input_tensors->at(4),
            input_tensors->at(5)};

        std::vector<Tensor> self_attention_output_tensors{
            Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, self_attn_output_},
            Tensor{MEMORY_GPU, data_type, self_k_cache_size, ((const T*)output_tensors->at(1).data) + cache_offset},
            Tensor{MEMORY_GPU, data_type, self_v_cache_size, ((const T*)output_tensors->at(2).data) + cache_offset}};

        self_attention_layer_->forward(&self_attention_output_tensors,
                                       &self_attention_input_tensors,
                                       &gpt_decoder_layer_weight->at(l)->self_attention_weights);

        invokeGeneralAddBiasResidualPreLayerNorm(
            self_attn_output_,
            normed_self_attn_output_,
            decoder_input,
            gpt_decoder_layer_weight->at(l)->self_attn_layernorm_weights.gamma,
            gpt_decoder_layer_weight->at(l)->self_attn_layernorm_weights.beta,
            gpt_decoder_layer_weight->at(l)->self_attention_weights.attention_output_weight.bias,
            local_batch_size,
            hidden_units_,
            stream_);
        sync_check_cuda_error();

        std::vector<Tensor> ffn_input_tensors{
            Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, normed_self_attn_output_}};
        std::vector<Tensor> ffn_output_tensors{
            Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, decoder_output}};
        ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &gpt_decoder_layer_weight->at(l)->ffn_weights);
        invokeAddBiasResidual(decoder_output,
                              self_attn_output_,
                              gpt_decoder_layer_weight->at(l)->ffn_weights.output_weight.bias,
                              local_batch_size,
                              hidden_units_,
                              stream_);
        sync_check_cuda_error();

        if (isLastLayerParallelId(l) == true && layer_para_rank_ != layer_para_size_ - 1 && layer_para_size_ > 1) {
            // ftNcclSend(decoder_output, local_batch_size * hidden_units_, layer_para_rank_ + 1, layer_para_comm_,
            // stream_);

            ftNcclSend(decoder_output + local_batch_size * hidden_units_ / tensor_para_size_ * tensor_para_rank_,
                       local_batch_size * hidden_units_ / tensor_para_size_,
                       layer_para_rank_ + 1,
                       layer_para_comm_,
                       stream_);
        }
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template class ParallelGptDecoder<float>;
template class ParallelGptDecoder<half>;

}  // namespace fastertransformer