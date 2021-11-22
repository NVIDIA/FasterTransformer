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

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptContextDecoder.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"

namespace fastertransformer {

template<typename T>
void ParallelGptContextDecoder<T>::initialize()
{
    self_attention_layer_ = new TensorParallelGptContextAttentionLayer<T>(max_batch_size_,
                                                                          max_seq_len_,
                                                                          head_num_,
                                                                          size_per_head_,
                                                                          tensor_para_size_,
                                                                          tensor_para_comm_,
                                                                          stream_,
                                                                          cublas_wrapper_,
                                                                          allocator_,
                                                                          is_free_buffer_after_forward_,
                                                                          is_qk_buf_float_,
                                                                          sparse_);

    ffn_layer_ = new TensorParallelGeluFfnLayer<T>(max_batch_size_,
                                                   max_seq_len_,
                                                   head_num_,
                                                   size_per_head_,
                                                   inter_size_,
                                                   tensor_para_size_,
                                                   tensor_para_comm_,
                                                   stream_,
                                                   cublas_wrapper_,
                                                   allocator_,
                                                   is_free_buffer_after_forward_,
                                                   sparse_);
}

template<typename T>
void ParallelGptContextDecoder<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        decoder_normed_input_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false));
        self_attn_output_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false));
        normed_self_attn_output_ = decoder_normed_input_;  // reuse the buffer
        decoder_layer_output_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false));
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void ParallelGptContextDecoder<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free(decoder_normed_input_);
        allocator_->free(self_attn_output_);
        allocator_->free(decoder_layer_output_);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool ParallelGptContextDecoder<T>::isValidBatchSize(size_t batch_size)
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
bool ParallelGptContextDecoder<T>::isValidSeqLen(size_t seq_len)
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
bool ParallelGptContextDecoder<T>::isValidLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / layer_para_size_));
    return l < num_layer_ && (l >= local_num_layer * layer_para_rank_)
           && (l < local_num_layer * (layer_para_rank_ + 1));
}

template<typename T>
bool ParallelGptContextDecoder<T>::isFirstLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / layer_para_size_));
    return l < num_layer_ && (l == local_num_layer * layer_para_rank_);
}

template<typename T>
bool ParallelGptContextDecoder<T>::isLastLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / layer_para_size_));
    return l < num_layer_ && (l == local_num_layer * (layer_para_rank_ + 1) - 1);
}

template<typename T>
int ParallelGptContextDecoder<T>::getFirstLayerParallelId()
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / layer_para_size_));
    return local_num_layer * layer_para_rank_;
}

template<typename T>
ParallelGptContextDecoder<T>::ParallelGptContextDecoder(size_t max_batch_size,
                                                        size_t max_seq_len,
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
    hidden_units_(head_num_ * size_per_head),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    tensor_para_comm_(tensor_para_comm),
    layer_para_size_(layer_para_size),
    layer_para_rank_(layer_para_rank),
    layer_para_comm_(layer_para_comm),
    is_qk_buf_float_(is_qk_buf_float)
{
    initialize();
}

template<typename T>
ParallelGptContextDecoder<T>::ParallelGptContextDecoder(ParallelGptContextDecoder<T> const& decoder):
    BaseLayer(decoder.stream_, decoder.cublas_wrapper_, decoder.allocator_, decoder.is_free_buffer_after_forward_),
    max_batch_size_(decoder.max_batch_size_),
    max_seq_len_(decoder.max_seq_len_),
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
    layer_para_comm_(decoder.layer_para_comm_),
    is_qk_buf_float_(decoder.is_qk_buf_float_)
{
    initialize();
}

template<typename T>
ParallelGptContextDecoder<T>::~ParallelGptContextDecoder()
{
    delete self_attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void ParallelGptContextDecoder<T>::forward(
    std::vector<Tensor>* output_tensors,
    const std::vector<Tensor>* input_tensors,
    const std::vector<ParallelGptDecoderLayerWeight<T>*>* gpt_decoder_layer_weight)
{
    // input tensors:
    //      decoder_input [batch_size, seq_len, hidden_dimension],
    //      attention_mask [batch_size, 1, seq_len, seq_len]
    //      input_lengths [batch_size]

    // output tensors:
    //      decoder_output [batch_size, seq_len, hidden_dimension],
    //      key_cache [num_layer, batch, local_head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [num_layer, batch, local_head_num, max_seq_len, size_per_head]
    //      last_token_hidden_units [batch_size, hidden_dimension]

    // To use layer/pipeline parallelism, we view the shape of 'batch_size' to 'ite * local_batch_size'.
    // For example, the shape of decoder_input becomes [ite, batch_size, seq_len, hidden_dimension] during
    // computing.

    FT_CHECK(input_tensors->size() == 3);
    FT_CHECK(output_tensors->size() == 4);
    isValidBatchSize(input_tensors->at(0).shape[0]);
    isValidSeqLen(input_tensors->at(0).shape[1]);
    allocateBuffer();

    const size_t batch_size = (size_t)input_tensors->at(0).shape[0];
    const size_t seq_len = (size_t)input_tensors->at(0).shape[1];
    const DataType data_type = getTensorType<T>();

    const size_t local_batch_size = getLocalBatchSize(batch_size, seq_len, layer_para_size_);
    FT_CHECK(batch_size % local_batch_size == 0);
    const size_t iteration_num = batch_size / local_batch_size;

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

    for (uint ite = 0; ite < iteration_num; ite++) {
        for (uint l = 0; l < num_layer_; l++) {
            if (isValidLayerParallelId(l) == false)
                continue;

            // const bool is_final = l == (num_layer_ - 1);
            const bool is_final = false;  // TODO(bhsueh) remove this flag
            T* decoder_input =
                (l == 0) ? (T*)(input_tensors->at(0).data) + (int)(ite * local_batch_size * seq_len * hidden_units_) :
                           decoder_layer_output_;
            T* decoder_output =
                (l == (num_layer_ - 1)) ?
                    (T*)(output_tensors->at(0).data) + (int)(ite * local_batch_size * seq_len * hidden_units_) :
                    decoder_layer_output_;

            if (isFirstLayerParallelId(l) && layer_para_rank_ != 0) {
                const int data_size = local_batch_size * seq_len * hidden_units_ / tensor_para_size_;
                ftNcclRecv(decoder_input + data_size * tensor_para_rank_,
                           data_size,
                           layer_para_rank_ - 1,
                           layer_para_comm_,
                           stream_);
                if (tensor_para_size_ > 1) {
                    ftNcclAllGather(
                        decoder_input, decoder_input, data_size, tensor_para_rank_, tensor_para_comm_, stream_);
                }
            }

            invokeGeneralLayerNorm(decoder_normed_input_,
                                   decoder_input,
                                   gpt_decoder_layer_weight->at(l)->pre_layernorm_weights.gamma,
                                   gpt_decoder_layer_weight->at(l)->pre_layernorm_weights.beta,
                                   local_batch_size * seq_len,
                                   hidden_units_,
                                   stream_);
            sync_check_cuda_error();

            std::vector<Tensor> self_attention_input_tensors{
                Tensor{MEMORY_GPU, data_type, {local_batch_size * seq_len, hidden_units_}, decoder_normed_input_},
                Tensor{MEMORY_GPU,
                       data_type,
                       {local_batch_size, 1, seq_len, seq_len},
                       (const T*)input_tensors->at(1).data + local_batch_size * ite * seq_len * seq_len},
                Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &is_final}};

            size_t cache_offset = l - getFirstLayerParallelId();
            for (auto t = output_tensors->at(1).shape.begin() + 1; t != output_tensors->at(1).shape.end(); ++t) {
                cache_offset *= *t;
            };
            size_t ite_cache_offset = ite * local_batch_size;
            for (auto t = output_tensors->at(1).shape.begin() + 2; t != output_tensors->at(1).shape.end(); ++t) {
                ite_cache_offset *= *t;
            }
            cache_offset += ite_cache_offset;

            std::vector<Tensor> self_attention_output_tensors{
                Tensor{MEMORY_GPU, data_type, {local_batch_size * seq_len, hidden_units_}, self_attn_output_},
                Tensor{MEMORY_GPU, data_type, self_k_cache_size, ((const T*)output_tensors->at(1).data) + cache_offset},
                Tensor{
                    MEMORY_GPU, data_type, self_v_cache_size, ((const T*)output_tensors->at(2).data) + cache_offset}};

            self_attention_layer_->forward(&self_attention_output_tensors,
                                           &self_attention_input_tensors,
                                           &gpt_decoder_layer_weight->at(l)->self_attention_weights);

            if (is_final == false) {
                invokeGeneralAddBiasResidualPreLayerNorm(
                    self_attn_output_,
                    normed_self_attn_output_,
                    decoder_input,
                    gpt_decoder_layer_weight->at(l)->self_attn_layernorm_weights.gamma,
                    gpt_decoder_layer_weight->at(l)->self_attn_layernorm_weights.beta,
                    gpt_decoder_layer_weight->at(l)->self_attention_weights.attention_output_weight.bias,
                    local_batch_size * seq_len,
                    hidden_units_,
                    stream_);
                sync_check_cuda_error();

                std::vector<Tensor> ffn_input_tensors{Tensor{
                    MEMORY_GPU, data_type, {local_batch_size * seq_len, hidden_units_}, normed_self_attn_output_}};
                std::vector<Tensor> ffn_output_tensors{
                    Tensor{MEMORY_GPU, data_type, {local_batch_size * seq_len, hidden_units_}, decoder_output}};

                ffn_layer_->forward(
                    &ffn_output_tensors, &ffn_input_tensors, &gpt_decoder_layer_weight->at(l)->ffn_weights);
                invokeAddBiasResidual(decoder_output,
                                      self_attn_output_,
                                      gpt_decoder_layer_weight->at(l)->ffn_weights.output_weight.bias,
                                      local_batch_size * seq_len,
                                      hidden_units_,
                                      stream_);
                sync_check_cuda_error();

                if (isLastLayerParallelId(l) == true && layer_para_rank_ != layer_para_size_ - 1) {
                    const int data_size = local_batch_size * seq_len * hidden_units_ / tensor_para_size_;
                    ftNcclSend(decoder_output + data_size * tensor_para_rank_,
                               data_size,
                               layer_para_rank_ + 1,
                               layer_para_comm_,
                               stream_);
                }
            }
        }
    }

    // TODO(bhsueh) We could optimize this point by only computing the last token for the last layer
    invokeLookupHiddenStateOfLastToken((T*)output_tensors->at(3).data,
                                       (T*)output_tensors->at(0).data,
                                       (int*)input_tensors->at(2).data,
                                       seq_len,
                                       batch_size,
                                       hidden_units_,
                                       stream_);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template class ParallelGptContextDecoder<float>;
template class ParallelGptContextDecoder<half>;

}  // namespace fastertransformer