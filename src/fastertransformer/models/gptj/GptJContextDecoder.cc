/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/gptj/GptJContextDecoder.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"

#include "src/fastertransformer/layers/TensorParallelGeluFfnLayer.h"
#include "src/fastertransformer/layers/attention_layers/TensorParallelGptContextAttentionLayer.h"

namespace fastertransformer {

template<typename T>
void GptJContextDecoder<T>::initialize()
{
    self_attention_layer_ = new TensorParallelGptContextAttentionLayer<T>(max_batch_size_,
                                                                          max_seq_len_,
                                                                          head_num_,
                                                                          size_per_head_,
                                                                          rotary_embedding_dim_,
                                                                          tensor_para_,
                                                                          stream_,
                                                                          cublas_wrapper_,
                                                                          allocator_,
                                                                          is_free_buffer_after_forward_,
                                                                          is_qk_buf_float_,
                                                                          false,
                                                                          custom_all_reduce_comm_,
                                                                          enable_custom_all_reduce_);

    ffn_layer_ = new TensorParallelGeluFfnLayer<T>(max_batch_size_,
                                                   max_seq_len_,
                                                   head_num_,
                                                   size_per_head_,
                                                   inter_size_,
                                                   tensor_para_,
                                                   stream_,
                                                   cublas_wrapper_,
                                                   allocator_,
                                                   is_free_buffer_after_forward_,
                                                   false,
                                                   0,
                                                   custom_all_reduce_comm_,
                                                   enable_custom_all_reduce_);
}

template<typename T>
void GptJContextDecoder<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        decoder_normed_input_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false));
        self_attn_output_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false));
        ffn_output_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false));
        decoder_layer_output_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false));
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void GptJContextDecoder<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free(decoder_normed_input_);
        allocator_->free(self_attn_output_);
        allocator_->free(ffn_output_);
        allocator_->free(decoder_layer_output_);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool GptJContextDecoder<T>::isValidBatchSize(size_t batch_size)
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
bool GptJContextDecoder<T>::isValidSeqLen(size_t seq_len)
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
bool GptJContextDecoder<T>::isValidLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l >= local_num_layer * pipeline_para_.rank_)
           && (l < local_num_layer * (pipeline_para_.rank_ + 1));
}

template<typename T>
bool GptJContextDecoder<T>::isFirstLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * pipeline_para_.rank_);
}

template<typename T>
bool GptJContextDecoder<T>::isLastLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * (pipeline_para_.rank_ + 1) - 1);
}

template<typename T>
int GptJContextDecoder<T>::getFirstLayerParallelId()
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return local_num_layer * pipeline_para_.rank_;
}

template<typename T>
GptJContextDecoder<T>::GptJContextDecoder(size_t max_batch_size,
                                          size_t max_seq_len,
                                          size_t head_num,
                                          size_t size_per_head,
                                          size_t inter_size,
                                          size_t num_layer,
                                          size_t rotary_embedding_dim,
                                          NcclParam tensor_para,
                                          NcclParam pipeline_para,
                                          cudaStream_t stream,
                                          cublasMMWrapper* cublas_wrapper,
                                          IAllocator* allocator,
                                          bool is_free_buffer_after_forward,
                                          bool is_qk_buf_float,
                                          std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                                          int enable_custom_all_reduce):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    rotary_embedding_dim_(rotary_embedding_dim),
    hidden_units_(head_num * size_per_head),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para),
    is_qk_buf_float_(is_qk_buf_float),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce)
{
    initialize();
}

template<typename T>
GptJContextDecoder<T>::GptJContextDecoder(GptJContextDecoder<T> const& decoder):
    BaseLayer(decoder.stream_, decoder.cublas_wrapper_, decoder.allocator_, decoder.is_free_buffer_after_forward_),
    max_batch_size_(decoder.max_batch_size_),
    max_seq_len_(decoder.max_seq_len_),
    head_num_(decoder.head_num_),
    size_per_head_(decoder.size_per_head_),
    inter_size_(decoder.inter_size_),
    num_layer_(decoder.num_layer_),
    rotary_embedding_dim_(decoder.rotary_embedding_dim_),
    hidden_units_(decoder.hidden_units_),
    tensor_para_(decoder.tensor_para_),
    pipeline_para_(decoder.pipeline_para_),
    is_qk_buf_float_(decoder.is_qk_buf_float_),
    custom_all_reduce_comm_(decoder.custom_all_reduce_comm_),
    enable_custom_all_reduce_(decoder.enable_custom_all_reduce_)
{
    initialize();
}

template<typename T>
GptJContextDecoder<T>::~GptJContextDecoder()
{
    delete self_attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void GptJContextDecoder<T>::forward(std::vector<Tensor>* output_tensors,
                                    const std::vector<Tensor>* input_tensors,
                                    const std::vector<GptJDecoderLayerWeight<T>>* gpt_decoder_layer_weight)
{
    std::unordered_map<std::string, Tensor> input_tensors_map{{"decoder_input", input_tensors->at(0)},
                                                              {"attention_mask", input_tensors->at(1)},
                                                              {"input_lengths", input_tensors->at(2)}};
    std::unordered_map<std::string, Tensor> output_tensors_map{{"decoder_output", output_tensors->at(0)},
                                                               {"key_cache", output_tensors->at(1)},
                                                               {"value_cache", output_tensors->at(2)},
                                                               {"last_token_hidden_units", output_tensors->at(3)}};

    forward(&output_tensors_map, &input_tensors_map, gpt_decoder_layer_weight);
}

template<typename T>
void GptJContextDecoder<T>::forward(std::unordered_map<std::string, Tensor>* output_tensors,
                                    const std::unordered_map<std::string, Tensor>* input_tensors,
                                    const std::vector<GptJDecoderLayerWeight<T>>* gpt_decoder_layer_weight)
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
    isValidBatchSize(input_tensors->at("decoder_input").shape[0]);
    isValidSeqLen(input_tensors->at("decoder_input").shape[1]);
    allocateBuffer();

    const int batch_size = input_tensors->at("decoder_input").shape[0];
    const int seq_len = input_tensors->at("decoder_input").shape[1];
    const DataType data_type = getTensorType<T>();

    T* decoder_input = (T*)input_tensors->at("decoder_input").data;
    T* decoder_output = (T*)output_tensors->at("decoder_output").data;
    const T* attention_mask = (const T*)input_tensors->at("attention_mask").data;

    const int local_batch_size = getLocalBatchSize(batch_size, seq_len, pipeline_para_.world_size_);
    FT_CHECK(batch_size % local_batch_size == 0);
    const int iteration_num = batch_size / local_batch_size;

    Tensor& k_cache = output_tensors->at("key_cache");
    Tensor& v_cache = output_tensors->at("value_cache");
    std::vector<size_t> self_k_cache_size;
    self_k_cache_size.push_back(local_batch_size);
    for (auto t = k_cache.shape.begin() + 2; t != k_cache.shape.end(); ++t) {
        self_k_cache_size.push_back(*t);
    }
    std::vector<size_t> self_v_cache_size;
    self_v_cache_size.push_back(local_batch_size);
    for (auto t = v_cache.shape.begin() + 2; t != v_cache.shape.end(); ++t) {
        self_v_cache_size.push_back(*t);
    }

    for (int ite = 0; ite < iteration_num; ite++) {
        for (int l = 0; l < num_layer_; l++) {
            if (isValidLayerParallelId(l) == false) {
                continue;
            }

            const bool is_final = false;  // TODO(bhsueh) remove this flag
            T* layer_input =
                ((l == 0) ? decoder_input : decoder_layer_output_) + ite * local_batch_size * seq_len * hidden_units_;
            T* layer_output = ((l == num_layer_ - 1) ? decoder_output : decoder_layer_output_)
                              + ite * local_batch_size * seq_len * hidden_units_;

            if (isFirstLayerParallelId(l) && pipeline_para_.rank_ != 0 && pipeline_para_.world_size_ > 1) {
                int data_size = local_batch_size * seq_len * hidden_units_ / tensor_para_.world_size_;
                ftNcclRecv(layer_input + data_size * tensor_para_.rank_,
                           data_size,
                           pipeline_para_.rank_ - 1,
                           pipeline_para_,
                           stream_);
                if (tensor_para_.world_size_ > 1) {
                    ftNcclAllGather(layer_input, layer_input, data_size, tensor_para_.rank_, tensor_para_, stream_);
                }
            }

            invokeGeneralLayerNorm(decoder_normed_input_,
                                   layer_input,
                                   gpt_decoder_layer_weight->at(l).pre_layernorm_weights.gamma,
                                   gpt_decoder_layer_weight->at(l).pre_layernorm_weights.beta,
                                   local_batch_size * seq_len,
                                   hidden_units_,
                                   stream_);
            sync_check_cuda_error();

            std::vector<Tensor> self_attention_input_tensors{
                Tensor{MEMORY_GPU,
                       data_type,
                       {(size_t)(local_batch_size * seq_len), (size_t)hidden_units_},
                       decoder_normed_input_},
                Tensor{MEMORY_GPU,
                       data_type,
                       {(size_t)local_batch_size, (size_t)1, (size_t)seq_len, (size_t)seq_len},
                       attention_mask + local_batch_size * ite * seq_len * seq_len},
                Tensor{MEMORY_CPU, TYPE_BOOL, {(size_t)1}, &is_final}};

            size_t cache_offset = l - getFirstLayerParallelId();
            for (auto t = k_cache.shape.begin() + 1; t != k_cache.shape.end(); ++t) {
                cache_offset *= *t;
            };
            size_t ite_cache_offset = ite * local_batch_size;
            for (auto t = k_cache.shape.begin() + 2; t != k_cache.shape.end(); ++t) {
                ite_cache_offset *= *t;
            }
            cache_offset += ite_cache_offset;

            std::vector<Tensor> self_attention_output_tensors{
                Tensor{MEMORY_GPU,
                       data_type,
                       {(size_t)(local_batch_size * seq_len), (size_t)hidden_units_},
                       self_attn_output_},
                Tensor{MEMORY_GPU, data_type, self_k_cache_size, ((const T*)k_cache.data) + cache_offset},
                Tensor{MEMORY_GPU, data_type, self_v_cache_size, ((const T*)v_cache.data) + cache_offset}};

            self_attention_layer_->forward(&self_attention_output_tensors,
                                           &self_attention_input_tensors,
                                           &gpt_decoder_layer_weight->at(l).self_attention_weights);

            if (is_final == false) {
                std::vector<Tensor> ffn_input_tensors{
                    Tensor{MEMORY_GPU,
                           data_type,
                           {(size_t)(local_batch_size * seq_len), (size_t)hidden_units_},
                           decoder_normed_input_}};
                std::vector<Tensor> ffn_output_tensors{Tensor{
                    MEMORY_GPU, data_type, {(size_t)(local_batch_size * seq_len), (size_t)hidden_units_}, ffn_output_}};
                ffn_layer_->forward(
                    &ffn_output_tensors, &ffn_input_tensors, &gpt_decoder_layer_weight->at(l).ffn_weights);

                invokeAddBiasAttentionFfnResidual(layer_output,
                                                  ffn_output_,
                                                  self_attn_output_,
                                                  layer_input,
                                                  gpt_decoder_layer_weight->at(l).ffn_weights.output_weight.bias,
                                                  local_batch_size * seq_len,
                                                  hidden_units_,
                                                  stream_);
                sync_check_cuda_error();

                if (isLastLayerParallelId(l) && pipeline_para_.rank_ != pipeline_para_.world_size_ - 1
                    && pipeline_para_.world_size_ > 1) {
                    int data_size = local_batch_size * seq_len * hidden_units_ / tensor_para_.world_size_;
                    ftNcclSend(layer_output + data_size * tensor_para_.rank_,
                               data_size,
                               pipeline_para_.rank_ + 1,
                               pipeline_para_,
                               stream_);
                }
            }
        }
    }

    // TODO(bhsueh) We could optimize this point by only computing the last token for the last layer
    invokeLookupHiddenStateOfLastToken((T*)output_tensors->at("last_token_hidden_units").data,
                                       (T*)output_tensors->at("decoder_output").data,
                                       (int*)input_tensors->at("input_lengths").data,
                                       seq_len,
                                       batch_size,
                                       hidden_units_,
                                       stream_);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template class GptJContextDecoder<float>;
template class GptJContextDecoder<half>;

}  // namespace fastertransformer
