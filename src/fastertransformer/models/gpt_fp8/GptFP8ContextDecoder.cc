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

#include "src/fastertransformer/models/gpt_fp8/GptFP8ContextDecoder.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"

namespace fastertransformer {

template<typename T1, typename T2>
void GptFP8ContextDecoder<T1, T2>::initialize()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    self_attention_layer_ = new TensorParallelGptContextAttentionFP8Layer<T1, T2>(head_num_,
                                                                                  size_per_head_,
                                                                                  0,
                                                                                  tensor_para_,
                                                                                  stream_,
                                                                                  cublas_wrapper_,
                                                                                  allocator_,
                                                                                  is_free_buffer_after_forward_,
                                                                                  is_qk_buf_float_,
                                                                                  sparse_);

    ffn_layer_ = new TensorParallelGeluFfnFP8Layer<T1, T2>(inter_size_,
                                                           tensor_para_,
                                                           2,  // fp8_mode
                                                           stream_,
                                                           cublas_wrapper_,
                                                           allocator_,
                                                           is_free_buffer_after_forward_,
                                                           sparse_);
}

template<typename T1, typename T2>
void GptFP8ContextDecoder<T1, T2>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T1, typename T2>
void GptFP8ContextDecoder<T1, T2>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    decoder_normed_input_ = reinterpret_cast<T1*>(
        allocator_->reMalloc(decoder_normed_input_, sizeof(T1) * batch_size * seq_len * hidden_units_, false));
    self_attn_output_ = reinterpret_cast<T2*>(
        allocator_->reMalloc(self_attn_output_, sizeof(T2) * batch_size * seq_len * hidden_units_, false));
    normed_self_attn_output_ = decoder_normed_input_;  // reuse the buffer
    decoder_layer_output_    = reinterpret_cast<T2*>(
        allocator_->reMalloc(decoder_layer_output_, sizeof(T2) * batch_size * seq_len * hidden_units_, false));
    is_allocate_buffer_ = true;
}

template<typename T1, typename T2>
void GptFP8ContextDecoder<T1, T2>::freeBuffer()
{
    if (is_allocate_buffer_) {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        allocator_->free((void**)(&decoder_normed_input_));
        allocator_->free((void**)(&self_attn_output_));
        allocator_->free((void**)(&decoder_layer_output_));
        is_allocate_buffer_ = false;
    }
}

template<typename T1, typename T2>
bool GptFP8ContextDecoder<T1, T2>::isValidLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l >= local_num_layer * pipeline_para_.rank_)
           && (l < local_num_layer * (pipeline_para_.rank_ + 1));
}

template<typename T1, typename T2>
bool GptFP8ContextDecoder<T1, T2>::isFirstLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * pipeline_para_.rank_);
}

template<typename T1, typename T2>
bool GptFP8ContextDecoder<T1, T2>::isLastLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * (pipeline_para_.rank_ + 1) - 1);
}

template<typename T1, typename T2>
int GptFP8ContextDecoder<T1, T2>::getFirstLayerParallelId()
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return local_num_layer * pipeline_para_.rank_;
}

template<typename T1, typename T2>
GptFP8ContextDecoder<T1, T2>::GptFP8ContextDecoder(size_t           head_num,
                                                   size_t           size_per_head,
                                                   size_t           inter_size,
                                                   size_t           num_layer,
                                                   NcclParam        tensor_para,
                                                   NcclParam        pipeline_para,
                                                   cudaStream_t     stream,
                                                   cublasMMWrapper* cublas_wrapper,
                                                   IAllocator*      allocator,
                                                   bool             is_free_buffer_after_forward,
                                                   bool             is_qk_buf_float,
                                                   bool             sparse):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    hidden_units_(head_num_ * size_per_head),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para),
    is_qk_buf_float_(is_qk_buf_float)
{
    initialize();
}

template<typename T1, typename T2>
GptFP8ContextDecoder<T1, T2>::GptFP8ContextDecoder(GptFP8ContextDecoder<T1, T2> const& decoder):
    BaseLayer(decoder.stream_, decoder.cublas_wrapper_, decoder.allocator_, decoder.is_free_buffer_after_forward_),
    head_num_(decoder.head_num_),
    size_per_head_(decoder.size_per_head_),
    inter_size_(decoder.inter_size_),
    num_layer_(decoder.num_layer_),
    hidden_units_(decoder.hidden_units_),
    tensor_para_(decoder.tensor_para_),
    pipeline_para_(decoder.pipeline_para_),
    is_qk_buf_float_(decoder.is_qk_buf_float_)
{
    initialize();
}

template<typename T1, typename T2>
GptFP8ContextDecoder<T1, T2>::~GptFP8ContextDecoder()
{
    delete self_attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T1, typename T2>
void GptFP8ContextDecoder<T1, T2>::forward(
    std::vector<Tensor>*                                  output_tensors,
    const std::vector<Tensor>*                            input_tensors,
    const std::vector<GptFP8DecoderLayerWeight<T1, T2>*>* gpt_decoder_layer_weight)
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

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() == 3);
    FT_CHECK(output_tensors->size() == 4);
    const size_t batch_size = (size_t)input_tensors->at(0).shape[0];
    const size_t seq_len    = (size_t)input_tensors->at(0).shape[1];
    allocateBuffer(batch_size, seq_len);

    const DataType data_type           = getTensorType<T1>();
    const DataType high_precision_type = getTensorType<T2>();

    const size_t local_batch_size = getLocalBatchSize(batch_size, seq_len, pipeline_para_.world_size_);
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
            T2*        decoder_input =
                (l == 0) ? (T2*)(input_tensors->at(0).data) + (int)(ite * local_batch_size * seq_len * hidden_units_) :
                                  decoder_layer_output_;
            T2* decoder_output =
                (l == (num_layer_ - 1)) ?
                    (T2*)(output_tensors->at(0).data) + (int)(ite * local_batch_size * seq_len * hidden_units_) :
                    decoder_layer_output_;

            if (isFirstLayerParallelId(l) && pipeline_para_.rank_ != 0) {
                const int data_size = local_batch_size * seq_len * hidden_units_ / tensor_para_.world_size_;
                ftNcclRecv(decoder_input + data_size * tensor_para_.rank_,
                           data_size,
                           pipeline_para_.rank_ - 1,
                           pipeline_para_,
                           stream_);
                if (tensor_para_.world_size_ > 1) {
                    ftNcclAllGather(decoder_input, decoder_input, data_size, tensor_para_.rank_, tensor_para_, stream_);
                }
            }

            {
                FP8LayerNormParam<T1, T2> param{
                    decoder_normed_input_,
                    decoder_input,
                    gpt_decoder_layer_weight->at(l)->pre_layernorm_weights.gamma,
                    gpt_decoder_layer_weight->at(l)->pre_layernorm_weights.beta,
                    gpt_decoder_layer_weight->at(l)->identity_scale,
                    gpt_decoder_layer_weight->at(l)->self_attention_weights.query_weight.input_scale_inv,
                    local_batch_size * seq_len,
                    hidden_units_,
                    stream_,
                    true};
                invokeFP8LayerNorm<T1, T2, 0>(param);
                sync_check_cuda_error();
            }
            TensorMap self_attention_input_tensors{
                {"input_query",
                 Tensor{MEMORY_GPU, data_type, {local_batch_size * seq_len, hidden_units_}, decoder_normed_input_}},
                {"attention_mask",
                 Tensor{MEMORY_GPU,
                        data_type,
                        {local_batch_size, 1, seq_len, seq_len},
                        (const T1*)input_tensors->at(1).data + local_batch_size * ite * seq_len * seq_len}},
                {"is_final_layer", Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &is_final}}};

            size_t cache_offset = l - getFirstLayerParallelId();
            for (auto t = output_tensors->at(1).shape.begin() + 1; t != output_tensors->at(1).shape.end(); ++t) {
                cache_offset *= *t;
            };
            size_t ite_cache_offset = ite * local_batch_size;
            for (auto t = output_tensors->at(1).shape.begin() + 2; t != output_tensors->at(1).shape.end(); ++t) {
                ite_cache_offset *= *t;
            }
            cache_offset += ite_cache_offset;

#ifdef FP8_MHA
            TensorMap self_attention_output_tensors{
                {"input_query",
                 Tensor{
                     MEMORY_GPU, high_precision_type, {local_batch_size * seq_len, hidden_units_}, self_attn_output_}},
                {"attention_mask",
                 Tensor{
                     MEMORY_GPU, data_type, self_k_cache_size, ((const T1*)output_tensors->at(1).data) + cache_offset}},
                {"is_final_layer",
                 Tensor{MEMORY_GPU,
                        data_type,
                        self_v_cache_size,
                        ((const T1*)output_tensors->at(2).data) + cache_offset}}};
#else
            TensorMap self_attention_output_tensors{
                {"attention_out",
                 Tensor{
                     MEMORY_GPU, high_precision_type, {local_batch_size * seq_len, hidden_units_}, self_attn_output_}},
                {"key_cache",
                 Tensor{MEMORY_GPU,
                        high_precision_type,
                        self_k_cache_size,
                        ((const T2*)output_tensors->at(1).data) + cache_offset}},
                {"value_cache",
                 Tensor{MEMORY_GPU,
                        high_precision_type,
                        self_v_cache_size,
                        ((const T2*)output_tensors->at(2).data) + cache_offset}}};
#endif
            self_attention_layer_->forward(
                &self_attention_output_tensors,
                &self_attention_input_tensors,
                (const AttentionWeight<T1>*)&gpt_decoder_layer_weight->at(l)->self_attention_weights);

            if (is_final == false) {
                {
                    GeneralFP8AddBiasResidualPreLayerNormParam<T1, T2> param{
                        normed_self_attn_output_,
                        self_attn_output_,
                        decoder_input,
                        gpt_decoder_layer_weight->at(l)->self_attention_weights.attention_output_weight.bias,
                        gpt_decoder_layer_weight->at(l)->self_attn_layernorm_weights.gamma,
                        gpt_decoder_layer_weight->at(l)->self_attn_layernorm_weights.beta,
                        nullptr,  // gpt_decoder_layer_weight->at(l)->self_attention_weights.attention_output_weight.weight_scale,
                        gpt_decoder_layer_weight->at(l)->ffn_weights.intermediate_weight.input_scale_inv,
                        local_batch_size * seq_len,
                        hidden_units_,
                        stream_,
                        true};
                    invokeGeneralFP8AddBiasResidualPreLayerNorm<T1, T2, 0>(param);
                    sync_check_cuda_error();
                }

                TensorMap ffn_input_tensors  = TensorMap(std::unordered_map<std::string, Tensor>{
                     {"input_hidden_state",
                      Tensor{MEMORY_GPU,
                            data_type,
                            std::vector<size_t>{local_batch_size * seq_len, hidden_units_},
                            normed_self_attn_output_}}});
                TensorMap ffn_output_tensors = TensorMap(std::unordered_map<std::string, Tensor>{
                    {"output_hidden_state",
                     Tensor{MEMORY_GPU,
                            high_precision_type,
                            std::vector<size_t>{local_batch_size * seq_len, hidden_units_},
                            decoder_output}}});

                ffn_layer_->forward(
                    &ffn_output_tensors, &ffn_input_tensors, &gpt_decoder_layer_weight->at(l)->ffn_weights);
                invokeAddBiasResidual(decoder_output,
                                      self_attn_output_,
                                      gpt_decoder_layer_weight->at(l)->ffn_weights.output_weight.bias,
                                      local_batch_size * seq_len,
                                      hidden_units_,
                                      stream_);
                sync_check_cuda_error();

                if (isLastLayerParallelId(l) == true && pipeline_para_.rank_ != pipeline_para_.world_size_ - 1) {
                    const int data_size = local_batch_size * seq_len * hidden_units_ / tensor_para_.world_size_;
                    ftNcclSend(decoder_output + data_size * tensor_para_.rank_,
                               data_size,
                               pipeline_para_.rank_ + 1,
                               pipeline_para_,
                               stream_);
                }
            }
        }
    }

    // TODO(bhsueh) We could optimize this point by only computing the last token for the last layer
    invokeLookupHiddenStateOfLastToken((T2*)output_tensors->at(3).data,
                                       (T2*)output_tensors->at(0).data,
                                       (int*)input_tensors->at(2).data,
                                       seq_len,
                                       batch_size,
                                       hidden_units_,
                                       stream_);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template class GptFP8ContextDecoder<__nv_fp8_e4m3, __nv_bfloat16>;

}  // namespace fastertransformer