/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/llama/LlamaDecoder.h"
#include "src/fastertransformer/layers/TensorParallelSiluFfnLayer.h"
#include "src/fastertransformer/layers/attention_layers/TensorParallelDecoderSelfAttentionLayer.h"

namespace fastertransformer {

template<typename T>
void LlamaDecoder<T>::initialize()
{
    self_attention_layer_ = new TensorParallelDecoderSelfAttentionLayer<T>(0,  // max_batch_size
                                                                           head_num_,
                                                                           size_per_head_,
                                                                           rotary_embedding_dim_,
                                                                           neox_rotary_style_,
                                                                           tensor_para_,
                                                                           stream_,
                                                                           cublas_wrapper_,
                                                                           allocator_,
                                                                           !use_gptj_residual_,
                                                                           is_free_buffer_after_forward_,
                                                                           false,
                                                                           0,
                                                                           custom_all_reduce_comm_,
                                                                           enable_custom_all_reduce_);

    ffn_layer_ = new TensorParallelSiluFfnLayer<T>(0,  // max_batch_size
                                                   1,
                                                   head_num_,
                                                   size_per_head_,
                                                   0,  // expert_num
                                                   inter_size_,
                                                   tensor_para_,
                                                   stream_,
                                                   cublas_wrapper_,
                                                   allocator_,
                                                   !use_gptj_residual_,
                                                   is_free_buffer_after_forward_,
                                                   false,
                                                   true,  // use_gated_activation = true;
                                                   custom_all_reduce_comm_,
                                                   enable_custom_all_reduce_);
}

template<typename T>
void LlamaDecoder<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void LlamaDecoder<T>::allocateBuffer(size_t batch_size)
{
    decoder_normed_input_ = reinterpret_cast<T*>(
        allocator_->reMalloc(decoder_normed_input_, sizeof(T) * batch_size * hidden_units_, false));
    self_attn_output_ =
        reinterpret_cast<T*>(allocator_->reMalloc(self_attn_output_, sizeof(T) * batch_size * hidden_units_, false));
    ffn_output_ =
        reinterpret_cast<T*>(allocator_->reMalloc(ffn_output_, sizeof(T) * batch_size * hidden_units_, false));
    decoder_layer_output_ = reinterpret_cast<T*>(
        allocator_->reMalloc(decoder_layer_output_, sizeof(T) * batch_size * hidden_units_, false));
    is_allocate_buffer_ = true;
}

template<typename T>
void LlamaDecoder<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free((void**)(&decoder_normed_input_));
        allocator_->free((void**)(&self_attn_output_));
        allocator_->free((void**)(&ffn_output_));
        allocator_->free((void**)(&decoder_layer_output_));
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool LlamaDecoder<T>::isValidLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l >= local_num_layer * pipeline_para_.rank_)
           && (l < local_num_layer * (pipeline_para_.rank_ + 1));
}

template<typename T>
bool LlamaDecoder<T>::isFirstLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * pipeline_para_.rank_);
}

template<typename T>
bool LlamaDecoder<T>::isLastLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * (pipeline_para_.rank_ + 1) - 1);
}

template<typename T>
int LlamaDecoder<T>::getFirstLayerParallelId()
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return local_num_layer * pipeline_para_.rank_;
}

template<typename T>
LlamaDecoder<T>::LlamaDecoder(size_t                              head_num,
                              size_t                              size_per_head,
                              size_t                              inter_size,
                              size_t                              num_layer,
                              size_t                              rotary_embedding_dim,
                              bool                                neox_rotary_style,
                              bool                                use_gptj_residual,
                              float                               layernorm_eps,
                              NcclParam                           tensor_para,
                              NcclParam                           pipeline_para,
                              cudaStream_t                        stream,
                              cublasMMWrapper*                    cublas_wrapper,
                              IAllocator*                         allocator,
                              bool                                is_free_buffer_after_forward,
                              std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                              int                                 enable_custom_all_reduce):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    rotary_embedding_dim_(rotary_embedding_dim),
    neox_rotary_style_(neox_rotary_style),
    use_gptj_residual_(use_gptj_residual),
    layernorm_eps_(layernorm_eps),
    hidden_units_(head_num_ * size_per_head),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce)
{
    initialize();
}

template<typename T>
LlamaDecoder<T>::LlamaDecoder(LlamaDecoder<T> const& decoder):
    BaseLayer(decoder.stream_, decoder.cublas_wrapper_, decoder.allocator_, decoder.is_free_buffer_after_forward_),
    head_num_(decoder.head_num_),
    size_per_head_(decoder.size_per_head_),
    inter_size_(decoder.inter_size_),
    num_layer_(decoder.num_layer_),
    rotary_embedding_dim_(decoder.rotary_embedding_dim_),
    neox_rotary_style_(decoder.neox_rotary_style_),
    use_gptj_residual_(decoder.use_gptj_residual_),
    layernorm_eps_(decoder.layernorm_eps_),
    hidden_units_(decoder.hidden_units_),
    tensor_para_(decoder.tensor_para_),
    pipeline_para_(decoder.pipeline_para_),
    custom_all_reduce_comm_(decoder.custom_all_reduce_comm_),
    enable_custom_all_reduce_(decoder.enable_custom_all_reduce_)
{
    initialize();
}

template<typename T>
LlamaDecoder<T>::~LlamaDecoder()
{
    delete self_attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void LlamaDecoder<T>::forward(std::vector<Tensor>*                              output_tensors,
                              const std::vector<Tensor>*                        input_tensors,
                              const std::vector<LlamaDecoderLayerWeight<T>*>*   gpt_decoder_layer_weight)
{
    FT_CHECK(false);
}

template<typename T>
void LlamaDecoder<T>::forward(std::unordered_map<std::string, Tensor>*          output_tensors,
                              const std::unordered_map<std::string, Tensor>*    input_tensors,
                              const std::vector<LlamaDecoderLayerWeight<T>*>* gpt_decoder_layer_weight)
{
    // input tensors:
    //      decoder_input [local_batch_size, hidden_dimension],
    //      finished [local_batch_size],
    //      sequence_lengths [local_batch_size]
    //      total_padding_tokens [local_batch_size],
    //      max_input_length [1] on cpu
    //      d_prefix_prompt_lengths [local_batch_size], on GPU
    //      max_prefix_prompt_length [1] on cpu
    //      step [1] on cpu
    //      ite [1] on cpu
    //      cache_indirection [local_batch_size / beam_width, beam_width, memory_len]
    //              Here, local_batch_size contains the beam_width, so local_batch_size / beam_width
    //              is real local_batch_size.
    //      masked_tokens[local_batch_size, memory_len]

    // output tensors:
    //      decoder_output [local_batch_size, hidden_dimension],
    //      key_cache [num_layer, batch_size, head_num, size_per_head // x, memory_len, x]
    //      value_cache [num_layer, batch_size, head_num, memory_len, size_per_head]

    FT_CHECK(input_tensors->size() == 11);
    FT_CHECK(output_tensors->size() == 3);

    const DataType data_type        = getTensorType<T>();
    const size_t   local_batch_size = input_tensors->at("decoder_input").shape[0];
    allocateBuffer(local_batch_size);
    const int ite = input_tensors->at("ite").getVal<const int>();

    T* decoder_input  = input_tensors->at("decoder_input").getPtr<T>();
    T* decoder_output = output_tensors->at("decoder_output").getPtr<T>();

    Tensor&             k_cache = output_tensors->at("key_cache");
    Tensor&             v_cache = output_tensors->at("value_cache");
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

    for (uint l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l) == false) {
            continue;
        }
        T* layer_input  = (l == 0) ? decoder_input : decoder_layer_output_;
        T* layer_output = (l == num_layer_ - 1) ? decoder_output : decoder_layer_output_;

        if (isFirstLayerParallelId(l) == true && pipeline_para_.rank_ != 0 && pipeline_para_.world_size_ > 1) {
            int data_size = local_batch_size * hidden_units_ / tensor_para_.world_size_;
            // ftNcclRecv(layer_input, local_batch_size * hidden_units_, pipeline_para_.rank_ - 1, pipeline_para_,
            // stream_);

            ftNcclRecv(layer_input + data_size * tensor_para_.rank_,
                       data_size,
                       pipeline_para_.rank_ - 1,
                       pipeline_para_,
                       stream_);
            if (tensor_para_.world_size_ > 1) {
                ftNcclAllGather(layer_input, layer_input, data_size, tensor_para_.rank_, tensor_para_, stream_);
            }
        }

        invokeGeneralT5LayerNorm(decoder_normed_input_,
                                 layer_input,
                                 gpt_decoder_layer_weight->at(l)->pre_layernorm_weights.gamma,
                                 (const T*)nullptr,
                                 layernorm_eps_,
                                 local_batch_size,
                                 hidden_units_,
                                 stream_);
        sync_check_cuda_error();

        TensorMap self_attention_input_tensors(*input_tensors);
        self_attention_input_tensors.insert(
            "input_query", Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, decoder_normed_input_});

        size_t cache_offset = l - getFirstLayerParallelId();
        for (auto t = k_cache.shape.begin() + 1; t != k_cache.shape.end(); ++t) {
            cache_offset *= *t;
        };
        size_t ite_cache_offset = ite * local_batch_size;
        for (auto t = k_cache.shape.begin() + 2; t != k_cache.shape.end(); ++t) {
            ite_cache_offset *= *t;
        }
        cache_offset += ite_cache_offset;

        TensorMap self_attention_output_tensors{
            {"hidden_features", Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, self_attn_output_}},
            {"key_cache", Tensor{MEMORY_GPU, data_type, self_k_cache_size, k_cache.getPtrWithOffset(cache_offset)}},
            {"value_cache", Tensor{MEMORY_GPU, data_type, self_v_cache_size, v_cache.getPtrWithOffset(cache_offset)}}};

        self_attention_layer_->forward(&self_attention_output_tensors,
                                       &self_attention_input_tensors,
                                       &gpt_decoder_layer_weight->at(l)->self_attention_weights);
        if (use_gptj_residual_) {
            invokeGeneralLayerNorm(decoder_normed_input_,
                                   layer_input,
                                   gpt_decoder_layer_weight->at(l)->post_attention_layernorm_weights.gamma,
                                   gpt_decoder_layer_weight->at(l)->post_attention_layernorm_weights.beta,
                                   layernorm_eps_,
                                   local_batch_size,
                                   hidden_units_,
                                   (float*)nullptr,
                                   0,
                                   stream_);
        }
        else {
            invokeGeneralAddResidualT5PreLayerNorm(
                self_attn_output_,
                decoder_normed_input_,
                layer_input,
                gpt_decoder_layer_weight->at(l)->post_attention_layernorm_weights.gamma,
                layernorm_eps_,
                local_batch_size,
                hidden_units_,
                stream_);
        }

        TensorMap ffn_input_tensors(
            {{"ffn_input", Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, decoder_normed_input_}}});
        TensorMap ffn_output_tensors({{"ffn_output",
                                       Tensor{MEMORY_GPU,
                                              data_type,
                                              {local_batch_size, hidden_units_},
                                              use_gptj_residual_ ? ffn_output_ : layer_output}}});
        ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &gpt_decoder_layer_weight->at(l)->ffn_weights);

        if (use_gptj_residual_) {
            // Original workflow:
            //      layer_output = layer_input + reduceSum(ffn_output + self_attn_output + ffn_output_bias)
            // Our workflow:
            //      layer_output = reduceSum(ffn_output + self_attn_output + ffn_output_bias + layer_input / TP_size)
            // They are equivalent on math, but we can use same buffer for layer_input and layer_output
            invokeAddBiasAttentionFfnResidual(layer_output,
                                              ffn_output_,
                                              self_attn_output_,
                                              layer_input,
                                              gpt_decoder_layer_weight->at(l)->ffn_weights.output_weight.bias,
                                              local_batch_size,
                                              hidden_units_,
                                              tensor_para_.world_size_,
                                              stream_);
            if (tensor_para_.world_size_ > 1) {
                ftNcclAllReduceSum(layer_output, layer_output, local_batch_size * hidden_units_, tensor_para_, stream_);
            }
        }
        else {
            invokeAddBiasResidual(layer_output,
                                  self_attn_output_,
                                  gpt_decoder_layer_weight->at(l)->ffn_weights.output_weight.bias,
                                  local_batch_size,
                                  hidden_units_,
                                  stream_);
        }

        sync_check_cuda_error();

        if (isLastLayerParallelId(l) == true && pipeline_para_.rank_ != pipeline_para_.world_size_ - 1
            && pipeline_para_.world_size_ > 1) {
            int data_size = local_batch_size * hidden_units_ / tensor_para_.world_size_;
            // ftNcclSend(layer_output, local_batch_size * hidden_units_, pipeline_para_.rank_ + 1, pipeline_para_,
            // stream_);

            ftNcclSend(layer_output + data_size * tensor_para_.rank_,
                       data_size,
                       pipeline_para_.rank_ + 1,
                       pipeline_para_,
                       stream_);
        }
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template class LlamaDecoder<float>;
template class LlamaDecoder<half>;
#ifdef ENABLE_BF16
template class LlamaDecoder<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
