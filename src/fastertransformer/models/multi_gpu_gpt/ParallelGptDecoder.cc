/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2022, SK Telecom Authored by A. Dialog
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
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    self_attention_layer_ = new TensorParallelDecoderSelfAttentionLayer<T>(max_batch_size_,
                                                                           head_num_,
                                                                           size_per_head_,
                                                                           tensor_para_,
                                                                           stream_,
                                                                           cublas_wrapper_,
                                                                           allocator_,
                                                                           true,
                                                                           is_free_buffer_after_forward_,
                                                                           sparse_,
                                                                           int8_mode_,
                                                                           custom_all_reduce_comm_,
                                                                           enable_custom_all_reduce_);

    bool use_gated_activation = activation_type_ == ActivationType::GeGLU || activation_type_ == ActivationType::ReGLU;
    size_t max_inter_size     = has_adapters_ ? std::max(inter_size_, adapter_inter_size_) : inter_size_;
    if (activation_type_ == ActivationType::Gelu || activation_type_ == ActivationType::GeGLU) {
        ffn_layer_ = new TensorParallelGeluFfnLayer<T>(max_batch_size_,
                                                       1,
                                                       head_num_,
                                                       size_per_head_,
                                                       0,  // expert_num
                                                       max_inter_size,
                                                       tensor_para_,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       true,
                                                       is_free_buffer_after_forward_,
                                                       sparse_,
                                                       int8_mode_,
                                                       use_gated_activation,
                                                       custom_all_reduce_comm_,
                                                       enable_custom_all_reduce_);
    }
    else if (activation_type_ == ActivationType::Relu || activation_type_ == ActivationType::ReGLU) {
        ffn_layer_ = new TensorParallelReluFfnLayer<T>(max_batch_size_,
                                                       1,
                                                       head_num_,
                                                       size_per_head_,
                                                       0,  // expert_num
                                                       max_inter_size,
                                                       tensor_para_,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       true,
                                                       is_free_buffer_after_forward_,
                                                       sparse_,
                                                       int8_mode_,
                                                       use_gated_activation,
                                                       custom_all_reduce_comm_,
                                                       enable_custom_all_reduce_);
    }
}

template<typename T>
ParallelGptDecoder<T>::ParallelGptDecoder(size_t                              max_batch_size,
                                          size_t                              head_num,
                                          size_t                              size_per_head,
                                          size_t                              inter_size,
                                          size_t                              num_layer,
                                          float                               layernorm_eps,
                                          gptVariantParams                    gpt_variant_params,
                                          NcclParam                           tensor_para,
                                          NcclParam                           pipeline_para,
                                          cudaStream_t                        stream,
                                          cublasMMWrapper*                    cublas_wrapper,
                                          IAllocator*                         allocator,
                                          bool                                is_free_buffer_after_forward,
                                          bool                                sparse,
                                          int                                 int8_mode,
                                          std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                                          int                                 enable_custom_all_reduce):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    max_batch_size_(max_batch_size),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    layernorm_eps_(layernorm_eps),
    layernorm_type_(gpt_variant_params.layernorm_type),
    activation_type_(gpt_variant_params.activation_type),
    adapter_inter_size_(gpt_variant_params.adapter_inter_size),
    has_adapters_(gpt_variant_params.has_adapters),
    hidden_units_(head_num_ * size_per_head_),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para),
    int8_mode_(int8_mode),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce)
{
    initialize();
}

template<typename T>
ParallelGptDecoder<T>::ParallelGptDecoder(ParallelGptDecoder<T> const& decoder):
    BaseLayer(decoder.stream_,
              decoder.cublas_wrapper_,
              decoder.allocator_,
              decoder.is_free_buffer_after_forward_,
              decoder.cuda_device_prop_,
              decoder.sparse_),
    max_batch_size_(decoder.max_batch_size_),
    head_num_(decoder.head_num_),
    size_per_head_(decoder.size_per_head_),
    inter_size_(decoder.inter_size_),
    num_layer_(decoder.num_layer_),
    layernorm_eps_(decoder.layernorm_eps_),
    layernorm_type_(decoder.layernorm_type_),
    activation_type_(decoder.activation_type_),
    adapter_inter_size_(decoder.adapter_inter_size_),
    has_adapters_(decoder.has_adapters_),
    hidden_units_(decoder.hidden_units_),
    tensor_para_(decoder.tensor_para_),
    pipeline_para_(decoder.pipeline_para_),
    int8_mode_(decoder.int8_mode_),
    custom_all_reduce_comm_(decoder.custom_all_reduce_comm_),
    enable_custom_all_reduce_(decoder.enable_custom_all_reduce_)
{
    initialize();
}

template<typename T>
void ParallelGptDecoder<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void ParallelGptDecoder<T>::allocateBuffer(size_t batch_size)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    decoder_layer_output_ = reinterpret_cast<T*>(
        allocator_->reMalloc(decoder_layer_output_, sizeof(T) * batch_size * hidden_units_, false));
    if (int8_mode_ == 2) {
        decoder_layer_output_int32_ = reinterpret_cast<int32_t*>(
            allocator_->reMalloc(decoder_layer_output_int32_, sizeof(int32_t) * batch_size * hidden_units_, false));
        self_attn_output_int32_ = reinterpret_cast<int32_t*>(
            allocator_->reMalloc(self_attn_output_int32_, sizeof(int32_t) * batch_size * hidden_units_, false));
    }
    decoder_normed_input_ = reinterpret_cast<T*>(
        allocator_->reMalloc(decoder_normed_input_, sizeof(T) * batch_size * hidden_units_, false));
    self_attn_output_ =
        reinterpret_cast<T*>(allocator_->reMalloc(self_attn_output_, sizeof(T) * batch_size * hidden_units_, false));
    normed_self_attn_output_ = reinterpret_cast<T*>(
        allocator_->reMalloc(normed_self_attn_output_, sizeof(T) * batch_size * hidden_units_, false));
    // only allocate additionl buffers when has adapters
    after_adapter_attn_output_ = has_adapters_ ? reinterpret_cast<T*>(allocator_->reMalloc(
                                     after_adapter_attn_output_, sizeof(T) * batch_size * hidden_units_, false)) :
                                                 self_attn_output_;
    is_allocate_buffer_        = true;
}

template<typename T>
void ParallelGptDecoder<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        allocator_->free((void**)(&decoder_layer_output_));
        allocator_->free((void**)(&decoder_normed_input_));
        allocator_->free((void**)(&self_attn_output_));
        allocator_->free((void**)(&normed_self_attn_output_));
        if (has_adapters_) {
            allocator_->free((void**)(&after_adapter_attn_output_));
        }
        if (int8_mode_ == 2) {
            allocator_->free((void**)(&self_attn_output_int32_));
            allocator_->free((void**)(&decoder_layer_output_int32_));
        }
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool ParallelGptDecoder<T>::isValidLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l >= local_num_layer * pipeline_para_.rank_)
           && (l < local_num_layer * (pipeline_para_.rank_ + 1));
}

template<typename T>
bool ParallelGptDecoder<T>::isFirstLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * pipeline_para_.rank_);
}

template<typename T>
bool ParallelGptDecoder<T>::isLastLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * (pipeline_para_.rank_ + 1) - 1);
}

template<typename T>
int ParallelGptDecoder<T>::getFirstLayerParallelId()
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return local_num_layer * pipeline_para_.rank_;
}

template<typename T>
ParallelGptDecoder<T>::~ParallelGptDecoder()
{
    delete self_attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void ParallelGptDecoder<T>::forward(std::unordered_map<std::string, Tensor>*              output_tensors,
                                    const std::unordered_map<std::string, Tensor>*        input_tensors,
                                    const std::vector<ParallelGptDecoderLayerWeight<T>*>* gpt_decoder_layer_weight)
{
    // input tensors:
    //      decoder_input [local_batch_size, hidden_dimension],
    //      finished [local_batch_size],
    //      input_lengths [local_batch_size],
    //      total_padding_tokens [local_batch_size]
    //      max_input_length [1] on cpu
    //      step [1] on cpu
    //      ite [1] on cpu
    //      cache_indirection [local_batch_size / beam_width, beam_width, memory_len]
    //          Here, local_batch_size contains the beam_width, so local_batch_size / beam_width
    //          is real local_batch_size. (optional.)
    //      masked_tokens [local_batch_size, memory_len]
    //      linear_bias_slopes [head_num], optional

    // output tensors:
    //      decoder_output [local_batch_size, hidden_dimension],
    //      key_cache [num_layer, batch_size, head_num, size_per_head // x, memory_len, x]
    //      value_cache [num_layer, batch_size, head_num, memory_len, size_per_head]

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    FT_CHECK(input_tensors->count("decoder_input"));
    FT_CHECK(input_tensors->count("finished"));
    FT_CHECK(input_tensors->count("input_lengths"));
    FT_CHECK(input_tensors->count("total_padding_tokens"));
    FT_CHECK(input_tensors->count("max_input_length"));
    FT_CHECK(input_tensors->count("step"));
    FT_CHECK(input_tensors->count("ite"));
    FT_CHECK(input_tensors->count("masked_tokens"));
    FT_CHECK(output_tensors->count("decoder_output"));
    FT_CHECK(output_tensors->count("key_cache"));
    FT_CHECK(output_tensors->count("value_cache"));

    const size_t local_batch_size = input_tensors->at("decoder_input").shape[0];
    allocateBuffer(local_batch_size);

    const DataType data_type = getTensorType<T>();

    const int ite = input_tensors->at("ite").getVal<int>();

    Tensor k_cache = output_tensors->at("key_cache");
    Tensor v_cache = output_tensors->at("value_cache");

    // The resize of the key cache buffer by
    //   (local_batch_size, local_head_num, size_per_head // x, max_seq_len, x) where x is constant.
    std::vector<size_t> self_k_cache_size(k_cache.shape.begin() + 2, k_cache.shape.end());
    self_k_cache_size.insert(self_k_cache_size.begin(), local_batch_size);

    // The resize of the value cache buffer by (local_batch_size, local_head_num, max_seq_len, size_per_head).
    std::vector<size_t> self_v_cache_size(v_cache.shape.begin() + 2, v_cache.shape.end());
    self_v_cache_size.insert(self_v_cache_size.begin(), local_batch_size);

    const auto activation_in_type  = int8_mode_ == 2 ? TYPE_INT8 : data_type;
    const auto activation_out_type = int8_mode_ == 2 ? TYPE_INT32 : data_type;

    for (uint l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l) == false) {
            continue;
        }
        T* decoder_input = (l == 0) ? input_tensors->at("decoder_input").getPtr<T>() : decoder_layer_output_;
        T* decoder_output =
            (l == num_layer_ - 1) ? output_tensors->at("decoder_output").getPtr<T>() : decoder_layer_output_;

        if (isFirstLayerParallelId(l) == true && pipeline_para_.rank_ != 0 && pipeline_para_.world_size_ > 1) {
            size_t data_size = local_batch_size * hidden_units_ / tensor_para_.world_size_;
            ftNcclRecv(decoder_input + data_size * tensor_para_.rank_,
                       data_size,
                       pipeline_para_.rank_ - 1,
                       pipeline_para_,
                       stream_);
            if (tensor_para_.world_size_ > 1) {
                ftNcclAllGather(decoder_input, decoder_input, data_size, tensor_para_.rank_, tensor_para_, stream_);
            }
        }

        ParallelGptDecoderLayerWeight<T>* layer_weight = gpt_decoder_layer_weight->at(l);

        if (layernorm_type_ == LayerNormType::pre_layernorm) {
            invokeGeneralLayerNorm(decoder_normed_input_,
                                   decoder_input,
                                   layer_weight->pre_layernorm_weights.gamma,
                                   layer_weight->pre_layernorm_weights.beta,
                                   layernorm_eps_,
                                   local_batch_size,
                                   hidden_units_,
                                   layer_weight->self_attention_weights.query_weight.scale,
                                   int8_mode_,
                                   stream_);
        }
        sync_check_cuda_error();

        TensorMap self_attention_input_tensors{
            {"input_query",
             Tensor{MEMORY_GPU,
                    activation_in_type,
                    {local_batch_size, hidden_units_},
                    layernorm_type_ == LayerNormType::pre_layernorm ? decoder_normed_input_ : decoder_input}},
            {"finished", input_tensors->at("finished")},
            {"sequence_lengths", input_tensors->at("input_lengths")},
            {"total_padding_tokens", input_tensors->at("total_padding_tokens")},
            {"max_input_length", input_tensors->at("max_input_length")},
            {"step", input_tensors->at("step")},
            {"masked_tokens", input_tensors->at("masked_tokens")}};
        if (input_tensors->count("cache_indirection")) {
            self_attention_input_tensors.insert("cache_indirection", input_tensors->at("cache_indirection"));
        }
        if (input_tensors->count("linear_bias_slopes")) {
            self_attention_input_tensors.insert("linear_bias_slopes", input_tensors->at("linear_bias_slopes"));
        }

        size_t cache_offset = l - getFirstLayerParallelId();
        for (auto t = k_cache.shape.begin() + 1; t != k_cache.shape.end(); ++t) {
            cache_offset *= *t;
        };
        size_t ite_cache_offset = ite * local_batch_size;
        for (auto t = k_cache.shape.begin() + 2; t != k_cache.shape.end(); ++t) {
            ite_cache_offset *= *t;
        }
        cache_offset += ite_cache_offset;

        T*        attention_out = (int8_mode_ == 2) ? reinterpret_cast<T*>(self_attn_output_int32_) : self_attn_output_;
        TensorMap self_attention_output_tensors{
            {"hidden_features",
             Tensor(MEMORY_GPU, activation_out_type, {local_batch_size, hidden_units_}, attention_out)},
            {"key_cache", Tensor(MEMORY_GPU, data_type, self_k_cache_size, k_cache.getPtrWithOffset<T>(cache_offset))},
            {"value_cache",
             Tensor(MEMORY_GPU, data_type, self_v_cache_size, v_cache.getPtrWithOffset<T>(cache_offset))}};

        self_attention_layer_->forward(
            &self_attention_output_tensors, &self_attention_input_tensors, &layer_weight->self_attention_weights);

        // the adapter after attention
        if (has_adapters_) {
            invokeGenericActivation<IdentityActivation, T, T>(
                self_attn_output_,
                layer_weight->self_attention_weights.attention_output_weight.bias,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                local_batch_size,
                hidden_units_,
                0,
                nullptr,
                nullptr,
                stream_);

            TensorMap ffn_input_tensors(
                {{"ffn_input", Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, self_attn_output_}}});
            TensorMap ffn_output_tensors(
                {{"ffn_output",
                  Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, after_adapter_attn_output_}}});

            ffn_layer_->resetInterSize(adapter_inter_size_ / tensor_para_.world_size_);
            ffn_layer_->forward(
                &ffn_output_tensors, &ffn_input_tensors, &layer_weight->after_attention_adapter_weights);
        }

        if (layernorm_type_ == LayerNormType::pre_layernorm) {
            invokeGeneralAddBiasResidualPreLayerNorm(
                // in case of has_adaptor false isn't it self_attn_output_? i.e.
                //   has_adapters_ ? after_adapter_attn_outpu_ : self_attn_output_,
                has_adapters_ ? after_adapter_attn_output_ : self_attn_output_,
                normed_self_attn_output_,
                has_adapters_ ? after_adapter_attn_output_ : attention_out,
                decoder_input,
                has_adapters_ ? self_attn_output_ : nullptr,
                layer_weight->self_attn_layernorm_weights.gamma,
                layer_weight->self_attn_layernorm_weights.beta,
                has_adapters_ ? layer_weight->after_attention_adapter_weights.output_weight.bias :
                                layer_weight->self_attention_weights.attention_output_weight.bias,
                layernorm_eps_,
                local_batch_size,
                hidden_units_,
                layer_weight->self_attention_weights.attention_output_weight.scale_inter,
                layer_weight->self_attention_weights.attention_output_weight.scale_out,
                layer_weight->ffn_weights.intermediate_weight.scale,
                int8_mode_,
                stream_);
        }
        else if (layernorm_type_ == LayerNormType::post_layernorm) {
            invokeAddBiasResidualLayerNorm(
                // check correctness.
                after_adapter_attn_output_,
                decoder_input,
                has_adapters_ ? layer_weight->after_attention_adapter_weights.output_weight.bias :
                                layer_weight->self_attention_weights.attention_output_weight.bias,
                layer_weight->pre_layernorm_weights.gamma,
                layer_weight->pre_layernorm_weights.beta,
                layernorm_eps_,
                local_batch_size,
                hidden_units_,
                stream_);
        }

        sync_check_cuda_error();

        T* ffn_output_ptr;
        if (int8_mode_ == 2) {
            ffn_output_ptr = reinterpret_cast<T*>(decoder_layer_output_int32_);
        }
        else if (has_adapters_) {
            ffn_output_ptr = self_attn_output_;
        }
        else {
            ffn_output_ptr = decoder_output;
        }

        TensorMap ffn_input_tensors(
            {{"ffn_input",
              Tensor{MEMORY_GPU,
                     activation_in_type,
                     {local_batch_size, hidden_units_},
                     layernorm_type_ == LayerNormType::pre_layernorm ? normed_self_attn_output_ :
                                                                       after_adapter_attn_output_}}});
        TensorMap ffn_output_tensors(
            {{"ffn_output",
              Tensor{MEMORY_GPU, activation_out_type, {local_batch_size, hidden_units_}, ffn_output_ptr}}});

        ffn_layer_->resetInterSize(inter_size_ / tensor_para_.world_size_);
        ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &layer_weight->ffn_weights);

        // the adapter after ffn
        if (has_adapters_) {
            invokeGenericActivation<IdentityActivation, T, T>(ffn_output_ptr,
                                                              layer_weight->ffn_weights.output_weight.bias,
                                                              nullptr,
                                                              nullptr,
                                                              nullptr,
                                                              nullptr,
                                                              local_batch_size,
                                                              hidden_units_,
                                                              0,
                                                              nullptr,
                                                              nullptr,
                                                              stream_);

            TensorMap ffn_input_tensors(
                {{"ffn_input", Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, ffn_output_ptr}}});
            TensorMap ffn_output_tensors(
                {{"ffn_output", Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, decoder_output}}});

            ffn_layer_->resetInterSize(adapter_inter_size_ / tensor_para_.world_size_);
            ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &layer_weight->after_ffn_adapter_weights);
        }

        if (layernorm_type_ == LayerNormType::pre_layernorm) {
            invokeAddBiasResidual(decoder_output,
                                  int8_mode_ == 2 ? reinterpret_cast<T*>(decoder_layer_output_int32_) : decoder_output,
                                  after_adapter_attn_output_,
                                  has_adapters_ ? ffn_output_ptr : nullptr,
                                  has_adapters_ ? layer_weight->after_ffn_adapter_weights.output_weight.bias :
                                                  layer_weight->ffn_weights.output_weight.bias,
                                  int8_mode_ == 2 ? layer_weight->ffn_weights.output_weight.scale_inter : nullptr,
                                  int8_mode_ == 2 ? layer_weight->ffn_weights.output_weight.scale_out : nullptr,
                                  local_batch_size,
                                  hidden_units_,
                                  int8_mode_,
                                  stream_);
        }
        else if (layernorm_type_ == LayerNormType::post_layernorm) {
            invokeAddBiasResidualLayerNorm(decoder_output,
                                           after_adapter_attn_output_,
                                           has_adapters_ ? layer_weight->after_ffn_adapter_weights.output_weight.bias :
                                                           layer_weight->ffn_weights.output_weight.bias,
                                           layer_weight->self_attn_layernorm_weights.gamma,
                                           layer_weight->self_attn_layernorm_weights.beta,
                                           layernorm_eps_,
                                           local_batch_size,
                                           hidden_units_,
                                           stream_);
        }
        sync_check_cuda_error();

        if (isLastLayerParallelId(l) == true && pipeline_para_.rank_ != pipeline_para_.world_size_ - 1
            && pipeline_para_.world_size_ > 1) {

            ftNcclSend(decoder_output
                           + local_batch_size * hidden_units_ / tensor_para_.world_size_ * tensor_para_.rank_,
                       local_batch_size * hidden_units_ / tensor_para_.world_size_,
                       pipeline_para_.rank_ + 1,
                       pipeline_para_,
                       stream_);
        }
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template class ParallelGptDecoder<float>;
template class ParallelGptDecoder<half>;
#ifdef ENABLE_BF16
template class ParallelGptDecoder<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
