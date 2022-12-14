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

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptContextDecoder.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"

namespace fastertransformer {

template<typename T>
void ParallelGptContextDecoder<T>::initialize()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    self_attention_layer_ = new TensorParallelGptContextAttentionLayer<T>(max_batch_size_,
                                                                          max_seq_len_,
                                                                          head_num_,
                                                                          size_per_head_,
                                                                          tensor_para_,
                                                                          stream_,
                                                                          cublas_wrapper_,
                                                                          allocator_,
                                                                          true,
                                                                          is_free_buffer_after_forward_,
                                                                          is_qk_buf_float_,
                                                                          sparse_,
                                                                          int8_mode_,
                                                                          custom_all_reduce_comm_,
                                                                          enable_custom_all_reduce_);

    bool use_gated_activation = activation_type_ == ActivationType::GeGLU || activation_type_ == ActivationType::ReGLU;
    size_t max_inter_size     = has_adapters_ ? std::max(inter_size_, adapter_inter_size_) : inter_size_;
    if (activation_type_ == ActivationType::Gelu || activation_type_ == ActivationType::GeGLU) {
        ffn_layer_ = new TensorParallelGeluFfnLayer<T>(max_batch_size_,
                                                       max_seq_len_,
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
                                                       max_seq_len_,
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
void ParallelGptContextDecoder<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void ParallelGptContextDecoder<T>::allocateBuffer(size_t batch_size, size_t seq_len, bool use_shared_contexts)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    decoder_normed_input_ = reinterpret_cast<T*>(
        allocator_->reMalloc(decoder_normed_input_, sizeof(T) * batch_size * seq_len * hidden_units_, false));
    self_attn_output_ = reinterpret_cast<T*>(
        allocator_->reMalloc(self_attn_output_, sizeof(T) * batch_size * seq_len * hidden_units_, false));
    normed_self_attn_output_ = decoder_normed_input_;  // reuse the buffer
    // only allocate additionl buffers when has adapters
    after_adapter_attn_output_ =
        has_adapters_ ? reinterpret_cast<T*>(
            allocator_->reMalloc(after_adapter_attn_output_, sizeof(T) * batch_size * seq_len * hidden_units_, false)) :
                        self_attn_output_;
    decoder_layer_output_ = reinterpret_cast<T*>(
        allocator_->reMalloc(decoder_layer_output_, sizeof(T) * batch_size * seq_len * hidden_units_, false));
    if (int8_mode_ == 2) {
        // TODO(mseznec): only use one buffer
        self_attn_output_int32_     = reinterpret_cast<int32_t*>(allocator_->reMalloc(
            self_attn_output_int32_, sizeof(int32_t) * batch_size * seq_len * hidden_units_, false));
        decoder_layer_output_int32_ = reinterpret_cast<int32_t*>(allocator_->reMalloc(
            decoder_layer_output_int32_, sizeof(int32_t) * batch_size * seq_len * hidden_units_, false));
    }
    token_num_ = reinterpret_cast<size_t*>(allocator_->reMalloc(token_num_, sizeof(size_t) * 1, false));
    padding_offset_ =
        reinterpret_cast<int*>(allocator_->reMalloc(padding_offset_, sizeof(int) * batch_size * seq_len, false));
    cu_seqlens_ = reinterpret_cast<int*>(allocator_->reMalloc(cu_seqlens_, sizeof(int) * (batch_size + 1), false));
    is_allocate_buffer_ = true;

    if (use_shared_contexts) {
        compact_decoder_features_ = reinterpret_cast<T*>(
            allocator_->reMalloc(compact_decoder_features_, sizeof(T) * batch_size * seq_len * hidden_units_, false));
        compact_attention_mask_ = reinterpret_cast<T*>(
            allocator_->reMalloc(compact_attention_mask_, sizeof(T) * batch_size * seq_len * seq_len, false));
        compact_input_lengths_ =
            reinterpret_cast<int*>(allocator_->reMalloc(compact_input_lengths_, sizeof(int) * batch_size, false));
        k_cache_layer_ = reinterpret_cast<T*>(
            allocator_->reMalloc(k_cache_layer_, sizeof(T) * batch_size * seq_len * hidden_units_, false));
        v_cache_layer_ = reinterpret_cast<T*>(
            allocator_->reMalloc(v_cache_layer_, sizeof(T) * batch_size * seq_len * hidden_units_, false));
    }
}

template<typename T>
void ParallelGptContextDecoder<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        allocator_->free((void**)(&decoder_normed_input_));
        allocator_->free((void**)(&self_attn_output_));
        if (has_adapters_) {
            allocator_->free((void**)(&after_adapter_attn_output_));
        }
        allocator_->free((void**)(&decoder_layer_output_));
        allocator_->free((void**)(&token_num_));
        allocator_->free((void**)(&padding_offset_));
        allocator_->free((void**)(&cu_seqlens_));
        if (compact_attention_mask_ != nullptr) {
            allocator_->free((void**)(&compact_decoder_features_));
            allocator_->free((void**)(&compact_attention_mask_));
            allocator_->free((void**)(&compact_input_lengths_));
            allocator_->free((void**)(&k_cache_layer_));
            allocator_->free((void**)(&v_cache_layer_));
        }
        if (int8_mode_ == 2) {
            allocator_->free((void**)(&decoder_layer_output_int32_));
            allocator_->free((void**)(&self_attn_output_int32_));
        }
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool ParallelGptContextDecoder<T>::isValidLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l >= local_num_layer * pipeline_para_.rank_)
           && (l < local_num_layer * (pipeline_para_.rank_ + 1));
}

template<typename T>
bool ParallelGptContextDecoder<T>::isFirstLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * pipeline_para_.rank_);
}

template<typename T>
bool ParallelGptContextDecoder<T>::isLastLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * (pipeline_para_.rank_ + 1) - 1);
}

template<typename T>
int ParallelGptContextDecoder<T>::getFirstLayerParallelId()
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return local_num_layer * pipeline_para_.rank_;
}

template<typename T>
ParallelGptContextDecoder<T>::ParallelGptContextDecoder(size_t           max_batch_size,
                                                        size_t           max_seq_len,
                                                        size_t           head_num,
                                                        size_t           size_per_head,
                                                        size_t           inter_size,
                                                        size_t           num_layer,
                                                        float            layernorm_eps,
                                                        gptVariantParams gpt_variant_params,
                                                        NcclParam        tensor_para,
                                                        NcclParam        pipeline_para,
                                                        cudaStream_t     stream,
                                                        cublasMMWrapper* cublas_wrapper,
                                                        IAllocator*      allocator,
                                                        bool             is_free_buffer_after_forward,
                                                        bool             is_qk_buf_float,
                                                        AttentionType    attention_type,
                                                        bool             sparse,
                                                        int              int8_mode,
                                                        std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                                                        int                                 enable_custom_all_reduce):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    layernorm_eps_(layernorm_eps),
    layernorm_type_(gpt_variant_params.layernorm_type),
    activation_type_(gpt_variant_params.activation_type),
    adapter_inter_size_(gpt_variant_params.adapter_inter_size),
    has_adapters_(gpt_variant_params.has_adapters),
    hidden_units_(head_num_ * size_per_head),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para),
    is_qk_buf_float_(is_qk_buf_float),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce),
    attention_type_(attention_type),
    int8_mode_(int8_mode)
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
    layernorm_eps_(decoder.layernorm_eps_),
    layernorm_type_(decoder.layernorm_type_),
    activation_type_(decoder.activation_type_),
    adapter_inter_size_(decoder.adapter_inter_size_),
    has_adapters_(decoder.has_adapters_),
    hidden_units_(decoder.hidden_units_),
    tensor_para_(decoder.tensor_para_),
    pipeline_para_(decoder.pipeline_para_),
    is_qk_buf_float_(decoder.is_qk_buf_float_),
    custom_all_reduce_comm_(decoder.custom_all_reduce_comm_),
    enable_custom_all_reduce_(decoder.enable_custom_all_reduce_),
    attention_type_(decoder.attention_type_),
    int8_mode_(decoder.int8_mode_)
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
    TensorMap*                                            output_tensors,
    const TensorMap*                                      input_tensors,
    const std::vector<ParallelGptDecoderLayerWeight<T>*>* gpt_decoder_layer_weight)
{
    // input tensors:
    //      decoder_input [batch_size, seq_len, hidden_dimension],
    //      attention_mask [batch_size, 1, seq_len, seq_len]
    //      input_lengths [batch_size]
    //      compact_idx [compact_size], optional
    //      batch_to_compact_idx [batch_size], optional
    //      linear_bias_slopes [head_num], optional

    // output tensors:
    //      decoder_output [batch_size, seq_len, hidden_dimension],
    //      key_cache [num_layer, batch, local_head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [num_layer, batch, local_head_num, max_seq_len, size_per_head]
    //      last_token_hidden_units [batch_size, hidden_dimension]

    // To use layer/pipeline parallelism, we view the shape of 'batch_size' to 'ite * local_batch_size'.
    // For example, the shape of decoder_input becomes [ite, batch_size, seq_len, hidden_dimension] during
    // computing.

    FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->isExist("decoder_input"));
    FT_CHECK(input_tensors->isExist("attention_mask"));
    FT_CHECK(input_tensors->isExist("input_lengths"));
    FT_CHECK(output_tensors->isExist("decoder_output"));
    FT_CHECK(output_tensors->isExist("key_cache"));
    FT_CHECK(output_tensors->isExist("value_cache"));
    FT_CHECK(output_tensors->isExist("last_token_hidden_units"));

    const bool use_shared_contexts = input_tensors->isExist("compact_idx");
    FT_CHECK(!use_shared_contexts || input_tensors->isExist("batch_to_compact_idx"));

    Tensor decoder_input_tensor = input_tensors->at("decoder_input");
    FT_CHECK(decoder_input_tensor.shape[2] == hidden_units_);

    // Request batch size
    const size_t request_batch_size = decoder_input_tensor.shape[0];
    // Maybe compacted batch size.
    const size_t batch_size =
        use_shared_contexts ? input_tensors->at("compact_idx").shape[0] : decoder_input_tensor.shape[0];
    // Request input length
    const size_t seq_len = decoder_input_tensor.shape[1];
    // The maximum length of generation.
    const size_t max_seq_len = output_tensors->at("value_cache").shape[3];

    const DataType data_type = getTensorType<T>();

    allocateBuffer(batch_size, seq_len, use_shared_contexts);

    if (use_shared_contexts) {
        invokeCompactInputs(compact_decoder_features_,
                            compact_attention_mask_,
                            compact_input_lengths_,
                            decoder_input_tensor.getPtr<T>(),
                            input_tensors->at("attention_mask").getPtr<T>(),
                            input_tensors->at("input_lengths").getPtr<int>(),
                            input_tensors->at("compact_idx").getPtr<int>(),
                            batch_size,
                            seq_len,
                            hidden_units_,
                            stream_);
    }

    const size_t local_batch_size = getLocalBatchSize(batch_size, seq_len, pipeline_para_.world_size_);
    FT_CHECK(batch_size % local_batch_size == 0);
    const size_t iteration_num = batch_size / local_batch_size;

    Tensor k_cache = output_tensors->at("key_cache");
    Tensor v_cache = output_tensors->at("value_cache");

    const auto activation_in_type  = int8_mode_ == 2 ? TYPE_INT8 : data_type;
    const auto activation_out_type = int8_mode_ == 2 ? TYPE_INT32 : data_type;

    // The resize of the key cache buffer by
    //   (local_batch_size, local_head_num, size_per_head // x, max_seq_len, x) where x is constant.
    std::vector<size_t> self_k_cache_size(k_cache.shape.begin() + 2, k_cache.shape.end());
    self_k_cache_size.insert(self_k_cache_size.begin(), local_batch_size);

    // The resize of the value cache buffer by
    //   (local_batch_size, local_head_num, max_seq_len, size_per_head).
    std::vector<size_t> self_v_cache_size(v_cache.shape.begin() + 2, v_cache.shape.end());
    self_v_cache_size.insert(self_v_cache_size.begin(), local_batch_size);

    if (use_shared_contexts) {
        // we use k_cache_layer_ and v_cache_layer_
        self_k_cache_size[3] = seq_len;
        self_v_cache_size[2] = seq_len;
    }

    AttentionType attention_type = (input_tensors->isExist("linear_bias_slopes") || is_qk_buf_float_ || int8_mode_ == 2) ?
        getUnfusedAttentionType(attention_type_) : attention_type_;
    const bool is_unpadded_mha = isUnPaddedMHA(attention_type);

    for (uint ite = 0; ite < iteration_num; ite++) {
        size_t h_token_num = local_batch_size * seq_len;
        if (is_unpadded_mha) {
            const int* base_input_lengths =
                (use_shared_contexts ? compact_input_lengths_ : input_tensors->at("input_lengths").getPtr<int>());
            invokeGetPaddingOffsetAndCuSeqLens(&h_token_num,
                                               token_num_,
                                               padding_offset_,
                                               cu_seqlens_,
                                               base_input_lengths + ite * local_batch_size,
                                               local_batch_size,
                                               seq_len,
                                               stream_);
        }

        for (uint l = 0; l < num_layer_; l++) {
            if (isValidLayerParallelId(l) == false) {
                continue;
            }

            if (l == 0 && is_unpadded_mha) {
                const T* base_input =
                    (use_shared_contexts ? compact_decoder_features_ : decoder_input_tensor.getPtr<T>());
                invokeRemovePadding(decoder_layer_output_,
                                    base_input + ite * local_batch_size * seq_len * hidden_units_,
                                    padding_offset_,
                                    h_token_num,
                                    hidden_units_,
                                    stream_);
            }

            ParallelGptDecoderLayerWeight<T>* layer_weight = gpt_decoder_layer_weight->at(l);

            T* decoder_input  = decoder_layer_output_;
            T* decoder_output = decoder_layer_output_;
            if (!is_unpadded_mha) {
                if (l == 0) {
                    decoder_input = use_shared_contexts ? compact_decoder_features_ : decoder_input_tensor.getPtr<T>();
                    decoder_input += ite * local_batch_size * seq_len * hidden_units_;
                }
                if (l == num_layer_ - 1) {
                    decoder_output = use_shared_contexts ? compact_decoder_features_ :
                                                           output_tensors->at("decoder_output").getPtr<T>();
                    decoder_output += ite * local_batch_size * seq_len * hidden_units_;
                }
            }

            if (isFirstLayerParallelId(l) && pipeline_para_.rank_ != 0) {
                const int data_size = h_token_num * hidden_units_ / tensor_para_.world_size_;
                ftNcclRecv(decoder_input + data_size * tensor_para_.rank_,
                           data_size,
                           pipeline_para_.rank_ - 1,
                           pipeline_para_,
                           stream_);
                if (tensor_para_.world_size_ > 1) {
                    ftNcclAllGather(decoder_input, decoder_input, data_size, tensor_para_.rank_, tensor_para_, stream_);
                }
            }

            if (layernorm_type_ == LayerNormType::pre_layernorm) {
                invokeGeneralLayerNorm(decoder_normed_input_,
                                       decoder_input,
                                       layer_weight->pre_layernorm_weights.gamma,
                                       layer_weight->pre_layernorm_weights.beta,
                                       layernorm_eps_,
                                       h_token_num,
                                       hidden_units_,
                                       layer_weight->self_attention_weights.query_weight.scale,
                                       int8_mode_,
                                       stream_);
            }
            sync_check_cuda_error();

            const bool is_final = false;  // TODO(bhsueh) remove this flag

            const T* attention_ptr =
                use_shared_contexts ? compact_attention_mask_ : input_tensors->at("attention_mask").getPtr<T>();

            TensorMap self_attention_input_tensors{
                {"input_query",
                 Tensor{MEMORY_GPU,
                        activation_in_type,
                        {h_token_num, hidden_units_},
                        layernorm_type_ == LayerNormType::pre_layernorm ? decoder_normed_input_ : decoder_input}},
                {"attention_mask",
                 Tensor{MEMORY_GPU,
                        data_type,
                        {local_batch_size, 1, seq_len, seq_len},
                        attention_ptr + local_batch_size * ite * seq_len * seq_len}},
                {"attention_type", Tensor{MEMORY_CPU, TYPE_VOID, {1}, &attention_type}},
                {"is_final_layer", Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &is_final}},
                {"layer_id", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &l}}};

            if (is_unpadded_mha) {
                self_attention_input_tensors.insert("padding_offset",
                                                    Tensor{MEMORY_GPU, TYPE_INT32, {h_token_num}, padding_offset_});
                self_attention_input_tensors.insert(
                    "cu_seqlens", Tensor{MEMORY_GPU, TYPE_INT32, {size_t(local_batch_size + 1)}, cu_seqlens_});
            }

            if (input_tensors->isExist("linear_bias_slopes")) {
                self_attention_input_tensors.insert("linear_bias_slopes", input_tensors->at("linear_bias_slopes"));
            }

            // The key/value cache stride per batch.
            const size_t cache_stride_per_batch = hidden_units_ / tensor_para_.world_size_ * max_seq_len;
            // The key/value cache offset of the layer.
            const size_t cache_layer_offset =
                (l - getFirstLayerParallelId()) * request_batch_size * cache_stride_per_batch;
            // The key/value cache offset of the current local batch iteration.
            const size_t ite_cache_offset = ite * local_batch_size * cache_stride_per_batch;
            const size_t cache_offset     = cache_layer_offset + ite_cache_offset;

            T* k_cache_ptr = use_shared_contexts ? k_cache_layer_ : k_cache.getPtrWithOffset<T>(cache_offset);
            T* v_cache_ptr = use_shared_contexts ? v_cache_layer_ : v_cache.getPtrWithOffset<T>(cache_offset);

            T* attention_out = (int8_mode_ == 2) ? reinterpret_cast<T*>(self_attn_output_int32_) : self_attn_output_;
            TensorMap self_attention_output_tensors{
                {"hidden_features",
                 Tensor{MEMORY_GPU, activation_out_type, {h_token_num, hidden_units_}, attention_out}},
                {"key_cache", Tensor{MEMORY_GPU, data_type, self_k_cache_size, k_cache_ptr}},
                {"value_cache", Tensor{MEMORY_GPU, data_type, self_v_cache_size, v_cache_ptr}}};

            self_attention_layer_->forward(
                &self_attention_output_tensors, &self_attention_input_tensors, &layer_weight->self_attention_weights);

            if (use_shared_contexts) {
                // Even with local batches, we must process the whole K/V caches as any
                // element in batch_idx_to_compact_idx may reference the local batch
                // we're processing. We also need to discard references that aren't in
                // that particular local batch.
                invokeUnCompactCaches(k_cache.getPtrWithOffset<T>(cache_layer_offset),
                                      v_cache.getPtrWithOffset<T>(cache_layer_offset),
                                      k_cache_layer_,
                                      v_cache_layer_,
                                      input_tensors->at("batch_to_compact_idx").getPtr<int>(),
                                      request_batch_size,  // batch_size (uncompact)
                                      v_cache.shape[2],    // local_head_num
                                      max_seq_len,
                                      seq_len,
                                      size_per_head_,
                                      local_batch_size,
                                      ite,
                                      stream_);
                sync_check_cuda_error();
            }

            // the adapter after attention (only pre layernorm currently)
            if (has_adapters_) {
                invokeGenericActivation<IdentityActivation, T, T>(
                    self_attn_output_,
                    layer_weight->self_attention_weights.attention_output_weight.bias,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    h_token_num,
                    hidden_units_,
                    0,
                    nullptr,
                    nullptr,
                    stream_);

                TensorMap ffn_input_tensors(
                    {{"ffn_input", Tensor{MEMORY_GPU, data_type, {h_token_num, hidden_units_}, self_attn_output_}}});
                TensorMap ffn_output_tensors(
                    {{"ffn_output",
                      Tensor{MEMORY_GPU, data_type, {h_token_num, hidden_units_}, after_adapter_attn_output_}}});

                ffn_layer_->resetInterSize(adapter_inter_size_ / tensor_para_.world_size_);
                ffn_layer_->forward(&ffn_output_tensors,
                                    &ffn_input_tensors,
                                    &gpt_decoder_layer_weight->at(l)->after_attention_adapter_weights);
            }

            if (layernorm_type_ == LayerNormType::pre_layernorm) {
                invokeGeneralAddBiasResidualPreLayerNorm(
                    has_adapters_ ? after_adapter_attn_output_ : self_attn_output_,
                    normed_self_attn_output_,
                    has_adapters_ ? after_adapter_attn_output_ : attention_out,
                    decoder_input,
                    has_adapters_ ? attention_out : nullptr,
                    layer_weight->self_attn_layernorm_weights.gamma,
                    layer_weight->self_attn_layernorm_weights.beta,
                    has_adapters_ ? layer_weight->after_attention_adapter_weights.output_weight.bias :
                                    layer_weight->self_attention_weights.attention_output_weight.bias,
                    layernorm_eps_,
                    h_token_num,
                    hidden_units_,
                    layer_weight->self_attention_weights.attention_output_weight.scale_inter,
                    layer_weight->self_attention_weights.attention_output_weight.scale_out,
                    layer_weight->ffn_weights.intermediate_weight.scale,
                    int8_mode_,
                    stream_);
            }
            else if (layernorm_type_ == LayerNormType::post_layernorm) {
                invokeAddBiasResidualLayerNorm(after_adapter_attn_output_,
                                               decoder_input,
                                               has_adapters_ ?
                                                   layer_weight->after_attention_adapter_weights.output_weight.bias :
                                                   layer_weight->self_attention_weights.attention_output_weight.bias,
                                               layer_weight->pre_layernorm_weights.gamma,
                                               layer_weight->pre_layernorm_weights.beta,
                                               layernorm_eps_,
                                               h_token_num,
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
                         {h_token_num, hidden_units_},
                         layernorm_type_ == LayerNormType::pre_layernorm ? normed_self_attn_output_ :
                                                                           after_adapter_attn_output_}}});
            TensorMap ffn_output_tensors(
                {{"ffn_output",
                  Tensor{MEMORY_GPU, activation_out_type, {h_token_num, hidden_units_}, ffn_output_ptr}}});

            ffn_layer_->resetInterSize(inter_size_ / tensor_para_.world_size_);
            ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &layer_weight->ffn_weights);

            // the adapter after ffn (only pre layernorm currently)
            if (has_adapters_) {
                invokeGenericActivation<IdentityActivation, T, T>(ffn_output_ptr,
                                                                  layer_weight->ffn_weights.output_weight.bias,
                                                                  nullptr,
                                                                  nullptr,
                                                                  nullptr,
                                                                  nullptr,
                                                                  h_token_num,
                                                                  hidden_units_,
                                                                  0,
                                                                  nullptr,
                                                                  nullptr,
                                                                  stream_);

                TensorMap ffn_input_tensors(
                    {{"ffn_input", Tensor{MEMORY_GPU, data_type, {h_token_num, hidden_units_}, ffn_output_ptr}}});
                TensorMap ffn_output_tensors(
                    {{"ffn_output", Tensor{MEMORY_GPU, data_type, {h_token_num, hidden_units_}, decoder_output}}});

                ffn_layer_->resetInterSize(adapter_inter_size_ / tensor_para_.world_size_);
                ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &layer_weight->after_ffn_adapter_weights);
            }

            if (layernorm_type_ == LayerNormType::pre_layernorm) {
                invokeAddBiasResidual(decoder_output,
                                      int8_mode_ == 2 ? reinterpret_cast<T*>(decoder_layer_output_int32_) :
                                                        decoder_output,
                                      after_adapter_attn_output_,
                                      has_adapters_ ? ffn_output_ptr : nullptr,
                                      has_adapters_ ? layer_weight->after_ffn_adapter_weights.output_weight.bias :
                                                      layer_weight->ffn_weights.output_weight.bias,
                                      int8_mode_ == 2 ? layer_weight->ffn_weights.output_weight.scale_inter : nullptr,
                                      int8_mode_ == 2 ? layer_weight->ffn_weights.output_weight.scale_out : nullptr,
                                      h_token_num,
                                      hidden_units_,
                                      int8_mode_,
                                      stream_);
            }
            else if (layernorm_type_ == LayerNormType::post_layernorm) {
                invokeAddBiasResidualLayerNorm(decoder_output,
                                               after_adapter_attn_output_,
                                               has_adapters_ ?
                                                   layer_weight->after_ffn_adapter_weights.output_weight.bias :
                                                   layer_weight->ffn_weights.output_weight.bias,
                                               layer_weight->self_attn_layernorm_weights.gamma,
                                               layer_weight->self_attn_layernorm_weights.beta,
                                               layernorm_eps_,
                                               h_token_num,
                                               hidden_units_,
                                               stream_);
            }
            sync_check_cuda_error();

            if (isLastLayerParallelId(l) == true && (pipeline_para_.rank_ != pipeline_para_.world_size_ - 1)) {
                const int data_size = h_token_num * hidden_units_ / tensor_para_.world_size_;
                ftNcclSend(decoder_output + data_size * tensor_para_.rank_,
                           data_size,
                           pipeline_para_.rank_ + 1,
                           pipeline_para_,
                           stream_);
            }

            if ((l == num_layer_ - 1) && is_unpadded_mha) {
                T* base_ptr =
                    use_shared_contexts ? compact_decoder_features_ : output_tensors->at("decoder_output").getPtr<T>();
                invokeRebuildPadding(base_ptr + ite * local_batch_size * seq_len * hidden_units_,
                                     decoder_layer_output_,
                                     padding_offset_,
                                     h_token_num,
                                     head_num_ * size_per_head_,
                                     stream_);
            }
        }
    }

    if (use_shared_contexts) {
        invokeUnCompactOutputs(output_tensors->at("decoder_output").getPtr<T>(),
                               compact_decoder_features_,
                               input_tensors->at("batch_to_compact_idx").getPtr<int>(),
                               request_batch_size,  // batch
                               seq_len * hidden_units_,
                               stream_);
    }

    // TODO(bhsueh) We could optimize this point by only computing the last token for the last layer
    invokeLookupHiddenStateOfLastToken(output_tensors->at("last_token_hidden_units").getPtr<T>(),
                                       output_tensors->at("decoder_output").getPtr<T>(),
                                       input_tensors->at("input_lengths").getPtr<int>(),
                                       seq_len,
                                       request_batch_size,
                                       hidden_units_,
                                       stream_);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template class ParallelGptContextDecoder<float>;
template class ParallelGptContextDecoder<half>;
#ifdef ENABLE_BF16
template class ParallelGptContextDecoder<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
