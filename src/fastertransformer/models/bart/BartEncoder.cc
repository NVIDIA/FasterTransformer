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

#include "src/fastertransformer/models/bart/BartEncoder.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/decoding_kernels.h"
#include "src/fastertransformer/utils/Tensor.h"

namespace fastertransformer {

template<typename T>
void BartEncoder<T>::initialize()
{
    if ((attention_type_ == AttentionType::FUSED_MHA || attention_type_ == AttentionType::FUSED_PADDED_MHA)
        && false  // disable the fused mha
        && std::is_same<T, half>::value == true && max_seq_len_ <= 384) {
        FT_CHECK(false);  // fused mha does not support relatvie attention bias now.
        // attention_layer_ = new FusedAttentionLayer<T>(max_batch_size_,
        //                                               max_seq_len_,
        //                                               head_num_,
        //                                               size_per_head_,
        //                                               sm_,
        //                                               q_scaling_ * (1.0 / sqrtf(size_per_head_ * 1.0f)),
        //                                               stream_,
        //                                               cublas_wrapper_,
        //                                               allocator_,
        //                                               is_free_buffer_after_forward_,
        //                                               sparse_);
    }
    else if (attention_type_ == AttentionType::UNFUSED_MHA || attention_type_ == AttentionType::UNFUSED_PADDED_MHA) {
        attention_layer_ =
            new TensorParallelUnfusedAttentionLayer<T>(max_batch_size_,
                                                       max_seq_len_,
                                                       head_num_,
                                                       size_per_head_,
                                                       d_model_,
                                                       q_scaling_,  // adjust according to checkpoint structure
                                                       tensor_para_,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       is_free_buffer_after_forward_,
                                                       sparse_,
                                                       custom_all_reduce_comm_,
                                                       enable_custom_all_reduce_);
    }
    else {
        throw std::runtime_error(std::string("[FT][ERROR] Invalid attention type \n"));
    }

    bool use_gated_activation = activation_type_ == ActivationType::GeGLU || activation_type_ == ActivationType::ReGLU
                                || activation_type_ == ActivationType::SiGLU;
    if (activation_type_ == ActivationType::Gelu || activation_type_ == ActivationType::GeGLU) {
        ffn_layer_ = new TensorParallelGeluFfnLayer<T>(max_batch_size_,
                                                       max_seq_len_,
                                                       1,
                                                       d_model_,
                                                       0,
                                                       inter_size_,
                                                       tensor_para_,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       true,
                                                       is_free_buffer_after_forward_,
                                                       sparse_,
                                                       0,
                                                       use_gated_activation,  // don't use GeGLU
                                                       custom_all_reduce_comm_,
                                                       enable_custom_all_reduce_);
    }
    else if (activation_type_ == ActivationType::Relu || activation_type_ == ActivationType::ReGLU) {
        ffn_layer_ = new TensorParallelReluFfnLayer<T>(max_batch_size_,
                                                       max_seq_len_,
                                                       1,
                                                       d_model_,
                                                       0,
                                                       inter_size_,
                                                       tensor_para_,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       true,
                                                       is_free_buffer_after_forward_,
                                                       sparse_,
                                                       0,
                                                       use_gated_activation,
                                                       custom_all_reduce_comm_,
                                                       enable_custom_all_reduce_);
    }
    else if (activation_type_ == ActivationType::Silu || activation_type_ == ActivationType::SiGLU) {
        ffn_layer_ = new TensorParallelSiluFfnLayer<T>(max_batch_size_,
                                                       max_seq_len_,
                                                       1,
                                                       d_model_,
                                                       0,
                                                       inter_size_,
                                                       tensor_para_,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       true,
                                                       is_free_buffer_after_forward_,
                                                       sparse_,
                                                       use_gated_activation,
                                                       custom_all_reduce_comm_,
                                                       enable_custom_all_reduce_);
    }
}

template<typename T>
BartEncoder<T>::BartEncoder(size_t                              max_batch_size,
                            size_t                              max_seq_len,
                            size_t                              head_num,
                            size_t                              size_per_head,
                            size_t                              inter_size,
                            size_t                              d_model,
                            size_t                              num_layer,
                            size_t                              num_bucket_or_max_seq_len,
                            size_t                              max_distance,
                            int                                 sm,
                            float                               q_scaling,
                            cudaStream_t                        stream,
                            cublasMMWrapper*                    cublas_wrapper,
                            IAllocator*                         allocator,
                            bool                                is_free_buffer_after_forward,
                            AttentionType                       attention_type,
                            bool                                sparse,
                            ActivationType                      activation_type,
                            LayerNormType                       layernorm_type,
                            NcclParam                           tensor_para,
                            NcclParam                           pipeline_para,
                            std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                            int                                 enable_custom_all_reduce):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    d_model_(d_model),
    hidden_units_(head_num_ * size_per_head_),
    num_layer_(num_layer),
    num_bucket_or_max_seq_len_(num_bucket_or_max_seq_len),
    max_distance_(max_distance),
    sm_(sm),
    q_scaling_(q_scaling),
    attention_type_(attention_type),
    sparse_(sparse),
    activation_type_(activation_type),
    layernorm_type_(layernorm_type),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce)
{
    initialize();
}

template<typename T>
BartEncoder<T>::BartEncoder(BartEncoder<T> const& bart_encoder):
    BaseLayer(bart_encoder),
    max_batch_size_(bart_encoder.max_batch_size_),
    max_seq_len_(bart_encoder.max_seq_len_),
    head_num_(bart_encoder.head_num_),
    size_per_head_(bart_encoder.size_per_head_),
    inter_size_(bart_encoder.inter_size_),
    d_model_(bart_encoder.d_model_),
    hidden_units_(bart_encoder.hidden_units_),
    num_layer_(bart_encoder.num_layer_),
    num_bucket_or_max_seq_len_(bart_encoder.num_bucket_or_max_seq_len_),
    max_distance_(bart_encoder.max_distance_),
    sm_(bart_encoder.sm_),
    q_scaling_(bart_encoder.q_scaling_),
    attention_type_(bart_encoder.attention_type_),
    sparse_(bart_encoder.sparse_),
    activation_type_(bart_encoder.activation_type_),
    layernorm_type_(bart_encoder.layernorm_type_),
    tensor_para_(bart_encoder.tensor_para_),
    pipeline_para_(bart_encoder.pipeline_para_),
    custom_all_reduce_comm_(bart_encoder.custom_all_reduce_comm_),
    enable_custom_all_reduce_(bart_encoder.enable_custom_all_reduce_)
{
    initialize();
}

template<typename T>
BartEncoder<T>::~BartEncoder()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    delete attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void BartEncoder<T>::setStream(cudaStream_t stream)
{
    attention_layer_->setStream(stream);
    ffn_layer_->setStream(stream);
    BaseLayer::setStream(stream);
}

template<typename T>
void BartEncoder<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        token_num_ = (size_t*)allocator_->reMalloc(token_num_, sizeof(size_t) * 1, false);
        padding_offset_ =
            (int*)allocator_->reMalloc(padding_offset_, sizeof(int) * max_batch_size_ * max_seq_len_, false);
        trt_mha_padding_offset_ =
            (int*)allocator_->reMalloc(trt_mha_padding_offset_, sizeof(int) * (2 * max_batch_size_ + 1), false);

        attention_mask_ =
            (T*)allocator_->reMalloc(attention_mask_, sizeof(T) * max_batch_size_ * max_seq_len_ * max_seq_len_, false);
        relative_attention_bias_ = (T*)allocator_->reMalloc(
            relative_attention_bias_, sizeof(T) * head_num_ * max_seq_len_ * max_seq_len_, false);

        bart_encoder_emb_buf_ = (T*)allocator_->reMalloc(
            bart_encoder_emb_buf_, sizeof(T) * max_batch_size_ * max_seq_len_ * d_model_, false);
        bart_encoder_in_buffer_ = (T*)allocator_->reMalloc(
            bart_encoder_in_buffer_, sizeof(T) * max_batch_size_ * max_seq_len_ * d_model_, false);
        attn_out_buf_ =
            (T*)allocator_->reMalloc(attn_out_buf_, sizeof(T) * max_batch_size_ * max_seq_len_ * d_model_, false);
        bart_encoder_out_buffer_ = (T*)allocator_->reMalloc(
            bart_encoder_out_buffer_, sizeof(T) * max_batch_size_ * max_seq_len_ * d_model_, false);

        if (layernorm_type_ == LayerNormType::post_layernorm) {
            normed_from_tensor_  = nullptr;
            normed_attn_out_buf_ = nullptr;
        }
        else {
            normed_from_tensor_ = (T*)allocator_->reMalloc(
                normed_from_tensor_, sizeof(T) * max_batch_size_ * max_seq_len_ * d_model_, false);
            normed_attn_out_buf_ = (T*)allocator_->reMalloc(
                normed_attn_out_buf_, sizeof(T) * max_batch_size_ * max_seq_len_ * d_model_, false);
        }
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void BartEncoder<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    token_num_      = (size_t*)allocator_->reMalloc(token_num_, sizeof(size_t) * 1, false);
    padding_offset_ = (int*)allocator_->reMalloc(padding_offset_, sizeof(int) * batch_size * seq_len, false);
    trt_mha_padding_offset_ =
        (int*)allocator_->reMalloc(trt_mha_padding_offset_, sizeof(int) * (2 * batch_size + 1), false);

    attention_mask_ = (T*)allocator_->reMalloc(attention_mask_, sizeof(T) * batch_size * seq_len * seq_len, false);
    relative_attention_bias_ =
        (T*)allocator_->reMalloc(relative_attention_bias_, sizeof(T) * head_num_ * seq_len * seq_len, false);

    bart_encoder_emb_buf_ =
        (T*)allocator_->reMalloc(bart_encoder_emb_buf_, sizeof(T) * batch_size * seq_len * d_model_, false);
    bart_encoder_in_buffer_ =
        (T*)allocator_->reMalloc(bart_encoder_in_buffer_, sizeof(T) * batch_size * seq_len * d_model_, false);
    attn_out_buf_ = (T*)allocator_->reMalloc(attn_out_buf_, sizeof(T) * batch_size * seq_len * d_model_, false);
    bart_encoder_out_buffer_ =
        (T*)allocator_->reMalloc(bart_encoder_out_buffer_, sizeof(T) * batch_size * seq_len * d_model_, false);

    if (layernorm_type_ == LayerNormType::post_layernorm) {
        normed_from_tensor_  = nullptr;
        normed_attn_out_buf_ = nullptr;
    }
    else {
        normed_from_tensor_ =
            (T*)allocator_->reMalloc(normed_from_tensor_, sizeof(T) * batch_size * seq_len * d_model_, false);
        normed_attn_out_buf_ =
            (T*)allocator_->reMalloc(normed_attn_out_buf_, sizeof(T) * batch_size * seq_len * d_model_, false);
    }
    is_allocate_buffer_ = true;
}

template<typename T>
void BartEncoder<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&token_num_));
        allocator_->free((void**)(&padding_offset_));
        allocator_->free((void**)(&trt_mha_padding_offset_));

        allocator_->free((void**)(&attention_mask_));
        allocator_->free((void**)(&relative_attention_bias_));
        allocator_->free((void**)(&bart_encoder_emb_buf_));
        allocator_->free((void**)(&bart_encoder_in_buffer_));
        allocator_->free((void**)(&attn_out_buf_));
        allocator_->free((void**)(&bart_encoder_out_buffer_));

        if (layernorm_type_ == LayerNormType::post_layernorm) {
            normed_from_tensor_  = nullptr;
            normed_attn_out_buf_ = nullptr;
        }
        else {
            allocator_->free((void**)(&normed_from_tensor_));
            allocator_->free((void**)(&normed_attn_out_buf_));
        }
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool BartEncoder<T>::isValidLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l >= local_num_layer * pipeline_para_.rank_)
           && (l < local_num_layer * (pipeline_para_.rank_ + 1));
}

template<typename T>
bool BartEncoder<T>::isFirstLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * pipeline_para_.rank_);
}

template<typename T>
bool BartEncoder<T>::isLastLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * (pipeline_para_.rank_ + 1) - 1);
}

template<typename T>
int BartEncoder<T>::getFirstLayerParallelId()
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return local_num_layer * pipeline_para_.rank_;
}

template<typename T>
void BartEncoder<T>::forward(TensorMap*                  output_tensors,
                             TensorMap*                  input_tensors,
                             const BartEncoderWeight<T>* bart_encoder_weights)
{
    // input_tensors:
    //      input_ids [batch, seqlen]
    //      sequence_length [batch]
    //      inputs_embeds [batch, seqlen, d_model_]
    // output tensors:
    //      output_hidden_state [batch, seqlen, d_model_]
    //      output_attentions [batch_size, layer_num, num_heads, seqlen, seqlen], optional

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    const bool use_inputs_embeds = input_tensors->isExist("inputs_embeds");
    if (use_inputs_embeds) {
        if (input_tensors->isExist("input_ids")) {
            FT_LOG_WARNING("Pass input_ids and inputs_embeds at the same time, using inputs_embeds");
        }
        FT_CHECK(input_tensors->at("inputs_embeds").shape.size() == 3);
        FT_LOG_INFO("Using inputs embeds instead of input_ids !");
    }
    else {
        FT_CHECK(input_tensors->at("input_ids").shape.size() == 2);
    }
    std::string  input_tensor_name  = use_inputs_embeds ? "inputs_embeds" : "input_ids";
    const size_t request_batch_size = input_tensors->at(input_tensor_name).shape[0];
    const size_t request_seq_len    = input_tensors->at(input_tensor_name).shape[1];
    const bool   return_attentions  = output_tensors->at("output_attentions", {}).size();
    FT_CHECK(input_tensors->size() >= 2 && input_tensors->size() <= 3);
    FT_CHECK(request_batch_size == input_tensors->at("sequence_length").shape[0]);
    FT_CHECK(input_tensors->at("sequence_length").shape.size() == 1);

    allocateBuffer(request_batch_size, request_seq_len);

    size_t attentions_size = request_batch_size * num_layer_ * head_num_ * request_seq_len * request_seq_len;
    if (return_attentions) {
        cudaMemsetAsync(output_tensors->at("output_attentions").getPtr<T>(), 0, sizeof(T) * attentions_size, stream_);
    }

    // Bart Structure Difference
    const bool            bart_with_bias          = bart_encoder_weights->bart_with_bias;
    const bool            mbart                   = bart_encoder_weights->mbart;
    PositionEmbeddingType position_embedding_type = bart_encoder_weights->position_embedding_type;

    const bool use_inputs_embeds_buffer =
        use_inputs_embeds && position_embedding_type == PositionEmbeddingType::relative;

    invokeBuildRelativeAttentionBias(relative_attention_bias_,
                                     bart_encoder_weights->absolute_or_relative_position_embedding,
                                     head_num_,
                                     request_seq_len,
                                     num_bucket_or_max_seq_len_,
                                     true,
                                     max_distance_,
                                     position_embedding_type,
                                     stream_);
    if (attention_type_ == AttentionType::UNFUSED_MHA || attention_type_ == AttentionType::FUSED_MHA) {
        // prevent undefined behavior of the padding parts
        cudaMemset(output_tensors->at("output_hidden_state").getPtr<T>(),
                   0,
                   sizeof(T) * request_batch_size * request_seq_len * d_model_);
    }
    const size_t local_batch_size = getLocalBatchSize(request_batch_size, request_seq_len, pipeline_para_.world_size_);
    const size_t iteration_num    = request_batch_size / local_batch_size;

    for (uint ite = 0; ite < iteration_num; ite++) {
        size_t id_offset      = ite * local_batch_size;
        size_t d_model_offset = id_offset * request_seq_len * d_model_;

        const int* sequence_lengths = input_tensors->at("sequence_length").getPtr<int>() + id_offset;

        if (position_embedding_type == PositionEmbeddingType::absolute) {
            invokeInputIdsEmbeddingLookupPosEncoding(
                bart_encoder_emb_buf_,
                nullptr,
                use_inputs_embeds ? input_tensors->at("inputs_embeds").getPtr<T>() :
                                    bart_encoder_weights->embedding_table,
                bart_encoder_weights->absolute_or_relative_position_embedding,
                pPromptTuningParam<T>{},  // p/prompt tuning
                use_inputs_embeds ? nullptr :
                                    input_tensors->at("input_ids").getPtrWithOffset<int>(id_offset * request_seq_len),
                1,
                request_seq_len,
                request_seq_len,
                local_batch_size,
                d_model_,
                stream_);
        }
        else {
            if (!use_inputs_embeds) {
                invokeEmbeddingLookupPosEncodingPadCount(
                    bart_encoder_emb_buf_,
                    bart_encoder_weights->embedding_table,
                    (const T*)nullptr,
                    input_tensors->at("input_ids").getPtrWithOffset<int>(id_offset * request_seq_len),
                    nullptr,
                    pPromptTuningParam<T>{},
                    local_batch_size * request_seq_len,
                    d_model_,
                    (T)1.0f,
                    0,
                    local_batch_size * request_seq_len,
                    0,
                    request_seq_len,
                    stream_);
            }
        }

        sync_check_cuda_error();

        size_t  h_token_num;
        T*      bart_encoder_input_ptr;
        T*      bart_encoder_output_ptr;
        Tensor* padding_offset_tensor_ptr;
        // preprocess (remove padding and build mask)
        switch (attention_type_) {
            case AttentionType::UNFUSED_MHA: {
                invokeBuildEncoderAttentionMask(
                    attention_mask_, sequence_lengths, local_batch_size, request_seq_len, stream_);

                sync_check_cuda_error();
                invokeGetPaddingOffset(&h_token_num,
                                       token_num_,
                                       padding_offset_,
                                       sequence_lengths,
                                       local_batch_size,
                                       request_seq_len,
                                       stream_);
                sync_check_cuda_error();

                if (pipeline_para_.rank_ == 0) {
                    invokeRemovePadding(bart_encoder_in_buffer_,
                                        use_inputs_embeds_buffer ?
                                            input_tensors->at("inputs_embeds").getPtrWithOffset<T>(d_model_offset) :
                                            bart_encoder_emb_buf_,
                                        padding_offset_,
                                        h_token_num,
                                        d_model_,
                                        stream_);
                    sync_check_cuda_error();
                }
                bart_encoder_input_ptr  = bart_encoder_in_buffer_;
                bart_encoder_output_ptr = bart_encoder_out_buffer_;

                padding_offset_tensor_ptr =
                    new Tensor(MEMORY_GPU, TYPE_INT32, std::vector<size_t>{h_token_num}, padding_offset_);
                break;
            }
            case AttentionType::UNFUSED_PADDED_MHA: {
                invokeBuildEncoderAttentionMask(
                    attention_mask_, sequence_lengths, local_batch_size, request_seq_len, stream_);

                h_token_num = local_batch_size * request_seq_len;
                if (use_inputs_embeds_buffer) {
                    cudaMemcpyAsync(bart_encoder_emb_buf_,
                                    input_tensors->at("inputs_embeds").getPtrWithOffset<T>(d_model_offset),
                                    sizeof(T) * h_token_num * d_model_,
                                    cudaMemcpyDeviceToDevice,
                                    stream_);
                }

                sync_check_cuda_error();
                h_token_num               = local_batch_size * request_seq_len;
                bart_encoder_input_ptr    = bart_encoder_emb_buf_;
                bart_encoder_output_ptr   = output_tensors->at("output_hidden_state").getPtr<T>() + d_model_offset;
                padding_offset_tensor_ptr = new Tensor(MEMORY_GPU, TYPE_INT32, std::vector<size_t>{0}, nullptr);
                break;
            }
            case AttentionType::FUSED_MHA: {
                FT_CHECK(false);  // not support FUSED_MHA now
                // invokeGetPaddingOffset(&h_token_num,
                //                        token_num_,
                //                        padding_offset_,
                //                        sequence_lengths,
                //                        local_batch_size,
                //                        request_seq_len,
                //                        stream_);

                // if (pipeline_para_.rank_ == 0) {
                //     invokeRemovePadding(
                //         bart_encoder_in_buffer_, bart_encoder_emb_buf_, padding_offset_, h_token_num, d_model_,
                //         stream_);
                //     sync_check_cuda_error();
                // }
                // sync_check_cuda_error();
                // bart_encoder_input_ptr = bart_encoder_in_buffer_;
                // bart_encoder_output_ptr = bart_encoder_out_buffer_;

                // invokeGetTrtPaddingOffset(trt_mha_padding_offset_, sequence_lengths, local_batch_size, stream_);

                // padding_offset_tensor_ptr = new Tensor(
                //     MEMORY_GPU, TYPE_INT32, std::vector<size_t>{local_batch_size + 1}, trt_mha_padding_offset_);
                // break;
            }
            case AttentionType::FUSED_PADDED_MHA: {
                FT_CHECK(false);  // not support FUSED_MHA now
                // h_token_num = local_batch_size * request_seq_len;
                // invokeGetTrtPaddingOffset(
                //     trt_mha_padding_offset_, sequence_lengths, local_batch_size, request_seq_len, stream_);
                // padding_offset_tensor_ptr = new Tensor(
                //     MEMORY_GPU, TYPE_INT32, std::vector<size_t>{local_batch_size * 2 + 1}, trt_mha_padding_offset_);
                // bart_encoder_input_ptr = bart_encoder_emb_buf_ + d_model_offset;
                // bart_encoder_output_ptr = output_tensors->at("output_hidden_state").getPtr<T>() + d_model_offset;
                // break;
            }
            default: {
                throw std::runtime_error(std::string("[FT][ERROR] Invalid attention type \n"));
            }
        }

        DataType data_type = getTensorType<T>();

        // Before layers, BART/mBART has a layernorm on the embeddings. Different from T5
        // should we put rank=0 condition here?
        invokeGeneralT5LayerNorm(bart_encoder_input_ptr,
                                 bart_encoder_input_ptr,
                                 bart_encoder_weights->pre_transformer_layernorm_weights.gamma,
                                 bart_encoder_weights->pre_transformer_layernorm_weights.beta,
                                 layernorm_eps_,
                                 h_token_num,
                                 d_model_,
                                 stream_);
        sync_check_cuda_error();

        // Encoder layers
        for (uint i = 0; i < num_layer_; i++) {
            if (!isValidLayerParallelId(i)) {
                continue;
            }
            T* from_tensor = (i == 0 ? bart_encoder_input_ptr : bart_encoder_output_ptr);
            T* out_tensor  = bart_encoder_output_ptr;

            BartEncoderLayerWeight<T>* layer_weight = bart_encoder_weights->bart_encoder_layer_weights[i];

            if (isFirstLayerParallelId(i) && pipeline_para_.rank_ != 0) {
                const int data_size = h_token_num * d_model_ / tensor_para_.world_size_;
                ftNcclRecv(from_tensor + data_size * tensor_para_.rank_,
                           data_size,
                           pipeline_para_.rank_ - 1,
                           pipeline_para_,
                           stream_);
                if (tensor_para_.world_size_ > 1) {
                    ftNcclAllGather(from_tensor, from_tensor, data_size, tensor_para_.rank_, tensor_para_, stream_);
                }
            }
            if (layernorm_type_ == LayerNormType::pre_layernorm) {
                invokeGeneralT5LayerNorm(normed_from_tensor_,
                                         from_tensor,
                                         layer_weight->attn_layernorm_weights_.gamma,
                                         layer_weight->attn_layernorm_weights_.beta,
                                         layernorm_eps_,
                                         h_token_num,
                                         d_model_,
                                         stream_);
            }

            {
                TensorMap attn_input_tensors{
                    {"input_query",
                     Tensor{MEMORY_GPU,
                            data_type,
                            std::vector<size_t>{h_token_num, d_model_},
                            layernorm_type_ == LayerNormType::pre_layernorm ? normed_from_tensor_ : from_tensor}},
                    {"attention_mask",
                     Tensor{MEMORY_GPU,
                            data_type,
                            std::vector<size_t>{local_batch_size, 1, request_seq_len, request_seq_len},
                            attention_mask_}}};

                attn_input_tensors.insertIfValid(
                    "relative_attention_bias",
                    Tensor{MEMORY_GPU,
                           data_type,
                           std::vector<size_t>{1, head_num_, request_seq_len, request_seq_len},
                           bart_encoder_weights->position_embedding_type == PositionEmbeddingType::relative ?
                               relative_attention_bias_ :
                               nullptr});
                attn_input_tensors.insertIfValid("padding_offset", *padding_offset_tensor_ptr);

                TensorMap attn_output_tensors{
                    {"hidden_features",
                     Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, d_model_}, attn_out_buf_}}};
                if (return_attentions) {
                    auto tp_offset = tensor_para_.rank_ * (head_num_ / tensor_para_.world_size_);
                    auto output_attentions_offset =
                        ((id_offset * num_layer_ + i) * head_num_ + tp_offset) * request_seq_len * request_seq_len;
                    attn_output_tensors.insert(
                        {"attentions",
                         output_tensors->at("output_attentions")
                             .slice({local_batch_size, 1, head_num_, request_seq_len, request_seq_len},
                                    output_attentions_offset)});
                }

                attention_layer_->forward(&attn_output_tensors, &attn_input_tensors, &layer_weight->attention_weights_);
            }

            if (layernorm_type_ == LayerNormType::post_layernorm) {
                invokeGeneralAddBiasResidualT5PreLayerNorm(
                    attn_out_buf_,
                    attn_out_buf_,
                    from_tensor,
                    layer_weight->attn_layernorm_weights_.gamma,
                    layer_weight->attn_layernorm_weights_.beta,
                    layer_weight->attention_weights_.attention_output_weight.bias,
                    layernorm_eps_,
                    h_token_num,
                    d_model_,
                    stream_);
            }
            else if (layernorm_type_ == LayerNormType::pre_layernorm) {
                invokeGeneralAddBiasResidualT5PreLayerNorm(
                    attn_out_buf_,
                    normed_attn_out_buf_,
                    from_tensor,
                    layer_weight->ffn_layernorm_weights_.gamma,
                    layer_weight->ffn_layernorm_weights_.beta,
                    layer_weight->attention_weights_.attention_output_weight.bias,
                    layernorm_eps_,
                    h_token_num,
                    d_model_,
                    stream_);
            }

            // FFN
            {
                TensorMap ffn_input_tensors(
                    {{"ffn_input",
                      Tensor{MEMORY_GPU,
                             data_type,
                             {h_token_num, d_model_},
                             layernorm_type_ == LayerNormType::pre_layernorm ? normed_attn_out_buf_ : attn_out_buf_}}});
                TensorMap ffn_output_tensors;
                ffn_output_tensors.insert("ffn_output",
                                          Tensor{MEMORY_GPU, data_type, {h_token_num, d_model_}, out_tensor});
                ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &layer_weight->ffn_weights_);
            }

            if (layernorm_type_ == LayerNormType::post_layernorm) {
                invokeGeneralAddBiasResidualT5PreLayerNorm(out_tensor,
                                                           out_tensor,
                                                           attn_out_buf_,
                                                           layer_weight->ffn_layernorm_weights_.gamma,
                                                           layer_weight->ffn_layernorm_weights_.beta,
                                                           layer_weight->ffn_weights_.output_weight.bias,
                                                           layernorm_eps_,
                                                           h_token_num,
                                                           d_model_,
                                                           stream_);
            }
            else if (layernorm_type_ == LayerNormType::pre_layernorm) {
                invokeT5AddBiasResidual(out_tensor,
                                        attn_out_buf_,
                                        layer_weight->ffn_weights_.output_weight.bias,
                                        h_token_num,
                                        d_model_,
                                        stream_);
            }
            sync_check_cuda_error();

            if (isLastLayerParallelId(i) == true && pipeline_para_.rank_ != pipeline_para_.world_size_ - 1) {
                const int data_size = h_token_num * d_model_ / tensor_para_.world_size_;
                ftNcclSend(out_tensor + data_size * tensor_para_.rank_,
                           data_size,
                           pipeline_para_.rank_ + 1,
                           pipeline_para_,
                           stream_);
            }
        }

        if (pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) {
            if (mbart) {  // mBART has final layernorm after Transformer block. BART doesn't
                invokeGeneralT5LayerNorm(bart_encoder_output_ptr,
                                         bart_encoder_output_ptr,
                                         bart_encoder_weights->post_transformer_layernorm_weights.gamma,
                                         bart_encoder_weights->post_transformer_layernorm_weights.beta,
                                         layernorm_eps_,
                                         h_token_num,
                                         d_model_,
                                         stream_);
            }

            // post process (rebuild padding)
            switch (attention_type_) {
                case AttentionType::UNFUSED_MHA: {
                    if (pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) {
                        invokeRebuildPadding(output_tensors->at("output_hidden_state").getPtr<T>() + d_model_offset,
                                             bart_encoder_out_buffer_,
                                             padding_offset_,
                                             h_token_num,
                                             d_model_,
                                             stream_);
                    }
                    break;
                }
                case AttentionType::UNFUSED_PADDED_MHA: {
                    break;
                }
                case AttentionType::FUSED_MHA: {
                    if (pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) {
                        invokeRebuildPadding(output_tensors->at("output_hidden_state").getPtr<T>() + d_model_offset,
                                             bart_encoder_out_buffer_,
                                             padding_offset_,
                                             h_token_num,
                                             d_model_,
                                             stream_);
                    }
                    break;
                }
                case AttentionType::FUSED_PADDED_MHA: {
                    break;
                }
                default: {
                    throw std::runtime_error(std::string("[FT][ERROR] Invalid attention type \n"));
                }
            }
        }

        delete padding_offset_tensor_ptr;
    }
    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();

    if (pipeline_para_.world_size_ > 1) {
        ftNcclGroupStart();
        const int data_size = request_batch_size * request_seq_len * d_model_ / tensor_para_.world_size_;
        ftNcclBroadCast(output_tensors->at("output_hidden_state").getPtr<T>() + data_size * tensor_para_.rank_,
                        data_size,
                        pipeline_para_.world_size_ - 1,
                        pipeline_para_,
                        stream_);
        if (return_attentions) {
            ftNcclAllReduceSum(output_tensors->at("output_attentions").getPtr<T>(),
                               output_tensors->at("output_attentions").getPtr<T>(),
                               attentions_size,
                               pipeline_para_,
                               stream_);
        }
        ftNcclGroupEnd();
        check_cuda_error(cudaStreamSynchronize(stream_));
        sync_check_cuda_error();
        if (tensor_para_.world_size_ > 1) {
            ftNcclAllGather(output_tensors->at("output_hidden_state").getPtr<T>(),
                            output_tensors->at("output_hidden_state").getPtr<T>(),
                            data_size,
                            tensor_para_.rank_,
                            tensor_para_,
                            stream_);
        }
    }
    if (tensor_para_.world_size_ > 1 && return_attentions) {
        ftNcclAllReduceSum(output_tensors->at("output_attentions").getPtr<T>(),
                           output_tensors->at("output_attentions").getPtr<T>(),
                           attentions_size,
                           tensor_para_,
                           stream_);
    }
}

template class BartEncoder<float>;
template class BartEncoder<half>;
#ifdef ENABLE_BF16
template class BartEncoder<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
