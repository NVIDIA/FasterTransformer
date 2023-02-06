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

#include <string>
#include <unordered_map>

#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/layernorm_fp8_kernels.h"
#include "src/fastertransformer/models/bert_fp8/BertFP8.h"
#include "src/fastertransformer/utils/nvtx_utils.h"

namespace fastertransformer {

template<typename T1, typename T2>
void BertFP8<T1, T2>::initialize()
{
    attention_layer_ = new SelfAttentionFP8Layer<T1, T2>(head_num_,
                                                         size_per_head_,
                                                         head_num_ * size_per_head_,
                                                         q_scaling_,
                                                         fp8_mode_,
                                                         sm_,
                                                         stream_,
                                                         cublas_wrapper_,
                                                         allocator_,
                                                         is_free_buffer_after_forward_,
                                                         sparse_);

    if (activation_type_ == ActivationType::Gelu) {
        ffn_layer_ = new GeluFfnFP8Layer<T1, T2>(
            inter_size_, fp8_mode_, stream_, cublas_wrapper_, allocator_, is_free_buffer_after_forward_, sparse_);
    }
    else if (activation_type_ == ActivationType::Relu) {
        ffn_layer_ = new ReluFfnFP8Layer<T1, T2>(
            inter_size_, fp8_mode_, stream_, cublas_wrapper_, allocator_, is_free_buffer_after_forward_, sparse_);
    }
}

template<typename T1, typename T2>
BertFP8<T1, T2>::BertFP8(size_t           head_num,
                         size_t           size_per_head,
                         size_t           d_model,
                         size_t           inter_size,
                         size_t           num_layer,
                         NcclParam        tensor_para,
                         NcclParam        pipeline_para,
                         int              sm,
                         float            q_scaling,
                         cudaStream_t     stream,
                         cublasMMWrapper* cublas_wrapper,
                         IAllocator*      allocator,
                         bool             is_free_buffer_after_forward,
                         AttentionType    attention_type,
                         bool             sparse,
                         ActivationType   activation_type,
                         LayerNormType    layernorm_type,
                         int              fp8_mode):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    head_num_(head_num),
    size_per_head_(size_per_head),
    d_model_(d_model),
    inter_size_(inter_size),
    num_layer_(num_layer),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para),
    sm_(sm),
    q_scaling_(q_scaling),
    attention_type_(attention_type),
    activation_type_(activation_type),
    layernorm_type_(layernorm_type),
    fp8_mode_(fp8_mode)
{
    initialize();
}

template<typename T1, typename T2>
BertFP8<T1, T2>::BertFP8(BertFP8<T1, T2> const& bert):
    BertFP8(bert.head_num_,
            bert.size_per_head_,
            bert.d_model_,
            bert.inter_size_,
            bert.num_layer_,
            bert.tensor_para_,
            bert.pipeline_para_,
            bert.sm_,
            bert.q_scaling_,
            bert.stream_,
            bert.cublas_wrapper_,
            bert.allocator_,
            bert.is_free_buffer_after_forward_,
            bert.attention_type_,
            bert.sparse_,
            bert.activation_type_,
            bert.layernorm_type_,
            bert.fp8_mode_)
{
}

template<typename T1, typename T2>
BertFP8<T1, T2>::~BertFP8()
{
    delete attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T1, typename T2>
void BertFP8<T1, T2>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T1, typename T2>
void BertFP8<T1, T2>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    h_pinned_token_num_ptr_ = (size_t*)allocator_->reMalloc(h_pinned_token_num_ptr_, sizeof(size_t), true, true);
    padding_offset_         = (int*)allocator_->reMalloc(padding_offset_, sizeof(int) * batch_size * seq_len, false);
    trt_mha_padding_offset_ =
        (int*)allocator_->reMalloc(trt_mha_padding_offset_, sizeof(int) * (2 * batch_size + 1), false);

    attention_mask_ = (T1*)allocator_->reMalloc(attention_mask_, sizeof(T1) * batch_size * seq_len * seq_len, false);

    bert_in_buffer_ = (T1*)allocator_->reMalloc(
        bert_in_buffer_, sizeof(T1) * batch_size * seq_len * head_num_ * size_per_head_, false);
    attn_out_buf_    = (T1*)allocator_->reMalloc(attn_out_buf_, sizeof(T1) * batch_size * seq_len * d_model_, false);
    bf16_out_tensor_ = (T2*)allocator_->reMalloc(bf16_out_tensor_, sizeof(T2) * batch_size * seq_len * d_model_, false);
    bert_out_buffer_ = (T1*)allocator_->reMalloc(
        bert_out_buffer_, sizeof(T1) * batch_size * seq_len * head_num_ * size_per_head_, false);

    if (layernorm_type_ == LayerNormType::post_layernorm) {
        normed_from_tensor_  = nullptr;
        normed_attn_out_buf_ = nullptr;
    }
    else {
        normed_from_tensor_ =
            (T1*)allocator_->reMalloc(normed_from_tensor_, sizeof(T1) * batch_size * seq_len * d_model_, false);
        normed_attn_out_buf_ =
            (T1*)allocator_->reMalloc(normed_attn_out_buf_, sizeof(T1) * batch_size * seq_len * d_model_, false);
    }

    first_token_tensor_ = (T2*)allocator_->reMalloc(first_token_tensor_, sizeof(T2) * batch_size * d_model_, false);
}

template<typename T1, typename T2>
void BertFP8<T1, T2>::freeBuffer()
{
    allocator_->free((void**)(&h_pinned_token_num_ptr_), true);
    allocator_->free((void**)(&padding_offset_));
    allocator_->free((void**)(&trt_mha_padding_offset_));

    allocator_->free((void**)(&attention_mask_));
    allocator_->free((void**)(&bert_in_buffer_));
    allocator_->free((void**)(&attn_out_buf_));
    allocator_->free((void**)(&bf16_out_tensor_));
    allocator_->free((void**)(&bert_out_buffer_));

    if (layernorm_type_ == LayerNormType::post_layernorm) {
        normed_from_tensor_  = nullptr;
        normed_attn_out_buf_ = nullptr;
    }
    else {
        allocator_->free((void**)(&normed_from_tensor_));
        allocator_->free((void**)(&normed_attn_out_buf_));
    }
    allocator_->free((void**)(&first_token_tensor_));
}

template<typename T1, typename T2>
void BertFP8<T1, T2>::forward(TensorMap*                   output_tensors,
                              TensorMap*                   input_tensors,
                              const BertFP8Weight<T1, T2>* bert_weights)
{
    // input_tensors:
    //      input_ids, int, [batch, seq_len]
    //      sequence_lengths, int, [batch]
    //      token_type_ids, int, [batch, seq_len]
    // output tensors:
    //      output_hidden_state [batch, seqlen, d_model_]
    //      ft_pooled_output, bfloat, [batch, d_model_], optional, the results of first token, only used for
    //      classification.

    const size_t request_batch_size = input_tensors->at("input_ids").shape[0];
    const size_t request_seq_len    = input_tensors->at("input_ids").shape[1];
    FT_CHECK(input_tensors->size() >= 2);
    FT_CHECK(request_batch_size == input_tensors->at("sequence_lengths").shape[0]);
    FT_CHECK(input_tensors->at("sequence_lengths").shape.size() == 1);
    allocateBuffer(request_batch_size, request_seq_len);
    FT_CHECK(output_tensors->at("output_hidden_state").type == DataType::TYPE_FP16);

    const int* sequence_lengths = reinterpret_cast<const int*>(input_tensors->at("sequence_lengths").data);

    size_t  h_token_num;
    Tensor* padding_offset_tensor_ptr     = nullptr;
    Tensor* trt_padding_offset_tensor_ptr = nullptr;

    AttentionType attention_type = attention_type_;

    FT_CHECK(attention_type != AttentionType::UNFUSED_MHA);  // not support UNFUSED_MHA now
    if (isFusedMHA(attention_type) == false) {
        PUSH_RANGE("invokeBuildEncoderAttentionMask");
        invokeBuildEncoderAttentionMask(
            attention_mask_, sequence_lengths, request_batch_size, request_seq_len, stream_);
        sync_check_cuda_error();
        POP_RANGE;
    }

    if (isPaddedMHA(attention_type)) {
        h_token_num = request_batch_size * request_seq_len;
        cudaMemsetAsync(padding_offset_, 0, sizeof(int) * h_token_num, stream_);

        PUSH_RANGE("invokeGetTrtPaddingOffset");
        invokeGetTrtPaddingOffset(
            trt_mha_padding_offset_, sequence_lengths, request_batch_size, request_seq_len, stream_);
        POP_RANGE;
        sync_check_cuda_error();

        padding_offset_tensor_ptr =
            new Tensor(MEMORY_GPU, TYPE_INT32, std::vector<size_t>{h_token_num}, padding_offset_);
        trt_padding_offset_tensor_ptr = new Tensor(
            MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size * 2 + 1}, trt_mha_padding_offset_);
    }
    else {
        PUSH_RANGE("invokeGetPaddingOffset");
        invokeGetPaddingOffset(h_pinned_token_num_ptr_,
                               &h_token_num,
                               padding_offset_,
                               sequence_lengths,
                               request_batch_size,
                               request_seq_len,
                               stream_);
        sync_check_cuda_error();
        POP_RANGE;

        PUSH_RANGE("invokeGetTrtPaddingOffset");
        invokeGetTrtPaddingOffset(trt_mha_padding_offset_, sequence_lengths, request_batch_size, stream_);
        POP_RANGE;
        sync_check_cuda_error();

        padding_offset_tensor_ptr =
            new Tensor(MEMORY_GPU, TYPE_INT32, std::vector<size_t>{h_token_num}, padding_offset_);
        trt_padding_offset_tensor_ptr =
            new Tensor(MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size + 1}, trt_mha_padding_offset_);
    }

    {
        RemovePaddingEmbLookupLayerNormFP8OutParam<T1, T2> param{
            bert_out_buffer_,
            input_tensors->at("input_ids").getPtr<int>(),
            input_tensors->isExist("position_ids") ? input_tensors->at("position_ids").getPtr<int>() : nullptr,
            input_tensors->isExist("token_type_ids") ? input_tensors->at("token_type_ids").getPtr<int>() : nullptr,
            padding_offset_,
            bert_weights->word_embeddings,
            bert_weights->position_embeddings,
            bert_weights->token_type_embeddings,
            bert_weights->embeddings_layernorm.gamma,
            bert_weights->embeddings_layernorm.beta,
            bert_weights->bert_layer_weights[0].attention_weights.query_weight.input_scale_inv,
            sequence_lengths,
            (int)h_token_num,
            (int)d_model_,
            (int)request_batch_size,
            (int)request_seq_len,
            0,  // pad_token_id
            stream_,
            false,
        };
        PUSH_RANGE("invokeRemovePaddingEmbLookupLayerNormFP8Out");
        invokeRemovePaddingEmbLookupLayerNormFP8Out(param);
        sync_check_cuda_error();
        POP_RANGE;
    }

    DataType data_type                = getTensorType<T1>();
    DataType high_precision_data_type = getTensorType<T2>();
    for (uint i = 0; i < num_layer_; i++) {
        const T1* from_tensor = bert_out_buffer_;
        T1*       out_tensor  = bert_out_buffer_;

        const BertFP8LayerWeight<T1, T2>* layer_weight = &bert_weights->bert_layer_weights[i];

        if (layernorm_type_ == LayerNormType::pre_layernorm) {
            if (fp8_mode_ == 1) {
                FT_CHECK(false);
            }
            else if (fp8_mode_ == 2) {
                // Need to layernorm kernel to support fp8 input.
                GeneralFP8IOPostLayerNormParam<T1, T2> param{
                    normed_from_tensor_,
                    from_tensor,
                    layer_weight->attn_layernorm_weights.gamma,
                    layer_weight->attn_layernorm_weights.beta,
                    nullptr,
                    layer_weight->attention_weights.query_weight.input_scale_inv,
                    (int)h_token_num,
                    (int)d_model_,
                    stream_,
                    true};
                sync_check_cuda_error();
            }
        }

        // Attention
        {
            TensorMap attn_input_tensors = TensorMap(std::unordered_map<std::string, Tensor>{
                {"input_hidden_state",
                 Tensor{MEMORY_GPU,
                        data_type,
                        {h_token_num, d_model_},
                        layernorm_type_ == LayerNormType::pre_layernorm ? normed_from_tensor_ : from_tensor}},
                {"attention_mask",
                 Tensor{MEMORY_GPU,
                        data_type,
                        {request_batch_size, 1, request_seq_len, request_seq_len},
                        attention_mask_}},
                {"attention_type", Tensor{MEMORY_CPU, TYPE_VOID, {1}, &attention_type}}});
            if (padding_offset_tensor_ptr != nullptr) {
                attn_input_tensors.insert("padding_offset", *padding_offset_tensor_ptr);
            }
            if (trt_padding_offset_tensor_ptr != nullptr) {
                attn_input_tensors.insert("trt_padding_offset", *trt_padding_offset_tensor_ptr);
            }

            TensorMap attn_output_tensors = TensorMap(std::unordered_map<std::string, Tensor>{
                {"output_hidden_state", Tensor{MEMORY_GPU, data_type, {h_token_num, d_model_}, attn_out_buf_}}});

            attention_layer_->forward(&attn_output_tensors, &attn_input_tensors, &layer_weight->attention_weights);
        }

        if (layernorm_type_ == LayerNormType::post_layernorm) {
            PUSH_RANGE("post layernorm 1");
            if (fp8_mode_ == 1) {
                FT_CHECK_WITH_INFO(false, "invokeGeneralFP8IOAddBiasResidualPostLayerNorm not support fp8_mode 1 now");
                // GeneralFP8IOAddBiasResidualPostLayerNormParam<T1, T2> param{
                //     attn_out_buf_,
                //     attn_out_buf_,
                //     from_tensor,
                //     layer_weight->attn_layernorm_weights.gamma,
                //     layer_weight->attn_layernorm_weights.beta,
                //     layer_weight->attention_weights.attention_output_weight.bias,
                //     layer_weight->attention_weights.attention_output_weight.output_scale,
                //     layer_weight->attention_weights.attention_output_weight.scale,
                //     layer_weight->attention_weights.attention_output_weight.per_channel_scale_min,
                //     layer_weight->ffn_weights.intermediate_weight.input_scale_inv,
                //     layer_weight->attention_weights.query_weight.input_scale,
                //     (int)h_token_num,
                //     (int)d_model_,
                //     stream_};
                // invokeGeneralFP8IOAddBiasResidualPostLayerNorm<T1, T2, PER_CHANNEL_WEIGHT_PER_TENSOR_ACT>(param);
            }
            else if (fp8_mode_ == 2) {
                GeneralFP8IOAddBiasResidualPostLayerNormParam<T1, T2> param{
                    attn_out_buf_,
                    attn_out_buf_,
                    from_tensor,
                    layer_weight->attn_layernorm_weights.gamma,
                    layer_weight->attn_layernorm_weights.beta,
#ifdef FUSE_GEMM_ACT
                    nullptr,
#else
                    layer_weight->attention_weights.attention_output_weight.bias,
#endif
                    layer_weight->attention_weights.attention_output_weight.output_scale,
                    nullptr,
                    nullptr,
                    layer_weight->ffn_weights.intermediate_weight.input_scale_inv,
                    layer_weight->attention_weights.query_weight.input_scale,
                    (int)h_token_num,
                    (int)d_model_,
                    stream_};
                invokeGeneralFP8IOAddBiasResidualPostLayerNorm(param);
            }
            POP_RANGE;
        }
        else if (layernorm_type_ == LayerNormType::pre_layernorm) {
            // GeneralFP8AddBiasResidualPreLayerNormParam<T1, T2> param{
            //     normed_attn_out_buf_,
            //     attn_out_buf_,
            //     from_tensor,
            //     layer_weight->attention_weights.attention_output_weight.bias,
            //     layer_weight->ffn_layernorm_weights.gamma,
            //     layer_weight->ffn_layernorm_weights.beta,
            //     (const float*)layer_weight->attention_weights.attention_output_weight.scale,
            //     (const float*)layer_weight->ffn_weights.intermediate_weight.input_scale,
            //     h_token_num,
            //     d_model_,
            //     stream_,
            //     true};
            // invokeGeneralFP8AddBiasResidualPreLayerNorm(param);
            // sync_check_cuda_error();
        }

        // FFN
        {
            TensorMap ffn_input_tensors  = TensorMap(std::unordered_map<std::string, Tensor>{
                 {"input_hidden_state",
                  Tensor{MEMORY_GPU,
                        data_type,
                        std::vector<size_t>{h_token_num, d_model_},
                        layernorm_type_ == LayerNormType::pre_layernorm ? normed_attn_out_buf_ : attn_out_buf_}}});
            TensorMap ffn_output_tensors = TensorMap(std::unordered_map<std::string, Tensor>{
                {"output_hidden_state",
                 Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, d_model_}, out_tensor}}});
            ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &layer_weight->ffn_weights);
        }

        if (layernorm_type_ == LayerNormType::post_layernorm) {
            PUSH_RANGE("post layernorm 2");
            if (fp8_mode_ == 1) {
                FT_CHECK_WITH_INFO(false, "invokeGeneralFP8IOAddBiasResidualPostLayerNorm not support fp8_mode 1 now");
                // GeneralFP8IOAddBiasResidualPostLayerNormParam<T1, T2> param{
                //     out_tensor,
                //     out_tensor,
                //     attn_out_buf_,
                //     layer_weight->ffn_layernorm_weights.gamma,
                //     layer_weight->ffn_layernorm_weights.beta,
                //     layer_weight->ffn_weights.output_weight.bias,
                //     layer_weight->ffn_weights.output_weight.output_scale,
                //     layer_weight->ffn_weights.output_weight.scale,
                //     layer_weight->ffn_weights.output_weight.per_channel_scale_min,
                //     i == num_layer_ - 1 ?
                //         layer_weight->attention_weights.query_weight.input_scale_inv :
                //         bert_weights->bert_layer_weights[i + 1]
                //             .attention_weights.query_weight.input_scale_inv,  // for next layer
                //     layer_weight->ffn_weights.intermediate_weight.input_scale,
                //     (int)h_token_num,
                //     (int)d_model_,
                //     stream_};
                // invokeGeneralFP8IOAddBiasResidualPostLayerNorm<T1, T2, PER_CHANNEL_WEIGHT_PER_TENSOR_ACT>(param);
            }
            else if (fp8_mode_ == 2) {
                GeneralFP8IOAddBiasResidualPostLayerNormParam<T1, T2> param{
                    out_tensor,
                    out_tensor,
                    attn_out_buf_,
                    layer_weight->ffn_layernorm_weights.gamma,
                    layer_weight->ffn_layernorm_weights.beta,
#ifdef FUSE_GEMM_ACT
                    layer_weight->ffn_weights.output_weight.bias,  // nullptr,
#else
                    layer_weight->ffn_weights.output_weight.bias,
#endif
                    layer_weight->ffn_weights.output_weight.output_scale,
                    nullptr,
                    nullptr,
                    i == num_layer_ - 1 ? layer_weight->attention_weights.query_weight.input_scale_inv :
                                          bert_weights->bert_layer_weights[i + 1]
                                              .attention_weights.query_weight.input_scale_inv,  // for next layer
                    layer_weight->ffn_weights.intermediate_weight.input_scale,
                    (int)h_token_num,
                    (int)d_model_,
                    stream_};
                invokeGeneralFP8IOAddBiasResidualPostLayerNorm(param);
            }
            POP_RANGE;
        }
        else if (layernorm_type_ == LayerNormType::pre_layernorm) {
            // invokeAddBiasResidual(out_tensor,
            //                       attn_out_buf_,
            //                       layer_weight->ffn_weights.output_weight.bias,
            //                       h_token_num,
            //                       d_model_,
            //                       stream_);
        }
        sync_check_cuda_error();
    }

    if (layernorm_type_ == LayerNormType::pre_layernorm) {
        // invokeGeneralLayerNorm(bert_out_buffer_,
        //                        bert_out_buffer_,
        //                        bert_weights->post_transformer_layernorm_weights.gamma,
        //                        bert_weights->post_transformer_layernorm_weights.beta,
        //                        h_token_num,
        //                        d_model_,
        //                        stream_);
    }

    {
        // because TRT does not support bfloat output now.
        QuantizeMatrixRebuildPaddingParam<half, T1, QUANTIZE_MODE::PER_TENSOR> param{
            output_tensors->at("output_hidden_state").getPtr<half>(),
            bert_out_buffer_,
            isPaddedMHA(attention_type) ? nullptr : padding_offset_,
            (int)h_token_num,
            (int)d_model_,
            bert_weights->bert_layer_weights[num_layer_ - 1].attention_weights.query_weight.input_scale,
            stream_};
        PUSH_RANGE("invokeQuantizeMatrixRebuildPadding");
        invokeQuantizeMatrixRebuildPadding<half, T1, QUANTIZE_MODE::PER_TENSOR>(param);
        sync_check_cuda_error();
        POP_RANGE;
    }

    if (output_tensors->isExist("ft_pooled_output")) {
        {
            getLastTokenDequantizeParam<T2, T1> param{
                first_token_tensor_,
                bert_out_buffer_,
                bert_weights->bert_layer_weights[num_layer_ - 1].attention_weights.query_weight.input_scale,
                (int)request_batch_size,
                (int)request_seq_len,
                (int)d_model_,
                stream_};
            invokeGetLastTokenDequantize<T2, T1>(param);
            sync_check_cuda_error();
        }

        {
            float alpha = 1.0f;
            float beta  = 0.0f;
            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  d_model_,
                                  request_batch_size,
                                  d_model_,
                                  &alpha,
                                  bert_weights->pooler_dense.kernel,
                                  CUDA_R_16BF,
                                  d_model_,
                                  first_token_tensor_,
                                  CUDA_R_16BF,
                                  d_model_,
                                  &beta,
                                  output_tensors->at("ft_pooled_output").getPtr<T2>(),
                                  CUDA_R_16BF,
                                  d_model_,
                                  CUDA_R_32F,
                                  cublasGemmAlgo_t(-1));
        }

        {
            invokeAddBiasTanh(output_tensors->at("ft_pooled_output").getPtr<T2>(),
                              bert_weights->pooler_dense.bias,
                              request_batch_size,
                              d_model_,
                              stream_);
            sync_check_cuda_error();
        }
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();

    if (padding_offset_tensor_ptr != nullptr) {
        delete padding_offset_tensor_ptr;
    }
    if (trt_padding_offset_tensor_ptr != nullptr) {
        delete trt_padding_offset_tensor_ptr;
    }
}

template class BertFP8<__nv_fp8_e4m3, __nv_bfloat16>;

}  // namespace fastertransformer
