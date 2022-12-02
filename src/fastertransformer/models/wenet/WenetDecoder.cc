/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2022.  Authored by Yuqing Ding.
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

#include "src/fastertransformer/models/wenet/WenetDecoder.h"
#include "src/fastertransformer/models/wenet/MultiHeadedAttentionLayer.h"
#include "src/fastertransformer/models/wenet/WenetKernels.h"

namespace fastertransformer {

template<typename T>
void WenetDecoder<T>::initialize()
{

    self_attention_layer_ = new MultiHeadedAttentionLayer<T>(0,
                                                             0,
                                                             head_num_,
                                                             size_per_head_,
                                                             qscaling_,
                                                             stream_,
                                                             cublas_wrapper_,
                                                             allocator_,
                                                             is_free_buffer_after_forward_);

    cross_attention_layer_ = new MultiHeadedAttentionLayer<T>(0,
                                                              0,
                                                              head_num_,
                                                              size_per_head_,
                                                              qscaling_,
                                                              stream_,
                                                              cublas_wrapper_,
                                                              allocator_,
                                                              is_free_buffer_after_forward_);

    ffn_layer_ = new ReluFfnLayer<T>(0,
                                     0,
                                     head_num_,
                                     size_per_head_,
                                     0,  // expert_num
                                     inter_size_,
                                     stream_,
                                     cublas_wrapper_,
                                     allocator_,
                                     is_free_buffer_after_forward_);
}

template<typename T>
void WenetDecoder<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void WenetDecoder<T>::allocateBuffer(size_t batch_size, size_t beam_size, size_t seq_len)
{
    size_t feature_size = batch_size * beam_size * seq_len * hidden_units_ * sizeof(T);

    pos_emb_tensor_ = reinterpret_cast<T*>(allocator_->reMalloc(pos_emb_tensor_, feature_size, false));
    encoder_output_repeated_ =
        reinterpret_cast<T*>(allocator_->reMalloc(encoder_output_repeated_, feature_size, false));
    encoder_sequence_length_repeated_ = reinterpret_cast<int*>(
        allocator_->reMalloc(encoder_sequence_length_repeated_, batch_size * beam_size * sizeof(int), false));
    self_attn_mask_ = reinterpret_cast<T*>(
        allocator_->reMalloc(self_attn_mask_, batch_size * beam_size * seq_len * seq_len * sizeof(T), false));
    cross_attn_mask_ = reinterpret_cast<T*>(
        allocator_->reMalloc(cross_attn_mask_, batch_size * beam_size * seq_len * seq_len * sizeof(T), false));

    decoder_normed_input_     = (T*)allocator_->reMalloc(decoder_normed_input_, feature_size, false);
    self_attn_output_         = (T*)allocator_->reMalloc(self_attn_output_, feature_size, false);
    normed_self_attn_output_  = (T*)allocator_->reMalloc(normed_self_attn_output_, feature_size, false);
    cross_attn_output_        = (T*)allocator_->reMalloc(cross_attn_output_, feature_size, false);
    normed_cross_attn_output_ = (T*)allocator_->reMalloc(normed_cross_attn_output_, feature_size, false);
    decoder_layer_output_     = (T*)allocator_->reMalloc(decoder_layer_output_, feature_size, false);

    log_probs_buf_ = reinterpret_cast<float*>(
        allocator_->reMalloc(log_probs_buf_, batch_size * beam_size * seq_len * vocab_size_ * sizeof(float), false));
    decoder_score_buf_ =
        reinterpret_cast<float*>(allocator_->reMalloc(decoder_score_buf_, batch_size * beam_size, false));
}

template<typename T>
void WenetDecoder<T>::freeBuffer()
{
    allocator_->free((void**)(&pos_emb_tensor_));
    allocator_->free((void**)(&encoder_output_repeated_));
    allocator_->free((void**)(&encoder_sequence_length_repeated_));
    allocator_->free((void**)(&self_attn_mask_));
    allocator_->free((void**)(&cross_attn_mask_));

    allocator_->free((void**)(&decoder_normed_input_));
    allocator_->free((void**)(&self_attn_output_));
    allocator_->free((void**)(&normed_self_attn_output_));
    allocator_->free((void**)(&cross_attn_output_));
    allocator_->free((void**)(&normed_cross_attn_output_));
    allocator_->free((void**)(&decoder_layer_output_));

    allocator_->free((void**)(&log_probs_buf_));
    allocator_->free((void**)(&decoder_score_buf_));
}

template<typename T>
WenetDecoder<T>::WenetDecoder(size_t           max_batch_size,
                              size_t           max_seq_len,
                              size_t           head_num,
                              size_t           size_per_head,
                              size_t           inter_size,
                              size_t           num_layer,
                              size_t           vocab_size,
                              size_t           max_len,
                              float            qscaling,
                              cudaStream_t     stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator*      allocator,
                              bool             is_free_buffer_after_forward):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    vocab_size_(vocab_size),
    max_len_(max_len),
    qscaling_(qscaling),
    hidden_units_(head_num_ * size_per_head)
{
    initialize();
}

template<typename T>
WenetDecoder<T>::WenetDecoder(WenetDecoder<T> const& decoder):
    BaseLayer(decoder.stream_, decoder.cublas_wrapper_, decoder.allocator_, decoder.is_free_buffer_after_forward_),
    head_num_(decoder.head_num_),
    size_per_head_(decoder.size_per_head_),
    inter_size_(decoder.inter_size_),
    num_layer_(decoder.num_layer_),
    vocab_size_(decoder.vocab_size_),
    max_len_(decoder.max_len_),
    qscaling_(decoder.qscaling_),
    hidden_units_(decoder.hidden_units_)
{
    initialize();
}

template<typename T>
WenetDecoder<T>::~WenetDecoder()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    delete self_attention_layer_;
    delete cross_attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void WenetDecoder<T>::setStream(cudaStream_t stream)
{
    self_attention_layer_->setStream(stream);
    cross_attention_layer_->setStream(stream);
    ffn_layer_->setStream(stream);
    BaseLayer::setStream(stream);
}

template<typename T>
void WenetDecoder<T>::forward(std::vector<Tensor>*         output_tensors,
                              const std::vector<Tensor>*   input_tensors,
                              const WenetDecoderWeight<T>* decoder_weight)
{
    TensorMap input_tensors_map = TensorMap({{"decoder_input", input_tensors->at(0)},
                                             {"decoder_sequence_length", input_tensors->at(1)},
                                             {"encoder_output", input_tensors->at(2)},
                                             {"encoder_sequence_length", input_tensors->at(3)},
                                             {"ctc_score", input_tensors->at(4)}});

    TensorMap output_tensors_map = TensorMap({
        {"decoder_output", output_tensors->at(0)},
        {"best_index", output_tensors->at(1)},
    });

    forward(&output_tensors_map, &input_tensors_map, decoder_weight);
}

template<typename T>
void WenetDecoder<T>::forward(TensorMap*                   output_tensors,
                              TensorMap*                   input_tensors,
                              const WenetDecoderWeight<T>* decoder_weight)
{
    // input tensors:
    //      decoder_input [batch_size, beam_width, seq_len],
    //      decoder_sequence_length [batch_size, beam_width],
    //      encoder_output [batch_size, mem_seq_len, memory_hidden_dimension],
    //      encoder_sequence_length [batch_size],
    //      ctc_score [batch_size, beam_width]

    // output tensors:
    //      decoder_output [batch_size, beam_width, seq_len-1, vocab_size],
    //      best_index [batch_size]

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() == 5);
    FT_CHECK(output_tensors->size() == 2);

    const size_t   beam_size  = (size_t)input_tensors->at("decoder_input").shape[1];
    const size_t   batch_size = (size_t)input_tensors->at("decoder_input").shape[0];
    const size_t   seq_len1   = (size_t)input_tensors->at("decoder_input").shape[2] - 1;
    const size_t   seq_len2   = (size_t)input_tensors->at("encoder_output").shape[1];
    const size_t   seq_len12  = std::max(seq_len1, seq_len2);
    const size_t   m          = batch_size * beam_size * seq_len1;
    const DataType data_type  = getTensorType<T>();
    const T        ctc_weight = (T)0.3f;
    allocateBuffer(batch_size, beam_size, seq_len12);

    const int* decoder_input           = input_tensors->getPtr<int>("decoder_input");
    const int* decoder_sequence_length = input_tensors->getPtr<int>("decoder_sequence_length");
    const T*   encoder_output          = input_tensors->getPtr<T>("encoder_output");
    const int* encoder_sequence_length = input_tensors->getPtr<int>("encoder_sequence_length");
    const T*   ctc_score               = input_tensors->getPtr<T>("ctc_score");
    int*       best_index              = output_tensors->getPtr<int>("best_index");

    invokeEmbedDecoderInput(pos_emb_tensor_,
                            decoder_input,
                            decoder_weight->decoder_embed_weights.data,
                            decoder_weight->positional_encoding_weights.data,
                            vocab_size_,
                            max_len_,
                            batch_size * beam_size,
                            seq_len1,
                            hidden_units_,
                            stream_);
    sync_check_cuda_error();

    invokeRepeatBeamSize(encoder_output_repeated_,
                         encoder_output,
                         batch_size,
                         seq_len2 * input_tensors->at("encoder_output").shape[2],
                         beam_size,
                         stream_);
    sync_check_cuda_error();

    invokeRepeatBeamSize(encoder_sequence_length_repeated_, encoder_sequence_length, batch_size, 1, beam_size, stream_);
    sync_check_cuda_error();

    invokeBuildDecoderAttentionMask<T, false>(
        self_attn_mask_, decoder_sequence_length, nullptr, batch_size * beam_size, seq_len1, seq_len1, stream_);
    sync_check_cuda_error();

    invokeBuildDecoderAttentionMask<T, true>(cross_attn_mask_,
                                             decoder_sequence_length,
                                             encoder_sequence_length_repeated_,
                                             batch_size * beam_size,
                                             seq_len1,
                                             seq_len2,
                                             stream_);

    sync_check_cuda_error();

    for (uint l = 0; l < num_layer_; l++) {
        const T* decoder_input = (const T*)((l == 0) ? pos_emb_tensor_ : decoder_layer_output_);

        invokeGeneralLayerNorm(decoder_normed_input_,
                               decoder_input,
                               decoder_weight->decoder_layer_weights[l]->pre_layernorm_weights.gamma,
                               decoder_weight->decoder_layer_weights[l]->pre_layernorm_weights.beta,
                               1e-6f,
                               m,
                               hidden_units_,
                               (float*)nullptr,
                               0,
                               stream_);
        sync_check_cuda_error();

        TensorMap self_attention_input_tensors{
            {"query_tensor",
             Tensor{MEMORY_GPU, data_type, {batch_size * beam_size, seq_len1, hidden_units_}, decoder_normed_input_}},
            {"key_tensor",
             Tensor{MEMORY_GPU, data_type, {batch_size * beam_size, seq_len1, hidden_units_}, decoder_normed_input_}},
            {"value_tensor",
             Tensor{MEMORY_GPU, data_type, {batch_size * beam_size, seq_len1, hidden_units_}, decoder_normed_input_}},
            {"attention_mask",
             Tensor{MEMORY_GPU, data_type, {batch_size * beam_size, 1, seq_len1, seq_len1}, self_attn_mask_}},
            // {"padding_offset", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size * beam_size}, nullptr}}
        };
        TensorMap self_attention_output_tensors{
            {"attention_out",
             Tensor{MEMORY_GPU, data_type, {batch_size * beam_size, seq_len1, hidden_units_}, self_attn_output_}}};
        self_attention_layer_->forward(&self_attention_output_tensors,
                                       &self_attention_input_tensors,
                                       &decoder_weight->decoder_layer_weights[l]->self_attention_weights);

        invokeGeneralScaleAddBiasResidualPreLayerNorm(
            self_attn_output_,
            normed_self_attn_output_,
            decoder_input,
            decoder_weight->decoder_layer_weights[l]->self_attn_layernorm_weights.gamma,
            decoder_weight->decoder_layer_weights[l]->self_attn_layernorm_weights.beta,
            decoder_weight->decoder_layer_weights[l]->self_attention_weights.attention_output_weight.bias,
            m,
            hidden_units_,
            stream_,
            2,
            1.0f,
            1.0f);
        sync_check_cuda_error();

        TensorMap cross_attention_input_tensors{
            {"query_tensor",
             Tensor{
                 MEMORY_GPU, data_type, {batch_size * beam_size, seq_len1, hidden_units_}, normed_self_attn_output_}},
            {"key_tensor",
             Tensor{
                 MEMORY_GPU, data_type, {batch_size * beam_size, seq_len2, hidden_units_}, encoder_output_repeated_}},
            {"value_tensor",
             Tensor{
                 MEMORY_GPU, data_type, {batch_size * beam_size, seq_len2, hidden_units_}, encoder_output_repeated_}},
            {"attention_mask",
             Tensor{MEMORY_GPU, data_type, {batch_size * beam_size, 1, seq_len1, seq_len2}, cross_attn_mask_}},
            // {"padding_offset", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size * beam_size}, nullptr}}
        };
        TensorMap cross_attention_output_tensors{
            {"attention_out",
             Tensor{MEMORY_GPU, data_type, {batch_size * beam_size, seq_len1, hidden_units_}, cross_attn_output_}}};
        cross_attention_layer_->forward(&cross_attention_output_tensors,
                                        &cross_attention_input_tensors,
                                        &decoder_weight->decoder_layer_weights[l]->cross_attention_weights);

        invokeGeneralScaleAddBiasResidualPreLayerNorm(
            cross_attn_output_,
            normed_cross_attn_output_,
            self_attn_output_,
            decoder_weight->decoder_layer_weights[l]->cross_attn_layernorm_weights.gamma,
            decoder_weight->decoder_layer_weights[l]->cross_attn_layernorm_weights.beta,
            decoder_weight->decoder_layer_weights[l]->cross_attention_weights.attention_output_weight.bias,
            m,
            hidden_units_,
            stream_,
            2,
            1.0f,
            1.0f);
        sync_check_cuda_error();

        std::vector<Tensor> ffn_input_tensors{
            Tensor{MEMORY_GPU, data_type, {m, hidden_units_}, normed_cross_attn_output_}};
        std::vector<Tensor> ffn_output_tensors{
            Tensor{MEMORY_GPU, data_type, {m, hidden_units_}, decoder_layer_output_}};
        ffn_layer_->forward(
            &ffn_output_tensors, &ffn_input_tensors, &decoder_weight->decoder_layer_weights[l]->ffn_weights);

        invokeScaleAddBiasResidual(decoder_layer_output_,
                                   cross_attn_output_,
                                   decoder_weight->decoder_layer_weights[l]->ffn_weights.output_weight.bias,
                                   m,
                                   hidden_units_,
                                   stream_,
                                   1.0f,
                                   1.0f);
        sync_check_cuda_error();
    }
    T* decoder_output = output_tensors->getPtr<T>("decoder_output");
    invokeGeneralLayerNorm(decoder_output,
                           decoder_layer_output_,
                           decoder_weight->after_norm_weights.gamma,
                           decoder_weight->after_norm_weights.beta,
                           1e-6f,
                           m,
                           hidden_units_,
                           (float*)nullptr,
                           0,
                           stream_);
    sync_check_cuda_error();

    int n = vocab_size_;
    int k = hidden_units_;
    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          n,
                          m,
                          k,
                          decoder_weight->output_layer_weights.kernel,
                          n,
                          decoder_output,
                          k,
                          (T*)log_probs_buf_,
                          n);

    float* decoder_output_ptr = output_tensors->getPtr<float>("decoder_output");
    // add bias and log_softmax
    invokeBiasLogSoftmax<T>(decoder_output_ptr,
                            (T*)log_probs_buf_,
                            decoder_weight->output_layer_weights.bias,
                            nullptr,
                            seq_len1,
                            batch_size * beam_size,
                            vocab_size_,
                            vocab_size_,
                            true,
                            stream_);
    sync_check_cuda_error();

    // WARN: use half cause larger error
    invokeMaskDecoderOutput<float>(decoder_score_buf_,
                                   decoder_output_ptr,
                                   decoder_sequence_length,
                                   decoder_input,
                                   batch_size * beam_size,
                                   seq_len1,
                                   vocab_size_,
                                   stream_);
    invokeBuildBestIndex<T>(best_index, decoder_score_buf_, ctc_score, ctc_weight, batch_size, beam_size, stream_);

    sync_check_cuda_error();

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template class WenetDecoder<float>;
template class WenetDecoder<half>;

}  // namespace fastertransformer