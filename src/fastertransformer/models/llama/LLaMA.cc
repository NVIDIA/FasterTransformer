/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/llama/LLaMA.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/decoding_kernels.h"
#include "src/fastertransformer/kernels/llama_kernels.h"
#include "src/fastertransformer/layers/beam_search_layers/BaseBeamSearchLayer.h"
#include "src/fastertransformer/utils/llama_utils.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include <algorithm>
#include <type_traits>

namespace fastertransformer {

template<typename T>
void LLaMA<T>::initialize()
{
    check_cuda_error(cudaStreamCreateWithFlags(&comm_stream_, cudaStreamNonBlocking));
    for (int i = 0; i < num_buffers_; ++i) {
        check_cuda_error(cudaEventCreate(&kern_event_[i]));
        check_cuda_error(cudaEventCreate(&comm_event_[i]));
    }
    llama_context_decoder_ = new LLaMAContextDecoder<T>(head_num_,
                                                        size_per_head_,
                                                        inter_size_,
                                                        num_layer_,
                                                        rotary_embedding_dim_,
                                                        layernorm_eps_,
                                                        pipeline_para_,
                                                        stream_,
                                                        cublas_wrapper_,
                                                        allocator_,
                                                        is_free_buffer_after_forward_,
                                                        is_context_qk_buf_float_,
                                                        attention_type_);
}

template<typename T>
void LLaMA<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void LLaMA<T>::allocateBuffer(size_t batch_size, size_t seq_len, size_t max_seq_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    const size_t self_cache_size = (num_layer_ / pipeline_para_.world_size_) * batch_size * max_seq_len * hidden_units_;

    padding_offset_ =
        reinterpret_cast<int*>(allocator_->reMalloc(padding_offset_, sizeof(int) * batch_size * seq_len, false));
    cu_seqlens_ = reinterpret_cast<int*>(allocator_->reMalloc(cu_seqlens_, sizeof(int) * (batch_size + 1), false));

    input_attention_mask_ =
        (T*)(allocator_->reMalloc(input_attention_mask_, sizeof(T) * batch_size * max_seq_len * max_seq_len, false));

    key_cache_   = (T*)(allocator_->reMalloc(key_cache_, sizeof(T) * self_cache_size * 2, false));
    value_cache_ = key_cache_ + self_cache_size;

    context_decoder_input_buf_ =
        (T*)(allocator_->reMalloc(context_decoder_input_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false));
    context_decoder_output_buf_ = (T*)(allocator_->reMalloc(
        context_decoder_output_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false));

    for (int i = 0; i < num_buffers_; ++i) {
        context_decoder_output_buf_clone_[i] = (T*)(allocator_->reMalloc(
            context_decoder_output_buf_clone_[i], sizeof(T) * batch_size * max_seq_len * hidden_units_, false));
    }

    normed_decoder_output_buf_ =
        (T*)(allocator_->reMalloc(normed_decoder_output_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false));
    logits_buf_ =
        (float*)(allocator_->reMalloc(logits_buf_, sizeof(float) * batch_size * seq_len * vocab_size_, false));
    log_likelihood_buf_ =
        (float*)(allocator_->reMalloc(log_likelihood_buf_, sizeof(float) * batch_size * seq_len * vocab_size_, false));

    is_allocate_buffer_ = true;
}

template<typename T>
void LLaMA<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&padding_offset_));
        allocator_->free((void**)(&cu_seqlens_));
        allocator_->free((void**)(&input_attention_mask_));
        allocator_->free((void**)(&key_cache_));
        allocator_->free((void**)(&context_decoder_input_buf_));
        allocator_->free((void**)(&context_decoder_output_buf_));
        for (int i = 0; i < num_buffers_; ++i) {
            allocator_->free((void**)(&context_decoder_output_buf_clone_[i]));
        }
        allocator_->free((void**)(&logits_buf_));
        is_allocate_buffer_ = false;
    }
}

template<typename T>
LLaMA<T>::LLaMA(size_t           head_num,
                size_t           size_per_head,
                size_t           inter_size,
                size_t           num_layer,
                size_t           vocab_size,
                size_t           rotary_embedding_dim,
                size_t           random_seed,
                size_t           max_seq_len,
                cudaStream_t     stream,
                cublasMMWrapper* cublas_wrapper,
                IAllocator*      allocator,
                bool             is_free_buffer_after_forward,
                cudaDeviceProp*  cuda_device_prop,
                AttentionType    attention_type):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    vocab_size_(vocab_size),
    rotary_embedding_dim_(rotary_embedding_dim),
    random_seed_(random_seed),
    max_seq_len_(max_seq_len),
    hidden_units_(head_num * size_per_head),
    attention_type_(attention_type)
{
    pipeline_para_.world_size_ = 1;
    pipeline_para_.rank_       = 0;
    initialize();
}

template<typename T>
LLaMA<T>::LLaMA(size_t           head_num,
                size_t           size_per_head,
                size_t           inter_size,
                size_t           num_layer,
                size_t           vocab_size,
                size_t           rotary_embedding_dim,
                size_t           random_seed,
                size_t           max_seq_len,
                NcclParam        tensor_para,
                NcclParam        pipeline_para,
                cudaStream_t     stream,
                cublasMMWrapper* cublas_wrapper,
                IAllocator*      allocator,
                bool             is_free_buffer_after_forward,
                cudaDeviceProp*  cuda_device_prop,
                AttentionType    attention_type):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    vocab_size_(vocab_size),
    rotary_embedding_dim_(rotary_embedding_dim),
    random_seed_(random_seed),
    max_seq_len_(max_seq_len),
    hidden_units_(head_num * size_per_head),
    pipeline_para_(pipeline_para),
    attention_type_(attention_type)
{
    initialize();
}

template<typename T>
LLaMA<T>::LLaMA(LLaMA<T> const& llama):
    BaseLayer(llama),
    head_num_(llama.head_num_),
    size_per_head_(llama.size_per_head_),
    inter_size_(llama.inter_size_),
    num_layer_(llama.num_layer_),
    vocab_size_(llama.vocab_size_),
    rotary_embedding_dim_(llama.rotary_embedding_dim_),
    random_seed_(llama.random_seed_),
    max_seq_len_(llama.max_seq_len_),
    hidden_units_(llama.hidden_units_),
    pipeline_para_(llama.pipeline_para_),
    attention_type_(llama.attention_type_)
{
    initialize();
}

template<typename T>
LLaMA<T>::~LLaMA()
{
    check_cuda_error(cudaStreamDestroy(comm_stream_));
    for (int i = 0; i < num_buffers_; ++i) {
        check_cuda_error(cudaEventDestroy(kern_event_[i]));
        check_cuda_error(cudaEventDestroy(comm_event_[i]));
    }

    delete llama_context_decoder_;
    freeBuffer();
}

template<typename T>
void LLaMA<T>::forward(std::vector<Tensor>*       output_tensors,
                       const std::vector<Tensor>* input_tensors,
                       const LLaMAWeight<T>*      llama_weights)
{
    FT_CHECK(false);
}

template<typename T>
void LLaMA<T>::forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                       const std::unordered_map<std::string, Tensor>* input_tensors,
                       const LLaMAWeight<T>*                          llama_weights)
{
    //
    // input_tensors:
    //      input_ids [num_tokens]
    //      input_lengths [batch_size]
    //      context_lengths [batch_size]
    //      num_tokens [1] int on cpu
    //      seq_len [1] int on cpu
    //      attn_len [1] int on cpu
    //      is_context [1] int on cpu

    // output_tensors:
    //      hidden_vector [num_tokens, hidden_size]
    //      log_probs [num_tokens, vocab_size]
    //      cum_probs [batch_size]

    FT_CHECK_WITH_INFO(input_tensors->size() == 7, "input_tensors->size() == 6");
    FT_CHECK(input_tensors->at("input_ids").shape.size() == 2);
    FT_CHECK(input_tensors->at("input_lengths").shape.size() == 1);

    const DataType data_type       = getTensorType<T>();
    const bool     is_unpadded_mha = isUnPaddedMHA(attention_type_);
    const size_t   batch_size      = input_tensors->at("input_lengths").shape[0];
    const int*     input_ids       = input_tensors->at("input_ids").getPtr<int>();
    const int*     context_lengths = input_tensors->at("context_lengths").getPtr<int>();
    const int*     input_lengths   = input_tensors->at("input_lengths").getPtr<int>();
    const int      num_tokens      = input_tensors->at("num_tokens").getVal<int>();
    const int      seq_len         = input_tensors->at("seq_len").getVal<int>();
    const int      attn_len        = input_tensors->at("attn_len").getVal<int>();
    const int      is_context      = input_tensors->at("is_context").getVal<int>();
    T*             hidden_vector   = output_tensors->at("hidden_vector").getPtr<T>();
    float*         log_probs       = output_tensors->at("log_probs").getPtr<float>();
    float*         cum_probs       = output_tensors->at("cum_probs").getPtr<float>();

    allocateBuffer(batch_size, seq_len, max_seq_len_);
    sync_check_cuda_error();

    if (is_unpadded_mha) {
        invokeLLaMAGetPaddingOffsetAndCuSeqLens(
            padding_offset_, cu_seqlens_, input_lengths, batch_size, seq_len, stream_);
        sync_check_cuda_error();

        //       invokeRemovePadding(decoder_layer_output_, decoder_input, padding_offset_, h_token_num, hidden_units_,
        //       stream_);
        //        sync_check_cuda_error();
    }

    invokeLLaMABuildDecoderAttentionMask(
        input_attention_mask_, input_lengths, context_lengths, batch_size, seq_len, attn_len, stream_);
    sync_check_cuda_error();

    if (pipeline_para_.rank_ == 0) {
        invokeLLaMAInputIdsEmbeddingLookup(context_decoder_input_buf_,
                                           llama_weights->pre_decoder_embedding_table,
                                           input_ids,
                                           num_tokens,
                                           hidden_units_,
                                           stream_);
        sync_check_cuda_error();
    }
    else {
        //        ftNcclGroupStart();
        //        ftNcclRecv(
        //            context_decoder_input_buf_, num_tokens * hidden_units_, pipeline_para_.rank_ - 1, pipeline_para_,
        //            stream_);
        //        ftNcclGroupEnd();
        //        sync_check_cuda_error();
    }

    std::unordered_map<std::string, Tensor> decoder_input_tensors{
        {"decoder_input",
         Tensor{MEMORY_GPU,
                data_type,
                {(size_t)num_tokens, hidden_units_},
                pipeline_para_.rank_ == 0 ? context_decoder_input_buf_ : hidden_vector}},
        {"attention_mask",
         Tensor{MEMORY_GPU, data_type, {batch_size, 1, (size_t)seq_len, (size_t)(attn_len)}, input_attention_mask_}},
        {"input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size}, input_lengths}},
        {"context_lengths", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size}, context_lengths}},
        {"num_tokens", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &num_tokens}},
        {"seq_len", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &seq_len}},
        {"attn_len", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &attn_len}}};

    if (is_unpadded_mha) {
        decoder_input_tensors.insert(
            {"padding_offset", Tensor{MEMORY_GPU, TYPE_INT32, {(size_t)num_tokens}, padding_offset_}});
        decoder_input_tensors.insert(
            {"cu_seqlens", Tensor{MEMORY_GPU, TYPE_INT32, {size_t(batch_size + 1)}, cu_seqlens_}});
    }

    std::unordered_map<std::string, Tensor> decoder_output_tensors{
        {"decoder_output",
         Tensor{MEMORY_GPU,
                data_type,
                {(size_t)num_tokens, hidden_units_},
                (pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) ? context_decoder_output_buf_ :
                                                                           hidden_vector}},
        {"key_cache",
         Tensor{MEMORY_GPU,
                data_type,
                {num_layer_ / pipeline_para_.world_size_, batch_size, head_num_, max_seq_len_, size_per_head_},
                key_cache_}},
        {"value_cache",
         Tensor{MEMORY_GPU,
                data_type,
                {num_layer_ / pipeline_para_.world_size_, batch_size, head_num_, max_seq_len_, size_per_head_},
                value_cache_}}};

    llama_context_decoder_->forward(
        &decoder_output_tensors, &decoder_input_tensors, &llama_weights->decoder_layer_weights);
    sync_check_cuda_error();

    //    if (is_unpadded_mha) {
    //        invokeRebuildPadding(
    //            decoder_output, decoder_layer_output_, padding_offset_, h_token_num, hidden_units_, stream_);
    //        sync_check_cuda_error();
    //    }

    if (pipeline_para_.rank_ < pipeline_para_.world_size_ - 1) {
        //        buf_no_ = (buf_no_ + 1) % num_buffers_;
        //        check_cuda_error(cudaEventSynchronize(comm_event_[buf_no_]));
        //        invokeLLaMACopyKernel(
        //            context_decoder_output_buf_clone_[buf_no_], context_decoder_output_buf_, num_tokens *
        //            hidden_units_, stream_);
        //        sync_check_cuda_error();
        //        check_cuda_error(cudaEventRecord(kern_event_[buf_no_], stream_));
        //        check_cuda_error(cudaStreamWaitEvent(comm_stream_, kern_event_[buf_no_]));
        //        ftNcclGroupStart();
        //        ftNcclSend(context_decoder_output_buf_clone_[buf_no_],
        //                   num_tokens * hidden_units_,
        //                   pipeline_para_.rank_ + 1,
        //                   pipeline_para_,
        //                   comm_stream_);
        //        ftNcclGroupEnd();
        //        check_cuda_error(cudaEventRecord(comm_event_[buf_no_], comm_stream_));
        //        sync_check_cuda_error();
    }
    else if (!is_context){
        invokeGeneralLLaMALayerNorm(normed_decoder_output_buf_,
                                    context_decoder_output_buf_,
                                    llama_weights->post_decoder_layernorm.gamma,
                                    layernorm_eps_,
                                    num_tokens,
                                    hidden_units_,
                                    stream_);
        sync_check_cuda_error();

        float alpha = 1.0f;
        float beta  = 0.0f;
        cublas_wrapper_->setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUDA_R_32F);
        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              vocab_size_,
                              num_tokens,
                              hidden_units_,
                              llama_weights->post_decoder_embedding.kernel,
                              vocab_size_,
                              normed_decoder_output_buf_,
                              hidden_units_,  // n
                              logits_buf_,
                              vocab_size_);
        sync_check_cuda_error();
        cublas_wrapper_->setFP16GemmConfig();

        invokeLLaMALogSoftmax(log_probs, logits_buf_, num_tokens, vocab_size_, stream_);
        sync_check_cuda_error();

        invokeLLaMAGatherTokens(
            cum_probs, log_probs, input_ids, input_lengths, cu_seqlens_, batch_size, vocab_size_, stream_);
        sync_check_cuda_error();
    }
}

template<typename T>
size_t LLaMA<T>::getPipelineParallelRank()
{
    return pipeline_para_.rank_;
}

template<typename T>
size_t LLaMA<T>::getPipelineParallelSize()
{
    return pipeline_para_.world_size_;
}

template class LLaMA<float>;
template class LLaMA<half>;

}  // namespace fastertransformer
