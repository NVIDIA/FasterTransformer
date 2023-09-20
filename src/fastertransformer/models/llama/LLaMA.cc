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
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/layers/beam_search_layers/BaseBeamSearchLayer.h"
#include <algorithm>

namespace fastertransformer {

template<typename T>
void LLaMA<T>::initialize()
{
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
void LLaMA<T>::allocateBuffer(size_t batch_size, size_t max_seq_len, size_t max_cache_seq_len, size_t max_input_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    const size_t self_cache_size =
        (num_layer_ / pipeline_para_.world_size_) * batch_size * max_cache_seq_len * hidden_units_;

    input_attention_mask_ = (T*)(allocator_->reMalloc(
        input_attention_mask_, sizeof(T) * batch_size * max_seq_len * max_cache_seq_len, false));
    decoder_output_buf_ =
        (T*)(allocator_->reMalloc(decoder_output_buf_, sizeof(T) * batch_size * hidden_units_, false));
    // logits_buf_       = (float*)(allocator_->reMalloc(logits_buf_, sizeof(float) * batch_size * max_seq_len *
    // vocab_size_, false));

    key_cache_   = (T*)(allocator_->reMalloc(key_cache_, sizeof(T) * self_cache_size * 2, false));
    value_cache_ = key_cache_ + self_cache_size;

    tiled_input_ids_buf_ =
        (int*)(allocator_->reMalloc(tiled_input_ids_buf_, sizeof(int) * batch_size * max_input_len, false));
    tiled_input_lengths_buf_ = (int*)(allocator_->reMalloc(tiled_input_lengths_buf_, sizeof(int) * batch_size, false));

    context_decoder_input_buf_  = (T*)(allocator_->reMalloc(
        context_decoder_input_buf_, sizeof(T) * batch_size * max_input_len * hidden_units_, false));
    context_decoder_output_buf_ = (T*)(allocator_->reMalloc(
        context_decoder_output_buf_, sizeof(T) * batch_size * max_input_len * hidden_units_, false));

    is_allocate_buffer_ = true;
}

template<typename T>
void LLaMA<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&input_attention_mask_));
        allocator_->free((void**)(&decoder_output_buf_));
        // allocator_->free((void**)(&logits_buf_));

        allocator_->free((void**)(&key_cache_));
        if (cache_indirections_[0] != nullptr) {
            allocator_->free((void**)(&cache_indirections_)[0]);
        }

        allocator_->free((void**)(&tiled_input_ids_buf_));
        allocator_->free((void**)(&tiled_input_lengths_buf_));

        allocator_->free((void**)(&context_decoder_input_buf_));
        allocator_->free((void**)(&context_decoder_output_buf_));

        is_allocate_buffer_ = false;
    }
}

template<typename T>
LLaMA<T>::LLaMA(size_t             head_num,
                size_t             size_per_head,
                size_t             inter_size,
                size_t             num_layer,
                size_t             vocab_size,
                size_t             rotary_embedding_dim,
                unsigned long long random_seed,
                cudaStream_t       stream,
                cublasMMWrapper*   cublas_wrapper,
                IAllocator*        allocator,
                bool               is_free_buffer_after_forward,
                cudaDeviceProp*    cuda_device_prop,
                AttentionType      attention_type):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    vocab_size_(vocab_size),
    rotary_embedding_dim_(rotary_embedding_dim),
    hidden_units_(head_num * size_per_head),
    attention_type_(attention_type)
{
    pipeline_para_.world_size_ = 1;
    pipeline_para_.rank_       = 0;
    initialize();
}

template<typename T>
LLaMA<T>::LLaMA(size_t             head_num,
                size_t             size_per_head,
                size_t             inter_size,
                size_t             num_layer,
                size_t             vocab_size,
                size_t             rotary_embedding_dim,
                unsigned long long random_seed,
                NcclParam          tensor_para,
                NcclParam          pipeline_para,
                cudaStream_t       stream,
                cublasMMWrapper*   cublas_wrapper,
                IAllocator*        allocator,
                bool               is_free_buffer_after_forward,
                cudaDeviceProp*    cuda_device_prop,
                AttentionType      attention_type):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    vocab_size_(vocab_size),
    rotary_embedding_dim_(rotary_embedding_dim),
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
    hidden_units_(llama.hidden_units_),
    pipeline_para_(llama.pipeline_para_),
    attention_type_(llama.attention_type_)
{
    initialize();
}

template<typename T>
LLaMA<T>::~LLaMA()
{
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
    // input_tensors:
    //      input_ids [batch_size, max_input_length]
    //      input_lengths [batch_size]
    //      output_seq_len [batch_size] on cpu
    //      min_length [1] or [batch_size] on cpu, optional, int
    //      random_seed [1] or [batch_size] on cpu, optional, unsigned long long int.
    //      max_cache_seq_len [batch_size] on cpu

    // output_tensors:
    //      output_logits [batch_size, max_output_seq_len, vocab_size]

    FT_CHECK_WITH_INFO(input_tensors->size() >= 3, "input_tensors->size() >= 3");
    FT_CHECK(input_tensors->at("input_ids").shape.size() == 2);
    FT_CHECK(input_tensors->at("input_lengths").shape.size() == 1);
    FT_CHECK(input_tensors->find("output_seq_len") != input_tensors->end()
             && input_tensors->at("output_seq_len").shape.size() == 1);

    const size_t batch_size = input_tensors->at("input_ids").shape[0];

    // NOTE: Prefix Prompt PreProcessing
    // get prefix_prompt_weight for each batch --> shape [batch, 1]
    // --> ptrs with shape [num_layers, 2, num_heads, perfix_seq_len, size_per_head]
    int max_input_length = input_tensors->at("input_ids").shape[1];

    // Prefix Soft Prompt
    const size_t max_output_seq_len = input_tensors->at("output_seq_len").max<uint32_t>();
    const size_t max_seq_len        = max_output_seq_len;
    // max cache seq len should include max prefix prompt length as it has k/v states
    const size_t max_cache_seq_len = input_tensors->at("max_cache_seq_len").max<uint32_t>();
    if (max_cache_seq_len < max_seq_len) {
        FT_LOG_WARNING("max_cache_seq_len (%d) is less than max_seq_len (%d). "
                       "Note that this reduces the memory cost of k/v cache, but may hurt the accuracy.",
                       max_cache_seq_len,
                       max_seq_len);
    }
    else if (max_cache_seq_len > max_seq_len) {
        FT_LOG_WARNING("max_cache_seq_len (%d) is larger than max_seq_len (%d). "
                       "This may lead to additional memory cost. Suggest to use smaller max_cache_seq_len.",
                       max_cache_seq_len,
                       max_seq_len);
    }
    const cudaDataType_t gemm_data_type = getCudaDataType<T>();

    allocateBuffer(batch_size, max_seq_len, max_cache_seq_len, max_input_length);
    sync_check_cuda_error();

    const DataType            data_type          = getTensorType<T>();
    const std::vector<size_t> self_k_cache_shape = {num_layer_ / pipeline_para_.world_size_,
                                                    batch_size,
                                                    head_num_,
                                                    size_per_head_ / (16 / sizeof(T)),
                                                    max_cache_seq_len,
                                                    16 / sizeof(T)};
    const std::vector<size_t> self_v_cache_shape = {
        num_layer_ / pipeline_para_.world_size_, batch_size, head_num_, max_cache_seq_len, size_per_head_};

    invokeTileGptInputs(tiled_input_ids_buf_,
                        tiled_input_lengths_buf_,
                        input_tensors->at("input_ids").getPtr<int>(),
                        input_tensors->at("input_lengths").getPtr<const int>(),
                        batch_size,
                        1,
                        max_input_length,
                        stream_);
    sync_check_cuda_error();

    invokeBuildDecoderAttentionMask(
        input_attention_mask_, tiled_input_lengths_buf_, nullptr, batch_size, max_input_length, 0, stream_);
    sync_check_cuda_error();

    if (pipeline_para_.rank_ == 0) {
        invokeInputIdsEmbeddingLookupPosEncoding(context_decoder_input_buf_,
                                                 nullptr,
                                                 llama_weights->pre_decoder_embedding_table,
                                                 llama_weights->position_encoding_table,
                                                 pPromptTuningParam<T>{},  // no p/prompt tuning
                                                 tiled_input_ids_buf_,
                                                 1,
                                                 max_input_length,
                                                 max_input_length,
                                                 batch_size,
                                                 hidden_units_,
                                                 stream_);
        sync_check_cuda_error();
    }

    std::unordered_map<std::string, Tensor> decoder_input_tensors{
        {"decoder_input",
         Tensor{
             MEMORY_GPU, data_type, {batch_size, (size_t)max_input_length, hidden_units_}, context_decoder_input_buf_}},
        {"attention_mask",
         Tensor{MEMORY_GPU,
                data_type,
                {batch_size, 1, (size_t)max_input_length, (size_t)(max_input_length)},
                input_attention_mask_}},
        {"input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size}, tiled_input_lengths_buf_}}};

    std::unordered_map<std::string, Tensor> decoder_output_tensors{
        {"decoder_output",
         Tensor{MEMORY_GPU,
                data_type,
                {batch_size, (size_t)max_input_length, hidden_units_},
                context_decoder_output_buf_}},
        {"key_cache", Tensor{MEMORY_GPU, data_type, self_k_cache_shape, key_cache_}},
        {"value_cache", Tensor{MEMORY_GPU, data_type, self_v_cache_shape, value_cache_}},
        {"last_token_hidden_units", Tensor{MEMORY_GPU, data_type, {batch_size, hidden_units_}, decoder_output_buf_}}};

    llama_context_decoder_->forward(
        &decoder_output_tensors, &decoder_input_tensors, &llama_weights->decoder_layer_weights);
    sync_check_cuda_error();

    if (pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) {
        invokeGeneralLLaMALayerNorm(context_decoder_input_buf_,
                                    context_decoder_output_buf_,
                                    llama_weights->post_decoder_layernorm.gamma,
                                    llama_weights->post_decoder_layernorm.beta,
                                    layernorm_eps_,
                                    batch_size * max_input_length,
                                    hidden_units_,
                                    stream_);
        sync_check_cuda_error();

        // FIXME: debugging
        T* output_logits = output_tensors->at("output_logits").getPtr<T>();
        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              vocab_size_,
                              batch_size * max_input_length,
                              hidden_units_,
                              llama_weights->post_decoder_embedding.kernel,
                              vocab_size_,
                              context_decoder_input_buf_,
                              hidden_units_,  // n
                              output_logits,
                              // logits_buf_,
                              vocab_size_);
        sync_check_cuda_error();
    }

    // sendTensorsToFirstPipelineNode(output_tensors, input_tensors);
}

template<typename T>
void LLaMA<T>::sendTensorsToFirstPipelineNode(std::unordered_map<std::string, Tensor>*       output_tensors,
                                              const std::unordered_map<std::string, Tensor>* input_tensors)
{
    NcclParam tensor_para(0, 1);

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (pipeline_para_.world_size_ == 1) {
        // throw errors when detected
        ftNcclStreamSynchronize(tensor_para, pipeline_para_, stream_);
        return;
    }
    const auto pp_rank = pipeline_para_.rank_;

    ftNcclGroupStart();
    for (auto const& it : *output_tensors) {
        if (it.second.data == nullptr) {
            continue;
        }

        if (pp_rank == pipeline_para_.world_size_ - 1) {
            ftNcclSend(it.second.getPtr<char>(), it.second.sizeBytes(), 0, pipeline_para_, stream_);
        }
        else if (pp_rank == 0) {
            ftNcclRecv(it.second.getPtr<char>(),
                       it.second.sizeBytes(),
                       pipeline_para_.world_size_ - 1,
                       pipeline_para_,
                       stream_);
        }
    }
    ftNcclGroupEnd();
    // throw errors when detected
    ftNcclStreamSynchronize(tensor_para, pipeline_para_, stream_);
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
