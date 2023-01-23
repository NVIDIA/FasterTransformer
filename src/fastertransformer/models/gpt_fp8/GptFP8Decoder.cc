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

#include "src/fastertransformer/models/gpt_fp8/GptFP8Decoder.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/layernorm_fp8_kernels.h"

namespace fastertransformer {

template<typename T1, typename T2>
void GptFP8Decoder<T1, T2>::initialize()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    self_attention_layer_ = new TensorParallelDecoderSelfAttentionFP8Layer<T1, T2>(head_num_,
                                                                                   size_per_head_,
                                                                                   0,
                                                                                   false,
                                                                                   head_num_ * size_per_head_,
                                                                                   1.0f,
                                                                                   tensor_para_,
                                                                                   stream_,
                                                                                   cublas_wrapper_,
                                                                                   allocator_,
                                                                                   is_free_buffer_after_forward_,
                                                                                   sparse_);

    ffn_layer_ = new TensorParallelGeluFfnFP8Layer<T1, T2>(
        inter_size_, tensor_para_, 2, stream_, cublas_wrapper_, allocator_, is_free_buffer_after_forward_, sparse_);
}

template<typename T1, typename T2>
GptFP8Decoder<T1, T2>::GptFP8Decoder(size_t           head_num,
                                     size_t           size_per_head,
                                     size_t           inter_size,
                                     size_t           num_layer,
                                     NcclParam        tensor_para,
                                     NcclParam        pipeline_para,
                                     cudaStream_t     stream,
                                     cublasMMWrapper* cublas_wrapper,
                                     IAllocator*      allocator,
                                     bool             is_free_buffer_after_forward,
                                     bool             sparse):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    hidden_units_(head_num_ * size_per_head_),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para)
{
    initialize();
}

template<typename T1, typename T2>
GptFP8Decoder<T1, T2>::GptFP8Decoder(GptFP8Decoder<T1, T2> const& decoder):
    BaseLayer(decoder.stream_,
              decoder.cublas_wrapper_,
              decoder.allocator_,
              decoder.is_free_buffer_after_forward_,
              decoder.cuda_device_prop_,
              decoder.sparse_),
    head_num_(decoder.head_num_),
    size_per_head_(decoder.size_per_head_),
    inter_size_(decoder.inter_size_),
    num_layer_(decoder.num_layer_),
    hidden_units_(decoder.hidden_units_),
    tensor_para_(decoder.tensor_para_),
    pipeline_para_(decoder.pipeline_para_)
{
    initialize();
}

template<typename T1, typename T2>
void GptFP8Decoder<T1, T2>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T1, typename T2>
void GptFP8Decoder<T1, T2>::allocateBuffer(size_t batch_size)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    decoder_layer_output_ = reinterpret_cast<T2*>(
        allocator_->reMalloc(decoder_layer_output_, sizeof(T2) * batch_size * hidden_units_, false));
    decoder_normed_input_ = reinterpret_cast<T1*>(
        allocator_->reMalloc(decoder_normed_input_, sizeof(T1) * batch_size * hidden_units_, false));
    self_attn_output_ =
        reinterpret_cast<T2*>(allocator_->reMalloc(self_attn_output_, sizeof(T2) * batch_size * hidden_units_, false));
    normed_self_attn_output_ = reinterpret_cast<T1*>(
        allocator_->reMalloc(normed_self_attn_output_, sizeof(T1) * batch_size * hidden_units_, false));
    is_allocate_buffer_ = true;
}

template<typename T1, typename T2>
void GptFP8Decoder<T1, T2>::freeBuffer()
{
    if (is_allocate_buffer_) {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        allocator_->free((void**)(&decoder_layer_output_));
        allocator_->free((void**)(&decoder_normed_input_));
        allocator_->free((void**)(&self_attn_output_));
        allocator_->free((void**)(&normed_self_attn_output_));
        is_allocate_buffer_ = false;
    }
}

template<typename T1, typename T2>
bool GptFP8Decoder<T1, T2>::isValidLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l >= local_num_layer * pipeline_para_.rank_)
           && (l < local_num_layer * (pipeline_para_.rank_ + 1));
}

template<typename T1, typename T2>
bool GptFP8Decoder<T1, T2>::isFirstLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * pipeline_para_.rank_);
}

template<typename T1, typename T2>
bool GptFP8Decoder<T1, T2>::isLastLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * (pipeline_para_.rank_ + 1) - 1);
}

template<typename T1, typename T2>
int GptFP8Decoder<T1, T2>::getFirstLayerParallelId()
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return local_num_layer * pipeline_para_.rank_;
}

template<typename T1, typename T2>
GptFP8Decoder<T1, T2>::~GptFP8Decoder()
{
    delete self_attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T1, typename T2>
void GptFP8Decoder<T1, T2>::forward(TensorMap*                                            output_tensors,
                                    const TensorMap*                                      input_tensors,
                                    const std::vector<GptFP8DecoderLayerWeight<T1, T2>*>* gpt_decoder_layer_weight)
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

    // output tensors:
    //      decoder_output [local_batch_size, hidden_dimension],
    //      key_cache [num_layer, batch_size, head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [num_layer, batch_size, head_num, max_seq_len, size_per_head]

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    FT_CHECK(input_tensors->isExist("decoder_input"));
    FT_CHECK(input_tensors->isExist("finished"));
    FT_CHECK(input_tensors->isExist("input_lengths"));
    FT_CHECK(input_tensors->isExist("total_padding_tokens"));
    FT_CHECK(input_tensors->isExist("max_input_length"));
    FT_CHECK(input_tensors->isExist("step"));
    FT_CHECK(input_tensors->isExist("ite"));
    FT_CHECK(input_tensors->isExist("masked_tokens"));
    FT_CHECK(output_tensors->isExist("decoder_output"));
    FT_CHECK(output_tensors->isExist("key_cache"));
    FT_CHECK(output_tensors->isExist("value_cache"));

    const size_t local_batch_size = input_tensors->at("decoder_input").shape[0];
    allocateBuffer(local_batch_size);

    const DataType data_type           = getTensorType<T1>();
    const DataType high_precision_type = getTensorType<T2>();
    const int      ite                 = input_tensors->getVal<int>("ite");

    std::vector<size_t> self_k_cache_size;
    self_k_cache_size.push_back(local_batch_size);
    for (auto t = output_tensors->at("key_cache").shape.begin() + 2; t != output_tensors->at("key_cache").shape.end();
         ++t) {
        self_k_cache_size.push_back(*t);
    }
    std::vector<size_t> self_v_cache_size;
    self_v_cache_size.push_back(local_batch_size);
    for (auto t = output_tensors->at("value_cache").shape.begin() + 2;
         t != output_tensors->at("value_cache").shape.end();
         ++t) {
        self_v_cache_size.push_back(*t);
    }

    for (uint l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l) == false) {
            continue;
        }
        T2* decoder_input = (T2*)((l == 0) ? input_tensors->getPtr<T2>("decoder_input") : decoder_layer_output_);
        T2* decoder_output =
            (T2*)((l == num_layer_ - 1) ? output_tensors->getPtr<T2>("decoder_output") : decoder_layer_output_);

        if (isFirstLayerParallelId(l) == true && pipeline_para_.rank_ != 0 && pipeline_para_.world_size_ > 1) {
            // ftNcclRecv(decoder_input, local_batch_size * hidden_units_, pipeline_para_.rank_ - 1, pipeline_para_,
            // stream_);

            ftNcclRecv(decoder_input + local_batch_size * hidden_units_ / tensor_para_.world_size_ * tensor_para_.rank_,
                       local_batch_size * hidden_units_ / tensor_para_.world_size_,
                       pipeline_para_.rank_ - 1,
                       pipeline_para_,
                       stream_);
            if (tensor_para_.world_size_ > 1) {
                ftNcclAllGather(decoder_input,
                                decoder_input,
                                local_batch_size * hidden_units_ / tensor_para_.world_size_,
                                tensor_para_.rank_,
                                tensor_para_,
                                stream_);
            }
        }

        size_t cache_offset = l - getFirstLayerParallelId();
        for (auto t = output_tensors->at("key_cache").shape.begin() + 1;
             t != output_tensors->at("key_cache").shape.end();
             ++t) {
            cache_offset *= *t;
        };
        size_t ite_cache_offset = ite * local_batch_size;
        for (auto t = output_tensors->at("key_cache").shape.begin() + 2;
             t != output_tensors->at("key_cache").shape.end();
             ++t) {
            ite_cache_offset *= *t;
        }
        cache_offset += ite_cache_offset;

        {
            FP8LayerNormParam<T1, T2> param{
                decoder_normed_input_,
                (T2*)decoder_input,
                gpt_decoder_layer_weight->at(l)->pre_layernorm_weights.gamma,
                gpt_decoder_layer_weight->at(l)->pre_layernorm_weights.beta,
                gpt_decoder_layer_weight->at(l)->identity_scale,
                gpt_decoder_layer_weight->at(l)->self_attention_weights.query_weight.input_scale_inv,
                (int)local_batch_size,
                (int)hidden_units_,
                stream_,
                true};
            invokeFP8LayerNorm<T1, T2, 0>(param);
            sync_check_cuda_error();
        }

        TensorMap self_attention_input_tensors{
            {"attention_input",
             Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, decoder_normed_input_}},
            {"finished", input_tensors->at("finished")},
            {"sequence_lengths", input_tensors->at("input_lengths")},
            {"total_padding_tokens", input_tensors->at("total_padding_tokens")},
            {"max_input_length", input_tensors->at("max_input_length")},
            {"step", input_tensors->at("step")},
            {"masked_tokens", input_tensors->at("masked_tokens")}};
        if (input_tensors->isExist("cache_indirection")) {
            self_attention_input_tensors.insert("cache_indirection", input_tensors->at("cache_indirection"));
        }

#ifdef FP8_MHA
        TensorMap self_attention_output_tensors{
            {"attention_output",
             Tensor{MEMORY_GPU, high_precision_type, {local_batch_size, hidden_units_}, self_attn_output_}},
            {"key_cache",
             Tensor{MEMORY_GPU,
                    high_precision_type,
                    self_k_cache_size,
                    ((const T1*)output_tensors->at("key_cache").data) + cache_offset}},
            {"value_cache",
             Tensor{MEMORY_GPU,
                    high_precision_type,
                    self_v_cache_size,
                    ((const T1*)output_tensors->at("value_cache").data) + cache_offset}}};
#else
        TensorMap self_attention_output_tensors{
            {"attention_output",
             Tensor{MEMORY_GPU, high_precision_type, {local_batch_size, hidden_units_}, self_attn_output_}},
            {"key_cache",
             Tensor{MEMORY_GPU,
                    high_precision_type,
                    self_k_cache_size,
                    ((const T2*)output_tensors->at("key_cache").data) + cache_offset}},
            {"value_cache",
             Tensor{MEMORY_GPU,
                    high_precision_type,
                    self_v_cache_size,
                    ((const T2*)output_tensors->at("value_cache").data) + cache_offset}}};
#endif

        self_attention_layer_->forward(
            &self_attention_output_tensors,
            &self_attention_input_tensors,
            (const AttentionWeight<T1>*)&gpt_decoder_layer_weight->at(l)->self_attention_weights);

        {
            GeneralFP8AddBiasResidualPreLayerNormParam<T1, T2> param{
                normed_self_attn_output_,
                self_attn_output_,
                decoder_input,
                gpt_decoder_layer_weight->at(l)->self_attention_weights.attention_output_weight.bias,
                gpt_decoder_layer_weight->at(l)->self_attn_layernorm_weights.gamma,
                gpt_decoder_layer_weight->at(l)->self_attn_layernorm_weights.beta,
                nullptr,  // (const
                          // float*)gpt_decoder_layer_weight->at(l)->self_attention_weights.attention_output_weight.scale,
                (const float*)gpt_decoder_layer_weight->at(l)->ffn_weights.intermediate_weight.input_scale_inv,
                (int)local_batch_size,
                (int)hidden_units_,
                stream_,
                true};
            invokeGeneralFP8AddBiasResidualPreLayerNorm(param);
            sync_check_cuda_error();
        }

        TensorMap ffn_input_tensors = TensorMap(
            std::unordered_map<std::string, Tensor>{{"input_hidden_state",
                                                     Tensor{MEMORY_GPU,
                                                            data_type,
                                                            std::vector<size_t>{local_batch_size, hidden_units_},
                                                            normed_self_attn_output_}}});
        TensorMap ffn_output_tensors = TensorMap(
            std::unordered_map<std::string, Tensor>{{"output_hidden_state",
                                                     Tensor{MEMORY_GPU,
                                                            high_precision_type,
                                                            std::vector<size_t>{local_batch_size, hidden_units_},
                                                            decoder_output}}});

        ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &gpt_decoder_layer_weight->at(l)->ffn_weights);

        invokeAddBiasResidual(decoder_output,
                              self_attn_output_,
                              gpt_decoder_layer_weight->at(l)->ffn_weights.output_weight.bias,
                              local_batch_size,
                              hidden_units_,
                              stream_);
        sync_check_cuda_error();

        if (isLastLayerParallelId(l) == true && pipeline_para_.rank_ != pipeline_para_.world_size_ - 1
            && pipeline_para_.world_size_ > 1) {
            // ftNcclSend(decoder_output, local_batch_size * hidden_units_, pipeline_para_.rank_ + 1, pipeline_para_,
            // stream_);

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

template class GptFP8Decoder<__nv_fp8_e4m3, __nv_bfloat16>;

}  // namespace fastertransformer