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

#include "src/fastertransformer/layers/attention_layers_fp8/DecoderSelfAttentionFP8Layer.h"
#include "src/fastertransformer/kernels/decoder_masked_multihead_attention.h"
#include "src/fastertransformer/utils/cuda_fp8_utils.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include "src/fastertransformer/utils/nvtx_utils.h"

namespace fastertransformer {

template<typename T1>
void fusedQKV_masked_attention_dispatch(const T1*    qkv_buf,
                                        const T1*    qkv_bias,
                                        const T1*    relative_attention_bias,
                                        const float* query_weight_output_scale,
                                        const float* attention_qk_scale,
                                        const float* attention_output_weight_input_scale_inv,
                                        T1*          key_cache,
                                        T1*          value_cache,
                                        const int*   cache_indir,
                                        T1*          context_buf,
                                        const bool*  finished,
                                        const int*   sequence_lengths,
                                        const int    max_batch_size,
                                        const int    inference_batch_size,
                                        const int    beam_width,
                                        const int    head_num,
                                        const int    size_per_head,
                                        const int    rotary_embedding_dim,
                                        const bool   neox_rotary_style,
                                        const int    memory_max_len,
                                        const int*   prefix_prompt_lengths,
                                        const int    max_prefix_prompt_length,
                                        const int    max_input_len,
                                        const int*   total_padding_tokens,
                                        const int    step,
                                        const float  q_scaling,
                                        const int    relative_attention_bias_stride,
                                        const T1*    linear_bias_slopes,
                                        const bool*  masked_tokens,
                                        const int*   ia3_tasks,
                                        const T1*    ia3_key_weights,
                                        const T1*    ia3_value_weights,
                                        cudaStream_t stream)
{
    using DataType = typename std::conditional<sizeof(T1) == 1, __nv_fp8_e4m3, __nv_bfloat16>::type;
    // using DataType = typename std::conditional<sizeof(T1) == 4, float, __nv_bfloat16>::type;
    Masked_multihead_attention_params<DataType> params;
    memset(&params, 0, sizeof(params));
    int hidden_units = head_num * size_per_head;
    if (qkv_bias != nullptr) {
        params.q_bias = reinterpret_cast<const DataType*>(qkv_bias);
        params.k_bias = reinterpret_cast<const DataType*>(qkv_bias) + hidden_units;
        params.v_bias = reinterpret_cast<const DataType*>(qkv_bias) + 2 * hidden_units;
    }
    else {
        params.q_bias = nullptr;
        params.k_bias = nullptr;
        params.v_bias = nullptr;
    }

    // Set the output buffer.
    params.out = reinterpret_cast<DataType*>(context_buf);

    // Set the input buffers.
    params.q        = reinterpret_cast<const DataType*>(qkv_buf);
    params.k        = reinterpret_cast<const DataType*>(qkv_buf) + hidden_units;
    params.v        = reinterpret_cast<const DataType*>(qkv_buf) + 2 * hidden_units;
    params.stride   = 3 * hidden_units;
    params.finished = const_cast<bool*>(finished);

    // Scale
    params.query_weight_output_scale               = query_weight_output_scale;
    params.attention_qk_scale                      = attention_qk_scale;
    params.attention_output_weight_input_scale_inv = attention_output_weight_input_scale_inv;

    params.k_cache                  = reinterpret_cast<DataType*>(key_cache);
    params.v_cache                  = reinterpret_cast<DataType*>(value_cache);
    params.cache_indir              = cache_indir;
    params.batch_size               = inference_batch_size;
    params.beam_width               = beam_width;
    params.memory_max_len           = memory_max_len;
    params.prefix_prompt_lengths    = prefix_prompt_lengths;
    params.max_prefix_prompt_length = max_prefix_prompt_length;
    params.length_per_sample        = sequence_lengths;  // max_input_length + current output length
    // timestep adding max_prefix_prompt_length for shared memory size calculation and rotary embedding computation
    params.timestep             = step + max_prefix_prompt_length - 1;
    params.num_heads            = head_num;
    params.hidden_size_per_head = size_per_head;
    params.rotary_embedding_dim = rotary_embedding_dim;
    params.neox_rotary_style    = neox_rotary_style;
    // Note: keep norm factor (sqrt(K_dim)) when adopting megatron T5 structure (may adjust)
    params.inv_sqrt_dh = 1.F / (sqrtf((float)params.hidden_size_per_head) * q_scaling);

    params.total_padding_tokens = total_padding_tokens;
    // TODO(bhsueh) Need better implementation
    FT_CHECK_WITH_INFO(rotary_embedding_dim == 0, "FP8 decoder still not support rotary embedding");
    if (relative_attention_bias != nullptr) {
        if (std::is_same<T1, __nv_fp8_e4m3>::value) {
            params.relative_attention_bias = reinterpret_cast<const DataType*>(relative_attention_bias);
        }
        else {
            FT_CHECK(false);
        }
    }
    params.relative_attention_bias_stride = relative_attention_bias_stride;
    params.masked_tokens                  = masked_tokens;

    if (linear_bias_slopes != nullptr) {
        params.linear_bias_slopes = reinterpret_cast<const DataType*>(linear_bias_slopes);
    }
    params.max_input_length = max_input_len;

    params.ia3_tasks         = ia3_tasks;
    params.ia3_key_weights   = reinterpret_cast<const DataType*>(ia3_key_weights);
    params.ia3_value_weights = reinterpret_cast<const DataType*>(ia3_value_weights);

    masked_multihead_attention(params, stream);
}

template<typename T1, typename T2>
void DecoderSelfAttentionFP8Layer<T1, T2>::allocateBuffer()
{
    // deprecated
    FT_CHECK(false);
}

template<typename T1, typename T2>
void DecoderSelfAttentionFP8Layer<T1, T2>::allocateBuffer(size_t batch_size)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
#ifdef FP8_MHA
    qkv_buf_ =
        reinterpret_cast<T1*>(allocator_->reMalloc(qkv_buf_, sizeof(T1) * batch_size * 3 * local_hidden_units_, false));
#else
    qkv_buf_ =
        reinterpret_cast<T2*>(allocator_->reMalloc(qkv_buf_, sizeof(T2) * batch_size * 3 * local_hidden_units_, false));
#endif
    context_buf_ =
        reinterpret_cast<T1*>(allocator_->reMalloc(context_buf_, sizeof(T1) * batch_size * local_hidden_units_, false));
    high_precision_context_buf_ = reinterpret_cast<T2*>(
        allocator_->reMalloc(high_precision_context_buf_, sizeof(T2) * batch_size * local_hidden_units_, false));
    is_allocate_buffer_ = true;
}

template<typename T1, typename T2>
void DecoderSelfAttentionFP8Layer<T1, T2>::freeBuffer()
{
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&qkv_buf_));
        allocator_->free((void**)(&context_buf_));
        allocator_->free((void**)(&high_precision_context_buf_));

        if (bf16_key_cache != nullptr) {
            deviceFree(bf16_key_cache);
        }
        if (bf16_value_cache != nullptr) {
            deviceFree(bf16_value_cache);
        }
        is_allocate_buffer_ = false;
    }
}

template<typename T1, typename T2>
DecoderSelfAttentionFP8Layer<T1, T2>::DecoderSelfAttentionFP8Layer(size_t           head_num,
                                                                   size_t           size_per_head,
                                                                   size_t           local_head_num,
                                                                   size_t           rotary_embedding_dim,
                                                                   bool             neox_rotary_style,
                                                                   size_t           d_model,
                                                                   const float      q_scaling,
                                                                   cudaStream_t     stream,
                                                                   cublasMMWrapper* cublas_wrapper,
                                                                   IAllocator*      allocator,
                                                                   bool             is_free_buffer_after_forward,
                                                                   bool             sparse):
    BaseAttentionLayer<T1>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num_ * size_per_head_),
    local_head_num_(local_head_num),
    local_hidden_units_(local_head_num_ * size_per_head_),
    rotary_embedding_dim_(rotary_embedding_dim),
    neox_rotary_style_(neox_rotary_style),
    d_model_(d_model),
    q_scaling_(q_scaling)
{
    FT_CHECK(size_per_head_ == 32 || size_per_head_ == 64 || size_per_head_ == 96 || size_per_head_ == 128
             || size_per_head_ == 160 || size_per_head_ == 192 || size_per_head_ == 224 || size_per_head_ == 256);
}

template<typename T1, typename T2>
DecoderSelfAttentionFP8Layer<T1, T2>::DecoderSelfAttentionFP8Layer(
    DecoderSelfAttentionFP8Layer<T1, T2> const& attention_layer):
    DecoderSelfAttentionFP8Layer<T1, T2>(attention_layer.head_num_,
                                         attention_layer.size_per_head_,
                                         attention_layer.local_head_num_,
                                         attention_layer.rotary_embedding_dim_,
                                         attention_layer.neox_rotary_style_,
                                         attention_layer.d_model_,
                                         attention_layer.q_scaling_,
                                         attention_layer.stream_,
                                         attention_layer.cublas_wrapper_,
                                         attention_layer.allocator_,
                                         attention_layer.is_free_buffer_after_forward_,
                                         attention_layer.sparse_)
{
}

template<typename T1, typename T2>
DecoderSelfAttentionFP8Layer<T1, T2>::~DecoderSelfAttentionFP8Layer()
{
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T1, typename T2>
void DecoderSelfAttentionFP8Layer<T1, T2>::forward(TensorMap*                 output_tensors,
                                                   TensorMap*                 input_tensors,
                                                   const AttentionWeight<T1>* attention_weights)
{
    // input tensors:
    //      attention_input [batch_size, d_model_],
    //      finished [batch_size],
    //      sequence_lengths [batch_size]
    //      max_input_length [1] on cpu
    //      total_padding_tokens [batch_size] (optional)
    //      masked_tokens [batch_size, memory_len], (optional)
    //      step [1] on cpu
    //      cache_indirection [batch_size / beam_width, beam_width, memory_max_len], optional
    //      relative_attention_bias [1, head_num, step, step] or [1, head_num, memory_max_len, memory_max_len]
    //      (optional) d_prefix_prompt_lengths [batch_size] (optional) max_prefix_prompt_length [1] on cpu (optional)
    //      linear_bias_slopes [head_num] (optional)

    // output tensors:
    //      attention_output [batch_size, d_model_],
    //      key_cache [batch, local_head_num, size_per_head // x, memory_max_len, x]
    //      value_cache [batch, local_head_num, memory_max_len, size_per_head]

    const AttentionFP8Weight<T1, T2>* attention_weights_ptr =
        reinterpret_cast<const AttentionFP8Weight<T1, T2>*>(attention_weights);

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() == 7 || input_tensors->size() == 8);
    FT_CHECK(output_tensors->size() == 3);
    FT_CHECK(output_tensors->at("key_cache").shape.size() == 5 || output_tensors->at("key_cache").shape.size() == 3);
    FT_CHECK(output_tensors->at("value_cache").shape.size() == 4
             || output_tensors->at("value_cache").shape.size() == 3);
    allocateBuffer(input_tensors->at("attention_input").shape[0]);

    const T1*   attention_input  = input_tensors->getPtr<T1>("attention_input");
    const bool* finished         = input_tensors->getPtr<bool>("finished", nullptr);
    const int*  sequence_lengths = input_tensors->getPtr<int>("sequence_lengths");
    const bool* masked_tokens    = input_tensors->getPtr<bool>("masked_tokens", nullptr);
    const int*  cache_indir      = input_tensors->getPtr<int>("cache_indirection", nullptr);
#ifdef FP8_MHA
    const T1* relative_attention_bias = input_tensors->getPtr<T1>("relative_attention_bias", nullptr);
#else
    const T2* relative_attention_bias = input_tensors->getPtr<T2>("relative_attention_bias", nullptr);
#endif
    const int relative_attention_bias_stride =
        input_tensors->isExist("relative_attention_bias") ? input_tensors->at("relative_attention_bias").shape[3] : 0;

#ifdef FP8_MHA
    const T1* linear_bias_slopes = input_tensors->getPtr<T1>("linear_bias_slopes", nullptr);
#else
    const T2* linear_bias_slopes      = input_tensors->getPtr<T2>("linear_bias_slopes", nullptr);
#endif

    T2* attention_out = output_tensors->getPtr<T2>("attention_output");
#ifdef FP8_MHA
    T1* key_cache   = output_tensors->getPtr<T1>("key_cache");
    T1* value_cache = output_tensors->getPtr<T1>("value_cache");
#else
    T2*       key_cache               = output_tensors->getPtr<T2>("key_cache");
    T2*       value_cache             = output_tensors->getPtr<T2>("value_cache");
#endif
    const int batch_size     = input_tensors->at("attention_input").shape[0];
    const int beam_width     = cache_indir != nullptr ? input_tensors->at("cache_indirection").shape[1] : 1;
    const int memory_max_len = output_tensors->at("key_cache").shape[3];

    const int* d_prefix_prompt_lengths  = input_tensors->getPtr<int>("d_prefix_prompt_lengths", nullptr);
    const int  max_prefix_prompt_length = input_tensors->getVal<int>("max_prefix_prompt_length", 0);

#ifdef FP8_MHA

    using qkv_output_type = T1;
#else

    using qkv_output_type = T2;
#endif

#ifdef SPARSITY_ENABLED
    FT_CHECK_WITH_INFO(false, "DecoderSelfAttentionFP8Layer does not support sparse now.");
    const int m_padded = 8 * div_up(batch_size, 8);
    if (sparse_ && cublas_wrapper_->isUseSparse(1, 3 * local_hidden_units_, m_padded, d_model_)) {
        // cublas_wrapper_->SpGemm(CUBLAS_OP_N,
        //                         CUBLAS_OP_N,
        //                         3 * local_hidden_units_,
        //                         m_padded,
        //                         d_model_,
        //                         attention_weights->query_weight.sp_kernel,
        //                         attention_input,
        //                         qkv_buf_);
    }
    else {
#endif
        {
            const float alpha = 1.0f;
            const float beta  = 0.0f;
            reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_)
                ->Gemm((qkv_output_type*)qkv_buf_,
                       (int)1,
                       (int)batch_size,
                       (int)3 * local_hidden_units_,
                       (int)d_model_,
                       (int64_t)0,
                       (int64_t)0,
                       (int64_t)0,
                       &alpha,
                       &beta,
                       attention_input,
                       attention_weights_ptr->query_weight.kernel,
                       attention_weights_ptr->query_weight.input_scale,
                       attention_weights_ptr->query_weight.weight_scale,
#ifdef FP8_MHA
                       attention_weights_ptr->query_weight.output_scale_inv,
#endif
                       stream_);
        }
#ifdef SPARSITY_ENABLED
    }
#endif
    sync_check_cuda_error();

#ifdef FP8_MHA
    fusedQKV_masked_attention_dispatch<T1>(
        (qkv_output_type*)qkv_buf_,
        attention_weights_ptr->query_weight.fp8_bias,
        relative_attention_bias,
        attention_weights_ptr->query_weight.output_scale,
        attention_weights_ptr->qk_scale,
        attention_weights_ptr->attention_output_weight.input_scale_inv,
        key_cache,
        value_cache,
        cache_indir,
        context_buf_,
        finished,
        sequence_lengths,  // NOTE: current seq len including padding (fixed after meeting the finished id)
        batch_size,
        batch_size,
        beam_width,
        local_head_num_,
        size_per_head_,
        rotary_embedding_dim_,
        neox_rotary_style_,
        memory_max_len,
        d_prefix_prompt_lengths,
        max_prefix_prompt_length,
        input_tensors->getVal<int>("max_input_length", 0),
        input_tensors->getPtr<int>("total_padding_tokens", nullptr),
        input_tensors->getVal<int>("step"),
        q_scaling_,
        relative_attention_bias_stride,
        linear_bias_slopes,
        masked_tokens,
        input_tensors->getPtr<int>("ia3_tasks", nullptr),
        nullptr,  // ia3_key_weight
        nullptr,  // ia3_value_weight
        stream_);
    sync_check_cuda_error();
#else
    fusedQKV_masked_attention_dispatch<T2>(
        (qkv_output_type*)qkv_buf_,
        attention_weights_ptr->query_weight.bias,
        relative_attention_bias,
        (const float*)nullptr,
        (const float*)nullptr,
        (const float*)nullptr,
        key_cache,
        value_cache,
        cache_indir,
        high_precision_context_buf_,
        finished,
        sequence_lengths,  // NOTE: current seq len including padding (fixed after meeting the finished id)
        batch_size,
        batch_size,
        beam_width,
        local_head_num_,
        size_per_head_,
        rotary_embedding_dim_,
        neox_rotary_style_,
        memory_max_len,
        d_prefix_prompt_lengths,
        max_prefix_prompt_length,
        input_tensors->getVal<int>("max_input_length", 0),
        input_tensors->getPtr<int>("total_padding_tokens", nullptr),
        input_tensors->getVal<int>("step"),
        q_scaling_,
        relative_attention_bias_stride,
        linear_bias_slopes,
        masked_tokens,
        input_tensors->getPtr<int>("ia3_tasks", nullptr),
        nullptr,  // ia3_key_weight
        nullptr,  // ia3_value_weight
        stream_);
    sync_check_cuda_error();
    invokeQuantizeMatrix<T1, T2, QUANTIZE_MODE::PER_TENSOR>(
        context_buf_,
        attention_weights_ptr->attention_output_weight.input_scale_inv,
        high_precision_context_buf_,
        batch_size * local_head_num_ * size_per_head_,
        1,
        stream_);
    sync_check_cuda_error();
#endif

#ifdef SPARSITY_ENABLED
    FT_CHECK_WITH_INFO(false, "DecoderSelfAttentionFP8Layer does not support sparse now.");
    if (sparse_ && cublas_wrapper_->isUseSparse(1, d_model_, m_padded, local_hidden_units_)) {
        // cublas_wrapper_->SpGemm(CUBLAS_OP_N,
        //                         CUBLAS_OP_N,
        //                         d_model_,
        //                         m_padded,
        //                         local_hidden_units_,
        //                         attention_weights_->attention_output_weight.sp_kernel,
        //                         context_buf_,
        //                         attention_out);
    }
    else {
#endif

        const float alpha = 1.0f;
        const float beta  = 0.0f;
        reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_)
            ->Gemm(attention_out,
                   (int)1,
                   (int)batch_size,
                   (int)d_model_,
                   (int)local_hidden_units_,
                   (int64_t)0,
                   (int64_t)0,
                   (int64_t)0,
                   &alpha,
                   &beta,
                   context_buf_,
                   attention_weights_ptr->attention_output_weight.kernel,
                   attention_weights_ptr->attention_output_weight.input_scale,
                   attention_weights_ptr->attention_output_weight.weight_scale,
                   stream_);
        sync_check_cuda_error();
#ifdef SPARSITY_ENABLED
    }
#endif

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template class DecoderSelfAttentionFP8Layer<__nv_fp8_e4m3, __nv_bfloat16>;

}  // namespace fastertransformer
