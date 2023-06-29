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

#include "src/fastertransformer/layers/attention_layers/DecoderSelfAttentionLayer.h"
#include "src/fastertransformer/kernels/decoder_masked_multihead_attention.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include "src/fastertransformer/utils/nvtx_utils.h"

namespace fastertransformer {

template<typename T>
struct SATypeConverter {
    using Type = T;
};

template<>
struct SATypeConverter<half> {
    using Type = uint16_t;
};

template<typename T>
void fusedQKV_masked_attention_dispatch(const T*     qkv_buf,
                                        const T*     qkv_bias,
                                        const T*     relative_attention_bias,
                                        T*           key_cache,
                                        T*           value_cache,
                                        const int*   cache_indir,
                                        T*           context_buf,
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
                                        const T*     linear_bias_slopes,
                                        const bool*  masked_tokens,
                                        const int*   ia3_tasks,
                                        const T*     ia3_key_weights,
                                        const T*     ia3_value_weights,
                                        const float* qkv_scale_out,
                                        const float* attention_out_scale,
                                        const int    int8_mode,
                                        cudaStream_t stream)
{
    using DataType = typename SATypeConverter<T>::Type;
    // Prepare the parameters.
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
    params.q = reinterpret_cast<const DataType*>(qkv_buf);
    if (int8_mode != 2) {
        params.k = reinterpret_cast<const DataType*>(qkv_buf) + hidden_units;
        params.v = reinterpret_cast<const DataType*>(qkv_buf) + 2 * hidden_units;
    }
    else {
        params.k = reinterpret_cast<const DataType*>(reinterpret_cast<const int8_t*>(qkv_buf) + hidden_units);
        params.v = reinterpret_cast<const DataType*>(reinterpret_cast<const int8_t*>(qkv_buf) + 2 * hidden_units);
    }
    params.stride   = 3 * hidden_units;
    params.finished = const_cast<bool*>(finished);

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
    if (relative_attention_bias != nullptr) {
        params.relative_attention_bias = reinterpret_cast<const DataType*>(relative_attention_bias);
    }
    params.relative_attention_bias_stride = relative_attention_bias_stride;
    params.masked_tokens                  = masked_tokens;

    // The slope of linear position bias per head, e.g., ALiBi.
    if (linear_bias_slopes != nullptr) {
        params.linear_bias_slopes = reinterpret_cast<const DataType*>(linear_bias_slopes);
    }
    params.max_input_length = max_input_len;

    params.ia3_tasks         = ia3_tasks;
    params.ia3_key_weights   = reinterpret_cast<const DataType*>(ia3_key_weights);
    params.ia3_value_weights = reinterpret_cast<const DataType*>(ia3_value_weights);

    params.int8_mode = int8_mode;
    if (int8_mode == 2) {
        params.qkv_scale_out       = qkv_scale_out;
        params.attention_out_scale = attention_out_scale;
    }

    PUSH_RANGE("scaled dot-product fusion");
    masked_multihead_attention(params, stream);
    POP_RANGE;
}

#define INSTANTIATE_FUSEDQKV_MASKED_ATTENTION_DISPATCH(T)                                                              \
    template void fusedQKV_masked_attention_dispatch(const T*     qkv_buf,                                             \
                                                     const T*     qkv_bias,                                            \
                                                     const T*     relative_attention_bias,                             \
                                                     T*           key_cache,                                           \
                                                     T*           value_cache,                                         \
                                                     const int*   cache_indir,                                         \
                                                     T*           context_buf,                                         \
                                                     const bool*  finished,                                            \
                                                     const int*   sequence_lengths,                                    \
                                                     const int    max_batch_size,                                      \
                                                     const int    inference_batch_size,                                \
                                                     const int    beam_width,                                          \
                                                     const int    head_num,                                            \
                                                     const int    size_per_head,                                       \
                                                     const int    rotary_embedding_dim,                                \
                                                     const bool   neox_rotary_style,                                   \
                                                     const int    memory_max_len,                                      \
                                                     const int*   prefix_prompt_lengths,                               \
                                                     const int    max_prefix_prompt_length,                            \
                                                     const int    max_input_len,                                       \
                                                     const int*   total_padding_tokens,                                \
                                                     const int    step,                                                \
                                                     const float  q_scaling,                                           \
                                                     const int    relative_attention_bias_stride,                      \
                                                     const T*     linear_bias_slopes,                                  \
                                                     const bool*  masked_tokens,                                       \
                                                     const int*   ia3_tasks,                                           \
                                                     const T*     ia3_key_weights,                                     \
                                                     const T*     ia3_value_weights,                                   \
                                                     const float* qkv_scale_out,                                       \
                                                     const float* attention_out_scale,                                 \
                                                     const int    int8_mode,                                           \
                                                     cudaStream_t stream)

INSTANTIATE_FUSEDQKV_MASKED_ATTENTION_DISPATCH(float);
INSTANTIATE_FUSEDQKV_MASKED_ATTENTION_DISPATCH(half);
#ifdef ENABLE_BF16
INSTANTIATE_FUSEDQKV_MASKED_ATTENTION_DISPATCH(__nv_bfloat16);
#endif

#undef INSTANTIATE_FUSEDQKV_MASKED_ATTENTION_DISPATCH

template<typename T>
void DecoderSelfAttentionLayer<T>::allocateBuffer()
{
    FT_CHECK_WITH_INFO(false, "Deprecated. Use `allocateBuffer(size_t batch_size)` instead");
}

template<typename T>
void DecoderSelfAttentionLayer<T>::allocateBuffer(size_t batch_size)
{
    const size_t type_size = int8_mode_ == 2 ? sizeof(int8_t) : sizeof(T);
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    qkv_buf_ =
        reinterpret_cast<T*>(allocator_->reMalloc(qkv_buf_, type_size * batch_size * 3 * local_hidden_units_, false));
    context_buf_ =
        reinterpret_cast<T*>(allocator_->reMalloc(context_buf_, type_size * batch_size * local_hidden_units_, false));

    if (int8_mode_ == 1) {
        // We use max_size for n and k since we reuse buffers for both FCs and want to allocate the max
        // possible memory that would be required by any of the individual gemms.
        const int max_size    = std::max(d_model_, 3 * local_hidden_units_);
        mixed_gemm_ws_bytes_  = weight_only_int8_fc_runner_->getWorkspaceSize(batch_size, max_size, max_size);
        mixed_gemm_workspace_ = (char*)allocator_->reMalloc(mixed_gemm_workspace_, mixed_gemm_ws_bytes_, false);
    }
    else if (int8_mode_ == 2) {
        const int max_size   = std::max(d_model_, 3 * local_hidden_units_);
        int8_gemm_ws_bytes_  = int8_fc_runner_->getWorkspaceSize(batch_size, max_size, max_size);
        int8_gemm_workspace_ = (char*)allocator_->reMalloc(int8_gemm_workspace_, int8_gemm_ws_bytes_, false);
    }

    is_allocate_buffer_ = true;
}

template<typename T>
void DecoderSelfAttentionLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&qkv_buf_));
        allocator_->free((void**)(&context_buf_));
        is_allocate_buffer_ = false;

        if (mixed_gemm_workspace_) {
            allocator_->free((void**)(&mixed_gemm_workspace_));
            mixed_gemm_ws_bytes_ = 0;
        }
    }
}

template<typename T>
bool DecoderSelfAttentionLayer<T>::isValidBatchSize(size_t batch_size)
{
    if (batch_size <= max_batch_size_) {
        return true;
    }
    else {
        freeBuffer();
        max_batch_size_ = batch_size * 1.2;
        return true;
    }
}

template<typename T>
DecoderSelfAttentionLayer<T>::DecoderSelfAttentionLayer(size_t           max_batch_size,
                                                        size_t           head_num,
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
                                                        bool             sparse,
                                                        int              int8_mode):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse),
    max_batch_size_(max_batch_size),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num_ * size_per_head_),
    local_head_num_(local_head_num),
    local_hidden_units_(local_head_num_ * size_per_head_),
    rotary_embedding_dim_(rotary_embedding_dim),
    neox_rotary_style_(neox_rotary_style),
    d_model_(d_model),
    q_scaling_(q_scaling),
    int8_fc_runner_(int8_mode == 2 ? std::make_shared<CutlassInt8GemmRunner<T>>() : nullptr),
    int8_mode_(int8_mode)
{
    FT_CHECK(size_per_head_ == 32 || size_per_head_ == 48 || size_per_head_ == 64 || size_per_head_ == 80
             || size_per_head_ == 96 || size_per_head_ == 112 || size_per_head_ == 128 || size_per_head_ == 144
             || size_per_head_ == 160 || size_per_head_ == 192 || size_per_head_ == 224 || size_per_head_ == 256);
    if (int8_mode_ == 1) {
        FT_CHECK_WITH_INFO(!(std::is_same<T, float>::value), "Weight only quant not supported for fp32.");
        weight_only_int8_fc_runner_ = std::make_shared<CutlassFpAIntBGemmRunner<T, uint8_t>>();
    }
}

template<typename T>
DecoderSelfAttentionLayer<T>::DecoderSelfAttentionLayer(size_t           max_batch_size,
                                                        size_t           head_num,
                                                        size_t           size_per_head,
                                                        cudaStream_t     stream,
                                                        cublasMMWrapper* cublas_wrapper,
                                                        IAllocator*      allocator,
                                                        bool             is_free_buffer_after_forward,
                                                        bool             sparse,
                                                        int              int8_mode):
    DecoderSelfAttentionLayer<T>(max_batch_size,
                                 head_num,
                                 size_per_head,
                                 head_num,
                                 0,
                                 false,
                                 head_num * size_per_head,
                                 1.0f,
                                 stream,
                                 cublas_wrapper,
                                 allocator,
                                 is_free_buffer_after_forward,
                                 sparse,
                                 int8_mode)
{
}

template<typename T>
DecoderSelfAttentionLayer<T>::DecoderSelfAttentionLayer(size_t           max_batch_size,
                                                        size_t           head_num,
                                                        size_t           size_per_head,
                                                        const float      q_scaling,
                                                        cudaStream_t     stream,
                                                        cublasMMWrapper* cublas_wrapper,
                                                        IAllocator*      allocator,
                                                        bool             is_free_buffer_after_forward,
                                                        bool             sparse,
                                                        int              int8_mode):
    DecoderSelfAttentionLayer<T>(max_batch_size,
                                 head_num,
                                 size_per_head,
                                 head_num,
                                 0,
                                 false,
                                 head_num * size_per_head,
                                 q_scaling,
                                 stream,
                                 cublas_wrapper,
                                 allocator,
                                 is_free_buffer_after_forward,
                                 sparse,
                                 int8_mode)
{
}

template<typename T>
DecoderSelfAttentionLayer<T>::DecoderSelfAttentionLayer(size_t           max_batch_size,
                                                        size_t           head_num,
                                                        size_t           size_per_head,
                                                        size_t           local_head_num,
                                                        cudaStream_t     stream,
                                                        cublasMMWrapper* cublas_wrapper,
                                                        IAllocator*      allocator,
                                                        bool             is_free_buffer_after_forward,
                                                        bool             sparse,
                                                        int              int8_mode):
    DecoderSelfAttentionLayer<T>(max_batch_size,
                                 head_num,
                                 size_per_head,
                                 local_head_num,
                                 0,
                                 false,
                                 head_num * size_per_head,
                                 1.0f,
                                 stream,
                                 cublas_wrapper,
                                 allocator,
                                 is_free_buffer_after_forward,
                                 sparse,
                                 int8_mode)
{
}

template<typename T>
DecoderSelfAttentionLayer<T>::DecoderSelfAttentionLayer(size_t           max_batch_size,
                                                        size_t           head_num,
                                                        size_t           size_per_head,
                                                        size_t           local_head_num,
                                                        size_t           d_model,
                                                        const float      q_scaling,
                                                        cudaStream_t     stream,
                                                        cublasMMWrapper* cublas_wrapper,
                                                        IAllocator*      allocator,
                                                        bool             is_free_buffer_after_forward,
                                                        bool             sparse,
                                                        int              int8_mode):
    DecoderSelfAttentionLayer<T>(max_batch_size,
                                 head_num,
                                 size_per_head,
                                 local_head_num,
                                 0,
                                 false,
                                 d_model,
                                 q_scaling,
                                 stream,
                                 cublas_wrapper,
                                 allocator,
                                 is_free_buffer_after_forward,
                                 sparse,
                                 int8_mode)
{
}

template<typename T>
DecoderSelfAttentionLayer<T>::DecoderSelfAttentionLayer(size_t           max_batch_size,
                                                        size_t           head_num,
                                                        size_t           size_per_head,
                                                        size_t           local_head_num,
                                                        size_t           rotary_embedding_dim,
                                                        bool             neox_rotary_style,
                                                        cudaStream_t     stream,
                                                        cublasMMWrapper* cublas_wrapper,
                                                        IAllocator*      allocator,
                                                        bool             is_free_buffer_after_forward,
                                                        bool             sparse,
                                                        int              int8_mode):
    DecoderSelfAttentionLayer<T>(max_batch_size,
                                 head_num,
                                 size_per_head,
                                 local_head_num,
                                 rotary_embedding_dim,
                                 neox_rotary_style,
                                 head_num * size_per_head,
                                 1.0f,
                                 stream,
                                 cublas_wrapper,
                                 allocator,
                                 is_free_buffer_after_forward,
                                 sparse,
                                 int8_mode)
{
}

template<typename T>
DecoderSelfAttentionLayer<T>::DecoderSelfAttentionLayer(DecoderSelfAttentionLayer<T> const& attention_layer):
    DecoderSelfAttentionLayer<T>(attention_layer.max_batch_size_,
                                 attention_layer.head_num_,
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
                                 attention_layer.sparse_,
                                 attention_layer.int8_mode_)
{
}

template<typename T>
DecoderSelfAttentionLayer<T>::~DecoderSelfAttentionLayer()
{
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void DecoderSelfAttentionLayer<T>::forward(TensorMap*                output_tensors,
                                           TensorMap*                input_tensors,
                                           const AttentionWeight<T>* attention_weights)
{
    // input tensors:
    //      input_query [batch_size, d_model_],
    //      sequence_lengths [batch_size]
    //      step [1] on cpu
    //      finished [batch_size] (optional)
    //      total_padding_tokens [batch_size] (optional)
    //      max_input_length [1] on cpu (optional)
    //      masked_tokens [batch_size, memory_len], (optional)
    //      cache_indirection [batch_size / beam_width, beam_width, memory_max_len] (optional)
    //      d_prefix_prompt_lengths [batch_size] (optional)
    //      max_prefix_prompt_length [1] on cpu (optional)
    //      relative_attention_bias [1, head_num, step, step] or [1, head_num, max_seq_len, max_seq_len] (optional)
    //      linear_bias_slopes [head_num] (optional)
    //      ia3_tasks [batch_size] (optional)

    // output tensors:
    //      attention_output [batch_size, d_model_],
    //      key_cache [batch, local_head_num, size_per_head // x, memory_max_len, x]
    //      value_cache [batch, local_head_num, memory_max_len, size_per_head]

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(output_tensors->at("key_cache").shape.size() == 5 || output_tensors->at("key_cache").shape.size() == 3);
    FT_CHECK(output_tensors->at("value_cache").shape.size() == 4
             || output_tensors->at("value_cache").shape.size() == 3);
    allocateBuffer(input_tensors->at("input_query").shape[0]);

    const T*    attention_input         = input_tensors->getPtr<T>("input_query");
    const int*  sequence_lengths        = input_tensors->getPtr<int>("sequence_lengths");
    const bool* finished                = input_tensors->getPtr<bool>("finished", nullptr);
    const bool* masked_tokens           = input_tensors->getPtr<bool>("masked_tokens", nullptr);
    const int*  cache_indir             = input_tensors->getPtr<int>("cache_indirection", nullptr);
    const T*    relative_attention_bias = input_tensors->getPtr<T>("relative_attention_bias", nullptr);
    const int   relative_attention_bias_stride =
        input_tensors->isExist("relative_attention_bias") ? input_tensors->at("relative_attention_bias").shape[3] : 0;
    const T*   linear_bias_slopes = input_tensors->getPtr<T>("linear_bias_slopes", nullptr);
    const bool has_ia3            = input_tensors->isExist("ia3_tasks");

    T* attention_out = output_tensors->getPtr<T>("hidden_features");
    T* key_cache     = output_tensors->getPtr<T>("key_cache");
    T* value_cache   = output_tensors->getPtr<T>("value_cache");

    const int batch_size     = input_tensors->at("input_query").shape[0];
    const int beam_width     = cache_indir != nullptr ? input_tensors->at("cache_indirection").shape[1] : 1;
    const int memory_max_len = output_tensors->at("key_cache").shape[3];

    const int* d_prefix_prompt_lengths  = input_tensors->getPtr<int>("d_prefix_prompt_lengths", nullptr);
    const int  max_prefix_prompt_length = input_tensors->getVal<int>("max_prefix_prompt_length", 0);

    const int m_padded = 8 * div_up(batch_size, 8);
#ifdef SPARSITY_ENABLED
    bool use_sparse_gemm = sparse_ && cublas_wrapper_->isUseSparse(1, 3 * local_hidden_units_, m_padded, d_model_);
#else
    constexpr bool use_sparse_gemm = false;
#endif

    PUSH_RANGE("qkv_gemm");
    if (use_sparse_gemm) {
#ifdef SPARSITY_ENABLED
        cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                3 * local_hidden_units_,
                                m_padded,
                                d_model_,
                                attention_weights->query_weight.sp_kernel,
                                attention_input,
                                qkv_buf_);
#endif
    }
    else {
        if (int8_mode_ == 1) {
            FT_CHECK(weight_only_int8_fc_runner_.get() != NULL && attention_weights->query_weight.int8_kernel != NULL
                     && attention_weights->query_weight.weight_only_quant_scale != NULL);

            weight_only_int8_fc_runner_->gemm(
                attention_input,
                reinterpret_cast<const uint8_t*>(attention_weights->query_weight.int8_kernel),
                attention_weights->query_weight.weight_only_quant_scale,
                qkv_buf_,
                batch_size,
                3 * local_hidden_units_,
                d_model_,
                mixed_gemm_workspace_,
                mixed_gemm_ws_bytes_,
                stream_);
        }
        else if (int8_mode_ == 2) {
            // Here, we set per_column_scaling to be true because q, k, v may
            // use different scales. So, we pass a pointer with shape [3, local_hidden_units_] like
            // [s_q, s_q, ..., s_q, s_k, s_k, ..., s_k, s_v, s_v, ..., s_v],
            // where s_q are scales of q, s_k are scales of k and s_v are scales of v.
            cublas_wrapper_->Int8Gemm(3 * local_hidden_units_,
                                      batch_size,
                                      d_model_,
                                      attention_weights->query_weight.int8_kernel,
                                      d_model_,
                                      input_tensors->getPtr<int8_t>("input_query"),
                                      d_model_,
                                      reinterpret_cast<int8_t*>(qkv_buf_),
                                      3 * local_hidden_units_,
                                      attention_weights->query_weight.scale_inter,
                                      true);
        }
        else {
            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  3 * local_hidden_units_,  // n
                                  batch_size,
                                  d_model_,  // k
                                  attention_weights->query_weight.kernel,
                                  3 * local_hidden_units_,  // n
                                  attention_input,
                                  d_model_,  // k
                                  qkv_buf_,
                                  3 * local_hidden_units_ /* n */);
        }
    }
    sync_check_cuda_error();
    POP_RANGE;
    fusedQKV_masked_attention_dispatch<T>(
        qkv_buf_,
        attention_weights->query_weight.bias,
        relative_attention_bias,
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
        has_ia3 ? attention_weights->ia3_key_weight.kernel : nullptr,
        has_ia3 ? attention_weights->ia3_value_weight.kernel : nullptr,
        int8_mode_ == 2 ? attention_weights->query_weight.scale_out : nullptr,
        int8_mode_ == 2 ? attention_weights->attention_output_weight.scale : nullptr,
        int8_mode_,
        stream_);
    sync_check_cuda_error();

    PUSH_RANGE("proj gemm");
#ifdef SPARSITY_ENABLED
    use_sparse_gemm = sparse_ && cublas_wrapper_->isUseSparse(1, d_model_, m_padded, local_hidden_units_);
#endif

    if (use_sparse_gemm) {
#ifdef SPARSITY_ENABLED
        cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                d_model_,
                                m_padded,
                                local_hidden_units_,
                                attention_weights->attention_output_weight.sp_kernel,
                                context_buf_,
                                attention_out);
#endif
    }
    else {
        if (int8_mode_ == 1) {
            FT_CHECK(weight_only_int8_fc_runner_.get() != NULL
                     && attention_weights->attention_output_weight.int8_kernel != NULL
                     && attention_weights->attention_output_weight.weight_only_quant_scale != NULL);

            weight_only_int8_fc_runner_->gemm(
                context_buf_,
                reinterpret_cast<const uint8_t*>(attention_weights->attention_output_weight.int8_kernel),
                attention_weights->attention_output_weight.weight_only_quant_scale,
                attention_out,
                batch_size,
                d_model_,
                local_hidden_units_,
                mixed_gemm_workspace_,
                mixed_gemm_ws_bytes_,
                stream_);
        }
        else if (int8_mode_ == 2) {
            int8_fc_runner_->gemm(reinterpret_cast<int8_t*>(context_buf_),
                                  attention_weights->attention_output_weight.int8_kernel,
                                  QuantMode::PerTensorQuant,
                                  attention_weights->attention_output_weight.scale_inter,
                                  attention_weights->attention_output_weight.scale_out,
                                  output_tensors->getPtr<T>("hidden_features"),
                                  batch_size,
                                  d_model_,
                                  local_hidden_units_,
                                  nullptr,
                                  0,
                                  stream_);
        }
        else {
            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  d_model_,  // n
                                  batch_size,
                                  local_hidden_units_,  // k
                                  attention_weights->attention_output_weight.kernel,
                                  d_model_,  // n
                                  context_buf_,
                                  local_hidden_units_,  // k
                                  attention_out,
                                  d_model_ /* n */);
        }
        sync_check_cuda_error();
    }
    POP_RANGE;

    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
}

template class DecoderSelfAttentionLayer<float>;
template class DecoderSelfAttentionLayer<half>;
#ifdef ENABLE_BF16
template class DecoderSelfAttentionLayer<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
