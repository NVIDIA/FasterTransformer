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

#include "src/fastertransformer/layers/attention_layers/DecoderSelfAttentionLayer.h"
#include "src/fastertransformer/kernels/decoder_masked_multihead_attention.h"
#include "src/fastertransformer/utils/logger.h"

namespace fastertransformer {

template<typename T>
struct TypeConverter {
    using Type = T;
};

template<>
struct TypeConverter<half> {
    using Type = uint16_t;
};

template<typename T>
void fusedQKV_masked_attention_dispatch(const T* qkv_buf,
                                        const T* qkv_bias,
                                        const T* relative_attention_bias,
                                        T* key_cache,
                                        T* value_cache,
                                        const int* cache_indir,
                                        T* context_buf,
                                        const bool* finished,
                                        const int* sequence_lengths,
                                        const int max_batch_size,
                                        const int inference_batch_size,
                                        const int beam_width,
                                        const int head_num,
                                        const int size_per_head,
                                        const int rotary_embedding_dim,
                                        const int max_seq_len,
                                        const int max_input_len,
                                        const int* input_lengths,
                                        const int step,
                                        const float q_scaling,
                                        const int relative_attention_bias_stride,
                                        cudaStream_t stream)
{
    using DataType = typename TypeConverter<T>::Type;
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
    params.k = reinterpret_cast<const DataType*>(qkv_buf) + hidden_units;
    params.v = reinterpret_cast<const DataType*>(qkv_buf) + 2 * hidden_units;
    params.stride = 3 * hidden_units;
    params.finished = const_cast<bool*>(finished);

    params.k_cache = reinterpret_cast<DataType*>(key_cache);
    params.v_cache = reinterpret_cast<DataType*>(value_cache);
    params.cache_indir = cache_indir;
    params.batch_size = inference_batch_size;
    params.beam_width = beam_width;
    params.seq_length = max_seq_len;
    params.length_per_sample = sequence_lengths;
    params.timestep = step - 1;
    params.num_heads = head_num;
    params.hidden_size_per_head = size_per_head;
    params.rotary_embedding_dim = rotary_embedding_dim;
    // Note: keep norm factor (sqrt(K_dim)) when adopting megatron T5 structure (may adjust)
    params.inv_sqrt_dh = 1.F / (sqrtf((float)params.hidden_size_per_head) * q_scaling);

    params.input_lengths = input_lengths;
    params.max_input_len = max_input_len;
    // TODO(bhsueh) Need better implementation
    if (relative_attention_bias != nullptr) {
        if (sizeof(T) == 4) {
            params.relative_attention_bias_float = reinterpret_cast<const float*>(relative_attention_bias);
        }
        else {
            params.relative_attention_bias_half = reinterpret_cast<const half*>(relative_attention_bias);
        }
    }
    params.relative_attention_bias_stride = relative_attention_bias_stride;

    masked_multihead_attention(params, stream);
}

template void fusedQKV_masked_attention_dispatch(const float* qkv_buf,
                                                 const float* qkv_bias,
                                                 const float* relative_attention_bias,
                                                 float* key_cache,
                                                 float* value_cache,
                                                 const int* cache_indir,
                                                 float* context_buf,
                                                 const bool* finished,
                                                 const int* sequence_lengths,
                                                 const int max_batch_size,
                                                 const int inference_batch_size,
                                                 const int beam_width,
                                                 const int head_num,
                                                 const int size_per_head,
                                                 const int rotary_embedding_dim,
                                                 const int max_seq_len,
                                                 const int max_input_len,
                                                 const int* input_lengths,
                                                 const int step,
                                                 const float q_scaling,
                                                 const int relative_attention_bias_stride,
                                                 cudaStream_t stream);

template void fusedQKV_masked_attention_dispatch(const half* qkv_buf,
                                                 const half* qkv_bias,
                                                 const half* relative_attention_bias,
                                                 half* key_cache,
                                                 half* value_cache,
                                                 const int* cache_indir,
                                                 half* context_buf,
                                                 const bool* finished,
                                                 const int* sequence_lengths,
                                                 const int max_batch_size,
                                                 const int inference_batch_size,
                                                 const int beam_width,
                                                 const int head_num,
                                                 const int size_per_head,
                                                 const int rotary_embedding_dim,
                                                 const int max_seq_len,
                                                 const int max_input_len,
                                                 const int* input_lengths,
                                                 const int step,
                                                 const float q_scaling,
                                                 const int relative_attention_bias_stride,
                                                 cudaStream_t stream);

template<typename T>
void DecoderSelfAttentionLayer<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        qkv_buf_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * 3 * local_hidden_units_, false));
        context_buf_ =
            reinterpret_cast<T*>(allocator_->malloc(sizeof(T) * max_batch_size_ * local_hidden_units_, false));
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void DecoderSelfAttentionLayer<T>::allocateBuffer(size_t batch_size)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    qkv_buf_ =
        reinterpret_cast<T*>(allocator_->reMalloc(qkv_buf_, sizeof(T) * batch_size * 3 * local_hidden_units_, false));
    context_buf_ =
        reinterpret_cast<T*>(allocator_->reMalloc(context_buf_, sizeof(T) * batch_size * local_hidden_units_, false));
    is_allocate_buffer_ = true;
}

template<typename T>
void DecoderSelfAttentionLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        allocator_->free(qkv_buf_);
        allocator_->free(context_buf_);
        is_allocate_buffer_ = false;
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
DecoderSelfAttentionLayer<T>::DecoderSelfAttentionLayer(size_t max_batch_size,
                                                        size_t head_num,
                                                        size_t size_per_head,
                                                        size_t local_head_num,
                                                        size_t rotary_embedding_dim,
                                                        size_t d_model,
                                                        const float q_scaling,
                                                        cudaStream_t stream,
                                                        cublasMMWrapper* cublas_wrapper,
                                                        IAllocator* allocator,
                                                        bool is_free_buffer_after_forward,
                                                        bool sparse,
                                                        int int8_mode):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse),
    max_batch_size_(max_batch_size),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num_ * size_per_head_),
    local_head_num_(local_head_num),
    local_hidden_units_(local_head_num_ * size_per_head_),
    rotary_embedding_dim_(rotary_embedding_dim),
    d_model_(d_model),
    q_scaling_(q_scaling),
    int8_mode_(int8_mode)
{
    FT_CHECK(size_per_head_ == 32 || size_per_head_ == 64 || size_per_head_ == 96 || size_per_head_ == 128
             || size_per_head_ == 160 || size_per_head_ == 192 || size_per_head_ == 224 || size_per_head_ == 256);
}

template<typename T>
DecoderSelfAttentionLayer<T>::DecoderSelfAttentionLayer(size_t max_batch_size,
                                                        size_t head_num,
                                                        size_t size_per_head,
                                                        cudaStream_t stream,
                                                        cublasMMWrapper* cublas_wrapper,
                                                        IAllocator* allocator,
                                                        bool is_free_buffer_after_forward,
                                                        bool sparse,
                                                        int int8_mode):
    DecoderSelfAttentionLayer<T>(max_batch_size,
                                 head_num,
                                 size_per_head,
                                 head_num,
                                 0,
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
DecoderSelfAttentionLayer<T>::DecoderSelfAttentionLayer(size_t max_batch_size,
                                                        size_t head_num,
                                                        size_t size_per_head,
                                                        const float q_scaling,
                                                        cudaStream_t stream,
                                                        cublasMMWrapper* cublas_wrapper,
                                                        IAllocator* allocator,
                                                        bool is_free_buffer_after_forward,
                                                        bool sparse,
                                                        int int8_mode):
    DecoderSelfAttentionLayer<T>(max_batch_size,
                                 head_num,
                                 size_per_head,
                                 head_num,
                                 0,
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
DecoderSelfAttentionLayer<T>::DecoderSelfAttentionLayer(size_t max_batch_size,
                                                        size_t head_num,
                                                        size_t size_per_head,
                                                        size_t local_head_num,
                                                        cudaStream_t stream,
                                                        cublasMMWrapper* cublas_wrapper,
                                                        IAllocator* allocator,
                                                        bool is_free_buffer_after_forward,
                                                        bool sparse,
                                                        int int8_mode):
    DecoderSelfAttentionLayer<T>(max_batch_size,
                                 head_num,
                                 size_per_head,
                                 local_head_num,
                                 0,
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
DecoderSelfAttentionLayer<T>::DecoderSelfAttentionLayer(size_t max_batch_size,
                                                        size_t head_num,
                                                        size_t size_per_head,
                                                        size_t local_head_num,
                                                        size_t d_model,
                                                        const float q_scaling,
                                                        cudaStream_t stream,
                                                        cublasMMWrapper* cublas_wrapper,
                                                        IAllocator* allocator,
                                                        bool is_free_buffer_after_forward,
                                                        bool sparse,
                                                        int int8_mode):
    DecoderSelfAttentionLayer<T>(max_batch_size,
                                 head_num,
                                 size_per_head,
                                 local_head_num,
                                 0,
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
DecoderSelfAttentionLayer<T>::DecoderSelfAttentionLayer(size_t max_batch_size,
                                                        size_t head_num,
                                                        size_t size_per_head,
                                                        size_t local_head_num,
                                                        size_t rotary_embedding_dim,
                                                        cudaStream_t stream,
                                                        cublasMMWrapper* cublas_wrapper,
                                                        IAllocator* allocator,
                                                        bool is_free_buffer_after_forward,
                                                        bool sparse,
                                                        int int8_mode):
    DecoderSelfAttentionLayer<T>(max_batch_size,
                                 head_num,
                                 size_per_head,
                                 local_head_num,
                                 rotary_embedding_dim,
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
void DecoderSelfAttentionLayer<T>::forward(std::vector<fastertransformer::Tensor>* output_tensors,
                                           const std::vector<fastertransformer::Tensor>* input_tensors,
                                           const AttentionWeight<T>* attention_weights)
{
    // input tensors:
    //      attention_input [batch_size, d_model_],
    //      finished [batch_size],
    //      sequence_lengths [batch_size]
    //      input_lengths [batch_size]
    //      max_input_length [1] on cpu
    //      step [1] on cpu
    //      cache_indirection [batch_size / beam_width, beam_width, max_seq_len]
    //      relative_attention_bias [1, head_num, step, step] or [1, head_num, max_seq_len, max_seq_len] (option)

    // output tensors:
    //      attention_output [batch_size, d_model_],
    //      key_cache [batch, local_head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [batch, local_head_num, max_seq_len, size_per_head]

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() == 7 || input_tensors->size() == 8);
    FT_CHECK(output_tensors->size() == 3);
    FT_CHECK(output_tensors->at(1).shape.size() == 5 || output_tensors->at(1).shape.size() == 3);
    FT_CHECK(output_tensors->at(2).shape.size() == 4 || output_tensors->at(2).shape.size() == 3);
    // FT_CHECK(isValidBatchSize(input_tensors->at(0).shape[0]));
    allocateBuffer(input_tensors->at(0).shape[0]);

    const T* attention_input = reinterpret_cast<const T*>(input_tensors->at(0).data);
    const bool* finished = reinterpret_cast<const bool*>(input_tensors->at(1).data);
    const int* sequence_lengths = reinterpret_cast<const int*>(input_tensors->at(2).data);
    const int* cache_indir = reinterpret_cast<const int*>(input_tensors->at(6).data);
    const T* relative_attention_bias =
        reinterpret_cast<const T*>(input_tensors->size() == 8 ? input_tensors->at(7).data : nullptr);
    const int relative_attention_bias_stride = input_tensors->size() == 8 ? input_tensors->at(7).shape[3] : 0;

    T* attention_out = (T*)(output_tensors->at(0).data);
    T* key_cache = (T*)(output_tensors->at(1).data);
    T* value_cache = (T*)(output_tensors->at(2).data);

    const int batch_size = input_tensors->at(0).shape[0];
    const int beam_width = input_tensors->at(6).shape[1];
    const int max_seq_len = output_tensors->at(1).shape[3];

#ifdef SPARSITY_ENABLED
    const int m_padded = 8 * div_up(batch_size, 8);
    if (sparse_ && cublas_wrapper_->isUseSparse(1, 3 * local_hidden_units_, m_padded, d_model_)) {
        cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                3 * local_hidden_units_,
                                m_padded,
                                d_model_,
                                attention_weights->query_weight.sp_kernel,
                                attention_input,
                                qkv_buf_);
    }
    else {
#endif
        if (int8_mode_ != 0 && batch_size <= 2) {
            FT_CHECK(attention_weights->query_weight.int8_kernel != NULL
                     && attention_weights->query_weight.scale != NULL);
            int8WeightPerChannelLdkMultiplicationLauncher(attention_weights->query_weight.int8_kernel,
                                                          attention_input,
                                                          attention_weights->query_weight.scale,
                                                          qkv_buf_,
                                                          batch_size,
                                                          3 * local_hidden_units_,
                                                          d_model_,
                                                          stream_);
        }
        else {
            if (int8_mode_ == 1) {
                FT_LOG_WARNING(
                    "[DecoderSelfAttentionLayer<T>::forward] int8 gpt doesn't support m > 2, run fp gpt instead.\n");
            }
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
#ifdef SPARSITY_ENABLED
    }
#endif
    sync_check_cuda_error();
    fusedQKV_masked_attention_dispatch<T>(qkv_buf_,
                                          attention_weights->query_weight.bias,
                                          relative_attention_bias,
                                          key_cache,
                                          value_cache,
                                          cache_indir,
                                          context_buf_,
                                          finished,
                                          sequence_lengths,
                                          batch_size,
                                          batch_size,
                                          beam_width,
                                          local_head_num_,
                                          size_per_head_,
                                          rotary_embedding_dim_,
                                          max_seq_len,
                                          *(int*)(input_tensors->at(4).data),
                                          (int*)(input_tensors->at(3).data),
                                          *(int*)(input_tensors->at(5).data),
                                          q_scaling_,
                                          relative_attention_bias_stride,
                                          stream_);
    sync_check_cuda_error();

#ifdef SPARSITY_ENABLED
    if (sparse_ && cublas_wrapper_->isUseSparse(1, d_model_, m_padded, local_hidden_units_)) {
        cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                d_model_,
                                m_padded,
                                local_hidden_units_,
                                attention_weights->attention_output_weight.sp_kernel,
                                context_buf_,
                                attention_out);
    }
    else {
#endif
        if (int8_mode_ != 0 && batch_size <= 2) {
            FT_CHECK(attention_weights->attention_output_weight.int8_kernel != NULL
                     && attention_weights->attention_output_weight.scale != NULL);
            int8WeightPerChannelLdkMultiplicationLauncher(attention_weights->attention_output_weight.int8_kernel,
                                                          context_buf_,
                                                          attention_weights->attention_output_weight.scale,
                                                          attention_out,
                                                          batch_size,
                                                          d_model_,
                                                          local_hidden_units_,
                                                          stream_);
        }
        else {
            if (int8_mode_ == 1) {
                FT_LOG_WARNING(
                    "[DecoderSelfAttentionLayer<T>::forward] int8 gpt doesn't support m > 2, run fp gpt instead.\n");
            }
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
#ifdef SPARSITY_ENABLED
    }
#endif

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template class DecoderSelfAttentionLayer<float>;
template class DecoderSelfAttentionLayer<half>;
#ifdef ENABLE_BF16
template class DecoderSelfAttentionLayer<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
