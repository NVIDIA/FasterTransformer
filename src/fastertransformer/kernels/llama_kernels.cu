#include "src/fastertransformer/kernels/llama_kernels.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"
#include "src/fastertransformer/utils/cuda_fp8_utils.h"

#include <algorithm>

#include <assert.h>
#include <cuda_fp16.h>
#include <stdio.h>

using namespace std;
namespace fastertransformer {

template<typename T>
__global__ void LLaMA_get_last_tokens(T* out, T* in, const int* cu_seqlens, int batch_size, int hidden_size)
{
    // in [num_tokens, hidden_size]
    // out [batch_size, hidden_size]
    int batch_idx = blockIdx.x;

    if (batch_idx >= batch_size)
        return;

    int pos = cu_seqlens[batch_idx + 1] - 1;

    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        out[batch_idx * hidden_size + idx] = in[pos * hidden_size + idx];
    }
}

template<typename T>
void invokeLLaMAGetLastTokens(
    T* out, T* in, const int* cu_seqlens, int batch_size, int hidden_size, cudaStream_t stream)
{
    dim3 grid(batch_size);
    dim3 block(256);
    LLaMA_get_last_tokens<<<grid, block, 0, stream>>>(out, in, cu_seqlens, batch_size, hidden_size);
}

template void invokeLLaMAGetLastTokens(
    float* out, float* in, const int* cu_seqlens, int batch_size, int hidden_size, cudaStream_t stream);
template void invokeLLaMAGetLastTokens(
    half* out, half* in, const int* cu_seqlens, int batch_size, int hidden_size, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeLLaMAGetLastTokens(
    __nv_bfloat16* out, __nv_bfloat16* in, const int* cu_seqlens, int batch_size, int hidden_size, cudaStream_t stream);
#endif

__global__ void LLaMA_extract_targets(float*     out,
                                      float*     in,
                                      const int* target_ids,
                                      const int* cu_seqlens,
                                      int        beam_width,
                                      int        batch_size,
                                      int        vocab_size,
                                      int        num_tokens)
{
    // in [batch_size, vocab_size]
    // target_ids [ beam_width, num_tokens ]
    // out [beam_width, batch_size]
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int beam_idx  = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx >= batch_size || beam_idx >= beam_width)
        return;

    int pos                                = cu_seqlens[batch_idx + 1] - 1;
    int target_idx                         = target_ids[beam_idx * num_tokens + pos];
    out[beam_idx * batch_size + batch_idx] = in[batch_idx * vocab_size + target_idx];
}

void invokeLLaMAExtractTargets(float*       out,
                               float*       in,
                               const int*   target_ids,
                               const int*   cu_seqlens,
                               int          beam_width,
                               int          batch_size,
                               int          vocab_size,
                               int          num_tokens,
                               cudaStream_t stream)
{
    dim3 block(32, 4);
    dim3 grid((batch_size + block.x - 1) / block.x, (beam_width + block.y - 1) / block.y);
    LLaMA_extract_targets<<<grid, block, 0, stream>>>(
        out, in, target_ids, cu_seqlens, beam_width, batch_size, vocab_size, num_tokens);
}

__global__ void LLaMA_log_softmax(float* out, const float* logits, const int num_tokens, const int vocab_size)
{
    // logits [T, V]
    // out [T, V]
    const int64_t    ti = blockIdx.x;
    __shared__ float s_sum, s_max;

    if (ti >= num_tokens)
        return;

    float local_max = -1e20f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float logit_val = logits[ti * vocab_size + i];
        local_max       = fmax(logit_val, local_max);
    }

    float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
    if (threadIdx.x == 0) {
        s_max = max_val;
    }
    __syncthreads();

    float local_sum = 0;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float logit_val = logits[ti * vocab_size + i];
        local_sum += __expf(logit_val - s_max);
    }
    float sum_val = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);
    if (threadIdx.x == 0) {
        // s_sum = sum_val + 1e-6f;
        s_sum = sum_val;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float logit_val          = logits[ti * vocab_size + i];
        out[ti * vocab_size + i] = (logit_val - s_max) - __logf(s_sum);
    }
}

void invokeLLaMALogSoftmax(
    float* out, const float* logits, const int num_tokens, const int vocab_size, cudaStream_t stream)
{
    dim3 grid(num_tokens);
    dim3 block(min(1024, vocab_size));
    LLaMA_log_softmax<<<grid, block, 0, stream>>>(out, logits, num_tokens, vocab_size);
}

__global__ void LLaMA_gather_tokens_kernel(float*       out,
                                           const float* probs,
                                           const int*   input_lengths,
                                           const int*   target_ids,
                                           const int*   cu_seqlens,
                                           const int    batch_size,
                                           const int    vocab_size,
                                           const int    num_tokens)
{
    // probs: [T, V]
    // target_ids: [T]
    // out: [batch_size]
    int batch_idx = blockIdx.x;

    if (batch_idx >= batch_size)
        return;

    float val = 0.f;
    for (int i = threadIdx.x; i < input_lengths[batch_idx]; i += blockDim.x) {
        int pos        = cu_seqlens[batch_idx] + i;
        int target_pos = target_ids[pos];
        val += (target_pos > 0) ? probs[pos * vocab_size + target_pos] : 0.f;
    }
    float sum = blockReduceSum<float>(val);

    if (threadIdx.x == 0)
        out[batch_idx] = sum;
}

void invokeLLaMAGatherTokens(float*       out,
                             const float* probs,
                             const int*   input_lengths,
                             const int*   target_ids,
                             const int*   cu_seqlens,
                             const int    batch_size,
                             const int    vocab_size,
                             const int    num_tokens,
                             cudaStream_t stream)
{
    dim3 grid(batch_size);
    dim3 block(256);
    LLaMA_gather_tokens_kernel<<<grid, block, 0, stream>>>(
        out, probs, input_lengths, target_ids, cu_seqlens, batch_size, vocab_size, num_tokens);
}

template<typename T>
__global__ void LLaMAstart_id_embedding_lookups_kernel(
    T* out, const T* embedding_table, const int* input_ids, const int num_tokens, const int64_t hidden_units)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < num_tokens * hidden_units;
         index += blockDim.x * gridDim.x) {

        // embedding lookup from word ids [batch, length] (part of [batch, length]) and [vocab, hidden] to generate
        // embedding [batch, length, hidden]
        const int word_index = index / hidden_units;
        const int col_index  = index % hidden_units;
        const int input_id   = input_ids[word_index];

        out[index] = embedding_table[input_id * hidden_units + col_index];
    }
}

template<typename T>
void invokeLLaMAInputIdsEmbeddingLookup(T*           out,
                                        const T*     embedding_table,
                                        const int*   input_ids,
                                        const int    num_tokens,
                                        const int    hidden_units,
                                        cudaStream_t stream)
{
    dim3 grid(min(num_tokens, 65536));
    dim3 block(min(hidden_units, 512));
    LLaMAstart_id_embedding_lookups_kernel<T>
        <<<grid, block, 0, stream>>>(out, embedding_table, input_ids, num_tokens, hidden_units);
}

template void invokeLLaMAInputIdsEmbeddingLookup(float*       out,
                                                 const float* embedding_table,
                                                 const int*   input_ids,
                                                 const int    num_tokens,
                                                 const int    hidden_units,
                                                 cudaStream_t stream);
template void invokeLLaMAInputIdsEmbeddingLookup(half*        out,
                                                 const half*  embedding_table,
                                                 const int*   input_ids,
                                                 const int    num_tokens,
                                                 const int    hidden_units,
                                                 cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeLLaMAInputIdsEmbeddingLookup(__nv_bfloat16*       out,
                                                 const __nv_bfloat16* embedding_table,
                                                 const int*           input_ids,
                                                 const int            num_tokens,
                                                 const int            hidden_units,
                                                 cudaStream_t         stream);
#endif

__global__ void LLaMAgetPaddingOffsetAndCuSeqLensKernel(
    int* padding_offset, int* cu_seqlens, const int* sequence_length, const int batch_size, const int seq_len)
{
    // do cumulated sum
    int total_seq_len = 0;
    int cum_offset    = 0;
    int index         = 0;
    for (int i = 0; i < batch_size; i++) {
        const int num_tokens = sequence_length[i];
        cu_seqlens[i]        = total_seq_len;
        for (int j = 0; j < num_tokens; j++) {
            padding_offset[index] = cum_offset;
            index++;
        }
        cum_offset += seq_len - num_tokens;
        total_seq_len += num_tokens;
    }
    cu_seqlens[batch_size] = total_seq_len;
}

void invokeLLaMAGetPaddingOffsetAndCuSeqLens(int*         padding_offset,
                                             int*         cu_seqlens,
                                             const int*   input_lengths,
                                             const int    batch_size,
                                             const int    seq_len,
                                             cudaStream_t stream)
{
    LLaMAgetPaddingOffsetAndCuSeqLensKernel<<<1, 1, 0, stream>>>(
        padding_offset, cu_seqlens, input_lengths, batch_size, seq_len);
}

template<typename T>
__global__ void LLaMAbuildDecoderAttentionMaskKernel(T*         attention_mask,
                                                     const int* sequence_lengths,
                                                     const int* context_lengths,
                                                     const int  batch_size,
                                                     const int  seq_len,
                                                     const int  attn_len)
{
    // attention_mask:
    // [batch_size, 1, seq_len, attn_len]
    const int batch_idx         = blockIdx.x;
    const int mask_size_per_seq = seq_len * attn_len;
    attention_mask += batch_idx * mask_size_per_seq;
    const int context_length = context_lengths[batch_idx];
    const int length         = sequence_lengths[batch_idx];

    for (int i = threadIdx.x; i < mask_size_per_seq; i += blockDim.x) {
        int row_id = i / attn_len;
        int col_id = i % attn_len;
        if (row_id < length && col_id <= (row_id + context_length)) {
            attention_mask[i] = (T)(1.0f);
        }
        else {
            attention_mask[i] = (T)(0.0f);
        }
    }
}

template<typename T>
void invokeLLaMABuildDecoderAttentionMask(T*           attention_mask,
                                          const int*   sequence_length,
                                          const int*   context_lengths,
                                          const int    batch_size,
                                          const int    seq_len,
                                          const int    attn_len,
                                          cudaStream_t stream)
{
    LLaMAbuildDecoderAttentionMaskKernel<T><<<batch_size, 256, 0, stream>>>(
        attention_mask, sequence_length, context_lengths, batch_size, seq_len, attn_len);
}

template void invokeLLaMABuildDecoderAttentionMask(float*       attention_mask,
                                                   const int*   sequence_length,
                                                   const int*   context_lengths,
                                                   const int    batch_size,
                                                   const int    seq_len,
                                                   const int    attn_len,
                                                   cudaStream_t stream);

template void invokeLLaMABuildDecoderAttentionMask(half*        attention_mask,
                                                   const int*   sequence_length,
                                                   const int*   context_lengths,
                                                   const int    batch_size,
                                                   const int    seq_len,
                                                   const int    attn_len,
                                                   cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeLLaMABuildDecoderAttentionMask(__nv_bfloat16* attention_mask,
                                                   const int*     sequence_length,
                                                   const int*     context_lengths,
                                                   const int      batch_size,
                                                   const int      seq_len,
                                                   const int      attn_len,
                                                   cudaStream_t   stream);
#endif

template<typename T>
__global__ void LLaMACopyKernel(T* dst, T* src, const int count)
{
    int           idx     = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int X_ELEMS = (sizeof(T) == 4) ? 4 : 8;
    if (idx * X_ELEMS >= count) {
        return;
    }

    auto v_dst = reinterpret_cast<uint4*>(dst);
    auto v_src = reinterpret_cast<uint4*>(src);
    v_dst[idx] = v_src[idx];
}

template<typename T>
void invokeLLaMACopyKernel(T* dst, T* src, const int count, cudaStream_t stream)
{
    constexpr int block_sz = 128;
    constexpr int x        = (sizeof(T) == 4) ? 4 : 8;
    assert(count % x == 0);
    int grid_sz = (count / x + block_sz - 1) / block_sz;
    LLaMACopyKernel<<<grid_sz, block_sz, 0, stream>>>(dst, src, count);
}

template void invokeLLaMACopyKernel(float* dst, float* src, const int count, cudaStream_t stream);
template void invokeLLaMACopyKernel(half* dst, half* src, const int count, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeLLaMACopyKernel(__nv_bfloat16* dst, __nv_bfloat16* src, const int count, cudaStream_t stream);
#endif

template<typename T>
__global__ void LLaMAMemset0Kernel(T* dst, const int count)
{
    int           idx     = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int X_ELEMS = (sizeof(T) == 4) ? 4 : 8;
    if (idx * X_ELEMS >= count) {
        return;
    }

    auto v_dst = reinterpret_cast<uint4*>(dst);
    v_dst[idx] = {0};
}

template<typename T>
void invokeLLaMAMemset0(T* dst, const int count, cudaStream_t stream)
{
    constexpr int block_sz = 128;
    constexpr int x        = (sizeof(T) == 4) ? 4 : 8;
    assert(count % x == 0);
    int grid_sz = (count / x + block_sz - 1) / block_sz;
    LLaMAMemset0Kernel<<<grid_sz, block_sz, 0, stream>>>(dst, count);
}

template void invokeLLaMAMemset0(float* dst, const int count, cudaStream_t stream);
template void invokeLLaMAMemset0(half* dst, const int count, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeLLaMAMemset0(__nv_bfloat16* dst, const int count, cudaStream_t stream);
#endif

}  // namespace fastertransformer
