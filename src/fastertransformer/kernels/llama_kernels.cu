#include "src/fastertransformer/kernels/llama_kernels.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"
#include "src/fastertransformer/utils/cuda_fp8_utils.h"

#include <algorithm>

#include <assert.h>
#include <cuda_fp16.h>
#include <stdio.h>

using namespace std;
namespace fastertransformer {

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
                                           const int*   input_ids,
                                           const int*   input_lengths,
                                           const int*   cu_seqlens,
                                           const int    batch_size,
                                           const int    vocab_size)
{
    /*
    // probs: [T, V]
    // input_ids: [T]
    int batch_idx = blockIdx.x;

    if (batch_idx >= batch_size)
        return;

    float val = 0.f;
    //    for (int i = cu_seqlens[batch_idx] + threadIdx.x; i < cu_seqlens[batch_idx + 1] - 1; i += blockDim.x) {
    //        int input_idx = input_ids[i + 1];
    //        val += probs[i * vocab_size + input_idx];
    //    }
    for (int t = cu_seqlens[batch_idx]; t < cu_seqlens[batch_idx + 1] - 1; ++t) {
        val += probs[t * vocab_size + input_ids[t + 1]];
    }
    //float sum = blockReduceSum<float>(val);

    if (threadIdx.x == 0)
        out[batch_idx] = val;
        */
    // for b in range(bsz):
    //     for i in range(choice_seq_lens_list[c][b]-1):
    //       t = choice_cum_seq_lens_list[c][b] + i
    //       choice_log_probs[b, c] = choice_log_probs[b, c] +  log_likelihoods[t, choice_tokens_list[c][t+1]]

    for (int b = 0; b < batch_size; ++b) {
        float val = 0.f;
        for (int i = 0; i < input_lengths[b] - 1; ++i) {
            int t = cu_seqlens[b] + i;
            val += probs[t * vocab_size + input_ids[t + 1]];
        }
        out[b] = val;
    }
}

void invokeLLaMAGatherTokens(float*       out,
                             const float* probs,
                             const int*   input_ids,
                             const int*   input_lengths,
                             const int*   cu_seqlens,
                             const int    batch_size,
                             const int    vocab_size,
                             cudaStream_t stream)
{
    LLaMA_gather_tokens_kernel<<<1, 1, 0, stream>>>(
        out, probs, input_ids, input_lengths, cu_seqlens, batch_size, vocab_size);
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

}  // namespace fastertransformer
