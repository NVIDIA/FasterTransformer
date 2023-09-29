#include "src/fastertransformer/kernels/llama_kernels.h"
#include "src/fastertransformer/utils/cuda_fp8_utils.h"

#include <assert.h>
#include <cuda_fp16.h>
#include <stdio.h>

namespace fastertransformer {

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
                                                     const int  max_length)
{
    // attention_mask:
    // [batch_size, 1, seq_len, max_length]
    const int batch_idx         = blockIdx.x;
    const int mask_size_per_seq = seq_len * max_length;
    attention_mask += batch_idx * mask_size_per_seq;
    const int context_length = context_lengths[batch_idx];
    const int length         = sequence_lengths[batch_idx];
    const int offset         = max_length - length;

    for (int i = threadIdx.x; i < mask_size_per_seq; i += blockDim.x) {
        int row_id = i / max_length;
        int col_id = i % max_length;
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
                                          const int    max_length,
                                          cudaStream_t stream)
{
    LLaMAbuildDecoderAttentionMaskKernel<T><<<batch_size, 256, 0, stream>>>(
        attention_mask, sequence_length, context_lengths, batch_size, seq_len, max_length);
}

template void invokeLLaMABuildDecoderAttentionMask(float*       attention_mask,
                                                   const int*   sequence_length,
                                                   const int*   context_lengths,
                                                   const int    batch_size,
                                                   const int    seq_len,
                                                   const int    max_length,
                                                   cudaStream_t stream);

template void invokeLLaMABuildDecoderAttentionMask(half*        attention_mask,
                                                   const int*   sequence_length,
                                                   const int*   context_lengths,
                                                   const int    batch_size,
                                                   const int    seq_len,
                                                   const int    max_length,
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
