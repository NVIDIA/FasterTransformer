#include "src/fastertransformer/utils/cuda_fp8_utils.h"
#include "src/fastertransformer/kernels/llama_kernels.h"

namespace fastertransformer {

template<typename T>
__global__ void LLaMAbuildDecoderAttentionMaskKernel(
    T* attention_mask, const int* sequence_lengths, const int batch_size, const int seq_len, const int start_pos)
{
    // sequence_lengths:
    // [batch_size]
    // attention_mask:
    // [batch_size, 1, seq_len, seq_len + start_pos]
    const int max_length        = seq_len + start_pos;
    const int mask_size_per_seq = seq_len * max_length;
    attention_mask += blockIdx.x * mask_size_per_seq;
    const int seq_length = sequence_lengths[blockIdx.x];

    for (int i = threadIdx.x; i < mask_size_per_seq; i += blockDim.x) {
        int row_id = i / max_length;
        int col_id = i % max_length;
        if (row_id < seq_length && col_id <= (row_id + start_pos)) {
            attention_mask[i] = (T)(1.0f);
        }
        else {
            attention_mask[i] = (T)(0.0f);
        }
    }
}

template<typename T>
void invokeLLaMABuildDecoderAttentionMask(T*           attention_mask,
                                          const int*   sequence_lengths,
                                          const int    batch_size,
                                          const int    seq_len,
                                          const int    start_pos,
                                          cudaStream_t stream)
{
    LLaMAbuildDecoderAttentionMaskKernel<T>
        <<<batch_size, 256, 0, stream>>>(attention_mask, sequence_lengths, batch_size, seq_len, start_pos);
}

template void invokeLLaMABuildDecoderAttentionMask(float*       attention_mask,
                                                   const int*   sequence_lengths,
                                                   const int    batch_size,
                                                   const int    seq_len,
                                                   const int    start_pos,
                                                   cudaStream_t stream);

template void invokeLLaMABuildDecoderAttentionMask(half*        attention_mask,
                                                   const int*   sequence_lengths,
                                                   const int    batch_size,
                                                   const int    seq_len,
                                                   const int    start_pos,
                                                   cudaStream_t stream);
}  // namespace fastertransformer
