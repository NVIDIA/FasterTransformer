#pragma once

#include "src/fastertransformer/utils/cuda_fp8_utils.h"
#include "src/fastertransformer/utils/memory_utils.h"
namespace fastertransformer {

void invokeLLaMAGetPaddingOffsetAndCuSeqLens(int*         padding_offset,
                                             int*         cu_seqlens,
                                             const int*   input_lengths,
                                             const int    batch_size,
                                             const int    seq_len,
                                             cudaStream_t stream);

template<typename T>
void invokeLLaMABuildDecoderAttentionMask(T*           attention_mask,
                                          const int*   sequence_length,
                                          const int*   context_lengths,
                                          const int    batch_size,
                                          const int    seq_len,
                                          const int    max_length,
                                          cudaStream_t stream);
template<typename T>
void invokeLLaMAInputIdsEmbeddingLookup(T*           from_tensor,
                                        const T*     embedding_table,
                                        const int*   input_ids,
                                        const int    num_tokens,
                                        const int    hidden_units,
                                        cudaStream_t stream);

template<typename T>
void invokeLLaMACopyKernel(T* dst, T* src, const int count, cudaStream_t stream);

template<typename T>
void invokeLLaMAMemset0(T* dst, const int count, cudaStream_t stream);

void invokeLLaMAGatherTokens(float*       out,
                             const float* probs,
                             const int*   input_lengths,
                             const int*   target_ids,
                             const int*   cu_seqlens,
                             const int    batch_size,
                             const int    vocab_size,
                             const int    num_tokens,
                             cudaStream_t stream);

void invokeLLaMALogSoftmax(
    float* out, const float* logits, const int num_tokens, const int vocab_size, cudaStream_t stream);

template<typename T>
void invokeLLaMAGetLastTokens(
    T* out, T* in, const int* cu_seqlens, int batch_size, int hidden_size, cudaStream_t stream);

void invokeLLaMAExtractTargets(float*       out,
                               float*       in,
                               const int*   target_ids,
                               const int*   cu_seqlens,
                               int          beam_width,
                               int          batch_size,
                               int          vocab_size,
                               int          num_tokens,
                               cudaStream_t stream);
}  // namespace fastertransformer
