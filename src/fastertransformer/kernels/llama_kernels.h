#pragma once


#include "src/fastertransformer/utils/cuda_fp8_utils.h"
#include "src/fastertransformer/utils/memory_utils.h"
namespace fastertransformer {

template<typename T>
void invokeLLaMABuildDecoderAttentionMask(T*           attention_mask,
                                          const int*   sequence_lengths,
                                          const int    batch_size,
                                          const int    seq_len,
                                          const int    start_pos,
                                          cudaStream_t stream);
} // namespace fastertransformer
