/***************************************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR 
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND 
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE 
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

#define CHECK_CUDA(call) do { \
  cudaError_t status_ = call; \
  if( status_ != cudaSuccess ) { \
    fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
    exit(1); \
  } \
} while(0)

////////////////////////////////////////////////////////////////////////////////////////////////////

// The structure of parameters for the masked multihead attention kernel.
//
// We use the following terminology to describe the different dimensions.
//
// B:  Batch size (number of sequences),
// L:  Sequence length,
// D:  Hidden dimension,
// H:  Number of heads,
// Dh: Hidden dimension per head - Dh = D / H.

template< typename T>
struct Multihead_attention_params_base {

  // The output buffer. Dimensions B x D.
  T *out;

  // The input Qs and the associated bias. Dimensions B x D and D, resp.
  const T *q, *q_bias;
  // The input Ks and the associated bias. Dimensions B x D and D, resp.
  const T *k, *k_bias;
  // The input Vs and the associated bias. Dimensions B x D and D, resp.
  const T *v, *v_bias;

  // The cache for the Ks. The size must be at least B x L x D.
  T *k_cache;
  // The cache for the Vs. The size must be at least B x L x D.
  T *v_cache;

  // Stride to handle the case when KQV is a single buffer
  int stride;

  // The batch size.
  int batch_size;
  // The sequence length.
  int seq_length;
  // The number of heads (H).
  int num_heads;
  // The hidden dimension per head (Dh).
  int hidden_size_per_head;
  // The per-head latent space reserved for rotary embeddings.
  int rotary_embedding_dim = 0;
  // The current timestep. TODO(bhsueh) Check that do we only this param in cross attention?
  int timestep;
  // The current timestep of each sentences (support different timestep for different sentences)

  // The 1.f / sqrt(Dh). Computed on the host.
  float inv_sqrt_dh;

  // Used when we have some input context like gpt
  const int *input_lengths;
  int max_input_len;

  const float* relative_attention_bias_float = nullptr;
  const half* relative_attention_bias_half = nullptr;
  int relative_attention_bias_stride;
};

template< typename T, bool CROSS_ATTENTION>
struct Multihead_attention_params : public Multihead_attention_params_base<T> {

  // allows to exist attention eary
  bool *finished;

  // required in case of cross attention
  // will need it here till if constexpr in c++17
  int *memory_length_per_sample;

  // required in case of masked attention with different length
  const int *length_per_sample;
};

template< typename T>
struct Multihead_attention_params<T, true> : public Multihead_attention_params_base<T> {
  // allows to exist attention eary
  bool *finished;

  // required in case of cross attention
  int *memory_length_per_sample;

  // required in case of masked attention with different length
  const int *length_per_sample;
};

template <class T>
using Masked_multihead_attention_params = Multihead_attention_params<T, false>;

template <class T>
using Cross_multihead_attention_params = Multihead_attention_params<T, true>;

////////////////////////////////////////////////////////////////////////////////////////////////////

void masked_multihead_attention    (const Masked_multihead_attention_params<float>    &params, const cudaStream_t &stream);
void masked_multihead_attention    (const Masked_multihead_attention_params<uint16_t> &params, const cudaStream_t &stream);
void cross_multihead_attention     (const Cross_multihead_attention_params<float>    &params, const cudaStream_t &stream);
void cross_multihead_attention     (const Cross_multihead_attention_params<uint16_t> &params, const cudaStream_t &stream);

////////////////////////////////////////////////////////////////////////////////////////////////////

