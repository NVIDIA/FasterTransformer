/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

#include "fastertransformer/th_op/gpt.h"

#ifdef USE_NVTX
  bool NVTX_ON = true;
#endif

namespace torch_ext {
using torch::Tensor;
using std::vector;

FasterTransformerGPT::FasterTransformerGPT(
    const int64_t batch_size,
    const int64_t head_num,
    const int64_t size_per_head,
    const int64_t vocab_size,
    const int64_t start_id,
    const int64_t end_id,
    const int64_t decoder_layers,
    const int64_t candidate_num,
    const double probability_threshold,
    const double temperature,
    const int64_t input_len,
    const int64_t output_len,
    const int64_t max_seq_len,
    const int64_t tensor_para_size,
    const int64_t layer_para_size,
    const int64_t layer_para_batch_size,
    const bool is_fuse_QKV,
    const int max_batch_size,
    Tensor embedding_table,
    Tensor position_encoding_table,
    vector<Tensor> self_layernorm_gamma,
    vector<Tensor> self_layernorm_beta,
    vector<Tensor> self_kernel,
    vector<Tensor> self_bias,
    vector<Tensor> self_output_kernel,
    vector<Tensor> self_output_bias,
    vector<Tensor> ffn_layernorm_gamma,
    vector<Tensor> ffn_layernorm_beta,
    vector<Tensor> ffn_kernel1,
    vector<Tensor> ffn_kernel2,
    vector<Tensor> ffn_bias1,
    vector<Tensor> ffn_bias2,
    Tensor layernorm_gamma,
    Tensor layernorm_beta)
    : st_(self_layernorm_gamma[0].scalar_type()),
      batch_size_(batch_size),
      input_len_(input_len),
      output_len_(output_len),
      max_seq_len_(max_seq_len),
      weights_transformer{self_layernorm_gamma,
              self_layernorm_beta,
              self_kernel,
              self_bias,
              self_output_kernel,
              self_output_bias,
              ffn_layernorm_gamma,
              ffn_layernorm_beta,
              ffn_kernel1,
              ffn_kernel2,
              ffn_bias1,
              ffn_bias2},
      weights{embedding_table,
              position_encoding_table,
              layernorm_gamma,
              layernorm_beta}
{
  CHECK_INPUT(embedding_table, st_); 
  CHECK_INPUT(position_encoding_table, st_);
  for (int i=0; i<decoder_layers; i++) {
    CHECK_INPUT(self_layernorm_gamma[i], st_); 
    CHECK_INPUT(self_layernorm_beta[i], st_);
    CHECK_INPUT(self_kernel[i], st_); 
    CHECK_INPUT(self_bias[i], st_);
    CHECK_INPUT(self_output_kernel[i], st_); 
    CHECK_INPUT(self_output_bias[i], st_);
    CHECK_INPUT(ffn_layernorm_gamma[i], st_); 
    CHECK_INPUT(ffn_layernorm_beta[i], st_);
    CHECK_INPUT(ffn_kernel1[i], st_); 
    CHECK_INPUT(ffn_kernel2[i], st_);
    CHECK_INPUT(ffn_bias1[i], st_); 
    CHECK_INPUT(ffn_bias2[i], st_);
  }
  CHECK_INPUT(layernorm_gamma, st_); 
  CHECK_INPUT(layernorm_beta, st_);

  switch (st_) {
  case at::ScalarType::Float:
    gpt = new torch_ext::GPT<float>(batch_size, head_num, size_per_head, vocab_size,
                                    start_id, end_id, decoder_layers, candidate_num, probability_threshold, temperature, 
                                    input_len, output_len, max_seq_len, 
                                    tensor_para_size, layer_para_size, layer_para_batch_size, is_fuse_QKV, max_batch_size, weights_transformer, weights);
    break;
  case at::ScalarType::Half:
    gpt = new torch_ext::GPT<half>(batch_size, head_num, size_per_head, vocab_size,
                                    start_id, end_id, decoder_layers, candidate_num, probability_threshold, temperature, 
                                    input_len, output_len, max_seq_len, 
                                    tensor_para_size, layer_para_size, layer_para_batch_size, is_fuse_QKV, max_batch_size, weights_transformer, weights);
    break;
  default:
    throw std::runtime_error("Wrong Tensor type.");
  }
};

FasterTransformerGPT::~FasterTransformerGPT() {
  delete gpt;
};

std::vector<Tensor> FasterTransformerGPT::forward(Tensor start_ids, Tensor start_lengths, Tensor attn_mask) {
  CHECK_INPUT(start_ids, at::ScalarType::Int);
  CHECK_INPUT(start_lengths, at::ScalarType::Int);
  CHECK_CUDA(attn_mask); CHECK_CONTIGUOUS(attn_mask);

  auto output_ids = torch::empty({input_len_ + output_len_, batch_size_}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

  attn_mask = attn_mask.to(st_);
  gpt->forward(start_ids, start_lengths, attn_mask, output_ids);

  return std::vector<Tensor>{output_ids};
};

} // namespace torch_ext