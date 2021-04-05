/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <vector>
#include <iostream>
#include <cuda_fp16.h>

#include <torch/script.h>
#include <torch/custom_class.h>
#include "torch/extension.h"
#include "torch/csrc/cuda/Stream.h"
#include <ATen/ATen.h>

#include "fastertransformer/gpt.h"
#include "fastertransformer/th_op/th_traits.h"
#include "fastertransformer/th_op/utils.h"

namespace torch_ext {
using namespace fastertransformer;
using torch_ext::get_ptr;
using torch::Tensor;
using std::vector;

class IGPT {
public:
  virtual ~IGPT() {}
  virtual void forward(Tensor& start_ids, Tensor& start_lengths, Tensor& attn_mask, Tensor& output_ids, int output_len) = 0;
};

template <typename T>
class GPT : public IGPT {
public:
  GPT(
      const int head_num,
      const int size_per_head,
      const int vocab_size,
      const int start_id,
      const int end_id,
      const int decoder_layers,
      const int candidate_num,
      const float probability_threshold,
      const float temperature,
      const int max_seq_len,
      const int tensor_para_size,
      const int layer_para_size,
      const int layer_para_batch_size,
      const bool is_fuse_QKV,
      const int max_batch_size,
      vector<vector<Tensor>> weights_transformer,
      vector<Tensor> weights)
      : max_seq_len_(max_seq_len), size_per_head_(size_per_head),
        vocab_size_(vocab_size), decoder_layers_(decoder_layers), start_id_(start_id), end_id_(end_id),
        candidate_num_(candidate_num), probability_threshold_(probability_threshold), temperature_(temperature),
        is_fuse_QKV_(is_fuse_QKV), max_batch_size_(max_batch_size)
  {
    const int local_head_num = head_num / tensor_para_size; 
    const int global_head_num = head_num; 
    const int local_hidden_units = local_head_num * size_per_head;
    const int global_hidden_units = global_head_num * size_per_head;
    const int local_inner_size = local_hidden_units * 4;

    global_head_num_ = global_head_num;

    // Set tensor and layer parallel
    int rank;
    MPICHECK( MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    const int tensor_para_rank = rank % tensor_para_size;
    const int layer_para_rank = rank / tensor_para_size;

    ncclUniqueId tensor_para_nccl_uid;
    ncclUniqueId layer_para_nccl_uid;
    init_nccl_comm(tensor_para_nccl_uid, layer_para_nccl_uid, rank, tensor_para_size, layer_para_size, tensor_para_rank, layer_para_rank);

    tensor_para_size_ = tensor_para_size;
    layer_para_size_ = layer_para_size;

    const int layers_per_group = decoder_layers / layer_para_size;
    assert(decoder_layers % layer_para_size == 0);

    tensor_parallel_param.rank = tensor_para_rank;
    tensor_parallel_param.world_size = tensor_para_size;
    tensor_parallel_param.nccl_comm = tensor_para_nccl_comm;
    tensor_parallel_param.local_head_num_ = local_head_num;
    tensor_parallel_param.local_hidden_units_ = local_hidden_units;

    layer_parallel_param.rank = layer_para_rank;
    layer_parallel_param.world_size = layer_para_size;
    layer_parallel_param.nccl_comm = layer_para_nccl_comm;
    layer_parallel_param.layers_per_group = layers_per_group;
    layer_parallel_param.local_batch_size = layer_para_batch_size;

    // Set weights
    vector<Tensor> self_layernorm_gamma = weights_transformer[0];
    vector<Tensor> self_layernorm_beta = weights_transformer[1];
    vector<Tensor> self_kernel = weights_transformer[2];
    vector<Tensor> self_bias = weights_transformer[3];
    vector<Tensor> self_output_kernel = weights_transformer[4];
    vector<Tensor> self_output_bias = weights_transformer[5];
    vector<Tensor> ffn_layernorm_gamma = weights_transformer[6];
    vector<Tensor> ffn_layernorm_beta = weights_transformer[7];
    vector<Tensor> ffn_kernel1 = weights_transformer[8];
    vector<Tensor> ffn_kernel2 = weights_transformer[9];
    vector<Tensor> ffn_bias1 = weights_transformer[10];
    vector<Tensor> ffn_bias2 = weights_transformer[11];
    Tensor embedding_table = weights[0];
    Tensor position_encoding_table = weights[1];
    Tensor layernorm_gamma = weights[2];
    Tensor layernorm_beta = weights[3];

    decoding_params.embedding_table = get_ptr<T>(embedding_table);
    decoding_params.position_encoding_table = get_ptr<T>(position_encoding_table);

    param = new DecoderInitParam<T>[decoder_layers];
    for (int i = 0; i < decoder_layers; i++)
    {
      if(layer_parallel_param.is_valid(i) == false) continue;

      T *self_Q_kernel, *self_K_kernel, *self_V_kernel;
      T *self_Q_bias, *self_K_bias, *self_V_bias;

      self_Q_kernel = get_ptr<T>(self_kernel[i]);
      self_K_kernel = self_Q_kernel + global_hidden_units * local_hidden_units;
      self_V_kernel = self_K_kernel + global_hidden_units * local_hidden_units;

      self_Q_bias = get_ptr<T>(self_bias[i]);
      self_K_bias = self_Q_bias + local_hidden_units;
      self_V_bias = self_K_bias + local_hidden_units;

      param[i].self_layernorm.gamma = get_ptr<T>(self_layernorm_gamma[i]);
      param[i].self_layernorm.beta = get_ptr<T>(self_layernorm_beta[i]);
      param[i].self_attention.query_weight.kernel = self_Q_kernel;
      param[i].self_attention.key_weight.kernel = self_K_kernel;
      param[i].self_attention.value_weight.kernel = self_V_kernel;
      param[i].self_attention.attention_output_weight.kernel = get_ptr<T>(self_output_kernel[i]);
      param[i].self_attention.query_weight.bias = self_Q_bias;
      param[i].self_attention.key_weight.bias = self_K_bias;
      param[i].self_attention.value_weight.bias = self_V_bias;
      param[i].self_attention.attention_output_weight.bias = get_ptr<T>(self_output_bias[i]);

      param[i].ffn_layernorm.gamma = get_ptr<T>(ffn_layernorm_gamma[i]);
      param[i].ffn_layernorm.beta = get_ptr<T>(ffn_layernorm_beta[i]);
      param[i].ffn.intermediate_weight.bias = get_ptr<T>(ffn_bias1[i]);
      param[i].ffn.output_weight.bias = get_ptr<T>(ffn_bias2[i]);
      param[i].ffn.intermediate_weight.kernel = get_ptr<T>(ffn_kernel1[i]);
      param[i].ffn.output_weight.kernel = get_ptr<T>(ffn_kernel2[i]);
    }

    decoding_params.embedding_kernel = get_ptr<T>(embedding_table);
    decoding_params.layernorm.gamma = get_ptr<T>(layernorm_gamma);
    decoding_params.layernorm.beta = get_ptr<T>(layernorm_beta);
  }

  void init_nccl_comm(ncclUniqueId &tensor_para_nccl_uid, ncclUniqueId &layer_para_nccl_uid,
                      int rank, int tensor_para_size, int layer_para_size,
                      int tensor_para_rank, int layer_para_rank)
  {
    // assume gpu_num = n * k,
    // tensor parallelism group size is n
    // layer parallelism group size is k

    if(tensor_para_rank == 0)
    {
      // get the uid of each tensor parallelism group
      // here, 0, 1, ..., n-1 are in group 0,
      //       n, ..., 2n - 1 are in group 1.
      NCCLCHECK( ncclGetUniqueId(&tensor_para_nccl_uid));
      for(int i = 1; i < tensor_para_size; i++)
      {
        printf("[INFO] rank %d sends tensor_para_nccl_uid to rank %d \n", rank, rank + i);
        MPICHECK( MPI_Send(&tensor_para_nccl_uid, sizeof(tensor_para_nccl_uid), MPI_BYTE, rank + i, 0, MPI_COMM_WORLD));
      }
    }
    else
    {
      MPI_Status status;
      printf("[INFO] rank %d receives tensor_para_nccl_uid from rank %d \n", rank, rank - tensor_para_rank);
      MPICHECK( MPI_Recv(&tensor_para_nccl_uid, sizeof(tensor_para_nccl_uid), MPI_BYTE, rank - tensor_para_rank, 0, MPI_COMM_WORLD, &status));
    }

    if(layer_para_rank == 0)
    {
      // get the uid of each layer parallelism group
      // 0, k, 2k, are in group 0
      // 1, k+1, 2k+1 are in group 1
      NCCLCHECK( ncclGetUniqueId(&layer_para_nccl_uid));
      for(int i = 1; i < layer_para_size; i++)
      {
        printf("[INFO] rank %d sends layer_para_nccl_uid to rank %d \n", rank, rank + i * tensor_para_size);
        MPICHECK( MPI_Send(&layer_para_nccl_uid, sizeof(layer_para_nccl_uid), MPI_BYTE, rank + i * tensor_para_size, 0, MPI_COMM_WORLD));
      }
    }
    else
    {
      MPI_Status status;
      printf("[INFO] rank %d receives layer_para_nccl_uid from rank %d \n", rank, rank % tensor_para_size);
      MPICHECK( MPI_Recv(&layer_para_nccl_uid, sizeof(layer_para_nccl_uid), MPI_BYTE, rank % tensor_para_size, 0, MPI_COMM_WORLD, &status));
    }

    NCCLCHECK( ncclCommInitRank(&tensor_para_nccl_comm, tensor_para_size, tensor_para_nccl_uid, tensor_para_rank));
    NCCLCHECK( ncclCommInitRank(&layer_para_nccl_comm, layer_para_size, layer_para_nccl_uid, layer_para_rank));
  }

  ~GPT() override {
    ncclCommDestroy(tensor_para_nccl_comm);
    ncclCommDestroy(layer_para_nccl_comm);

    delete [] param;
  }

  void forward(Tensor& start_ids, Tensor& start_lengths, Tensor& attn_mask, Tensor& output_ids, int output_len) override
  {
    // Set cudaStream, and cublasHandlers.
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cublasHandle_t cublasHandle;
    cublasLtHandle_t cublasLtHandle;
    check_cuda_error(cublasCreate(&cublasHandle));
    check_cuda_error(cublasLtCreate(&cublasLtHandle));
    check_cuda_error(cublasSetStream(cublasHandle, stream));

    decoding_params.cublas_handle = cublasHandle;
    decoding_params.cublaslt_handle = cublasLtHandle;
    decoding_params.stream = stream;

    int batch_size = start_ids.size(0);
    int input_len = at::min(start_lengths).item().to<int>();

    for (int i = 0; i < decoder_layers_; i++) {
      if(layer_parallel_param.is_valid(i) == false) continue;
      param[i].request_batch_size = batch_size;
      param[i].stream = stream;
      param[i].cublas_handle = cublasHandle;
      param[i].cublaslt_handle = cublasLtHandle;
    }

    fastertransformer::Allocator<AllocatorType::TH> allocator;
    // TODO This is a workaround. Better to move it into the constructor, but need to fix the "Caught signal 11 (Segmentation fault: address not mapped to object at address" error first.
    DecodingGpt<THTraits<T>::OpType> *decoding = new DecodingGpt<THTraits<T>::OpType>(allocator, max_batch_size_,
                                                                                        max_seq_len_, global_head_num_, size_per_head_,
                                                                                        vocab_size_, decoder_layers_,
                                                                                        start_id_, end_id_,
                                                                                        candidate_num_, probability_threshold_,
                                                                                        temperature_, tensor_para_size_, layer_para_size_, is_fuse_QKV_);
    
    decoding->set_tensor_parallel_param(tensor_parallel_param);
    decoding->set_layer_parallel_param(layer_parallel_param);

    decoding_params.request_batch_size = batch_size;
    decoding_params.request_input_len = input_len;
    decoding_params.request_output_len = output_len;
    decoding_params.d_start_ids = get_ptr<int>(start_ids);
    decoding_params.d_attn_mask = get_ptr<T>(attn_mask);
    decoding_params.max_input_len = at::max(start_lengths).item().to<int>();
    decoding_params.output_ids = get_ptr<int>(output_ids);

    decoding->forward_context(param, decoding_params);
    decoding->forward(param, decoding_params);

    // TODO This is a workaround. Better to move it into the deconstructor, but need to fix the "Caught signal 11 (Segmentation fault: address not mapped to object at address" error first.
    delete decoding;
  }

private:
  const int max_seq_len_;
  int global_head_num_;
  const int size_per_head_;
  const int vocab_size_;
  const int decoder_layers_;
  const int start_id_;
  const int end_id_;
  const int candidate_num_;
  const float probability_threshold_;
  const float temperature_;
  int tensor_para_size_;
  int layer_para_size_;
  const int is_fuse_QKV_;
  const int max_batch_size_;
  DecoderInitParam<T> *param;
  DecodingInitParam<T> decoding_params;
  TensorParallelParam tensor_parallel_param;
  LayerParallelParam layer_parallel_param;
  ncclComm_t tensor_para_nccl_comm, layer_para_nccl_comm;
};

class FasterTransformerGPT : public torch::jit::CustomClassHolder {
public:
  FasterTransformerGPT(
    const int64_t head_num,
    const int64_t size_per_head,
    const int64_t vocab_size,
    const int64_t start_id,
    const int64_t end_id,
    const int64_t decoder_layers,
    const int64_t candidate_num,
    const double probability_threshold,
    const double temperature,
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
    Tensor layernorm_beta
   );

  ~FasterTransformerGPT();
  
  vector<Tensor> forward(Tensor start_ids, Tensor start_lengths, Tensor attn_mask, int64_t output_len);

private:
  const at::ScalarType st_;
  int64_t max_seq_len_;
  vector<vector<Tensor>> weights_transformer;
  vector<Tensor> weights;
  IGPT* gpt;
};

} // namespace torch_ext