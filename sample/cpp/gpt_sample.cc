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

#include "fastertransformer/open_decoder.h"
#include "fastertransformer/gpt.h"
#include "fastertransformer/utils/INIReader.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_profiler_api.h>
#include <iostream>
#include <sys/time.h>
#include <cuda_fp16.h>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include "fastertransformer/utils/nvtx_utils.h"

static std::string MODEL_PATH_PREFIX;

#ifdef USE_NVTX
  bool NVTX_ON = true;
#endif

static inline std::string path_to_weights(const char *file, int layernum = -1, int gpu_num = 1)
{
  if (layernum == -1)
    return MODEL_PATH_PREFIX + std::to_string(gpu_num) + "-gpu/model." + file;
  else
  {
    return MODEL_PATH_PREFIX + std::to_string(gpu_num) + "-gpu/model.layers." + std::to_string(layernum) + "." + file;
  }
}

static inline std::string add_rank_to_path(std::string str, int rank)
{
  return str + std::to_string(rank) + ".bin";
}

using namespace fastertransformer;

template <typename T>
void device_malloc(T **ptr, int size);

template <typename T>
void decoding_sample(const INIReader reader);

int main(int argc, char *argv[])
{
  MPICHECK( MPI_Init(&argc, &argv));
  srand(0);
  struct cudaDeviceProp prop;
  check_cuda_error(cudaGetDeviceProperties(&prop, 0));
  printf("Device %s\n", prop.name);

  std::string ini_name;
  if(argc == 2)
    ini_name = std::string(argv[1]);
  else
    ini_name = "../sample/cpp/gpt_config.ini";

  INIReader reader = INIReader(ini_name);
  if (reader.ParseError() < 0) {
    std::cout << "[ERROR] Can't load '" << ini_name << "'\n";
    return -1;
  }
  const int is_half = reader.GetInteger("ft_instance_hyperparameter", "is_half");
  MODEL_PATH_PREFIX = reader.Get("ft_instance_hyperparameter", "model_path_prefix");

  if (is_half == 0)
    decoding_sample<float>(reader);
  else if (is_half == 1)
    decoding_sample<half>(reader);
  else
  {
    printf("[ERROR] is_fp16 should be 0 (use float) or 1 (use half). \n");
    return -1;
  }
  MPI_Finalize();
  return 0;
}

template <typename T>
void device_malloc(T **ptr, int size)
{
  check_cuda_error(cudaMalloc((void **)ptr, sizeof(T) * size));
  cuda_random_uniform_kernelLauncher(*ptr, size);
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());
}

template <typename T>
void device_malloc_zero(T **ptr, int size)
{
  check_cuda_error(cudaMalloc((void **)ptr, sizeof(T) * size));
  check_cuda_error(cudaMemset(*ptr, 0, sizeof(T) * size));
}

template <typename T>
int init_device_from_bin(T **ptr, std::vector<int> shape, std::string filename, int split = 1)
{
  if (shape.size() > 2)
  {
    printf("[ERROR] shape should have less than two dims \n");
    return -1;
  }
  int dim0 = shape[0], dim1 = 1;
  if (shape.size() == 2)
  {
    dim1 = shape[1];
  }
  size_t size = dim0 * dim1;

  std::vector<float> host_array(size);

  std::ifstream in(filename, std::ios::in | std::ios::binary);
  if(!in.is_open())
  {
    printf("[WARNING] file %s cannot be opened, initializing weights with random values! \n", filename.c_str());
    device_malloc(ptr, size);
    return 0;
  }

  size_t float_data_size = sizeof(float) * size;
  in.read((char*)host_array.data(), float_data_size);

  size_t in_get_size = in.gcount();
  if(in_get_size != float_data_size)
  {
    printf("[WARNING] file %s only has %ld, but request %ld, initializing weights with random values! \n",
      filename.c_str(), in_get_size, float_data_size);
    device_malloc(ptr, size);
    return 0;
  }

  check_cuda_error(cudaMalloc((void **)ptr, sizeof(T) * size));
  if(std::is_same<T, float>::value == true)
    cudaMemcpy(*ptr, host_array.data(), sizeof(T) * size, cudaMemcpyHostToDevice);
  else
  {
    std::vector<T> host_array_2(size);
    for(size_t i = 0; i < size; i++)
    {
      host_array_2[i] = __float2half(host_array[i]);
    }
    cudaMemcpy(*ptr, host_array_2.data(), sizeof(T) * size, cudaMemcpyHostToDevice);
  }
  return 0;
}

template <typename T>
int init_device_from_csv(T **ptr, std::vector<int> shape, std::string filename, int split = 1)
{
  if (shape.size() > 2)
  {
    printf("[ERROR] shape should have less than two dims \n");
    return -1;
  }
  int dim0 = shape[0], dim1 = 1;
  if (shape.size() == 2)
  {
    dim1 = shape[1];
  }
  size_t size = dim0 * dim1;

  int split_boundary = (dim1 + split - 1) / split;
  size_t size_each = size / split;
  size_t dim1_each = dim1 / split;

  bool dim0_reached = false, dim1_reached = false;
  int i0 = 0, i1;

  std::ifstream file(filename);
  std::vector<T> host_array(size);

  if (file.is_open())
  {
    std::string line;
    while (std::getline(file, line))
    {
      if (i0 == dim0)
      {
        dim0_reached = true;
        break;
      }

      std::stringstream lineStream(line);
      std::string vals;
      i1 = 0;
      while (std::getline(lineStream, vals, ','))
      {
        if (i1 == dim1)
        {
          dim1_reached = true;
          break;
        }
        if (split > 1)
        {
          int idx = i1 / split_boundary;
          int i11 = i1 % split_boundary;
          if (sizeof(T) == sizeof(float))
            host_array[i0 * dim1_each + (idx * size_each) + i11] = std::stof(vals);
          else
            host_array[i0 * dim1_each + (idx * size_each) + i11] = __float2half(std::stof(vals));
        }
        else
        {
          if (sizeof(T) == sizeof(float))
            host_array[i0 * dim1 + i1] = std::stof(vals);
          else
            host_array[i0 * dim1 + i1] = __float2half(std::stof(vals));
        }
        i1++;
      }
      i0++;
    }
  }
  else
  {
    printf("[WARNING] file %s cannot be opened, initializing weights with random values! \n", filename.c_str());
    device_malloc(ptr, size);
    return 0;
  }
  check_cuda_error(cudaMalloc((void **)ptr, sizeof(T) * size));
  cudaMemcpy(*ptr, host_array.data(), sizeof(T) * size, cudaMemcpyHostToDevice);
  if (dim0_reached)
    printf("[WARNING] the file dimension does not match with input dim0! %s, dim0=%d, i0=%d\n", filename.c_str(), dim0, i0);
  if (dim1_reached)
    printf("[WARNING] the file dimension does not match with input dim1! %s, dim1=%d, i1=%d\n", filename.c_str(), dim1, i1);

  return 0;
}

template <typename T>
int init_device_from_file(T **ptr, std::vector<int> shape, std::string filename, int split = 1, std::string type="bin")
{
  std::cout << "[INFO] load ckpt from " << filename << "                                  \r" << std::flush;
  if(type == "bin")
  {
    init_device_from_bin(ptr, shape, filename, split);
  }
  else if(type == "csv")
  {
    init_device_from_csv(ptr, shape, filename, split);
  }
  else
  {
    printf("[ERROR] not support type: %s \n", type.c_str());
    exit(-1);
  }
  return 0;
}

int read_start_ids(int batch_size, std::vector<int>*v_start_lengths, std::vector<int>*v_start_ids, 
                   int& max_input_len, const int end_id)
{
  std::vector<std::vector<int>> tmp_start_ids;

  std::ifstream start_id_file("../sample/cpp/start_ids.csv", std::ios::in);
  if (start_id_file.is_open())
  {
    std::string line;
    int i0 = 0;
    while (std::getline(start_id_file, line))
    {
      std::stringstream lineStream(line);
      std::string vals;
      int i1 = 0;
      std::vector<int> tmp_vec; 
      while (std::getline(lineStream, vals, ','))
      {
        tmp_vec.push_back(std::stoi(vals));
        i1++;
      }
      tmp_start_ids.push_back(tmp_vec);
      v_start_lengths->push_back(i1);
      i0++;
    }
  }
  else
  {
    printf("[ERROR] Cannot open the file '../sample/cpp/start_ids.csv'. \n");
    exit(-1);
  }

  max_input_len = v_start_lengths->data()[0];
  for(uint i = 1; i < (uint)v_start_lengths->size(); i++)
  {
    max_input_len = max_input_len > v_start_lengths->data()[i] ? max_input_len : v_start_lengths->data()[i];
  }

  while((int)v_start_lengths->size() < batch_size)
  {
    std::vector<int> padding_ids;
    for(int i = 0; i < max_input_len; i++)
      padding_ids.push_back(50256);
    tmp_start_ids.push_back(padding_ids);
    v_start_lengths->push_back(max_input_len);
  }

  // Add padding
  for(int i = 0; i < (int)tmp_start_ids.size(); i++)
  {
    for(int j = (int)tmp_start_ids[i].size(); j < max_input_len; j++)
    {
      tmp_start_ids[i].push_back(end_id);
    }
  }

  for(int i = 0; i < (int)tmp_start_ids.size(); i++)
  {
    for(int j = 0; j < (int)tmp_start_ids[i].size(); j++)
    {
      v_start_ids->push_back(tmp_start_ids[i][j]);
    }
  }
  return 0;
}

template <typename T>
void decoding_sample(const INIReader reader)
{
  const std::string model_name = reader.Get("ft_instance_hyperparameter", "model_name");
  const int max_batch_size = reader.GetInteger("ft_instance_hyperparameter", "max_batch_size");
  const int max_seq_len = reader.GetInteger("ft_instance_hyperparameter", "max_seq_len");
  const int candidate_num = reader.GetInteger("ft_instance_hyperparameter", "candidate_num");
  const float probability_threshold = reader.GetFloat("ft_instance_hyperparameter", "probability_threshold");
  const float temperature = reader.GetFloat("ft_instance_hyperparameter", "temperature");
  const int tensor_para_size = reader.GetInteger("ft_instance_hyperparameter", "tensor_para_size");
  const int layer_para_size = reader.GetInteger("ft_instance_hyperparameter", "layer_para_size");
  const int layer_para_batch_size = reader.GetInteger("ft_instance_hyperparameter", "layer_para_batch_size");
  const bool is_fuse_QKV = (bool)(reader.GetInteger("ft_instance_hyperparameter", "is_fuse_QKV"));

  const int head_num = reader.GetInteger(model_name, "head_num");
  const int size_per_head = reader.GetInteger(model_name, "size_per_head");
  const int vocab_size = reader.GetInteger(model_name, "vocab_size");
  const int decoder_layers = reader.GetInteger(model_name, "decoder_layers");
  
  const int request_batch_size = reader.GetInteger("request", "request_batch_size");
  const int request_input_len = reader.GetInteger("request", "request_input_len"); // The length of the conditioned context
  const int request_output_len = reader.GetInteger("request", "request_output_len"); // The length of tokens we hope this model to generate
  const int total_output_len = request_input_len + request_output_len;

  if(is_fuse_QKV != true)
    MODEL_PATH_PREFIX = MODEL_PATH_PREFIX + "unfusedQKV-";

  const int start_id = 50256;
  const int end_id = 50256;

  // Read ids of request from file.
  std::vector<int> v_start_lengths;
  std::vector<int> v_start_ids;
  int max_input_len = -1;
  read_start_ids(request_batch_size, &v_start_lengths, &v_start_ids, 
                 max_input_len, end_id);

  int* start_ids = v_start_ids.data();
  for(int i = 0; i < request_batch_size; i++)
  {
    if(request_input_len > v_start_lengths[i]) 
    {
      printf("[ERROR] input length (%d) should be smaller or equal to all start lengths (%d). \n", request_input_len, v_start_lengths[i]);
      exit(-1);
    }
  }

  if(request_input_len < 0)
  {
    printf("[ERROR] request_input_len should be >= 0 because this model requires some start ids as inputs. \n");
    exit(-1);
  }
  else if(request_input_len == 0)
  {
    printf("[WARNING] No inputs. Use start_id as the first token.");
    v_start_ids.clear();
    for(int i = 0; i < request_batch_size; i++) v_start_ids.push_back(start_id);
  }

  if(total_output_len > max_seq_len)
  {
    printf("[ERROR] total_output_len (%d) should be <= max_seq_len (%d). \n", total_output_len, max_seq_len);
    exit(-1);
  }

  // Prepare the parallelism parameters
  int rank, world_size, device, device_count;
  MPICHECK( MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK( MPI_Comm_size(MPI_COMM_WORLD, &world_size));
  assert(head_num % tensor_para_size == 0);
  if (rank==0) printf("Total ranks: %d.\n", world_size);
  CUDACHECK(cudaGetDeviceCount(&device_count));
  CUDACHECK(cudaSetDevice(rank % device_count));
  CUDACHECK(cudaGetDevice(&device));

  printf("P%d is runing with %d GPU.\n", rank, device);

  if(tensor_para_size * layer_para_size != world_size)
  {
    printf("[ERROR] tensor_para_size * layer_para_size should equal to world_size \n");
    exit(-1);
  }

  const int tensor_para_rank = rank % tensor_para_size;
  const int layer_para_rank = rank / tensor_para_size;
  const int layers_per_group = decoder_layers / layer_para_size;
  if(layers_per_group * layer_para_size != decoder_layers)
  {
    printf("[ERROR] layers_per_group (%d) * layer_para_size (%d) should equal to decoder_layers (%d) \n", layers_per_group, layer_para_size, decoder_layers);
    exit(-1);
  }
  ncclUniqueId tensor_para_nccl_uid;
  ncclUniqueId layer_para_nccl_uid;
  
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

  ncclComm_t tensor_para_nccl_comm, layer_para_nccl_comm;
  NCCLCHECK( ncclCommInitRank(&tensor_para_nccl_comm, tensor_para_size, tensor_para_nccl_uid, tensor_para_rank));
  NCCLCHECK( ncclCommInitRank(&layer_para_nccl_comm, layer_para_size, layer_para_nccl_uid, layer_para_rank));

  cublasHandle_t cublasHandle;
  cublasLtHandle_t cublasLtHandle;
  check_cuda_error(cublasCreate(&cublasHandle));
  check_cuda_error(cublasLtCreate(&cublasLtHandle));

  cudaStream_t stream;
  check_cuda_error(cudaStreamCreate(&stream));
  check_cuda_error(cublasSetStream(cublasHandle, stream));

  if(head_num % tensor_para_size != 0)
  {
    printf("[ERROR] head_num mod tensor_para_size should be 0. Here, head_num is %d, and tensor_para_size is %d. \n", head_num, tensor_para_size);
    exit(-1);
  }

  const int local_head_num = head_num / tensor_para_size; 
  const int global_head_num = head_num; 
  const int local_hidden_units = local_head_num * size_per_head;
  const int global_hidden_units = global_head_num * size_per_head;
  const int local_inner_size = local_hidden_units * 4;

  TensorParallelParam tensor_parallel_param;
  tensor_parallel_param.rank = tensor_para_rank;
  tensor_parallel_param.world_size = tensor_para_size;
  tensor_parallel_param.nccl_comm = tensor_para_nccl_comm;
  tensor_parallel_param.local_head_num_ = local_head_num;
  tensor_parallel_param.local_hidden_units_ = local_hidden_units;

  LayerParallelParam layer_parallel_param;
  layer_parallel_param.rank = layer_para_rank;
  layer_parallel_param.world_size = layer_para_size;
  layer_parallel_param.nccl_comm = layer_para_nccl_comm;
  layer_parallel_param.layers_per_group = layers_per_group;
  layer_parallel_param.local_batch_size = layer_para_batch_size;

  fastertransformer::Allocator<AllocatorType::CUDA> allocator(device);
  DecoderInitParam<T> *decoder_param = new DecoderInitParam<T>[decoder_layers];

  for (int i = 0; i < decoder_layers; i++)
  {
    if(layer_parallel_param.is_valid(i) == false) continue;
    decoder_param[i].request_batch_size = request_batch_size;
    decoder_param[i].stream = stream;
    decoder_param[i].cublas_handle = cublasHandle;
    decoder_param[i].cublaslt_handle = cublasLtHandle;

    T *d_self_Q_kernel, *d_self_K_kernel, *d_self_V_kernel, *d_self_output_kernel;
    T *d_self_bias;
    T *d_self_Q_bias, *d_self_K_bias, *d_self_V_bias, *d_self_output_bias;
    T *d_ffn_kernel1, *d_ffn_bias1, *d_ffn_kernel2, *d_ffn_bias2;
    T *d_self_gamma, *d_self_beta;
    T *d_ffn_gamma, *d_ffn_beta;

    init_device_from_file(&d_self_Q_kernel, {global_hidden_units, local_hidden_units * 3}, path_to_weights(add_rank_to_path("attention.query_key_value.weight.", tensor_para_rank).c_str(), i, tensor_para_size), 3);
    d_self_K_kernel = d_self_Q_kernel + global_hidden_units * local_hidden_units;
    d_self_V_kernel = d_self_K_kernel + global_hidden_units * local_hidden_units;

    init_device_from_file(&d_self_output_kernel, {local_hidden_units, global_hidden_units}, path_to_weights(add_rank_to_path("attention.dense.weight.", tensor_para_rank).c_str(), i, tensor_para_size));

    init_device_from_file(&d_self_bias, {local_hidden_units * 3}, path_to_weights(add_rank_to_path("attention.query_key_value.bias.", tensor_para_rank).c_str(), i, tensor_para_size));
    d_self_Q_bias = d_self_bias;
    d_self_K_bias = d_self_Q_bias + local_hidden_units;
    d_self_V_bias = d_self_K_bias + local_hidden_units;
    
    init_device_from_file(&d_self_output_bias, {global_hidden_units}, path_to_weights("attention.dense.bias.bin", i, tensor_para_size));

    init_device_from_file(&d_ffn_bias1, {local_inner_size}, path_to_weights(add_rank_to_path("mlp.dense_h_to_4h.bias.", tensor_para_rank).c_str(), i, tensor_para_size));
    init_device_from_file(&d_ffn_bias2, {global_hidden_units}, path_to_weights("mlp.dense_4h_to_h.bias.bin", i, tensor_para_size));

    init_device_from_file(&d_ffn_kernel1, {global_hidden_units, local_inner_size}, path_to_weights(add_rank_to_path("mlp.dense_h_to_4h.weight.", tensor_para_rank).c_str(), i, tensor_para_size));
    init_device_from_file(&d_ffn_kernel2, {local_inner_size, global_hidden_units}, path_to_weights(add_rank_to_path("mlp.dense_4h_to_h.weight.", tensor_para_rank).c_str(), i, tensor_para_size));

    init_device_from_file(&d_self_gamma, {global_hidden_units}, path_to_weights("input_layernorm.weight.bin", i, tensor_para_size));
    init_device_from_file(&d_self_beta, {global_hidden_units}, path_to_weights("input_layernorm.bias.bin", i, tensor_para_size));
    init_device_from_file(&d_ffn_gamma, {global_hidden_units}, path_to_weights("post_attention_layernorm.weight.bin", i, tensor_para_size));
    init_device_from_file(&d_ffn_beta, {global_hidden_units}, path_to_weights("post_attention_layernorm.bias.bin", i, tensor_para_size));

    decoder_param[i].self_layernorm.gamma = d_self_gamma;
    decoder_param[i].self_layernorm.beta = d_self_beta;
    decoder_param[i].self_attention.query_weight.kernel = d_self_Q_kernel;
    decoder_param[i].self_attention.key_weight.kernel = d_self_K_kernel;
    decoder_param[i].self_attention.value_weight.kernel = d_self_V_kernel;
    decoder_param[i].self_attention.attention_output_weight.kernel = d_self_output_kernel;
    decoder_param[i].self_attention.query_weight.bias = d_self_Q_bias;
    decoder_param[i].self_attention.key_weight.bias = d_self_K_bias;
    decoder_param[i].self_attention.value_weight.bias = d_self_V_bias;
    decoder_param[i].self_attention.attention_output_weight.bias = d_self_output_bias;

    decoder_param[i].ffn_layernorm.gamma = d_ffn_gamma;
    decoder_param[i].ffn_layernorm.beta = d_ffn_beta;
    decoder_param[i].ffn.intermediate_weight.bias = d_ffn_bias1;
    decoder_param[i].ffn.output_weight.bias = d_ffn_bias2;
    decoder_param[i].ffn.intermediate_weight.kernel = d_ffn_kernel1;
    decoder_param[i].ffn.output_weight.kernel = d_ffn_kernel2;
  }

  DecodingInitParam<T> decoding_params;

  T *d_embedding_table;
  T *d_position_encoding_table;
  T *d_embedding_kernel;
  int *d_output_ids;
  T *d_gamma, *d_beta;

  init_device_from_file(&d_embedding_table, {vocab_size, global_hidden_units}, path_to_weights("wte.bin", -1, tensor_para_size));
  init_device_from_file(&d_position_encoding_table, {max_seq_len, global_hidden_units}, path_to_weights("wpe.bin", -1, tensor_para_size));
  d_embedding_kernel = d_embedding_table;

  check_cuda_error(cudaMalloc((void **)&d_output_ids, sizeof(int) * (request_input_len + request_output_len) * request_batch_size));
  init_device_from_file(&d_gamma, {global_hidden_units}, path_to_weights("final_layernorm.weight.bin", -1, tensor_para_size));
  init_device_from_file(&d_beta, {global_hidden_units}, path_to_weights("final_layernorm.bias.bin", -1, tensor_para_size));

  decoding_params.cublas_handle = cublasHandle;
  decoding_params.cublaslt_handle = cublasLtHandle;
  decoding_params.stream = stream;
  decoding_params.embedding_table = d_embedding_table;
  decoding_params.position_encoding_table = d_position_encoding_table;
  decoding_params.embedding_kernel = d_embedding_kernel;
  decoding_params.output_ids = d_output_ids;
  decoding_params.layernorm.gamma = d_gamma;
  decoding_params.layernorm.beta = d_beta;

  decoding_params.request_batch_size = request_batch_size;
  decoding_params.request_input_len = request_input_len;
  decoding_params.request_output_len = request_output_len;
  decoding_params.max_input_len = max_input_len;

  const fastertransformer::OperationType type = std::is_same<T, float>::value ? OperationType::FP32 : OperationType::FP16;

  DecodingGpt<type> *decoding = new DecodingGpt<type>(allocator, max_batch_size, 
                                                        max_seq_len, global_head_num, size_per_head,
                                                        vocab_size, decoder_layers,
                                                        start_id, end_id,
                                                        candidate_num, probability_threshold,
                                                        temperature, tensor_para_size, layer_para_size, is_fuse_QKV);
  decoding->set_tensor_parallel_param(tensor_parallel_param);
  decoding->set_layer_parallel_param(layer_parallel_param);

  int* d_start_ids;
  cudaMalloc((void **)&d_start_ids, sizeof(int) * request_batch_size * max_input_len);
  cudaMemcpyAsync(d_start_ids, start_ids, sizeof(int) * request_batch_size * max_input_len, cudaMemcpyHostToDevice, stream);
  T* h_attn_mask = new T[request_batch_size * request_input_len * request_input_len];
  memset(h_attn_mask, 0, sizeof(T) * request_batch_size * request_input_len * request_input_len);
  for(int i = 0; i < request_batch_size; i++)
  {
    for(int j = 0; j < request_input_len; j++)
    {
      for(int k = 0; k <= j; k++)
      {
        h_attn_mask[i * request_input_len * request_input_len + j * request_input_len + k] = (T)1.0f;
      }
    }
  }
  T* d_attn_mask;
  cudaMalloc((void **)&d_attn_mask, sizeof(T) * request_batch_size * request_input_len * request_input_len);
  cudaMemcpyAsync(d_attn_mask, h_attn_mask, sizeof(T) * request_batch_size * request_input_len * request_input_len, cudaMemcpyHostToDevice, stream);
  decoding_params.d_start_ids = d_start_ids;
  decoding_params.d_attn_mask = d_attn_mask;
  
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());
  MPI_Barrier(MPI_COMM_WORLD);

  print_mem_usage();

  int ite = 1;
  cudaDeviceSynchronize();
  MPI_Barrier(MPI_COMM_WORLD);
  
  cudaProfilerStart();
  // warm up
  ite = 1;
  nvtx::set_scope("warmup_time");
  PUSH_RANGE("warmup time")
  for (int i = 0; i < ite; ++i)
  {
    decoding->forward_context(decoder_param, decoding_params);
    decoding->forward(decoder_param, decoding_params);
  }
  cudaDeviceSynchronize();
  MPI_Barrier(MPI_COMM_WORLD);
  POP_RANGE;
  nvtx::reset_scope();

  struct timeval start, end;
  struct timeval context_start, context_end;
  cudaDeviceSynchronize();
  MPI_Barrier(MPI_COMM_WORLD);
  gettimeofday(&start, NULL);

  nvtx::set_scope("total_time");
  PUSH_RANGE("total time")
  for (int i = 0; i < ite; ++i)
  {
    gettimeofday(&context_start, NULL);
    decoding->forward_context(decoder_param, decoding_params);
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&context_end, NULL);
    decoding->forward(decoder_param, decoding_params);
  }

  cudaDeviceSynchronize();
  MPI_Barrier(MPI_COMM_WORLD);
  POP_RANGE;
  nvtx::reset_scope();
  gettimeofday(&end, NULL);

  cudaProfilerStop();


  printf("[INFO] batch_size %d head_num %d size_per_head %d total_output_len %d"
         " decoder_layers %d vocab_size %d FT-CPP-decoding-beamsearch-time %.2f ms (context time: %.2f ms)\n",
         request_batch_size, head_num, size_per_head, total_output_len, decoder_layers, vocab_size,
         ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001) / ite,
         ((context_end.tv_sec - context_start.tv_sec) * 1000 + (context_end.tv_usec - context_start.tv_usec) * 0.001) / ite);
    
  if(rank == 0)
  {
    
    std::string fName = "out";
    auto outFile = std::ofstream(fName, std::ios::out);
    if(!outFile.is_open())
    {
      printf("[WARNING] Cannot write results into output file %s \n", fName.c_str());
    }
    else
    {
      size_t outCount = total_output_len * request_batch_size;
      int *hBuf = new int[outCount];
      cudaDeviceSynchronize();
      cudaMemcpyAsync(hBuf, d_output_ids, outCount * sizeof(int), cudaMemcpyDeviceToHost, stream);
      cudaDeviceSynchronize();

      {
        std::cout << "Writing " << outCount << " elements\n";
        int zerroCount = 0;
        for (size_t i = 0; i < outCount; i++)
        {
            if (hBuf[i] == int(0)) zerroCount++;
            outFile << hBuf[i] << " ";
            if((i+1) % (request_batch_size) == 0) outFile << std::endl;

            if( i < 10) printf("%5d ", hBuf[i]);
            if((i+1) % (request_batch_size) == 0 && i < 10) std::cout << std::endl;
        }
        std::cout << std::endl << "zerroCount = " << zerroCount << std::endl;
      }
      delete [] hBuf;
    }
  }

  ncclCommDestroy(tensor_para_nccl_comm);
  ncclCommDestroy(layer_para_nccl_comm);
  delete [] decoder_param;
  delete [] h_attn_mask;
  delete decoding;
  return;
}
