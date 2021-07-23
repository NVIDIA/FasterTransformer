
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

#include "fastertransformer/triton_backend/gpt_triton_backend.hpp"

using namespace fastertransformer;

#ifdef USE_NVTX
  bool NVTX_ON = true;
#endif

#ifdef USE_TRITONSERVER_DATATYPE
#include "triton/core/tritonbackend.h"
#endif

std::shared_ptr<AbstractTransformerModel> AbstractTransformerModel::createGptModel (std::string inifile)
{
  INIReader reader = INIReader(inifile);
  if (reader.ParseError() < 0) {
    std::cout << "[ERROR] Can't load '" << inifile << "'\n";
    return nullptr;
  }

  const std::string model_name = reader.Get("ft_instance_hyperparameter", "model_name");
  const int is_half = reader.GetInteger("ft_instance_hyperparameter", "is_half");
  if (is_half)
    return std::make_shared<GptModel<fastertransformer::OperationType::FP16>>
                 (reader.GetInteger("ft_instance_hyperparameter", "max_batch_size"),
                  reader.GetInteger("ft_instance_hyperparameter", "candidate_num"),
                  reader.GetInteger(model_name, "head_num"),
                  reader.GetInteger(model_name, "size_per_head"),
                  reader.GetInteger(model_name, "vocab_size"),
                  reader.GetInteger("ft_instance_hyperparameter", "max_seq_len"),
                  reader.GetInteger(model_name, "decoder_layers"),
                  reader.GetInteger("ft_instance_hyperparameter", "tensor_para_size"),
                  reader.GetInteger("ft_instance_hyperparameter", "layer_para_size"),
                  reader.GetInteger("ft_instance_hyperparameter", "layer_para_batch_size"),
                  reader.GetFloat("ft_instance_hyperparameter", "probability_threshold"),
                  reader.GetInteger("ft_instance_hyperparameter", "is_fuse_QKV"),
		  reader.GetFloat("ft_instance_hyperparameter", "temperature"),
                  reader.GetFloat("ft_instance_hyperparameter", "repetition_penalty"),
                  reader.Get("ft_instance_hyperparameter", "model_name"),
                  reader.Get("ft_instance_hyperparameter", "model_path_prefix"));
  else
    return std::make_shared<GptModel<fastertransformer::OperationType::FP32>>
                 (reader.GetInteger("ft_instance_hyperparameter", "max_batch_size"),
                  reader.GetInteger("ft_instance_hyperparameter", "candidate_num"),
                  reader.GetInteger(model_name, "head_num"),
                  reader.GetInteger(model_name, "size_per_head"),
                  reader.GetInteger(model_name, "vocab_size"),
                  reader.GetInteger("ft_instance_hyperparameter", "max_seq_len"),
                  reader.GetInteger(model_name, "decoder_layers"),
                  reader.GetInteger("ft_instance_hyperparameter", "tensor_para_size"),
                  reader.GetInteger("ft_instance_hyperparameter", "layer_para_size"),
                  reader.GetInteger("ft_instance_hyperparameter", "layer_para_batch_size"),
                  reader.GetFloat("ft_instance_hyperparameter", "probability_threshold"),
                  reader.GetInteger("ft_instance_hyperparameter", "is_fuse_QKV"),
		  reader.GetFloat("ft_instance_hyperparameter", "temperature"),
                  reader.GetFloat("ft_instance_hyperparameter", "repetition_penalty"),
                  reader.Get("ft_instance_hyperparameter", "model_name"),
                  reader.Get("ft_instance_hyperparameter", "model_path_prefix"));
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

int read_start_ids(int batch_size, std::vector<int>*v_start_lengths, std::vector<u_int32_t>*v_start_ids,
                   int& max_start_len, const int end_id, std::string start_id_filename)
{
  std::vector<std::vector<u_int32_t>> tmp_start_ids;

  std::ifstream start_id_file(start_id_filename.c_str(), std::ios::in);
  if (start_id_file.is_open())
  {
    std::string line;
    int i0 = 0;
    while (std::getline(start_id_file, line))
    {
      std::stringstream lineStream(line);
      std::string vals;
      int i1 = 0;
      std::vector<u_int32_t> tmp_vec;
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
    printf("[ERROR] Cannot open the file '%s'. \n", start_id_filename.c_str());
    exit(-1);
  }

  for(uint i = 1; i < (uint)v_start_lengths->size(); i++)
  {
    max_start_len = max_start_len > v_start_lengths->data()[i] ? max_start_len : v_start_lengths->data()[i];
  }

  while((int)v_start_lengths->size() < batch_size)
  {
    std::vector<u_int32_t> padding_ids;
    for(int i = 0; i < max_start_len; i++)
      padding_ids.push_back(50256);
    tmp_start_ids.push_back(padding_ids);
    v_start_lengths->push_back(max_start_len);
  }

  // Add padding
  for(int i = 0; i < (int)tmp_start_ids.size(); i++)
  {
    for(int j = (int)tmp_start_ids[i].size(); j < max_start_len; j++)
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

//template <fastertransformer::OperationType OpType>
//std::shared_ptr<Request> prepareRequest(std::shared_ptr<GptModel<OpType>> model)
std::shared_ptr<std::vector<Tensor>> prepareRequest(std::string request_config_filename, std::string start_id_filename)
{
  INIReader reader = INIReader(request_config_filename);
  if (reader.ParseError() < 0) {
    std::cout << "[ERROR] Can't load '" << request_config_filename << "'\n";
    return nullptr;
  }

  const int batch_size = reader.GetInteger("request", "request_batch_size");
  const int end_id = 50256;

  cudaStream_t stream;
  check_cuda_error(cudaStreamCreate(&stream));

  std::vector<int> v_start_lengths;
  std::vector<u_int32_t> v_start_ids;
  int max_start_len = -1;
  read_start_ids(batch_size, &v_start_lengths, &v_start_ids,
                 max_start_len, end_id, start_id_filename);

  int *start_lengths = new int[batch_size];
  int *start_ids = new int[batch_size * max_start_len];
  int *output_len = new int[batch_size];
  for(int i = 0; i < batch_size; i++) start_lengths[i] = v_start_lengths[i];
  for(int i = 0; i < batch_size * max_start_len; i++) start_ids[i] = v_start_ids[i];
  for(int i = 0; i < batch_size; i++) output_len[i] = 0;

  printf("[INFO] request data at host is prepared\n");

  return std::shared_ptr<std::vector<Tensor>>(new std::vector<Tensor>{
      Tensor{MEMORY_CPU, TYPE_UINT32, std::vector<int64_t>{batch_size, max_start_len}, (void *) start_ids},
      Tensor{MEMORY_CPU, TYPE_UINT32, std::vector<int64_t>{batch_size}, (void*)start_lengths},
      Tensor{MEMORY_CPU, TYPE_UINT32, std::vector<int64_t>{batch_size}, (void*)output_len}});
}

//template <fastertransformer::OperationType OpType>
//std::pair<int*, GptModelInstance<OpType>::DataType*> GptModelInstance<OpType>::prepareRequestAttentionMask2(std::shared_ptr<Request> request) {}

template <fastertransformer::OperationType OpType>
std::tuple<int*, int*, void*> GptModelInstance<OpType>::prepareRequestAttentionMask(std::shared_ptr<std::vector<Tensor>> input_tensors, const int input_len)
{
  auto shape = (*input_tensors)[0].shape;
  auto start_ids = (*input_tensors)[0].data;
  auto start_lengths = (*input_tensors)[1].data;
  assert(shape.size() == 2);
  auto batch_size = shape[0], max_start_len = shape[1];
  const int start_len = input_len;

  //cudaStream_t stream;
  //check_cuda_error(cudaStreamCreate(&stream));

  int* d_start_ids;
  check_cuda_error(cudaMalloc((void **)&d_start_ids, sizeof(int) * batch_size * max_start_len));
  check_cuda_error(cudaMemcpyAsync(d_start_ids, start_ids, sizeof(int) * batch_size * max_start_len, cudaMemcpyHostToDevice, stream));

  int* d_start_lengths;
  check_cuda_error(cudaMalloc((void **)&d_start_lengths, sizeof(int) * batch_size));
  check_cuda_error(cudaMemcpyAsync(d_start_lengths, start_lengths, sizeof(int) * batch_size, cudaMemcpyHostToDevice, stream));

  DataType* h_attn_mask = new DataType[batch_size * start_len * start_len];
  memset(h_attn_mask, 0, sizeof(DataType) * batch_size * start_len * start_len);
  for(int i = 0; i < batch_size; i++)
  {
    for(int j = 0; j < start_len; j++)
    {
      for(int k = 0; k <= j; k++)
      {
        h_attn_mask[i * start_len * start_len + j * start_len + k] = (DataType)1.0f;
      }
    }
  }
  DataType* d_attn_mask;
  check_cuda_error(cudaMalloc((void **)&d_attn_mask, sizeof(DataType) * batch_size * start_len * start_len));
  check_cuda_error(cudaMemcpyAsync(d_attn_mask, h_attn_mask, sizeof(DataType) * batch_size * start_len * start_len, cudaMemcpyHostToDevice, stream));

  // cudaDeviceSynchronize();
  // check_cuda_error(cudaGetLastError());

  return std::make_tuple(d_start_ids, d_start_lengths, (void *) d_attn_mask);
}

template <fastertransformer::OperationType OpType>
std::unique_ptr<AbstractTransformerModelInstance> GptModel<OpType>::createModelInstance (int node_id, int device_id, int world_size, cudaStream_t stream)
{
  const int global_head_num = head_num;

  std::unique_ptr<fastertransformer::Allocator<AllocatorType::CUDA>>
    allocator(new fastertransformer::Allocator<AllocatorType::CUDA>(device_id));

  const int start_id = 50256; // In fact, there is no start id in GPT model.
  const int end_id = 50256;

  auto decoding = std::make_unique<DecodingGpt<OpType>>((*allocator.get()), batch_size,
                                                          max_seq_len, global_head_num, size_per_head,
                                                          vocab_size, decoder_layers,
                                                          start_id, end_id,
                                                          candidate_num, probability_threshold,
                                                          temperature, tensor_para_size, layer_para_size, is_fuse_QKV, repetition_penalty);

  return std::unique_ptr<GptModelInstance<OpType>>
    (new GptModelInstance<OpType>(
      stream,
      std::move(allocator),
      std::move(decoding),
      batch_size,
      max_seq_len));
}

template <fastertransformer::OperationType OpType>
std::unique_ptr<AbstractParamInstance> GptModel<OpType>::createParamInstance (int node_id, int device_id,
                                                                              int world_size, cudaStream_t stream,
                                                                              std::vector<ncclUniqueId> nccl_ids)
{
  uint64_t rank = (uint64_t)(node_id * 8 + device_id);

  return std::unique_ptr<GptParamInstance<OpType>>
    (new GptParamInstance<OpType>(
      batch_size,
      head_num,
      size_per_head,
      tensor_para_size,
      layer_para_size,
      world_size,
      rank,
      decoder_layers,
      vocab_size,
      max_seq_len,
      layer_para_batch_size,
      model_path_prefix,
      stream,
      nccl_ids));
}

void check_inputs(std::shared_ptr<std::vector<Tensor>> output_tensors, const char* filename)
{
  auto& output = output_tensors->at(0);
  auto shape = output.shape;
  assert(shape.size() == 2);
  assert(output.type == TYPE_UINT32);
  auto batch_size = shape[0];
  auto length = shape[1];
  std::string fName = filename;
  auto file = std::ofstream(fName, std::ios::out);
  if(!file.is_open())  {
  } else {
    for(int i=0; i<batch_size; i++) {
      for(int j=0; j<length; j++) {
        file << ((uint32_t*)output.data)[i*batch_size + j] << " ";
      }
      file << std::endl;
    }
  }
}
void check_outputs(std::shared_ptr<std::vector<Tensor>> output_tensors, const char* filename)
{
  auto& output = output_tensors->at(0);
  auto shape = output.shape;
  assert(shape.size() == 2);
  assert(output.type == TYPE_UINT32);
  auto batch_size = shape[0];
  auto max_seq_len = shape[1];
  if(true)
  {
    std::string fName = filename;
    auto outFile = std::ofstream(fName, std::ios::out);
    if(!outFile.is_open())
    {
      printf("[WARNING] Cannot write results into output file %s \n", fName.c_str());
    }
    else
    {
      size_t outCount = max_seq_len * batch_size;
      int *hBuf = new int[outCount];
      cudaDeviceSynchronize();
      cudaMemcpy(hBuf, output.data, outCount * sizeof(int), cudaMemcpyDeviceToHost);

      {
        std::cout << "Writing " << outCount << " elements\n";
        int zeroCount = 0;
        for (size_t i = 0; i < outCount; i++)
        {
            if (hBuf[i] == int(0)) zeroCount++;
            outFile << hBuf[i] << " ";
            if((i+1) % (batch_size) == 0) outFile << std::endl;

            if( i < 10) printf("%5d ", hBuf[i]);
            if((i+1) % (batch_size) == 0 && i < 10) std::cout << std::endl;
        }
        std::cout << std::endl << "zeroCount = " << zeroCount << std::endl;
      }
      delete [] hBuf;
    }
  }
}

template <fastertransformer::OperationType OpType>
int GptParamInstance<OpType>::init_device_from_bin(DataType **ptr, std::vector<size_t> shape, std::string filename, int split)
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

  check_cuda_error(cudaMalloc((void **)ptr, sizeof(DataType) * size));
  if(std::is_same<DataType, float>::value == true)
    cudaMemcpy(*ptr, host_array.data(), sizeof(DataType) * size, cudaMemcpyHostToDevice);
  else
  {
    std::vector<DataType> host_array_2(size);
    for(size_t i = 0; i < size; i++)
    {
      host_array_2[i] = __float2half(host_array[i]);
    }
    cudaMemcpy(*ptr, host_array_2.data(), sizeof(DataType) * size, cudaMemcpyHostToDevice);
  }
  return 0;
}

template <fastertransformer::OperationType OpType>
int GptParamInstance<OpType>::init_device_from_csv(DataType **ptr, std::vector<size_t> shape, std::string filename, int split)
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
  std::vector<DataType> host_array(size);

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
          if (sizeof(DataType) == sizeof(float))
            host_array[i0 * dim1_each + (idx * size_each) + i11] = std::stof(vals);
          else
            host_array[i0 * dim1_each + (idx * size_each) + i11] = __float2half(std::stof(vals));
        }
        else
        {
          if (sizeof(DataType) == sizeof(float))
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
  check_cuda_error(cudaMalloc((void **)ptr, sizeof(DataType) * size));
  cudaMemcpy(*ptr, host_array.data(), sizeof(DataType) * size, cudaMemcpyHostToDevice);
  if (dim0_reached)
    printf("[WARNING] the file dimension does not match with input dim0! %s, dim0=%d, i0=%d\n", filename.c_str(), dim0, i0);
  if (dim1_reached)
    printf("[WARNING] the file dimension does not match with input dim1! %s, dim1=%d, i1=%d\n", filename.c_str(), dim1, i1);

  return 0;
}

template <fastertransformer::OperationType OpType>
int GptParamInstance<OpType>::init_device_from_file(DataType **ptr, std::vector<size_t> shape, std::string filename, int split, std::string type)
{
  std::cout << "[INFO] load ckpt from " << filename << "                                  \r" << std::flush;
  if(type == "bin")
  {
    return init_device_from_bin(ptr, shape, filename, split);
  }
  else if(type == "csv")
  {
    return init_device_from_csv(ptr, shape, filename, split);
  }
  else
  {
    printf("[ERROR] not support type: %s \n", type.c_str());
    exit(-1);
  }
}

template <fastertransformer::OperationType OpType>
void GptParamInstance<OpType>::setup_parallel_param(std::vector<ncclUniqueId> nccl_ids)
{
  uint64_t rank = rank_;

  int device, device_count;
  uint64_t tensor_para_size = tensor_para_size_;

  assert(head_num_ % tensor_para_size_ == 0);
  if (rank==0) printf("Total ranks: %ld.\n", world_size_);
  CUDACHECK(cudaGetDeviceCount(&device_count));
  CUDACHECK(cudaSetDevice(rank % device_count));
  CUDACHECK(cudaGetDevice(&device));

  printf("P%ld is runing with %d GPU.\n", rank, device);

  const size_t layer_para_size = layer_para_size_;
  if(tensor_para_size * layer_para_size != world_size_)
  {
    printf("[ERROR] tensor_para_size * layer_para_size should equal to world_size \n");
    exit(-1);
  }

  const size_t tensor_para_rank = rank % tensor_para_size;
  const size_t layer_para_rank = rank / tensor_para_size;
  const size_t layers_per_group = decoder_layers_ / layer_para_size;
  if(layers_per_group * layer_para_size != decoder_layers_)
  {
    printf("[ERROR] layers_per_group (%ld) * layer_para_size (%ld) should equal to decoder_layers_ (%ld) \n", layers_per_group, layer_para_size, decoder_layers_);
    exit(-1);
  }

  ncclUniqueId tensor_para_nccl_uid = nccl_ids[rank / tensor_para_size];
  ncclUniqueId layer_para_nccl_uid = nccl_ids[world_size_ / tensor_para_size + rank % tensor_para_size];
  /*
  // assume gpu_num = n * k,
  // tensor parallelism group size is n
  // layer parallelism group size is k
  if(tensor_para_rank == 0){
    // get the uid of each tensor parallelism group
    // here, 0, 1, ..., n-1 are in group 0,
    //       n, ..., 2n - 1 are in group 1.
    NCCLCHECK( ncclGetUniqueId(&tensor_para_nccl_uid));
    for(size_t i = 1; i < tensor_para_size; i++)
      {
        printf("[INFO] rank %ld sends tensor_para_nccl_uid to rank %ld \n", rank, rank + i);
        //MPICHECK( MPI_Send(&tensor_para_nccl_uid, sizeof(tensor_para_nccl_uid), MPI_BYTE, rank + i, 0, MPI_COMM_WORLD));
      }
  }
  else {
    //MPI_Status status;
    printf("[INFO] rank %ld receives tensor_para_nccl_uid from rank %ld \n", rank, rank - tensor_para_rank);
    //MPICHECK( MPI_Recv(&tensor_para_nccl_uid, sizeof(tensor_para_nccl_uid), MPI_BYTE, rank - tensor_para_rank, 0, MPI_COMM_WORLD, &status));
  }

  if(layer_para_rank == 0)	{
    // get the uid of each layer parallelism group
    // 0, k, 2k, are in group 0
    // 1, k+1, 2k+1 are in group 1
    NCCLCHECK( ncclGetUniqueId(&layer_para_nccl_uid));
    for(size_t i = 1; i < layer_para_size; i++)
    {
      printf("[INFO] rank %ld sends layer_para_nccl_uid to rank %ld \n", rank, rank + i * tensor_para_size);
      //MPICHECK( MPI_Send(&layer_para_nccl_uid, sizeof(layer_para_nccl_uid), MPI_BYTE, rank + i * tensor_para_size, 0, MPI_COMM_WORLD));
    }
  }
  else  {
    //MPI_Status status;
    printf("[INFO] rank %ld receives layer_para_nccl_uid from rank %ld \n", rank, rank % tensor_para_size);
    //MPICHECK( MPI_Recv(&layer_para_nccl_uid, sizeof(layer_para_nccl_uid), MPI_BYTE, rank % tensor_para_size, 0, MPI_COMM_WORLD, &status));
  }
  */

  ncclComm_t tensor_para_nccl_comm, layer_para_nccl_comm;
  NCCLCHECK( ncclCommInitRank(&tensor_para_nccl_comm, tensor_para_size, tensor_para_nccl_uid, tensor_para_rank));
  NCCLCHECK( ncclCommInitRank(&layer_para_nccl_comm, layer_para_size, layer_para_nccl_uid, layer_para_rank));

  if(head_num_ % tensor_para_size != 0)
  {
    printf("[ERROR] head_num mod tensor_para_size should be 0. Here, head_num is %ld, and tensor_para_size is %ld. \n", head_num_, tensor_para_size);
    exit(-1);
  }

  const int local_head_num = head_num_ / tensor_para_size;
  const int local_hidden_units = local_head_num * size_per_head_;

  // TensorParallelParam tensor_parallel_param;
  tensor_parallel_params.rank = tensor_para_rank;
  tensor_parallel_params.world_size = tensor_para_size;
  tensor_parallel_params.nccl_comm = tensor_para_nccl_comm;
  tensor_parallel_params.local_head_num_ = local_head_num;
  tensor_parallel_params.local_hidden_units_ = local_hidden_units;

  // LayerParallelParam layer_parallel_param;
  layer_parallel_params.rank = layer_para_rank;
  layer_parallel_params.world_size = layer_para_size;
  layer_parallel_params.nccl_comm = layer_para_nccl_comm;
  layer_parallel_params.layers_per_group = layers_per_group;
  layer_parallel_params.local_batch_size = layer_para_batch_size_;
}
template <fastertransformer::OperationType OpType>
void GptParamInstance<OpType>::setup_parallel_param_ranks()
{
  uint64_t rank = rank_;

  int device, device_count;
  uint64_t tensor_para_size = tensor_para_size_;

  assert(head_num_ % tensor_para_size_ == 0);
  if (rank==0) printf("Total ranks: %ld.\n", world_size_);
  CUDACHECK(cudaGetDeviceCount(&device_count));
  CUDACHECK(cudaSetDevice(rank % device_count));
  CUDACHECK(cudaGetDevice(&device));

  printf("P%ld is runing with %d GPU.\n", rank, device);

  const size_t layer_para_size = layer_para_size_;
  if(tensor_para_size * layer_para_size != world_size_)
  {
    printf("[ERROR] tensor_para_size * layer_para_size should equal to world_size \n");
    exit(-1);
  }

  const size_t tensor_para_rank = rank % tensor_para_size;
  const size_t layer_para_rank = rank / tensor_para_size;
  const size_t layers_per_group = decoder_layers_ / layer_para_size;
  if(layers_per_group * layer_para_size != decoder_layers_)
  {
    printf("[ERROR] layers_per_group (%ld) * layer_para_size (%ld) should equal to decoder_layers_ (%ld) \n", layers_per_group, layer_para_size, decoder_layers_);
    exit(-1);
  }

  if(head_num_ % tensor_para_size != 0)
  {
    printf("[ERROR] head_num mod tensor_para_size should be 0. Here, head_num is %ld, and tensor_para_size is %ld. \n", head_num_, tensor_para_size);
    exit(-1);
  }

  const int local_head_num = head_num_ / tensor_para_size;
  const int local_hidden_units = local_head_num * size_per_head_;

  // TensorParallelParam tensor_parallel_param;
  tensor_parallel_params.rank = tensor_para_rank;
  tensor_parallel_params.world_size = tensor_para_size;
  //tensor_parallel_params.nccl_comm = tensor_para_nccl_comm;
  tensor_parallel_params.local_head_num_ = local_head_num;
  tensor_parallel_params.local_hidden_units_ = local_hidden_units;

  // LayerParallelParam layer_parallel_param;
  layer_parallel_params.rank = layer_para_rank;
  layer_parallel_params.world_size = layer_para_size;
  //layer_parallel_params.nccl_comm = layer_para_nccl_comm;
  layer_parallel_params.layers_per_group = layers_per_group;
  layer_parallel_params.local_batch_size = layer_para_batch_size_;
}
template <fastertransformer::OperationType OpType>
void GptParamInstance<OpType>::setup_parallel_param_nccls(std::vector<ncclUniqueId> nccl_ids)
{
  uint64_t rank = rank_;

  int device, device_count;
  uint64_t tensor_para_size = tensor_para_size_;

  assert(head_num_ % tensor_para_size_ == 0);
  if (rank==0) printf("Total ranks: %ld.\n", world_size_);
  CUDACHECK(cudaGetDeviceCount(&device_count));
  CUDACHECK(cudaSetDevice(rank % device_count));
  CUDACHECK(cudaGetDevice(&device));

  printf("P%ld is runing with %d GPU.\n", rank, device);

  const size_t layer_para_size = layer_para_size_;
  if(tensor_para_size * layer_para_size != world_size_)
  {
    printf("[ERROR] tensor_para_size * layer_para_size should equal to world_size \n");
    exit(-1);
  }

  const size_t tensor_para_rank = rank % tensor_para_size;
  const size_t layer_para_rank = rank / tensor_para_size;
  const size_t layers_per_group = decoder_layers_ / layer_para_size;
  if(layers_per_group * layer_para_size != decoder_layers_)
  {
    printf("[ERROR] layers_per_group (%ld) * layer_para_size (%ld) should equal to decoder_layers_ (%ld) \n", layers_per_group, layer_para_size, decoder_layers_);
    exit(-1);
  }

  printf("rank = %ld, tensor size = %ld, layer size = %ld, world size = %ld\n", rank, tensor_para_size, layer_para_size, world_size_);

  ncclUniqueId tensor_para_nccl_uid = nccl_ids[rank / tensor_para_size];
  ncclUniqueId layer_para_nccl_uid = nccl_ids[world_size_ / tensor_para_size + rank % tensor_para_size];

  ncclComm_t tensor_para_nccl_comm, layer_para_nccl_comm;
  NCCLCHECK( ncclCommInitRank(&tensor_para_nccl_comm, tensor_para_size, tensor_para_nccl_uid, tensor_para_rank));
  NCCLCHECK( ncclCommInitRank(&layer_para_nccl_comm, layer_para_size, layer_para_nccl_uid, layer_para_rank));

  if(head_num_ % tensor_para_size != 0)
  {
    printf("[ERROR] head_num mod tensor_para_size should be 0. Here, head_num is %ld, and tensor_para_size is %ld. \n", head_num_, tensor_para_size);
    exit(-1);
  }

  const int local_head_num = head_num_ / tensor_para_size;
  const int local_hidden_units = local_head_num * size_per_head_;

  // TensorParallelParam tensor_parallel_param;
  // tensor_parallel_params.rank = tensor_para_rank;
  // tensor_parallel_params.world_size = tensor_para_size;
  tensor_parallel_params.nccl_comm = tensor_para_nccl_comm;
  // tensor_parallel_params.local_head_num_ = local_head_num;
  // tensor_parallel_params.local_hidden_units_ = local_hidden_units;
  //
  // // LayerParallelParam layer_parallel_param;
  // layer_parallel_params.rank = layer_para_rank;
  // layer_parallel_params.world_size = layer_para_size;
  layer_parallel_params.nccl_comm = layer_para_nccl_comm;
  // layer_parallel_params.layers_per_group = layers_per_group;
  // layer_parallel_params.local_batch_size = layer_para_batch_size_;
}

template <fastertransformer::OperationType OpType>
void GptParamInstance<OpType>::load_gpt_model_param()
{
  decoder_params = std::unique_ptr<DecoderInitParam<DataType>[]>(new DecoderInitParam<DataType>[decoder_layers_]);

  check_cuda_error(cublasCreate(&cublasHandle));
  check_cuda_error(cublasSetStream(cublasHandle, stream));

  uint64_t tensor_para_size = tensor_parallel_params.world_size;
  uint64_t tensor_para_rank = tensor_parallel_params.rank;
  uint64_t global_hidden_units = head_num_ * size_per_head_;
  uint64_t local_hidden_units = global_hidden_units / tensor_para_size;
  uint64_t local_inner_size = local_hidden_units * 4;

  for (uint i = 0; i < (uint)decoder_layers_; i++)
  {
    if(layer_parallel_params.is_valid(i) == false) continue;
    decoder_params[i].stream = stream;
    decoder_params[i].cublas_handle = cublasHandle;

    DataType *d_self_Q_kernel, *d_self_K_kernel, *d_self_V_kernel, *d_self_output_kernel;
    DataType *d_self_bias;
    DataType *d_self_Q_bias, *d_self_K_bias, *d_self_V_bias, *d_self_output_bias;
    DataType *d_ffn_kernel1, *d_ffn_bias1, *d_ffn_kernel2, *d_ffn_bias2;
    DataType *d_self_gamma, *d_self_beta;
    DataType *d_ffn_gamma, *d_ffn_beta;

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


    decoder_params[i].self_layernorm.gamma = d_self_gamma;
    decoder_params[i].self_layernorm.beta = d_self_beta;
    decoder_params[i].self_attention.query_weight.kernel = d_self_Q_kernel;
    decoder_params[i].self_attention.key_weight.kernel = d_self_K_kernel;
    decoder_params[i].self_attention.value_weight.kernel = d_self_V_kernel;
    decoder_params[i].self_attention.attention_output_weight.kernel = d_self_output_kernel;
    decoder_params[i].self_attention.query_weight.bias = d_self_Q_bias;
    decoder_params[i].self_attention.key_weight.bias = d_self_K_bias;
    decoder_params[i].self_attention.value_weight.bias = d_self_V_bias;
    decoder_params[i].self_attention.attention_output_weight.bias = d_self_output_bias;

    decoder_params[i].ffn_layernorm.gamma = d_ffn_gamma;
    decoder_params[i].ffn_layernorm.beta = d_ffn_beta;
    decoder_params[i].ffn.intermediate_weight.bias = d_ffn_bias1;
    decoder_params[i].ffn.output_weight.bias = d_ffn_bias2;
    decoder_params[i].ffn.intermediate_weight.kernel = d_ffn_kernel1;
    decoder_params[i].ffn.output_weight.kernel = d_ffn_kernel2;
  }

  DataType *d_embedding_table;
  DataType *d_position_encoding_table;
  DataType *d_embedding_kernel;
  int *d_output_ids;
  DataType *d_gamma, *d_beta;

  init_device_from_file(&d_embedding_table, {vocab_size_, global_hidden_units}, path_to_weights("wte.bin", -1, tensor_para_size));
  init_device_from_file(&d_position_encoding_table, {max_seq_len_, global_hidden_units}, path_to_weights("wpe.bin", -1, tensor_para_size));
  d_embedding_kernel = d_embedding_table;

  check_cuda_error(cudaMalloc((void **)&d_output_ids, sizeof(int) * max_seq_len_ * batch_size_));
  init_device_from_file(&d_gamma, {global_hidden_units}, path_to_weights("final_layernorm.weight.bin", -1, tensor_para_size));
  init_device_from_file(&d_beta, {global_hidden_units}, path_to_weights("final_layernorm.bias.bin", -1, tensor_para_size));

  decoding_params.cublas_handle = cublasHandle;
  decoding_params.stream = stream;
  decoding_params.embedding_table = d_embedding_table;
  decoding_params.position_encoding_table = d_position_encoding_table;
  decoding_params.embedding_kernel = d_embedding_kernel;
  decoding_params.output_ids = d_output_ids;
  decoding_params.layernorm.gamma = d_gamma;
  decoding_params.layernorm.beta = d_beta;
}

template <fastertransformer::OperationType OpType>
void GptParamInstance<OpType>::free_model_param()
{
  check_cuda_error(cublasDestroy(cublasHandle));

  for (uint i = 0; i < (uint)decoder_layers_; i++)
  {
    if(layer_parallel_params.is_valid(i) == false) continue;

    free_param(&decoder_params[i].self_layernorm.gamma);
    free_param(&decoder_params[i].self_layernorm.beta);
    free_param(&decoder_params[i].self_attention.query_weight.kernel);
    free_param(&decoder_params[i].self_attention.key_weight.kernel);
    free_param(&decoder_params[i].self_attention.value_weight.kernel);
    free_param(&decoder_params[i].self_attention.attention_output_weight.kernel);
    free_param(&decoder_params[i].self_attention.query_weight.bias);
    free_param(&decoder_params[i].self_attention.key_weight.bias);
    free_param(&decoder_params[i].self_attention.value_weight.bias);
    free_param(&decoder_params[i].self_attention.attention_output_weight.bias);

    free_param(&decoder_params[i].ffn_layernorm.gamma);
    free_param(&decoder_params[i].ffn_layernorm.beta);
    free_param(&decoder_params[i].ffn.intermediate_weight.bias);
    free_param(&decoder_params[i].ffn.output_weight.bias);
    free_param(&decoder_params[i].ffn.intermediate_weight.kernel);
    free_param(&decoder_params[i].ffn.output_weight.kernel);
  }

  free_param(&decoding_params.embedding_table);
  free_param(&decoding_params.position_encoding_table);
  free_param(&decoding_params.embedding_kernel);
  free_param(&decoding_params.output_ids);
  free_param(&decoding_params.layernorm.gamma);
  free_param(&decoding_params.layernorm.beta);

  ncclCommDestroy(tensor_parallel_params.nccl_comm);
  ncclCommDestroy(layer_parallel_params.nccl_comm);
}
