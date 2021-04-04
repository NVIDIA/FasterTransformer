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

int main(int argc, char *argv[])
{
  /*
    Prepare the nccl ids, node id, device id and world size 
    by MPI or triton
  */ 

  int node_id = 0;
  int device_id = 0;
  int world_size = 1;

  bool use_mpi = argc >= 3 && std::stoi(argv[2]) == 1 ? true : false;

  if(use_mpi)
  {
    // run by mpi
    MPICHECK( MPI_Init(&argc, &argv));

    int rank, device_count;
    MPICHECK( MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPICHECK( MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    if (rank==0) printf("Total ranks: %d.\n", world_size);
    CUDACHECK(cudaGetDeviceCount(&device_count));
    node_id = rank / device_count;
    device_id = rank % device_count;
    CUDACHECK(cudaSetDevice(device_id));
  }
  else
  {
    // run by triton
  }


  struct cudaDeviceProp prop;
  check_cuda_error(cudaGetDeviceProperties(&prop, 0));
  printf("Device %s\n", prop.name);

  std::string ini_name;
  if(argc >= 2)
    ini_name = std::string(argv[1]);
  else
    ini_name = "../sample/cpp/gpt_config.ini";

  auto model = AbstractTransformerModel::createGptModel(ini_name);

  std::vector<ncclUniqueId> nccl_ids;
  if(use_mpi)
  {
    if(node_id == 0 && device_id == 0)
    {
      nccl_ids = model->create_nccl_ids(world_size);
    }
    int nccl_size = nccl_ids.size();
    MPI_Barrier(MPI_COMM_WORLD);
    MPICHECK(MPI_Bcast(&nccl_size, 1, MPI_INT, 0, MPI_COMM_WORLD));
    if(node_id != 0 || device_id != 0) 
      nccl_ids = std::vector<ncclUniqueId>(nccl_size);
    MPI_Barrier(MPI_COMM_WORLD);
    for(int i = 0; i < nccl_ids.size(); i++)
      MPICHECK( MPI_Bcast(&nccl_ids[i], sizeof(nccl_ids[i]), MPI_BYTE, 0, MPI_COMM_WORLD));
  }
  else
  {
    // generate nccl ids by triton and send to other
    if(node_id == 0 && device_id == 0)
    {
      nccl_ids = model->create_nccl_ids(world_size);
    }
  }

  printf("model parameter is created \n");

  std::cout << model->to_string();

  cudaStream_t stream;

  check_cuda_error(cudaStreamCreate(&stream));
  auto modelInstance = model->createModelInstance(node_id, device_id, world_size, stream);
  auto param_instance = model->createParamInstance(node_id, device_id, world_size, stream, nccl_ids);
  modelInstance->set_param(param_instance.get());
  printf("model instance is created \n");
  
  auto request = prepareRequest(ini_name);
    
  printf("request is created \n");

  print_mem_usage();

  auto output = modelInstance->forward(request);

  if(node_id == 0 && device_id == 0)
    check_inputs(request);

  output = modelInstance->forward(request);

  if(node_id == 0 && device_id == 0)
    check_outputs(output);
  
  if(use_mpi)
  {
    MPICHECK(MPI_Finalize());
  }
  return 0;
}
