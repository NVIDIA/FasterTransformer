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

#include <thread>
#include "fastertransformer/triton_backend/gpt_triton_backend.hpp"

using namespace fastertransformer;

int thread_main(int argc, char* argv[],
                int node_id, int device_id, int world_size,
                std::shared_ptr<AbstractTransformerModel> model,
                std::vector<ncclUniqueId> nccl_ids,
                ncclComm_t tensor_para_nccl_comm,
                ncclComm_t layer_para_nccl_comm)
{
  CUDACHECK(cudaSetDevice(device_id));
  struct cudaDeviceProp prop;
  check_cuda_error(cudaGetDeviceProperties(&prop, device_id));
  printf("Device %s node_id = %d, device_id = %d, world_size = %d\n", prop.name, node_id, device_id, world_size);

  cudaStream_t stream;
  check_cuda_error(cudaStreamCreate(&stream));


  auto modelInstance = model->createModelInstance(node_id, device_id, world_size, stream);
  auto param_instance = model->createParamInstance(node_id, device_id, world_size, stream, nccl_ids);
  //param_instance->init_nccl_from_ids(nccl_ids);
  param_instance->init_nccl_from_comms(tensor_para_nccl_comm, layer_para_nccl_comm);
  modelInstance->set_param(param_instance.get());
  printf("model instance is created \n");

  std::string ini_name;
  if(argc >= 2)
    ini_name = std::string(argv[1]);
  else
    ini_name = "../sample/cpp/gpt_config.ini";
  auto request = prepareRequest(ini_name, "./start_ids.csv");
  if(node_id == 0 && device_id == 0)
    check_inputs(request);

  printf("request is created \n");

  size_t free_bytes, total_bytes;
  check_cuda_error(cudaMemGetInfo(&free_bytes, &total_bytes));
  float free = (float)(free_bytes) / 1024.0 / 1024.0 / 1024.0;
  float total = (float)(total_bytes) / 1024.0 / 1024.0 / 1024.0;
  printf("after allocation, free %.2f GB total %.2f GB\n", free, total);

  auto output = modelInstance->forward(request);

  if(node_id == 0 && device_id == 0)
    check_outputs(output);

  printf("Thread node id = %d, devide id = %d ends\n", node_id, device_id);
  return 0;
}

int main(int argc, char *argv[])
{
  std::string ini_name;
  if(argc >= 2)
    ini_name = std::string(argv[1]);
  else
    ini_name = "../sample/cpp/gpt_config.ini";
  int node_id = argc >= 5 ? std::stoi(argv[2]) : 0;
  int gpu_size = argc >= 5 ? std::stoi(argv[3]) : 1;
  int world_size = argc >= 5 ? std::stoi(argv[4]) : 1;

  int devices = 0;
  CUDACHECK(cudaGetDeviceCount(&devices));
  assert(gpu_size <= devices);

  auto model = AbstractTransformerModel::createGptModel(ini_name);
  int tensor_para_size = model->get_tensor_para_size();
  int layer_para_size =  model->get_layer_para_size();

  std::vector<ncclUniqueId> nccl_ids = model->create_nccl_ids(world_size);

  printf("model parameter is created \n");

  std::cout << model->to_string();

  ncclComm_t tensor_nccl_comms[gpu_size];
  ncclComm_t layer_nccl_comms[gpu_size];

  NCCLCHECK(ncclGroupStart());
  for (int gid=0; gid < gpu_size; gid++) {

    int rank = node_id * 8 + gid;
    size_t tensor_para_rank = rank % tensor_para_size;
    size_t layer_para_rank = rank / tensor_para_size;

    ncclUniqueId tensor_para_nccl_uid = nccl_ids[rank / tensor_para_size];
    ncclUniqueId layer_para_nccl_uid  = nccl_ids[world_size / tensor_para_size + rank % tensor_para_size];

    CUDACHECK(cudaSetDevice(gid));
    NCCLCHECK( ncclCommInitRank(&tensor_nccl_comms[gid], tensor_para_size, tensor_para_nccl_uid, tensor_para_rank));
    NCCLCHECK( ncclCommInitRank(&layer_nccl_comms[gid], layer_para_size, layer_para_nccl_uid, layer_para_rank));

  }
  NCCLCHECK(ncclGroupEnd());

  std::vector<std::thread> threads;
  for(int gid = 0; gid < gpu_size; gid ++) {
    threads.push_back(std::thread(thread_main, argc, argv,
                                  node_id, gid, world_size,
                                  model, nccl_ids,
                                  tensor_nccl_comms[gid],
                                  layer_nccl_comms[gid]));
  }

  for(auto & t : threads) {
    t.join();
  }

  printf("ALL THREADs END\n");
}
