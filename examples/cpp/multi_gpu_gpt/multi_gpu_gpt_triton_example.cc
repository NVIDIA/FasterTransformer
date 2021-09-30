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

#include "3rdparty/INIReader.h"
#include "examples/cpp/multi_gpu_gpt/gpt_example_utils.h"
#include "src/fastertransformer/triton_backend/multi_gpu_gpt/ParallelGptTritonModel.h"
#include "src/fastertransformer/triton_backend/multi_gpu_gpt/ParallelGptTritonModelInstance.h"
#include "src/fastertransformer/utils/mpi_utils.h"

#include <memory>
#include <thread>

namespace ft = fastertransformer;

std::vector<std::shared_ptr<std::vector<triton::Tensor>>> broadCastRequest(const std::vector<int>& v_start_ids,
                                                                           const std::vector<int>& v_start_lengths,
                                                                           const int node_id,
                                                                           const int gpu_count,
                                                                           const int beam_width,
                                                                           const int request_output_len,
                                                                           std::vector<void*>* pointer_record)
{
    // broadcast the request to all nodes, and copy "gpu_count" copies on different gpu
    int size_1 = v_start_ids.size();
    int size_2 = v_start_lengths.size();
    MPICHECK(MPI_Bcast(&size_1, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&size_2, 1, MPI_INT, 0, MPI_COMM_WORLD));

    std::vector<int> v_input_ids(size_1);
    std::vector<int> v_input_lengths(size_2);

    if (node_id == 0) {
        memcpy(v_input_ids.data(), v_start_ids.data(), size_1 * sizeof(int));
        memcpy(v_input_lengths.data(), v_start_lengths.data(), size_2 * sizeof(int));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int request_batch_size = size_2 / beam_width;
    int max_input_len = size_1 / size_2;

    MPICHECK(MPI_Bcast(v_input_ids.data(), size_1, MPI_INT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(v_input_lengths.data(), size_2, MPI_INT, 0, MPI_COMM_WORLD));

    std::vector<std::shared_ptr<std::vector<triton::Tensor>>> request_list;
    for (int device_id = 0; device_id < gpu_count; device_id++) {
        ft::check_cuda_error(cudaSetDevice(device_id));

        int* d_input_ids;
        int* d_input_lengths;

        if (max_input_len == 0) {
            // unconditional case, no input ids, so do nothing.
            d_input_ids = nullptr;
            d_input_lengths = nullptr;
            max_input_len = 0;
        }
        else {
            // conditional case.
            ft::deviceMalloc(&d_input_ids, size_1, false);
            ft::deviceMalloc(&d_input_lengths, size_2, false);
            ft::cudaH2Dcpy(d_input_ids, v_input_ids.data(), size_1);
            ft::cudaH2Dcpy(d_input_lengths, v_input_lengths.data(), size_2);
        }

        int* request_output_len_ptr = new int((int)(request_output_len));

        request_list.push_back(std::shared_ptr<std::vector<triton::Tensor>>(new std::vector<triton::Tensor>{
            triton::Tensor{triton::MEMORY_GPU,
                           triton::TYPE_INT32,
                           std::vector<size_t>{(size_t)request_batch_size, (size_t)beam_width, (size_t)max_input_len},
                           d_input_ids},
            triton::Tensor{triton::MEMORY_GPU,
                           triton::TYPE_INT32,
                           std::vector<size_t>{(size_t)request_batch_size, (size_t)beam_width},
                           d_input_lengths},
            triton::Tensor{
                triton::MEMORY_CPU, triton::TYPE_INT32, std::vector<size_t>{(size_t)1}, request_output_len_ptr}}));

        pointer_record->push_back(d_input_ids);
        pointer_record->push_back(d_input_lengths);
        pointer_record->push_back(request_output_len_ptr);
    }

    return request_list;
}

std::vector<std::shared_ptr<std::vector<triton::Tensor>>>
prepareRequest(std::string ini_name, const int node_id, const int gpu_count, std::vector<void*>* pointer_record)
{
    INIReader reader = INIReader(ini_name);
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << ini_name << "'\n";
        ft::FT_CHECK(false);
    }

    const size_t request_batch_size = reader.GetInteger("request", "request_batch_size");
    const size_t beam_width = reader.GetInteger("ft_instance_hyperparameter", "beam_width");
    const size_t request_output_len = reader.GetInteger("request", "request_output_len");

    const int end_id = 50256;

    std::vector<int> v_start_ids;
    std::vector<int> v_start_lengths;

    int max_input_len = 0;
    ft::read_start_ids(request_batch_size,
                       &v_start_lengths,
                       &v_start_ids,
                       max_input_len,
                       end_id,
                       beam_width,
                       "../examples/cpp/multi_gpu_gpt/start_ids.csv");

    auto request_list = broadCastRequest(
        v_start_ids, v_start_lengths, node_id, gpu_count, beam_width, request_output_len, pointer_record);
    return request_list;
}

int threadCreateModelInstances(std::shared_ptr<AbstractTransformerModel> model,
                               std::vector<std::unique_ptr<AbstractTransformerModelInstance>>* model_instances,
                               const int device_id,
                               const int rank,
                               std::pair<std::vector<ncclComm_t>, std::vector<ncclComm_t>> nccl_comms)
{
    printf("[INFO] rank = %d \n", rank);
    ft::check_cuda_error(cudaSetDevice(device_id));
    cudaStream_t stream;
    ft::check_cuda_error(cudaStreamCreate(&stream));
    auto model_instance = model->createModelInstance(device_id, rank, stream, nccl_comms);
    model_instances->at(device_id) = std::move(model_instance);
    printf("model instance %d is created \n", device_id);
    ft::print_mem_usage();
    return 0;
}

int threadForward(std::unique_ptr<AbstractTransformerModelInstance>* model_instance,
                  std::shared_ptr<std::vector<triton::Tensor>> request,
                  std::shared_ptr<std::vector<triton::Tensor>>* output_tensors,
                  const int device_id)
{
    ft::check_cuda_error(cudaSetDevice(device_id));
    *output_tensors = (*model_instance)->forward(request);
    return 0;
}

int main(int argc, char* argv[])
{
    /*
        Prepare the nccl ids, node id, device id and world size
        by MPI or triton
    */

    MPICHECK(MPI_Init(&argc, &argv));
    int node_id;
    int node_num;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &node_id));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &node_num));

    // Note: Only supports that all nodes have same gpu count
    const int gpu_count = ft::getDeviceCount();
    const int world_size = node_num * gpu_count;
    std::string ini_name = argc >= 2 ? std::string(argv[1]) : "../examples/cpp/multi_gpu_gpt/gpt_config.ini";

    // step 1: Create model
    std::shared_ptr<AbstractTransformerModel> model = AbstractTransformerModel::createGptModel(ini_name);
    std::cout << model->toString();

    // step 2: Initialize the NCCL
    std::vector<ncclUniqueId> nccl_ids;
    if (node_id == 0) {
        nccl_ids = model->createNcclIds(world_size);
    }
    int nccl_size = nccl_ids.size();
    MPI_Barrier(MPI_COMM_WORLD);
    MPICHECK(MPI_Bcast(&nccl_size, 1, MPI_INT, 0, MPI_COMM_WORLD));
    if (node_id != 0) {
        nccl_ids.resize(nccl_size);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (size_t i = 0; i < nccl_ids.size(); i++) {
        MPICHECK(MPI_Bcast(&nccl_ids[i], sizeof(nccl_ids[i]), MPI_BYTE, 0, MPI_COMM_WORLD));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    std::pair<std::vector<ncclComm_t>, std::vector<ncclComm_t>> nccl_comms = model->createNcclComms(nccl_ids, node_id);
    cudaDeviceSynchronize();

    // step 3: Create model instances
    std::vector<std::unique_ptr<AbstractTransformerModelInstance>> model_instances((size_t)gpu_count);
    std::vector<std::thread> threads;
    for (int device_id = 0; device_id < gpu_count; device_id++) {
        const int rank = node_id * gpu_count + device_id;
        threads.push_back(
            std::thread(threadCreateModelInstances, model, &model_instances, device_id, rank, nccl_comms));
    }
    for (auto& t : threads) {
        t.join();
    }

    // step 4: prepare request
    std::vector<void*> pointer_record;  // Used to prevent the pointers are release after leaving functions
    std::vector<std::shared_ptr<std::vector<triton::Tensor>>> request_list =
        prepareRequest(ini_name, node_id, gpu_count, &pointer_record);
    printf("[INFO] request is created \n");

    // step 5: Forward
    std::vector<std::shared_ptr<std::vector<triton::Tensor>>> output_tensors_lists((size_t)gpu_count);
    for (int i = 0; i < 2; i++) {
        threads.clear();
        for (int device_id = 0; device_id < gpu_count; device_id++) {
            threads.push_back(std::thread(threadForward,
                                          &model_instances[device_id],
                                          request_list[device_id],
                                          &output_tensors_lists[device_id],
                                          device_id));
        }
        for (auto& t : threads) {
            t.join();
        }
    }
    printf("[INFO] forward is completed. \n");

    const int* d_output_ids = (const int*)output_tensors_lists[0].get()->at(0).data;
    const int batch_size = output_tensors_lists[0].get()->at(0).shape[0];
    const int beam_width = output_tensors_lists[0].get()->at(0).shape[1];
    const int seq_len = output_tensors_lists[0].get()->at(0).shape[2]; 
    // step 6: check results
    if (node_id == 0) {

        std::string fName = "out";
        auto outFile = std::ofstream(fName, std::ios::out);
        if (!outFile.is_open()) {
            printf("[WARNING] Cannot write results into output file %s \n", fName.c_str());
        }
        else {
            size_t outCount = batch_size * beam_width * seq_len;
            int* hBuf = new int[outCount];
            ft::cudaD2Hcpy(hBuf, d_output_ids, outCount);

            {
                std::cout << "Writing " << outCount << " elements\n";
                int zeroCount = 0;
                for (size_t i = 0; i < outCount; i++) {
                    if (hBuf[i] == int(0))
                        zeroCount++;
                    outFile << hBuf[i] << " ";
                    if ((i + 1) % (seq_len) == 0)
                        outFile << std::endl;

                    if (i < 10)
                        printf("%5d ", hBuf[i]);
                    if ((i + 1) % (seq_len) == 0 && i < 10)
                        std::cout << std::endl;
                }
                std::cout << std::endl << "zeroCount = " << zeroCount << std::endl;
            }
            delete[] hBuf;
        }
    }

    // test time
    struct timeval start, end;
    MPI_Barrier(MPI_COMM_WORLD);
    cudaDeviceSynchronize();
    gettimeofday(&start, NULL);

    const int ite = 1;
    for (int i = 0; i < ite; i++) {
        threads.clear();
        for (int device_id = 0; device_id < gpu_count; device_id++) {
            threads.push_back(std::thread(threadForward,
                                          &model_instances[device_id],
                                          request_list[device_id],
                                          &output_tensors_lists[device_id],
                                          device_id));
        }
        for (auto& t : threads) {
            t.join();
        }
    }

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    gettimeofday(&end, NULL);

    printf("[INFO] batch_size %d beam_width %d seq_len %d"
           " FT-CPP-GPT-Triton-time %.2f ms\n",
           batch_size,
           beam_width,
           seq_len,
           ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001) / ite);

    MPICHECK(MPI_Finalize());
    return 0;
}
