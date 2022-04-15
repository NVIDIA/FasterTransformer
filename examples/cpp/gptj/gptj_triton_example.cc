/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
#include "src/fastertransformer/triton_backend/gptj/GptJTritonModel.h"
#include "src/fastertransformer/triton_backend/gptj/GptJTritonModelInstance.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"
#include "src/fastertransformer/utils/mpi_utils.h"
#include "src/fastertransformer/utils/word_list.h"

#include <memory>
#include <thread>

namespace ft = fastertransformer;

struct RequestParam {
    int beam_width;
    int request_output_len;
    float beam_search_diversity_rate;
    int runtime_top_k;
    float runtime_top_p;
    float temperature;
    float len_penalty;
    float repetition_penalty;
    unsigned long long int random_seed;
    int start_id;
    int end_id;
};

std::vector<std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>>
broadCastRequest(const std::vector<int>& v_start_ids,
                 const std::vector<int>& v_start_lengths,
                 const std::vector<int>& v_bad_words,
                 const int node_id,
                 const int gpu_count,
                 const RequestParam param,
                 std::vector<void*>* pointer_record)
{
    // broadcast the request to all nodes, and copy "gpu_count" copies on different gpu
    int size_1 = v_start_ids.size();
    int size_2 = v_start_lengths.size();
    int size_bad_words = v_bad_words.size();
    MPICHECK(MPI_Bcast(&size_1, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&size_2, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&size_bad_words, 1, MPI_INT, 0, MPI_COMM_WORLD));

    std::vector<int> v_input_ids(size_1);
    std::vector<int> v_input_lengths(size_2);
    std::vector<int> v_input_bad_words(size_bad_words);

    if (node_id == 0) {
        memcpy(v_input_ids.data(), v_start_ids.data(), size_1 * sizeof(int));
        memcpy(v_input_lengths.data(), v_start_lengths.data(), size_2 * sizeof(int));
        memcpy(v_input_bad_words.data(), v_bad_words.data(), size_bad_words * sizeof(int));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int request_batch_size = size_2;
    int max_input_len = size_1 / size_2;

    MPICHECK(MPI_Bcast(v_input_ids.data(), size_1, MPI_INT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(v_input_lengths.data(), size_2, MPI_INT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(v_input_bad_words.data(), size_bad_words, MPI_INT, 0, MPI_COMM_WORLD));

    std::vector<std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>> request_list;
    for (int device_id = 0; device_id < gpu_count; device_id++) {
        ft::check_cuda_error(cudaSetDevice(device_id));

        int* d_input_ids;
        int* d_input_lengths;
        int* d_input_bad_words;

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
        ft::deviceMalloc(&d_input_bad_words, size_bad_words, false);
        ft::cudaH2Dcpy(d_input_bad_words, v_input_bad_words.data(), size_bad_words);

        int* request_output_len_ptr = new int((int)(param.request_output_len));

        int* start_ids_ptr = (int*)malloc(request_batch_size * sizeof(int));
        int* end_ids_ptr = (int*)malloc(request_batch_size * sizeof(int));
        for (int i = 0; i < request_batch_size; i++) {
            start_ids_ptr[i] = param.start_id;
            end_ids_ptr[i] = param.end_id;
        }
        pointer_record->push_back(start_ids_ptr);
        pointer_record->push_back(end_ids_ptr);

        request_list.push_back(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>(
            new std::unordered_map<std::string, triton::Tensor>{
                {"input_ids",
                 triton::Tensor{triton::MEMORY_GPU,
                                triton::TYPE_INT32,
                                std::vector<size_t>{(size_t)request_batch_size, (size_t)max_input_len},
                                d_input_ids}},
                {"input_lengths",
                 triton::Tensor{triton::MEMORY_GPU,
                                triton::TYPE_INT32,
                                std::vector<size_t>{(size_t)request_batch_size},
                                d_input_lengths}},
                {"request_output_len",
                 triton::Tensor{
                     triton::MEMORY_CPU, triton::TYPE_INT32, std::vector<size_t>{(size_t)1}, request_output_len_ptr}},
                {"bad_words_list",
                 triton::Tensor{
                     triton::MEMORY_GPU, triton::TYPE_INT32, {2, v_input_bad_words.size() / 2}, d_input_bad_words}},
                {"start_id",
                 triton::Tensor{triton::MEMORY_CPU, triton::TYPE_INT32, {(size_t)request_batch_size}, start_ids_ptr}},
                {"end_id",
                 triton::Tensor{triton::MEMORY_CPU, triton::TYPE_INT32, {(size_t)request_batch_size}, end_ids_ptr}}}));

        int* beam_width_ptr = new int(param.beam_width);
        pointer_record->push_back(beam_width_ptr);
        request_list[device_id]->insert(
            {"beam_width",
             triton::Tensor{triton::MEMORY_CPU, triton::TYPE_INT32, std::vector<size_t>{1}, beam_width_ptr}});
        if (param.beam_width > 1) {
            float* beam_search_diversity_rate_ptr = new float(param.beam_search_diversity_rate);
            pointer_record->push_back(beam_search_diversity_rate_ptr);
            request_list[device_id]->insert(
                {"beam_search_diversity_rate",
                 triton::Tensor{
                     triton::MEMORY_CPU, triton::TYPE_FP32, std::vector<size_t>{1}, beam_search_diversity_rate_ptr}});
        }
        else {
            if (param.runtime_top_p != 0.0f) {
                float* runtime_top_p_ptr = new float(param.runtime_top_p);
                pointer_record->push_back(runtime_top_p_ptr);
                request_list[device_id]->insert(
                    {"runtime_top_p",
                     triton::Tensor{triton::MEMORY_CPU, triton::TYPE_FP32, std::vector<size_t>{1}, runtime_top_p_ptr}});
            }
            if (param.runtime_top_k != 0) {
                int* runtime_top_k_ptr = new int(param.runtime_top_k);
                pointer_record->push_back(runtime_top_k_ptr);
                request_list[device_id]->insert(
                    {"runtime_top_k",
                     triton::Tensor{
                         triton::MEMORY_CPU, triton::TYPE_INT32, std::vector<size_t>{1}, runtime_top_k_ptr}});
            }
        }
        float* temperature_ptr = new float(param.temperature);
        pointer_record->push_back(temperature_ptr);
        request_list[device_id]->insert(
            {"temperature",
             triton::Tensor{triton::MEMORY_CPU, triton::TYPE_FP32, std::vector<size_t>{1}, temperature_ptr}});
        float* len_penalty_ptr = new float(param.len_penalty);
        pointer_record->push_back(len_penalty_ptr);
        request_list[device_id]->insert(
            {"len_penalty",
             triton::Tensor{triton::MEMORY_CPU, triton::TYPE_FP32, std::vector<size_t>{1}, len_penalty_ptr}});
        float* repetition_penalty_ptr = new float(param.repetition_penalty);
        pointer_record->push_back(repetition_penalty_ptr);
        request_list[device_id]->insert(
            {"repetition_penalty",
             triton::Tensor{triton::MEMORY_CPU, triton::TYPE_FP32, std::vector<size_t>{1}, repetition_penalty_ptr}});
        unsigned long long int* random_seed_ptr = new unsigned long long int(param.random_seed);
        pointer_record->push_back(random_seed_ptr);
        request_list[device_id]->insert(
            {"random_seed",
             triton::Tensor{triton::MEMORY_CPU, triton::TYPE_UINT64, std::vector<size_t>{1}, random_seed_ptr}});

        pointer_record->push_back(d_input_ids);
        pointer_record->push_back(d_input_lengths);
        pointer_record->push_back(d_input_bad_words);
        pointer_record->push_back(request_output_len_ptr);
    }

    return request_list;
}

std::vector<std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>>
prepareRequest(std::string ini_name, const int node_id, const int gpu_count, std::vector<void*>* pointer_record)
{
    INIReader reader = INIReader(ini_name);
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << ini_name << "'\n";
        ft::FT_CHECK(false);
    }

    const size_t request_batch_size = reader.GetInteger("request", "request_batch_size");

    const int start_id = reader.GetInteger("gptj_6B", "start_id");
    const int end_id = reader.GetInteger("gptj_6B", "end_id");

    std::vector<int> v_start_ids;
    std::vector<int> v_start_lengths;

    int max_input_len = 0;
    ft::read_start_ids(request_batch_size,
                       &v_start_lengths,
                       &v_start_ids,
                       max_input_len,
                       end_id,
                       1,
                       "../examples/cpp/gptj/start_ids.csv");

    std::vector<int> v_bad_words;
    ft::read_word_list("../examples/cpp/gptj/bad_words.csv", v_bad_words);

    RequestParam param;
    param.beam_width = reader.GetInteger("ft_instance_hyperparameter", "beam_width");
    param.request_output_len = reader.GetInteger("request", "request_output_len");
    param.beam_search_diversity_rate = reader.GetFloat("ft_instance_hyperparameter", "beam_search_diversity_rate");
    param.runtime_top_k = reader.GetInteger("ft_instance_hyperparameter", "top_k");
    param.runtime_top_p = reader.GetFloat("ft_instance_hyperparameter", "top_p");
    param.temperature = reader.GetFloat("ft_instance_hyperparameter", "temperature");
    param.len_penalty = reader.GetFloat("ft_instance_hyperparameter", "len_penalty");
    param.repetition_penalty = reader.GetFloat("ft_instance_hyperparameter", "repetition_penalty");
    param.random_seed = (unsigned long long int)0;
    param.start_id = start_id;
    param.end_id = end_id;

    auto request_list =
        broadCastRequest(v_start_ids, v_start_lengths, v_bad_words, node_id, gpu_count, param, pointer_record);
    return request_list;
}

int threadCreateModelInstances(std::shared_ptr<AbstractTransformerModel> model,
                               std::vector<std::unique_ptr<AbstractTransformerModelInstance>>* model_instances,
                               const int device_id,
                               const int rank,
                               std::pair<std::vector<ncclComm_t>, std::vector<ncclComm_t>> nccl_comms,
                               std::shared_ptr<ft::AbstractCustomComm> custom_all_reduce_comm = nullptr)
{
    printf("[INFO] rank = %d \n", rank);
    ft::check_cuda_error(cudaSetDevice(device_id));
    cudaStream_t stream;
    ft::check_cuda_error(cudaStreamCreate(&stream));
    auto model_instance = model->createModelInstance(device_id, rank, stream, nccl_comms, custom_all_reduce_comm);
    model_instances->at(device_id) = std::move(model_instance);
    printf("model instance %d is created \n", device_id);
    ft::print_mem_usage();
    return 0;
}

int threadForward(std::unique_ptr<AbstractTransformerModelInstance>* model_instance,
                  std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> request,
                  std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>* output_tensors,
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
    std::string ini_name = argc >= 2 ? std::string(argv[1]) : "../examples/cpp/gptj/gptj_config.ini";

    // step 1: Create model
    std::shared_ptr<AbstractTransformerModel> model = AbstractTransformerModel::createGptJModel(ini_name);
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

    // Optional Step: create custom all reduce comm
    std::vector<std::shared_ptr<ft::AbstractCustomComm>> custom_all_reduce_comms;
    model->createCustomComms(&custom_all_reduce_comms, world_size);

    // step 3: Create model instances
    std::vector<std::unique_ptr<AbstractTransformerModelInstance>> model_instances((size_t)gpu_count);
    std::vector<std::thread> threads;
    for (int device_id = 0; device_id < gpu_count; device_id++) {
        const int rank = node_id * gpu_count + device_id;
        threads.push_back(std::thread(threadCreateModelInstances,
                                      model,
                                      &model_instances,
                                      device_id,
                                      rank,
                                      nccl_comms,
                                      custom_all_reduce_comms[rank]));
    }
    for (auto& t : threads) {
        t.join();
    }

    // step 4: prepare request
    std::vector<void*> pointer_record;  // Used to prevent the pointers are release after leaving functions
    std::vector<std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>> request_list =
        prepareRequest(ini_name, node_id, gpu_count, &pointer_record);
    printf("[INFO] request is created \n");

    // step 5: Forward
    std::vector<std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>> output_tensors_lists(
        (size_t)gpu_count);
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

    const int* d_output_ids = (const int*)output_tensors_lists[0].get()->at("output_ids").data;
    const int batch_size = output_tensors_lists[0].get()->at("output_ids").shape[0];
    const int beam_width = output_tensors_lists[0].get()->at("output_ids").shape[1];
    const int seq_len = output_tensors_lists[0].get()->at("output_ids").shape[2];
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
                    if (hBuf[i] == int(0)) {
                        zeroCount++;
                    }
                    outFile << hBuf[i] << " ";
                    if ((i + 1) % (seq_len) == 0) {
                        outFile << std::endl;
                    }

                    if (i < 10) {
                        printf("%5d ", hBuf[i]);
                    }
                    if ((i + 1) % (seq_len) == 0 && i < 10) {
                        std::cout << std::endl;
                    }
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
