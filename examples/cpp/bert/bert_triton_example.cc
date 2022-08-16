/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "3rdparty/INIReader.h"
#include "src/fastertransformer/triton_backend/bert/BertTritonModel.h"
#include "src/fastertransformer/utils/mpi_utils.h"
#include "src/fastertransformer/utils/nccl_utils.h"

namespace ft = fastertransformer;

template<typename T>
std::vector<std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>>
broadcastRequest(const std::vector<T>&   h_input_hidden_state,
                 const std::vector<int>& h_input_seq_len,
                 const size_t            request_batch_size,
                 const size_t            request_seq_len,
                 const size_t            head_num,
                 const size_t            size_per_head,
                 const int               node_id,
                 const int               gpu_count,
                 std::vector<void*>*     pointer_record)
{
    std::vector<std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>> request_list;
    for (int device_id = 0; device_id < gpu_count; device_id++) {
        ft::check_cuda_error(cudaSetDevice(device_id));

        T*   d_input_hidden_state;
        int* d_input_seq_len;

        ft::deviceMalloc(&d_input_hidden_state, h_input_hidden_state.size(), false);
        ft::deviceMalloc(&d_input_seq_len, h_input_seq_len.size(), false);
        ft::cudaH2Dcpy(d_input_hidden_state, h_input_hidden_state.data(), h_input_hidden_state.size());
        ft::cudaH2Dcpy(d_input_seq_len, h_input_seq_len.data(), h_input_seq_len.size());

        request_list.push_back(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>(
            new std::unordered_map<std::string, triton::Tensor>(std::unordered_map<std::string, triton::Tensor>{
                {"input_hidden_state",
                 triton::Tensor{triton::MEMORY_GPU,
                                std::is_same<T, float>::value ? triton::TYPE_FP32 : triton::TYPE_FP16,
                                std::vector<size_t>{request_batch_size, request_seq_len, head_num * size_per_head},
                                d_input_hidden_state}},
                {"sequence_lengths",
                 triton::Tensor{triton::MEMORY_GPU,
                                triton::TYPE_INT32,
                                std::vector<size_t>{request_batch_size},
                                d_input_seq_len}}})));

        pointer_record->push_back(d_input_hidden_state);
        pointer_record->push_back(d_input_seq_len);
    }

    return request_list;
}

template<typename T>
std::vector<std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>>
prepareRequest(std::string ini_name, const int node_id, const int gpu_count, std::vector<void*>* pointer_record)
{
    INIReader reader = INIReader(ini_name);
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << ini_name << "'\n";
        ft::FT_CHECK(false);
    }

    const size_t      request_batch_size = reader.GetInteger("request", "request_batch_size");
    const size_t      request_seq_len    = reader.GetInteger("request", "request_seq_len");
    const std::string model_name         = reader.Get("ft_instance_hyperparameter", "model_name");
    const std::string model_dir          = reader.Get("ft_instance_hyperparameter", "model_dir");

    INIReader model_config_reader = INIReader(model_dir + "/config.ini");
    if (model_config_reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << model_dir << "/config.ini"
                  << "'\n";
        ft::FT_CHECK(false);
    }

    const size_t head_num      = model_config_reader.GetInteger("bert", "head_num");
    const size_t size_per_head = model_config_reader.GetInteger("bert", "size_per_head");

    std::vector<T>   h_input_hidden_state;
    std::vector<int> h_input_seq_len;
    srand(0);
    for (size_t i = 0; i < request_batch_size * request_seq_len * head_num * size_per_head; ++i) {
        T random_num = (T)((random() % 1000) / 1000.f - 0.5f);
        h_input_hidden_state.push_back(random_num);
    }
    for (uint i = 0; i < request_batch_size; i++) {
        h_input_seq_len.push_back(random() % request_seq_len);
    }

    auto request_list = broadcastRequest(h_input_hidden_state,
                                         h_input_seq_len,
                                         request_batch_size,
                                         request_seq_len,
                                         head_num,
                                         size_per_head,
                                         node_id,
                                         gpu_count,
                                         pointer_record);
    return request_list;
}

int threadCreateModelInstances(std::shared_ptr<AbstractTransformerModel>                         model,
                               std::vector<std::unique_ptr<AbstractTransformerModelInstance>>*   model_instances,
                               const int                                                         device_id,
                               const int                                                         rank,
                               std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> nccl_params,
                               std::shared_ptr<ft::AbstractCustomComm> custom_all_reduce_comm = nullptr)
{
    ft::check_cuda_error(cudaSetDevice(device_id));
    cudaStream_t stream;
    ft::check_cuda_error(cudaStreamCreate(&stream));
    model->createSharedWeights(device_id, rank);
    auto model_instance = model->createModelInstance(device_id, rank, stream, nccl_params, custom_all_reduce_comm);
    model_instances->at(device_id) = std::move(model_instance);
    FT_LOG_INFO("model instance %d is created", device_id);
    ft::print_mem_usage();
    return 0;
}

int threadForward(std::unique_ptr<AbstractTransformerModelInstance>*                model_instance,
                  std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>  request,
                  std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>* output_tensors,
                  const int                                                         device_id)
{
    ft::check_cuda_error(cudaSetDevice(device_id));
    *output_tensors = (*model_instance)->forward(request);
    return 0;
}

template<typename T>
int bert_triton_example(int argc, char* argv[])
{
    /*
        Prepare the nccl ids, node id, device id and world size
        by MPI or triton
    */

    ft::mpi::initialize(&argc, &argv);
    int node_id  = ft::mpi::getCommWorldRank();
    int node_num = ft::mpi::getCommWorldSize();

    // Note: Only supports that all nodes have same gpu count
    const int   gpu_count  = ft::getDeviceCount();
    const int   world_size = node_num * gpu_count;
    std::string ini_name   = argc >= 2 ? std::string(argv[1]) : "../examples/cpp/bert/bert_config.ini";

    // step 1: Create model
    INIReader                                 reader = INIReader(ini_name);
    std::shared_ptr<AbstractTransformerModel> model  = std::make_shared<BertTritonModel<T>>(
        reader.GetInteger("ft_instance_hyperparameter", "tensor_para_size"),
        reader.GetInteger("ft_instance_hyperparameter", "pipeline_para_size"),
        reader.GetInteger("ft_instance_hyperparameter", "enable_custom_all_reduce", 0),
        reader.Get("ft_instance_hyperparameter", "model_dir"),
        reader.GetInteger("ft_instance_hyperparameter", "int8_mode"),
        reader.GetInteger("ft_instance_hyperparameter", "is_sparse"),
        reader.GetInteger("ft_instance_hyperparameter", "is_remove_padding"));
    std::cout << model->toString();
    int tensor_para_size   = model->getTensorParaSize();
    int pipeline_para_size = model->getPipelineParaSize();
    ft::FT_CHECK_WITH_INFO(world_size == (tensor_para_size * pipeline_para_size),
                           fmtstr("World Size(%d) != Tensor Parallel Size (%d) * Pipeline Parallel Size (%d) !",
                                  world_size,
                                  tensor_para_size,
                                  pipeline_para_size));

    // step 2: Initialize the NCCL
    std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> nccl_params = model->createNcclParams(node_id);
    cudaDeviceSynchronize();

    // Optional Step: create custom all reduce comm
    std::vector<std::shared_ptr<ft::AbstractCustomComm>> custom_all_reduce_comms;
    model->createCustomComms(&custom_all_reduce_comms, world_size);

    // step 3: Create model instances
    std::vector<std::unique_ptr<AbstractTransformerModelInstance>> model_instances((size_t)gpu_count);
    std::vector<std::thread>                                       threads;

    threads.clear();

    for (int device_id = 0; device_id < gpu_count; device_id++) {
        const int rank = node_id * gpu_count + device_id;
        threads.push_back(std::thread(threadCreateModelInstances,
                                      model,
                                      &model_instances,
                                      device_id,
                                      rank,
                                      nccl_params,
                                      custom_all_reduce_comms[rank]));
    }
    for (auto& t : threads) {
        t.join();
    }

    // step 4: prepare request
    std::vector<void*> pointer_record;  // Used to prevent the pointers are release after leaving functions
    std::vector<std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>> request_list =
        prepareRequest<T>(ini_name, node_id, gpu_count, &pointer_record);
    FT_LOG_INFO("request is created");

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
    FT_LOG_INFO("forward is completed.");
    const size_t request_batch_size = output_tensors_lists[0].get()->at("output_hidden_state").shape[0];
    const size_t request_seq_len    = output_tensors_lists[0].get()->at("output_hidden_state").shape[1];
    const size_t hidden_dim         = output_tensors_lists[0].get()->at("output_hidden_state").shape[1];

    if (node_id == 0) {
        ft::print_abs_mean((T*)output_tensors_lists[0].get()->at("output_hidden_state").data,
                           request_batch_size * request_seq_len * hidden_dim,
                           (cudaStream_t)0,
                           "output_tensors_lists[0].at(\"output_hidden_state\").data");
    }

    // test time
    struct timeval start, end;
    ft::mpi::barrier();
    cudaDeviceSynchronize();
    gettimeofday(&start, NULL);

    const int ite = 20;
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
    ft::mpi::barrier();

    gettimeofday(&end, NULL);

    FT_LOG_INFO("request_batch_size %d request_seq_len %d"
                " FT-CPP-BERT-Triton-time %.2f ms",
                request_batch_size,
                request_seq_len,
                ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001) / ite);

    ft::mpi::finalize();
    return 0;
}

template int bert_triton_example<float>(int argc, char* argv[]);
template int bert_triton_example<half>(int argc, char* argv[]);

int main(int argc, char* argv[])
{
    std::string ini_name = argc >= 2 ? std::string(argv[1]) : "../examples/cpp/bert/bert_config.ini";
    INIReader   reader   = INIReader(ini_name);
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << ini_name << "'\n";
        ft::FT_CHECK(false);
    }

    const std::string data_type = reader.Get("ft_instance_hyperparameter", "data_type");
    if (data_type == "fp32") {
        bert_triton_example<float>(argc, argv);
    }
    else if (data_type == "fp16") {
        bert_triton_example<half>(argc, argv);
    }

    return 0;
}
