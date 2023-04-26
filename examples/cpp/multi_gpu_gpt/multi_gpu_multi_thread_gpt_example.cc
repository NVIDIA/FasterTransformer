/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.h"
#include "src/fastertransformer/utils/mpi_utils.h"
#include "src/fastertransformer/utils/nccl_utils.h"
#include "src/fastertransformer/utils/nvtx_utils.h"

#include <cuda_profiler_api.h>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <tuple>
#include <vector>
#include <thread>

using namespace fastertransformer;


int createNcclParams(std::pair<std::vector<NcclParam>, std::vector<NcclParam>>& nccl_comms, std::vector<NcclUid>& nccl_ids, const int tp, const int pp, const int node_id, const int device_id_start, const bool multi_node)
{
    const int gpu_count          = getDeviceCount();
    const int tensor_para_size   = tp;
    const int pipeline_para_size = pp;
    const int local_comm_size    = tensor_para_size * pipeline_para_size;
    FT_CHECK(tensor_para_size > 0 && pipeline_para_size > 0);
    FT_CHECK(device_id_start + (int)local_comm_size <= gpu_count);
    
    if (tensor_para_size > 1 || pipeline_para_size > 1) {
        nccl_ids.resize(tensor_para_size + pipeline_para_size);
        if (node_id == 0) {
            for (uint32_t i = 0; i < nccl_ids.size(); i++) {
                ftNcclGetUniqueId(nccl_ids[i]);
            }
        }
    }

    std::vector<NcclParam>& tensor_para_params = nccl_comms.first;
    std::vector<NcclParam>& pipeline_para_params = nccl_comms.second;;

    tensor_para_params.resize(local_comm_size);
    pipeline_para_params.resize(local_comm_size);

    // Don't init comm when size == 1
    if (tensor_para_size > 1) {
        ftNcclGroupStart();
        for (int gid = device_id_start; gid < device_id_start + local_comm_size; gid++) {
            int rank               = node_id * gpu_count + gid - device_id_start;
            int tensor_para_rank   = rank % tensor_para_size;
            int pipeline_para_rank = rank / tensor_para_size;

            NcclUid tensor_para_nccl_uid = nccl_ids[pipeline_para_rank];
            check_cuda_error(cudaSetDevice(gid));
            ftNcclCommInitRank(
                tensor_para_params[gid - device_id_start], tensor_para_rank, tensor_para_size, tensor_para_nccl_uid);
        }
        ftNcclGroupEnd();
    }
    if (pipeline_para_size > 1) {
        ftNcclGroupStart();
        for (int gid = device_id_start; gid < device_id_start + local_comm_size; gid++) {
            int rank               = node_id * gpu_count + gid - device_id_start;
            int tensor_para_rank   = rank % tensor_para_size;
            int pipeline_para_rank = rank / tensor_para_size;

            NcclUid pipeline_para_nccl_uid = nccl_ids[pipeline_para_size + tensor_para_rank];
            check_cuda_error(cudaSetDevice(gid));
            ftNcclCommInitRank(pipeline_para_params[gid - device_id_start],
                                   pipeline_para_rank,
                                   pipeline_para_size,
                                   pipeline_para_nccl_uid);
        }
        ftNcclGroupEnd();
    }
    return 0;
}

template<typename T>
int threadCreateModelInstances(std::vector<std::shared_ptr<ParallelGpt<T>>>&                     model_instances,
                               std::vector<std::shared_ptr<ParallelGptWeight<T>>>&               model_weights,
                               std::vector<std::unique_ptr<Allocator<AllocatorType::CUDA>>>& v_allocator,
                                std::vector<std::unique_ptr<cublasAlgoMap>>&                      v_cublas_algo_map,
                                std::vector<std::unique_ptr<std::mutex>>&                             v_cublas_wrapper_mutex,
                                std::vector<std::unique_ptr<cublasMMWrapper>>&                  v_cublas_wrapper,
                                std::vector<std::unique_ptr<cudaDeviceProp>>&                     v_cuda_device_prop_ptr,
                               model_config_t                                                   model_config,
                               request_config_t                                                 request_config,
                               int                                                         device_id,
                               const int                                                         rank,
                               std::pair<std::vector<NcclParam>, std::vector<NcclParam>> nccl_comms)
{
    FT_LOG_INFO("rank = %d", rank);
    check_cuda_error(cudaSetDevice(device_id));
    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream));
    
    const int comms_rank = device_id % (model_config.tensor_para_size * model_config.pipeline_para_size);
    NcclParam tensor_para   = nccl_comms.first[comms_rank];
    NcclParam pipeline_para = nccl_comms.second[comms_rank];
    std::cout<<"tensor_para.world_size:"<<tensor_para.world_size_<<" tensor_para.rank:"<<tensor_para.rank_<<" pipeline_para.world_size:"<<pipeline_para.world_size_<<" pipeline_para.rank:"<<pipeline_para.rank_<<std::endl;

    model_weights[device_id] = std::make_shared< ParallelGptWeight<T> > (model_config.hidden_units,
                                                                        model_config.inter_size,
                                                                        model_config.vocab_size,
                                                                        model_config.decoder_layers,
                                                                        model_config.max_seq_len,
                                                                        tensor_para.world_size_,
                                                                        tensor_para.rank_,
                                                                        pipeline_para.world_size_,
                                                                        pipeline_para.rank_,
                                                                        0); //int8 mode:0
    model_weights[device_id]->loadModel(model_config.model_dir);

    std::unique_ptr<Allocator<AllocatorType::CUDA>> allocator(
        new Allocator<AllocatorType::CUDA>(device_id));

    allocator->setStream(stream);

    cublasHandle_t   cublas_handle;
    cublasLtHandle_t cublaslt_handle;

    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
    cublasSetStream(cublas_handle, stream);

    std::unique_ptr<cublasAlgoMap>       cublas_algo_map(new cublasAlgoMap("gemm_config.in"));
    std::unique_ptr<std::mutex>          cublas_wrapper_mutex(new std::mutex());
    std::unique_ptr<cublasMMWrapper>     cublas_wrapper(new cublasMMWrapper(
        cublas_handle, cublaslt_handle, stream, cublas_algo_map.get(), cublas_wrapper_mutex.get(), allocator.get()));

    std::unique_ptr<cudaDeviceProp> cuda_device_prop_ptr(new cudaDeviceProp);
    check_cuda_error(cudaGetDeviceProperties(cuda_device_prop_ptr.get(), device_id));

    if (std::is_same<T, half>::value) {
        cublas_wrapper->setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
    } else if (std::is_same<T, float>::value) {
        cublas_wrapper->setFP32GemmConfig();
    }


    AttentionType attention_type = getAttentionType<T>(model_config.size_per_head,
                                                       getSMVersion(),
                                                       request_config.remove_padding,  // remove_padding
                                                       0,                            // gpt supports any-seq-length fmha
                                                       model_config.int8_mode != 2,  // is_fuse
                                                       false,                        // with_relative_position_bias
                                                       true);                        // causal_mask

    model_instances[device_id] = std::make_shared< ParallelGpt<T> > ( 
                                        ParallelGpt<T>(0,  // max_batch_size, FT will adjust the buffer automatically.
                                        0,  // max_seq_len, FT will adjust the buffer automatically.
                                        0,  // max_input_len, FT will adjust the buffer automatically.
                                        0,  // beam_width, FT will adjust the buffer automatically.
                                        model_config.head_num,
                                        model_config.size_per_head,
                                        model_config.inter_size,
                                        model_config.decoder_layers,
                                        0,   // expert_num
                                        0,   // moe_k
                                        {},  // moe_layer_index
                                        model_config.vocab_size,
                                        model_config.start_id,
                                        model_config.end_id,
                                        // p/prompt tuning virtual token start id
                                        model_config.prompt_learning_start_id,
                                        model_config.prompt_learning_type,
                                        model_config.gpt_variants,
                                        0.0f,  // beam_search_diversity_rate,
                                        0,     // top_k,
                                        0.0,   // top_p,
                                        0,     // random_seed,
                                        1.0f,  // temperature,
                                        0.0f,  // len_penalty,
                                        1.0f,  // repetition_penalty,
                                        tensor_para,
                                        pipeline_para,
                                        stream,
                                        cublas_wrapper.get(),
                                        allocator.get(),
                                        false,
                                        cuda_device_prop_ptr.get(),
                                        attention_type,
                                        model_config.sparse,
                                        0,
                                        nullptr,
                                        0,
                                        request_config.shared_contexts_ratio));

    v_allocator[device_id] = std::move(allocator);
    v_cublas_algo_map[device_id] = std::move(cublas_algo_map);
    v_cublas_wrapper_mutex[device_id] = std::move(cublas_wrapper_mutex);
    v_cublas_wrapper[device_id] = std::move(cublas_wrapper);
    v_cuda_device_prop_ptr[device_id] = std::move(cuda_device_prop_ptr);


    FT_LOG_INFO("model instance %d is created", device_id);
    print_mem_usage();
    return 0;
}


int prepareRequest(std::vector<std::shared_ptr<std::unordered_map<std::string, Tensor>>>& request_list, 
                   std::vector<std::shared_ptr<std::unordered_map<std::string, Tensor>>>& output_list,
                   std::vector<void*>* pointer_record,
                   int device_id, 
                   model_config_t& model_config,
                   request_config_t& request_config,
                   std::string in_csv)
{
    check_cuda_error(cudaSetDevice(device_id));


    int*                                    d_input_ids     = nullptr;
    int*                                    d_input_lengths = nullptr;
    std::vector<int>                        v_start_lengths;
    std::vector<int>                        output_seq_len;
    std::vector<int>                        p_prompt_tuning_task_ids;

    int*                                    d_output_ids                = nullptr;
    int*                                    d_sequence_lengths          = nullptr;
    float*                                  d_output_log_probs          = nullptr;
    float*                                  d_cum_log_probs             = nullptr;
    float*                                  d_output_context_embeddings = nullptr;

    // Read ids of request from file.
    size_t     max_input_len      = 0;
    int* random_seed_ptr = new int(0);
    auto       request_batch_size = request_config.request_batch_size;
    const auto beam_width         = request_config.beam_width;
    const auto request_output_len = request_config.request_output_len;

    std::vector<int> v_start_ids;
    request_batch_size = read_start_ids(
        request_batch_size, &v_start_lengths, &v_start_ids, max_input_len, model_config.end_id, 1, in_csv);


    if (max_input_len > 0) {
        // conditional case.
        deviceMalloc(&d_input_ids, request_batch_size * max_input_len, false);
        deviceMalloc(&d_input_lengths, request_batch_size, false);

        cudaH2Dcpy(d_input_ids, v_start_ids.data(), request_batch_size * max_input_len);
        cudaH2Dcpy(d_input_lengths, v_start_lengths.data(), request_batch_size);
    }
    const size_t total_output_len = max_input_len + request_output_len;



    output_seq_len = std::vector<int>(request_batch_size, total_output_len);

     uint32_t* request_output_len_ptr = (uint32_t*)malloc(request_batch_size * sizeof(uint32_t));
    for (int i = 0; i < request_batch_size; i++) {
        request_output_len_ptr[i] = request_output_len;
    }

    request_list[device_id] = std::shared_ptr<std::unordered_map<std::string, Tensor>>(
            new std::unordered_map<std::string, Tensor>{
                {"input_ids",
                 Tensor{MEMORY_GPU,
                                TYPE_INT32,
                                std::vector<size_t>{(size_t)request_batch_size, (size_t)max_input_len},
                                d_input_ids}}});

    //request_list[device_id]->insert({"input_ids", {MEMORY_GPU, TYPE_INT32, {request_batch_size, max_input_len}, d_input_ids}});
    request_list[device_id]->insert({"input_lengths", {MEMORY_GPU, TYPE_INT32, {request_batch_size}, d_input_lengths}});
    request_list[device_id]->insert({"input_lengths_h", {MEMORY_CPU, TYPE_INT32, {request_batch_size}, v_start_lengths.data()}});
    request_list[device_id]->insert({"output_seq_len", {MEMORY_CPU, TYPE_UINT32, {request_batch_size}, request_output_len_ptr}});
    request_list[device_id]->insert({"temperature", {MEMORY_CPU, TYPE_FP32, {1}, &request_config.temperature}});
    request_list[device_id]->insert({"len_penalty", {MEMORY_CPU, TYPE_FP32, {1}, &request_config.len_penalty}});
    if (request_config.repetition_penalty != 1.0f) {
        request_list[device_id]->insert({"repetition_penalty", {MEMORY_CPU, TYPE_FP32, {1}, &request_config.repetition_penalty}});
    }
    if (request_config.presence_penalty != 0.0f) {
        request_list[device_id]->insert({"presence_penalty", {MEMORY_CPU, TYPE_FP32, {1}, &request_config.presence_penalty}});
    }
    request_list[device_id]->insert({"min_length", {MEMORY_CPU, TYPE_FP32, {1}, &request_config.min_length}});
    request_list[device_id]->insert({"random_seed", {MEMORY_CPU, TYPE_UINT64, {1}, random_seed_ptr}});

    if (request_config.top_k == 0 && request_config.top_p == 0.0f) {
        FT_CHECK(beam_width > 1);
        request_list[device_id]->insert(
            {"beam_search_diversity_rate", {MEMORY_CPU, TYPE_FP32, {1}, &request_config.beam_search_diversity_rate}});
    }
    else if (request_config.top_p != 0.0f) {
        request_list[device_id]->insert({"runtime_top_p", {MEMORY_CPU, TYPE_FP32, {1}, &request_config.top_p}});
    }
    else if (request_config.top_k != 0) {
        request_list[device_id]->insert({"runtime_top_k", {MEMORY_CPU, TYPE_UINT32, {1}, &request_config.top_k}});
    }


    deviceMalloc(&d_output_ids, request_batch_size * request_config.beam_width * total_output_len, false);
    deviceMalloc(&d_sequence_lengths, request_batch_size * request_config.beam_width, false);

    output_list[device_id] = std::shared_ptr<std::unordered_map<std::string, Tensor>>(
            new std::unordered_map<std::string, Tensor>{
                {"output_ids", {MEMORY_GPU, TYPE_INT32, {request_batch_size, beam_width, total_output_len}, d_output_ids}}}
                );
    output_list[device_id]->insert(
        {"sequence_length", {MEMORY_GPU, TYPE_INT32, {request_batch_size, beam_width}, d_sequence_lengths}});


    pointer_record->push_back(d_input_ids);
    pointer_record->push_back(d_input_lengths);
    pointer_record->push_back(d_output_ids);
    pointer_record->push_back(d_sequence_lengths);
    pointer_record->push_back(d_input_ids);
    //pointer_record->push_back(random_seed_ptr);
    //pointer_record->push_back(v_start_lengths.data());
    std::cout<<"prepare request success"<<std::endl;

    return 0;
}


template<typename T>
int prepareRequest_and_forward(std::vector<std::shared_ptr<std::unordered_map<std::string, Tensor>>>& request_list, 
                   std::vector<std::shared_ptr<std::unordered_map<std::string, Tensor>>>& output_list,
                   std::shared_ptr<ParallelGpt<T>>                              model_instance,
                   std::shared_ptr<ParallelGptWeight<T>>                        gpt_weights,
                   std::vector<void*>* pointer_record,
                   int device_id, 
                   model_config_t& model_config,
                   request_config_t& request_config,
                   std::string in_csv)
{
    check_cuda_error(cudaSetDevice(device_id));


    int*                                    d_input_ids     = nullptr;
    int*                                    d_input_lengths = nullptr;
    std::vector<int>                        v_start_lengths;
    std::vector<int>                        output_seq_len;
    std::vector<int>                        p_prompt_tuning_task_ids;

    int*                                    d_output_ids                = nullptr;
    int*                                    d_sequence_lengths          = nullptr;
    float*                                  d_output_log_probs          = nullptr;
    float*                                  d_cum_log_probs             = nullptr;
    float*                                  d_output_context_embeddings = nullptr;

    // Read ids of request from file.
    size_t     max_input_len      = 0;
    int* random_seed_ptr = new int(0);
    auto       request_batch_size = request_config.request_batch_size;
    const auto beam_width         = request_config.beam_width;
    const auto request_output_len = request_config.request_output_len;

    std::vector<int> v_start_ids;
    request_batch_size = read_start_ids(
        request_batch_size, &v_start_lengths, &v_start_ids, max_input_len, model_config.end_id, 1, in_csv);


    if (max_input_len > 0) {
        // conditional case.
        deviceMalloc(&d_input_ids, request_batch_size * max_input_len, false);
        deviceMalloc(&d_input_lengths, request_batch_size, false);

        cudaH2Dcpy(d_input_ids, v_start_ids.data(), request_batch_size * max_input_len);
        cudaH2Dcpy(d_input_lengths, v_start_lengths.data(), request_batch_size);
    }
    const size_t total_output_len = max_input_len + request_output_len;



    output_seq_len = std::vector<int>(request_batch_size, total_output_len);

     uint32_t* request_output_len_ptr = (uint32_t*)malloc(request_batch_size * sizeof(uint32_t));
    for (int i = 0; i < request_batch_size; i++) {
        request_output_len_ptr[i] = request_output_len;
    }

    request_list[device_id] = std::shared_ptr<std::unordered_map<std::string, Tensor>>(
            new std::unordered_map<std::string, Tensor>{
                {"input_ids",
                 Tensor{MEMORY_GPU,
                                TYPE_INT32,
                                std::vector<size_t>{(size_t)request_batch_size, (size_t)max_input_len},
                                d_input_ids}}});

    request_list[device_id]->insert({"input_lengths", {MEMORY_GPU, TYPE_INT32, {request_batch_size}, d_input_lengths}});
    request_list[device_id]->insert({"input_lengths_h", {MEMORY_CPU, TYPE_INT32, {request_batch_size}, v_start_lengths.data()}});
    request_list[device_id]->insert({"output_seq_len", {MEMORY_CPU, TYPE_UINT32, {request_batch_size}, request_output_len_ptr}});
    request_list[device_id]->insert({"temperature", {MEMORY_CPU, TYPE_FP32, {1}, &request_config.temperature}});
    request_list[device_id]->insert({"len_penalty", {MEMORY_CPU, TYPE_FP32, {1}, &request_config.len_penalty}});
    if (request_config.repetition_penalty != 1.0f) {
        request_list[device_id]->insert({"repetition_penalty", {MEMORY_CPU, TYPE_FP32, {1}, &request_config.repetition_penalty}});
    }
    if (request_config.presence_penalty != 0.0f) {
        request_list[device_id]->insert({"presence_penalty", {MEMORY_CPU, TYPE_FP32, {1}, &request_config.presence_penalty}});
    }
    request_list[device_id]->insert({"min_length", {MEMORY_CPU, TYPE_FP32, {1}, &request_config.min_length}});
    request_list[device_id]->insert({"random_seed", {MEMORY_CPU, TYPE_UINT64, {1}, random_seed_ptr}});

    if (request_config.top_k == 0 && request_config.top_p == 0.0f) {
        FT_CHECK(beam_width > 1);
        request_list[device_id]->insert(
            {"beam_search_diversity_rate", {MEMORY_CPU, TYPE_FP32, {1}, &request_config.beam_search_diversity_rate}});
    }
    else if (request_config.top_p != 0.0f) {
        request_list[device_id]->insert({"runtime_top_p", {MEMORY_CPU, TYPE_FP32, {1}, &request_config.top_p}});
    }
    else if (request_config.top_k != 0) {
        request_list[device_id]->insert({"runtime_top_k", {MEMORY_CPU, TYPE_UINT32, {1}, &request_config.top_k}});
    }




    deviceMalloc(&d_output_ids, request_batch_size * request_config.beam_width * total_output_len, false);
    deviceMalloc(&d_sequence_lengths, request_batch_size * request_config.beam_width, false);

    output_list[device_id] = std::shared_ptr<std::unordered_map<std::string, Tensor>>(
            new std::unordered_map<std::string, Tensor>{
                {"output_ids", {MEMORY_GPU, TYPE_INT32, {request_batch_size, beam_width, total_output_len}, d_output_ids}}}
                );
    output_list[device_id]->insert(
        {"sequence_length", {MEMORY_GPU, TYPE_INT32, {request_batch_size, beam_width}, d_sequence_lengths}});

    std::cout<<"prepare request success"<<std::endl;
    model_instance->forward(output_list[device_id].get(), request_list[device_id].get(), gpt_weights.get());
    std::cout<<"predict success"<<std::endl;
    safe_free(d_input_ids);
    safe_free(d_input_lengths);
    safe_free(d_output_ids);
    safe_free(d_sequence_lengths);

    return 0;
}



template<typename T>
int threadForward(std::shared_ptr<ParallelGpt<T>>                              model_instance,
                  std::shared_ptr<std::unordered_map<std::string, Tensor>>     request,
                  std::shared_ptr<std::unordered_map<std::string, Tensor>>*    output_tensors,
                  std::shared_ptr<ParallelGptWeight<T>>                        gpt_weights,
                  int                                                    device_id)
{
    check_cuda_error(cudaSetDevice(device_id));
    model_instance->forward((*output_tensors).get(), request.get(), gpt_weights.get());
    return 0;
}



template<typename T>
void multi_gpu_gpt_example(const INIReader reader, std::string in_csv);


int main(int argc, char* argv[])
{
    srand(0);

    //prase args
    std::string ini_name;
    if (argc >= 2) {
        ini_name = std::string(argv[1]);
    }
    else {
        ini_name = "../examples/cpp/multi_gpu_gpt/gpt_config.ini";
    }

    std::string in_csv;
    if (argc == 3) {
        in_csv = std::string(argv[2]);
    }
    else {
        in_csv = "../examples/cpp/multi_gpu_gpt/start_ids.csv";
    }

    INIReader reader = INIReader(ini_name);
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << ini_name << "'\n";
        return -1;
    }
    const std::string data_type = reader.Get("ft_instance_hyperparameter", "data_type");

  
    if (data_type == "fp32") {
        multi_gpu_gpt_example<float>(reader, in_csv);
    }
    else if (data_type == "fp16") {
        multi_gpu_gpt_example<half>(reader, in_csv);
    } else {
        printf("[ERROR] data_type should be fp32, fp16 or bf16 ! \n");
        return -1;
    }
    return 0;
}

template<typename T>
void multi_gpu_gpt_example(const INIReader reader, std::string in_csv)
{
    auto model_config   = read_model_config(reader);
    auto request_config = read_request_config(reader);

    const int   gpu_count  = 2;
    const int   node_id    = 0;
    const int   tp         = model_config.tensor_para_size;
    const int   pp         = model_config.pipeline_para_size;

    // step 1: Initialize the NCCL
    std::pair<std::vector<NcclParam>, std::vector<NcclParam>> nccl_comms;
    std::vector<NcclUid> nccl_ids;
    createNcclParams(nccl_comms, nccl_ids, tp, pp, 0, 0, 0);
    cudaDeviceSynchronize();

    // step 2: Create model instances
    std::vector<std::shared_ptr<ParallelGpt<T>>>       model_instances((size_t)gpu_count);
    std::vector<std::shared_ptr<ParallelGptWeight<T>>> model_weights((size_t)gpu_count);
    std::vector<std::thread>                           threads;

    std::vector<std::unique_ptr<Allocator<AllocatorType::CUDA>>>     allocator((size_t)gpu_count);
    std::vector<std::unique_ptr<cublasAlgoMap>>                      cublas_algo_map((size_t)gpu_count);
    std::vector<std::unique_ptr<std::mutex>>                         cublas_wrapper_mutex((size_t)gpu_count);
    std::vector<std::unique_ptr<cublasMMWrapper>>                    cublas_wrapper((size_t)gpu_count);
    std::vector<std::unique_ptr<cudaDeviceProp>>                     cuda_device_prop_ptr((size_t)gpu_count);

    threads.clear();

    for (int device_id = 0; device_id < gpu_count; device_id++) {
        const int rank = node_id * gpu_count + device_id;
        threads.push_back(std::thread(threadCreateModelInstances<T>,
                                      std::ref(model_instances),
                                      std::ref(model_weights),
                                      std::ref(allocator),
                                      std::ref(cublas_algo_map),
                                      std::ref(cublas_wrapper_mutex),
                                      std::ref(cublas_wrapper),
                                      std::ref(cuda_device_prop_ptr),
                                      model_config,
                                      request_config,
                                      device_id,
                                      rank,
                                      nccl_comms));
    }
    for (auto& t : threads) {
        t.join();
    }

    std::vector<void*> pointer_record;  // Used to prevent the pointers are release after leaving functions
    std::vector<std::shared_ptr<std::unordered_map<std::string, Tensor>>> request_list((size_t)gpu_count);
    std::vector<std::shared_ptr<std::unordered_map<std::string, Tensor>>> output_list((size_t)gpu_count);
    threads.clear();


    //For combine prepareRequest and forward
    /*for (int device_id = 0; device_id < gpu_count; device_id++) {
        //const int rank = node_id * gpu_count + device_id;
        threads.push_back(std::thread(prepareRequest_and_forward<T>,
                                      std::ref(request_list),
                                      std::ref(output_list),
                                      model_instances[device_id],
                                      model_weights[device_id],
                                      &pointer_record,
                                      device_id,
                                      std::ref(model_config),
                                      std::ref(request_config),
                                      in_csv));
    }
    for (auto& t : threads) {
        t.join();
    }*/
    

    
    for (int device_id = 0; device_id < gpu_count; device_id++) {
        //const int rank = node_id * gpu_count + device_id;
        threads.push_back(std::thread(prepareRequest,
                                      std::ref(request_list),
                                      std::ref(output_list),
                                      &pointer_record,
                                      device_id,
                                      std::ref(model_config),
                                      std::ref(request_config),
                                      in_csv));
    }
    for (auto& t : threads) {
        t.join();
    }

    print_mem_usage();

    int ite = 10;
    struct timeval start, end;
    cudaDeviceSynchronize();
    gettimeofday(&start, NULL);

    ft_nvtx::setScope("warmup_time");
    PUSH_RANGE("warmup time")
    for (int i = 0; i < ite; i++) {
        threads.clear();
        for (int device_id = 0; device_id < gpu_count; device_id++) {
            threads.push_back(std::thread(threadForward<T>,
                                          model_instances[device_id],
                                          request_list[device_id],
                                          &output_list[device_id],
                                          model_weights[device_id],
                                          device_id));
        }
        for (auto& t : threads) {
            t.join();
        }
    }
    threads.clear();
    FT_LOG_INFO("forward is completed.");
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);

    POP_RANGE;
    ft_nvtx::resetScope();
    
    write_output_tensors(*(output_list[0]));

    cudaProfilerStop();

    const auto total_output_len = output_list[0]->at("output_ids").shape[2];
    printf("[INFO] request_batch_size %ld beam_width %ld head_num %ld size_per_head %ld total_output_len %ld"
           " decoder_layers %ld vocab_size %ld FT-CPP-decoding-beamsearch-time %.2f ms\n",
           request_config.request_batch_size,
           request_config.beam_width,
           model_config.head_num,
           model_config.size_per_head,
           total_output_len,
           model_config.decoder_layers,
           model_config.vocab_size,
           ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001) / ite);

    std::cout<<"start to free"<<std::endl;
    model_instances.clear();
    std::cout<<"free done"<<std::endl;
}
