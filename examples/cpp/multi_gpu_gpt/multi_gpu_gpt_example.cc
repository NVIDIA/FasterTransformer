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
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.h"
#include "src/fastertransformer/utils/mpi_utils.h"
#include "src/fastertransformer/utils/nvtx_utils.h"

#include <cuda_profiler_api.h>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <vector>

#ifdef USE_NVTX
bool NVTX_ON = true;
#endif

using namespace fastertransformer;

template<typename T>
void multi_gpu_gpt_example(const INIReader reader);

int main(int argc, char* argv[])
{
    MPICHECK(MPI_Init(&argc, &argv));
    srand(0);

    std::string ini_name;
    if (argc == 2) {
        ini_name = std::string(argv[1]);
    }
    else {
        ini_name = "../examples/cpp/multi_gpu_gpt/gpt_config.ini";
    }

    INIReader reader = INIReader(ini_name);
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << ini_name << "'\n";
        return -1;
    }
    const std::string data_type = reader.Get("ft_instance_hyperparameter", "data_type");

    if (data_type == "fp32") {
        multi_gpu_gpt_example<float>(reader);
    }
    else if (data_type == "fp16") {
        multi_gpu_gpt_example<half>(reader);
    }
#ifdef ENABLE_BF16
    else if (data_type == "bf16") {
        multi_gpu_gpt_example<__nv_bfloat16>(reader);
    }
#endif
    else {
        printf("[ERROR] data_type should be fp32, fp16 or bf16 ! \n");
        return -1;
    }
    MPI_Finalize();
    return 0;
}

template<typename T>
void multi_gpu_gpt_example(const INIReader reader)
{
    const std::string model_name = reader.Get("ft_instance_hyperparameter", "model_name");
    const size_t max_batch_size = (size_t)reader.GetInteger("ft_instance_hyperparameter", "max_batch_size");
    const size_t max_seq_len = (size_t)reader.GetInteger("ft_instance_hyperparameter", "max_seq_len");
    const size_t beam_width = (size_t)reader.GetInteger("ft_instance_hyperparameter", "beam_width");
    const int top_k = reader.GetInteger("ft_instance_hyperparameter", "top_k");
    const float top_p = reader.GetFloat("ft_instance_hyperparameter", "top_p");
    const float temperature = reader.GetFloat("ft_instance_hyperparameter", "temperature");
    const float repetition_penalty = reader.GetFloat("ft_instance_hyperparameter", "repetition_penalty");
    const std::string model_dir = std::string(reader.Get("ft_instance_hyperparameter", "model_dir"));
    const bool sparse = static_cast<bool>(reader.GetInteger("ft_instance_hyperparameter", "sparse"));
    const int int8_mode = reader.GetInteger("ft_instance_hyperparameter", "int8_mode");
    const float len_penalty = 1.0f;
    const float beam_search_diversity_rate = 0.0f;

    const int tensor_para_size = reader.GetInteger("ft_instance_hyperparameter", "tensor_para_size");
    const int pipeline_para_size = reader.GetInteger("ft_instance_hyperparameter", "pipeline_para_size");

    const size_t head_num = (size_t)reader.GetInteger(model_name, "head_num");
    const size_t size_per_head = (size_t)reader.GetInteger(model_name, "size_per_head");
    const size_t vocab_size = (size_t)reader.GetInteger(model_name, "vocab_size");
    const size_t decoder_layers = (size_t)reader.GetInteger(model_name, "decoder_layers");
    const size_t hidden_units = head_num * size_per_head;
    const size_t inter_size = 4 * hidden_units;

    const size_t request_batch_size = reader.GetInteger("request", "request_batch_size");
    // The length of tokens we hope this model to generate
    const int request_output_len = reader.GetInteger("request", "request_output_len");
    const bool is_return_log_probs = reader.GetBoolean("request", "return_log_probs", false);
    // Whether to include input contexts in computing the cumulative log probabilities.
    const bool is_return_context_cum_log_probs = reader.GetBoolean("request", "context_log_probs", false);
    if (is_return_log_probs && !is_return_context_cum_log_probs) {
        FT_LOG_WARNING("context_log_probs will be ignored since return_log_probs is disabled.");
    }

    const int start_id = 50256;
    const int end_id = 50256;

    FT_CHECK(head_num % tensor_para_size == 0);
    FT_CHECK(decoder_layers % pipeline_para_size == 0);

    // Prepare the parallelism parameters
    int rank, world_size, device, device_count;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    if (rank == 0) {
        printf("Total ranks: %d.\n", world_size);
    }
    check_cuda_error(cudaGetDeviceCount(&device_count));
    check_cuda_error(cudaSetDevice(rank % device_count));
    check_cuda_error(cudaGetDevice(&device));

    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, device));
    printf("Device %s\n", prop.name);

    printf("P%d is runing with %d GPU.\n", rank, device);

    if (tensor_para_size * pipeline_para_size != world_size) {
        printf("[ERROR] tensor_para_size * pipeline_para_size should equal to world_size \n");
        exit(-1);
    }

    const int tensor_para_rank = rank % tensor_para_size;
    const int pipeline_para_rank = rank / tensor_para_size;
    const int layers_per_group = decoder_layers / pipeline_para_size;
    if (layers_per_group * pipeline_para_size != (int)decoder_layers) {
        printf("[ERROR] layers_per_group (%d) * pipeline_para_size (%d) should equal to decoder_layers (%ld) \n",
               layers_per_group,
               pipeline_para_size,
               decoder_layers);
        exit(-1);
    }

    // assume gpu_num = k * n,
    // tensor parallelism group size is n
    // pipeline parallelism group size is k

    // convert WORLD communicator into 2D grid (k * n) communicator
    // comms of the same row means they are in the same tensor parallel group
    // comms of the same col means they are in the same pipeline parallel group
    MPI_Comm grid_comm;
    int dims[2] = {pipeline_para_size, tensor_para_size};
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);

    MPI_Comm comm_tensor_parallel, comm_pipeline_parallel;

    int remain_dims_tensor_parallel[2] = {false, true};
    int remain_dims_pipeline_parallel[2] = {true, false};
    // split 2D communicator into rows and cols, each row = one tensor parallel group, each col = one pipeline parallel
    // group
    MPI_Cart_sub(grid_comm, remain_dims_tensor_parallel, &comm_tensor_parallel);
    MPI_Cart_sub(grid_comm, remain_dims_pipeline_parallel, &comm_pipeline_parallel);

    int rank_tensor_parallel, rank_pipeline_parallel;
    MPI_Comm_rank(comm_tensor_parallel, &rank_tensor_parallel);
    MPI_Comm_rank(comm_pipeline_parallel, &rank_pipeline_parallel);

    ncclUniqueId tensor_para_nccl_uid;
    ncclUniqueId pipeline_para_nccl_uid;
    // root of tensor parallel group and pipeline parallel group creates the nccl uid
    if (rank_tensor_parallel == 0) {
        NCCLCHECK(ncclGetUniqueId(&tensor_para_nccl_uid));
    }

    if (rank_pipeline_parallel == 0) {
        NCCLCHECK(ncclGetUniqueId(&pipeline_para_nccl_uid));
    }
    // broadcast nccl uid to the comms in the same tensor parallel group or pipeline parallel group
    MPI_Bcast(&tensor_para_nccl_uid, sizeof(tensor_para_nccl_uid), MPI_BYTE, 0, comm_tensor_parallel);
    MPI_Bcast(&pipeline_para_nccl_uid, sizeof(pipeline_para_nccl_uid), MPI_BYTE, 0, comm_pipeline_parallel);

    ncclComm_t tensor_para_nccl_comm, pipeline_para_nccl_comm;
    NCCLCHECK(ncclCommInitRank(&tensor_para_nccl_comm, tensor_para_size, tensor_para_nccl_uid, tensor_para_rank));
    NCCLCHECK(
        ncclCommInitRank(&pipeline_para_nccl_comm, pipeline_para_size, pipeline_para_nccl_uid, pipeline_para_rank));

    // Read ids of request from file.
    int max_input_len = -1;
    std::vector<int> v_start_lengths;
    std::vector<int> v_start_ids;
    read_start_ids(request_batch_size,
                   &v_start_lengths,
                   &v_start_ids,
                   max_input_len,
                   end_id,
                   1,
                   "../examples/cpp/multi_gpu_gpt/start_ids.csv");

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
        deviceMalloc(&d_input_ids, request_batch_size * max_input_len, false);
        deviceMalloc(&d_input_lengths, request_batch_size, false);
        cudaH2Dcpy(d_input_ids, v_start_ids.data(), request_batch_size * max_input_len);
        cudaH2Dcpy(d_input_lengths, v_start_lengths.data(), request_batch_size);
    }

    const int total_output_len = max_input_len + request_output_len;
    if (total_output_len > (int)max_seq_len) {
        printf("[ERROR] total_output_len (%d) should be <= max_seq_len (%ld). \n", total_output_len, max_seq_len);
        exit(-1);
    }

    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStreamCreate(&stream);
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
    cublasSetStream(cublas_handle, stream);
#ifdef SPARSITY_ENABLED
    cusparseLtHandle_t cusparselt_handle;
    CHECK_CUSPARSE(cusparseLtInit(&cusparselt_handle));
    cublasAlgoMap* cublas_algo_map = new cublasAlgoMap(GEMM_CONFIG, SPGEMM_CONFIG);
#else
    cublasAlgoMap* cublas_algo_map = new cublasAlgoMap(GEMM_CONFIG);
#endif

    Allocator<AllocatorType::CUDA> allocator(getDevice());

    std::mutex* cublas_wrapper_mutex = new std::mutex();
#ifdef SPARSITY_ENABLED
    cublasMMWrapper cublas_wrapper = cublasMMWrapper(
        cublas_handle, cublaslt_handle, cusparselt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, &allocator);
#else
    cublasMMWrapper cublas_wrapper =
        cublasMMWrapper(cublas_handle, cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, &allocator);
#endif

    if (std::is_same<T, half>::value) {
        cublas_wrapper.setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        cublas_wrapper.setBF16GemmConfig();
    }
#endif
    else if (std::is_same<T, float>::value) {
        cublas_wrapper.setFP32GemmConfig();
    }

    fastertransformer::ParallelGptWeight<T> gpt_weights(hidden_units,
                                                        inter_size,
                                                        vocab_size,
                                                        decoder_layers,
                                                        max_seq_len,
                                                        tensor_para_size,
                                                        tensor_para_rank,
                                                        pipeline_para_size,
                                                        pipeline_para_rank,
                                                        int8_mode);
    gpt_weights.loadModel(model_dir);
#ifdef SPARSITY_ENABLED
    if (sparse) {
        printf("[INFO] Compress weights for sparse inference\n");
        gpt_weights.compress_weights(cublas_wrapper);
    }
#endif

    unsigned long long random_seed;
    if (rank == 0) {
        random_seed = (unsigned long long)(0);
    }
    if (world_size > 1) {
        MPICHECK(MPI_Bcast(&random_seed, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD));
    }

    NcclParam tensor_para(tensor_para_rank, tensor_para_size, tensor_para_nccl_comm);
    NcclParam pipeline_para(pipeline_para_rank, pipeline_para_size, pipeline_para_nccl_comm);

    ParallelGpt<T> gpt = ParallelGpt<T>(0,  // max_batch_size, FT will adjust the buffer automatically.
                                        0,  // max_seq_len, FT will adjust the buffer automatically.
                                        0,  // max_input_len, FT will adjust the buffer automatically.
                                        beam_width,
                                        head_num,
                                        size_per_head,
                                        inter_size,
                                        decoder_layers,
                                        vocab_size,
                                        start_id,
                                        end_id,
                                        0.0f,  // beam_search_diversity_rate,
                                        0,     // top_k,
                                        0.0,   // top_p,
                                        0,     // random_seed,
                                        1.0f,  // temperature,
                                        1.0f,  // len_penalty,
                                        1.0f,  // repetition_penalty,
                                        tensor_para,
                                        pipeline_para,
                                        stream,
                                        &cublas_wrapper,
                                        &allocator,
                                        false,
                                        &prop,
                                        sparse,
                                        int8_mode);

    int* d_output_ids;
    int* d_parent_ids;
    int* d_sequence_lengths;
    deviceMalloc(&d_output_ids, request_batch_size * beam_width * total_output_len, false);
    deviceMalloc(&d_parent_ids, request_batch_size * beam_width * total_output_len, false);
    deviceMalloc(&d_sequence_lengths, request_batch_size * beam_width, false);

    std::unordered_map<std::string, Tensor> input_tensors = std::unordered_map<std::string, Tensor>{
        {"input_ids",
         Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size, (size_t)max_input_len}, d_input_ids}},
        {"input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size}, d_input_lengths}},
        {"max_output_seq_len", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{1}, &total_output_len}}};
    if (top_k == 0 && top_p == 0.0f) {
        FT_CHECK(beam_width > 1);
        input_tensors.insert({"beam_search_diversity_rate",
                              Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &beam_search_diversity_rate}});
    }
    else {
        if (top_p != 0.0f) {
            input_tensors.insert({"runtime_top_p", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &top_p}});
        }
        if (top_k != 0) {
            input_tensors.insert({"runtime_top_k", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{1}, &top_k}});
        }
    }
    input_tensors.insert({"temperature", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &temperature}});
    input_tensors.insert({"len_penalty", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &len_penalty}});
    input_tensors.insert(
        {"repetition_penalty", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &repetition_penalty}});
    input_tensors.insert({"random_seed", Tensor{MEMORY_CPU, TYPE_UINT64, std::vector<size_t>{1}, &random_seed}});

    std::unordered_map<std::string, Tensor> output_tensors = std::unordered_map<std::string, Tensor>{
        {"output_ids",
         Tensor{MEMORY_GPU,
                TYPE_INT32,
                std::vector<size_t>{request_batch_size, beam_width, (size_t)total_output_len},
                d_output_ids}},
        {"parent_ids",
         Tensor{MEMORY_GPU,
                TYPE_INT32,
                std::vector<size_t>{(size_t)total_output_len, request_batch_size, beam_width},
                d_parent_ids}},
        {"sequence_length",
         Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size, beam_width}, d_sequence_lengths}}};

    float* output_log_probs = nullptr;
    float* d_cum_log_probs = nullptr;
    if (is_return_log_probs) {
        deviceMalloc(&output_log_probs, request_batch_size * beam_width * request_output_len);
        output_tensors.insert({"output_log_probs",
                               Tensor{MEMORY_GPU,
                                      TYPE_FP32,
                                      std::vector<size_t>{request_batch_size, beam_width, (size_t)request_output_len},
                                      output_log_probs}});
        deviceMalloc(&d_cum_log_probs, request_batch_size * beam_width);
        output_tensors.insert(
            {"cum_log_probs",
             Tensor{MEMORY_GPU, TYPE_FP32, std::vector<size_t>{request_batch_size, beam_width}, d_cum_log_probs}});
        input_tensors.insert({"is_return_context_cum_log_probs",
                              Tensor{MEMORY_CPU, TYPE_BOOL, std::vector<size_t>{1}, &is_return_context_cum_log_probs}});
    }

    print_mem_usage();

    int ite = 1;
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    cudaProfilerStart();
    // warm up
    ite = 1;
    nvtx::setScope("warmup_time");
    PUSH_RANGE("warmup time")
    for (int i = 0; i < ite; ++i) {
        gpt.forward(&output_tensors, &input_tensors, &gpt_weights);
    }
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    POP_RANGE;
    nvtx::resetScope();

    if (rank == 0) {
        std::string fName = "out";
        auto outFile = std::ofstream(fName, std::ios::out);
        if (!outFile.is_open()) {
            printf("[WARNING] Cannot write results into output file %s \n", fName.c_str());
        }
        else {
            size_t outCount = total_output_len * request_batch_size * beam_width;
            int* hBuf = new int[outCount];
            cudaD2Hcpy(hBuf, d_output_ids, outCount);

            {
                std::cout << "Writing " << outCount << " elements\n";
                int zeroCount = 0;
                for (size_t i = 0; i < outCount; i++) {
                    if (hBuf[i] == int(0)) {
                        zeroCount++;
                    }
                    outFile << hBuf[i] << " ";
                    if ((i + 1) % (total_output_len) == 0) {
                        outFile << std::endl;
                    }

                    if (i < 10) {
                        printf("%5d ", hBuf[i]);
                    }
                    if ((i + 1) % (total_output_len) == 0 && i < 10) {
                        std::cout << std::endl;
                    }
                }
                std::cout << std::endl << "zeroCount = " << zeroCount << std::endl;
            }
            delete[] hBuf;
        }
        outFile.close();

        if (d_cum_log_probs != nullptr) {
            std::string logprob_fname = "logprob.out";
            std::ofstream logprob_file = std::ofstream("logprob.out", std::ios::out);
            if (!logprob_file.is_open()) {
                printf("[WARNING] Cannot write results into output file %s \n", logprob_fname.c_str());
            }
            else {
                size_t cum_log_probs_size = request_batch_size * beam_width;
                printf("[INFO] Writing %ld elements (log probs)\n", cum_log_probs_size);
                float* h_buf = new float[cum_log_probs_size];
                cudaD2Hcpy(h_buf, d_cum_log_probs, cum_log_probs_size);
                for (size_t i = 0; i < cum_log_probs_size; i++) {
                    logprob_file << h_buf[i] << std::endl;
                    if (i < 10) {
                        printf(" %10.6f\n", h_buf[i]);
                    }
                }
                delete[] h_buf;
            }
            logprob_file.close();
        }
    }

    // test time
    struct timeval start, end;
    MPI_Barrier(MPI_COMM_WORLD);
    cudaDeviceSynchronize();
    gettimeofday(&start, NULL);

    nvtx::setScope("total_time");
    PUSH_RANGE("total time")
    for (int i = 0; i < ite; ++i) {
        gpt.forward(&output_tensors, &input_tensors, &gpt_weights);
    }

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    POP_RANGE;
    nvtx::resetScope();
    gettimeofday(&end, NULL);

    cudaProfilerStop();

    printf("[INFO] request_batch_size %ld beam_width %ld head_num %ld size_per_head %ld total_output_len %d"
           " decoder_layers %ld vocab_size %ld FT-CPP-decoding-beamsearch-time %.2f ms\n",
           request_batch_size,
           beam_width,
           head_num,
           size_per_head,
           total_output_len,
           decoder_layers,
           vocab_size,
           ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001) / ite);

    ncclCommDestroy(tensor_para_nccl_comm);
    ncclCommDestroy(pipeline_para_nccl_comm);

#ifdef SPARSITY_ENABLED
    cusparseLtDestroy(&cusparselt_handle);
#endif
    delete cublas_algo_map;
    delete cublas_wrapper_mutex;
    return;
}
