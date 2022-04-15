/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/decoding/Decoding.h"

#include <cuda_profiler_api.h>
#include <sys/time.h>

using namespace fastertransformer;

template<typename T>
int decodingExample(const size_t batch_size,
                    const size_t beam_width,
                    const size_t head_num,
                    const size_t size_per_head,
                    const size_t inter_size,
                    const size_t vocab_size,
                    const size_t num_layers,
                    const size_t max_seq_len,
                    const size_t memory_max_seq_len,
                    const size_t memory_hidden_units,
                    const int top_k,
                    const float top_p);

int main(int argc, char** argv)
{
    if (argc != 14) {
        printf("[ERROR] decoding_example batch_size beam_width head_num size_per_head inter_size vocab_size"
               " num_layers max_seq_len memory_max_seq_len memory_hidden_units top_k top_p is_fp16\n");
        printf("e.g., ./bin/decoding_example 4 1 8 64 2048 30000 6 32 32 512 0 0.6 1\n");
        return 0;
    }

    int batch_size = atoi(argv[1]);
    int beam_width = atoi(argv[2]);
    int head_num = atoi(argv[3]);
    int size_per_head = atoi(argv[4]);
    int inter_size = atoi(argv[5]);
    int vocab_size = atoi(argv[6]);
    int num_layers = atoi(argv[7]);
    int max_seq_len = atoi(argv[8]);
    int memory_max_seq_len = atoi(argv[9]);
    int memory_hidden_units = atoi(argv[10]);
    int top_k = atoi(argv[11]);
    float top_p = atof(argv[12]);

    if (atoi(argv[13]) == 0) {
        return decodingExample<float>(batch_size,
                                      beam_width,
                                      head_num,
                                      size_per_head,
                                      inter_size,
                                      vocab_size,
                                      num_layers,
                                      max_seq_len,
                                      memory_max_seq_len,
                                      memory_hidden_units,
                                      top_k,
                                      top_p);
    }
    else if (atoi(argv[13]) == 1) {
        return decodingExample<half>(batch_size,
                                     beam_width,
                                     head_num,
                                     size_per_head,
                                     inter_size,
                                     vocab_size,
                                     num_layers,
                                     max_seq_len,
                                     memory_max_seq_len,
                                     memory_hidden_units,
                                     top_k,
                                     top_p);
    }
    else {
        throw std::runtime_error(std::string("[FT][ERROR] is_fp16 should be 0 (use float)"
                                             "or 1 (use half). \n "));
    }
}

template<typename T>
int decodingExample(const size_t batch_size,
                    const size_t beam_width,
                    const size_t head_num,
                    const size_t size_per_head,
                    const size_t inter_size,
                    const size_t vocab_size,
                    const size_t num_layers,
                    const size_t max_seq_len,
                    const size_t memory_max_seq_len,
                    const size_t memory_hidden_units,
                    const int top_k,
                    const float top_p)
{
    const size_t hidden_units = head_num * size_per_head;
    const int start_id = 50256;
    const int end_id = 50256;

    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStreamCreate(&stream);
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
    cublasSetStream(cublas_handle, stream);
    cublasAlgoMap* cublas_algo_map = new cublasAlgoMap("gemm_config.in");

    Allocator<AllocatorType::CUDA> allocator(getDevice());

    std::mutex* cublas_wrapper_mutex = new std::mutex();
    cublasMMWrapper cublas_wrapper =
        cublasMMWrapper(cublas_handle, cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, &allocator);
    if (std::is_same<T, half>::value) {
        cublas_wrapper.setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper.setFP32GemmConfig();
    }
    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));

    fastertransformer::DecodingWeight<T> decoding_weights(
        hidden_units, inter_size, vocab_size, num_layers, max_seq_len, memory_hidden_units);
    Decoding<T> decoding = Decoding<T>(batch_size,
                                       max_seq_len,
                                       memory_max_seq_len,
                                       beam_width,
                                       head_num,
                                       size_per_head,
                                       inter_size,
                                       num_layers,
                                       vocab_size,
                                       start_id,
                                       end_id,
                                       0.0f,
                                       top_k,
                                       top_p,
                                       1.0,   // temperature
                                       1.0f,  // len_penalty,
                                       1.0,   // repetition_penalty
                                       stream,
                                       &cublas_wrapper,
                                       &allocator,
                                       false,
                                       &prop);

    T* d_memory_tensor;
    int* d_memory_sequence_lengths;
    deviceMalloc(&d_memory_tensor, memory_hidden_units * memory_max_seq_len * batch_size * beam_width);
    deviceMalloc(&d_memory_sequence_lengths, batch_size * beam_width);
    int* h_memory_sequence_lengths = new int[batch_size * beam_width];
    for (int i = 0; i < (int)(batch_size * beam_width); i++) {
        h_memory_sequence_lengths[i] = memory_max_seq_len;
    }
    cudaH2Dcpy(d_memory_sequence_lengths, h_memory_sequence_lengths, batch_size * beam_width);

    int* d_output_ids;
    int* d_parent_ids;
    int* d_sequence_lengths;
    deviceMalloc(&d_output_ids, batch_size * beam_width * max_seq_len, false);
    deviceMalloc(&d_parent_ids, batch_size * beam_width * max_seq_len, false);
    deviceMalloc(&d_sequence_lengths, batch_size * beam_width, false);

    std::vector<Tensor> input_tensors = std::vector<Tensor>{
        Tensor{MEMORY_GPU,
               getTensorType<T>(),
               std::vector<size_t>{batch_size * beam_width, memory_max_seq_len, memory_hidden_units},
               d_memory_tensor},
        Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size * beam_width}, d_memory_sequence_lengths}};

    std::vector<Tensor> output_tensors = std::vector<Tensor>{
        Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{(size_t)max_seq_len, batch_size, beam_width}, d_output_ids},
        Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{(size_t)max_seq_len, batch_size, beam_width}, d_parent_ids},
        Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size, beam_width}, d_sequence_lengths}};
    print_mem_usage();

    cudaProfilerStart();
    const int ite = 10;
    // warm up
    for (int i = 0; i < ite; ++i) {
        decoding.forward(&output_tensors, &input_tensors, &decoding_weights);
    }
    cudaDeviceSynchronize();

    struct timeval start, end;
    cudaDeviceSynchronize();
    gettimeofday(&start, NULL);

    for (int i = 0; i < ite; ++i) {
        decoding.forward(&output_tensors, &input_tensors, &decoding_weights);
    }

    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);

    cudaProfilerStop();

    printf("[INFO] batch_size %ld beam_width %ld head_num %ld size_per_head %ld max_seq_len %ld"
           " num_layers %ld vocab_size %ld, top_k %d, top_p %.3f, FT-CPP-decoding-time %.2f ms\n",
           batch_size,
           beam_width,
           head_num,
           size_per_head,
           max_seq_len,
           num_layers,
           vocab_size,
           top_k,
           top_p,
           ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001) / ite);

    delete cublas_algo_map;
    delete cublas_wrapper_mutex;
    return 0;
}
