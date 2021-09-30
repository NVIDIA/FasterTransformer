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

#include "cnpy.h"
#include "src/fastertransformer/models/xlnet/Xlnet.h"
using namespace fastertransformer;

template<typename T>
int xlnetExample(size_t batch_size, size_t num_layers, size_t seq_len, size_t head_num, size_t size_per_head);

int main(int argc, char** argv)
{
    if (argc != 7) {
        printf("[ERROR] xlnet_example <batch_size> <num_layers> <seq_len> <head_num> "
               "<size_per_head> <is_fp16>\n");
        printf("e.g., ./bin/xlnet_example 8 12 128 12 64 0\n");
        return 0;
    }

    int batch_size = atoi(argv[1]);
    int num_layers = atoi(argv[2]);
    int seq_len = atoi(argv[3]);
    int head_num = atoi(argv[4]);
    int size_per_head = atoi(argv[5]);
    bool is_fp16 = atoi(argv[6]);

    if (is_fp16 == 0) {
        return xlnetExample<float>(batch_size, num_layers, seq_len, head_num, size_per_head);
    }
    else if (is_fp16 == 1) {
        return xlnetExample<half>(batch_size, num_layers, seq_len, head_num, size_per_head);
    }
    else {
        throw std::runtime_error(std::string("[FT][ERROR] is_fp16 should be 0 (use float)"
                                             "or 1 (use half). \n "));
    }
}

template<typename T>
int xlnetExample(size_t batch_size, size_t num_layers, size_t seq_len, size_t head_num, size_t size_per_head)
{
    printf("[INFO] Device: %s \n", getDeviceName().c_str());

    const size_t hidden_units = head_num * size_per_head;
    const size_t inter_size = 4 * hidden_units;

    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStreamCreate(&stream);
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);

    cublasSetStream(cublas_handle, stream);
    cublasAlgoMap* cublas_algo_map = new cublasAlgoMap("gemm_config.in", "");

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

    // Set layer weight
    std::vector<XlnetLayerWeight<T>> xlnet_layer_weights(num_layers, XlnetLayerWeight<T>(hidden_units, inter_size));

    // Allocate Input & Output
    T* word_emb_k;
    deviceMalloc(&word_emb_k, batch_size * seq_len * hidden_units, false);
    float* input_mask;
    deviceMalloc(&input_mask, batch_size * seq_len, false);
    int* seg_id;
    deviceMalloc(&seg_id, batch_size * seq_len, false);

    T* out_tensor;
    deviceMalloc(&out_tensor, batch_size * seq_len * hidden_units, false);

    std::vector<Tensor> input_tensors = std::vector<Tensor>{
        Tensor{MEMORY_GPU, getTensorType<T>(), std::vector<size_t>{batch_size, seq_len, hidden_units}, word_emb_k},
        Tensor{MEMORY_GPU, getTensorType<float>(), std::vector<size_t>{batch_size, seq_len}, input_mask},
        Tensor{MEMORY_GPU, getTensorType<int>(), std::vector<size_t>{batch_size, seq_len}, seg_id}};

    std::vector<Tensor> output_tensors = std::vector<Tensor>{
        Tensor{MEMORY_GPU, getTensorType<T>(), std::vector<size_t>{batch_size, seq_len, hidden_units}, out_tensor}};

    Xlnet<T> xlnet = Xlnet<T>(batch_size,
                              seq_len,
                              head_num,
                              size_per_head,
                              inter_size,
                              num_layers,
                              1.0f,
                              stream,
                              &cublas_wrapper,
                              &allocator,
                              false);

    // warmup
    for (int i = 0; i < 10; i++) {
        xlnet.forward(&output_tensors, &input_tensors, &xlnet_layer_weights);
    }

    // profile time
    const int ite = 10;
    CudaTimer cuda_timer(stream);
    cuda_timer.start();
    for (int i = 0; i < ite; i++) {
        xlnet.forward(&output_tensors, &input_tensors, &xlnet_layer_weights);
    }
    float total_time = cuda_timer.stop();

    printf("[INFO] batch_size %ld seq_len %ld layer %ld "
           "FT-CPP-time %.2f ms (%d iterations) \n",
           batch_size,
           seq_len,
           num_layers,
           total_time / ite,
           ite);

    delete cublas_algo_map;
    delete cublas_wrapper_mutex;

    cudaFree(word_emb_k);
    cudaFree(input_mask);
    cudaFree(seg_id);
    cudaFree(out_tensor);

    return 0;
}
