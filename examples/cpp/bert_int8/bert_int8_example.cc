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

#include "src/fastertransformer/models/bert_int8/BertINT8.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

using namespace fastertransformer;

template<typename T>
int bertINT8Example(size_t batch_size,
                    size_t num_layers,
                    size_t seq_len,
                    size_t head_num,
                    size_t size_per_head,
                    int int8_mode,
                    bool is_remove_padding,
                    bool allow_gemm_test = false);

int main(int argc, char** argv)
{
    if (argc != 9 && argc != 10) {
        printf("[ERROR] bert_int8_example batch_size num_layers seq_len head_num"
               "size_per_head is_fp16 is_remove_padding int8_mode\n");
        printf("e.g., ./bin/bert_int8_example 1 12 128 12 64 1 0 2\n");
        return 0;
    }
    bool allow_gemm_test = false;
    if (argc == 10) {
        allow_gemm_test = (atoi(argv[9]) == 1) ? true : false;
    }

    int batch_size = atoi(argv[1]);
    int num_layers = atoi(argv[2]);
    int seq_len = atoi(argv[3]);
    int head_num = atoi(argv[4]);
    int size_per_head = atoi(argv[5]);
    int int8_mode = atoi(argv[8]);
    bool is_remove_padding = static_cast<bool>(atoi(argv[7]));

    if (atoi(argv[6]) == 0) {
        return bertINT8Example<float>(
            batch_size, num_layers, seq_len, head_num, size_per_head, int8_mode, is_remove_padding, allow_gemm_test);
    }
    else if (atoi(argv[6]) == 1) {
        return bertINT8Example<half>(
            batch_size, num_layers, seq_len, head_num, size_per_head, int8_mode, is_remove_padding, allow_gemm_test);
    }
    else {
        throw std::runtime_error(std::string("[FT][ERROR] is_fp16 should be 0 (use float)"
                                             "or 1 (use half). \n "));
    }
}

template<typename T>
int bertINT8Example(size_t batch_size,
                    size_t num_layers,
                    size_t seq_len,
                    size_t head_num,
                    size_t size_per_head,
                    int int8_mode,
                    bool is_remove_padding,
                    bool allow_gemm_test)
{
    printf("[INFO] Device: %s \n", getDeviceName().c_str());

    const size_t hidden_units = head_num * size_per_head;
    const size_t inter_size = 4 * hidden_units;

    cudaStream_t stream;
    cublasLtHandle_t cublaslt_handle;
    cudaStreamCreate(&stream);
    cublasLtCreate(&cublaslt_handle);
#ifdef SPARSITY_ENABLED
    cusparseLtHandle_t cusparselt_handle;
    CHECK_CUSPARSE(cusparseLtInit(&cusparselt_handle));
#endif
    cublasAlgoMap* cublas_algo_map = new cublasAlgoMap("igemm_config.in");

    int sm = getSMVersion();

    Allocator<AllocatorType::CUDA> allocator(getDevice());

    std::mutex* cublas_wrapper_mutex = new std::mutex();

    bool use_ORDER_COL32_2R_4R4 = false;
#if (CUDART_VERSION >= 11000)
    if (sm >= 80) {
        use_ORDER_COL32_2R_4R4 = true;
    }
#endif

#ifdef SPARSITY_ENABLED
    cublasINT8MMWrapper cublas_wrapper = cublasINT8MMWrapper(
        cublaslt_handle, cusparselt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, use_ORDER_COL32_2R_4R4);
#else
    cublasINT8MMWrapper cublas_wrapper =
        cublasINT8MMWrapper(cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, use_ORDER_COL32_2R_4R4);
#endif

    std::vector<BertLayerINT8Weight<T>> bert_layer_weights(num_layers,
                                                           BertLayerINT8Weight<T>(hidden_units, inter_size));

    AttentionType attention_type = getAttentionTypeINT8<T>(size_per_head, sm, is_remove_padding, seq_len, int8_mode);

    BertINT8<T> bert_int8 = BertINT8<T>(batch_size,
                                        seq_len,
                                        head_num,
                                        size_per_head,
                                        inter_size,
                                        num_layers,
                                        sm,
                                        1.0f,
                                        int8_mode,
                                        stream,
                                        &cublas_wrapper,
                                        &allocator,
                                        false,
                                        attention_type);

    T* out_tensor;
    T* from_tensor;
    deviceMalloc(&out_tensor, batch_size * seq_len * head_num * size_per_head, false);
    deviceMalloc(&from_tensor, batch_size * seq_len * head_num * size_per_head, false);

    int* h_sequence_lengths = new int[batch_size];
    unsigned int seed = 0;
    for (uint i = 0; i < batch_size; i++) {
        h_sequence_lengths[i] = rand_r(&seed) % seq_len;
    }
    int* d_sequence_lengths;
    deviceMalloc(&d_sequence_lengths, batch_size);
    cudaMemcpy(d_sequence_lengths, h_sequence_lengths, sizeof(int) * batch_size, cudaMemcpyHostToDevice);
    delete[] h_sequence_lengths;

    std::vector<Tensor> input_tensors =
        std::vector<Tensor>{Tensor{MEMORY_GPU,
                                   getTensorType<T>(),
                                   std::vector<size_t>{batch_size, seq_len, (size_t)(head_num * size_per_head)},
                                   from_tensor},
                            Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size}, d_sequence_lengths}};

    std::vector<Tensor> output_tensors =
        std::vector<Tensor>{Tensor{MEMORY_GPU,
                                   getTensorType<T>(),
                                   std::vector<size_t>{batch_size, seq_len, (size_t)(head_num * size_per_head)},
                                   out_tensor}};

    // warmup
    for (int i = 0; i < 100; i++) {
        bert_int8.forward(&output_tensors, &input_tensors, &bert_layer_weights);
    }

    // profile time
    const int ite = 100;
    CudaTimer cuda_timer(stream);
    cuda_timer.start();
    for (int i = 0; i < ite; i++) {
        bert_int8.forward(&output_tensors, &input_tensors, &bert_layer_weights);
    }
    float total_time = cuda_timer.stop();

    printf("[INFO] batch_size %ld seq_len %ld layer %ld "
           "FT-CPP-time %.2f ms (%d iterations) \n",
           batch_size,
           seq_len,
           num_layers,
           total_time / ite,
           ite);

#ifdef SPARSITY_ENABLED
    cusparseLtDestroy(&cusparselt_handle);
#endif
    delete cublas_algo_map;
    delete cublas_wrapper_mutex;
    return 0;
}
