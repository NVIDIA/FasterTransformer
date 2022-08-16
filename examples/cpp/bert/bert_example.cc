/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/bert/Bert.h"
#include "src/fastertransformer/utils/logger.h"

using namespace fastertransformer;

template<typename T>
int bertExample(size_t batch_size,
                size_t num_layers,
                size_t seq_len,
                size_t head_num,
                size_t size_per_head,
                bool   is_remove_padding);

int main(int argc, char** argv)
{
    if (argc != 8) {
        FT_LOG_ERROR("bert_example batch_size num_layers seq_len head_num size_per_head data_type is_remove_padding");
        FT_LOG_ERROR("e.g., ./bin/bert_example 32 12 32 12 64 0 0");
        return 0;
    }

    int                  batch_size        = atoi(argv[1]);
    int                  num_layers        = atoi(argv[2]);
    int                  seq_len           = atoi(argv[3]);
    int                  head_num          = atoi(argv[4]);
    int                  size_per_head     = atoi(argv[5]);
    bool                 is_remove_padding = static_cast<bool>(atoi(argv[7]));
    const CublasDataType data_type         = static_cast<CublasDataType>(atoi(argv[6]));  // 0 FP32, 1 FP16, 2 BF 16

    if (data_type == FLOAT_DATATYPE) {
        return bertExample<float>(
            batch_size, num_layers, seq_len, head_num, size_per_head, is_remove_padding);
    }
    else if (data_type == HALF_DATATYPE) {
        return bertExample<half>(
            batch_size, num_layers, seq_len, head_num, size_per_head, is_remove_padding);
    }
#ifdef ENABLE_BF16
    else if (data_type == BFLOAT16_DATATYPE) {
        return bertExample<__nv_bfloat16>(
            batch_size, num_layers, seq_len, head_num, size_per_head, is_remove_padding);
    }
#endif
    else {
        throw std::runtime_error(std::string("[FT][ERROR] is_fp16 should be 0 (use float)"
                                             "or 1 (use half). \n "));
    }
}

template<typename T>
int bertExample(size_t batch_size,
                size_t num_layers,
                size_t seq_len,
                size_t head_num,
                size_t size_per_head,
                bool   is_remove_padding)
{
    printf("[INFO] Device: %s \n", getDeviceName().c_str());
    print_mem_usage("Before loading model");
    const size_t hidden_units = head_num * size_per_head;
    const size_t inter_size   = 4 * hidden_units;

    cudaStream_t     stream;
    cublasHandle_t   cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStreamCreate(&stream);
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
#ifdef SPARSITY_ENABLED
    cusparseLtHandle_t cusparselt_handle;
    CHECK_CUSPARSE(cusparseLtInit(&cusparselt_handle));
#endif
    cublasSetStream(cublas_handle, stream);
    cublasAlgoMap* cublas_algo_map = new cublasAlgoMap("gemm_config.in", "");

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
        cublas_wrapper.setFP16GemmConfig();
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        cublas_wrapper.setBF16GemmConfig();
    }
#endif
    else if (std::is_same<T, float>::value) {
        cublas_wrapper.setFP32GemmConfig();
    }

    BertWeight<T> bert_weights(hidden_units, inter_size, num_layers);

    AttentionType attention_type = getAttentionType<T>(size_per_head, getSMVersion(), is_remove_padding, seq_len);

    Bert<T> bert = Bert<T>(0,  // max_batch_size_, deprecated
                           0,  // max_seq_len_, deprecated
                           head_num,
                           size_per_head,
                           inter_size,
                           num_layers,
                           getSMVersion(),
                           1.0f,
                           stream,
                           &cublas_wrapper,
                           &allocator,
                           false,
                           attention_type,
                           false,
                           ActivationType::Gelu,
                           LayerNormType::post_layernorm);

    T* out_tensor;
    T* from_tensor;
    deviceMalloc(&out_tensor, batch_size * seq_len * head_num * size_per_head, false);
    deviceMalloc(&from_tensor, batch_size * seq_len * head_num * size_per_head, false);

    int*         h_sequence_lengths = new int[batch_size];
    unsigned int seed               = 0;
    for (uint i = 0; i < batch_size; i++) {
        h_sequence_lengths[i] = rand_r(&seed) % seq_len;
    }
    int* d_sequence_lengths;
    deviceMalloc(&d_sequence_lengths, batch_size, false);
    cudaH2Dcpy(d_sequence_lengths, h_sequence_lengths, batch_size);
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
    print_mem_usage("After loading model");

    // warmup
    for (int i = 0; i < 10; i++) {
        bert.forward(&output_tensors, &input_tensors, &bert_weights);
    }
    print_mem_usage("After inference");

    // profile time
    const int ite = 10;
    CudaTimer cuda_timer(stream);
    cuda_timer.start();
    for (int i = 0; i < ite; i++) {
        bert.forward(&output_tensors, &input_tensors, &bert_weights);
    }
    float total_time = cuda_timer.stop();

    FT_LOG_INFO("batch_size %ld seq_len %ld layer %ld "
                "FT-CPP-time %.2f ms (%d iterations) ",
                batch_size,
                seq_len,
                num_layers,
                total_time / ite,
                ite);

#ifdef SPARSITY_ENABLED
    cusparseLtDestroy(&cusparselt_handle);
#endif
    deviceFree(d_sequence_lengths);
    deviceFree(from_tensor);
    deviceFree(out_tensor);
    delete cublas_algo_map;
    delete cublas_wrapper_mutex;
    return 0;
}
