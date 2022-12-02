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

#include "src/fastertransformer/models/wenet/WenetEncoder.h"
#include "src/fastertransformer/utils/logger.h"

using namespace fastertransformer;

template<typename T>
int wenetEncoderExample(size_t batch_size, size_t num_layers, size_t seq_len, size_t head_num, size_t size_per_head);

int main(int argc, char** argv)
{
    if (argc != 7) {
        FT_LOG_ERROR("wenet_encoder_example batch_size num_layers seq_len head_num size_per_head is_fp16");
        FT_LOG_ERROR("e.g., ./build/bin/wenet_encoder_example 16 12 256 4 64 1");
        return 0;
    }

    int batch_size    = atoi(argv[1]);
    int num_layers    = atoi(argv[2]);
    int seq_len       = atoi(argv[3]);
    int head_num      = atoi(argv[4]);
    int size_per_head = atoi(argv[5]);
    // bool is_remove_padding = static_cast<bool>(atoi(argv[7])); // not supported yet

    if (atoi(argv[6]) == 0) {
        return wenetEncoderExample<float>(batch_size, num_layers, seq_len, head_num, size_per_head);
    }
    else if (atoi(argv[6]) == 1) {
        return wenetEncoderExample<half>(batch_size, num_layers, seq_len, head_num, size_per_head);
    }
    else {
        throw std::runtime_error(std::string("[FT][ERROR] data_type should be 0 (use float) or 1 (use half). \n"));
    }
}

template<typename T>
int wenetEncoderExample(size_t batch_size, size_t num_layers, size_t seq_len, size_t head_num, size_t size_per_head)
{
    const size_t hidden_units = head_num * size_per_head;
    const size_t d_model      = hidden_units;

    const size_t feature_size            = 80;
    const size_t max_len                 = 5000;
    const size_t inter_size              = 2048;
    const size_t vocab_size              = 4233;
    const size_t conv_module_kernel_size = 15;
    const int    sm                      = getSMVersion();
    const float  q_scaling               = 1.0f;

    const size_t kernel_size = 3;
    const size_t stride      = 2;
    const size_t seq_len1    = (seq_len - kernel_size) / stride + 1;
    const size_t seq_len2    = (seq_len1 - kernel_size) / stride + 1;

    cudaStream_t     stream;
    cublasHandle_t   cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudnnHandle_t    cudnn_handle;
    check_cuda_error(cudaStreamCreate(&stream));
    check_cuda_error(cublasCreate(&cublas_handle));
    check_cuda_error(cublasLtCreate(&cublaslt_handle));
    checkCUDNN(cudnnCreate(&cudnn_handle));
#ifdef SPARSITY_ENABLED
    cusparseLtHandle_t cusparselt_handle;
    CHECK_CUSPARSE(cusparseLtInit(&cusparselt_handle));
#endif
    check_cuda_error(cublasSetStream(cublas_handle, stream));
    checkCUDNN(cudnnSetStream(cudnn_handle, stream));
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
    else if (std::is_same<T, float>::value) {
        cublas_wrapper.setFP32GemmConfig();
    }

    WenetEncoderWeight<T> wenet_encoder_weights(head_num,
                                                size_per_head,
                                                inter_size,
                                                d_model,
                                                vocab_size,
                                                conv_module_kernel_size,
                                                feature_size,
                                                max_len,
                                                num_layers);
    // std::string paraFilePath = "/target/python/enc/bin_model/";
    // wenet_encoder_weights->loadModel(paraFilePath);

    AttentionType attention_type = AttentionType::UNFUSED_MHA;

    WenetEncoder<T> wenet_encoder = WenetEncoder<T>(0,  // max_batch_size_, deprecated
                                                    0,  // max_seq_len_, deprecated
                                                    head_num,
                                                    size_per_head,
                                                    feature_size,
                                                    max_len,
                                                    inter_size,
                                                    d_model,
                                                    num_layers,
                                                    vocab_size,
                                                    conv_module_kernel_size,
                                                    sm,
                                                    q_scaling,
                                                    cudnn_handle,
                                                    0,  // stream placeholder
                                                    &cublas_wrapper,
                                                    &allocator,
                                                    false,
                                                    attention_type,
                                                    false,
                                                    ActivationType::Silu);
    wenet_encoder.setStream(stream);

    T*     out_tensor1;
    int*   out_tensor2;
    float* out_tensor3;
    T*     from_tensor;
    deviceMalloc(&out_tensor1, batch_size * seq_len2 * d_model, false);
    deviceMalloc(&out_tensor2, batch_size, false);
    deviceMalloc(&out_tensor3, batch_size * seq_len2 * vocab_size, false);
    deviceMalloc(&from_tensor, batch_size * seq_len * feature_size, false);

    int*         h_sequence_lengths = new int[batch_size];
    unsigned int seed               = 0;
    for (uint i = 0; i < batch_size; i++) {
        h_sequence_lengths[i] = rand_r(&seed) % seq_len;
    }
    int* d_sequence_lengths;
    deviceMalloc(&d_sequence_lengths, batch_size, false);
    cudaH2Dcpy(d_sequence_lengths, h_sequence_lengths, batch_size);
    delete[] h_sequence_lengths;

    TensorMap input_tensors{
        {"speech",
         Tensor{MEMORY_GPU, getTensorType<T>(), std::vector<size_t>{batch_size, seq_len, feature_size}, from_tensor}},
        {"sequence_length", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size}, d_sequence_lengths}}};

    TensorMap output_tensors{
        {"output_hidden_state",
         Tensor{MEMORY_GPU, getTensorType<T>(), std::vector<size_t>{batch_size, seq_len2, d_model}, out_tensor1}},
        {"encoder_out_lens", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size}, out_tensor2}},
        {"ctc_log_probs",
         Tensor{MEMORY_GPU, TYPE_FP32, std::vector<size_t>{batch_size, seq_len2, vocab_size}, out_tensor3}},
    };

    // 1. init device buffer
    wenet_encoder.forward(&output_tensors, &input_tensors, &wenet_encoder_weights);

    // 2. CUDA graph capture
    // cudaGraph_t graph;
    // cudaGraphExec_t instance;
    // cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    // wenet_encoder.forward(&output_tensors, &input_tensors, &wenet_encoder_weights);
    // cudaStreamEndCapture(stream, &graph);
    // cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    // // 3. CUDA graph enqueue
    // cudaGraphLaunch(instance, stream);
    // cudaStreamSynchronize(stream);

    // warmup
    for (int i = 0; i < 10; i++) {
        wenet_encoder.forward(&output_tensors, &input_tensors, &wenet_encoder_weights);
        // cudaGraphLaunch(instance, stream);
        // cudaStreamSynchronize(stream);
    }

    // profile time
    const int ite = 100;
    CudaTimer cuda_timer(stream);
    cuda_timer.start();
    for (int i = 0; i < ite; i++) {
        wenet_encoder.forward(&output_tensors, &input_tensors, &wenet_encoder_weights);
        // cudaGraphLaunch(instance, stream);
        // cudaStreamSynchronize(stream);
    }
    float total_time = cuda_timer.stop();

    FT_LOG_INFO("batch_size %ld seq_len %ld layer %ld FT-CPP-time %.2f ms (%d iterations) ",
                batch_size,
                seq_len,
                num_layers,
                total_time / ite,
                ite);

    check_cuda_error(cudaFree(from_tensor));
    check_cuda_error(cudaFree(out_tensor1));
    check_cuda_error(cudaFree(out_tensor2));
    check_cuda_error(cudaFree(out_tensor3));
    check_cuda_error(cublasDestroy(cublas_handle));
    check_cuda_error(cublasLtDestroy(cublaslt_handle));
    checkCUDNN(cudnnDestroy(cudnn_handle));
#ifdef SPARSITY_ENABLED
    CHECK_CUSPARSE(cusparseLtDestroy(&cusparselt_handle));
#endif
    delete cublas_algo_map;
    delete cublas_wrapper_mutex;
    return 0;
}
