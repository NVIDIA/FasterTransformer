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

#include "src/fastertransformer/models/wenet/WenetDecoder.h"
#include "src/fastertransformer/utils/logger.h"

using namespace fastertransformer;

template<typename T>
int wenetDecoderExample(size_t batch_size, size_t num_layers, size_t seq_len, size_t head_num, size_t size_per_head);

int main(int argc, char** argv)
{
    if (argc != 7) {
        FT_LOG_ERROR("wenet_decoder_example batch_size num_layers seq_len head_num size_per_head is_fp16");
        FT_LOG_ERROR("e.g., ./build/bin/wenet_decoder_example 16 6 256 4 64 1");
        return 0;
    }

    int batch_size    = atoi(argv[1]);
    int num_layers    = atoi(argv[2]);
    int seq_len       = atoi(argv[3]);
    int head_num      = atoi(argv[4]);
    int size_per_head = atoi(argv[5]);
    // bool is_remove_padding = static_cast<bool>(atoi(argv[7])); // not supported yet

    if (atoi(argv[6]) == 0) {
        return wenetDecoderExample<float>(batch_size, num_layers, seq_len, head_num, size_per_head);
    }
    else if (atoi(argv[6]) == 1) {
        return wenetDecoderExample<half>(batch_size, num_layers, seq_len, head_num, size_per_head);
    }
    else {
        throw std::runtime_error(std::string("[FT][ERROR] is_fp16 should be 0 (use float)"
                                             "or 1 (use half). \n "));
    }
}

template<typename T>
int wenetDecoderExample(size_t batch_size, size_t num_layers, size_t seq_len, size_t head_num, size_t size_per_head)
{
    // printf("[INFO] Device: %s \n", getDeviceName().c_str());

    const size_t hidden_units = head_num * size_per_head;
    const size_t d_model      = hidden_units;

    const size_t max_len    = 5000;
    const size_t inter_size = 2048;
    const size_t vocab_size = 4233;
    const size_t beam_width = 10;
    const int    sm         = getSMVersion();
    const float  q_scaling  = 1.0f;

    const size_t seq_len1 = 63;

    cudaStream_t     stream;
    cublasHandle_t   cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    check_cuda_error(cudaStreamCreate(&stream));
    check_cuda_error(cublasCreate(&cublas_handle));
    check_cuda_error(cublasLtCreate(&cublaslt_handle));
#ifdef SPARSITY_ENABLED
    cusparseLtHandle_t cusparselt_handle;
    CHECK_CUSPARSE(cusparseLtInit(&cusparselt_handle));
#endif
    check_cuda_error(cublasSetStream(cublas_handle, stream));
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

    WenetDecoderWeight<T> wenet_Decoder_weights(head_num, size_per_head, inter_size, num_layers, vocab_size, max_len);
    // std::string paraFilePath = "/target/python/dec/bin_model/";
    // wenet_Decoder_weights->loadModel(paraFilePath);
    WenetDecoder<T> wenet_Decoder = WenetDecoder<T>(0,  // max_batch_size_, deprecated
                                                    0,  // max_seq_len_, deprecated
                                                    head_num,
                                                    size_per_head,
                                                    inter_size,
                                                    num_layers,
                                                    vocab_size,
                                                    max_len,
                                                    q_scaling,
                                                    0,  // stream placeholder
                                                    &cublas_wrapper,
                                                    &allocator,
                                                    false);

    wenet_Decoder.setStream(stream);

    int*   from_tensor1;
    T*     from_tensor2;
    T*     from_tensor3;
    float* out_tensor1;
    int*   out_tensor2;
    deviceMalloc(&from_tensor1, batch_size * beam_width * (seq_len1 + 1), false);
    deviceMalloc(&from_tensor2, batch_size * seq_len * d_model, false);
    deviceMalloc(&from_tensor3, batch_size * beam_width, false);
    deviceMalloc(&out_tensor1, batch_size * beam_width * seq_len1 * vocab_size, false);
    deviceMalloc(&out_tensor2, batch_size, false);

    int*         h_sequence_lengths = new int[batch_size];
    unsigned int seed               = 0;
    for (uint i = 0; i < batch_size; i++) {
        h_sequence_lengths[i] = rand_r(&seed) % std::min(seq_len, seq_len1);
    }
    int* encoder_out_lens;
    int* hyps_lens_sos;
    deviceMalloc(&encoder_out_lens, batch_size, false);
    deviceMalloc(&hyps_lens_sos, batch_size * beam_width, false);
    cudaH2Dcpy(encoder_out_lens, h_sequence_lengths, batch_size);
    cudaH2Dcpy(hyps_lens_sos, h_sequence_lengths, batch_size * beam_width);
    delete[] h_sequence_lengths;

    TensorMap input_tensors{
        {"decoder_input",
         Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size, beam_width, seq_len1 + 1}, from_tensor1}},
        {"decoder_sequence_length",
         Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size, beam_width}, hyps_lens_sos}},
        {"encoder_output",
         Tensor{MEMORY_GPU, getTensorType<T>(), std::vector<size_t>{batch_size, seq_len, d_model}, from_tensor2}},
        {"encoder_sequence_length", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size}, encoder_out_lens}},
        {"ctc_score",
         Tensor{MEMORY_GPU, getTensorType<T>(), std::vector<size_t>{batch_size, beam_width}, from_tensor3}}};

    TensorMap output_tensors{
        {"decoder_output",
         Tensor{MEMORY_GPU, TYPE_FP32, std::vector<size_t>{batch_size, beam_width, seq_len1, vocab_size}, out_tensor1}},
        {"best_index", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size}, out_tensor2}}};

    // 1. init device buffer
    wenet_Decoder.forward(&output_tensors, &input_tensors, &wenet_Decoder_weights);

    // 2. CUDA graph capture
    // cudaGraph_t graph;
    // cudaGraphExec_t instance;
    // cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    // wenet_Decoder.forward(&output_tensors, &input_tensors, &wenet_Decoder_weights);
    // cudaStreamEndCapture(stream, &graph);
    // cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    // // 3. CUDA graph enqueue
    // cudaGraphLaunch(instance, stream);
    // cudaStreamSynchronize(stream);

    // warmup
    for (int i = 0; i < 10; i++) {
        wenet_Decoder.forward(&output_tensors, &input_tensors, &wenet_Decoder_weights);
        // cudaGraphLaunch(instance, stream);
        // cudaStreamSynchronize(stream);
    }

    // profile time
    const int ite = 300;
    CudaTimer cuda_timer(stream);
    cuda_timer.start();
    for (int i = 0; i < ite; i++) {
        wenet_Decoder.forward(&output_tensors, &input_tensors, &wenet_Decoder_weights);
        // cudaGraphLaunch(instance, stream);
        // cudaStreamSynchronize(stream);
    }
    float total_time = cuda_timer.stop();

    FT_LOG_INFO("batch_size %ld seq_len %ld layer %ld "
                "FT-CPP-time %.2f ms (%d iterations) ",
                batch_size,
                seq_len,
                num_layers,
                total_time / ite,
                ite);

    check_cuda_error(cudaFree(from_tensor1));
    check_cuda_error(cudaFree(from_tensor2));
    check_cuda_error(cudaFree(from_tensor3));
    check_cuda_error(cudaFree(out_tensor1));
    check_cuda_error(cudaFree(out_tensor2));
    check_cuda_error(cudaFree(encoder_out_lens));
    check_cuda_error(cudaFree(hyps_lens_sos));
    check_cuda_error(cublasDestroy(cublas_handle));
    check_cuda_error(cublasLtDestroy(cublaslt_handle));
#ifdef SPARSITY_ENABLED
    CHECK_CUSPARSE(cusparseLtDestroy(&cusparselt_handle));
#endif
    delete cublas_algo_map;
    delete cublas_wrapper_mutex;
    return 0;
}
