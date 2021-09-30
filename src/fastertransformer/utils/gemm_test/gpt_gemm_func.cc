/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/utils/gemm_test/gpt_gemm_func.h"

namespace fastertransformer {

template<typename T>
void generate_gpt_gemm_config(int batch_size,
                              int beam_width,
                              int max_input_len,
                              int head_num,
                              int size_per_head,
                              int inter_size,
                              int vocab_size,
                              int tensor_para_size,
                              void* buffer_in,
                              bool isAppend)
{
    FT_CHECK(head_num % tensor_para_size == 0);
    void* cublas_workspace;
    void* buffer;
    int workSpaceSize;
    if (std::is_same<T, half>::value) {
        // cublas_workspace_ should be the start pointer of cudaMalloc()
        // to ensure 16B alignemnet
        cublas_workspace = buffer_in;
        buffer = (void*)((char*)cublas_workspace + CUBLAS_WORKSPACE_SIZE);
        workSpaceSize = CUBLAS_WORKSPACE_SIZE;
    }
    else {
        cublas_workspace = nullptr;
        buffer = buffer_in;
        workSpaceSize = 0;
    }

    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));
    printf("Device %s\n", prop.name);

    // check config
    FILE* fd;
    int line_count = 0;
    if (!isAppend) {
        fd = fopen(GEMM_CONFIG, "w+");
    }
    else {
        fd = fopen(GEMM_CONFIG, "a+");
        std::vector<std::string> config;
        char line[1024];
        while (fgets(line, 1024, fd) != NULL) {
            config.push_back(std::string(line));
        }
        line_count = config.size();
        if (config.size() >= (MAX_CONFIG_NUM * GEMM_NUM + 1))  // 6 cublas/cublasLt, first row is not included
        {
            int startIdx = config.size() - ((MAX_CONFIG_NUM - 1) * GEMM_NUM);
            fclose(fd);
            fd = fopen(GEMM_CONFIG, "w+");
            fprintf(fd, "%s", config[0].c_str());
            for (uint i = startIdx; i < config.size(); i++) {
                fprintf(fd, "%s", config[i].c_str());
            }
            line_count = config.size() - (GEMM_NUM + 3);
        }
    }

    const int hidden_units = head_num * size_per_head;
    const int local_head_num = head_num / tensor_para_size;
    const int local_hidden_units = local_head_num * size_per_head;
    const int gemm_num = 11;
    int M[gemm_num];
    int N[gemm_num];
    int K[gemm_num];
    int batchCount[gemm_num];
    char mess[gemm_num][256];

    // gemm 0
    M[0] = batch_size * beam_width * max_input_len;
    K[0] = hidden_units;
    N[0] = 3 * local_hidden_units;
    batchCount[0] = 1;
    strcpy(mess[0], "context from_tensor * weightQKV");

    // gemm 1
    M[1] = max_input_len;
    K[1] = size_per_head;
    N[1] = max_input_len;
    batchCount[1] = batch_size * beam_width * local_head_num;
    strcpy(mess[1], "context batch gemm Q*K^T");

    // gemm 2
    M[2] = max_input_len;
    K[2] = max_input_len;
    N[2] = size_per_head;
    batchCount[2] = batch_size * beam_width * local_head_num;
    strcpy(mess[2], "context batch gemm QK*V^T");

    // gemm 3
    M[3] = batch_size * beam_width * max_input_len;
    K[3] = local_hidden_units;
    N[3] = hidden_units;
    batchCount[3] = 1;
    strcpy(mess[3], "context attr * output_kernel");

    // gemm 4
    M[4] = batch_size * beam_width * max_input_len;
    K[4] = hidden_units;
    N[4] = inter_size;
    batchCount[4] = 1;
    strcpy(mess[4], "context ffn gemm 1");

    // gemm 5
    M[5] = batch_size * beam_width * max_input_len;
    K[5] = inter_size;
    N[5] = hidden_units;
    batchCount[5] = 1;
    strcpy(mess[5], "context ffn gemm 2");

    // gemm 6
    M[6] = batch_size * beam_width;
    K[6] = hidden_units;
    N[6] = 3 * local_hidden_units;
    batchCount[6] = 1;
    strcpy(mess[6], "from_tensor * weightQKV");

    // gemm 7
    M[7] = batch_size * beam_width;
    K[7] = local_hidden_units;
    N[7] = hidden_units;
    batchCount[7] = 1;
    strcpy(mess[7], "attr * output_kernel");

    // gemm 8
    M[8] = batch_size * beam_width;
    K[8] = hidden_units;
    N[8] = inter_size;
    batchCount[8] = 1;
    strcpy(mess[8], "ffn gemm 1");

    // gemm 9
    M[9] = batch_size * beam_width;
    K[9] = inter_size;
    N[9] = hidden_units;
    batchCount[9] = 1;
    strcpy(mess[9], "ffn gemm 2");

    // gemm 10
    M[10] = batch_size * beam_width;
    K[10] = hidden_units;
    N[10] = ceil(vocab_size / 8.) * 8 / tensor_para_size;
    batchCount[10] = 1;
    strcpy(mess[10], "logits gemm");

    cublasHandle_t cublas_handle;
    check_cuda_error(cublasCreate(&cublas_handle));
    cublasLtHandle_t ltHandle;
    check_cuda_error(cublasLtCreate(&ltHandle));

    cudaDataType_t AType;
    cudaDataType_t BType;
    cudaDataType_t CType;
    cudaDataType_t computeType;
    int startAlgo, endAlgo;
    const int ites = 100;
    struct timeval start, end;

    if (sizeof(T) == sizeof(float)) {
        AType = CUDA_R_32F;
        BType = CUDA_R_32F;
        CType = CUDA_R_32F;
        computeType = CUDA_R_32F;
        startAlgo = (int)CUBLAS_GEMM_DEFAULT;
        endAlgo = (int)CUBLAS_GEMM_ALGO23;
    }
    else {
        AType = CUDA_R_16F;
        BType = CUDA_R_16F;
        CType = CUDA_R_16F;
        computeType = CUDA_R_32F; // Use fp32 computeType to prevent overflow in gpt
        startAlgo = (int)CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        endAlgo = (int)CUBLAS_GEMM_ALGO15_TENSOR_OP;
    }

    float alpha = (float)1.0f;
    float beta = (float)0.0f;

    printf("***Encoder Gemm Testing Begin***\n");
    printf("***Cublas Gemm Testing Begin***\n");
    if (line_count == 0) {
        fprintf(fd,
                "batch_size, seq_len, head_num, size_per_head dataType ### batchCount, n, m, k, algoId, "
                "customOption, tile, numSplitsK, swizzle, reductionScheme, workspaceSize, stages, exec_time\n");
    }
    for (int i = 0; i < gemm_num; ++i) {
        int seq_len = i <= 5 ? max_input_len : 1;

        int m = M[i], n = N[i], k = K[i];
        printf("\n-----------------------------\n");
        printf("GEMM test %d: [M: %d, K: %d, N: %d] %s\n", i, m, k, n, mess[i]);
        T* d_A = (T*)buffer;
        T* d_B = d_A + m * k * batchCount[i];
        T* d_C = d_B + k * n * batchCount[i];

        float exec_time = 99999.0f;
        int fast_algo = 0;
        for (int algo = startAlgo; algo <= endAlgo; algo++) {
            cublasStatus_t status;
            cudaDeviceSynchronize();
            gettimeofday(&start, NULL);
            for (int ite = 0; ite < ites; ++ite) {
                if (i == 1) {
                    status = cublasGemmStridedBatchedEx(cublas_handle,
                                                        CUBLAS_OP_T,
                                                        CUBLAS_OP_N,
                                                        max_input_len,
                                                        max_input_len,
                                                        size_per_head,
                                                        &alpha,
                                                        d_B,
                                                        BType,
                                                        size_per_head,
                                                        max_input_len * size_per_head,
                                                        d_A,
                                                        AType,
                                                        size_per_head,
                                                        max_input_len * size_per_head,
                                                        &beta,
                                                        d_C,
                                                        CUDA_R_32F, // CType,
                                                        max_input_len,
                                                        max_input_len * max_input_len,
                                                        batchCount[i],
                                                        computeType,
                                                        static_cast<cublasGemmAlgo_t>(algo));
                }
                else if (i == 2) {
                    status = cublasGemmStridedBatchedEx(cublas_handle,
                                                        CUBLAS_OP_N,
                                                        CUBLAS_OP_N,
                                                        size_per_head,
                                                        max_input_len,
                                                        max_input_len,
                                                        &alpha,
                                                        d_B,
                                                        BType,
                                                        size_per_head,
                                                        max_input_len * size_per_head,
                                                        d_A,
                                                        AType,
                                                        max_input_len,
                                                        max_input_len * max_input_len,
                                                        &beta,
                                                        d_C,
                                                        CType,
                                                        size_per_head,
                                                        max_input_len * size_per_head,
                                                        batchCount[i],
                                                        computeType,
                                                        static_cast<cublasGemmAlgo_t>(algo));
                }
                else {
                    status = cublasGemmEx(cublas_handle,
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
                                          n,
                                          m,
                                          k,
                                          &alpha,
                                          d_B,
                                          BType,
                                          n,
                                          d_A,
                                          AType,
                                          k,
                                          &beta,
                                          d_C,
                                          CType,
                                          n,
                                          computeType,
                                          static_cast<cublasGemmAlgo_t>(algo));
                }

                if (status != CUBLAS_STATUS_SUCCESS)
                    break;
            }
            cudaDeviceSynchronize();
            gettimeofday(&end, NULL);
            if (status == CUBLAS_STATUS_SUCCESS) {
                printf("algo_%d costs %.3fms \n", algo, diffTime(start, end) / ites);
                if (diffTime(start, end) / ites < exec_time) {
                    exec_time = diffTime(start, end) / ites;
                    fast_algo = algo;
                }
            }
            sync_check_cuda_error();
        }

        printf("fast_algo %d costs %.3f ms\n", fast_algo, exec_time);
        int is_fp16 = 0;
        if (sizeof(T) == sizeof(half))
            is_fp16 = 1;

        // for fp16, we compare cublasLt
        if (is_fp16 == 1 && i != 1 && i != 2) {
            printf("***cublasLt Gemm Testing Beign***\n");
            // Let try a fixed number of combinations
            int ALGO_COMBINATIONS = 5000;
            customMatmulPerf_t perfResults[ALGO_COMBINATIONS];

            //for gpt, computeType & scaleType should be FP32
            LtHgemmCustomFind<T, float>(ltHandle,
                                        batch_size * beam_width,
                                        i == 1 || i == 2 ? max_input_len : 1,
                                        head_num,
                                        size_per_head,
                                        n,
                                        m,
                                        k,
                                        &alpha,
                                        d_B,
                                        d_A,
                                        &beta,
                                        d_C,
                                        cublas_workspace,
                                        workSpaceSize,
                                        fd,
                                        perfResults,
                                        ALGO_COMBINATIONS);
            if (perfResults[0].time < exec_time) {
                printPerfStructure(
                    batch_size * beam_width, seq_len, head_num, size_per_head, n, m, k, perfResults[0], fd, is_fp16, 0);
            }
            else {
                fprintf(fd,
                        "%d %d %d %d %d ### %d %d %d %d %d -1 -1 -1 -1 -1 -1 -1 %f\n",
                        batch_size * beam_width,
                        seq_len,
                        head_num,
                        size_per_head,
                        is_fp16 ? HALF_DATATYPE : FLOAT_DATATYPE,
                        batchCount[i],
                        n,
                        m,
                        k,
                        fast_algo,
                        exec_time);
            }
            printf("***cublasLt Gemm Testing End***\n");
        }
        else {
            fprintf(fd,
                    "%d %d %d %d %d ### %d %d %d %d %d -1 -1 -1 -1 -1 -1 -1 %f\n",
                    batch_size * beam_width,
                    seq_len,
                    head_num,
                    size_per_head,
                    is_fp16 ? HALF_DATATYPE : FLOAT_DATATYPE,
                    batchCount[i],
                    n,
                    m,
                    k,
                    fast_algo,
                    exec_time);
        }
        sync_check_cuda_error();
    }
    printf("***cublas Gemm Testing End***\n\n");
    fclose(fd);
    printf("***GPT Gemm Testing End***\n");
    return;
}

template void generate_gpt_gemm_config<float>(int batch_size,
                                              int beam_width,
                                              int max_input_len,
                                              int head_num,
                                              int size_per_head,
                                              int inter_size,
                                              int vocab_size,
                                              int tensor_para_size,
                                              void* buffer_in,
                                              bool isAppend);

template void generate_gpt_gemm_config<half>(int batch_size,
                                             int beam_width,
                                             int max_input_len,
                                             int head_num,
                                             int size_per_head,
                                             int inter_size,
                                             int vocab_size,
                                             int tensor_para_size,
                                             void* buffer_in,
                                             bool isAppend);

size_t calGptGemmTestBufSizeInByte(int batch_size,
                                   int beam_width,
                                   int max_input_len,
                                   int head_num,
                                   int size_per_head,
                                   int inter_size,
                                   int vocab_size,
                                   int tensor_para_size,
                                   int is_fp16)
{
    size_t buf_size_in_byte = 0;
    const size_t hidden_units = head_num * size_per_head;
    const size_t local_head_num = head_num / tensor_para_size;
    const size_t local_hidden_units = local_head_num * size_per_head;

    int wordSize = (is_fp16 == 1 ? sizeof(half) : sizeof(float));

    size_t m = batch_size * beam_width * max_input_len;
    std::vector<size_t> buff_size;
    // for context qkv gemm
    buff_size.push_back(m * hidden_units + hidden_units * 3 * local_hidden_units + m * 3 * local_hidden_units);
    // for context batch gemm
    buff_size.push_back(m * local_hidden_units + m * local_hidden_units + m * head_num * max_input_len * max_input_len);
    // for context ffn gemm
    buff_size.push_back(m * inter_size / tensor_para_size + hidden_units * inter_size / tensor_para_size
                        + m * hidden_units);
    // for vocab
    buff_size.push_back(m * hidden_units + hidden_units * ceil(vocab_size / 8.) * 8 / tensor_para_size
                        + m * ceil(vocab_size / 8.) * 8 / tensor_para_size);

    for (auto t : buff_size) {
        buf_size_in_byte = buf_size_in_byte > t ? buf_size_in_byte : t;
    }
    buf_size_in_byte *= wordSize;
    buf_size_in_byte += ((is_fp16 == 1) ? CUBLAS_WORKSPACE_SIZE : 0);

    return buf_size_in_byte;
}

}  // namespace fastertransformer
