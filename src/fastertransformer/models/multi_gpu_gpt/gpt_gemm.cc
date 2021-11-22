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
#include "src/fastertransformer/utils/memory_utils.h"

namespace ft = fastertransformer;

int main(int argc, char* argv[])
{
    if (argc != 9 && argc != 10) {
        printf(
            "[ERROR] gpt_gemm batch_size beam_width max_input_len head_number size_per_head inter_size vocab_size is_fp16 tensor_para_size\n");
        printf("e.g. ./bin/gpt_gemm 8 4 32 96 128 49152 51200 8 1\n");
        return 0;
    }

    const int batch_size = atoi(argv[1]);
    const int beam_width = atoi(argv[2]);
    const int max_input_len = atoi(argv[3]);
    const int head_num = atoi(argv[4]);
    const int size_per_head = atoi(argv[5]);
    const int inter_size = atoi(argv[6]);
    const int vocab_size = atoi(argv[7]);
    const int is_fp16 = atoi(argv[8]);
    const int tensor_para_size = argc <= 9 ? 1 : atoi(argv[9]);
    const int is_fp16_compute_type = argc <= 10 ? 0 : atoi(argv[10]);

    printf("[INFO] arguments: \n");
    printf("  batch_size: %d \n", batch_size);
    printf("  beam_width: %d \n", beam_width);
    printf("  max_input_len: %d \n", max_input_len);
    printf("  head_num: %d \n", head_num);
    printf("  size_per_head: %d \n", size_per_head);
    printf("  inter_size: %d \n", inter_size);
    printf("  vocab_size: %d \n", vocab_size);
    printf("  is_fp16: %d \n", is_fp16);
    printf("  tensor_para_size: %d \n", tensor_para_size);
    std::cout << std::endl;

    void* gemm_test_buf;
    size_t buf_size_in_byte = ft::calGptGemmTestBufSizeInByte(batch_size,
                                                              beam_width,
                                                              max_input_len,
                                                              head_num,
                                                              size_per_head,
                                                              inter_size,
                                                              vocab_size,
                                                              tensor_para_size,
                                                              is_fp16);
    size_t total, free;
    ft::check_cuda_error(cudaMemGetInfo(&free, &total));
    if (free < buf_size_in_byte + 10 * 1024 * 1024) {
        printf("[ERROR] There is no enough device memory for gemm test!\n"
               " %ld Bytes is needed, but only %ld Bytes is free.\n",
               buf_size_in_byte,
               free);
        gemm_test_buf = NULL;
        return -1;
    }
    else {
        ft::deviceMalloc(reinterpret_cast<char**>(&gemm_test_buf), buf_size_in_byte, false);
    }

    if (is_fp16 == 0) {
        ft::generate_gpt_gemm_config<float>(batch_size,
                                            beam_width,
                                            max_input_len,
                                            head_num,
                                            size_per_head,
                                            inter_size,
                                            vocab_size,
                                            tensor_para_size,
                                            gemm_test_buf,
                                            false);
    }
    else if (is_fp16 == 1) {
        ft::generate_gpt_gemm_config<half>(batch_size,
                                           beam_width,
                                           max_input_len,
                                           head_num,
                                           size_per_head,
                                           inter_size,
                                           vocab_size,
                                           tensor_para_size,
                                           gemm_test_buf,
                                           false);
    }
    else {
        printf("[ERROR] is_fp16 should be 0 (use float) or 1 (use half). \n");
        return -1;
    }

    ft::check_cuda_error(cudaFree(gemm_test_buf));
    return 0;
}
