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

#include "src/fastertransformer/utils/gemm_test/decoding_gemm_func.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace ft = fastertransformer;

int main(int argc, char* argv[])
{
    if (argc != 10 && argc != 11) {
        printf(
            "[ERROR] decoding_gemm batch_size beam_width head_num size_per_head inter_size vocab_size max_mem_seq_len memory_hidden_units is_fp16 \n");
        printf("e.g. ./bin/decoding_gemm 32 4 8 64 2048 30000 32 512 0\n");
        return 0;
    }

    const int batch_size = atoi(argv[1]);
    const int beam_width = atoi(argv[2]);
    const int head_num = atoi(argv[3]);
    const int size_per_head = atoi(argv[4]);
    const int inter_size = atoi(argv[5]);
    const int vocab_size = atoi(argv[6]);
    const int max_mem_seq_len = atoi(argv[7]);
    const int memory_hidden_units = atoi(argv[8]);
    const int is_fp16 = atoi(argv[9]);
    const bool is_append = argc == 11 ? ((bool)atoi(argv[10])) : false;

    printf("[INFO] arguments: \n");
    printf("  batch_size: %d \n", batch_size);
    printf("  beam_width: %d \n", beam_width);
    printf("  head_num: %d \n", head_num);
    printf("  size_per_head: %d \n", size_per_head);
    printf("  inter_size: %d \n", inter_size);
    printf("  vocab_size: %d \n", vocab_size);
    printf("  max_mem_seq_len: %d \n", max_mem_seq_len);
    printf("  memory_hidden_units: %d \n", memory_hidden_units);
    printf("  is_fp16: %d \n", is_fp16);
    std::cout << std::endl;

    void* gemm_test_buf;
    size_t buf_size_in_byte = ft::calDecodingGemmTestBufSizeInByte(batch_size,
                                                                   beam_width,
                                                                   max_mem_seq_len,
                                                                   head_num,
                                                                   size_per_head,
                                                                   inter_size,
                                                                   memory_hidden_units,
                                                                   vocab_size,
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
        ft::generate_decoding_gemm_config<float>(batch_size,
                                                 beam_width,
                                                 max_mem_seq_len,
                                                 head_num,
                                                 size_per_head,
                                                 inter_size,
                                                 vocab_size,
                                                 memory_hidden_units,
                                                 gemm_test_buf,
                                                 is_append);
    }
    else if (is_fp16 == 1) {
        ft::generate_decoding_gemm_config<half>(batch_size,
                                                beam_width,
                                                max_mem_seq_len,
                                                head_num,
                                                size_per_head,
                                                inter_size,
                                                vocab_size,
                                                memory_hidden_units,
                                                gemm_test_buf,
                                                is_append);
    }
    else {
        printf("[ERROR] is_fp16 should be 0 (use float) or 1 (use half). \n");
        return -1;
    }

    ft::check_cuda_error(cudaFree(gemm_test_buf));
    std::cout << "[INFO] Finish the decoding gemm test" << std::endl;
    return 0;
}
