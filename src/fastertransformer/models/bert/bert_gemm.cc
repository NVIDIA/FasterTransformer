/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/utils/gemm_test/encoder_gemm_func.h"
#include "src/fastertransformer/utils/gemm_test/encoder_igemm_func.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace ft = fastertransformer;

int main(int argc, char* argv[])
{
    if (argc != 7) {
        printf("[ERROR] bert_gemm batch_size seq_len head_number size_per_head data_type int8_mode \n");
        printf("e.g. ./bin/bert_gemm 1 32 12 64 0 0\n");
        return 0;
    }

    const int batch_size = atoi(argv[1]);
    const int seq_len = atoi(argv[2]);
    const int head_num = atoi(argv[3]);
    const int size_per_head = atoi(argv[4]);
    const ft::CublasDataType data_type = static_cast<ft::CublasDataType>(atoi(argv[5]));  // 0 FP32, 1 FP16, 2 BF 16
    const int int8_mode = atoi(argv[6]);

    const int inter_size = 4 * head_num * size_per_head;

    printf("[INFO] arguments: \n");
    printf("  batch_size: %d \n", batch_size);
    printf("  head_num: %d \n", head_num);
    printf("  size_per_head: %d \n", size_per_head);
    printf("  data_type: %d \n", data_type);
    printf("  int8_mode: %d \n", int8_mode);
    std::cout << std::endl;

    void* gemm_test_buf;
    size_t buf_size_in_byte =
        ft::calGemmTestBufSizeInByte(batch_size, seq_len, head_num, size_per_head, inter_size, 0, int8_mode, data_type);
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

    if (int8_mode != 0) {
        ft::generate_encoder_igemm_config(batch_size, seq_len, head_num, size_per_head, gemm_test_buf, false);
    }
    else if (data_type == ft::FLOAT_DATATYPE) {
        ft::generate_encoder_gemm_config<float>(batch_size, seq_len, head_num, size_per_head, gemm_test_buf, false);
    }
    else if (data_type == ft::HALF_DATATYPE) {
        ft::generate_encoder_gemm_config<half>(batch_size, seq_len, head_num, size_per_head, gemm_test_buf, false);
    }
#ifdef ENABLE_BF16
    else if (data_type == ft::BFLOAT16_DATATYPE) {
        ft::generate_encoder_gemm_config<__nv_bfloat16>(
            batch_size, seq_len, head_num, size_per_head, gemm_test_buf, false);
    }
#endif
    else {
        printf("[ERROR] data type only supports fp32(0), fp16(1), bf16(2). \n");
        return -1;
    }

    ft::check_cuda_error(cudaFree(gemm_test_buf));
    return 0;
}
