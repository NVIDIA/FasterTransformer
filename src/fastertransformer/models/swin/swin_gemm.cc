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

#include "src/fastertransformer/utils/gemm_test/swin_gemm_func.h"
#include "src/fastertransformer/utils/gemm_test/swin_igemm_func.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace ft = fastertransformer;

int main(int argc, char* argv[])
{
    if (argc != 8) {
        printf(
            "[ERROR] swin_gemm batch_img image_width window_width head_num(of the first block) size_per_head data_type is_int8\n");
        printf("e.g. ./bin/swin_gemm 32 224 7 3 32 0 1\n");
        return 0;
    }

    const int batch_img = atoi(argv[1]);
    const int image_width = atoi(argv[2]);
    const int window_width = atoi(argv[3]);
    const int head_num = atoi(argv[4]);
    const int size_per_head = atoi(argv[5]);
    const ft::CublasDataType data_type = static_cast<ft::CublasDataType>(atoi(argv[6]));  // 0 FP32, 1 FP16, 2 BF 16
    const int is_int8 = atoi(argv[7]);

    printf("[INFO] arguments: \n");
    printf("  batch_img: %d \n", batch_img);
    printf("  image_width: %d \n", image_width);
    printf("  window_width: %d \n", window_width);
    printf("  head_num: %d \n", head_num);
    printf("  size_per_head: %d \n", size_per_head);
    printf("  data_type: %d \n", data_type);
    printf("  is_int8: %d \n", is_int8);
    std::cout << std::endl;

    const int patch_width = 4;
    const int batch_size =
        batch_img * (image_width / (patch_width * window_width)) * (image_width / (patch_width * window_width));
    const int seq_len = window_width * window_width;
    const int inter_size = 4 * head_num * size_per_head;

    void* gemm_test_buf;
    size_t buf_size_in_byte = ft::calGemmTestBufSizeInByte(
        batch_size, seq_len, head_num, size_per_head, inter_size, 0, 1, ft::FLOAT_DATATYPE);
    int batch_tmp = batch_size;
    int head_num_tmp = head_num;
    for (int i = 1; i < 4; i++) {
        batch_tmp /= 4;
        head_num_tmp *= 2;
        size_t buf_size_tmp = ft::calGemmTestBufSizeInByte(
            batch_tmp, seq_len, head_num_tmp, size_per_head, 4 * head_num_tmp * size_per_head, 0, is_int8, data_type);
        if (buf_size_tmp > buf_size_in_byte) {
            buf_size_in_byte = buf_size_tmp;
        }
    }
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
        ft::deviceMalloc(reinterpret_cast<char**>(&gemm_test_buf), buf_size_in_byte, true);
    }

    if (is_int8 != 0) {
        ft::generate_swin_igemm_config(batch_size, seq_len, head_num, size_per_head, gemm_test_buf, false);
    }
    else if (data_type == ft::FLOAT_DATATYPE) {
        ft::generate_swin_gemm_config<float>(batch_size, seq_len, head_num, size_per_head, gemm_test_buf, false);
    }
    else if (data_type == ft::HALF_DATATYPE) {
        ft::generate_swin_gemm_config<half>(batch_size, seq_len, head_num, size_per_head, gemm_test_buf, false);
    }
#ifdef ENABLE_BF16
    else if (data_type == ft::BFLOAT16_DATATYPE) {
        ft::generate_swin_gemm_config<__nv_bfloat16>(
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
