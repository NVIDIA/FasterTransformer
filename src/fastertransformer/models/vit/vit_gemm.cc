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

#include "src/fastertransformer/utils/gemm_test/encoder_gemm_func.h"
#include "src/fastertransformer/utils/gemm_test/encoder_igemm_func.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace ft = fastertransformer;

int main(int argc, char* argv[])
{
    if (argc != 9) {
        printf(
            "[ERROR] vit_gemm batch_size img_size patch_size embed_dim head_number with_cls_token data_type int8_mode \n");
        printf("e.g. ./bin/vit_gemm 1 224 16 768 12 1 0 0\n");
        return 0;
    }

    const int batch_size = atoi(argv[1]);
    const int img_size = atoi(argv[2]);
    const int patch_size = atoi(argv[3]);
    const int embed_dim = atoi(argv[4]);
    const int head_num = atoi(argv[5]);
    const int with_cls_token = atoi(argv[6]);
    const ft::CublasDataType data_type = static_cast<ft::CublasDataType>(atoi(argv[7]));  // 0 FP32, 1 FP16, 2 BF 16
    const int int8_mode = atoi(argv[8]);

    printf("[INFO] arguments: \n");
    printf("  batch_size: %d \n", batch_size);
    printf("  img_size: %d \n", img_size);
    printf("  patch_size: %d \n", patch_size);
    printf("  embed_dim: %d \n", embed_dim);
    printf("  head_num: %d \n", head_num);
    printf("  with_cls_token: %d \n", with_cls_token);
    printf("  data_type: %d \n", data_type);
    printf("  int8_mode: %d \n", int8_mode);

    if (img_size % patch_size != 0) {
        printf("[ERROR] Invalid img_size and patch_size, (i=%d mod p=%d) !=0\n", img_size, patch_size);
        return -1;
    }

    if (embed_dim % head_num != 0) {
        printf("[ERROR] Invalid embed_dim and head_num, (e=%d mod h=%d) != 0\n", embed_dim, head_num);
    }

    const int patch_resol = img_size / patch_size;
    int seq_len = patch_resol * patch_resol + (with_cls_token != 0 ? 1 : 0);
    if (atoi(argv[7]) == 1 && seq_len > 384 && seq_len % 8 != 0) {
        seq_len = (seq_len + 7) / 8 * 8;
    }
    const int size_per_head = embed_dim / head_num;

    const int inter_size = 4 * head_num * size_per_head;

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
    else {
        printf("[ERROR] is_fp16 should be 0 (use float) or 1 (use half). \n");
        return -1;
    }

    ft::check_cuda_error(cudaFree(gemm_test_buf));
    return 0;
}
