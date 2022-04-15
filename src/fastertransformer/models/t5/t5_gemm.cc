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

#include "src/fastertransformer/utils/gemm_test/t5_gemm_func.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace ft = fastertransformer;

int main(int argc, char* argv[])
{
    if (argc != 14 && argc != 15 && argc != 16 && argc != 17) {
        FT_LOG_ERROR("[ERROR] ./bin/t5_gemm  \\ \n"
                     "              batch_size \\ \n"
                     "              beam_width \\ \n"
                     "              max_mem_seq_len \\ \n"
                     "              encoder_d_model \\ \n"
                     "              encoder_head_num \\ \n"
                     "              encoder_size_per_head \\ \n"
                     "              encoder_inter_size \\ \n"
                     "              decoder_d_model \\ \n"
                     "              decoder_head_num \\ \n"
                     "              decoder_size_per_head \\ \n"
                     "              decoder_inter_size \\ \n"
                     "              decoder_vocab_size \\ \n"
                     "              data_type \\ \n"
                     "              tensor_para_size \\ \n"
                     "              is_fp16_compute_type \\ \n"
                     "              is_append");
        FT_LOG_ERROR("e.g. ./bin/t5_gemm 8 4 32 512 8 64 2048 512 8 64 2048 32100 1 2 1 0");
        return 0;
    }

    const int batch_size = atoi(argv[1]);
    const int beam_width = atoi(argv[2]);
    const int max_mem_seq_len = atoi(argv[3]);

    const int encoder_d_model = atoi(argv[4]);
    const int encoder_head_num = atoi(argv[5]);
    const int encoder_size_per_head = atoi(argv[6]);
    const int encoder_inter_size = atoi(argv[7]);

    const int decoder_d_model = atoi(argv[8]);
    const int decoder_head_num = atoi(argv[9]);
    const int decoder_size_per_head = atoi(argv[10]);
    const int decoder_inter_size = atoi(argv[11]);
    const int decoder_vocab_size = atoi(argv[12]);

    const ft::CublasDataType data_type = static_cast<ft::CublasDataType>(atoi(argv[13]));  // 0 FP32, 1 FP16, 2 BF 16
    const int tensor_para_size = argc <= 15 ? 1 : atoi(argv[14]);
    int is_fp16_compute_type = argc <= 16 ? 0 : atoi(argv[15]);
    if (data_type == ft::BFLOAT16_DATATYPE && is_fp16_compute_type != 0) {
        printf("[ERROR] BFLOAT16_DATATYPE does not support is_fp16_compute_type = True\n");
        return 0;
    }
    const bool is_append = argc <= 17 ? false : (bool)(atoi(argv[16]));

    std::cout << "[INFO] arguments: " << std::endl
              << "    batch_size: " << batch_size << std::endl
              << "    beam_width: " << beam_width << std::endl
              << "    max_mem_seq_len: " << max_mem_seq_len << std::endl
              << "    encoder_d_model: " << encoder_d_model << std::endl
              << "    encoder_head_num: " << encoder_head_num << std::endl
              << "    encoder_size_per_head: " << encoder_size_per_head << std::endl
              << "    encoder_inter_size: " << encoder_inter_size << std::endl
              << "    decoder_d_model: " << decoder_d_model << std::endl
              << "    decoder_head_num: " << decoder_head_num << std::endl
              << "    decoder_size_per_head: " << decoder_size_per_head << std::endl
              << "    decoder_inter_size: " << decoder_inter_size << std::endl
              << "    decoder_vocab_size: " << decoder_vocab_size << std::endl
              << "    data_type: " << data_type << std::endl
              << "    tensor_para_size: " << tensor_para_size << std::endl
              << "    is_fp16_compute_type: " << is_fp16_compute_type << std::endl;

    void* gemm_test_buf;
    size_t buf_size_in_byte = ft::calT5GemmTestBufSizeInByte(batch_size,
                                                             beam_width,
                                                             max_mem_seq_len,
                                                             encoder_d_model,
                                                             encoder_head_num,
                                                             encoder_size_per_head,
                                                             encoder_inter_size,
                                                             decoder_d_model,
                                                             decoder_head_num,
                                                             decoder_size_per_head,
                                                             decoder_inter_size,
                                                             decoder_vocab_size,
                                                             tensor_para_size,
                                                             data_type);
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

    if (data_type == ft::FLOAT_DATATYPE) {
        ft::generate_t5_gemm_config<float>(batch_size,
                                           beam_width,
                                           max_mem_seq_len,
                                           encoder_d_model,
                                           encoder_head_num,
                                           encoder_size_per_head,
                                           encoder_inter_size,
                                           decoder_d_model,
                                           decoder_head_num,
                                           decoder_size_per_head,
                                           decoder_inter_size,
                                           decoder_vocab_size,
                                           tensor_para_size,
                                           gemm_test_buf,
                                           is_append,
                                           is_fp16_compute_type);
    }
    else if (data_type == ft::HALF_DATATYPE) {
        ft::generate_t5_gemm_config<half>(batch_size,
                                          beam_width,
                                          max_mem_seq_len,
                                          encoder_d_model,
                                          encoder_head_num,
                                          encoder_size_per_head,
                                          encoder_inter_size,
                                          decoder_d_model,
                                          decoder_head_num,
                                          decoder_size_per_head,
                                          decoder_inter_size,
                                          decoder_vocab_size,
                                          tensor_para_size,
                                          gemm_test_buf,
                                          is_append,
                                          is_fp16_compute_type);
    }
#ifdef ENABLE_BF16
    else if (data_type == ft::BFLOAT16_DATATYPE) {
        ft::generate_t5_gemm_config<__nv_bfloat16>(batch_size,
                                                   beam_width,
                                                   max_mem_seq_len,
                                                   encoder_d_model,
                                                   encoder_head_num,
                                                   encoder_size_per_head,
                                                   encoder_inter_size,
                                                   decoder_d_model,
                                                   decoder_head_num,
                                                   decoder_size_per_head,
                                                   decoder_inter_size,
                                                   decoder_vocab_size,
                                                   tensor_para_size,
                                                   gemm_test_buf,
                                                   is_append,
                                                   is_fp16_compute_type);
    }
#endif
    else {
        printf("[ERROR] data type only supports fp32(0), fp16(1), bf16(2). \n");
        return -1;
    }

    ft::check_cuda_error(cudaFree(gemm_test_buf));
    return 0;
}
