/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "decoding_gemm.h"

int main(int argc, char* argv[])
{
  if(argc != 9)
  {
    printf("[ERROR] gpt_gemm local_batch_size context_local_batch_size head_number size_per_head vocab_size start_len tensor_para_size is_fp16\n");
    printf("e.g. ./bin/gpt_gemm 8 8 96 128 51200 32 8 1\n");
    return 0;
  }
  const int local_batch_size = atoi(argv[1]);
  const int context_local_batch_size = atoi(argv[2]);
  const int head_number = atoi(argv[3]);
  const int size_per_head = atoi(argv[4]);
  const int vocab_size = atoi(argv[5]);
  const int start_len = atoi(argv[6]);
  const int tensor_para_size = atoi(argv[7]);

  struct cudaDeviceProp prop;
  check_cuda_error(cudaGetDeviceProperties(&prop, 0));
  printf("Device %s\n", prop.name);

  if(atoi(argv[8]) == 0)
    generate_gpt_gemm_config<float>(local_batch_size, context_local_batch_size, head_number, size_per_head, vocab_size, start_len, tensor_para_size);
  else if(atoi(argv[8]) == 1)
    generate_gpt_gemm_config<half>(local_batch_size, context_local_batch_size, head_number, size_per_head, vocab_size, start_len, tensor_para_size);
  else
  {
    printf("[ERROR] is_fp16 should be 0 (use float) or 1 (use half). \n");
    return -1;
  }

  return 0;
}