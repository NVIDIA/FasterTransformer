# Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.8)

add_library(Bert STATIC Bert.cc)
set_property(TARGET Bert PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Bert PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Bert PUBLIC -lcudart bert_preprocess_kernels cublasMMWrapper 
                      UnfusedAttentionLayer FusedAttentionLayer FfnLayer layernorm_kernels
                      add_residual_kernels)

add_executable(bert_gemm bert_gemm.cc)
target_link_libraries(bert_gemm PUBLIC -lcublas -lcublasLt -lcudart encoder_gemm_func encoder_igemm_func memory_utils)