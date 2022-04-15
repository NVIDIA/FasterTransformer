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

add_library(ViT STATIC ViT.cc)
set_property(TARGET ViT PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ViT PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ViT PUBLIC -lcudart -lcublasLt -lcublas cublasMMWrapper 
                      UnfusedAttentionLayer FusedAttentionLayer FfnLayer layernorm_kernels
                      add_residual_kernels activation_kernels vit_kernels bert_preprocess_kernels)

add_executable(vit_gemm vit_gemm.cc)
target_link_libraries(vit_gemm PUBLIC -lcublas -lcublasLt -lcudart encoder_gemm_func encoder_igemm_func memory_utils)