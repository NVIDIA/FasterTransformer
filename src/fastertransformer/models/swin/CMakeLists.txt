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

add_library(SwinBlock STATIC SwinBlock.cc)
set_property(TARGET SwinBlock PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET SwinBlock PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(SwinBlock PUBLIC -lcublasLt -lcublas -lcudart WindowAttention 
    activation_kernels add_residual_kernels layernorm_kernels)

add_library(SwinBasicLayer STATIC SwinBasicLayer.cc)
set_property(TARGET SwinBasicLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET SwinBasicLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(SwinBasicLayer PUBLIC -lcublasLt -lcublas -lcudart SwinBlock)

add_library(Swin STATIC Swin.cc)
set_property(TARGET Swin PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Swin PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Swin PUBLIC -lcudart SwinBasicLayer memory_utils)

add_executable(swin_gemm swin_gemm.cc)
target_link_libraries(swin_gemm PUBLIC -lcublas -lcublasLt -lcudart swin_igemm_func swin_gemm_func memory_utils)