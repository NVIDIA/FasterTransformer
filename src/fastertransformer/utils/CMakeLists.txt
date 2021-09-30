# Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

add_subdirectory(gemm_test)

add_library(cublasAlgoMap STATIC cublasAlgoMap.cc)
set_property(TARGET cublasAlgoMap PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cublasAlgoMap PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(cublasAlgoMap PUBLIC -lcublas -lcudart -lcurand)

add_library(cublasMMWrapper STATIC cublasMMWrapper.cc)
set_property(TARGET cublasMMWrapper PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cublasMMWrapper PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
if (SPARSITY_SUPPORT)
target_link_libraries(cublasMMWrapper PUBLIC -lcublas -lcudart -lcurand -lcusparse -lcusparseLt cublasAlgoMap)
else()
target_link_libraries(cublasMMWrapper PUBLIC -lcublas -lcudart -lcurand cublasAlgoMap)
endif()

add_library(nvtx_utils STATIC nvtx_utils.cc)
set_property(TARGET nvtx_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET nvtx_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(nvtx_utils PUBLIC -lnvToolsExt)

add_library(memory_utils STATIC memory_utils.cu)
set_property(TARGET memory_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET memory_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(memory_utils PUBLIC -lnvToolsExt)

if(BUILD_GPT)
    add_library(nccl_utils STATIC nccl_utils.cc)
    set_property(TARGET nccl_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
    set_property(TARGET nccl_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
    target_link_libraries(nccl_utils PUBLIC -lnccl)
endif()

add_library(cublasINT8MMWrapper STATIC cublasINT8MMWrapper.cc)
set_property(TARGET cublasINT8MMWrapper PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cublasINT8MMWrapper PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(cublasINT8MMWrapper PUBLIC -lcublasLt -lcudart -lcurand cublasAlgoMap cublasMMWrapper)
