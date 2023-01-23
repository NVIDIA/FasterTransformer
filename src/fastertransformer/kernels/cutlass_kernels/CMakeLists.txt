# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

add_library(cutlass_heuristic STATIC cutlass_heuristic.cc)
set_property(TARGET cutlass_heuristic PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(cutlass_preprocessors STATIC cutlass_preprocessors.cc)
set_property(TARGET cutlass_preprocessors PROPERTY POSITION_INDEPENDENT_CODE ON)

set(moe_gemm_files "")
file(GLOB moe_gemm_files ${moe_gemm_files} ./moe_gemm/*.cu)
add_library(moe_gemm_kernels STATIC ${moe_gemm_files})
# target_compile_options(moe_gemm_kernels PRIVATE "-Xptxas=-v")
set_property(TARGET moe_gemm_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET moe_gemm_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(moe_gemm_kernels PRIVATE cutlass_heuristic)

set(fpA_intB_files "")
file(GLOB fpA_intB_files ${fpA_intB_files} ./fpA_intB_gemm/*.cu)
add_library(fpA_intB_gemm STATIC ${fpA_intB_files})
# target_compile_options(fpA_intB_gemm PRIVATE "-Xptxas=-v")
set_property(TARGET fpA_intB_gemm PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET fpA_intB_gemm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(fpA_intB_gemm PRIVATE cutlass_heuristic)

set(int8_files "")
file(GLOB int8_files ${int8_files} ./int8_gemm/*.cu)
add_library(int8_gemm STATIC ${int8_files})
# target_compile_options(int8_gemm PRIVATE "-Xptxas=-v")
set_property(TARGET int8_gemm PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET int8_gemm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(int8_gemm PRIVATE cutlass_heuristic)