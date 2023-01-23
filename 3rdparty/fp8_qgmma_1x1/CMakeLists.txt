# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

set(fp8_gemm_1x1_files
    fp8_qgmma_1x1_utils.cu
)

file(GLOB fp8_gemm_1x1_files ${fp8_gemm_1x1_files} cubins/*.cubin.cpp)

add_library(fp8_qgmma_1x1_utils STATIC ${fp8_gemm_1x1_files})
target_link_libraries(fp8_qgmma_1x1_utils PUBLIC cuda_driver_wrapper)
set_property(TARGET fp8_qgmma_1x1_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET fp8_qgmma_1x1_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
