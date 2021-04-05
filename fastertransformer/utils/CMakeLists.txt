# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

add_library(nccl_utils STATIC nccl_utils.cpp)
set_property(TARGET nccl_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET nccl_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
if(BUILD_GPT)
    target_link_libraries(nccl_utils PUBLIC -lmpi -lnccl)
endif()

add_library(nvtx_utils STATIC nvtx_utils.cpp)
set_property(TARGET nvtx_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET nvtx_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(nvtx_utils PUBLIC -lnvToolsExt)