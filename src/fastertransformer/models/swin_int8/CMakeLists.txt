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

add_library(SwinBlockINT8 STATIC SwinBlockINT8.cc)
set_property(TARGET SwinBlockINT8 PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET SwinBlockINT8 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(SwinBlockINT8 PUBLIC -lcublasLt -lcublas -lcudart 
    WindowAttentionINT8 activation_int8_kernels add_residual_kernels)

add_library(SwinBasicLayerINT8 STATIC SwinBasicLayerINT8.cc)
set_property(TARGET SwinBasicLayerINT8 PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET SwinBasicLayerINT8 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(SwinBasicLayerINT8 PUBLIC -lcublasLt -lcublas -lcudart SwinBlockINT8 dequantize_kernels)

add_library(SwinINT8 STATIC SwinINT8.cc)
set_property(TARGET SwinINT8 PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET SwinINT8 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(SwinINT8 PUBLIC -lcudart SwinBasicLayerINT8 activation_kernels memory_utils)
