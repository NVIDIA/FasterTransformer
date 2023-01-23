# Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

add_subdirectory(attention_layers)
add_subdirectory(attention_layers_int8)
if(ENABLE_FP8)
add_subdirectory(attention_layers_fp8)
endif()
add_subdirectory(xlnet_attention_layers)
add_subdirectory(beam_search_layers)
add_subdirectory(sampling_layers)

add_library(FfnLayer STATIC FfnLayer.cc)
set_property(TARGET FfnLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET FfnLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(FfnLayer PUBLIC -lcublas -lcudart cublasMMWrapper activation_kernels transpose_int8_kernels memory_utils matrix_vector_multiplication tensor moe_kernels fpA_intB_gemm int8_gemm nvtx_utils)

add_library(FfnLayerINT8 STATIC FfnLayerINT8.cc)
set_property(TARGET FfnLayerINT8 PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET FfnLayerINT8 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(FfnLayerINT8 PUBLIC -lcublasLt -lcublas -lcudart cublasMMWrapper cublasINT8MMWrapper activation_int8_kernels memory_utils tensor moe_kernels nvtx_utils)

if(ENABLE_FP8)
add_library(FfnFP8Layer STATIC FfnFP8Layer.cc)
set_property(TARGET FfnFP8Layer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET FfnFP8Layer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(FfnFP8Layer PUBLIC -lcublasLt -lcublas -lcudart cublasMMWrapper 
                        cublasFP8MMWrapper activation_fp8_kernels memory_utils nvtx_utils)
endif()

add_library(TensorParallelGeluFfnLayer STATIC TensorParallelGeluFfnLayer.cc)
set_property(TARGET TensorParallelGeluFfnLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET TensorParallelGeluFfnLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(TensorParallelGeluFfnLayer PUBLIC -lcudart FfnLayer nccl_utils tensor nvtx_utils)

if(ENABLE_FP8)
add_library(TensorParallelGeluFfnFP8Layer STATIC TensorParallelGeluFfnFP8Layer.cc)
set_property(TARGET TensorParallelGeluFfnFP8Layer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET TensorParallelGeluFfnFP8Layer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(TensorParallelGeluFfnFP8Layer PUBLIC -lcudart FfnFP8Layer nccl_utils)
endif()

add_library(TensorParallelReluFfnLayer STATIC TensorParallelReluFfnLayer.cc)
set_property(TARGET TensorParallelReluFfnLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET TensorParallelReluFfnLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(TensorParallelReluFfnLayer PUBLIC -lcudart FfnLayer nccl_utils tensor nvtx_utils)

add_library(DynamicDecodeLayer STATIC DynamicDecodeLayer.cc)
set_property(TARGET DynamicDecodeLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET DynamicDecodeLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(DynamicDecodeLayer PUBLIC -lcudart
                        TopKSamplingLayer TopPSamplingLayer
                        OnlineBeamSearchLayer BeamSearchLayer ban_bad_words stop_criteria
                        gpt_kernels tensor nvtx_utils)

add_library(TensorParallelSiluFfnLayer STATIC TensorParallelSiluFfnLayer.cc)
set_property(TARGET TensorParallelSiluFfnLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET TensorParallelSiluFfnLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(TensorParallelSiluFfnLayer PUBLIC -lcudart FfnLayer nccl_utils nvtx_utils)

add_subdirectory(adapter_layers)
