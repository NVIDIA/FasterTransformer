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

add_library(GptNeoXDecoderLayerWeight STATIC GptNeoXDecoderLayerWeight.cc)
set_property(TARGET GptNeoXDecoderLayerWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptNeoXDecoderLayerWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptNeoXDecoderLayerWeight PUBLIC memory_utils)

add_library(GptNeoXDecoder STATIC GptNeoXDecoder.cc)
set_property(TARGET GptNeoXDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptNeoXDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptNeoXDecoder PUBLIC -lcudart cublasMMWrapper
                      TensorParallelDecoderSelfAttentionLayer
                      TensorParallelGeluFfnLayer
                      layernorm_kernels
                      add_residual_kernels
                      GptNeoXDecoderLayerWeight
                      tensor
                      nccl_utils)

add_library(GptNeoXContextDecoder STATIC GptNeoXContextDecoder.cc)
set_property(TARGET GptNeoXContextDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptNeoXContextDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptNeoXContextDecoder PUBLIC -lcudart cublasMMWrapper
                      TensorParallelGptContextAttentionLayer
                      TensorParallelGeluFfnLayer
                      layernorm_kernels
                      add_residual_kernels
                      gpt_kernels
                      tensor
                      nccl_utils)

add_library(GptNeoXWeight STATIC GptNeoXWeight.cc)
set_property(TARGET GptNeoXWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptNeoXWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptNeoXWeight PUBLIC GptNeoXDecoderLayerWeight)

add_library(GptNeoX STATIC GptNeoX.cc)
set_property(TARGET GptNeoX PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptNeoX PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptNeoX PUBLIC -lcudart
                      GptNeoXDecoder
                      GptNeoXContextDecoder
                      decoding_kernels
                      gpt_kernels
                      DynamicDecodeLayer
                      BaseBeamSearchLayer
                      bert_preprocess_kernels
                      tensor
                      GptNeoXWeight)
