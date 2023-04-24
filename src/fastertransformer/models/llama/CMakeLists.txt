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

add_library(LlamaDecoderLayerWeight STATIC LlamaDecoderLayerWeight.cc)
set_property(TARGET LlamaDecoderLayerWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET LlamaDecoderLayerWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(LlamaDecoderLayerWeight PUBLIC memory_utils cuda_utils logger)

add_library(LlamaDecoder STATIC LlamaDecoder.cc)
set_property(TARGET LlamaDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET LlamaDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(LlamaDecoder PUBLIC -lcudart cublasMMWrapper
                      TensorParallelDecoderSelfAttentionLayer
                      TensorParallelSiluFfnLayer
                      layernorm_kernels
                      add_residual_kernels
                      LlamaDecoderLayerWeight
                      tensor
                      nccl_utils
                      cuda_utils
                      logger)

add_library(LlamaContextDecoder STATIC LlamaContextDecoder.cc)
set_property(TARGET LlamaContextDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET LlamaContextDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(LlamaContextDecoder PUBLIC -lcudart cublasMMWrapper
                      TensorParallelGptContextAttentionLayer
                      TensorParallelSiluFfnLayer
                      layernorm_kernels
                      add_residual_kernels
                      gpt_kernels
                      tensor
                      nccl_utils
                      cuda_utils
                      logger)

add_library(LlamaWeight STATIC LlamaWeight.cc)
set_property(TARGET LlamaWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET LlamaWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(LlamaWeight PUBLIC LlamaDecoderLayerWeight cuda_utils logger)

add_library(Llama STATIC Llama.cc)
set_property(TARGET Llama PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Llama PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Llama PUBLIC -lcudart
                      LlamaDecoder
                      LlamaContextDecoder
                      decoding_kernels
                      gpt_kernels
                      DynamicDecodeLayer
                      BaseBeamSearchLayer
                      bert_preprocess_kernels
                      tensor
                      LlamaWeight
                      cuda_utils
                      logger)
