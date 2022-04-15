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

add_library(ParallelGptDecoderLayerWeight STATIC ParallelGptDecoderLayerWeight.cc)
set_property(TARGET ParallelGptDecoderLayerWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelGptDecoderLayerWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelGptDecoderLayerWeight PUBLIC memory_utils calibrate_quantize_weight_kernels)

add_library(ParallelGptWeight STATIC ParallelGptWeight.cc)
set_property(TARGET ParallelGptWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelGptWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelGptWeight PUBLIC ParallelGptDecoderLayerWeight)

add_library(ParallelGptContextDecoder STATIC ParallelGptContextDecoder.cc)
set_property(TARGET ParallelGptContextDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelGptContextDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelGptContextDecoder PUBLIC -lcudart TensorParallelGeluFfnLayer
                                                TensorParallelGptContextAttentionLayer layernorm_kernels
                                                add_residual_kernels nccl_utils gpt_kernels)

add_library(ParallelGptDecoder STATIC ParallelGptDecoder.cc)
set_property(TARGET ParallelGptDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelGptDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelGptDecoder PUBLIC -lcudart TensorParallelGeluFfnLayer
                                                TensorParallelDecoderSelfAttentionLayer layernorm_kernels
                                                add_residual_kernels nccl_utils)

add_library(ParallelGpt STATIC ParallelGpt.cc)
set_property(TARGET ParallelGpt PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelGpt PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelGpt PUBLIC -lcudart ParallelGptDecoder ParallelGptContextDecoder decoding_kernels gpt_kernels
                      DynamicDecodeLayer BaseBeamSearchLayer bert_preprocess_kernels ParallelGptWeight custom_ar_comm logprob_kernels)

add_executable(gpt_gemm gpt_gemm.cc)
target_link_libraries(gpt_gemm PUBLIC -lcudart gpt_gemm_func memory_utils)
