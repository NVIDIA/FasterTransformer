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

# add_compile_definitions(FP8_MHA)

add_library(GptFP8DecoderLayerWeight STATIC GptFP8DecoderLayerWeight.cc)
set_property(TARGET GptFP8DecoderLayerWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptFP8DecoderLayerWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptFP8DecoderLayerWeight PUBLIC memory_utils cuda_fp8_utils)

add_library(GptFP8Weight STATIC GptFP8Weight.cc)
set_property(TARGET GptFP8Weight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptFP8Weight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptFP8Weight PUBLIC GptFP8DecoderLayerWeight)

add_library(GptFP8ContextDecoder STATIC GptFP8ContextDecoder.cc)
set_property(TARGET GptFP8ContextDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptFP8ContextDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptFP8ContextDecoder PUBLIC -lcudart TensorParallelGeluFfnFP8Layer
                                                TensorParallelGptContextAttentionFP8Layer layernorm_fp8_kernels
                                                add_residual_kernels nccl_utils gpt_kernels)

add_library(GptFP8Decoder STATIC GptFP8Decoder.cc)
set_property(TARGET GptFP8Decoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptFP8Decoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptFP8Decoder PUBLIC -lcudart TensorParallelGeluFfnLayer
                                                TensorParallelDecoderSelfAttentionFP8Layer layernorm_fp8_kernels
                                                add_residual_kernels nccl_utils)

add_library(GptFP8 STATIC GptFP8.cc)
set_property(TARGET GptFP8 PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptFP8 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptFP8 PUBLIC -lcudart GptFP8Decoder GptFP8ContextDecoder
                        bert_preprocess_kernels decoding_kernels gpt_kernels layernorm_kernels
                        DynamicDecodeLayer BaseBeamSearchLayer GptFP8Weight)

add_executable(gpt_fp8_gemm gpt_fp8_gemm.cc)
target_link_libraries(gpt_fp8_gemm PUBLIC -lcudart gpt_gemm_func memory_utils)
