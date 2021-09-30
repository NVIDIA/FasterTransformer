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

add_library(GptDecoderLayerWeight STATIC GptDecoderLayerWeight.cc)
set_property(TARGET GptDecoderLayerWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptDecoderLayerWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptDecoderLayerWeight PUBLIC memory_utils)

add_library(GptDecoder STATIC GptDecoder.cc)
set_property(TARGET GptDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptDecoder PUBLIC -lcudart cublasMMWrapper 
                                    DecoderSelfAttentionLayer FfnLayer layernorm_kernels add_residual_kernels GptDecoderLayerWeight)

add_library(GptContextDecoder STATIC GptContextDecoder.cc)
set_property(TARGET GptContextDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptContextDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptContextDecoder PUBLIC -lcudart cublasMMWrapper 
                        GptContextAttentionLayer FfnLayer layernorm_kernels add_residual_kernels)

add_library(GptWeight STATIC GptWeight.cc)
set_property(TARGET GptWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptWeight PUBLIC GptDecoderLayerWeight)

add_library(Gpt STATIC Gpt.cc)
set_property(TARGET Gpt PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Gpt PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Gpt PUBLIC -lcudart GptDecoder GptContextDecoder decoding_kernels gpt_kernels
                        OnlineBeamSearchLayer BeamSearchLayer TopKSamplingLayer TopPSamplingLayer 
                        TopKTopPSamplingLayer bert_preprocess_kernels GptWeight)