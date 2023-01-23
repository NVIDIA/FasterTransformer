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

cmake_minimum_required(VERSION 3.8)

add_library(DecoderSelfAttentionFP8Layer STATIC DecoderSelfAttentionFP8Layer.cc)
set_property(TARGET DecoderSelfAttentionFP8Layer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET DecoderSelfAttentionFP8Layer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(DecoderSelfAttentionFP8Layer PUBLIC -lcublasLt -lcublas -lcudart cublasMMWrapper cublasFP8MMWrapper decoder_masked_multihead_attention memory_utils cuda_fp8_utils)

add_library(TensorParallelDecoderSelfAttentionFP8Layer STATIC TensorParallelDecoderSelfAttentionFP8Layer.cc)
set_property(TARGET TensorParallelDecoderSelfAttentionFP8Layer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET TensorParallelDecoderSelfAttentionFP8Layer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(TensorParallelDecoderSelfAttentionFP8Layer PUBLIC -lcublasLt -lcublas -lcudart cublasMMWrapper cublasFP8MMWrapper DecoderSelfAttentionFP8Layer cuda_fp8_utils)

add_library(GptContextAttentionFP8Layer STATIC GptContextAttentionFP8Layer.cc)
set_property(TARGET GptContextAttentionFP8Layer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptContextAttentionFP8Layer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptContextAttentionFP8Layer PUBLIC -lcublasLt -lcublas -lcudart cublasMMWrapper cublasFP8MMWrapper unfused_attention_fp8_kernels memory_utils unfused_attention_kernels cuda_fp8_utils)

add_library(TensorParallelGptContextAttentionFP8Layer STATIC TensorParallelGptContextAttentionFP8Layer.cc)
set_property(TARGET TensorParallelGptContextAttentionFP8Layer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET TensorParallelGptContextAttentionFP8Layer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(TensorParallelGptContextAttentionFP8Layer PUBLIC -lcublasLt -lcublas -lcudart cublasMMWrapper cublasFP8MMWrapper GptContextAttentionFP8Layer cuda_fp8_utils)

add_library(SelfAttentionFP8Layer STATIC SelfAttentionFP8Layer.cc)
set_property(TARGET SelfAttentionFP8Layer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET SelfAttentionFP8Layer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(SelfAttentionFP8Layer PUBLIC -lcublasLt -lcublas -lcudart cublasMMWrapper cublasFP8MMWrapper 
                    GptContextAttentionFP8Layer trt_fused_multi_head_attention cuda_fp8_utils nvtx_utils)
