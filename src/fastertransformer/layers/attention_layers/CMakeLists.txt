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

add_library(UnfusedAttentionLayer STATIC UnfusedAttentionLayer.cc)
set_property(TARGET UnfusedAttentionLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET UnfusedAttentionLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(UnfusedAttentionLayer PUBLIC -lcublas -lcudart cublasMMWrapper memory_utils unfused_attention_kernels)

add_library(FusedAttentionLayer STATIC FusedAttentionLayer.cu)
set_property(TARGET FusedAttentionLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET FusedAttentionLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(FusedAttentionLayer PUBLIC -lcublas -lcudart cublasMMWrapper memory_utils trt_fused_multi_head_attention)

add_library(DecoderCrossAttentionLayer STATIC DecoderCrossAttentionLayer.cu)
set_property(TARGET DecoderCrossAttentionLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET DecoderCrossAttentionLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(DecoderCrossAttentionLayer PUBLIC -lcublas -lcudart cublasMMWrapper memory_utils decoder_masked_multihead_attention)

add_library(LongformerAttentionLayer STATIC LongformerAttentionLayer.cc)
set_property(TARGET LongformerAttentionLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET LongformerAttentionLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(LongformerAttentionLayer PUBLIC -lcublas -lcudart cublasMMWrapper longformer_kernels)

add_library(DecoderSelfAttentionLayer STATIC DecoderSelfAttentionLayer.cc)
set_property(TARGET DecoderSelfAttentionLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET DecoderSelfAttentionLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(DecoderSelfAttentionLayer PUBLIC -lcublas -lcudart cublasMMWrapper memory_utils decoder_masked_multihead_attention matrix_vector_multiplication)

add_library(GptContextAttentionLayer STATIC GptContextAttentionLayer.cc)
set_property(TARGET GptContextAttentionLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptContextAttentionLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptContextAttentionLayer PUBLIC -lcublas -lcudart cublasMMWrapper memory_utils unfused_attention_kernels)

add_library(TensorParallelDecoderSelfAttentionLayer STATIC TensorParallelDecoderSelfAttentionLayer.cc)
set_property(TARGET TensorParallelDecoderSelfAttentionLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET TensorParallelDecoderSelfAttentionLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(TensorParallelDecoderSelfAttentionLayer PUBLIC -lcudart DecoderSelfAttentionLayer nccl_utils custom_ar_kernels)

add_library(TensorParallelDecoderCrossAttentionLayer STATIC TensorParallelDecoderCrossAttentionLayer.cc)
set_property(TARGET TensorParallelDecoderCrossAttentionLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET TensorParallelDecoderCrossAttentionLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(TensorParallelDecoderCrossAttentionLayer PUBLIC -lcudart DecoderCrossAttentionLayer nccl_utils)

add_library(TensorParallelGptContextAttentionLayer STATIC TensorParallelGptContextAttentionLayer.cc)
set_property(TARGET TensorParallelGptContextAttentionLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET TensorParallelGptContextAttentionLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(TensorParallelGptContextAttentionLayer PUBLIC -lcudart GptContextAttentionLayer nccl_utils custom_ar_kernels)

add_library(TensorParallelUnfusedAttentionLayer STATIC TensorParallelUnfusedAttentionLayer.cc)
set_property(TARGET TensorParallelUnfusedAttentionLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET TensorParallelUnfusedAttentionLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(TensorParallelUnfusedAttentionLayer PUBLIC -lcudart UnfusedAttentionLayer nccl_utils)

add_library(WindowAttention STATIC WindowAttention.cc)
set_property(TARGET WindowAttention PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET WindowAttention PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(WindowAttention PUBLIC -lcublasLt -lcublas -lcudart transform_mask_kernels 
    unfused_attention_kernels trt_fused_multi_head_attention reverse_roll_kernels activation_kernels)