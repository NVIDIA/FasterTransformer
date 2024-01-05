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

add_library(M2MDecoder STATIC M2MDecoder.cc M2MDecoderLayerWeight.cc)
set_property(TARGET M2MDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET M2MDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(M2MDecoder PUBLIC -lcudart cublasMMWrapper TensorParallelDecoderSelfAttentionLayer
                    TensorParallelDecoderCrossAttentionLayer TensorParallelReluFfnLayer TensorParallelSiluFfnLayer
                    layernorm_kernels add_residual_kernels nccl_utils memory_utils tensor)

add_library(M2MDecoding STATIC M2MDecoding.cc M2MDecodingWeight.cc)
set_property(TARGET M2MDecoding PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET M2MDecoding PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(M2MDecoding PUBLIC -lcudart cublasMMWrapper M2MDecoder bert_preprocess_kernels
                                        decoding_kernels DynamicDecodeLayer BaseBeamSearchLayer 
                                        beam_search_topk_kernels gpt_kernels tensor)

add_library(M2MEncoder STATIC M2MEncoder.cc M2MEncoderWeight.cc M2MEncoderLayerWeight.cc)
set_property(TARGET M2MEncoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET M2MEncoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(M2MEncoder PUBLIC -lcudart bert_preprocess_kernels cublasMMWrapper 
                        TensorParallelUnfusedAttentionLayer FusedAttentionLayer TensorParallelReluFfnLayer
                        TensorParallelGeluFfnLayer TensorParallelSiluFfnLayer layernorm_kernels add_residual_kernels nccl_utils tensor)
