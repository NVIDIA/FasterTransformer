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

add_library(BartDecoder STATIC BartDecoder.cc BartDecoderLayerWeight.cc)
set_property(TARGET BartDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET BartDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(BartDecoder PUBLIC -lcudart cublasMMWrapper TensorParallelDecoderSelfAttentionLayer
                    TensorParallelDecoderCrossAttentionLayer TensorParallelReluFfnLayer TensorParallelSiluFfnLayer
                    layernorm_kernels add_residual_kernels nccl_utils memory_utils tensor)

add_library(BartDecoding STATIC BartDecoding.cc BartDecodingWeight.cc)
set_property(TARGET BartDecoding PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET BartDecoding PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(BartDecoding PUBLIC -lcudart cublasMMWrapper BartDecoder bert_preprocess_kernels
                                        decoding_kernels DynamicDecodeLayer BaseBeamSearchLayer 
                                        beam_search_topk_kernels gpt_kernels tensor)

add_library(BartEncoder STATIC BartEncoder.cc BartEncoderWeight.cc BartEncoderLayerWeight.cc)
set_property(TARGET BartEncoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET BartEncoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(BartEncoder PUBLIC -lcudart bert_preprocess_kernels cublasMMWrapper 
                        TensorParallelUnfusedAttentionLayer FusedAttentionLayer TensorParallelReluFfnLayer
                        TensorParallelGeluFfnLayer TensorParallelSiluFfnLayer layernorm_kernels add_residual_kernels nccl_utils tensor)
