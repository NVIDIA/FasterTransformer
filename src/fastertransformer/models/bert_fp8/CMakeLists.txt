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


add_library(BertFP8Weight STATIC BertFP8Weight.cc BertFP8LayerWeight.cc)
set_property(TARGET BertFP8Weight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET BertFP8Weight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(BertFP8Weight PUBLIC memory_utils cuda_fp8_utils fp8_qgmma_1x1_utils)

add_library(BertFP8 STATIC BertFP8.cc)
set_property(TARGET BertFP8 PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET BertFP8 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(BertFP8 PUBLIC -lcudart bert_preprocess_kernels  
                      FfnFP8Layer layernorm_kernels layernorm_fp8_kernels add_residual_kernels SelfAttentionFP8Layer
                      BertFP8Weight)
