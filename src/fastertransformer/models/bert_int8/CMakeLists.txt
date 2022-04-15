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

add_library(BertLayerINT8 STATIC BertLayerINT8.cc)
set_property(TARGET BertLayerINT8 PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET BertLayerINT8 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(BertLayerINT8 PUBLIC -lcublasLt -lcublas -lcudart -lcurand cublasMMWrapper 
                      cublasINT8MMWrapper UnfusedAttentionLayerINT8 FusedAttentionLayerINT8 
                      FfnLayerINT8 layernorm_int8_kernels
                      layout_transformer_int8_kernels quantization_int8_kernels)

add_library(BertINT8 STATIC BertINT8.cc)
set_property(TARGET BertINT8 PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET BertINT8 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(BertINT8 PUBLIC -lcublasLt -lcublas -lcudart -lcurand BertLayerINT8 bert_preprocess_kernels)
