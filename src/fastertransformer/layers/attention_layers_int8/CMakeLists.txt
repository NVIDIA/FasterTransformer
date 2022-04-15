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

add_library(UnfusedAttentionLayerINT8 STATIC UnfusedAttentionLayerINT8.cc)
set_property(TARGET UnfusedAttentionLayerINT8 PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET UnfusedAttentionLayerINT8 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(UnfusedAttentionLayerINT8 PUBLIC -lcublasLt -lcublas -lcudart cublasMMWrapper cublasINT8MMWrapper softmax_int8_kernels transpose_int8_kernels unfused_attention_int8_kernels)

add_library(FusedAttentionLayerINT8 STATIC FusedAttentionLayerINT8.cu)
set_property(TARGET FusedAttentionLayerINT8 PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET FusedAttentionLayerINT8 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(FusedAttentionLayerINT8 PUBLIC -lcublasLt -lcublas -lcudart cublasMMWrapper cublasINT8MMWrapper trt_fused_multi_head_attention)

add_library(WindowAttentionINT8 STATIC WindowAttentionINT8.cu)
set_property(TARGET WindowAttentionINT8 PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET WindowAttentionINT8 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(WindowAttentionINT8 PUBLIC -lcublasLt -lcublas -lcudart cublasINT8MMWrapper reverse_roll_kernels 
    trt_fused_multi_head_attention layout_transformer_int8_kernels transpose_int8_kernels unfused_attention_int8_kernels
    layernorm_int8_kernels softmax_int8_kernels layernorm_kernels transform_mask_kernels)
