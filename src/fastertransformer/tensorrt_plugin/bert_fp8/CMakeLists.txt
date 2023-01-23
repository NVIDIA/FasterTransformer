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

set(bert_fp8_trt_files
  bertFp8Plugin.cu
)

if(BUILD_TRT)
  set(LIB_NAME "bert_fp8_plugin")
  add_library(${LIB_NAME} SHARED ${bert_fp8_trt_files})
  set_target_properties(${LIB_NAME} PROPERTIES
                        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
		target_link_libraries(${LIB_NAME} BertFP8 BertFP8Weight activation_kernels activation_fp8_kernels
                          memory_utils layernorm_fp8_kernels cuda_fp8_utils bert_preprocess_kernels
                          FfnFP8Layer cublasFP8MMWrapper cublasMMWrapper cublasAlgoMap SelfAttentionFP8Layer
                          unfused_attention_fp8_kernels trt_fused_multi_head_attention nvtx_utils
                          fp8_qgmma_1x1_utils tensor -lcudnn -lcublas -lcudart -lnvinfer)
endif()
