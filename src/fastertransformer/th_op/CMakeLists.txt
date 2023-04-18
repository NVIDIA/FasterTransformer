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

add_definitions(-DTORCH_CUDA=1)

add_library(th_utils STATIC th_utils.cu)
set_property(TARGET th_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET th_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(th_utils PUBLIC "${TORCH_LIBRARIES}" -lcublas -lcudart -lcurand tensor)

add_subdirectory(common)
add_subdirectory(decoder)
add_subdirectory(decoding)
add_subdirectory(encoder)
add_subdirectory(longformer)
add_subdirectory(swin)
add_subdirectory(vit)
add_subdirectory(multi_gpu_gpt)
add_subdirectory(gptneox)
add_subdirectory(t5)
add_subdirectory(bart)
add_subdirectory(bert)
add_subdirectory(deberta)

add_library(th_transformer SHARED
            $<TARGET_OBJECTS:th_bart>
            $<TARGET_OBJECTS:th_bert>
            $<TARGET_OBJECTS:th_common>
            $<TARGET_OBJECTS:th_deberta>
            $<TARGET_OBJECTS:th_decoder>
            $<TARGET_OBJECTS:th_decoding>
            $<TARGET_OBJECTS:th_encoder>
            $<TARGET_OBJECTS:th_gather_tree>
            $<TARGET_OBJECTS:th_longformer>
            $<TARGET_OBJECTS:th_parallel_gpt>
            $<TARGET_OBJECTS:th_gptneox>
            $<TARGET_OBJECTS:th_swintransformer>
            $<TARGET_OBJECTS:th_t5>
            $<TARGET_OBJECTS:th_utils>
            $<TARGET_OBJECTS:th_vit>
)
target_link_libraries(th_transformer PUBLIC "${TORCH_LIBRARIES}"
                      th_bart
                      th_bert
                      th_common
                      th_deberta
                      th_decoder
                      th_decoding
                      th_encoder
                      th_gather_tree
                      th_longformer
                      th_parallel_gpt
                      th_gptneox
                      th_swintransformer
                      th_t5
                      th_utils
                      th_vit
)

if(ENABLE_FP8)
add_subdirectory(gpt_fp8)
target_link_libraries(th_transformer PUBLIC $<TARGET_OBJECTS:th_gpt_fp8> th_gpt_fp8)
endif()
