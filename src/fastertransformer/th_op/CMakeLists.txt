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

add_definitions(-DTORCH_CUDA=1)

add_library(th_utils STATIC th_utils.cu)
set_property(TARGET th_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET th_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(th_utils PUBLIC "${TORCH_LIBRARIES}" -lcublas -lcudart -lcurand)

add_subdirectory(bert)
add_subdirectory(encoder)
add_subdirectory(decoder)
add_subdirectory(decoding)
add_subdirectory(gpt)
add_subdirectory(longformer)
add_subdirectory(swin)
add_subdirectory(vit)

if(BUILD_MULTI_GPU)
    add_subdirectory(multi_gpu_gpt)
    add_subdirectory(t5)
endif()
