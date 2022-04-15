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

add_library(cnpy STATIC cnpy.cpp)
target_link_libraries(cnpy PUBLIC -lz)
set_property(TARGET cnpy PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(xlnet_example xlnet_example.cc)
target_link_libraries(xlnet_example PUBLIC -lcublas -lcublasLt -lcudart -lz cnpy Xlnet)

add_executable(xlnet_correctness_example xlnet_correctness_example.cc)
target_link_libraries(xlnet_correctness_example PUBLIC -lcublas -lcublasLt -lcudart -lz cnpy Xlnet)

