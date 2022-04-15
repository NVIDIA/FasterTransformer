# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

add_library(gpt_example_utils STATIC gpt_example_utils.cc)
target_link_libraries(gpt_example_utils PUBLIC -lcudart)

add_executable(multi_gpu_gpt_example multi_gpu_gpt_example.cc)
target_link_libraries(multi_gpu_gpt_example PUBLIC -lcublas -lcublasLt -lcudart
                            ParallelGpt nvtx_utils -lmpi gpt_example_utils)

add_executable(multi_gpu_gpt_async_example multi_gpu_gpt_async_example.cc)
target_link_libraries(multi_gpu_gpt_async_example PUBLIC -lcublas -lcublasLt -lcudart
                            ParallelGpt nvtx_utils -lmpi gpt_example_utils)

add_executable(multi_gpu_gpt_triton_example multi_gpu_gpt_triton_example.cc)
target_link_libraries(multi_gpu_gpt_triton_example PUBLIC -lcublas -lcublasLt -lcudart
                            ParallelGptTritonBackend memory_utils custom_ar_comm -lmpi gpt_example_utils -lpthread)
